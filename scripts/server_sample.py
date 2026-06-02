from flask import Flask, request, jsonify
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "simulator.yaml"
BASE_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "first_yolo11n" / "weights" / "best.pt"
FINETUNED_MODEL_PATHS = [
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2_focus_150" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2_focus_continue_150" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2_valfix_30" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2" / "weights" / "best.pt",
]
CLASS_ALIASES = {
    "blue": "person",
    "red": "person",
    "tank": "Tank",
}
IGNORED_CLASSES = {"car"}
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_MODEL_CONF", "0.04"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_DEFAULT_CONF", "0.20"))
CLASS_CONFIDENCE_THRESHOLDS = {
    "wall": float(os.getenv("YOLO_WALL_CONF", "0.06")),
}
CLOSE_WALL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CLOSE_WALL_CONF", "0.04"))
CLOSE_WALL_AREA_RATIO = float(os.getenv("YOLO_CLOSE_WALL_AREA_RATIO", "0.08"))
CLOSE_WALL_MIN_HEIGHT_RATIO = float(os.getenv("YOLO_CLOSE_WALL_MIN_HEIGHT_RATIO", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.70"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "100"))


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_CUDA_DEVICE = torch.cuda.is_available() and YOLO_DEVICE.lower() != "cpu"
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_HALF = USE_CUDA_DEVICE and env_flag("YOLO_HALF", True)
YOLO_TIMING = env_flag("YOLO_TIMING", False)
YOLO_DETECT_DEBUG = env_flag("YOLO_DETECT_DEBUG", False)
YOLO_WARMUP_RUNS = int(os.getenv("YOLO_WARMUP_RUNS", "2"))
SHADOW_FILTER_ENABLED = env_flag("YOLO_SHADOW_FILTER", True)
SHADOW_FILTER_SIGMA = float(os.getenv("YOLO_SHADOW_SIGMA", "35.0"))
SHADOW_FILTER_STRENGTH = float(os.getenv("YOLO_SHADOW_STRENGTH", "0.75"))
SHADOW_FILTER_WORK_SCALE = float(os.getenv("YOLO_SHADOW_WORK_SCALE", "0.35"))
SHADOW_FILTER_MAX_SIDE = int(os.getenv("YOLO_SHADOW_MAX_SIDE", "960"))
SHADOW_FILTER_CLAHE = env_flag("YOLO_SHADOW_CLAHE", False)
SHADOW_FILTER_CLAHE_CLIP = float(os.getenv("YOLO_SHADOW_CLAHE_CLIP", "1.5"))

if USE_CUDA_DEVICE:
    torch.backends.cudnn.benchmark = env_flag("YOLO_CUDNN_BENCHMARK", False)


def load_config(path=CONFIG_PATH):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_model_names(names):
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def get_public_class_name(class_id):
    class_name = model_names.get(class_id)
    if class_name is None:
        return None
    return CLASS_ALIASES.get(class_name, class_name)


def get_box_size_ratios(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    box_width = max(0.0, float(x2 - x1))
    box_height = max(0.0, float(y2 - y1))
    frame_area = max(1.0, float(frame_width * frame_height))
    area_ratio = (box_width * box_height) / frame_area
    height_ratio = box_height / max(1.0, float(frame_height))
    return area_ratio, height_ratio


def is_close_wall_candidate(class_name, confidence, box, frame_shape):
    if class_name != "wall" or confidence < CLOSE_WALL_CONFIDENCE_THRESHOLD:
        return False
    area_ratio, height_ratio = get_box_size_ratios(box, frame_shape)
    return area_ratio >= CLOSE_WALL_AREA_RATIO or height_ratio >= CLOSE_WALL_MIN_HEIGHT_RATIO


def should_return_detection(class_name, confidence, box, frame_shape):
    if class_name in IGNORED_CLASSES:
        return False
    if is_close_wall_candidate(class_name, confidence, box, frame_shape):
        return True
    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_CONFIDENCE_THRESHOLD)
    return confidence >= threshold


def decode_uploaded_image(image):
    image_bytes = image.read()
    if not image_bytes:
        return None
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)


def clamp_float(value, lower, upper):
    return max(lower, min(upper, value))


def remove_shadow_with_gaussian(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    value_float = value.astype(np.float32)
    frame_height, frame_width = value.shape[:2]
    work_scale = clamp_float(SHADOW_FILTER_WORK_SCALE, 0.1, 1.0)
    sigma = max(SHADOW_FILTER_SIGMA * work_scale, 1.0)
    if work_scale < 1.0:
        work_width = max(16, int(frame_width * work_scale))
        work_height = max(16, int(frame_height * work_scale))
        value_for_blur = cv2.resize(
            value_float,
            (work_width, work_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        value_for_blur = value_float

    illumination = cv2.GaussianBlur(
        value_for_blur,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    if work_scale < 1.0:
        illumination = cv2.resize(
            illumination,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

    illumination = np.maximum(illumination, 1.0)
    scale = max(float(np.mean(illumination)), 1.0)
    normalized_value = cv2.divide(value_float, illumination, scale=scale)
    normalized_value = np.clip(normalized_value, 0, 255).astype(np.uint8)

    if SHADOW_FILTER_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=SHADOW_FILTER_CLAHE_CLIP,
            tileGridSize=(8, 8),
        )
        normalized_value = clahe.apply(normalized_value)

    strength = clamp_float(SHADOW_FILTER_STRENGTH, 0.0, 1.0)
    corrected_value = cv2.addWeighted(value, 1.0 - strength, normalized_value, strength, 0)
    corrected_hsv = cv2.merge((hue, saturation, corrected_value))
    return cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)


def resize_for_shadow_filter(frame):
    frame_height, frame_width = frame.shape[:2]
    max_side = max(frame_height, frame_width)
    if SHADOW_FILTER_MAX_SIDE <= 0 or max_side <= SHADOW_FILTER_MAX_SIDE:
        return frame, 1.0, 1.0

    resize_scale = SHADOW_FILTER_MAX_SIDE / float(max_side)
    resized_width = max(16, int(frame_width * resize_scale))
    resized_height = max(16, int(frame_height * resize_scale))
    resized_frame = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized_frame, frame_width / float(resized_width), frame_height / float(resized_height)


def preprocess_frame_for_detection(frame):
    if not SHADOW_FILTER_ENABLED:
        return frame, 1.0, 1.0
    resized_frame, scale_x, scale_y = resize_for_shadow_filter(frame)
    return remove_shadow_with_gaussian(resized_frame), scale_x, scale_y


def scale_box_to_original_frame(box, scale_x, scale_y):
    scaled_box = box.copy()
    scaled_box[0] *= scale_x
    scaled_box[2] *= scale_x
    scaled_box[1] *= scale_y
    scaled_box[3] *= scale_y
    return scaled_box


def make_warmup_image():
    sample_path = SCRIPT_DIR / "temp_image.jpg"
    sample_image = cv2.imread(str(sample_path))
    if sample_image is not None:
        return sample_image

    height = int(os.getenv("YOLO_WARMUP_HEIGHT", "1080"))
    width = int(os.getenv("YOLO_WARMUP_WIDTH", "1920"))
    return np.zeros((height, width, 3), dtype=np.uint8)


def resolve_model_path():
    env_model_path = os.getenv("YOLO_MODEL_PATH")
    if env_model_path:
        path = Path(env_model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    for model_path in FINETUNED_MODEL_PATHS:
        if model_path.exists():
            return model_path
    return BASE_MODEL_PATH

app = Flask(__name__)
MODEL_PATH = resolve_model_path()
model = YOLO(str(MODEL_PATH))
model_names = normalize_model_names(model.names)
print(f"Loaded YOLO model: {MODEL_PATH}")
print(f"Model labels: {model_names}")
print(
    "YOLO runtime: "
    f"device={YOLO_DEVICE}, half={YOLO_HALF}, imgsz={YOLO_IMGSZ}, "
    f"conf={MODEL_CONFIDENCE_THRESHOLD}, wall_conf={CLASS_CONFIDENCE_THRESHOLDS['wall']}, "
    f"shadow_filter={SHADOW_FILTER_ENABLED}, shadow_sigma={SHADOW_FILTER_SIGMA}, "
    f"shadow_scale={SHADOW_FILTER_WORK_SCALE}, shadow_max_side={SHADOW_FILTER_MAX_SIDE}, "
    f"cudnn={torch.backends.cudnn.enabled}, "
    f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
)

if USE_CUDA_DEVICE:
    warmup_image, _, _ = preprocess_frame_for_detection(make_warmup_image())
    warmup_started_at = time.perf_counter()
    for _ in range(max(1, YOLO_WARMUP_RUNS)):
        with torch.inference_mode():
            model.predict(
                source=warmup_image,
                conf=MODEL_CONFIDENCE_THRESHOLD,
                imgsz=YOLO_IMGSZ,
                device=YOLO_DEVICE,
                half=YOLO_HALF,
                iou=YOLO_IOU,
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - warmup_started_at) * 1000
    print(f"YOLO CUDA warmup complete ({warmup_ms:.1f} ms, shape={warmup_image.shape})")
combined_commands = [
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "D", "weight": 1.0},
        "turretQE": {"command": "Q", "weight": 0.7},
        "turretRF": {"command": "R", "weight": 0.5},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.6},
        "moveAD": {"command": "A", "weight": 0.4},
        "turretQE": {"command": "E", "weight": 0.8},
        "turretRF": {"command": "R", "weight": 0.3},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.5},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "E", "weight": 0.4},
        "turretRF": {"command": "R", "weight": 0.6},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.3},
        "moveAD": {"command": "D", "weight": 0.3},
        "turretQE": {"command": "E", "weight": 0.5},
        "turretRF": {"command": "R", "weight": 0.7},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "E", "weight": 0.5},
        "turretRF": {"command": "R", "weight": 0.5},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.8},
        "moveAD": {"command": "A", "weight": 0.6},
        "turretQE": {"command": "E", "weight": 0.9},
        "turretRF": {"command": "R", "weight": 0.2},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "D", "weight": 1.0},
        "turretQE": {"command": "E", "weight": 1.0},
        "turretRF": {"command": "R", "weight": 1.0},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.2},
        "moveAD": {"command": "A", "weight": 0.9},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "R", "weight": 0.9},
        "fire": False
    },
    {
        "moveWS": {"command": "S", "weight": 0.4},
        "moveAD": {"command": "D", "weight": 0.4},
        "turretQE": {"command": "E", "weight": 0.6},
        "turretRF": {"command": "F", "weight": 0.6},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.8},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "Q", "weight": 0.5},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False
    },
    {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": True
    },
    {
        "moveWS": {"command": "S", "weight": 0.2},
        "moveAD": {"command": "A", "weight": 0.2},
        "turretQE": {"command": "E", "weight": 0.2},
        "turretRF": {"command": "F", "weight": 0.2},
        "fire": False
    }
]


@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    frame = decode_uploaded_image(image)
    if frame is None:
        return jsonify({"error": "Invalid image received"}), 400

    original_shape = frame.shape
    started_at = time.perf_counter()
    preprocess_started_at = time.perf_counter()
    frame, scale_x, scale_y = preprocess_frame_for_detection(frame)
    preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000
    with torch.inference_mode():
        results = model.predict(
            source=frame,
            conf=MODEL_CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            device=YOLO_DEVICE,
            half=YOLO_HALF,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
            verbose=False,
        )

    boxes = results[0].boxes
    detections = boxes.data.detach().cpu().numpy() if boxes is not None else np.empty((0, 6))
    filtered_results = []
    raw_detections = []
    for box in detections:
        class_id = int(box[5])
        class_name = get_public_class_name(class_id)
        if class_name is None:
            continue
        confidence = float(box[4])
        scaled_box = scale_box_to_original_frame(box, scale_x, scale_y)
        raw_detections.append(f"{class_name}:{confidence:.2f}")
        if not should_return_detection(class_name, confidence, scaled_box, original_shape):
            continue

        filtered_results.append({
            'className': class_name,
            'bbox': [float(coord) for coord in scaled_box[:4]],
            'confidence': confidence,
            'color': '#00FF00',
            'filled': False,
            'updateBoxWhileMoving': False
        })

    if YOLO_DETECT_DEBUG:
        print("Raw detections:", ", ".join(raw_detections) if raw_detections else "none")
        print("Returned detections:", filtered_results)
    if YOLO_TIMING:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(
            f"YOLO detect: {elapsed_ms:.1f} ms, "
            f"preprocess={preprocess_ms:.1f} ms, "
            f"raw={len(raw_detections)}, returned={len(filtered_results)}"
        )
    return jsonify(filtered_results)

@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return jsonify({"result": "error", "message": str(e)}), 500

    return jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    #print("📨 /info data received:", data)

    # Auto-pause after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"status": "success", "control": "pause"})
    # Auto-reset after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"stsaatus": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    if combined_commands:
        command = combined_commands.pop(0)
    else:
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    print("🔁 Sent Combined Action:", command)
    return jsonify(command)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": False,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    config = load_config()

    host = config["simulator"]["host"]
    port = config["simulator"]["port"]
    app.run(host=host, port=port, threaded=True)
