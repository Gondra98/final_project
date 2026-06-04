from flask import Flask, request, jsonify
import os
from pathlib import Path
from threading import Lock
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
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2-2" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "finetune_tankkk2" / "weights" / "best.pt",
]
CLASS_ALIASES = {
    "blue": "person",
    "red": "person",
    "tank": "tank",
}
IGNORED_CLASSES = {"car"}
CLASS_COLORS = {
    "person": "#00FFFF",
    "rock": "#FFA500",
    "tank": "#FF0000",
    "wall": "#00FF00",
    "tent": "#FFFF00",
}
DEFAULT_BOX_COLOR = "#00FF00"
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_MODEL_CONF", "0.10"))
FALLBACK_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_FALLBACK_MODEL_CONF", "0.05"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_DEFAULT_CONF", "0.20"))
CLASS_CONFIDENCE_THRESHOLDS = {
    "wall": float(os.getenv("YOLO_WALL_CONF", "0.15")),
}
CLOSE_WALL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CLOSE_WALL_CONF", "0.12"))
CLOSE_WALL_AREA_RATIO = float(os.getenv("YOLO_CLOSE_WALL_AREA_RATIO", "0.08"))
CLOSE_WALL_MIN_HEIGHT_RATIO = float(os.getenv("YOLO_CLOSE_WALL_MIN_HEIGHT_RATIO", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.70"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "20"))
MAX_RETURN_DETECTIONS = int(os.getenv("YOLO_MAX_RETURN", "5"))
DEBUG_DETECTION_LIMIT = int(os.getenv("YOLO_DEBUG_DET_LIMIT", "10"))


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_CUDA_DEVICE = torch.cuda.is_available() and YOLO_DEVICE.lower() != "cpu"
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "512"))
YOLO_HALF = USE_CUDA_DEVICE and env_flag("YOLO_HALF", True)
YOLO_TIMING = env_flag("YOLO_TIMING", env_flag("DEBUG_PERF_LOG", False))
YOLO_DETECT_DEBUG = env_flag("YOLO_DETECT_DEBUG", False)
YOLO_RECOGNITION_LOG = env_flag("YOLO_RECOGNITION_LOG", True)
YOLO_RECOGNITION_LOG_CACHE = env_flag("YOLO_RECOGNITION_LOG_CACHE", False)
YOLO_RECOGNITION_LOG_EMPTY = env_flag("YOLO_RECOGNITION_LOG_EMPTY", False)
YOLO_WARMUP_RUNS = int(os.getenv("YOLO_WARMUP_RUNS", "2"))
SHADOW_FILTER_ENABLED = env_flag("YOLO_SHADOW_FILTER", False)
SHADOW_FILTER_SIGMA = float(os.getenv("YOLO_SHADOW_SIGMA", "35.0"))
SHADOW_FILTER_STRENGTH = float(os.getenv("YOLO_SHADOW_STRENGTH", "0.75"))
SHADOW_FILTER_WORK_SCALE = float(os.getenv("YOLO_SHADOW_WORK_SCALE", "0.35"))
SHADOW_FILTER_MAX_SIDE = int(os.getenv("YOLO_SHADOW_MAX_SIDE", "960"))
SHADOW_FILTER_CLAHE = env_flag("YOLO_SHADOW_CLAHE", False)
SHADOW_FILTER_CLAHE_CLIP = float(os.getenv("YOLO_SHADOW_CLAHE_CLIP", "1.5"))
ENABLE_DETECT_CACHE = env_flag("YOLO_DETECT_CACHE", True)
YOLO_MIN_INTERVAL = float(os.getenv("YOLO_MIN_INTERVAL", "0.12"))
YOLO_BYPASS_RETURN_FILTER = env_flag("YOLO_BYPASS_RETURN_FILTER", False)
YOLO_LOW_CONF_FALLBACK = env_flag("YOLO_LOW_CONF_FALLBACK", False)
YOLO_RETURN_FALLBACK_DETECTIONS = env_flag("YOLO_RETURN_FALLBACK_DETECTIONS", True)
DETECT_MODE = env_flag("SIM_DETECT_MODE", True)
SERVER_THREADED = env_flag("FLASK_THREADED", False)
SERVER_MODE = "yolo_only_fast"

if USE_CUDA_DEVICE:
    torch.backends.cudnn.benchmark = env_flag("YOLO_CUDNN_BENCHMARK", False)


def load_config(path=CONFIG_PATH):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_model_names(names):
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def normalize_public_class_name(class_name):
    class_name = str(class_name).strip().lower()
    return CLASS_ALIASES.get(class_name, class_name)


def get_box_color(class_name: str) -> str:
    normalized = str(class_name).strip().lower()
    return CLASS_COLORS.get(normalized, DEFAULT_BOX_COLOR)


def get_public_class_name(class_id):
    class_name = model_names.get(class_id)
    if class_name is None:
        return None
    return normalize_public_class_name(class_name)


def get_public_model_names():
    return {
        class_id: normalize_public_class_name(class_name)
        for class_id, class_name in model_names.items()
    }


def get_box_size_ratios(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    box_width = max(0.0, float(x2 - x1))
    box_height = max(0.0, float(y2 - y1))
    frame_area = max(1.0, float(frame_width * frame_height))
    area_ratio = (box_width * box_height) / frame_area
    height_ratio = box_height / max(1.0, float(frame_height))
    return area_ratio, height_ratio


def is_valid_box(box):
    if len(box) < 4:
        return False
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    return x2 > x1 and y2 > y1


def is_close_wall_candidate(class_name, confidence, box, frame_shape):
    if class_name != "wall" or confidence < CLOSE_WALL_CONFIDENCE_THRESHOLD:
        return False
    area_ratio, height_ratio = get_box_size_ratios(box, frame_shape)
    return area_ratio >= CLOSE_WALL_AREA_RATIO or height_ratio >= CLOSE_WALL_MIN_HEIGHT_RATIO


def evaluate_detection_for_return(class_name, confidence, box, frame_shape, bypass_return_filter):
    if class_name is None:
        return False, "class_name_none", None
    if not is_valid_box(box):
        return False, "invalid_box", None
    if class_name in IGNORED_CLASSES:
        return False, "ignored_class", None
    if bypass_return_filter:
        return True, None, None
    if is_close_wall_candidate(class_name, confidence, box, frame_shape):
        return True, None, CLOSE_WALL_CONFIDENCE_THRESHOLD
    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, DEFAULT_CONFIDENCE_THRESHOLD)
    if confidence >= threshold:
        return True, None, threshold
    if class_name in CLASS_CONFIDENCE_THRESHOLDS:
        return False, "below_class_threshold", threshold
    return False, "below_default_threshold", threshold


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
    size = max(32, YOLO_IMGSZ)
    return np.zeros((size, size, 3), dtype=np.uint8)


def resolve_model_path():
    if YOLO_MODEL_PATH_ENV:
        path = Path(YOLO_MODEL_PATH_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    for model_path in FINETUNED_MODEL_PATHS:
        if model_path.exists():
            return model_path
    return BASE_MODEL_PATH


def get_model_path_candidates():
    candidates = [*FINETUNED_MODEL_PATHS, BASE_MODEL_PATH]
    return [
        {
            "path": str(path),
            "exists": path.exists(),
        }
        for path in candidates
    ]

app = Flask(__name__)
YOLO_MODEL_PATH_ENV = os.getenv("YOLO_MODEL_PATH")
MODEL_PATH_FROM_ENV = bool(YOLO_MODEL_PATH_ENV)
MODEL_PATH = resolve_model_path()
model = YOLO(str(MODEL_PATH))
model_names = normalize_model_names(model.names)
public_names = get_public_model_names()
detect_state_lock = Lock()
yolo_predict_lock = Lock()
detect_state = {
    "latest_detections": [],
    "latest_detection_timestamp": 0.0,
    "latest_detect_cached": False,
    "latest_detect_ms": 0.0,
    "latest_decode_ms": 0.0,
    "latest_preprocess_ms": 0.0,
    "latest_yolo_ms": 0.0,
    "latest_postprocess_ms": 0.0,
    "latest_raw_detection_count": 0,
    "latest_returned_detection_count": 0,
    "latest_raw_detections": [],
    "latest_returned_detections": [],
    "latest_rejected_detections": [],
    "latest_cache_reason": None,
    "latest_frame_shape": None,
    "latest_frame_mean": None,
    "latest_frame_std": None,
    "latest_model_conf_used": MODEL_CONFIDENCE_THRESHOLD,
    "latest_fallback_used": False,
}
print(f"YOLO_MODEL_PATH env set: {MODEL_PATH_FROM_ENV}")
print(f"Loaded YOLO model: {MODEL_PATH}")
print(f"Model labels: {model_names}")
print(f"Public labels: {public_names}")
print(
    "YOLO runtime: "
    f"device={YOLO_DEVICE}, half={YOLO_HALF}, imgsz={YOLO_IMGSZ}, "
    f"model_conf={MODEL_CONFIDENCE_THRESHOLD}, default_conf={DEFAULT_CONFIDENCE_THRESHOLD}, "
    f"fallback_conf={FALLBACK_MODEL_CONFIDENCE_THRESHOLD}, low_conf_fallback={YOLO_LOW_CONF_FALLBACK}, "
    f"wall_conf={CLASS_CONFIDENCE_THRESHOLDS['wall']}, "
    f"max_det={YOLO_MAX_DET}, max_return={MAX_RETURN_DETECTIONS}, "
    f"cache={ENABLE_DETECT_CACHE}, min_interval={YOLO_MIN_INTERVAL}, "
    f"bypass_return_filter={YOLO_BYPASS_RETURN_FILTER}, "
    f"flask_threaded={SERVER_THREADED}, "
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
print("YOLO server ready")


def get_cuda_device_name():
    if not torch.cuda.is_available():
        return None
    try:
        device_index = 0 if YOLO_DEVICE.lower() == "cuda" else int(YOLO_DEVICE)
    except ValueError:
        device_index = 0
    return torch.cuda.get_device_name(device_index)


def get_cached_detections(now_seconds):
    if not ENABLE_DETECT_CACHE:
        return None
    with detect_state_lock:
        latest_timestamp = detect_state["latest_detection_timestamp"]
        if latest_timestamp <= 0.0:
            return None
        if now_seconds - latest_timestamp > YOLO_MIN_INTERVAL:
            return None
        return list(detect_state["latest_detections"]), latest_timestamp


def get_latest_detections():
    with detect_state_lock:
        latest_timestamp = detect_state["latest_detection_timestamp"]
        if latest_timestamp <= 0.0:
            return [], latest_timestamp
        return list(detect_state["latest_detections"]), latest_timestamp


def update_detect_state(
    detections,
    detection_timestamp,
    detect_ms,
    decode_ms,
    preprocess_ms,
    yolo_ms,
    postprocess_ms,
    raw_detection_count,
    cached,
    raw_detections=None,
    rejected_detections=None,
    cache_reason=None,
    frame_shape=None,
    frame_mean=None,
    frame_std=None,
    model_conf_used=None,
    fallback_used=False,
):
    with detect_state_lock:
        detect_state["latest_detections"] = list(detections)
        detect_state["latest_detection_timestamp"] = detection_timestamp
        detect_state["latest_detect_cached"] = cached
        detect_state["latest_cache_reason"] = cache_reason
        detect_state["latest_detect_ms"] = detect_ms
        if not cached:
            detect_state["latest_decode_ms"] = decode_ms
            detect_state["latest_preprocess_ms"] = preprocess_ms
            detect_state["latest_yolo_ms"] = yolo_ms
            detect_state["latest_postprocess_ms"] = postprocess_ms
            detect_state["latest_raw_detection_count"] = raw_detection_count
            detect_state["latest_raw_detections"] = list(raw_detections or [])[:DEBUG_DETECTION_LIMIT]
            detect_state["latest_rejected_detections"] = list(rejected_detections or [])[:DEBUG_DETECTION_LIMIT]
            detect_state["latest_frame_shape"] = frame_shape
            detect_state["latest_frame_mean"] = frame_mean
            detect_state["latest_frame_std"] = frame_std
            detect_state["latest_model_conf_used"] = model_conf_used
            detect_state["latest_fallback_used"] = fallback_used
        detect_state["latest_returned_detection_count"] = len(detections)
        detect_state["latest_returned_detections"] = list(detections)[:DEBUG_DETECTION_LIMIT]


def log_detect_perf(
    decode_ms,
    preprocess_ms,
    yolo_ms,
    postprocess_ms,
    total_ms,
    raw_detection_count,
    returned_detection_count,
    cached,
):
    if not YOLO_TIMING:
        return
    print(
        "[perf:/detect] "
        f"decode={decode_ms:.1f}ms "
        f"preprocess={preprocess_ms:.1f}ms "
        f"yolo={yolo_ms:.1f}ms "
        f"post={postprocess_ms:.1f}ms "
        f"total={total_ms:.1f}ms "
        f"raw={raw_detection_count} "
        f"returned={returned_detection_count} "
        f"cached={cached}"
    )


def return_cached_response(started_at, detections, detection_timestamp, reason):
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    update_detect_state(
        detections,
        detection_timestamp,
        elapsed_ms,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        True,
        cache_reason=reason,
    )
    if YOLO_TIMING:
        print(f"[perf:/detect] cached response reason={reason}")
    log_detect_perf(0.0, 0.0, 0.0, 0.0, elapsed_ms, 0, len(detections), True)
    log_recognized_detections(detections, cached=True)
    return jsonify(detections)


def log_recognized_detections(detections, cached=False):
    if not YOLO_RECOGNITION_LOG:
        return
    if cached and not YOLO_RECOGNITION_LOG_CACHE:
        return
    if not detections:
        if YOLO_RECOGNITION_LOG_EMPTY:
            print("[detect] no object recognized")
        return

    print(f"[detect] {len(detections)} object(s) recognized")
    for detection in detections:
        bbox = detection.get("bbox", [])
        bbox_text = ", ".join(f"{float(coord):.1f}" for coord in bbox[:4])
        print(
            f"[detect] class={detection.get('className')} "
            f"conf={float(detection.get('confidence', 0.0)):.2f} "
            f"bbox=[{bbox_text}]"
        )


def make_detection_response(class_name, box, confidence):
    return {
        "className": class_name,
        "bbox": [float(coord) for coord in box[:4]],
        "confidence": confidence,
        "color": get_box_color(class_name),
        "filled": False,
        "updateBoxWhileMoving": False,
    }


def make_debug_detection(
    model_class_name,
    class_name,
    box,
    confidence,
    returned,
    reject_reason,
    threshold,
):
    item = {
        "modelClassName": model_class_name,
        "className": class_name,
        "bbox": [float(coord) for coord in box[:4]],
        "confidence": confidence,
        "returned": returned,
        "rejectReason": reject_reason,
        "threshold": threshold,
    }
    return item


def result_box_count(results):
    if not results:
        return 0
    boxes = results[0].boxes
    if boxes is None:
        return 0
    return len(boxes)


def get_debug_state_payload():
    with detect_state_lock:
        state = dict(detect_state)
    return {
        "serverMode": SERVER_MODE,
        "modelPath": str(MODEL_PATH),
        "modelPathFromEnv": MODEL_PATH_FROM_ENV,
        "modelPathEnvValue": YOLO_MODEL_PATH_ENV,
        "modelPathCandidates": get_model_path_candidates(),
        "modelNames": model_names,
        "publicNames": public_names,
        "yoloImgsz": YOLO_IMGSZ,
        "modelConf": MODEL_CONFIDENCE_THRESHOLD,
        "fallbackModelConf": FALLBACK_MODEL_CONFIDENCE_THRESHOLD,
        "lowConfFallbackEnabled": YOLO_LOW_CONF_FALLBACK,
        "returnFallbackDetections": YOLO_RETURN_FALLBACK_DETECTIONS,
        "defaultConf": DEFAULT_CONFIDENCE_THRESHOLD,
        "classThresholds": CLASS_CONFIDENCE_THRESHOLDS,
        "closeWallConf": CLOSE_WALL_CONFIDENCE_THRESHOLD,
        "ignoredClasses": sorted(IGNORED_CLASSES),
        "classColors": CLASS_COLORS,
        "defaultBoxColor": DEFAULT_BOX_COLOR,
        "recognitionLogEnabled": YOLO_RECOGNITION_LOG,
        "recognitionLogCacheEnabled": YOLO_RECOGNITION_LOG_CACHE,
        "recognitionLogEmptyEnabled": YOLO_RECOGNITION_LOG_EMPTY,
        "bypassReturnFilter": YOLO_BYPASS_RETURN_FILTER,
        "yoloIou": YOLO_IOU,
        "yoloMaxDet": YOLO_MAX_DET,
        "maxReturnDetections": MAX_RETURN_DETECTIONS,
        "debugDetectionLimit": DEBUG_DETECTION_LIMIT,
        "shadowFilterEnabled": SHADOW_FILTER_ENABLED,
        "detectCacheEnabled": ENABLE_DETECT_CACHE,
        "yoloMinInterval": YOLO_MIN_INTERVAL,
        "detectMode": DETECT_MODE,
        "flaskThreaded": SERVER_THREADED,
        "latestCacheReason": state["latest_cache_reason"],
        "latestDetectMs": state["latest_detect_ms"],
        "latestDecodeMs": state["latest_decode_ms"],
        "latestYoloMs": state["latest_yolo_ms"],
        "latestPreprocessMs": state["latest_preprocess_ms"],
        "latestPostprocessMs": state["latest_postprocess_ms"],
        "latestRawDetectionCount": state["latest_raw_detection_count"],
        "latestReturnedDetectionCount": state["latest_returned_detection_count"],
        "latestRawDetections": state["latest_raw_detections"],
        "latestReturnedDetections": state["latest_returned_detections"],
        "latestRejectedDetections": state["latest_rejected_detections"],
        "latestFrameShape": state["latest_frame_shape"],
        "latestFrameMean": state["latest_frame_mean"],
        "latestFrameStd": state["latest_frame_std"],
        "latestModelConfUsed": state["latest_model_conf_used"],
        "latestFallbackUsed": state["latest_fallback_used"],
        "latestDetectCached": state["latest_detect_cached"],
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDeviceName": get_cuda_device_name(),
    }
# Fire/destruction is deferred; the current server only exercises movement and perception.
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
        "fire": False
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
        "fire": False
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
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "D", "weight": 1.0},
        "turretQE": {"command": "E", "weight": 1.0},
        "turretRF": {"command": "R", "weight": 1.0},
        "fire": False
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
        "fire": False
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
        "fire": False
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
    started_at = time.perf_counter()
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    cached = get_cached_detections(time.time())
    if cached is not None:
        cached_results, cached_timestamp = cached
        return return_cached_response(started_at, cached_results, cached_timestamp, "fresh_interval")

    if not yolo_predict_lock.acquire(blocking=False):
        latest_results, latest_timestamp = get_latest_detections()
        return return_cached_response(started_at, latest_results, latest_timestamp, "inference_busy")

    try:
        decode_started_at = time.perf_counter()
        frame = decode_uploaded_image(image)
        decode_ms = (time.perf_counter() - decode_started_at) * 1000
        if frame is None:
            return jsonify({"error": "Invalid image received"}), 400

        original_shape = frame.shape
        frame_shape = [int(value) for value in original_shape]
        frame_mean = float(np.mean(frame))
        frame_std = float(np.std(frame))
        preprocess_started_at = time.perf_counter()
        frame, scale_x, scale_y = preprocess_frame_for_detection(frame)
        preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000

        yolo_started_at = time.perf_counter()
        model_conf_used = MODEL_CONFIDENCE_THRESHOLD
        fallback_used = False
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
            if (
                YOLO_LOW_CONF_FALLBACK
                and FALLBACK_MODEL_CONFIDENCE_THRESHOLD < MODEL_CONFIDENCE_THRESHOLD
                and result_box_count(results) == 0
            ):
                fallback_used = True
                model_conf_used = FALLBACK_MODEL_CONFIDENCE_THRESHOLD
                results = model.predict(
                    source=frame,
                    conf=FALLBACK_MODEL_CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMGSZ,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    iou=YOLO_IOU,
                    max_det=YOLO_MAX_DET,
                    verbose=False,
                )
        yolo_ms = (time.perf_counter() - yolo_started_at) * 1000
    finally:
        yolo_predict_lock.release()

    postprocess_started_at = time.perf_counter()
    boxes = results[0].boxes
    detections = boxes.data.detach().cpu().numpy() if boxes is not None else np.empty((0, 6))
    filtered_results = []
    raw_detections = []
    rejected_detections = []
    for box in detections:
        class_id = int(box[5])
        model_class_name = model_names.get(class_id)
        class_name = get_public_class_name(class_id)
        confidence = float(box[4])
        scaled_box = scale_box_to_original_frame(box, scale_x, scale_y)
        bypass_return_filter = YOLO_BYPASS_RETURN_FILTER or (
            fallback_used and YOLO_RETURN_FALLBACK_DETECTIONS
        )
        returned, reject_reason, threshold = evaluate_detection_for_return(
            class_name,
            confidence,
            scaled_box,
            original_shape,
            bypass_return_filter,
        )
        debug_detection = make_debug_detection(
            model_class_name,
            class_name,
            scaled_box,
            confidence,
            returned,
            reject_reason,
            threshold,
        )
        raw_detections.append(debug_detection)
        if not returned:
            rejected_detections.append(debug_detection)
            continue

        filtered_results.append(make_detection_response(class_name, scaled_box, confidence))

    filtered_results.sort(key=lambda detection: detection["confidence"], reverse=True)
    if not YOLO_BYPASS_RETURN_FILTER:
        filtered_results = filtered_results[:MAX_RETURN_DETECTIONS]
    postprocess_ms = (time.perf_counter() - postprocess_started_at) * 1000
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    update_detect_state(
        filtered_results,
        time.time(),
        elapsed_ms,
        decode_ms,
        preprocess_ms,
        yolo_ms,
        postprocess_ms,
        len(raw_detections),
        False,
        raw_detections=raw_detections,
        rejected_detections=rejected_detections,
        frame_shape=frame_shape,
        frame_mean=frame_mean,
        frame_std=frame_std,
        model_conf_used=model_conf_used,
        fallback_used=fallback_used,
    )
    if YOLO_DETECT_DEBUG:
        raw_summary = ", ".join(
            f"{item['className']}:{item['confidence']:.2f}:{item['rejectReason'] or 'returned'}"
            for item in raw_detections
        )
        print("Raw detections:", raw_summary if raw_summary else "none")
        print("Returned detections:", filtered_results)
    log_detect_perf(
        decode_ms,
        preprocess_ms,
        yolo_ms,
        postprocess_ms,
        elapsed_ms,
        len(raw_detections),
        len(filtered_results),
        False,
    )
    log_recognized_detections(filtered_results, cached=False)
    return jsonify(filtered_results)


@app.route('/debug_state', methods=['GET'])
@app.route('/debug/perf', methods=['GET'])
def debug_state():
    return jsonify(get_debug_state_payload())


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
        "detectMode": DETECT_MODE,
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
    app.run(host=host, port=port, threaded=SERVER_THREADED)
