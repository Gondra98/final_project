"""
Tank 시뮬레이터와 YOLO 모델을 연결하는 Flask 서버.

주 사용처:
- 시뮬레이터가 detectMode로 실행될 때 이미지 프레임을 받아 객체 탐지를 수행한다.
- 탐지 결과를 시뮬레이터 API가 요구하는 `className`, `bbox`, `confidence` 형태로 맞춰준다.
- 단순 탐지뿐 아니라 초기 설정(`/init`)과 테스트용 이동 명령(`/get_action`)도 함께 제공한다.

큰 흐름:
1. 환경변수와 `configs/simulator.yaml`에서 실행 설정을 읽는다.
2. 사용할 YOLO `best_final.engine` 모델을 고르고 서버 시작 시 한 번 로드한다.
3. 시뮬레이터가 `/detect`로 보낸 이미지를 YOLO로 추론한다.
4. 탐지 결과를 시뮬레이터가 기대하는 JSON 형식으로 필터링해 반환한다.
5. `/init`, `/get_action` 등 시뮬레이터 제어용 API도 같은 서버에서 처리한다.
"""

from flask import Flask, request, jsonify
import os
import math
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

# 프로젝트 경로, 기본 모델, 탐지 임계값 같은 전역 실행 설정입니다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "simulator.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tank_detector" / "best_final.engine"
DEFAULT_FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "tank_detector" / "best.pt"

# 큰 흐름:
# 1. 서버 시작 시 설정값과 YOLO 모델을 로드합니다. TensorRT engine이 실패하면 best.pt로 fallback할 수 있습니다.
# 2. simulator가 /detect로 이미지를 보내면 서버가 이미지를 디코딩하고 YOLO 추론을 실행합니다.
# 3. 모델 class id를 simulator가 쓰는 공개 className으로 변환한 뒤 confidence, 무시 클래스, 반환 개수 기준으로 필터링합니다.
# 4. 마지막 탐지 결과와 성능 지표는 detect_state에 저장되어 /debug_state에서 확인할 수 있습니다.
# 5. /init, /info, /get_action 등은 simulator 실행에 필요한 제어/상태 API를 같은 Flask 서버에서 제공합니다.
# `YOLO_MODEL_PATH` 환경변수를 주면 기본 모델 대신 해당 weight를 사용합니다.
CLASS_ALIASES = {
    "blue": "person",
    "red": "person",
    "tank": "tank",
}
# 학습 데이터에는 남아 있어도 시뮬레이터 제어에는 쓰지 않는 클래스입니다.
def parse_ignored_classes(default_value):
    value = os.getenv("YOLO_IGNORED_CLASSES", default_value)
    return {item.strip().lower() for item in value.split(",") if item.strip()}


IGNORED_CLASSES = parse_ignored_classes("car,wall")
CLASS_COLORS_BGR = {
    "tank": (0, 0, 255),
    "wall": (255, 0, 0),
    "rock": (0, 255, 255),
    "person": (0, 255, 0),
    "tent": (255, 255, 0),
    "car": (255, 0, 255),
}
CLASS_COLOR_PALETTE_BGR = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]
CLASS_HEIGHTS_M = {
    "tank": float(os.getenv("YOLO_DISTANCE_TANK_HEIGHT_M", "2.4")),
    "wall": float(os.getenv("YOLO_DISTANCE_WALL_HEIGHT_M", "2.0")),
    "rock": float(os.getenv("YOLO_DISTANCE_ROCK_HEIGHT_M", "0.8")),
    "person": float(os.getenv("YOLO_DISTANCE_PERSON_HEIGHT_M", "1.7")),
    "tent": float(os.getenv("YOLO_DISTANCE_TENT_HEIGHT_M", "1.6")),
    "car": float(os.getenv("YOLO_DISTANCE_CAR_HEIGHT_M", "1.5")),
}


def get_class_bgr_color(class_name, class_id=0):
    normalized_name = str(class_name).strip().lower()
    if normalized_name in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[normalized_name]
    return CLASS_COLOR_PALETTE_BGR[class_id % len(CLASS_COLOR_PALETTE_BGR)]


def bgr_to_hex(color):
    blue, green, red = (int(value) for value in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def get_class_hex_color(class_name, class_id=0):
    return bgr_to_hex(get_class_bgr_color(class_name, class_id))


def get_class_height_m(class_name):
    return CLASS_HEIGHTS_M.get(str(class_name).strip().lower())


def estimate_distance_by_height(class_name, box, frame_shape):
    if box is None or len(box) < 4 or frame_shape is None:
        return None
    reference_height_m = get_class_height_m(class_name)
    if reference_height_m is None or reference_height_m <= 0:
        return None

    box_height_px = max(0.0, float(box[3]) - float(box[1]))
    if box_height_px < YOLO_DISTANCE_MIN_BOX_HEIGHT:
        return None

    frame_height_px = max(1.0, float(frame_shape[0]))
    if not 0 < YOLO_DISTANCE_FOV_DEG < 180:
        return None
    half_fov_rad = math.radians(YOLO_DISTANCE_FOV_DEG) / 2.0
    focal_length_px = frame_height_px / (2.0 * math.tan(half_fov_rad))
    distance_m = (reference_height_m * focal_length_px) / box_height_px
    return round(float(distance_m), 2)

# 모델 입력 단계의 confidence와 응답 반환 단계의 confidence를 분리했습니다.
# 낮은 모델 confidence는 후보를 넓게 잡기 위한 값이고, DEFAULT_CONFIDENCE_THRESHOLD가 실제 반환 필터입니다.
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_MODEL_CONF", "0.10"))
FALLBACK_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_FALLBACK_MODEL_CONF", "0.05"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_DEFAULT_CONF", "0.20"))
# 벽은 가까이 있을 때 박스가 크게 잡히므로 일반 객체보다 낮은 confidence도 허용합니다.
CLOSE_WALL_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CLOSE_WALL_CONF", "0.12"))
CLOSE_WALL_AREA_RATIO = float(os.getenv("YOLO_CLOSE_WALL_AREA_RATIO", "0.08"))
CLOSE_WALL_MIN_HEIGHT_RATIO = float(os.getenv("YOLO_CLOSE_WALL_MIN_HEIGHT_RATIO", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.70"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "20"))
YOLO_DISTANCE_FOV_DEG = float(os.getenv("YOLO_DISTANCE_FOV_DEG", "60"))
YOLO_DISTANCE_MIN_BOX_HEIGHT = float(os.getenv("YOLO_DISTANCE_MIN_BOX_HEIGHT", "2"))
MAX_RETURN_DETECTIONS = int(os.getenv("YOLO_MAX_RETURN", "30"))
DEBUG_DETECTION_LIMIT = int(os.getenv("YOLO_DEBUG_DET_LIMIT", "10"))


def env_flag(name, default=False):
    """환경변수 문자열을 bool로 해석합니다. 값이 없으면 호출자가 준 기본값을 사용합니다."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# CUDA가 있으면 기본적으로 GPU 0번을 사용하고, 명시적으로 cpu를 지정하면 CPU로 고정합니다.
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_CUDA_DEVICE = torch.cuda.is_available() and YOLO_DEVICE.lower() != "cpu"
REQUESTED_YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "416"))
YOLO_IMGSZ = REQUESTED_YOLO_IMGSZ
YOLO_HALF = USE_CUDA_DEVICE and env_flag("YOLO_HALF", True)
YOLO_TIMING = env_flag("YOLO_TIMING", env_flag("DEBUG_PERF_LOG", False))
YOLO_DETECT_DEBUG = env_flag("YOLO_DETECT_DEBUG", False)
YOLO_RECOGNITION_LOG = env_flag("YOLO_RECOGNITION_LOG", True)
YOLO_RECOGNITION_LOG_CACHE = env_flag("YOLO_RECOGNITION_LOG_CACHE", False)
YOLO_RECOGNITION_LOG_EMPTY = env_flag("YOLO_RECOGNITION_LOG_EMPTY", False)
YOLO_WARMUP_RUNS = int(os.getenv("YOLO_WARMUP_RUNS", "2"))
# 짧은 시간 안에 중복 요청이 오면 마지막 결과를 재사용해서 시뮬레이터 프레임 드롭을 줄입니다.
ENABLE_DETECT_CACHE = env_flag("YOLO_DETECT_CACHE", True)
YOLO_MIN_INTERVAL = float(os.getenv("YOLO_MIN_INTERVAL", "0.12"))
YOLO_BYPASS_RETURN_FILTER = env_flag("YOLO_BYPASS_RETURN_FILTER", False)
YOLO_ALLOW_MODEL_FALLBACK = env_flag("YOLO_ALLOW_MODEL_FALLBACK", True)
# 아무 후보도 없을 때만 낮은 confidence로 한 번 더 추론하는 디버그/실험용 옵션입니다.
YOLO_LOW_CONF_FALLBACK = env_flag("YOLO_LOW_CONF_FALLBACK", False)
YOLO_RETURN_FALLBACK_DETECTIONS = env_flag("YOLO_RETURN_FALLBACK_DETECTIONS", True)
DETECT_MODE = env_flag("SIM_DETECT_MODE", True)
SERVER_THREADED = env_flag("FLASK_THREADED", False)
SERVER_MODE = "yolo_only_fast"

if USE_CUDA_DEVICE:
    torch.backends.cudnn.benchmark = env_flag("YOLO_CUDNN_BENCHMARK", False)


# 설정 로드, 라벨 정리, 박스 검증처럼 여러 단계에서 공유하는 유틸 함수들입니다.
def load_config(path=CONFIG_PATH):
    """시뮬레이터 host/port 같은 실행 설정을 YAML에서 읽습니다."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_model_names(names):
    """Ultralytics가 dict/list 어느 형태로 주든 class id -> 이름 dict로 통일합니다."""
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def normalize_public_class_name(class_name):
    """학습 라벨명을 시뮬레이터에 노출할 공개 라벨명으로 변환합니다."""
    class_name = str(class_name).strip().lower()
    return CLASS_ALIASES.get(class_name, class_name)


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
    """박스가 전체 화면에서 차지하는 면적/높이 비율을 계산합니다."""
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
    """큰 벽 박스는 낮은 confidence라도 시뮬레이터 회피에 중요하므로 따로 판정합니다."""
    if class_name != "wall" or confidence < CLOSE_WALL_CONFIDENCE_THRESHOLD:
        return False
    area_ratio, height_ratio = get_box_size_ratios(box, frame_shape)
    return area_ratio >= CLOSE_WALL_AREA_RATIO or height_ratio >= CLOSE_WALL_MIN_HEIGHT_RATIO


def evaluate_detection_for_return(class_name, confidence, box, frame_shape, bypass_return_filter):
    """모델 후보 하나를 실제 JSON 응답에 포함할지 결정하고, 거절 이유를 함께 반환합니다."""
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
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if confidence >= threshold:
        return True, None, threshold
    return False, "below_default_threshold", threshold


def decode_uploaded_image(image):
    """Flask 업로드 파일을 OpenCV BGR 이미지 배열로 변환합니다."""
    image_bytes = image.read()
    if not image_bytes:
        return None
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)


def make_warmup_image():
    """CUDA warm-up에 사용할 더미 이미지를 만듭니다."""
    size = max(32, YOLO_IMGSZ)
    return np.zeros((size, size, 3), dtype=np.uint8)


def resolve_model_path():
    """환경변수가 있으면 해당 weight를, 없으면 정해진 최종 weight 파일을 선택합니다."""
    if YOLO_MODEL_PATH_ENV:
        path = Path(YOLO_MODEL_PATH_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_MODEL_PATH


def resolve_fallback_model_path():
    if YOLO_FALLBACK_MODEL_PATH_ENV:
        path = Path(YOLO_FALLBACK_MODEL_PATH_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_FALLBACK_MODEL_PATH


def get_model_path_candidates():
    """디버그 화면에서 현재 사용할 모델 경로가 존재하는지 확인할 수 있도록 목록을 만듭니다."""
    candidates = []
    if YOLO_MODEL_PATH_ENV:
        env_path = Path(YOLO_MODEL_PATH_ENV)
        candidates.append(env_path if env_path.is_absolute() else PROJECT_ROOT / env_path)
    candidates.append(DEFAULT_MODEL_PATH)
    fallback_path = resolve_fallback_model_path()
    if fallback_path not in candidates:
        candidates.append(fallback_path)
    return [
        {
            "path": str(path),
            "exists": path.exists(),
        }
        for path in candidates
    ]


# Flask 앱, YOLO 모델, 최신 탐지 상태를 서버 시작 시 한 번 초기화합니다.
def ensure_model_runtime(model_path):
    if model_path.suffix.lower() != ".engine":
        return
    try:
        __import__("tensorrt")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorRT Python package is required to load .engine models. "
            "Install it in the active environment with: "
            "python -m pip install --upgrade tensorrt-cu12"
        ) from exc


def get_tensorrt_engine_imgsz(model_path):
    if model_path.suffix.lower() != ".engine":
        return None
    try:
        import json
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with model_path.open("rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                json.loads(f.read(meta_len).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                f.seek(0)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None
        is_trt10 = hasattr(engine, "num_io_tensors")
        indices = range(engine.num_io_tensors) if is_trt10 else range(engine.num_bindings)
        for index in indices:
            if is_trt10:
                tensor_name = engine.get_tensor_name(index)
                if engine.get_tensor_mode(tensor_name) != trt.TensorIOMode.INPUT:
                    continue
                shape = tuple(int(dim) for dim in engine.get_tensor_shape(tensor_name))
            else:
                if not engine.binding_is_input(index):
                    continue
                shape = tuple(int(dim) for dim in engine.get_binding_shape(index))
            if len(shape) >= 4 and shape[1] in {1, 3, 4}:
                height, width = shape[2], shape[3]
            elif len(shape) >= 3:
                height, width = shape[1], shape[2]
            else:
                continue
            if height > 0 and height == width:
                return height
    except Exception:
        return None
    return None


def get_engine_load_error_message(model_path):
    return (
        f"Failed to load TensorRT engine: {model_path}\n"
        "TensorRT .engine files are tied to the platform/GPU/TensorRT runtime "
        "they were built for. Re-export best_final.engine on this machine from "
        "the original .pt or .onnx model.\n"
        "Example:\n"
        "  yolo export model=models/tank_detector/best.pt format=engine "
        "task=detect imgsz=416 half=True device=0\n"
        "Then rename/copy the generated engine to "
        "models/tank_detector/best_final.engine."
    )


def load_yolo_model(model_path):
    try:
        loaded_model = YOLO(str(model_path), task="detect")
        loaded_names = normalize_model_names(loaded_model.names)
    except (AttributeError, RuntimeError, ValueError) as exc:
        if model_path.suffix.lower() == ".engine":
            raise RuntimeError(get_engine_load_error_message(model_path)) from exc
        raise
    return loaded_model, loaded_names


def load_yolo_model_with_fallback(model_path, fallback_path):
    try:
        ensure_model_runtime(model_path)
        loaded_model, loaded_names = load_yolo_model(model_path)
        return model_path, loaded_model, loaded_names, False, None
    except (AttributeError, RuntimeError, ValueError, OSError, ModuleNotFoundError) as exc:
        can_fallback = (
            model_path.suffix.lower() == ".engine"
            and YOLO_ALLOW_MODEL_FALLBACK
            and fallback_path.exists()
            and fallback_path.resolve() != model_path.resolve()
        )
        if not can_fallback:
            raise
        print(f"[WARNING] Could not load requested TensorRT engine: {model_path}")
        print(f"[WARNING] {exc}")
        print(f"[WARNING] Falling back to YOLO model: {fallback_path}")
        ensure_model_runtime(fallback_path)
        loaded_model, loaded_names = load_yolo_model(fallback_path)
        return fallback_path, loaded_model, loaded_names, True, str(exc)


app = Flask(__name__)
YOLO_MODEL_PATH_ENV = os.getenv("YOLO_MODEL_PATH")
YOLO_FALLBACK_MODEL_PATH_ENV = os.getenv("YOLO_FALLBACK_MODEL_PATH")
MODEL_PATH_FROM_ENV = bool(YOLO_MODEL_PATH_ENV)
MODEL_PATH = resolve_model_path()
REQUESTED_MODEL_PATH = MODEL_PATH
MODEL_FALLBACK_PATH = resolve_fallback_model_path()

# 모델은 서버 시작 시 한 번만 올립니다. 이후 /detect 요청은 이 전역 model 객체를 재사용합니다.
print(f"Loading YOLO model: {MODEL_PATH}")
MODEL_PATH, model, model_names, MODEL_LOAD_FALLBACK_USED, MODEL_LOAD_ERROR = (
    load_yolo_model_with_fallback(MODEL_PATH, MODEL_FALLBACK_PATH)
)
engine_imgsz = get_tensorrt_engine_imgsz(MODEL_PATH)
if engine_imgsz is not None and YOLO_IMGSZ != engine_imgsz:
    print(
        f"Overriding YOLO_IMGSZ={YOLO_IMGSZ} to TensorRT engine input size "
        f"{engine_imgsz}."
    )
    YOLO_IMGSZ = engine_imgsz
public_names = get_public_model_names()
detect_state_lock = Lock()
# Ultralytics 추론은 무거운 작업이라 요청이 겹칠 때 동시에 여러 번 돌리지 않도록 별도 lock을 둡니다.
yolo_predict_lock = Lock()
# `/detect`가 마지막으로 처리한 결과와 성능 정보를 모아 `/debug_state`와 캐시 응답에서 공유합니다.
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
print(f"Requested YOLO model: {REQUESTED_MODEL_PATH}")
print(f"Loaded YOLO model: {MODEL_PATH}")
print(f"Model load fallback used: {MODEL_LOAD_FALLBACK_USED}")
print(f"Model labels: {model_names}")
print(f"Public labels: {public_names}")
print(
    "YOLO runtime: "
    f"device={YOLO_DEVICE}, half={YOLO_HALF}, imgsz={YOLO_IMGSZ}, "
    f"model_conf={MODEL_CONFIDENCE_THRESHOLD}, default_conf={DEFAULT_CONFIDENCE_THRESHOLD}, "
    f"fallback_conf={FALLBACK_MODEL_CONFIDENCE_THRESHOLD}, low_conf_fallback={YOLO_LOW_CONF_FALLBACK}, "
    f"max_det={YOLO_MAX_DET}, max_return={MAX_RETURN_DETECTIONS}, "
    f"cache={ENABLE_DETECT_CACHE}, min_interval={YOLO_MIN_INTERVAL}, "
    f"bypass_return_filter={YOLO_BYPASS_RETURN_FILTER}, "
    f"flask_threaded={SERVER_THREADED}, "
    f"cudnn={torch.backends.cudnn.enabled}, "
    f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
)

if USE_CUDA_DEVICE:
    # CUDA 첫 추론 지연을 줄이기 위해 더미 이미지로 모델을 미리 한 번 돌립니다.
    warmup_image = make_warmup_image()
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


# 캐시, 로그, 디버그 상태는 `/detect` 요청을 빠르게 처리하고 문제를 추적하기 위한 보조 흐름입니다.
def get_cuda_device_name():
    """디버그 응답에 표시할 CUDA 장치 이름을 안전하게 조회합니다."""
    if not torch.cuda.is_available():
        return None
    try:
        device_index = 0 if YOLO_DEVICE.lower() == "cuda" else int(YOLO_DEVICE)
    except ValueError:
        device_index = 0
    return torch.cuda.get_device_name(device_index)


def get_cached_detections(now_seconds):
    """최근 탐지 결과가 아직 신선하면 YOLO를 다시 돌리지 않고 재사용합니다."""
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
    """추론이 이미 진행 중일 때 즉시 반환할 마지막 결과를 읽습니다."""
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
    """최신 탐지 결과와 디버그 지표를 한 번에 갱신합니다."""
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
    """성능 로그 옵션이 켜져 있을 때 `/detect` 단계별 시간을 출력합니다."""
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
    """캐시로 응답할 때도 디버그 상태와 로그 형식을 일반 응답과 맞춥니다."""
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
    """인식된 객체를 콘솔에서 빠르게 확인하기 위한 선택 로그입니다."""
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
        distance = detection.get("distance")
        distance_text = "N/A" if distance is None else f"{float(distance):.2f}m"
        print(
            f"[detect] class={detection.get('className')} "
            f"conf={float(detection.get('confidence', 0.0)):.2f} "
            f"distance={distance_text} "
            f"bbox=[{bbox_text}]"
        )


def make_detection_response(class_name, box, confidence, class_id=0, frame_shape=None):
    """시뮬레이터가 기대하는 detection JSON 필드만 남깁니다."""
    distance = estimate_distance_by_height(class_name, box, frame_shape)
    return {
        "className": class_name,
        "bbox": [float(coord) for coord in box[:4]],
        "confidence": confidence,
        "distance": distance,
        "color": get_class_hex_color(class_name, class_id),
        "filled": False,
        "updateBoxWhileMoving": False,
    }


def make_debug_detection(
    model_class_name,
    class_name,
    class_id,
    box,
    confidence,
    returned,
    reject_reason,
    threshold,
    frame_shape=None,
):
    """필터 통과/거절 이유까지 포함한 디버그용 detection 항목을 만듭니다."""
    distance = estimate_distance_by_height(class_name, box, frame_shape)
    item = {
        "modelClassName": model_class_name,
        "className": class_name,
        "bbox": [float(coord) for coord in box[:4]],
        "confidence": confidence,
        "distance": distance,
        "color": get_class_hex_color(class_name, class_id),
        "returned": returned,
        "rejectReason": reject_reason,
        "threshold": threshold,
    }
    return item


def result_box_count(results):
    """fallback 추론 여부를 판단하기 위해 Ultralytics 결과의 박스 수를 셉니다."""
    if not results:
        return 0
    boxes = results[0].boxes
    if boxes is None:
        return 0
    return len(boxes)


def get_debug_state_payload():
    """현재 서버 설정과 마지막 탐지 상태를 `/debug_state` 응답으로 묶습니다."""
    with detect_state_lock:
        state = dict(detect_state)
    return {
        "serverMode": SERVER_MODE,
        "modelPath": str(MODEL_PATH),
        "requestedModelPath": str(REQUESTED_MODEL_PATH),
        "modelPathFromEnv": MODEL_PATH_FROM_ENV,
        "modelPathEnvValue": YOLO_MODEL_PATH_ENV,
        "modelFallbackPath": str(MODEL_FALLBACK_PATH),
        "modelFallbackPathEnvValue": YOLO_FALLBACK_MODEL_PATH_ENV,
        "modelFallbackAllowed": YOLO_ALLOW_MODEL_FALLBACK,
        "modelLoadFallbackUsed": MODEL_LOAD_FALLBACK_USED,
        "modelLoadError": MODEL_LOAD_ERROR,
        "modelPathCandidates": get_model_path_candidates(),
        "modelNames": model_names,
        "publicNames": public_names,
        "yoloImgsz": YOLO_IMGSZ,
        "requestedYoloImgsz": REQUESTED_YOLO_IMGSZ,
        "distanceFovDeg": YOLO_DISTANCE_FOV_DEG,
        "distanceClassHeightsM": CLASS_HEIGHTS_M,
        "modelConf": MODEL_CONFIDENCE_THRESHOLD,
        "fallbackModelConf": FALLBACK_MODEL_CONFIDENCE_THRESHOLD,
        "lowConfFallbackEnabled": YOLO_LOW_CONF_FALLBACK,
        "returnFallbackDetections": YOLO_RETURN_FALLBACK_DETECTIONS,
        "defaultConf": DEFAULT_CONFIDENCE_THRESHOLD,
        "closeWallConf": CLOSE_WALL_CONFIDENCE_THRESHOLD,
        "ignoredClasses": sorted(IGNORED_CLASSES),
        "recognitionLogEnabled": YOLO_RECOGNITION_LOG,
        "recognitionLogCacheEnabled": YOLO_RECOGNITION_LOG_CACHE,
        "recognitionLogEmptyEnabled": YOLO_RECOGNITION_LOG_EMPTY,
        "bypassReturnFilter": YOLO_BYPASS_RETURN_FILTER,
        "yoloIou": YOLO_IOU,
        "yoloMaxDet": YOLO_MAX_DET,
        "maxReturnDetections": MAX_RETURN_DETECTIONS,
        "debugDetectionLimit": DEBUG_DETECTION_LIMIT,
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


# 현재는 발사/파괴 로직보다 이동과 인식 연결을 확인하기 위한 고정 행동 시퀀스입니다.
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


# 메인 탐지 엔드포인트: 이미지 수신 -> YOLO 추론 -> 필터링 -> JSON 응답.
@app.route('/detect', methods=['POST'])
def detect():
    # /detect 처리 순서:
    # 이미지 수신 -> 캐시/동시 실행 여부 확인 -> YOLO 추론 -> 반환용 detection 필터링 -> debug 상태 저장.
    started_at = time.perf_counter()
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    # 너무 짧은 간격으로 들어온 요청은 최신 결과를 재사용해 프레임 지연을 줄입니다.
    cached = get_cached_detections(time.time())
    if cached is not None:
        cached_results, cached_timestamp = cached
        return return_cached_response(started_at, cached_results, cached_timestamp, "fresh_interval")

    # YOLO 추론은 무겁기 때문에 동시에 하나만 실행하고, 겹치는 요청은 캐시로 응답합니다.
    if not yolo_predict_lock.acquire(blocking=False):
        latest_results, latest_timestamp = get_latest_detections()
        return return_cached_response(started_at, latest_results, latest_timestamp, "inference_busy")

    try:
        # 시뮬레이터가 보낸 이미지 파일을 OpenCV 프레임으로 디코딩합니다.
        decode_started_at = time.perf_counter()
        frame = decode_uploaded_image(image)
        decode_ms = (time.perf_counter() - decode_started_at) * 1000
        if frame is None:
            return jsonify({"error": "Invalid image received"}), 400

        original_shape = frame.shape
        frame_shape = [int(value) for value in original_shape]
        frame_mean = float(np.mean(frame))
        frame_std = float(np.std(frame))
        # 전처리 단계는 제거했지만 디버그 응답 호환성을 위해 0.0으로 기록합니다.
        preprocess_ms = 0.0

        # 기본 confidence로 먼저 추론하고, 옵션이 켜져 있으면 빈 결과에 한해 낮은 confidence로 재시도합니다.
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

    # 모델 출력 박스를 시뮬레이터 응답 형식으로 바꾸고 confidence/클래스 기준으로 거릅니다.
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
        box_coords = box[:4]
        bypass_return_filter = YOLO_BYPASS_RETURN_FILTER or (
            fallback_used and YOLO_RETURN_FALLBACK_DETECTIONS
        )
        returned, reject_reason, threshold = evaluate_detection_for_return(
            class_name,
            confidence,
            box_coords,
            original_shape,
            bypass_return_filter,
        )
        debug_detection = make_debug_detection(
            model_class_name,
            class_name,
            class_id,
            box_coords,
            confidence,
            returned,
            reject_reason,
            threshold,
            original_shape,
        )
        raw_detections.append(debug_detection)
        if not returned:
            rejected_detections.append(debug_detection)
            continue

        filtered_results.append(
            make_detection_response(
                class_name,
                box_coords,
                confidence,
                class_id,
                original_shape,
            )
        )

    filtered_results.sort(key=lambda detection: detection["confidence"], reverse=True)
    if not YOLO_BYPASS_RETURN_FILTER:
        filtered_results = filtered_results[:MAX_RETURN_DETECTIONS]
    postprocess_ms = (time.perf_counter() - postprocess_started_at) * 1000
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    # 마지막 탐지 상태를 저장해 `/debug_state`와 캐시 응답에서 재사용합니다.
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


# 디버그용 엔드포인트와 시뮬레이터가 호출하는 보조 API들입니다.
@app.route('/debug_state', methods=['GET'])
@app.route('/debug/perf', methods=['GET'])
def debug_state():
    return jsonify(get_debug_state_payload())


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    """스테레오 카메라 모드를 켰을 때 좌/우 이미지 수신 여부만 확인하는 자리입니다."""
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    return jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    """시뮬레이터 상태 tick을 받는 엔드포인트입니다. 필요하면 여기서 pause/reset을 반환할 수 있습니다."""
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
    """현재는 AI 제어 로직 대신 준비된 명령 시퀀스를 하나씩 꺼내 보내는 테스트용 액션 API입니다."""
    data = request.get_json(force=True)

    # 위치/포탑 값은 현재 로깅만 하지만, 이후 제어 정책을 붙일 때 입력 특징으로 사용할 수 있습니다.
    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    # combined_commands는 pop으로 소비되므로 서버를 재시작하면 처음부터 반복됩니다.
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
    """탄착/피격 정보를 받아 로그로 남깁니다. 명중 기반 보상이나 분석을 붙일 때 쓰는 자리입니다."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    """외부에서 목적지를 문자열 좌표로 지정할 때 형식을 검증해 시뮬레이터에 돌려줍니다."""
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
    """장애물 상태 업데이트를 받아 추후 경로 계획/회피 로직에서 사용할 수 있게 만든 엔드포인트입니다."""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    """충돌 이벤트를 받아 어떤 객체와 어느 위치에서 부딪혔는지 기록합니다."""
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


# 에피소드가 시작될 때 시뮬레이터가 읽는 초기 설정입니다.
@app.route('/init', methods=['GET'])
def init():
    """시뮬레이터 에피소드 시작 시 월드/카메라/로그 옵션을 전달합니다."""
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
    """시뮬레이터가 시작 신호를 확인하는 간단한 헬스체크성 엔드포인트입니다."""
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    # Flask 바인딩 주소는 YAML 설정을 따르고, YOLO 관련 옵션은 위의 환경변수를 따릅니다.
    config = load_config()

    host = config["simulator"]["host"]
    port = config["simulator"]["port"]
    app.run(host=host, port=port, threaded=SERVER_THREADED)
