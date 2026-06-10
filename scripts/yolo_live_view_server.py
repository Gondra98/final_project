"""
Tank simulator -> Flask -> Web live view + async YOLO(best_final.engine)

이 파일은 디버깅/시각화에 초점을 둔 서버다.
`yolo_detection_server.py`처럼 `/detect`를 제공하지만, 탐지는 백그라운드 worker가 처리하고
웹 페이지(`/view`)는 최신 프레임 위에 마지막 bbox와 track ID를 덧그려 보여준다.

핵심 구조
- /detect: 시뮬레이터 이미지 수신 후 원본 프레임을 즉시 latest_frame에 저장하고 바로 최신 detection을 반환
- yolo_worker: 백그라운드에서 최신 프레임 1장만 YOLO Tracking 추론
- /view: 최신 원본 프레임 + 최신 bbox를 그려서 웹으로 스트리밍

웹에서 YOLO를 다시 실행하지 않는다.
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

# =========================
# 환경 설정
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tank_detector" / "best_final.pt"
DEFAULT_FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "tank_detector" / "best.pt"
# 웹 확인용 서버는 기본적으로 최종 best_final.engine를 사용하지만, YOLO_MODEL_PATH로 다른 weight를 지정할 수 있습니다.
YOLO_MODEL_PATH_ENV = os.getenv("YOLO_MODEL_PATH")
YOLO_FALLBACK_MODEL_PATH_ENV = os.getenv("YOLO_FALLBACK_MODEL_PATH")


def resolve_project_path(path_value) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


YOLO_MODEL_PATH = resolve_project_path(YOLO_MODEL_PATH_ENV or DEFAULT_MODEL_PATH)
REQUESTED_YOLO_MODEL_PATH = YOLO_MODEL_PATH
YOLO_FALLBACK_MODEL_PATH = resolve_project_path(
    YOLO_FALLBACK_MODEL_PATH_ENV or DEFAULT_FALLBACK_MODEL_PATH
)
YOLO_ALLOW_MODEL_FALLBACK = os.getenv("YOLO_ALLOW_MODEL_FALLBACK", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_LOAD_FALLBACK_USED = False
MODEL_LOAD_ERROR: Optional[str] = None


# 큰 흐름:
# 1. 서버 시작 시 YOLO 모델을 한 번 로드합니다. TensorRT engine이 실패하면 best.pt로 fallback할 수 있습니다.
# 2. /detect는 simulator가 보낸 최신 이미지를 저장하고 worker를 깨운 뒤, 직전 YOLO 결과를 즉시 반환합니다.
# 3. yolo_worker_loop는 항상 가장 최신 프레임 하나만 골라 YOLO 추론을 수행하고 latest_detections를 갱신합니다.
# 4. /view와 /video_feed는 최신 원본 프레임 위에 latest_detections를 그려서 브라우저에서 확인하게 해줍니다.
# 5. /debug_state는 현재 모델, 프레임 번호, 추론 시간, 최근 에러를 한 번에 확인하는 진단용 API입니다.


def parse_ignored_classes(default_value: str) -> set[str]:
    value = os.getenv("YOLO_IGNORED_CLASSES", default_value)
    return {item.strip().lower() for item in value.split(",") if item.strip()}


HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "5000"))

YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available() and YOLO_DEVICE.lower() != "cpu"
# half precision은 CUDA에서만 의미가 있으므로 CPU 실행일 때는 자동으로 꺼집니다.
YOLO_HALF = os.getenv("YOLO_HALF", "true").lower() in {"1", "true", "yes", "on"} and USE_CUDA
REQUESTED_YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "416"))
YOLO_IMGSZ = REQUESTED_YOLO_IMGSZ
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.70"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "30"))
# YOLO Tracking 설정. 추적이 필요 없으면 YOLO_TRACKING=false로 기존 predict 경로를 사용할 수 있습니다.
YOLO_TRACKING = os.getenv("YOLO_TRACKING", "true").lower() in {"1", "true", "yes", "on"}
YOLO_TRACKER = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
YOLO_TRACK_PERSIST = os.getenv("YOLO_TRACK_PERSIST", "true").lower() in {"1", "true", "yes", "on"}
YOLO_DISTANCE_FOV_DEG = float(os.getenv("YOLO_DISTANCE_FOV_DEG", "60"))
YOLO_DISTANCE_MIN_BOX_HEIGHT = float(os.getenv("YOLO_DISTANCE_MIN_BOX_HEIGHT", "2"))
IGNORED_CLASSES = parse_ignored_classes("wall")

# 클래스별 고정 ID입니다. 같은 클래스가 여러 개 잡히면 classFixedId는 같고, trackId로 객체를 구분합니다.
CLASS_FIXED_IDS = {
    "tank": 1,
    "rock": 2,
    "person": 3,
    "tent": 4,
}

WEB_FPS = float(os.getenv("WEB_FPS", "20"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DETECT_MODE = os.getenv("SIM_DETECT_MODE", "true").lower() in {"1", "true", "yes", "on"}
PRINT_TIMING_LOG = os.getenv("YOLO_TIMING", "true").lower() in {"1", "true", "yes", "on"}
PRINT_DETECTION_LOG = os.getenv("YOLO_RECOGNITION_LOG", "false").lower() in {"1", "true", "yes", "on"}

# 선택 옵션: 너무 자주 YOLO를 돌리지 않도록 최소 간격 제한. 0이면 제한 없음.
YOLO_MIN_INTERVAL = float(os.getenv("YOLO_MIN_INTERVAL", "0.00"))

# =========================
# Flask / YOLO 초기화
# =========================
app = Flask(__name__)

if not YOLO_MODEL_PATH.exists():
    print(f"[WARNING] YOLO model not found: {YOLO_MODEL_PATH}")
    print("          YOLO_MODEL_PATH 환경변수 또는 best_final.engine 위치를 확인하세요.")

def ensure_model_runtime(model_path: Path) -> None:
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


def get_tensorrt_engine_imgsz(model_path: Path) -> Optional[int]:
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


def normalize_model_names(names):
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def get_engine_load_error_message(model_path: Path) -> str:
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


def load_yolo_model(model_path: Path):
    try:
        loaded_model = YOLO(str(model_path), task="detect")
        loaded_names = normalize_model_names(loaded_model.names)
    except (AttributeError, RuntimeError, ValueError) as exc:
        if model_path.suffix.lower() == ".engine":
            raise RuntimeError(get_engine_load_error_message(model_path)) from exc
        raise
    return loaded_model, loaded_names


def load_yolo_model_with_fallback(model_path: Path):
    try:
        ensure_model_runtime(model_path)
        loaded_model, loaded_names = load_yolo_model(model_path)
        return model_path, loaded_model, loaded_names, False, None
    except (AttributeError, RuntimeError, ValueError, OSError, ModuleNotFoundError) as exc:
        fallback_path = YOLO_FALLBACK_MODEL_PATH
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


# 시작 시 모델 로드는 한 번만 수행합니다. 이후 /detect 요청마다 모델을 다시 만들지 않습니다.
print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
YOLO_MODEL_PATH, model, model_names, MODEL_LOAD_FALLBACK_USED, MODEL_LOAD_ERROR = (
    load_yolo_model_with_fallback(YOLO_MODEL_PATH)
)
engine_imgsz = get_tensorrt_engine_imgsz(YOLO_MODEL_PATH)
if engine_imgsz is not None and YOLO_IMGSZ != engine_imgsz:
    print(
        f"Overriding YOLO_IMGSZ={YOLO_IMGSZ} to TensorRT engine input size "
        f"{engine_imgsz}."
    )
    YOLO_IMGSZ = engine_imgsz
if MODEL_LOAD_FALLBACK_USED:
    print(f"Loaded fallback YOLO model: {YOLO_MODEL_PATH}")
print(f"Model labels: {model_names}")
print(
    "YOLO runtime: "
    f"device={YOLO_DEVICE}, half={YOLO_HALF}, imgsz={YOLO_IMGSZ}, "
    f"conf={YOLO_CONF}, iou={YOLO_IOU}, max_det={YOLO_MAX_DET}, "
    f"tracking={YOLO_TRACKING}, tracker={YOLO_TRACKER}, persist={YOLO_TRACK_PERSIST}, "
    f"web_fps={WEB_FPS}, jpeg_quality={JPEG_QUALITY}"
)

# =========================
# 공유 상태
# =========================
state_lock = Lock()
# worker가 새 프레임을 기다릴 때 쓰는 조건 변수입니다. busy-wait 없이 `/detect`가 깨워줍니다.
frame_condition = Condition()

latest_frame: Optional[np.ndarray] = None          # 웹에 즉시 보여줄 최신 원본 프레임
latest_frame_seq: int = 0                         # /detect가 새 프레임을 받을 때마다 증가
latest_frame_timestamp: float = 0.0

processed_frame_seq: int = 0                      # YOLO가 마지막으로 처리한 프레임 번호
latest_detections: List[Dict[str, Any]] = []
latest_lidar_points: List[Dict] = []
latest_yolo_ms: float = 0.0
latest_post_ms: float = 0.0
latest_worker_total_ms: float = 0.0
latest_detect_response_ms: float = 0.0
latest_decode_ms: float = 0.0
latest_result_timestamp: float = 0.0
latest_frame_shape: Optional[List[int]] = None
latest_error: Optional[str] = None
request_count: int = 0
worker_count: int = 0

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


def get_class_bgr_color(class_name: str, class_id: int = 0) -> Tuple[int, int, int]:
    normalized_name = str(class_name).strip().lower()
    if normalized_name in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[normalized_name]
    return CLASS_COLOR_PALETTE_BGR[class_id % len(CLASS_COLOR_PALETTE_BGR)]


def bgr_to_hex(color: Tuple[int, int, int]) -> str:
    blue, green, red = (int(value) for value in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def get_class_hex_color(class_name: str, class_id: int = 0) -> str:
    return bgr_to_hex(get_class_bgr_color(class_name, class_id))


def get_class_height_m(class_name: str) -> Optional[float]:
    return CLASS_HEIGHTS_M.get(str(class_name).strip().lower())


# def estimate_distance_by_height(
#     class_name: str,
#     bbox: List[float],
#     frame_shape: Tuple[int, ...],
# ) -> Optional[float]:
#     if len(bbox) < 4 or not frame_shape:
#         return None
#     bbox_h_px = float(bbox[3]) - float(bbox[1])
#     img_h = float(frame_shape[0])
#     if bbox_h_px <= 0 or img_h <= 0:
#         return None
#     height = bbox_h_px / img_h
#     class_name_lower = class_name.lower()

#     if class_name_lower == "tank":
#         calib = [
#             (216/1057, 30), (147/1057, 40), (111/1057, 50),
#             (94/1057,  60), (73/1057,  70), (63/1057,  80),
#             (59/1057,  90), (52/1057, 100),
#         ]
#     elif class_name_lower in ("person"):
#         calib = [
#             (191/1057, 30), (139/1057, 40),
#             (113/1057, 50), (88/1057,  60),
#         ]
#     elif class_name_lower == "tent":
#         calib = [
#             (472/1057,30),(370/1057,40),(328/1057,50),
#             (269/1057,60),(226/1057,70),(196/1057,80),
#             (169/1057,90),(150/1057,100),(139/1057,110),
#             (116/1057,130),
#         ]
#     elif class_name_lower == "rock":
#         calib = [
#             (407/1057, 20), (292/1057, 30), (213/1057, 40),
#             (160/1057, 50), (128/1057, 60), (110/1057, 70),
#             (94/1057, 80),  (88/1057, 90),  (78/1057, 100),
#             (68/1057, 120),
#         ]
#     else:
#         return float(round(1.0 / height, 1)) if height > 0 else None
    
#     heights = np.array([h for h, d in calib])
#     distances = np.array([d for h, d in calib])
#     coeffs = np.polyfit(1.0 / heights, distances, 1)
#     a, b = coeffs[0], coeffs[1]
#     return float(round(max(0.0, a / height + b), 1))


def estimate_distance_by_lidar(
    bbox: List[float],
    frame_shape: Tuple[int, ...],
    lidar_points: List[Dict],
    fov_deg: float = 30.0,
) -> Optional[float]:
    if not lidar_points or not frame_shape:
        return None
    bbox_cx = (bbox[0] + bbox[2]) / 2.0
    img_w = float(frame_shape[1])
    angle = (bbox_cx / img_w - 0.5) * fov_deg
    if angle < 0:
        angle += 360
    candidates = [
        p["distance"] for p in lidar_points
        if abs(p.get("verticalAngle", 90)) < 5.0
        and min(abs(p.get("angle", 999) - angle), 360 - abs(p.get("angle", 999) - angle)) < 5.0
    ]
    all_same_angle = [
        p["distance"] for p in lidar_points
        if min(abs(p.get("angle", 999) - angle), 360 - abs(p.get("angle", 999) - angle)) < 5.0
    ]
    ground_dist = max(all_same_angle) if all_same_angle else 999
    objects = [d for d in candidates if d < ground_dist - 1.5]

    print(f"[lidar] bbox_cx={bbox_cx:.0f} angle={angle:.1f}° candidates={candidates[:3]} ground={ground_dist:.1f} objects={objects[:3]}")

    if objects:
        return float(round(min(objects), 1))
    elif candidates:
        return float(round(min(candidates), 1))
    return None


# =========================
# 유틸 함수
# =========================
def decode_uploaded_image(image_file) -> Optional[np.ndarray]:
    """Flask 업로드 파일을 OpenCV BGR 프레임으로 디코딩합니다."""
    image_bytes = image_file.read()
    if not image_bytes:
        return None
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)


def run_yolo_only(frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float, float]:
    """YOLO Tracking/Detection 수행 후 bbox 결과와 소요 시간을 반환합니다.

    YOLO_TRACKING=True이면 model.track()을 사용하여 프레임 간 trackId를 유지합니다.
    YOLO_TRACKING=False이면 기존처럼 model.predict()만 수행합니다.
    화면에 그릴 이미지는 여기서 만들지 않습니다. worker는 숫자 결과만 저장하고,
    스트리밍 루프가 최신 원본 프레임에 bbox를 덧그려 웹 응답을 만듭니다.
    """
    # worker에서만 호출되는 핵심 추론 함수입니다.
    # 모델 결과를 simulator/view가 공통으로 쓰는 detection dict 목록으로 변환합니다.
    yolo_started = time.perf_counter()
    with torch.inference_mode():
        if YOLO_TRACKING:
            results = model.track(
                source=frame,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                device=YOLO_DEVICE,
                half=YOLO_HALF,
                iou=YOLO_IOU,
                max_det=YOLO_MAX_DET,
                tracker=YOLO_TRACKER,
                persist=YOLO_TRACK_PERSIST,
                verbose=False,
            )
        else:
            results = model.predict(
                source=frame,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                device=YOLO_DEVICE,
                half=YOLO_HALF,
                iou=YOLO_IOU,
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
        if USE_CUDA:
            torch.cuda.synchronize()
    yolo_ms = (time.perf_counter() - yolo_started) * 1000

    with state_lock:
        lidar_pts = list(latest_lidar_points)

    post_started = time.perf_counter()
    detections: List[Dict[str, Any]] = []
    boxes = results[0].boxes if results and results[0].boxes is not None else None
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()

        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.detach().cpu().numpy()
        else:
            track_ids = [None] * len(xyxy)

        for bbox_xyxy, conf, cls_id, track_id in zip(xyxy, confs, clss, track_ids):
            x1, y1, x2, y2 = bbox_xyxy
            class_id = int(cls_id)
            class_name = str(model_names.get(class_id, class_id))
            normalized_class_name = class_name.strip().lower()
            if normalized_class_name in IGNORED_CLASSES:
                continue
            fixed_id = CLASS_FIXED_IDS.get(normalized_class_name)
            color = get_class_hex_color(class_name, class_id)
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            center_x = (float(x1) + float(x2)) / 2.0
            center_y = (float(y1) + float(y2)) / 2.0
            distance = estimate_distance_by_lidar(bbox, frame.shape, lidar_pts)
            detections.append(
                {
                    "id": fixed_id,
                    "classFixedId": fixed_id,
                    "trackId": None if track_id is None else int(track_id),
                    "className": class_name,
                    "classId": class_id,
                    "confidence": float(conf),
                    "bbox": bbox,
                    "center": [float(center_x), float(center_y)],
                    "distance": distance,
                    "color": color,
                    "filled": False,
                    "updateBoxWhileMoving": False,
                }
            )
    post_ms = (time.perf_counter() - post_started) * 1000
    return detections, yolo_ms, post_ms


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """프레임 복사본에 bbox와 현재 처리 상태 텍스트를 덧그립니다."""
    drawn = frame.copy()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        class_name = det.get("className", "object")
        class_id = int(det.get("classId", 0))
        fixed_id = det.get("classFixedId", det.get("id"))
        track_id = det.get("trackId")
        fixed_text = "" if fixed_id is None else f" ID:{fixed_id}"
        track_text = "" if track_id is None else f" T:{track_id}"
        id_text = f"{fixed_text}{track_text}"
        conf = float(det.get("confidence", 0.0))
        distance = det.get("distance")
        distance_text = f" {distance:.1f}m" if distance is not None else " N/A"
        color = get_class_bgr_color(class_name, class_id)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            drawn,
            f"{class_name}{id_text} {conf:.2f}{distance_text}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # 화면 상태 표시
    with state_lock:
        yolo_ms = latest_yolo_ms
        result_age = time.time() - latest_result_timestamp if latest_result_timestamp else None
        frame_seq = latest_frame_seq
        proc_seq = processed_frame_seq

    status = f"frame={frame_seq} yolo_seq={proc_seq} det={len(detections)} yolo={yolo_ms:.1f}ms"
    if result_age is not None:
        status += f" age={result_age*1000:.0f}ms"
    else:
        status += " age=none"

    cv2.putText(drawn, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return drawn


def make_blank_frame(message: str = "Waiting for simulator image...") -> np.ndarray:
    """아직 `/detect`가 들어오지 않았을 때 웹 스트림에 보여줄 대기 화면입니다."""
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

# =========================
# YOLO 백그라운드 worker
# =========================
def yolo_worker_loop() -> None:
    """새 프레임이 들어올 때마다 가장 최신 프레임 하나만 YOLO로 처리하는 worker입니다."""
    global processed_frame_seq, latest_detections, latest_yolo_ms, latest_post_ms
    global latest_worker_total_ms, latest_result_timestamp, latest_error, worker_count

    print("YOLO worker started")
    last_run_time = 0.0

    while True:
        with frame_condition:
            frame_condition.wait_for(lambda: latest_frame_seq > processed_frame_seq)

        # 최신 프레임만 복사한다. 중간에 쌓인 오래된 프레임은 버린다.
        # 이 방식은 모든 프레임을 처리하는 정확도보다 화면/시뮬레이터 응답성을 우선합니다.
        with state_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            seq_to_process = latest_frame_seq

        if frame is None:
            continue

        # 너무 잦은 추론을 제한하고 싶을 때만 사용
        if YOLO_MIN_INTERVAL > 0:
            now = time.time()
            wait_sec = YOLO_MIN_INTERVAL - (now - last_run_time)
            if wait_sec > 0:
                time.sleep(wait_sec)
            last_run_time = time.time()

        started = time.perf_counter()
        try:
            detections, yolo_ms, post_ms = run_yolo_only(frame)
            total_ms = (time.perf_counter() - started) * 1000
            with state_lock:
                # worker 결과는 한 번에 갱신해서 `/detect`, `/view`, `/debug_state`가 일관된 값을 읽게 합니다.
                processed_frame_seq = seq_to_process
                latest_detections = list(detections)
                latest_yolo_ms = yolo_ms
                latest_post_ms = post_ms
                latest_worker_total_ms = total_ms
                latest_result_timestamp = time.time()
                latest_error = None
                worker_count += 1

            if PRINT_TIMING_LOG:
                print(
                    f"[worker] seq={seq_to_process} yolo={yolo_ms:.1f}ms "
                    f"post={post_ms:.1f}ms total={total_ms:.1f}ms det={len(detections)}"
                )
            if PRINT_DETECTION_LOG and detections:
                for det in detections:
                    distance = det.get("distance")
                    distance_text = "N/A" if distance is None else f"{distance:.2f}m"
                    track_text = "None" if det.get("trackId") is None else str(det.get("trackId"))
                    fixed_text = "None" if det.get("classFixedId") is None else str(det.get("classFixedId"))
                    print(
                        f"[det] id={fixed_text} trackId={track_text} class={det['className']} "
                        f"conf={det['confidence']:.2f} distance={distance_text} "
                        f"center={det.get('center')} bbox={det['bbox']}"
                    )
        except Exception as exc:  # noqa: BLE001
            with state_lock:
                latest_error = str(exc)
            print(f"[worker:error] {exc}")
            time.sleep(0.05)

# =========================
# 시뮬레이터 API
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    """
    시뮬레이터가 호출하는 객체 탐지 API.
    중요: YOLO를 여기서 기다리지 않는다.
    1) 이미지 디코딩
    2) latest_frame 즉시 갱신
    3) worker 깨우기
    4) 현재까지의 최신 detection 결과 바로 반환
    """
    global latest_frame, latest_frame_seq, latest_frame_timestamp, latest_frame_shape
    global latest_detect_response_ms, latest_decode_ms, latest_error, request_count

    # /detect는 실시간성을 위해 YOLO 완료를 기다리지 않습니다.
    # 새 프레임은 worker에게 넘기고, 응답은 직전에 완성된 detection을 즉시 돌려줍니다.
    request_started = time.perf_counter()
    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image received"}), 400

    decode_started = time.perf_counter()
    frame = decode_uploaded_image(image)
    decode_ms = (time.perf_counter() - decode_started) * 1000
    if frame is None:
        with state_lock:
            latest_error = "Invalid image received"
        return jsonify({"error": "Invalid image received"}), 400

    with state_lock:
        request_count += 1
        latest_frame_seq += 1
        current_seq = latest_frame_seq
        latest_frame = frame.copy()  # 웹에서 바로 표시할 원본 프레임을 YOLO 전에 저장
        latest_frame_timestamp = time.time()
        latest_frame_shape = [int(v) for v in frame.shape]
        latest_decode_ms = decode_ms
        detections_to_return = list(latest_detections)  # 직전 최신 결과

    # worker가 잠들어 있으면 깨우지만, 응답은 worker 완료를 기다리지 않고 즉시 보냅니다.
    with frame_condition:
        frame_condition.notify()

    response_ms = (time.perf_counter() - request_started) * 1000
    with state_lock:
        latest_detect_response_ms = response_ms
        proc_seq = processed_frame_seq
        result_age = time.time() - latest_result_timestamp if latest_result_timestamp else None

    if PRINT_TIMING_LOG:
        age_text = "none" if result_age is None else f"{result_age*1000:.1f}ms"
        print(
            f"[/detect] enqueue_seq={current_seq} return_seq={proc_seq} "
            f"decode={decode_ms:.1f}ms response={response_ms:.1f}ms "
            f"result_age={age_text} det={len(detections_to_return)}"
        )

    return jsonify(detections_to_return)


@app.route("/init", methods=["GET"])
def init():
    """시뮬레이터 시작 시 필요한 초기 설정을 반환합니다."""
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
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
        "destoryObstaclesOnHit": True,
    }
    print("[/init]", config)
    return jsonify(config)


@app.route("/info", methods=["POST"])
def info():
    """시뮬레이터 상태 tick에 대해 별도 제어 없이 정상 응답만 반환합니다."""
    global latest_lidar_points
    data = request.get_json(silent=True) or {}
    lidar_points = data.get("lidarPoints", [])
    if lidar_points:
        with state_lock:
            latest_lidar_points = lidar_points
    return jsonify({"status": "success", "control": ""})



@app.route("/get_action", methods=["POST"])
def get_action():
    """웹 확인용 서버에서는 이동 제어를 하지 않고 중립 명령을 보냅니다."""
    command = {
        "moveWS": {"command": "", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }
    return jsonify(command)

# =========================
# 웹 표시 API
# =========================
@app.route("/view")
def view():
    """브라우저에서 확인할 수 있는 단일 페이지를 문자열 템플릿으로 제공합니다."""
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <title>Live YOLO Tracking View</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; text-align: center; }
            header { padding: 12px 20px; background: #1e1e1e; border-bottom: 1px solid #333; }
            .wrap { padding: 16px; }
            img { max-width: 96vw; max-height: 82vh; border: 2px solid #00ff00; background: #000; }
            .hint { margin-top: 10px; color: #aaa; font-size: 14px; }
            a { color: #7dd3fc; }
        </style>
    </head>
    <body>
        <header><h2>Live YOLO Tracking Result</h2></header>
        <div class="wrap">
            <img src="/video_feed" alt="YOLO stream">
            <div class="hint">/detect로 들어온 원본 프레임은 즉시 표시하고, YOLO bbox와 track ID는 백그라운드 결과를 덧그립니다.</div>
            <div class="hint">상태 확인: <a href="/debug_state">/debug_state</a></div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


def generate_video_stream():
    """MJPEG 스트림을 생성합니다. 브라우저는 각 JPEG 조각을 이어서 영상처럼 표시합니다."""
    # /view의 <img> 태그가 이 generator를 계속 읽습니다.
    # latest_frame과 latest_detections는 worker가 갱신한 최신 상태를 사용합니다.
    interval = 1.0 / max(1.0, WEB_FPS)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    while True:
        with state_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            detections = list(latest_detections)

        if frame is None:
            frame = make_blank_frame()
        else:
            # worker 결과가 약간 이전 프레임 기준일 수 있지만, 실시간 디버깅에서는 지연을 줄이는 것이 더 중요합니다.
            frame = draw_detections(frame, detections)

        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        time.sleep(interval)


@app.route("/video_feed")
def video_feed():
    """`/view`의 img 태그가 구독하는 MJPEG 응답입니다."""
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/debug_state")
def debug_state():
    """프레임 수신, worker 처리, 최근 에러와 성능 값을 JSON으로 확인합니다."""
    with state_lock:
        result_age = time.time() - latest_result_timestamp if latest_result_timestamp else None
        frame_age = time.time() - latest_frame_timestamp if latest_frame_timestamp else None
        payload = {
            "modelPath": str(YOLO_MODEL_PATH),
            "requestedModelPath": str(REQUESTED_YOLO_MODEL_PATH),
            "modelPathEnvValue": YOLO_MODEL_PATH_ENV,
            "modelFallbackPath": str(YOLO_FALLBACK_MODEL_PATH),
            "modelFallbackAllowed": YOLO_ALLOW_MODEL_FALLBACK,
            "modelFallbackUsed": MODEL_LOAD_FALLBACK_USED,
            "modelLoadError": MODEL_LOAD_ERROR,
            "modelNames": model_names,
            "ignoredClasses": sorted(IGNORED_CLASSES),
            "classFixedIds": CLASS_FIXED_IDS,
            "tracking": YOLO_TRACKING,
            "tracker": YOLO_TRACKER,
            "trackPersist": YOLO_TRACK_PERSIST,
            "device": YOLO_DEVICE,
            "cudaAvailable": torch.cuda.is_available(),
            "half": YOLO_HALF,
            "imgsz": YOLO_IMGSZ,
            "requestedImgsz": REQUESTED_YOLO_IMGSZ,
            "distanceFovDeg": YOLO_DISTANCE_FOV_DEG,
            "distanceClassHeightsM": CLASS_HEIGHTS_M,
            "conf": YOLO_CONF,
            "latestFrameSeq": latest_frame_seq,
            "processedFrameSeq": processed_frame_seq,
            "requestCount": request_count,
            "workerCount": worker_count,
            "latestDetectionCount": len(latest_detections),
            "latestDetections": latest_detections,
            "latestDetectResponseMs": latest_detect_response_ms,
            "latestDecodeMs": latest_decode_ms,
            "latestYoloMs": latest_yolo_ms,
            "latestPostMs": latest_post_ms,
            "latestWorkerTotalMs": latest_worker_total_ms,
            "latestFrameShape": latest_frame_shape,
            "latestFrameAgeMs": None if frame_age is None else frame_age * 1000,
            "latestResultAgeMs": None if result_age is None else result_age * 1000,
            "latestError": latest_error,
            "webFps": WEB_FPS,
            "jpegQuality": JPEG_QUALITY,
        }
    return jsonify(payload)

# =========================
# Warm-up
# =========================
def warmup_yolo() -> None:
    """첫 실제 요청에서 생기는 CUDA/모델 초기화 지연을 줄이기 위해 더미 추론을 미리 실행합니다."""
    if not YOLO_MODEL_PATH.exists():
        return
    dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    started = time.perf_counter()
    try:
        for _ in range(3):
            with torch.inference_mode():
                model.predict(
                    source=dummy,
                    conf=YOLO_CONF,
                    imgsz=YOLO_IMGSZ,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    iou=YOLO_IOU,
                    max_det=YOLO_MAX_DET,
                    verbose=False,
                )
        if USE_CUDA:
            torch.cuda.synchronize()
        print(f"YOLO warm-up complete: {(time.perf_counter() - started) * 1000:.1f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup:error] {exc}")


if __name__ == "__main__":
    # 서버 시작 전에 모델을 한 번 깨우고, 이후에는 데몬 worker가 최신 프레임만 계속 처리합니다.
    warmup_yolo()
    Thread(target=yolo_worker_loop, daemon=True).start()
    print(f"Server running: http://127.0.0.1:{PORT}/view")
    app.run(host=HOST, port=PORT, threaded=True)
