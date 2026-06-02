# Camera-LiDAR Fusion 연결 가이드

이 문서는 `perception/camera_lidar_fusion.py`와 `perception/risk_classifier.py`를 기존 Flask 서버의 `/detect` 흐름에 연결하는 방법을 정리한다. 이 단계에서는 서버 코드를 크게 바꾸지 않고, YOLO 탐지 결과에 LiDAR 거리와 위험도 정보를 덧붙이는 것을 목표로 한다.

## 1. 목적

정밀 3D calibration을 바로 구현하지 않고 sector matching 방식을 먼저 사용한다. Tank Challenge 1차 자율주행 장애물 회피에서는 “화면의 어느 방향에 객체가 있는지”와 “그 방향의 LiDAR 거리가 얼마나 가까운지”만 알아도 충분히 의미 있는 감속, 정지, 회피, 재계획 판단을 만들 수 있다.

역할 분담은 단순하다.

- YOLO: `className`, `bbox`, `confidence`로 객체 종류와 화면 위치를 판단한다.
- Camera bearing: bbox 중심 x좌표를 카메라 좌우 각도 `bearingDeg`로 변환한다.
- LiDAR: `bearingDeg`와 가까운 `angle`의 `distance`를 제공한다.
- Risk classifier: `className + distanceM + bearingDeg`로 `riskZone`과 제어 힌트를 만든다.

이 방식은 정확한 3D 위치 추정보다는 빠른 장애물 회피용 1차 구현에 적합하다.

## 2. 전체 흐름

```text
YOLO detection
  className, bbox, confidence
        |
        v
bbox center_x
        |
        v
camera bearingDeg
        |
        v
nearest LiDAR angle
        |
        v
LiDAR distance
        |
        v
riskZone
        |
        v
slow / stop / avoid / replan
```

## 3. 서버 상태 추가

기존 Flask 서버에 전역 상태 dict가 있다면 거기에 추가한다. 없다면 `/detect`, `/info`, `/get_action`에서 공유할 수 있는 작은 dict를 만든다.

```python
planner_state = {
    "latest_lidar_points": [],
    "latest_lidar_timestamp": 0.0,
    "vision_lidar_risk": None,
    "needs_replan": False,
}
```

이미 `planner_state`가 있다면 아래 값만 추가하면 된다.

```python
planner_state["latest_lidar_points"] = []
planner_state["latest_lidar_timestamp"] = 0.0
```

## 4. LiDAR 저장 위치

시뮬레이터가 LiDAR 데이터를 `/info` payload에 포함해서 보내거나, 별도의 `/update_lidar` 같은 엔드포인트로 보낸다면 가장 최근 LiDAR만 저장한다.

```python
import time


def update_latest_lidar(payload):
    lidar_points = lidar_items_from_payload(payload)
    if lidar_points:
        planner_state["latest_lidar_points"] = lidar_points
        planner_state["latest_lidar_timestamp"] = time.time()
```

`lidar_items_from_payload`는 서버 payload 구조에 맞춰 작성한다. 예를 들어 payload가 `{"lidarPoints": [...]}` 형태라면 아래처럼 충분하다.

```python
def lidar_items_from_payload(payload):
    if not isinstance(payload, dict):
        return []
    return (
        payload.get("lidarPoints")
        or payload.get("lidar_points")
        or payload.get("lidar")
        or []
    )
```

`/info`에서 처리하는 예시는 다음과 같다.

```python
@app.route("/info", methods=["POST"])
def info():
    payload = request.get_json(force=True)
    update_latest_lidar(payload)
    return jsonify({"status": "success", "control": ""})
```

별도 엔드포인트를 둘 경우:

```python
@app.route("/update_lidar", methods=["POST"])
def update_lidar():
    payload = request.get_json(force=True)
    update_latest_lidar(payload)
    return jsonify({"status": "success"})
```

## 5. `/detect` 내부에서 fusion 적용

서버 상단 import에 perception 모듈을 추가한다.

```python
from perception.camera_lidar_fusion import fuse_detections_with_lidar
from perception.risk_classifier import attach_risk_to_detections, summarize_risks
```

`/detect` 안에서는 기존 YOLO 결과인 `filtered_results`를 만든 뒤, 반환 직전에 fusion을 적용한다.

```python
image_width = original_shape[1]

fused = fuse_detections_with_lidar(
    detections=filtered_results,
    lidar_points=planner_state["latest_lidar_points"],
    image_width=image_width,
    hfov_deg=47.81061,
    tolerance_deg=5.0,
)

fused = attach_risk_to_detections(fused)
risk_summary = summarize_risks(fused)

planner_state["vision_lidar_risk"] = risk_summary

if risk_summary["shouldStop"]:
    planner_state["needs_replan"] = True

return jsonify(fused)
```

## 6. Sensor stale 처리

LiDAR는 오래된 데이터를 쓰면 위험하다. 최신 LiDAR timestamp가 너무 오래되었으면 fusion을 하지 않고, YOLO detection에 거리 없음 상태를 붙여서 반환한다.

```python
SENSOR_STALE_SECONDS = 1.5
```

예시 helper:

```python
def is_lidar_fresh():
    latest_ts = planner_state.get("latest_lidar_timestamp", 0.0)
    return time.time() - latest_ts <= SENSOR_STALE_SECONDS
```

stale일 때 distance 없는 detection으로 반환:

```python
def attach_no_lidar_distance(detections):
    output = []
    for detection in detections:
        item = dict(detection)
        item["bearingDeg"] = None
        item["lidarMatched"] = False
        item["lidarAngle"] = None
        item["lidarDistance"] = None
        item["distanceM"] = None
        item["distanceSource"] = "none"
        output.append(item)
    return output
```

`/detect` 적용 예시:

```python
if is_lidar_fresh():
    fused = fuse_detections_with_lidar(
        detections=filtered_results,
        lidar_points=planner_state["latest_lidar_points"],
        image_width=image_width,
        hfov_deg=47.81061,
        tolerance_deg=5.0,
    )
else:
    fused = attach_no_lidar_distance(filtered_results)

fused = attach_risk_to_detections(fused)
risk_summary = summarize_risks(fused)
planner_state["vision_lidar_risk"] = risk_summary
```

## 7. Planner state에 risk summary 저장

`risk_summary`는 `/get_action` 또는 주행 planner에서 바로 참고할 수 있게 저장한다.

```python
planner_state["vision_lidar_risk"] = risk_summary

if risk_summary["shouldStop"]:
    planner_state["needs_replan"] = True
```

제어 단계에서는 이 값을 직접적인 행동 명령으로 바로 바꾸기보다, 기존 주행 로직의 우선순위 입력으로 사용한다.

```python
if planner_state.get("needs_replan"):
    # 기존 경로 추종을 멈추고 회피 경로 또는 정지 명령을 선택한다.
    pass
```

## 8. Calibration 실험 절차

### 실험 1: 정면 정렬 확인

1. 전차 정면에 `wall`을 배치한다.
2. YOLO bbox의 `center_x`가 `image_width / 2` 근처인지 확인한다.
3. `bearingDeg`가 0 근처인지 확인한다.
4. LiDAR 최단 `angle`이 0 근처인지 확인한다.

기대값:

```text
bearingDeg ~= 0
lidarAngle ~= 0
```

### 실험 2: 좌우 부호 확인

1. 전차 왼쪽에 `wall`을 배치한다.
2. `bearingDeg`가 음수인지 확인한다.
3. LiDAR `angle`도 음수로 정규화되거나, 원본 값이 350도 근처인지 확인한다.

기대값:

```text
bearingDeg < 0
lidarAngle < 0 또는 raw angle ~= 350
```

### 실험 3: 오른쪽 확인

1. 전차 오른쪽에 `wall`을 배치한다.
2. `bearingDeg`가 양수인지 확인한다.
3. LiDAR `angle`도 양수인지 확인한다.

기대값:

```text
bearingDeg > 0
lidarAngle > 0
```

### 실험 4: HFOV 보정

화면 가장자리 근처에 객체를 배치하고, 해당 객체와 매칭되는 LiDAR angle을 확인한다. bbox 중심의 normalized x값과 LiDAR angle을 이용해 HFOV를 역추정한다.

```text
normalized_x = (center_x - image_width / 2) / (image_width / 2)
bearingDeg = normalized_x * (hfov_deg / 2)

hfov_deg ~= 2 * lidarAngle / normalized_x
```

여러 위치에서 측정한 뒤 평균값을 `hfov_deg` 기본값 대신 사용한다.

## 9. 한계

- 정밀 3D calibration이 아니다.
- bbox 중심과 LiDAR ray가 같은 객체를 가리킨다고 가정한다.
- 객체가 여러 개 겹치면 오매칭 가능성이 있다.
- LiDAR 각도와 카메라 FOV는 실험 보정이 필요하다.
- `verticalAngle`은 1차 구현에서 사용하지 않는다.
- `worldPosition` 추정은 후속 단계다.

## 10. 후속 개선

- bbox smoothing
- 최근 N프레임 tracking
- `verticalAngle` 활용
- 단안 bbox 거리추정 fallback
- `worldPosition` 추정
- A* dynamic obstacle 반영

## 11. 최소 연결 예시

아래는 기존 `/detect`의 `filtered_results` 생성 이후에 넣을 수 있는 최소 스니펫이다.

```python
image_width = original_shape[1]

if is_lidar_fresh():
    response_detections = fuse_detections_with_lidar(
        detections=filtered_results,
        lidar_points=planner_state["latest_lidar_points"],
        image_width=image_width,
        hfov_deg=47.81061,
        tolerance_deg=5.0,
    )
else:
    response_detections = attach_no_lidar_distance(filtered_results)

response_detections = attach_risk_to_detections(response_detections)
risk_summary = summarize_risks(response_detections)

planner_state["vision_lidar_risk"] = risk_summary
planner_state["needs_replan"] = risk_summary["shouldStop"]

return jsonify(response_detections)
```

