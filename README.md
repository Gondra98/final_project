# TankSimulation

Tank 시뮬레이터 API 서버와 YOLO 인식 유틸리티입니다.

## 팀원 빠른 시작

팀원이 새로 클론한 상태에서 이 브랜치를 실행하려면 아래 순서대로 진행하면 됩니다.

```powershell
git clone https://github.com/RT-FINAL-2TEAM/TankSimulation.git
cd TankSimulation

git remote add evann https://github.com/Evann9/TankSimulation-yolo.git
git fetch evann
git checkout -b fix/yolo evann/fix/yolo
```

가상환경을 만들고 활성화합니다:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

필요한 패키지를 설치합니다:

```powershell
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
```

모델 가중치는 Git에 커밋하지 않습니다. 학습된 최종 `best_final.engine` 파일은 팀원에게 따로 공유한 뒤,
아래 경로에 넣어 주세요:

```text
models/tank_detector/best_final.engine
```

또는 실행 전에 모델 경로를 직접 지정할 수 있습니다:

```powershell
$env:YOLO_MODEL_PATH="models/tank_detector/best_final.engine"
python scripts/yolo_detection_server.py
```

서버가 정상 실행 중인지 확인합니다:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/debug_state
```

서버는 기본적으로 `0.0.0.0:5000`에 바인딩됩니다. 시뮬레이터가 다른 PC에서 실행된다면,
서버를 실행한 PC의 IP 주소와 `5000` 포트로 연결하면 됩니다.

## 프로젝트 구조

- `configs/`: 시뮬레이터 연결 설정
- `scripts/yolo_detection_server.py`: YOLO 탐지를 포함한 Flask 서버
- `scripts/yolo_live_view_server.py`: YOLO 탐지 결과를 웹 화면에서 확인하는 디버그 서버
- `scripts/run_simulator_client.py`: 시뮬레이터 클라이언트 실행 진입점
- `scripts/train_yolo_detector.py`: Roboflow 데이터셋 병합 및 YOLO 파인튜닝 스크립트
- `src/`: 재사용 가능한 시뮬레이터, 인식, 경로 계획, RL 모듈
- `tests/`: 자동화 테스트

생성된 데이터셋, 학습 결과, 모델 가중치, 캐시, 에디터 파일은 Git에 올라가지 않도록 제외되어 있습니다.

## 설치

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
```

## YOLO 서버 실행

```bash
python scripts/yolo_detection_server.py
```

기본 실행 설정은 YOLO 단독 데모 탐지에 맞춰져 있습니다:

- `YOLO_IMGSZ=512`
- `YOLO_MODEL_CONF=0.10`
- `YOLO_DEFAULT_CONF=0.20`
- `YOLO_WALL_CONF=0.15`
- `YOLO_MAX_DET=20`
- `YOLO_MAX_RETURN=5`
- `YOLO_DETECT_CACHE=true`
- `YOLO_MIN_INTERVAL=0.12`
- `YOLO_LOW_CONF_FALLBACK=false`
- `YOLO_TIMING=false`
- `YOLO_RECOGNITION_LOG=true`
- `YOLO_RECOGNITION_LOG_CACHE=false`
- `YOLO_RECOGNITION_LOG_EMPTY=false`
- `SIM_DETECT_MODE=true`
- `FLASK_THREADED=false`

Windows PowerShell에서 데모용 실행 옵션:

```powershell
$env:YOLO_MODEL_PATH="models/tank_detector/best_final.engine"
$env:YOLO_IMGSZ="512"
$env:YOLO_MODEL_CONF="0.10"
$env:YOLO_DEFAULT_CONF="0.20"
$env:YOLO_WALL_CONF="0.15"
$env:YOLO_MAX_DET="20"
$env:YOLO_MAX_RETURN="5"
$env:YOLO_DETECT_CACHE="true"
$env:YOLO_MIN_INTERVAL="0.12"
$env:YOLO_LOW_CONF_FALLBACK="false"
$env:YOLO_TIMING="false"
$env:YOLO_RECOGNITION_LOG="true"
$env:YOLO_RECOGNITION_LOG_CACHE="false"
$env:YOLO_RECOGNITION_LOG_EMPTY="false"
$env:FLASK_THREADED="false"
python scripts/yolo_detection_server.py
```

더 빠른 대신 정확도는 낮아질 수 있는 테스트:

```powershell
$env:YOLO_IMGSZ="512"
python scripts/yolo_detection_server.py
```

현재 서버 상태를 확인합니다:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/debug_state
```

최종 `/detect` 결과에 객체가 포함되면 콘솔에 인식 로그가 출력됩니다:

```text
[detect] 2 object(s) recognized
[detect] class=rock conf=0.86 bbox=[45.9, 414.9, 251.7, 502.2]
[detect] class=tank conf=0.81 bbox=[500.1, 320.4, 700.5, 480.2]
```

로그가 너무 많이 출력된다면:

```powershell
$env:YOLO_RECOGNITION_LOG="false"
python scripts/yolo_detection_server.py
```

캐시된 탐지 응답도 인식 로그에 포함하려면:

```powershell
$env:YOLO_RECOGNITION_LOG_CACHE="true"
python scripts/yolo_detection_server.py
```

탐지 결과가 비어 있는 경우도 로그로 남기려면:

```powershell
$env:YOLO_RECOGNITION_LOG_EMPTY="true"
python scripts/yolo_detection_server.py
```

디버그용 실행 옵션은 더 무겁지만 탐지 실패 원인을 찾을 때 유용합니다. `YOLO_LOW_CONF_FALLBACK=true`는 결과가 비어 있을 때 YOLO를 한 번 더 실행할 수 있으므로, 문제를 진단하는 상황이 아니라면 데모에서는 꺼두는 편이 좋습니다.

```powershell
$env:YOLO_MODEL_PATH="models/tank_detector/best_final.engine"
$env:YOLO_DETECT_CACHE="false"
$env:YOLO_LOW_CONF_FALLBACK="true"
$env:YOLO_RETURN_FALLBACK_DETECTIONS="true"
$env:YOLO_MODEL_CONF="0.15"
$env:YOLO_FALLBACK_MODEL_CONF="0.05"
$env:YOLO_DEFAULT_CONF="0.20"
$env:YOLO_WALL_CONF="0.15"
$env:YOLO_DETECT_DEBUG="true"
$env:YOLO_TIMING="true"
$env:YOLO_RECOGNITION_LOG="true"
$env:FLASK_THREADED="false"
python scripts/yolo_detection_server.py
```

반환 필터를 완전히 우회하려면 다음 옵션도 추가합니다:

```powershell
$env:YOLO_BYPASS_RETURN_FILTER="true"
$env:YOLO_MODEL_CONF="0.05"
$env:YOLO_DEFAULT_CONF="0.05"
$env:YOLO_WALL_CONF="0.05"
```

`/debug_state`로 실패 원인을 나눠서 확인할 수 있습니다:

- `modelPath`: 어떤 `best_final.engine`가 로드되었는지 확인
- `modelNames`와 `publicNames`: 모델 클래스 ID가 반환되는 `className` 값과 어떻게 매핑되는지 확인
- `recognitionLogEnabled`: 콘솔 인식 로그가 켜져 있는지 확인
- `latestRawDetectionCount=0`: 모델 경로, 모델 품질, 입력 이미지 문제 가능성
- `latestRawDetectionCount>0`와 `latestReturnedDetectionCount=0`: 반환 임계값 또는 필터 문제 가능성
- `latestDetectCached=true`: 캐시된 결과가 반환된 상태이므로 `latestCacheReason` 확인
- `modelPathFromEnv=false`: 서버가 `YOLO_MODEL_PATH`가 아니라 기본 최종 가중치를 사용한 상태
- `latestFrameShape`, `latestFrameMean`, `latestFrameStd`: 시뮬레이터 이미지가 정상 디코딩되는지 확인
- `latestRejectedDetections`: `below_default_threshold` 같은 박스별 필터 제외 이유 확인
- `latestFallbackUsed=true`: 일반 confidence 탐지에서는 결과가 없었고 low-confidence fallback이 최신 결과를 만든 상태

## YOLO 학습 또는 파인튜닝

데이터셋을 다운로드하기 전에 Roboflow API 키를 설정합니다:

```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/train_yolo_detector.py
```

Windows PowerShell에서는 다음처럼 설정합니다:

```powershell
$env:ROBOFLOW_API_KEY="your_key_here"
python scripts/train_yolo_detector.py
```

모델 가중치나 다운로드한 데이터셋은 커밋하지 마세요. 팀에서 학습된 가중치를 공유해야 한다면 Git LFS나 릴리스 아티팩트를 사용하세요.
