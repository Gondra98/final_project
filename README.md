# TankSimulation YOLO Detection Server

> YOLO-based object detection module for a tank simulator environment.  
> 전차 시뮬레이터 환경에서 객체를 인식하고, Flask API를 통해 탐지 결과를 제공하는 Computer Vision 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-green)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Project-orange)

---

## 1. Project Overview

이 저장소는 전차 시뮬레이터에서 전달되는 이미지를 기반으로 주변 객체를 탐지하고, 탐지 결과를 API 형태로 반환하는 프로젝트입니다.

단순히 YOLO 모델을 실행하는 데서 끝나는 것이 아니라, 실제 시뮬레이터와 연결하기 위해 다음 요소를 함께 구성했습니다.

- YOLO 기반 객체 탐지 서버
- Flask 기반 `/detect` API
- 탐지 결과 캐싱 및 반환 필터링
- confidence threshold 조정
- 디버깅용 상태 확인 API
- Roboflow 데이터셋 기반 YOLO 파인튜닝 스크립트
- 시뮬레이터 클라이언트 및 경로 계획/RL 모듈 연동 구조

이 프로젝트는 **Computer Vision, Object Detection, Simulation, AI-based Situation Awareness** 역량을 보여주기 위한 포트폴리오 프로젝트로 정리했습니다.

---

## 2. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| AI / CV | YOLO, OpenCV |
| API Server | Flask |
| Data / Training | Roboflow, custom training script |
| Environment | Windows PowerShell, Python virtual environment |
| Project Management | Git, GitHub |

---

## 3. Main Features

### YOLO Detection API

시뮬레이터 이미지 입력을 받아 YOLO 모델로 객체를 탐지하고, 결과를 API 응답으로 반환합니다.

주요 파일:

```text
scripts/yolo_detection_server.py
```

### Live View Debug Server

탐지 결과를 웹 화면에서 확인하기 위한 디버그 서버입니다.

```text
scripts/yolo_live_view_server.py
```

### YOLO Fine-tuning Script

Roboflow 데이터셋을 활용해 YOLO 모델을 학습하거나 파인튜닝할 수 있습니다.

```text
scripts/train_yolo_detector.py
```

### Simulator Client

시뮬레이터와 서버를 연결하기 위한 클라이언트 실행 진입점입니다.

```text
scripts/run_simulator_client.py
```

---

## 4. Project Structure

```text
TankSimulation-yolo/
├─ configs/                         # 시뮬레이터 연결 설정
├─ models/                          # 로컬 모델 가중치 저장 위치, Git 제외 권장
├─ scripts/
│  ├─ yolo_detection_server.py       # YOLO 탐지 Flask 서버
│  ├─ yolo_live_view_server.py       # 탐지 결과 확인용 디버그 서버
│  ├─ run_simulator_client.py        # 시뮬레이터 클라이언트
│  └─ train_yolo_detector.py         # YOLO 파인튜닝 스크립트
├─ src/                              # 재사용 가능한 인식, 경로 계획, RL 모듈
├─ tests/                            # 테스트 코드
├─ requirements.txt
├─ requirements-torch-cu128.txt
└─ README.md
```

생성된 데이터셋, 학습 결과, 모델 가중치, 캐시, 에디터 파일은 Git에 올리지 않도록 관리합니다.

---

## 5. Quick Start

### 5.1 Clone Repository

```bash
git clone https://github.com/Evann9/TankSimulation-yolo.git
cd TankSimulation-yolo
```

팀 프로젝트 원본 저장소에서 `fix/yolo` 브랜치를 가져와야 하는 경우에는 아래 방식을 사용할 수 있습니다.

```powershell
git clone https://github.com/RT-FINAL-2TEAM/TankSimulation.git
cd TankSimulation

git remote add evann https://github.com/Evann9/TankSimulation-yolo.git
git fetch evann
git checkout -b fix/yolo evann/fix/yolo
```

### 5.2 Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 5.3 Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
```

---

## 6. Model Weights

학습된 모델 가중치는 Git에 커밋하지 않습니다.

최종 TensorRT 엔진 파일은 아래 경로에 배치합니다.

```text
models/tank_detector/best_final.engine
```

또는 실행 전에 직접 모델 경로를 지정할 수 있습니다.

```powershell
$env:YOLO_MODEL_PATH="models/tank_detector/best_final.engine"
python scripts/yolo_detection_server.py
```

모델 가중치나 다운로드한 데이터셋을 팀원과 공유해야 한다면 Git LFS 또는 GitHub Release Artifact 사용을 권장합니다.

---

## 7. Run YOLO Detection Server

```bash
python scripts/yolo_detection_server.py
```

서버는 기본적으로 `0.0.0.0:5000`에 바인딩됩니다.  
시뮬레이터가 다른 PC에서 실행된다면, 서버를 실행한 PC의 IP 주소와 `5000` 포트로 연결하면 됩니다.

서버 상태 확인:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/debug_state
```

---

## 8. Recommended Demo Settings

Windows PowerShell 기준 데모용 실행 옵션입니다.

```powershell
$env:YOLO_MODEL_PATH="models/tank_detector/best_final.engine"
$env:YOLO_IMGSZ="416"
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

탐지 결과가 정상적으로 반환되면 콘솔에 다음과 같은 인식 로그가 출력됩니다.

```text
[detect] 2 object(s) recognized
[detect] class=rock conf=0.86 bbox=[45.9, 414.9, 251.7, 502.2]
[detect] class=tank conf=0.81 bbox=[500.1, 320.4, 700.5, 480.2]
```

---

## 9. Debugging Guide

### 인식 로그 끄기

```powershell
$env:YOLO_RECOGNITION_LOG="false"
python scripts/yolo_detection_server.py
```

### 캐시된 탐지 응답도 로그에 포함하기

```powershell
$env:YOLO_RECOGNITION_LOG_CACHE="true"
python scripts/yolo_detection_server.py
```

### 탐지 결과가 비어 있는 경우도 로그로 남기기

```powershell
$env:YOLO_RECOGNITION_LOG_EMPTY="true"
python scripts/yolo_detection_server.py
```

### 탐지 실패 원인 진단용 옵션

디버그용 옵션은 더 무겁지만, 탐지 실패 원인을 찾을 때 유용합니다.  
`YOLO_LOW_CONF_FALLBACK=true`는 결과가 비어 있을 때 YOLO를 한 번 더 실행할 수 있으므로, 데모 상황이 아니라 문제 진단 상황에서만 사용하는 것이 좋습니다.

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

반환 필터를 완전히 우회하려면 다음 옵션도 추가합니다.

```powershell
$env:YOLO_BYPASS_RETURN_FILTER="true"
$env:YOLO_MODEL_CONF="0.05"
$env:YOLO_DEFAULT_CONF="0.05"
$env:YOLO_WALL_CONF="0.05"
```

---

## 10. `/debug_state` Checklist

`/debug_state` 응답에서 다음 항목을 확인하면 문제 원인을 빠르게 좁힐 수 있습니다.

| Field | Meaning |
|---|---|
| `modelPath` | 어떤 `best_final.engine` 파일이 로드되었는지 확인 |
| `modelNames`, `publicNames` | 모델 클래스 ID와 반환되는 `className` 매핑 확인 |
| `recognitionLogEnabled` | 콘솔 인식 로그 활성화 여부 확인 |
| `latestRawDetectionCount=0` | 모델 경로, 모델 품질, 입력 이미지 문제 가능성 |
| `latestRawDetectionCount>0`, `latestReturnedDetectionCount=0` | 반환 임계값 또는 필터 문제 가능성 |
| `latestDetectCached=true` | 캐시 결과가 반환된 상태이므로 `latestCacheReason` 확인 |
| `modelPathFromEnv=false` | 환경변수가 아닌 기본 경로의 모델을 사용 중 |
| `latestFrameShape`, `latestFrameMean`, `latestFrameStd` | 시뮬레이터 이미지 디코딩 상태 확인 |
| `latestRejectedDetections` | `below_default_threshold` 같은 필터 제외 이유 확인 |
| `latestFallbackUsed=true` | 일반 탐지 결과가 없어 low-confidence fallback이 사용된 상태 |

---

## 11. YOLO Training / Fine-tuning

Roboflow 데이터셋을 다운로드하기 전에 API 키를 설정합니다.

```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/train_yolo_detector.py
```

Windows PowerShell에서는 다음처럼 설정합니다.

```powershell
$env:ROBOFLOW_API_KEY="your_key_here"
python scripts/train_yolo_detector.py
```

---

## 12. Portfolio Notes

이 프로젝트에서 보여줄 수 있는 핵심 역량은 다음과 같습니다.

- YOLO 기반 객체 탐지 모델 활용
- 시뮬레이터 이미지 입력과 AI 서버 연동
- Flask API 서버 구축
- confidence threshold 및 반환 필터 설계
- 실시간 탐지를 위한 캐싱 구조 적용
- 디버깅 가능한 AI 추론 파이프라인 구성
- 데이터셋 관리와 모델 가중치 관리 방식 이해

---

## 13. Future Improvements

- 탐지 결과 시각화 이미지 추가
- 실제 inference latency 측정 결과 정리
- YOLO 학습 데이터셋 구성 방식 문서화
- 모델 성능 지표 mAP, precision, recall 정리
- Docker 기반 실행 환경 추가
- GitHub Actions 기반 테스트 자동화

---

## Author

**Evann9**  
Computer Vision · Object Detection · GeoAI-oriented AI Projects
