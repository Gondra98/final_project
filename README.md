# TankSimulation

Tank simulator API server and YOLO perception utilities.

## Project Layout

- `configs/`: simulator connection settings
- `docs/`: project and API notes
- `scripts/run_yolo_server.py`: Flask server with YOLO detection
- `scripts/run_simulator_client.py`: simulator client entrypoint
- `scripts/train_yolo_wall_detector.py`: Roboflow dataset merge and YOLO fine-tuning script
- `src/`: reusable simulator, perception, planning, and RL modules
- `tests/`: automated tests

Generated datasets, training runs, model weights, caches, and editor files are intentionally ignored by Git.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
```

## Run The YOLO Server

```bash
python scripts/run_yolo_server.py
```

Default runtime behavior is tuned for YOLO-only demo detection:

- `YOLO_IMGSZ=512`
- `YOLO_MODEL_CONF=0.10`
- `YOLO_DEFAULT_CONF=0.20`
- `YOLO_WALL_CONF=0.15`
- `YOLO_MAX_DET=20`
- `YOLO_MAX_RETURN=5`
- `YOLO_SHADOW_FILTER=false`
- `YOLO_DETECT_CACHE=true`
- `YOLO_MIN_INTERVAL=0.12`
- `YOLO_LOW_CONF_FALLBACK=false`
- `YOLO_TIMING=false`
- `YOLO_RECOGNITION_LOG=true`
- `YOLO_RECOGNITION_LOG_CACHE=false`
- `YOLO_RECOGNITION_LOG_EMPTY=false`
- `SIM_DETECT_MODE=true`
- `FLASK_THREADED=false`

Demo runtime flags on Windows PowerShell:

```powershell
$env:YOLO_MODEL_PATH="runs/detect/finetune_tankkk2_valfix_30/weights/best.pt"
$env:YOLO_IMGSZ="512"
$env:YOLO_MODEL_CONF="0.10"
$env:YOLO_DEFAULT_CONF="0.20"
$env:YOLO_WALL_CONF="0.15"
$env:YOLO_MAX_DET="20"
$env:YOLO_MAX_RETURN="5"
$env:YOLO_SHADOW_FILTER="false"
$env:YOLO_DETECT_CACHE="true"
$env:YOLO_MIN_INTERVAL="0.12"
$env:YOLO_LOW_CONF_FALLBACK="false"
$env:YOLO_TIMING="false"
$env:YOLO_RECOGNITION_LOG="true"
$env:YOLO_RECOGNITION_LOG_CACHE="false"
$env:YOLO_RECOGNITION_LOG_EMPTY="false"
$env:FLASK_THREADED="false"
python scripts/run_yolo_server.py
```

For a faster accuracy tradeoff test:

```powershell
$env:YOLO_IMGSZ="416"
python scripts/run_yolo_server.py
```

Debug the active server state:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/debug_state
```

Bounding box colors by returned `className`:

- `person`: `#00FFFF`
- `rock`: `#FFA500`
- `tank`: `#FF0000`
- `wall`: `#00FF00`
- `tent`: `#FFFF00`

Recognition logs print to the console when final `/detect` results contain objects:

```text
[detect] 2 object(s) recognized
[detect] class=rock conf=0.86 bbox=[45.9, 414.9, 251.7, 502.2]
[detect] class=tank conf=0.81 bbox=[500.1, 320.4, 700.5, 480.2]
```

If logs are too noisy:

```powershell
$env:YOLO_RECOGNITION_LOG="false"
python scripts/run_yolo_server.py
```

To include cached detection responses in recognition logs:

```powershell
$env:YOLO_RECOGNITION_LOG_CACHE="true"
python scripts/run_yolo_server.py
```

To log empty detection results:

```powershell
$env:YOLO_RECOGNITION_LOG_EMPTY="true"
python scripts/run_yolo_server.py
```

Debug runtime flags are heavier but better for finding detection failures. `YOLO_LOW_CONF_FALLBACK=true` can run a second YOLO pass on empty results, so keep it off for demos unless you are diagnosing a problem.

```powershell
$env:YOLO_MODEL_PATH="runs/detect/finetune_tankkk2_valfix_30/weights/best.pt"
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
python scripts/run_yolo_server.py
```

For a full filter-bypass pass, add:

```powershell
$env:YOLO_BYPASS_RETURN_FILTER="true"
$env:YOLO_MODEL_CONF="0.05"
$env:YOLO_DEFAULT_CONF="0.05"
$env:YOLO_WALL_CONF="0.05"
```

Use `/debug_state` to separate failure causes:

- `modelPath`: confirms which `best.pt` was loaded
- `modelNames` and `publicNames`: confirm model class IDs map to returned `className` values
- `classColors`: confirms returned `color` values for each class
- `recognitionLogEnabled`: confirms console recognition logging is enabled
- `latestRawDetectionCount=0`: model path, model quality, or input image issue
- `latestRawDetectionCount>0` and `latestReturnedDetectionCount=0`: return threshold/filter issue
- `latestDetectCached=true`: cached result was returned; check `latestCacheReason`
- `modelPathFromEnv=false`: server used auto-selected weights, not `YOLO_MODEL_PATH`
- `latestFrameShape`, `latestFrameMean`, `latestFrameStd`: confirms the simulator image is being decoded
- `latestRejectedDetections`: shows per-box filter reasons such as `below_default_threshold`
- `latestFallbackUsed=true`: the normal confidence pass found nothing and the low-confidence fallback produced the latest result

## Train Or Fine-Tune YOLO

Set the Roboflow API key before downloading datasets:

```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/train_yolo_wall_detector.py
```

On Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="your_key_here"
python scripts/train_yolo_wall_detector.py
```

Do not commit model weights or downloaded datasets. Use Git LFS or a release artifact if the team needs to share trained weights.
