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

Useful runtime flags:

```bash
YOLO_TIMING=1 python scripts/run_yolo_server.py
YOLO_SHADOW_FILTER=0 python scripts/run_yolo_server.py
YOLO_MODEL_PATH=runs/detect/your_run/weights/best.pt python scripts/run_yolo_server.py
```

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
