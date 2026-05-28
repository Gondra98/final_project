from flask import Flask, request, jsonify, Response
from datetime import datetime
import threading

app = Flask(__name__)

lock = threading.Lock()

latest_state = {
    "info": None,
    "position": None,
    "turret": None,
    "destination": None,
    "obstacle": None,
    "collision": None,
    "bullet": None,
    "updated_at": None,
}

latest_action = {
    "moveWS": {"command": "STOP", "weight": 1.0},
    "moveAD": {"command": "", "weight": 0.0},
    "turretQE": {"command": "", "weight": 0.0},
    "turretRF": {"command": "", "weight": 0.0},
    "fire": False,
}

latest_detections = []
latest_image_bytes = None
latest_path = []


def now():
    return datetime.now().isoformat(timespec="seconds")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "updated_at": now()
    })


@app.route("/init", methods=["GET"])
def init():
    config = {
        "startMode": "start",

        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,

        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,

        "trackingMode": True,

        # 중요:
        # 시뮬레이터가 /detect로 이미지를 보내고
        # 서버가 클라이언트 YOLO 결과를 반환하게 하려면 True
        "detectMode": True,

        "logMode": False,
        "stereoCameraMode": False,
        "enemyTracking": False,

        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,

        "lux": 30000,
        "destoryObstaclesOnHit": True,
    }

    print("🛠️ /init sent")
    return jsonify(config)


@app.route("/start", methods=["GET"])
def start():
    print("🚀 /start received")
    return jsonify({"control": ""})


@app.route("/info", methods=["POST"])
def info():
    data = request.get_json(force=True)

    with lock:
        latest_state["info"] = data
        latest_state["updated_at"] = now()

    return jsonify({
        "status": "success",
        "control": ""
    })


@app.route("/get_action", methods=["POST"])
def get_action():
    data = request.get_json(force=True)

    with lock:
        latest_state["position"] = data.get("position", {})
        latest_state["turret"] = data.get("turret", {})
        latest_state["updated_at"] = now()

        action = latest_action.copy()

    print("📨 position:", data.get("position", {}))
    print("📤 action:", action)

    return jsonify(action)


@app.route("/detect", methods=["POST"])
def detect():
    """
    시뮬레이터가 이미지 전송.
    서버는 YOLO를 돌리지 않고 이미지만 저장한다.
    반환값은 yolo_client.py가 미리 넣어둔 latest_detections.
    """

    global latest_image_bytes

    image = request.files.get("image")

    if image is None:
        return jsonify([])

    image_bytes = image.read()

    with lock:
        latest_image_bytes = image_bytes
        detections = list(latest_detections)

    return jsonify(detections)


@app.route("/stereo_image", methods=["POST"])
def stereo_image():
    return jsonify({"result": "success"})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    data = request.get_json(force=True)

    if not data or "destination" not in data:
        return jsonify({
            "status": "ERROR",
            "message": "Missing destination data"
        }), 400

    try:
        x, y, z = map(float, data["destination"].split(","))

        with lock:
            latest_state["destination"] = {
                "x": x,
                "y": y,
                "z": z,
            }
            latest_state["updated_at"] = now()

        print("🎯 destination:", latest_state["destination"])

        return jsonify({
            "status": "OK",
            "destination": latest_state["destination"]
        })

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 400


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    data = request.get_json(force=True)

    with lock:
        latest_state["obstacle"] = data
        latest_state["updated_at"] = now()

    print("🪨 obstacle:", data)

    return jsonify({
        "status": "success",
        "message": "Obstacle data received"
    })


@app.route("/collision", methods=["POST"])
def collision():
    data = request.get_json(force=True)

    with lock:
        latest_state["collision"] = data
        latest_state["updated_at"] = now()

    print("💥 collision:", data)

    return jsonify({
        "status": "success",
        "message": "Collision data received"
    })


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    data = request.get_json(force=True)

    with lock:
        latest_state["bullet"] = data
        latest_state["updated_at"] = now()

    print("💥 bullet:", data)

    return jsonify({
        "status": "OK",
        "message": "Bullet impact data received"
    })


# ==========================
# 클라이언트 모듈용 API
# ==========================

@app.route("/worker/state", methods=["GET"])
def worker_get_state():
    with lock:
        state = dict(latest_state)

    return jsonify(state)


@app.route("/worker/action", methods=["POST"])
def worker_set_action():
    global latest_action

    data = request.get_json(force=True)

    required_keys = ["moveWS", "moveAD", "turretQE", "turretRF", "fire"]

    for key in required_keys:
        if key not in data:
            return jsonify({
                "status": "error",
                "message": f"Missing key: {key}"
            }), 400

    with lock:
        latest_action = data

    print("✅ action updated by client:", latest_action)

    return jsonify({
        "status": "success",
        "action": latest_action
    })


@app.route("/worker/frame", methods=["GET"])
def worker_get_frame():
    """
    yolo_client.py가 최신 시뮬레이터 이미지를 가져가는 API.
    """

    with lock:
        image_bytes = latest_image_bytes

    if image_bytes is None:
        return jsonify({
            "status": "error",
            "message": "No image received yet"
        }), 404

    return Response(image_bytes, mimetype="image/jpeg")


@app.route("/worker/detections", methods=["POST"])
def worker_set_detections():
    """
    yolo_client.py가 YOLO 결과를 서버에 넣는 API.
    이 값이 다음 /detect 응답으로 시뮬레이터에 전달됨.
    """

    global latest_detections

    data = request.get_json(force=True)

    if not isinstance(data, list):
        return jsonify({
            "status": "error",
            "message": "detections must be list"
        }), 400

    with lock:
        latest_detections = data

    print(f"✅ detections updated: {len(latest_detections)} boxes")

    return jsonify({
        "status": "success",
        "count": len(latest_detections)
    })


@app.route("/worker/path", methods=["POST"])
def worker_set_path():
    global latest_path

    data = request.get_json(force=True)
    path = data.get("path", [])

    if not isinstance(path, list):
        return jsonify({
            "status": "error",
            "message": "path must be list"
        }), 400

    with lock:
        latest_path = path

    print(f"🧭 path updated: {len(latest_path)} points")

    return jsonify({
        "status": "success",
        "path_length": len(latest_path)
    })


@app.route("/debug", methods=["GET"])
def debug():
    with lock:
        data = {
            "state": latest_state,
            "action": latest_action,
            "detections": latest_detections,
            "path": latest_path,
            "has_image": latest_image_bytes is not None,
        }

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)