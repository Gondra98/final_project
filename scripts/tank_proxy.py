from flask import Flask, request, Response
import requests

app = Flask(__name__)

UBUNTU_SERVER = "http://192.168.0.14:5000"  # Ubuntu PC IP로 수정

@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    target_url = f"{UBUNTU_SERVER}/{path}"

    try:
        if request.method == "GET":
            resp = requests.get(
                target_url,
                params=request.args,
                timeout=10
            )

        elif request.method == "POST":
            files = {}
            for key, file in request.files.items():
                files[key] = (
                    file.filename,
                    file.stream,
                    file.content_type
                )

            data = request.form.to_dict()
            json_data = request.get_json(silent=True)

            if files:
                resp = requests.post(
                    target_url,
                    files=files,
                    data=data,
                    timeout=30
                )
            elif json_data is not None:
                resp = requests.post(
                    target_url,
                    json=json_data,
                    timeout=10
                )
            else:
                resp = requests.post(
                    target_url,
                    data=request.get_data(),
                    headers={
                        "Content-Type": request.headers.get("Content-Type", "")
                    },
                    timeout=10
                )

        else:
            return Response("Unsupported method", status=405)

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type")
        )

    except Exception as e:
        return Response(f"Proxy error: {e}", status=500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
