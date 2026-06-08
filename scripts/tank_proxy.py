from flask import Flask, request, Response
import requests

app = Flask(__name__)

UBUNTU_SERVER = "http://192.168.0.14:5000"  # Ubuntu PC IP로 수정

# 큰 흐름:
# 1. 로컬 Windows 쪽에서 받은 HTTP 요청을 그대로 Ubuntu 서버로 전달합니다.
# 2. GET은 query string, POST는 파일/form/json/raw body 형태를 보존해서 넘깁니다.
# 3. Ubuntu 서버의 응답 body/status/content-type을 다시 로컬 호출자에게 돌려줍니다.


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    target_url = f"{UBUNTU_SERVER}/{path}"

    try:
        # GET 요청은 URL 파라미터만 보존하면 되므로 가장 단순하게 전달합니다.
        if request.method == "GET":
            resp = requests.get(
                target_url,
                params=request.args,
                timeout=10
            )

        elif request.method == "POST":
            # POST는 simulator가 이미지 파일을 보낼 수도 있고 JSON/body를 보낼 수도 있어
            # 실제 payload 형태를 먼저 판별한 뒤 같은 형태로 Ubuntu 서버에 재전송합니다.
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

        # proxy는 응답을 해석하지 않고, 원격 서버 응답을 최대한 그대로 통과시킵니다.
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type")
        )

    except Exception as e:
        return Response(f"Proxy error: {e}", status=500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
