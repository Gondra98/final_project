"""시뮬레이터 클라이언트 진입점.

실제 실행 로직은 `src.main.main()`에 두고, 이 파일은 `scripts` 폴더에서
간단히 실행할 수 있는 얇은 wrapper 역할만 합니다.
"""

from src.main import main


if __name__ == "__main__":
    # `python scripts/run_simulator_client.py`로 실행했을 때 src.main.main()을 호출합니다.
    main()
