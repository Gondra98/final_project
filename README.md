# Tank Autonomous Driving Project

전차 시뮬레이터 환경에서 장애물 회피 및 강화학습 기반 자율주행 경로 생성을 구현하는 프로젝트입니다.

## Goals

- 시뮬레이터 API 연동
- 라이다 및 장애물 데이터 분석
- 장애물 회피 알고리즘 구현
- 강화학습 기반 경로 생성
- 주행 성능 평가 및 시각화

## Project Structure

- `src/simulator`: 시뮬레이터 API 통신
- `src/perception`: 장애물 데이터 처리
- `src/planning`: 경로 생성 및 장애물 회피
- `src/control`: 전차 제어 명령 생성
- `src/rl`: 강화학습 환경 및 학습 코드
- `data`: 수집 데이터
- `models`: 학습된 모델
- `docs`: 프로젝트 문서
- `tests`: 테스트 코드

## Setup

```bash
pip install -r requirements.txt