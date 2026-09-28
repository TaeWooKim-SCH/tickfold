# Tickfold

호가창(LOB) 데이터를 깊이·수량정밀도·가격정밀도·시간해상도 네 축으로 줄였을 때, 저장 용량이 얼마나 줄고 DeepLOB 예측 정확도가 얼마나 떨어지는지 재는 연구. 12월 국내 저널 투고.
살아 있는 계획은 @docs/build-plan-v2.md (v1 저장 포맷·베이스라인 벤치마크는 보류, `docs/build-plan-v1.md`에 보관)

## 스택
Python 3.12, asyncio + websockets + aiohttp, numpy, zstandard, pytest.
단계마다 추가: metrics 착수 시 prometheus_client, 복원기 착수 시 pyarrow, 모델 착수 시 torch·scikit-learn(별도 dependency group), 그림 단계에 matplotlib.

## 명령
- 테스트: `uv run pytest -q`
- 수집기 로컬 실행: `uv run --env-file .env python -m tickfold.collector.main`
- 수집 서비스: `systemctl --user status tickfold-collector` · 로그는 `journalctl --user -u tickfold-collector -f`
- 운영 스택: `docker compose -f deploy/docker-compose.yml up -d` (2절 착수 후)

## 구조
- `tickfold/collector/` 수집기: orderbook.py 순번 검증 · binance_rest.py REST · writer.py 원본 저장 · binance_stream.py combined stream 수집과 갭 재동기화 · main.py 진입점. 예정: metrics.py
- `scripts/` 픽스처 캡처 · `tests/` 픽스처 기반 테스트(`fakes.py` 는 aiohttp 흉내) · `docs/` 계획과 분석 보고서
- 예정: `tickfold/replay/` 복원기(3절) · `tickfold/reduce/` 축소 변환기(4절) · `tickfold/model/` 라벨·학습(5절) · `experiments/` 실험 하네스(6절)
- `deploy/` 운영: tickfold-collector.service systemd 사용자 유닛. 예정: docker-compose(2절)
- `data/`는 git 밖. 절대 삭제·수정하지 말 것. 2026-09-27 부터 수집 서비스가 `data/raw` 에 쓰고 있다

## 규칙
- 원본 페이로드는 재직렬화하지 않는다 (바이트 그대로 저장)
- `tickfold/replay/`를 수정하면 픽스처 재생 결과가 `tests/fixtures/btcusdt_snapshot.json`과 일치하는지 확인한다. `tickfold/reduce/`를 수정하면 항등 설정(깊이 전체, 정밀도 원본, 시간 원해상도)에서 입력 == 출력을 확인한다
- 테스트는 네트워크 없이 돌아야 한다 (fixtures 사용)
- 이벤트를 하나씩 도는 Python 루프 대신 numpy 컬럼 연산. 예외는 복원기 재생 루프 하나 — 순번 검증 때문에 순차여야 한다. 배열이 된 뒤부터는 컬럼 연산
- 호가 데이터 품질은 "오차 없음"이 아니라 측정한 불일치율과 커버리지로 말한다. 방법은 분석 보고서 5.2절
- 새 설계 결정은 docs/build-plan-v2.md 해당 절 메모의 맞는 주제 아래에 넣는다. 이미 있는 줄과 겹치면 줄을 더하지 않고 그 줄을 고친다. 체크리스트와 맨 위 판정 기준도 같이 맞춘다. v1 파일은 보관용이라 고치지 않는다
- 패키지 설치나 실행은 무조건 가상환경 위에서
