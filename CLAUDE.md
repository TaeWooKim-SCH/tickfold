# Tickfold

호가창(LOB) 델타 스트림 저장 포맷 + 공개 벤치마크. 구현 순서와 항목은 @docs/build-plan.md

## 스택
Python 3.12, asyncio + websockets, numpy, zstandard, pyarrow, pytest

## 명령
- 테스트: `pytest -q`
- 수집기 로컬 실행: `python -m tickfold.collector.main`
- 운영 스택: `docker compose -f deploy/docker-compose.yml up -d`

## 구조
- `tickfold/collector/` 수집기 · `tickfold/io/` 파서 · `tickfold/format/` 인코더·디코더·쿼리
- `tickfold/baselines/` 베이스라인 적재 · `tickfold/bench/` 하네스 · `deploy/` 운영
- `data/`는 git 밖. 절대 삭제·수정하지 말 것

## 규칙
- 원본 페이로드는 재직렬화하지 않는다 (바이트 그대로 저장)
- `tickfold/format/`을 수정하면 반드시 `pytest tests/test_roundtrip.py` 실행
- 테스트는 네트워크 없이 돌아야 한다 (fixtures 사용)
- 이벤트를 하나씩 도는 Python 루프 대신 numpy 컬럼 연산
- 새 설계 결정은 docs/build-plan.md의 메모에 한 줄 추가
- 패키지 설치나 실행은 무조건 가상환경 위에서