# Tickfold

비트코인 호가창(LOB) 스트림을 빠짐없이 모으고 품질을 수치로 증명하는 수집기. 12월 국내 저널 투고 논문은 이 수집기의 설계와 구현이고, 활용 예로 깊이 20 vs 10 DeepLOB 실험 하나를 붙인다. 10월 할 일(복원기, 전체 기간 대조, 72시간 판정)이 10월 안에 끝나면 그 실험의 예측으로 모의 매매 손익까지 낸다(2026-10-12, 조건은 계획서 5절). 학술제(본선 2026-11-10) 시연용으로 호가창 재생 화면을 10월 말까지 만든다. 네 축(깊이·수량 정밀도·가격 정밀도·시간 해상도) 축소 실험(2026-10-07)과 체결 모사 시뮬레이터(2026-10-12)는 다음 논문으로 미뤘다.
살아 있는 계획은 @docs/build-plan-v2.md (v1 저장 포맷·베이스라인 벤치마크는 보류, `docs/build-plan-v1.md`에 보관)

## 스택
Python 3.12, asyncio + websockets + aiohttp, numpy, zstandard, pytest.
단계마다 추가: metrics 착수 시 prometheus_client, 복원기 착수 시 pyarrow, 모델 착수 시 torch·scikit-learn(별도 dependency group), 그림 단계에 matplotlib(별도 dependency group `figures`), 시연 화면 착수 시 streamlit·plotly(별도 dependency group).

## 명령
- 테스트: `uv run pytest -q` (화면 자바스크립트 시험은 node 로 돈다. PATH 에 node 가 없으면 건너뛴다. screen·model 그룹을 안 깔았으면 그 그룹을 쓰는 시험을 건너뛴다. `-rs` 로 건너뛴 것을 본다)
- 수집기 로컬 실행: `uv run --env-file .env python -m tickfold.collector.main`
- 수집 서비스: `systemctl --user status tickfold-collector` · 로그는 `journalctl --user -u tickfold-collector -f`
- 재생 화면용 구간 내보내기: `nice -n 19 uv run python scripts/export_replay_segment.py 2026-10-07T07:25:00 2026-10-07T07:40:00 segments/gap-0732.npz` (시각은 UTC 거래소 시각이고 구간은 2시간까지. `segments/` 는 git 밖)
- 화면용 패키지 설치: `uv sync --group screen --inexact` (수집 서비스가 같은 `.venv` 를 쓰므로 `--inexact` 로 더하기만 한다)
- 모델용 패키지 설치: `uv sync --group model --inexact` (torch·scikit-learn. 위와 같은 이유로 `--inexact`)
- 그림용 패키지 설치: `uv sync --group figures --inexact` (matplotlib. 모의 매매 손익 곡선. 위와 같은 이유로 `--inexact`)
- 날짜별 복원과 합계: `nice -n 19 ionice -c3 uv run python scripts/restore_days.py restore 2026-09-27 2026-10-16` 뒤 `nice -n 19 ionice -c3 uv run python scripts/restore_days.py summarize` (끝난 UTC 날짜만 받는다. 날짜마다 `restored/BTCUSDT/2026-10-08.npz` 하나, 0.58GB. 넷까지 같이 돌려 20일에 8분. summarize 는 전체 기간과 72시간 판정 창을 찍고, `--window START END` 를 주면 72시간 창 대신 준 창들을 찍는다. `restored/` 는 git 밖)
- 수집기 journal 보관: `journalctl --user -u tickfold-collector --since "2026-09-27 00:00" --until "2026-10-18 09:00" -o short-iso-precise --no-pager > evidence/journal-tickfold-collector-2026-09-27_2026-10-17.txt` 와 같은 범위의 `-o json` 을 뜨고 `sha256sum evidence/journal-* > evidence/SHA256SUMS` (시각은 KST. 판정을 새로 하면 범위를 늘려 다시 뜬다)
- 깊이 20 vs 10 학습: `nice -n 19 ionice -c3 uv run --group model python scripts/train_deeplob.py` (`restored/BTCUSDT` 를 읽어 `runs/deeplob/` 에 깊이·시드마다 result.json·predictions.npz·model.pt, 끝에 summary.json. 평가 구간은 설정을 다 정한 뒤 `--evaluate` 를 준 실행에서만 예측하고 채점한다. 깊이 둘 × 시드 셋 25분쯤. GPU 가 안 잡히면 멈춘다. GPU 학습은 서버에서 한 번에 하나. `runs/` 는 git 밖)
- 모의 매매: `nice -n 19 ionice -c3 uv run --group figures python scripts/paper_trading.py --predictions runs/deeplob/depth20-seed0/predictions.npz --figure runs/deeplob/depth20-seed0/paper_trading.png` (분할과 horizon 은 예측 옆 `result.json` 을 따른다. 모델 예측 전에는 `--predictions` 대신 `--random-predictions 0` 이고 그림은 `runs/` 아래에 둔다. 복원 폴더는 `--days-directory`, 기본 `restored/BTCUSDT`. 검증 구간에서 신호 문턱을 고르고 평가 구간을 한 번 돈다. 무작위 예측으로 열흘치에 4분, 최대 메모리 1.4GB. 서버에 한글 글꼴이 없어 그림 글자는 영어다)
- 재생 화면: `nice -n 19 uv run --group screen streamlit run tickfold/screen/app.py` (구간 폴더는 `TICKFOLD_SEGMENTS`, 기본 `segments`. 127.0.0.1 로만 받는다. 서버 밖에서는 SSH 터널로 8501 포트)
- 운영 스택: `docker compose -f deploy/docker-compose.yml up -d` (2절 착수 후)

## 구조
- `tickfold/collector/` 수집기: orderbook.py 순번 검증 · binance_rest.py REST · writer.py 원본 저장 · binance_stream.py combined stream 수집과 갭 재동기화 · connection.py 웹소켓 연결 유지와 12시간마다 갈아타기 · main.py 진입점. 예정: metrics.py
- `scripts/` 픽스처 캡처 · 재생 화면용 구간 내보내기(export_replay_segment.py) · 날짜별 복원과 합계(restore_days.py. restore 는 날짜마다 프로세스 하나, summarize 는 쓴 파일만 읽는다) · DeepLOB 깊이 20 vs 10 학습(train_deeplob.py) · 모의 매매(paper_trading.py. 검증 구간에서 문턱을 골라 평가 구간의 손익, 기준선 셋, 누적 손익 그림) · `tests/` 픽스처 기반 테스트(`fakes.py` 는 aiohttp·websockets 흉내, `*.test.js` 는 화면 자바스크립트의 node 시험이고 test_screen_javascript.py 가 부른다) · `docs/` 계획과 분석 보고서, 학술제 녹화 가이드(showcase-recording.md)
- `tickfold/replay/` 복원기(3절): files.py 저장 파일을 쓰인 순서로 찾아 읽기 · repeats.py 두 파일에 겹쳐 나온 프레임 가려내기 · depth20.py 거래소가 준 상위 20레벨 프레임을 배열로 읽기 · depth_sequence.py depth 순번과 거래소 시각 읽기, 순번 사슬에서 갭 찾기, depth20 프레임에 거래소 시각 붙이기 · segment.py 화면용 구간 하나를 거래소 시각으로 잘라 npz 로 쓰고 읽기 · book_replay.py 범위 앞의 가장 가까운 스냅샷부터 호가창을 다시 돌리기(스냅샷은 순번 자리에, 갭 뒤는 다음 스냅샷까지 무효) · top_levels.py 재생하며 프레임마다 상위 N 레벨을 depth20 과 같은 모양의 배열로 뽑기(정렬된 가격 목록을 옆에 두고, 상태와 스냅샷 범위 끝도 프레임마다 담는다) · valid_mask.py 상위 N 행마다 valid 두 컬럼(동기화됨, N번째 레벨이 스냅샷 범위 안)과 갭 뒤 첫 행 표시, 거래소 시각 창 하나를 갭·동기화 안 됨·범위 밖 시간으로 나눈 커버리지 · comparison.py 재생한 상위 20 을 저장된 depth20 과, 끼우기 직전 호가창을 REST 스냅샷과 가격끼리 맞춰 견주고 범위 안·밖 줄의 개수로 담기(날짜별 결과를 더하면 한 범위 값이다. REST 는 스냅샷마다 끼운 행과 결과도 낸다) · restored_day.py 한 날짜를 복원해(재생 두 바퀴, 상위 20, valid, depth20 대조, 스냅샷마다 REST 대조) npz 하나로 쓰고 읽기, 저장된 날짜 파일만으로 창 하나의 대조·커버리지·갭 합계
- `tickfold/screen/` 학술제 시연 화면(5절): app.py 구간 파일을 골라 숫자 띠를 보이고 재생 컴포넌트에 넘기는 Streamlit 화면 · player.py 브라우저에 한 번 넘길 값과 컴포넌트 모듈(plotly.js 뒤에 화면 자바스크립트 셋을 이어 붙인다) · player.js 브라우저 쪽 화면과 조작 · playback.js 재생 시계와 재생 상태 · book.js 호가창 한 프레임의 막대와 글자 · figures.py 숫자 띠 값, 수량 상한, mid 선 솎기. 재생과 조작은 브라우저 안에서만 돈다. 구간 파일만 읽고 `data/raw` 에는 붙지 않는다. `.streamlit/config.toml` 은 화면이 127.0.0.1 로만 받게 하는 설정. 학술제에는 녹화 영상을 쓴다(녹화 절차는 docs/showcase-recording.md)
- `tickfold/model/` 깊이 20 vs 10 학습(5절): settings.py 설정 한 곳(#3 의 값) · inputs.py 날짜 파일을 이어 읽어 DeepLOB 열 배치(레벨마다 매도가·매도량·매수가·매수량. 가격은 mid 기준 상대값, 수량은 log1p)로 바꾸고 학습 구간 통계로 정규화 · labels.py 식 3 라벨과 보합 폭, 입력 창과 라벨 구간이 갭·쓸 수 없는 행·구간 경계를 넘지 않는 표본, 모델 없는 기준 · deeplob.py DeepLOB-N(세 번째 블록 첫 커널이 1×N) · training.py 시드 고정 학습, 검증 손실로 일찍 멈춤, 검증 행마다(평가는 넣을 때만) 세 확률 · evaluation.py 정확도·macro-F1·혼동행렬·다수 클래스·모델 없는 기준·날짜별 값과 result.json·predictions.npz·model.pt 쓰기
- `tickfold/paper_trading/` 모의 매매(5절, 조건부). 모델 예측과 기준선이 같은 규칙과 같은 셈을 탄다: market.py 날짜별 복원 파일에서 행마다 최우선 호가와 쓸 수 있는지(valid 두 컬럼이 다 참)를 꺼내 날짜를 이어 붙이기 · walk.py 신호를 받아 포지션 하나로 사고팔기(규칙 설정은 TradingRule 한 곳) · performance.py 수수료 전·후 손익, 거래당 bp, 최대 낙폭, 승률 · baselines.py 항상 보유, 최우선 호가 수량 불균형 규칙, 거래 수를 맞춘 무작위 신호 · predictions.py 모델 갈래가 쓴 predictions.npz 읽기와 신호 문턱, 모델 전 무작위 예측
- 예정: `tickfold/reduce/` 축소 변환기(4절) · `experiments/` 실험 하네스(6절)
- `deploy/` 운영: tickfold-collector.service systemd 사용자 유닛. 예정: docker-compose(2절)
- `data/`는 git 밖. 절대 삭제·수정하지 말 것. 2026-09-27 부터 수집 서비스가 `data/raw` 에 쓰고 있다
- `evidence/` 는 git 밖. 72시간 판정의 근거(재시작 0회, 끊김과 재동기화)로 떠 둔 수집기 journal 의 텍스트·JSON 과 SHA256SUMS. journal 은 시스템이 오래된 것부터 지우므로 이 사본을 지우지 말 것

## 규칙
- 원본 페이로드는 재직렬화하지 않는다 (바이트 그대로 저장)
- `tickfold/replay/`를 수정하면 픽스처 재생 결과가 `tests/fixtures/btcusdt_snapshot.json`과 일치하는지 확인한다. `tickfold/reduce/`를 수정하면 항등 설정(깊이 전체, 정밀도 원본, 시간 원해상도)에서 입력 == 출력을 확인한다
- 테스트는 네트워크 없이 돌아야 한다 (fixtures 사용)
- 이벤트를 하나씩 도는 Python 루프 대신 numpy 컬럼 연산. 예외는 복원기 재생 루프 하나 — 순번 검증 때문에 순차여야 한다. 배열이 된 뒤부터는 컬럼 연산
- 호가 데이터 품질은 "오차 없음"이 아니라 측정한 불일치율과 커버리지로 말한다. 방법은 분석 보고서 5.2절
- 새 설계 결정은 docs/build-plan-v2.md 해당 절 메모의 맞는 주제 아래에 넣는다. 이미 있는 줄과 겹치면 줄을 더하지 않고 그 줄을 고친다. 체크리스트와 맨 위 판정 기준도 같이 맞춘다. v1 파일은 보관용이라 고치지 않는다
- 패키지 설치나 실행은 무조건 가상환경 위에서
