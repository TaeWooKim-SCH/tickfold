"""
UTC 날짜마다 호가창을 복원해 종목·날짜별 .npz 로 쓰고(restore), 쓴 파일만 읽어 대조와 커버리지를 더한다(summarize).

사용법 (프로젝트 루트에서):
    nice -n 19 ionice -c3 uv run python scripts/restore_days.py restore 2026-09-27 2026-10-16
    nice -n 19 ionice -c3 uv run python scripts/restore_days.py summarize

restore 는 날짜마다 프로세스를 하나 띄워 넷까지 같이 돌린다. summarize 는 전체 기간과 72시간 판정 창(--window 를 주면 그 창들)을 찍는다.
"""

import argparse
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple
from datetime import date, datetime, timedelta, timezone
from itertools import repeat
from pathlib import Path
from time import monotonic

from tickfold.replay.restored_day import WindowSummary, restore_day, save_restored_day, summarize_window
from tickfold.replay.segment import DELIVERY_MARGIN

# 갈아타기 코드를 배포해 다시 띄운 뒤 처음 저장된 프레임부터 72시간이다(2026-10-12 결정)
JUDGMENT_WINDOW = ("2026-10-06T05:14:40.814", "2026-10-09T05:14:40.814")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def utc_time(text: str) -> datetime:
    """명령줄의 시각을 읽는다. 시간대가 없으면 UTC 로 본다."""
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def restore_and_save(symbol_directory: Path, days_directory: Path, day: date) -> str:
    """날짜 하나를 복원해 쓰고, 걸린 시간과 최대 메모리와 파일 크기를 한 줄로 돌려준다. 날짜마다 새 프로세스에서 돈다."""
    began = monotonic()
    restored = restore_day(symbol_directory, day)
    path = days_directory / f"{day}.npz"
    save_restored_day(restored, path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # 리눅스는 KB 로 준다
    return (f"{day}: 행 {len(restored.top.exchange_time):,}, REST {len(restored.snapshots)}개, "
            f"{monotonic() - began:.0f}초, 최대 메모리 {peak:.2f}GB, {path.stat().st_size / 1e9:.3f}GB")


def describe(start: datetime, end: datetime, summary: WindowSummary) -> str:
    """창 하나의 숫자를 줄들로 만든다. 논문 4장 표와 판정 기준에 옮겨 적는 값이다."""
    depth20, snapshots, coverage = summary.depth20, summary.snapshots, summary.coverage
    lines = [f"창 {start.isoformat()} ~ {end.isoformat()}, 행 {summary.rows:,}, "
             f"depth20 동기화 안 됨 {depth20.unsynced_points:,}, 짝 없는 depth20 {depth20.unpaired_depth20:,}, "
             f"짝 없는 행 {depth20.unpaired_rows:,}"]
    for name, line in (("범위 안", depth20.inside_range), ("범위 밖", depth20.outside_range), ("전체", depth20.whole)):
        lines.append(
            f"  depth20 {name}: {line.points:,}시점 중 불일치 {line.mismatched_points:,}({line.point_mismatch_rate:.4%}), "
            f"레벨 {_levels(line.level_counts)}, 누락 중 범위 밖 가격 {line.missing_beyond_known_range:,}"
        )
    lines.append(f"  REST 견줌 {snapshots.compared_snapshots:,}, 견줄 수 없음 {snapshots.uncomparable_snapshots:,}, "
                 f"범위 안 레벨 {_levels(snapshots.inside_range)}, 범위 밖 레벨 {_levels(snapshots.outside_range)}")
    window, gap, unsynced, outside = (f"{value / 1000:,.3f}초" for value in astuple(coverage))
    first, last = ((_EPOCH + timedelta(milliseconds=edge)).isoformat(timespec="milliseconds")
                   for edge in (summary.coverage_start, summary.coverage_end))
    lines.append(f"  커버리지 {first} ~ {last} {window}: 갭 {gap}, 동기화 안 됨 {unsynced}, 범위 밖 {outside}. "
                 f"갭만 뺀 {coverage.gap_free_ratio:.4%}, 재동기화까지 뺀 {coverage.synced_ratio:.4%}(판정), "
                 f"범위 밖까지 뺀 {coverage.valid_ratio:.4%}")
    waits = ["못 봄" if wait is None else f"{wait / 1000:.3f}초" for _, wait in summary.gaps]
    gaps = [f"{length / 1000:.3f}초(재동기화 {wait})" for (length, _), wait in zip(summary.gaps, waits)]
    lines.append(f"  갭 {len(gaps)}개: {', '.join(gaps)}")
    return "\n".join(lines)


def _levels(levels) -> str:
    return (f"{levels.exchange_levels:,}개 중 누락 {levels.missing:,} 수량 {levels.quantity_differences:,} "
            f"남음 {levels.extra:,}({levels.mismatch_rate:.5%})")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="BTCUSDT")
    common.add_argument("--data-root", type=Path, default=Path(os.environ.get("TICKFOLD_DATA_ROOT", "data/raw")))
    common.add_argument("--output", type=Path, default=Path("restored"), help="날짜별 .npz 폴더. 그 아래 종목 폴더에 쓴다")
    parser = argparse.ArgumentParser(description="호가창을 UTC 날짜별 .npz 로 복원하고, 쓴 파일로 합계를 낸다")
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore", parents=[common], help="날짜 범위를 날짜마다 .npz 하나로 쓴다")
    restore.add_argument("first_day", type=date.fromisoformat, help="첫 날짜(포함). UTC, 예: 2026-09-27")
    restore.add_argument("last_day", type=date.fromisoformat, help="끝 날짜(포함). UTC")
    # 수집기와 같은 서버라 넷까지만 같이 돌린다
    restore.add_argument("--processes", type=int, default=4, choices=range(1, 5))
    summarize = commands.add_parser("summarize", parents=[common], help="쓴 파일만 읽어 전체 기간과 창을 잰다")
    summarize.add_argument(
        "--window", nargs=2, type=utc_time, action="append", metavar=("START", "END"),
        help="UTC 거래소 시각 창. 여러 번 줄 수 있고, 안 주면 72시간 판정 창",
    )
    arguments = parser.parse_args()
    symbol_directory, days_directory = arguments.data_root / arguments.symbol, arguments.output / arguments.symbol

    if arguments.command == "restore":
        first, last = arguments.first_day, arguments.last_day
        if first > last:
            parser.error(f"첫 날짜 {first} 가 끝 날짜 {last} 보다 뒤다")
        # 오늘은 수집 중이다. 끝난 날짜도 끝 프레임은 늦게 와 5초마다 쓰이고, 재생은 날짜 끝 뒤 DELIVERY_MARGIN 까지 읽는다
        if last >= (datetime.now(timezone.utc) - DELIVERY_MARGIN - timedelta(seconds=5)).date():
            parser.error(f"{last} 는 아직 안 끝났거나 막 끝난 UTC 날짜다. 끝 프레임까지 다 쓰인 날짜만 만든다")
        # 원본 폴더에는 수집기만 쓴다
        if arguments.output.resolve().is_relative_to(arguments.data_root.resolve()):
            parser.error(f"{arguments.output} 은 원본 폴더 {arguments.data_root} 아래다. 다른 곳에 쓴다")
        days_directory.mkdir(parents=True, exist_ok=True)
        days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
        # 날짜마다 새 프로세스를 띄운다. 최대 메모리가 날짜별로 재지고 앞 날짜가 잡은 메모리를 다음 날짜가 안 물려받는다
        with ProcessPoolExecutor(arguments.processes, max_tasks_per_child=1) as pool:
            for line in pool.map(restore_and_save, repeat(symbol_directory), repeat(days_directory), days):
                print(line, flush=True)
        return

    stored = sorted(days_directory.glob("????-??-??.npz"))
    if not stored:
        parser.error(f"{days_directory} 에 날짜 파일이 없다")
    first, last = (utc_time(path.stem) for path in (stored[0], stored[-1]))
    windows = [(first, last + timedelta(days=1)), *(arguments.window or [tuple(map(utc_time, JUDGMENT_WINDOW))])]
    for start, end in windows:
        milliseconds = [(edge - _EPOCH) // timedelta(milliseconds=1) for edge in (start, end)]
        print(describe(start, end, summarize_window(days_directory, *milliseconds, symbol_directory)), flush=True)


if __name__ == "__main__":
    main()