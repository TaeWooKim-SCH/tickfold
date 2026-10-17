import json
import runpy
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from test_replay_book_replay import (
    SECOND, SNAPSHOT, advanced, delta, depth_lines, one_day_later, received_at, utc_day,
)
from test_replay_comparison import as_snapshot, store_across_midnight, store_with_depth20
from test_replay_depth20 import fixture_depth20_lines
from tickfold.replay.book_replay import BookReplay
from tickfold.replay.comparison import compare_depth20, compare_snapshots
from tickfold.replay.restored_day import load_restored_day, restore_day, save_restored_day, summarize_window
from tickfold.replay.top_levels import collect_top_levels
from tickfold.replay.valid_mask import mark_valid, measure_coverage

SCRIPT = str(Path(__file__).parent.parent / "scripts" / "restore_days.py")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def milliseconds(time: datetime) -> int:
    return (time - EPOCH) // timedelta(milliseconds=1)


def utc(milliseconds: int) -> datetime:
    return EPOCH + timedelta(milliseconds=milliseconds)


def save_days(symbol_directory: Path, days_directory: Path, days: list[int]) -> None:
    """날짜마다 복원해 쓰고, 다시 읽은 값이 만든 값과 같은지 본다."""
    days_directory.mkdir(parents=True)
    for day in days:
        restored = restore_day(symbol_directory, date(2026, 9, day))
        save_restored_day(restored, days_directory / f"2026-09-{day}.npz")
        loaded = load_restored_day(days_directory / f"2026-09-{day}.npz")
        arrays = {**vars(restored.top), **vars(restored.valid), "snapshot_rows": restored.snapshot_rows}
        loaded_arrays = {**vars(loaded.top), **vars(loaded.valid), "snapshot_rows": loaded.snapshot_rows}
        for name, array in arrays.items():
            assert np.array_equal(loaded_arrays[name], array, equal_nan=True), name
            assert loaded_arrays[name].dtype == array.dtype, name
        assert (loaded.start, loaded.end, loaded.depth20, loaded.snapshots) == (
            restored.start, restored.end, restored.depth20, restored.snapshots
        )


def test_saved_days_give_what_one_replay_gives_for_both_days_and_for_a_window_cut_inside_each_day(tmp_path):
    symbol_directory, depth = store_across_midnight(tmp_path / "raw")
    days_directory = tmp_path / "restored" / "BTCUSDT"
    save_days(symbol_directory, days_directory, [27, 28])
    replay = BookReplay(symbol_directory, utc_day(27), utc_day(29))
    both = mark_valid(collect_top_levels(replay), replay)
    # 창은 27일 스냅샷을 끼운 151번 프레임부터 28일 스냅샷을 끼운 451번 프레임까지다. 끼운 행이 창 첫 행과 끝 행이라
    # 스냅샷을 한 행만 밀려 골라도 빠진다. 날마다 바꾼 depth20 하나와 28일 스냅샷의 틀린 수량이 창 안이다
    windows = [
        (milliseconds(utc_day(27)), milliseconds(utc_day(29))),
        (delta(depth[151])["E"], delta(depth[452])["E"]),
    ]

    for start, end in windows:
        summary = summarize_window(days_directory, start, end, symbol_directory)

        replay = BookReplay(symbol_directory, utc(start), utc(end))
        top = collect_top_levels(replay)
        assert summary.rows == len(top.exchange_time)
        assert summary.depth20 == compare_depth20(top, mark_valid(top, replay), symbol_directory)
        assert summary.depth20.whole.mismatched_points == 2
        # REST 대조는 재생을 한 번 더 돈다. collect_top_levels 가 steps() 를 다 쓴다
        assert summary.snapshots == compare_snapshots(BookReplay(symbol_directory, utc(start), utc(end)))
        assert summary.snapshots.inside_range.quantity_differences == 1
        # 수집 첫 프레임 앞과 끝 프레임 뒤는 모르는 시간이라 커버리지 창이 줄어든다. 창 끝 뒤 프레임은 두 날짜 재생에 있다
        edges = (max(start, both.exchange_time[0]), min(end, both.exchange_time[-1]))
        assert (summary.coverage_start, summary.coverage_end) == edges
        assert summary.coverage == measure_coverage(both, summary.coverage_start, summary.coverage_end)


def test_coverage_of_each_saved_day_adds_up_to_both_days_when_a_gap_crosses_midnight(tmp_path):
    depth, first = depth_lines(), json.loads(SNAPSHOT)
    # 100 ~ 102번 프레임을 빼고 103번부터 하루 뒤로 옮긴다. 300번 프레임 자리의 스냅샷으로 다시 동기화된다
    later = [one_day_later(line) for line in depth[103:]]
    bids, asks = advanced(first, depth[:301])
    # 갭 뒤 스냅샷은 양쪽 10레벨만 준다. 그 뒤 행은 동기화됐지만 20번째 레벨이 범위 밖이라 저장한 valid 두 컬럼이 갈린다
    best_ten_bids = {price: bids[price] for price in sorted(bids, key=float, reverse=True)[:10]}
    best_ten_asks = {price: asks[price] for price in sorted(asks, key=float)[:10]}
    second = as_snapshot(delta(depth[300])["u"], best_ten_bids, best_ten_asks)
    snapshots = [
        (received_at(depth[20]) + SECOND // 2, SNAPSHOT),
        (received_at(later[300 - 103]) + SECOND // 2, second),
    ]
    # depth20 도 갭에서 같이 비고, 갭 뒤 것은 수신 시각만 하루 뒤로 간다. depth20 에는 거래소 시각이 없다
    resumed_from = delta(depth[103])["U"]
    depth20 = [
        one_day_later(line) if json.loads(line)["m"]["data"]["lastUpdateId"] >= resumed_from else line
        for line in fixture_depth20_lines()
        if not delta(depth[99])["u"] < json.loads(line)["m"]["data"]["lastUpdateId"] < resumed_from
    ]
    symbol_directory = store_with_depth20(tmp_path / "raw", depth[:100] + later, depth20, snapshots)
    days_directory = tmp_path / "restored" / "BTCUSDT"
    save_days(symbol_directory, days_directory, [27, 28])
    midnight = milliseconds(utc_day(28))

    day27 = summarize_window(days_directory, milliseconds(utc_day(27)), midnight, symbol_directory)
    day28 = summarize_window(days_directory, midnight, milliseconds(utc_day(29)), symbol_directory)
    both = summarize_window(days_directory, milliseconds(utc_day(27)), milliseconds(utc_day(29)), symbol_directory)

    # 27일 끝 조각은 28일 첫 행을 봐야 갭인 줄 안다
    assert day27.coverage.gap_milliseconds == midnight - delta(depth[99])["E"]
    assert day28.coverage.gap_milliseconds == delta(later[0])["E"] - midnight
    assert day27.coverage + day28.coverage == both.coverage
    assert both.coverage.outside_range_milliseconds > 0
    replay = BookReplay(symbol_directory, utc_day(27), utc_day(29))
    replayed = mark_valid(collect_top_levels(replay), replay)
    assert both.coverage == measure_coverage(replayed, both.coverage_start, both.coverage_end)
    gap = delta(later[0])["E"] - delta(depth[99])["E"]
    assert both.gaps == ((gap, delta(later[301 - 103])["E"] - delta(later[0])["E"]),)
    # 72시간 판정 창처럼 갭 뒤 첫 프레임에서 시작하는 창은 그 갭을 안 센다
    resumed_at = delta(later[0])["E"]
    assert summarize_window(days_directory, resumed_at, milliseconds(utc_day(29)), symbol_directory).gaps == ()
    # 27일 파일 없이 28일부터 재면 창 시작이 갭 뒤 첫 행으로 줄어 자정에 걸친 갭이 조용히 빠진다
    (days_directory / "2026-09-27.npz").unlink()
    with pytest.raises(FileNotFoundError):
        summarize_window(days_directory, midnight, milliseconds(utc_day(29)), symbol_directory)


def test_summary_stops_when_a_day_inside_the_window_was_not_restored(tmp_path):
    symbol_directory, _ = store_across_midnight(tmp_path / "raw")
    days_directory = tmp_path / "restored" / "BTCUSDT"
    save_days(symbol_directory, days_directory, [27])
    # 28일 파일 없이 재면 창 끝이 27일 끝 행으로 줄어 28일 하루가 조용히 빠진다
    with pytest.raises(FileNotFoundError):
        summarize_window(days_directory, milliseconds(utc_day(27)), milliseconds(utc_day(29)), symbol_directory)


@pytest.mark.parametrize(
    "days, output",
    [
        (["2026-09-27", "2026-09-27"], "restored/../raw/restored"),
        (["2026-09-27", "2026-09-27"], "raw"),
        (["2026-09-27", "today"], "restored"),
    ],
    ids=["under_the_raw_folder", "the_raw_folder_itself", "ending_on_a_day_still_being_collected"],
)
def test_restore_script_refuses_to_write_into_the_raw_folder_or_a_day_still_being_collected(
    tmp_path, monkeypatch, days, output
):
    (tmp_path / "raw").mkdir()
    today = str(datetime.now(timezone.utc).date())
    # 원본 폴더는 기본값처럼 상대 경로로 준다. 출력 경로를 돌려 적어도 원본 폴더 아래 어디든 막아야 한다
    monkeypatch.chdir(tmp_path)
    days = [today if day == "today" else day for day in days]
    monkeypatch.setattr(sys, "argv", [SCRIPT, "restore", *days, "--output", output, "--data-root", "raw"])

    with pytest.raises(SystemExit):
        runpy.run_path(SCRIPT, run_name="__main__")

    assert not any((tmp_path / "raw").iterdir()) and not (tmp_path / "restored").exists()