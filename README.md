# Tickfold

Storage size vs. deep-learning accuracy trade-offs for reduced limit order book (LOB) data.

**Status:** early development. The collector is being built first; replay, reduction transforms, and the model pipeline follow. See `docs/build-plan-v2.md` for the build order.

## What this is

Exchanges publish order book updates as delta streams: each message lists the new total quantity at a handful of price levels, not the full book. Storing years of these streams means choosing how much detail to keep. Tickfold measures what each of those choices costs, in bytes and in prediction accuracy.

The project has three parts:

1. **A collector and replay pipeline** for Binance spot depth streams, with sequence validation, gap resynchronisation, and byte-exact storage of the original payloads.
2. **Reduction transforms** along four axes: book depth, quantity precision, price precision (tick coarsening), and time resolution.
3. **An experiment harness** that trains DeepLOB on each reduced variant under identical settings and plots the Pareto frontier of storage size against accuracy and macro-F1.

## Layout

```
tickfold/           Python package
  collector/        Binance depth collector with sequence validation
docs/               build plan and analysis notes
scripts/            fixture capture
tests/              pytest suite (no network required)
```

Additional packages (`replay/`, `reduce/`, `model/`, `experiments/`, `deploy/`) will be added as each stage begins; names are settled when the stage starts.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Run the collector locally (entry point lands with plan section 1, `tickfold/collector/main.py`):

```bash
uv run python -m tickfold.collector.main
```

Configuration will be via environment variables once that entry point exists; the list will live in `docs/build-plan-v2.md`.

## Data

Raw market data is collected from Binance's public spot streams and is **not** redistributed in this repository. The public release ships with derived statistics and, where terms permit, a small sample; otherwise a synthetic generator calibrated to the published statistics is provided.

Data quality is reported as measured mismatch rates and coverage, not as a claim of zero error. Reconstructed books are checked against independent exchange channels (the partial depth stream, the book ticker stream, and periodic REST snapshots); the method is described in `docs/analysis-2026-09-15-v2-transition.md`.

## Documentation

- `docs/build-plan-v2.md` — live plan: schedule, sections 1 through 7
- `docs/build-plan-v1.md` — archived v1 plan (storage format and baseline benchmark, on hold)
- `docs/analysis-2026-09-15-v2-transition.md` — v1 to v2 transition analysis, measurements, open decisions

## License

MIT — see `LICENSE`.

## Citation

A paper on storage-size versus DeepLOB-accuracy trade-offs for reduced LOB data is in preparation. A citation entry will be added here when the paper is available.
