# Tickfold

A storage format and public benchmark for limit order book (LOB) delta streams.

**Status:** early development. The collector is being built first; the format, baselines, and benchmark harness will follow. See `docs/build-plan.md` for the build order.

## What this is

Exchanges publish order book updates as delta streams: each message lists the new total quantity at a handful of price levels, not the full book. Tickfold addresses the problem of storing years of these streams compactly while keeping any point-in-time book quick to reconstruct.

The project has three parts:

1. **Domain transforms** that exploit LOB structure: tick-size integerization, column separation with delta / delta-of-delta encoding, level indexing relative to the best price, and separation of structure and quantity streams.
2. **A design-space benchmark** that decomposes storage gains along three axes (transform, columnar layout, checkpoint policy) rather than reporting a single winner.
3. **A reproducible harness** comparing the proposed format against Parquet, DBN, DuckDB, ClickHouse, and ArcticDB on size, write throughput, and four query workloads.

## Layout

```
tickfold/           Python package
  collector/        Binance depth/trade collector with sequence validation
docs/               build plan and notes
tests/              pytest suite (no network required)
```

Additional packages (`format/`, `baselines/`, `bench/`, `deploy/`) will be added as each stage begins.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Run the collector locally:

```bash
uv run python -m tickfold.collector.main
```

Configuration is via environment variables; see `docs/build-plan.md` for the list.

## Data

Raw market data is collected from Binance's public spot streams and is **not** redistributed in this repository. The benchmark ships with derived statistics and, where terms permit, a small sample; otherwise a synthetic generator calibrated to the published statistics is provided.

## Documentation

- `docs/build-plan.md` — components, build order, and design notes

## License

TBD.

## Citation

A paper describing Tickfold is in preparation. A citation entry will be added here when the paper is available.