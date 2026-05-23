"""
Step 1 of the rebuild: convert raw Kaggle CSVs to typed Parquet.

Reads:
  - market_train_full.csv          (~1.1 GB,  ~4.07M rows, 17 cols inc. unnamed idx)
  - news_train_from2013.csv        (~4.7 GB,  ~9.3M rows, 36 cols inc. unnamed idx)

Writes:
  - market.parquet
  - news.parquet

Design notes (per CLAUDE.md):
  - Explicit dtypes via pyarrow.csv ConvertOptions; type failures raise loudly.
  - Streaming reader -> ParquetWriter so peak RAM stays bounded.
  - No row drops, no imputation here. Step 1 preserves raw fidelity.
  - Any unexpected condition (row count mismatch, unexpected time range, dtype
    surprise) is reported with a [WARN] line and counted in `warnings`.
  - `universe` left as float64 because NaN means "outside training universe"
    and pyarrow ints cannot represent NaN. Casting choice is the only data-shaping
    decision in this script and is documented here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as papq


ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

MARKET_COLUMNS = [
    "_idx",            # unnamed pandas-index column in source; dropped after read
    "time",
    "assetCode",
    "assetName",
    "volume",
    "close",
    "open",
    "returnsClosePrevRaw1",
    "returnsOpenPrevRaw1",
    "returnsClosePrevMktres1",
    "returnsOpenPrevMktres1",
    "returnsClosePrevRaw10",
    "returnsOpenPrevRaw10",
    "returnsClosePrevMktres10",
    "returnsOpenPrevMktres10",
    "returnsOpenNextMktres10",
    "universe",
]

MARKET_TYPES = {
    "_idx": pa.int64(),
    "time": pa.timestamp("ns", tz="UTC"),
    "assetCode": pa.string(),
    "assetName": pa.string(),
    "volume": pa.float64(),
    "close": pa.float64(),
    "open": pa.float64(),
    "returnsClosePrevRaw1": pa.float64(),
    "returnsOpenPrevRaw1": pa.float64(),
    "returnsClosePrevMktres1": pa.float64(),
    "returnsOpenPrevMktres1": pa.float64(),
    "returnsClosePrevRaw10": pa.float64(),
    "returnsOpenPrevRaw10": pa.float64(),
    "returnsClosePrevMktres10": pa.float64(),
    "returnsOpenPrevMktres10": pa.float64(),
    "returnsOpenNextMktres10": pa.float64(),
    "universe": pa.float64(),
}

NEWS_COLUMNS = [
    "_idx",
    "time",
    "sourceTimestamp",
    "firstCreated",
    "sourceId",
    "headline",
    "urgency",
    "takeSequence",
    "provider",
    "subjects",
    "audiences",
    "bodySize",
    "companyCount",
    "headlineTag",
    "marketCommentary",
    "sentenceCount",
    "wordCount",
    "assetCodes",
    "assetName",
    "firstMentionSentence",
    "relevance",
    "sentimentClass",
    "sentimentNegative",
    "sentimentNeutral",
    "sentimentPositive",
    "sentimentWordCount",
    "noveltyCount12H",
    "noveltyCount24H",
    "noveltyCount3D",
    "noveltyCount5D",
    "noveltyCount7D",
    "volumeCounts12H",
    "volumeCounts24H",
    "volumeCounts3D",
    "volumeCounts5D",
    "volumeCounts7D",
]

NEWS_TYPES = {
    "_idx": pa.int64(),
    "time": pa.timestamp("ns", tz="UTC"),
    "sourceTimestamp": pa.timestamp("ns", tz="UTC"),
    "firstCreated": pa.timestamp("ns", tz="UTC"),
    "sourceId": pa.string(),
    "headline": pa.string(),
    "urgency": pa.int8(),
    "takeSequence": pa.int16(),
    "provider": pa.string(),
    "subjects": pa.string(),
    "audiences": pa.string(),
    "bodySize": pa.int32(),
    "companyCount": pa.int16(),
    "headlineTag": pa.string(),
    "marketCommentary": pa.bool_(),
    "sentenceCount": pa.int16(),
    "wordCount": pa.int32(),
    "assetCodes": pa.string(),
    "assetName": pa.string(),
    "firstMentionSentence": pa.int16(),
    "relevance": pa.float32(),
    "sentimentClass": pa.int8(),
    "sentimentNegative": pa.float32(),
    "sentimentNeutral": pa.float32(),
    "sentimentPositive": pa.float32(),
    "sentimentWordCount": pa.int32(),
    "noveltyCount12H": pa.int16(),
    "noveltyCount24H": pa.int16(),
    "noveltyCount3D": pa.int16(),
    "noveltyCount5D": pa.int16(),
    "noveltyCount7D": pa.int16(),
    "volumeCounts12H": pa.int16(),
    "volumeCounts24H": pa.int16(),
    "volumeCounts3D": pa.int16(),
    "volumeCounts5D": pa.int16(),
    "volumeCounts7D": pa.int16(),
}


# -----------------------------------------------------------------------------
# Conversion driver
# -----------------------------------------------------------------------------

def warn(context: str, expected, actual, fallback="<<none>>"):
    print(
        f"[WARN] {context}: expected={expected}, got={actual}, fallback={fallback}",
        flush=True,
    )


def count_csv_rows(path: Path) -> int:
    """Cheap row count by reading raw bytes line by line. Used as ground truth."""
    print(f"  counting raw rows in {path.name} ...", flush=True)
    t0 = time.time()
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    n -= 1  # subtract header
    print(f"  raw row count = {n:,}  (took {time.time() - t0:.1f}s)", flush=True)
    return n


def convert(
    csv_path: Path,
    parquet_path: Path,
    column_names: list[str],
    column_types: dict[str, pa.DataType],
    expected_min_year: int,
    expected_max_year: int,
) -> dict:
    print(f"\n=== Converting {csv_path.name} -> {parquet_path.name} ===", flush=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    expected_rows = count_csv_rows(csv_path)

    read_options = pacsv.ReadOptions(
        skip_rows=1,                  # skip original header; we name cols explicitly
        column_names=column_names,
        block_size=64 * 1024 * 1024,  # 64 MB CSV-read chunks
    )
    parse_options = pacsv.ParseOptions(delimiter=",", quote_char='"')
    convert_options = pacsv.ConvertOptions(
        column_types=column_types,
        null_values=["", "NA", "NaN", "nan"],
        true_values=["True", "true"],
        false_values=["False", "false"],
        strings_can_be_null=True,
    )

    t0 = time.time()
    rows_written = 0
    writer: papq.ParquetWriter | None = None
    min_ts: pa.scalar | None = None
    max_ts: pa.scalar | None = None

    try:
        reader = pacsv.open_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

        for batch in reader:
            # Drop the unnamed index column so the Parquet has clean schema.
            if "_idx" in batch.schema.names:
                batch = batch.drop_columns(["_idx"])

            if writer is None:
                writer = papq.ParquetWriter(
                    parquet_path,
                    batch.schema,
                    compression="snappy",
                )

            writer.write_batch(batch)
            rows_written += batch.num_rows

            ts_min = pc.min(batch["time"]).as_py()
            ts_max = pc.max(batch["time"]).as_py()
            if min_ts is None or (ts_min is not None and ts_min < min_ts):
                min_ts = ts_min
            if max_ts is None or (ts_max is not None and ts_max > max_ts):
                max_ts = ts_max

            if rows_written % 1_000_000 < batch.num_rows:
                elapsed = time.time() - t0
                rate = rows_written / max(elapsed, 1e-6)
                print(
                    f"  ... {rows_written:>12,} rows  "
                    f"({elapsed:5.1f}s, {rate:,.0f} rows/s)",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  ({rows_written:,} rows written)", flush=True)

    # ---- Verification ----
    warnings = 0

    if rows_written != expected_rows:
        warn(
            "row_count_after_write",
            expected=expected_rows,
            actual=rows_written,
            fallback="continued (data may be corrupt; investigate)",
        )
        warnings += 1

    # Roundtrip read to confirm Parquet is readable end-to-end
    print(f"  roundtrip-reading {parquet_path.name} ...", flush=True)
    t0 = time.time()
    pf = papq.ParquetFile(parquet_path)
    rt_rows = pf.metadata.num_rows
    rt_cols = pf.schema_arrow.names
    print(
        f"  roundtrip ok: {rt_rows:,} rows, {len(rt_cols)} cols, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    if rt_rows != rows_written:
        warn(
            "roundtrip_row_count",
            expected=rows_written,
            actual=rt_rows,
            fallback="continued (Parquet write may be broken)",
        )
        warnings += 1

    expected_cols = [c for c in column_names if c != "_idx"]
    if set(rt_cols) != set(expected_cols):
        warn(
            "roundtrip_columns",
            expected=expected_cols,
            actual=rt_cols,
            fallback="continued",
        )
        warnings += 1

    # Time-range sanity check
    if min_ts is None or max_ts is None:
        warn("time_range", expected="non-null", actual="None", fallback="continued")
        warnings += 1
    else:
        print(f"  time range: {min_ts} -> {max_ts}", flush=True)
        if min_ts.year != expected_min_year:
            warn(
                "time_range_min_year",
                expected=expected_min_year,
                actual=min_ts.year,
                fallback="continued (range mismatch may be benign; verify)",
            )
            warnings += 1
        if max_ts.year != expected_max_year:
            warn(
                "time_range_max_year",
                expected=expected_max_year,
                actual=max_ts.year,
                fallback="continued",
            )
            warnings += 1

    out_bytes = parquet_path.stat().st_size
    in_bytes = csv_path.stat().st_size
    print(
        f"  size: {in_bytes / 1e9:.2f} GB CSV -> "
        f"{out_bytes / 1e9:.2f} GB Parquet "
        f"({100 * out_bytes / in_bytes:.1f}%)",
        flush=True,
    )

    return {
        "csv_rows": expected_rows,
        "parquet_rows": rt_rows,
        "warnings": warnings,
        "time_min": min_ts,
        "time_max": max_ts,
    }


def main():
    market_stats = convert(
        csv_path=ROOT / "market_train_full.csv",
        parquet_path=ROOT / "market.parquet",
        column_names=MARKET_COLUMNS,
        column_types=MARKET_TYPES,
        expected_min_year=2007,
        expected_max_year=2016,
    )

    news_stats = convert(
        csv_path=ROOT / "news_train_from2013.csv",
        parquet_path=ROOT / "news.parquet",
        column_names=NEWS_COLUMNS,
        column_types=NEWS_TYPES,
        expected_min_year=2007,
        expected_max_year=2016,
    )

    print("\n=== Summary ===")
    for name, s in [("market", market_stats), ("news", news_stats)]:
        print(
            f"  {name:6s}: csv_rows={s['csv_rows']:,} "
            f"parquet_rows={s['parquet_rows']:,} "
            f"range={s['time_min']} -> {s['time_max']} "
            f"warnings={s['warnings']}"
        )

    total_warnings = market_stats["warnings"] + news_stats["warnings"]
    if total_warnings > 0:
        print(f"\n[WARN] total warnings across step 1: {total_warnings}")
        sys.exit(1)
    print("\nStep 1 OK.")


if __name__ == "__main__":
    main()
