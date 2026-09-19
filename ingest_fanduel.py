"""
Parse a FanDuel NFL entries upload template into a clean player-pool table.

FanDuel's template is not a normal one-header CSV. The first section contains
entry metadata/instructions and a second header row introduces the player pool.
This parser deliberately ignores entry_id / contest_id when exporting the pool
so a public repository or downstream artifact does not leak a user's entry data.

Usage:
    python ingest_fanduel.py path/to/FanDuel-...-entries-upload-template.csv

Output:
    tables/fanduel_player_pool.parquet
    tables/fanduel_player_pool.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import polars as pl

PLAYER_HEADER = "Player ID + Player Name"
KEEP = [
    "Player ID + Player Name",
    "Id",
    "Position",
    "First Name",
    "Nickname",
    "Last Name",
    "FPPG",
    "Played",
    "Salary",
    "Game",
    "Team",
    "Opponent",
    "Injury Indicator",
    "Injury Details",
    "Roster Position",
]


def parse_fanduel_template(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header_row = None
    start_col = None

    for i, row in enumerate(rows):
        if PLAYER_HEADER in row:
            header_row = i
            start_col = row.index(PLAYER_HEADER)
            break

    if header_row is None or start_col is None:
        raise ValueError(
            "Could not find the FanDuel player-pool header. "
            "Expected a column named 'Player ID + Player Name'."
        )

    headers = rows[header_row][start_col:]
    records = []

    for row in rows[header_row + 1 :]:
        if len(row) <= start_col:
            continue
        vals = row[start_col:]
        if len(vals) < len(headers):
            vals = vals + [""] * (len(headers) - len(vals))
        rec = dict(zip(headers, vals))
        if rec.get("Id") and rec.get("Position"):
            records.append({k: rec.get(k, "") for k in KEEP})

    if not records:
        raise ValueError("FanDuel template contained no player-pool rows.")

    df = pl.DataFrame(records).with_columns([
        pl.col("Salary").cast(pl.Int64, strict=False),
        pl.col("FPPG").cast(pl.Float64, strict=False),
        pl.col("Played").cast(pl.Int64, strict=False),
    ])

    return df.sort(["Position", "Salary"], descending=[False, True])


def main(path: str) -> None:
    os.makedirs("tables", exist_ok=True)
    df = parse_fanduel_template(path)

    df.write_parquet("tables/fanduel_player_pool.parquet")
    df.write_csv("tables/fanduel_player_pool.csv")

    print(f"players: {len(df)}")
    print(f"games:   {df['Game'].n_unique()}")
    print(df.group_by("Position").agg([
        pl.len().alias("players"),
        pl.col("Salary").min().alias("min_salary"),
        pl.col("Salary").max().alias("max_salary"),
    ]).sort("Position"))
    print(f"injury-flagged rows: {len(df.filter(pl.col('Injury Indicator') != ''))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python ingest_fanduel.py "
            "FanDuel-NFL-YYYY-MM-DD-...-entries-upload-template.csv"
        )
    main(sys.argv[1])
