"""
Fill an EXISTING FanDuel entries upload template from optimizer_output.json.

The raw FanDuel template contains user-specific entry/contest identifiers, so
it is never committed to this public repository. This utility receives that
original file locally, preserves the metadata exactly, and fills one existing
entry with a selected optimized lineup.

Example:
    python fanduel_export.py \
      --template FanDuel-NFL-...-entries-upload-template.csv \
      --optimizer-output tables/optimizer_output.json \
      --rank 1 \
      --out FanDuel-filled.csv
"""
from __future__ import annotations

import argparse, csv, json


def assign_slots(lineup: dict) -> dict[str, list[str]]:
    """Map one legal lineup to FanDuel's QB/RB/RB/WR/WR/WR/TE/FLEX/DEF columns."""
    by: dict[str, list[dict]] = {}
    for row in lineup["players"]:
        by.setdefault(row["position"], []).append(row)
    for rows in by.values():
        rows.sort(key=lambda r: float(r["projection"]), reverse=True)

    flex_pos = lineup["flex_position"]
    flex_player = by[flex_pos].pop()

    slots = {
        "QB": [by["QB"][0]["fanduel_id"]],
        "RB": [r["fanduel_id"] for r in by["RB"]],
        "WR": [r["fanduel_id"] for r in by["WR"]],
        "TE": [by["TE"][0]["fanduel_id"]],
        "FLEX": [flex_player["fanduel_id"]],
        "DEF": [by["DEF"][0]["fanduel_id"]],
    }
    expected = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DEF": 1}
    got = {k: len(v) for k, v in slots.items()}
    if got != expected:
        raise ValueError(f"lineup cannot fill FanDuel slots: {got}")
    flat = [x for vals in slots.values() for x in vals]
    if len(flat) != 9 or len(set(flat)) != 9:
        raise ValueError("lineup has duplicate or missing FanDuel player IDs")
    return slots


def fill_template(template_csv: str, lineup: dict, out_csv: str,
                  entry_number: int = 1) -> str:
    with open(template_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError("empty FanDuel template")

    header = rows[0]
    required = ["entry_id", "contest_id", "QB", "RB", "WR", "TE", "FLEX", "DEF"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"FanDuel template missing columns: {missing}")

    entry_rows = [
        i for i, row in enumerate(rows[1:], start=1)
        if len(row) > 1 and row[0].strip() and row[1].strip()
    ]
    if not entry_rows:
        raise ValueError("no existing FanDuel entry row found")
    if entry_number < 1 or entry_number > len(entry_rows):
        raise ValueError(f"entry_number {entry_number} outside 1..{len(entry_rows)}")

    ri = entry_rows[entry_number - 1]
    if len(rows[ri]) < len(header):
        rows[ri].extend([""] * (len(header) - len(rows[ri])))

    slots = assign_slots(lineup)
    used = {k: 0 for k in slots}
    for ci, col in enumerate(header):
        if col not in slots:
            continue
        vals = slots[col]
        j = used[col]
        if j >= len(vals):
            raise ValueError(f"too many {col} columns in FanDuel template")
        rows[ri][ci] = vals[j]
        used[col] += 1

    expected = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DEF": 1}
    if used != expected:
        raise ValueError(f"unexpected FanDuel roster columns: wrote {used}")

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--optimizer-output", default="tables/optimizer_output.json")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--entry-number", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    payload = json.load(open(a.optimizer_output, encoding="utf-8"))
    matches = [x for x in payload["lineups"] if int(x["rank"]) == a.rank]
    if not matches:
        raise SystemExit(f"lineup rank {a.rank} not found")
    fill_template(a.template, matches[0], a.out, a.entry_number)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
