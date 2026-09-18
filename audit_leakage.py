"""
Leakage audit. Runs before any model is fit.

Each as-of join carried its source timestamp through. For a row played in
week w, the source row must be stamped no later than w - lag. Any violation
means the model can see the game it is predicting.
"""
import polars as pl

CHECKS = [
    ("player history",   "tindex_p",  1),
    ("team volume",      "tindex_t",  1),
    ("opponent allowed", "tindex_d",  1),
    ("offense scheme",   "tindex_os", 2),
    ("defense scheme",   "tindex_ds", 2),
]

def audit(path="tables/frame.parquet"):
    f = pl.read_parquet(path)
    ok = True
    print(f"{'source':20s} {'rows':>7s} {'violations':>11s} {'max lead':>9s}")
    for name, col, lag in CHECKS:
        sub = f.filter(pl.col(col).is_not_null())
        # gap is measured in weeks; cross-season gaps are large and always legal
        same_season = sub.filter(pl.col(col) // 100 == pl.col("tindex") // 100)
        gap = (same_season["tindex"] - same_season[col])
        viol = (gap < lag).sum()
        worst = int(gap.min()) if len(gap) else None
        ok &= (viol == 0)
        print(f"{name:20s} {len(sub):7d} {viol:11d} {str(worst):>9s}")

    # cross-check: no post-game schedule column made it into the frame
    banned = [c for c in f.columns if c in
              ("total", "result", "home_score", "away_score", "away_qb_id", "home_qb_id")]
    print("\npost-game columns present in frame:", banned or "none")
    ok &= not banned

    # week 1 rows must have no same-season history
    w1 = f.filter(pl.col("week") == 1)
    same = w1.filter(pl.col("tindex_p") // 100 == pl.col("tindex") // 100)
    print(f"week-1 rows sourcing same-season player history: {len(same)} (must be 0)")
    ok &= (len(same) == 0)

    print("\nLEAKAGE AUDIT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if audit() else 1)
