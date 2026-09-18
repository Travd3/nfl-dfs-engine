"""
Stage 4: does scheme earn projection weight?

Bootstrap resamples whole slate-weeks, not individual rows. Player-weeks
inside one week are correlated, so a row-level bootstrap would report a
confidence interval several times too narrow.
"""
import numpy as np, polars as pl
from scipy import stats

R = pl.read_parquet("tables/predictions.parquet")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    return dict(n=len(y), MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                spearman=stats.spearmanr(y, p).statistic,
                r2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def block_bootstrap(df, a="base", b="scheme", B=2000, seed=11):
    """Paired difference in MAE, resampling (season, week) blocks."""
    rng = np.random.default_rng(seed)
    df = df.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk"))
    blocks = df["blk"].unique().to_list()
    by = {k: v for k, v in zip(blocks, [df.filter(pl.col("blk") == k) for k in blocks])}
    diffs = []
    for _ in range(B):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        s = pl.concat([by[k] for k in pick])
        y = s["y"].to_numpy(); pa = s[a].to_numpy(); pb = s[b].to_numpy()
        diffs.append(np.mean(np.abs(y - pa)) - np.mean(np.abs(y - pb)))
    d = np.array(diffs)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def calibration(df, col, bins=10):
    d = df.drop_nulls([col, "y"])
    q = d.with_columns(pl.col(col).qcut(bins, labels=[str(i) for i in range(bins)]).alias("b"))
    t = q.group_by("b").agg([pl.col(col).mean().alias("pred"),
                             pl.col("y").mean().alias("actual"),
                             pl.len().alias("n")]).sort("pred")
    sl = stats.linregress(t["pred"].to_numpy(), t["actual"].to_numpy())
    return t, sl


def report():
    print("=" * 66)
    print("1 & 2. HELD-OUT ACCURACY  (walk-forward: train < season, test = season)")
    print("=" * 66)
    print(f"{'model':10s} {'n':>6s} {'MAE':>7s} {'RMSE':>7s} {'Spearman':>9s} {'R2':>7s}")
    for lab, col in [("naive", "naive"), ("baseline", "base"), ("+scheme", "scheme")]:
        m = metrics(R["y"], R[col])
        print(f"{lab:10s} {m['n']:6d} {m['MAE']:7.3f} {m['RMSE']:7.3f} "
              f"{m['spearman']:9.4f} {m['r2']:7.4f}")

    print()
    print("=" * 66)
    print("3. BOOTSTRAP CI FOR SCHEME IMPROVEMENT  (positive = scheme better)")
    print("=" * 66)
    mu, lo, hi, pw = block_bootstrap(R)
    print(f"  MAE improvement: {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(better)={pw:.3f}")
    print(f"  as a share of baseline MAE: {mu / metrics(R['y'], R['base'])['MAE'] * 100:+.2f}%")

    print()
    print("=" * 66)
    print("4. CALIBRATION  (actual = a + b * predicted; b=1, a=0 is perfect)")
    print("=" * 66)
    for lab, col in [("baseline", "base"), ("+scheme", "scheme")]:
        t, sl = calibration(R, col)
        print(f"  {lab:9s} slope={sl.slope:6.3f}  intercept={sl.intercept:+6.3f}  "
              f"r={sl.rvalue:.4f}")
    t, _ = calibration(R, "base")
    print("\n  baseline decile table")
    print(f"  {'pred':>7s} {'actual':>7s} {'n':>6s}")
    for r in t.iter_rows(named=True):
        print(f"  {r['pred']:7.2f} {r['actual']:7.2f} {r['n']:6d}")

    print()
    print("=" * 66)
    print("5. BY CHANNEL")
    print("=" * 66)
    print(f"{'channel':10s} {'n':>6s} {'base MAE':>9s} {'schm MAE':>9s} {'delta':>8s} {'CI':>22s}")
    for ch in R["channel"].unique().sort().to_list():
        s = R.filter(pl.col("channel") == ch)
        mb, ms = metrics(s["y"], s["base"]), metrics(s["y"], s["scheme"])
        mu, lo, hi, _ = block_bootstrap(s, B=600)
        print(f"{ch:10s} {mb['n']:6d} {mb['MAE']:9.3f} {ms['MAE']:9.3f} "
              f"{mu:+8.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print()
    print("=" * 66)
    print("6. EARLY VS LATE SEASON")
    print("=" * 66)
    for lab, f in [("weeks 1-6", pl.col("week") <= 6),
                   ("weeks 7-11", (pl.col("week") > 6) & (pl.col("week") <= 11)),
                   ("weeks 12+", pl.col("week") > 11)]:
        s = R.filter(f)
        mb, ms = metrics(s["y"], s["base"]), metrics(s["y"], s["scheme"])
        mu, lo, hi, _ = block_bootstrap(s, B=600)
        print(f"{lab:11s} n={mb['n']:5d}  base {mb['MAE']:.3f}  scheme {ms['MAE']:.3f}  "
              f"delta {mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    report()
