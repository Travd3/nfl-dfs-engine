"""
Which arm is actually better?

MAE and RMSE disagree here, and the disagreement is not noise. Fantasy points
are right-skewed, so the conditional median sits below the conditional mean.
A model fit to absolute error predicts the median and wins MAE by
systematically under-projecting. For DFS that is the wrong direction: lineup
value is expected points, and the players it shades down hardest are the
high-ceiling ones you build tournaments around.

So this reports, for each arm:
  RMSE bootstrap against naive, block-resampled by slate week
  mean bias, predicted minus actual
  bias in the top decile of actual outcomes, where ceilings live
"""
import numpy as np, polars as pl

R = pl.read_parquet("tables/baseline_v2.parquet")
ARMS = ["v1", "v2a", "v2b", "v2c", "v2d", "v2e"]


def boot_rmse(col, ref="naive", B=1500, seed=23):
    rng = np.random.default_rng(seed)
    d = R.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk")).drop_nulls(["y", col, ref])
    blks = d["blk"].unique().to_list()
    by = {k: d.filter(pl.col("blk") == k) for k in blks}
    o = []
    for _ in range(B):
        s = pl.concat([by[k] for k in rng.choice(blks, len(blks), replace=True)])
        r = lambda c: float((((s["y"] - s[c]) ** 2).mean()) ** 0.5)
        o.append(r(ref) - r(col))
    o = np.array(o)
    return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o > 0).mean()


def main():
    print("=" * 76)
    print("RMSE AGAINST NAIVE  (positive = better; this is the DFS-relevant metric)")
    print("=" * 76)
    print(f"{'arm':6s} {'RMSE':>7s} {'delta':>9s} {'95% CI':>22s} {'P>0':>6s}")
    dn = R.drop_nulls(["y", "naive"])
    print(f"{'naive':6s} {float((((dn['y']-dn['naive'])**2).mean())**0.5):7.4f}")
    for c in ARMS:
        d = R.drop_nulls(["y", c])
        rm = float((((d["y"] - d[c]) ** 2).mean()) ** 0.5)
        mu, lo, hi, p = boot_rmse(c)
        print(f"{c:6s} {rm:7.4f} {mu:+9.4f}  [{lo:+.4f}, {hi:+.4f}] {p:6.3f}")

    print()
    print("=" * 76)
    print("BIAS  (predicted minus actual; negative = under-projecting)")
    print("=" * 76)
    cut = R["y"].quantile(0.90)
    print(f"{'arm':6s} {'overall':>9s} {'top decile of actual':>22s}")
    for c in ["naive"] + ARMS:
        d = R.drop_nulls(["y", c])
        overall = float((d[c] - d["y"]).mean())
        top = d.filter(pl.col("y") >= cut)
        tb = float((top[c] - top["y"]).mean())
        print(f"{c:6s} {overall:+9.3f} {tb:+22.3f}")

    print(f"\n  top decile threshold: {cut:.1f} points, n={len(R.filter(pl.col('y') >= cut))}")
    print(f"  mean actual: {float(R['y'].mean()):.3f}   median actual: {float(R['y'].median()):.3f}")


if __name__ == "__main__":
    main()
