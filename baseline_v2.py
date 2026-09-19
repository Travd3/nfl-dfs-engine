"""
Baseline V2. Assigned task: beat naive persistence, or show it cannot be done
with what is available.

Arms, each a single change on the one before it so the source of any gain is
identifiable:

  naive   carry forward the player's exponentially weighted points
  v1      role x efficiency, ridge, unweighted            (the current baseline)
  v2a     v1 with the efficiency regression weighted by opportunities
  v2b     v2a plus snap share, red zone role, pace, rest, venue
  v2c     one direct model on points, same features
  v2d     direct model, gradient boosting, squared error
  v2e     direct model, gradient boosting, absolute error

Two notes on method.

Efficiency weighting: the target is points/opportunities. A one-target game
where the ball found the end zone reads as 7.0 points per opportunity and an
unweighted regression treats it as equal evidence to a 30-carry game. Weighting
by opportunities is the fix, not a tweak.

Boosting iterations are chosen on a temporal inner split, the last training
season, never a random split. A random validation fold would let the model
tune against the future.
"""
import numpy as np, polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES_ROLE, FEATURES_EFF
from features_v2 import NEW_ROLE, NEW_EFF
from model import prep, CHANNELS, TEST_SEASONS, MIN_HIST

ALPHAS = np.logspace(-2, 4, 25)
ROLE2 = FEATURES_ROLE + NEW_ROLE
EFF2 = FEATURES_EFF + NEW_EFF
DIRECT = sorted(set(ROLE2 + EFF2))


def ridge(Xtr, ytr, w=None):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr, sample_weight=w)
    return lambda X: m.predict(sc.transform(X))


def boost(Xtr, ytr, seasons_tr, loss="squared_error"):
    """Iteration count picked on the latest training season, held out in time."""
    last = seasons_tr.max()
    va = seasons_tr == last
    if va.sum() < 200 or (~va).sum() < 400:
        va = np.zeros(len(ytr), bool); va[int(len(ytr) * 0.8):] = True
    best_n, best_err = 60, None
    for n in (60, 120, 200, 320):
        m = HistGradientBoostingRegressor(
            loss=loss, max_iter=n, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        m.fit(Xtr[~va], ytr[~va])
        e = np.mean(np.abs(ytr[va] - m.predict(Xtr[va])))
        if best_err is None or e < best_err:
            best_err, best_n = e, n
    m = HistGradientBoostingRegressor(
        loss=loss, max_iter=best_n, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xtr, ytr)
    return m.predict


def run():
    frame = pl.read_parquet("tables/frame_v2.parquet").filter(pl.col("p_ngames") >= MIN_HIST)
    arms = ["naive", "v1", "v2a", "v2b", "v2c", "v2d", "v2e"]
    out = []

    for test_season in TEST_SEASONS:
        tr = frame.filter(pl.col("season") < test_season)
        te = frame.filter(pl.col("season") == test_season)

        for ch in CHANNELS:
            a = tr.filter(pl.col("channel") == ch)
            b = te.filter(pl.col("channel") == ch)
            if len(a) < 400 or len(b) < 100:
                continue

            y_opps = a["y_opps"].to_numpy().astype(float)
            y_ppo = a["y_ppo"].to_numpy().astype(float)
            y_pts = a["y_pts"].to_numpy().astype(float)
            seasons_tr = a["season"].to_numpy()
            w = y_opps.copy()

            def mat(cols, df_a, df_b):
                A, med = prep(df_a.select(cols).to_numpy())
                B, _ = prep(df_b.select(cols).to_numpy(), med)
                return A, B

            # ---- v1: original role x efficiency
            R1a, R1b = mat(FEATURES_ROLE, a, b)
            E1a, E1b = mat(FEATURES_EFF, a, b)
            opps1 = np.clip(ridge(R1a, y_opps)(R1b), 0, None)
            v1 = opps1 * ridge(E1a, y_ppo)(E1b)

            # ---- v2a: weight efficiency by opportunities
            v2a = opps1 * ridge(E1a, y_ppo, w=w)(E1b)

            # ---- v2b: add the new role features to both halves
            R2a, R2b = mat(ROLE2, a, b)
            E2a, E2b = mat(EFF2, a, b)
            opps2 = np.clip(ridge(R2a, y_opps)(R2b), 0, None)
            v2b = opps2 * ridge(E2a, y_ppo, w=w)(E2b)

            # ---- direct models on points
            Da, Db = mat(DIRECT, a, b)
            v2c = np.clip(ridge(Da, y_pts)(Db), 0, None)
            v2d = np.clip(boost(Da, y_pts, seasons_tr, "squared_error")(Db), 0, None)
            v2e = np.clip(boost(Da, y_pts, seasons_tr, "absolute_error")(Db), 0, None)

            out.append(pl.DataFrame({
                "season": [test_season] * len(b), "week": b["week"],
                "channel": [ch] * len(b), "pos": b["pos"],
                "y": b["y_pts"].cast(float),
                "naive": b["pts__ewm"].cast(float),
                "v1": v1, "v2a": v2a, "v2b": v2b,
                "v2c": v2c, "v2d": v2d, "v2e": v2e}))

    R = pl.concat(out)
    R.write_parquet("tables/baseline_v2.parquet")
    report(R, arms)
    return R


def block_boot(R, col, ref="naive", B=1500, seed=17):
    rng = np.random.default_rng(seed)
    d = R.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk")).drop_nulls([col, ref])
    blks = d["blk"].unique().to_list()
    by = {k: d.filter(pl.col("blk") == k) for k in blks}
    o = []
    for _ in range(B):
        s = pl.concat([by[k] for k in rng.choice(blks, len(blks), replace=True)])
        o.append(float((s["y"] - s[ref]).abs().mean()) - float((s["y"] - s[col]).abs().mean()))
    o = np.array(o)
    return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o > 0).mean()


def report(R, arms):
    def mae(c):
        d = R.drop_nulls(["y", c]); return float((d["y"] - d[c]).abs().mean())
    def rmse(c):
        d = R.drop_nulls(["y", c]); return float((((d["y"] - d[c]) ** 2).mean()) ** 0.5)

    print("=" * 78)
    print("BASELINE V2, walk-forward. Deltas are vs NAIVE. Positive = better.")
    print("=" * 78)
    print(f"{'arm':6s} {'MAE':>7s} {'RMSE':>7s} {'MAE delta':>10s} {'95% CI':>22s} {'P>0':>6s}")
    for c in arms:
        if c == "naive":
            print(f"{c:6s} {mae(c):7.4f} {rmse(c):7.4f}")
            continue
        mu, lo, hi, p = block_boot(R, c)
        print(f"{c:6s} {mae(c):7.4f} {rmse(c):7.4f} {mu:+10.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}] {p:6.3f}")

    best = min([c for c in arms if c != "naive"], key=mae)
    print(f"\nbest by MAE: {best}")
    print("\nby channel (MAE)")
    hdr = f"{'channel':10s}" + "".join(f"{c:>9s}" for c in arms)
    print(hdr)
    for ch in R["channel"].unique().sort().to_list():
        s = R.filter(pl.col("channel") == ch)
        row = f"{ch:10s}"
        for c in arms:
            d = s.drop_nulls(["y", c])
            row += f"{float((d['y'] - d[c]).abs().mean()):9.3f}"
        print(row)


if __name__ == "__main__":
    run()
