"""
Stage 5: ablation.

The first run added 14 scheme columns to a ridge with a fixed penalty. That
confounds "scheme is uninformative" with "the model was under-regularized and
extra columns cost accuracy". This run fixes both problems:

  * the penalty is chosen by inner cross-validation on training data only,
    so a useless feature should be shrunk toward zero rather than hurt
  * a NOISE arm adds the same number of random columns, giving the amount of
    damage a definitionally worthless feature set does under this setup

Arms:
  baseline        Vegas + volume + opportunity + efficiency
  + team rates    baseline + offense/defense observed tendencies
  + sensitivity   baseline + the single partially pooled points-per-opp term
  + both
  + noise         baseline + 14 random columns (control)
"""
import numpy as np, polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy import stats

from features import FEATURES_ROLE, FEATURES_EFF, FEATURES_SCHEME_TEAM
from model import prep, fit_sensitivities, scheme_term, CHANNELS, TEST_SEASONS, MIN_HIST

ALPHAS = np.logspace(-2, 4, 25)
rng = np.random.default_rng(3)


def ridgecv(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return sc, m, m.alpha_


def run():
    frame = pl.read_parquet("tables/frame.parquet").filter(pl.col("p_ngames") >= MIN_HIST)
    opps = pl.read_parquet("tables/opportunities.parquet").with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("tindex"))

    arms = ["base", "team", "sens", "both", "noise"]
    rows, alphas = [], []

    for test_season in TEST_SEASONS:
        tr = frame.filter(pl.col("season") < test_season)
        te = frame.filter(pl.col("season") == test_season)
        opps_tr = opps.filter(pl.col("season") < test_season)

        for ch in CHANNELS:
            a = tr.filter(pl.col("channel") == ch)
            b = te.filter(pl.col("channel") == ch)
            if len(a) < 400 or len(b) < 100:
                continue

            Xr_tr, med_r = prep(a.select(FEATURES_ROLE).to_numpy())
            Xr_te, _ = prep(b.select(FEATURES_ROLE).to_numpy(), med_r)
            scr, mr, _ = ridgecv(Xr_tr, a["y_opps"].to_numpy().astype(float))
            opps_hat = np.clip(mr.predict(scr.transform(Xr_te)), 0, None)

            slopes, conds = fit_sensitivities(opps_tr, ch)
            s_tr, s_te = scheme_term(a, slopes, conds), scheme_term(b, slopes, conds)

            E_tr = a.select(FEATURES_EFF).to_numpy().astype(float)
            E_te = b.select(FEATURES_EFF).to_numpy().astype(float)
            T_tr = a.select(FEATURES_SCHEME_TEAM).to_numpy().astype(float)
            T_te = b.select(FEATURES_SCHEME_TEAM).to_numpy().astype(float)
            k = T_tr.shape[1] + 1
            N_tr = rng.normal(size=(len(a), k)); N_te = rng.normal(size=(len(b), k))

            blocks = {
                "base":  (E_tr, E_te),
                "team":  (np.hstack([E_tr, T_tr]), np.hstack([E_te, T_te])),
                "sens":  (np.hstack([E_tr, s_tr[:, None]]), np.hstack([E_te, s_te[:, None]])),
                "both":  (np.hstack([E_tr, T_tr, s_tr[:, None]]),
                          np.hstack([E_te, T_te, s_te[:, None]])),
                "noise": (np.hstack([E_tr, N_tr]), np.hstack([E_te, N_te])),
            }

            preds = {}
            for arm, (Xa, Xb) in blocks.items():
                Xa, med = prep(Xa); Xb, _ = prep(Xb, med)
                sc, m, al = ridgecv(Xa, a["y_ppo"].to_numpy().astype(float))
                preds[arm] = opps_hat * m.predict(sc.transform(Xb))
                alphas.append({"season": test_season, "channel": ch, "arm": arm, "alpha": al})

            rows.append(pl.DataFrame({
                "season": [test_season] * len(b), "week": b["week"],
                "channel": [ch] * len(b), "y": b["y_pts"].cast(float),
                "naive": b["pts__ewm"].cast(float),
                **{arm: preds[arm] for arm in arms}}))

    R = pl.concat(rows)
    R.write_parquet("tables/ablation.parquet")
    pl.DataFrame(alphas).write_parquet("tables/alphas.parquet")

    def mae(c):
        d = R.drop_nulls(["y", c]); return float((d["y"] - d[c]).abs().mean())

    def boot(c, B=1500, seed=5):
        r = np.random.default_rng(seed)
        d = R.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk"))
        blks = d["blk"].unique().to_list()
        by = {k: d.filter(pl.col("blk") == k) for k in blks}
        out = []
        for _ in range(B):
            s = pl.concat([by[k] for k in r.choice(blks, len(blks), replace=True)])
            out.append(float((s["y"] - s["base"]).abs().mean()) -
                       float((s["y"] - s[c]).abs().mean()))
        o = np.array(out)
        return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5)

    print("=" * 72)
    print("ABLATION, penalty tuned per arm by inner CV. Positive delta = better.")
    print("=" * 72)
    print(f"{'arm':14s} {'MAE':>7s} {'delta vs base':>14s} {'95% CI':>24s}")
    print(f"{'naive':14s} {mae('naive'):7.4f}")
    print(f"{'base':14s} {mae('base'):7.4f}")
    for arm in ["team", "sens", "both", "noise"]:
        mu, lo, hi = boot(arm)
        print(f"{arm:14s} {mae(arm):7.4f} {mu:+14.4f}   [{lo:+.4f}, {hi:+.4f}]")

    al = pl.read_parquet("tables/alphas.parquet")
    print("\nselected penalty by arm (median across channels/seasons)")
    print(al.group_by("arm").agg(pl.col("alpha").median().alias("median_alpha")).sort("arm"))


if __name__ == "__main__":
    run()
