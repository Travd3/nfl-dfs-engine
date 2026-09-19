"""
Issue #2: availability hurdle against frozen V3.

    E[points] = P(record an opportunity) x E[points | opportunity]

V3 is imported, never modified. The frame, the feature list, the
winsorization, the minimum-history filter, the rolling-origin splits and the
scoring profile all come from baseline_v3 unchanged, so the only difference
between arms is the estimator.

Three arms are scored on identical rows:

  naive     carry forward the player's exponentially weighted points
  v3        frozen direct ridge on points, the confirmed interim baseline
  hurdle    P(play) x E[points | play]

Both hurdle components are also reported alone, because the question in the
issue is specifically whether the conditional half fixes the played-row
weakness without surrendering the zero-row advantage. A product that ties
overall could still be hiding one half that works and one that does not.

Regularization for the classifier is chosen on a TEMPORAL inner split, the
last training season, never a random fold. That is stricter than the
leave-one-out generalized CV that frozen V3's RidgeCV performs, so the hurdle
arm cannot be accused of a looser search.

LEAKAGE NOTE: the row universe is active-roster players, inherited unchanged
from V3. If weekly roster status is recorded after games rather than before,
that encodes post-lock information. It is a shared assumption, so the
comparison between arms is unaffected, but P(play) leans on it more heavily
than a points model does and the concern is larger here. No injury or
inactive-list field is used by either arm.
"""
import numpy as np, polars as pl
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from baseline_v3 import (build_frame, prep, FEATURES, ALPHAS, FIRST_TEST,
                         RMSE, MAE, BIAS, spearman, calib, block_boot)

CS = np.logspace(-3, 3, 13)
POSITIONS = ["QB", "RB", "WR", "TE"]


def fit_classifier(A, y, seasons_tr):
    """P(opportunity). C picked on the latest training season."""
    last = seasons_tr.max()
    va = seasons_tr == last
    if va.sum() < 200 or (~va).sum() < 400:
        va = np.zeros(len(y), bool)
        va[int(len(y) * 0.8):] = True
    sc = StandardScaler().fit(A[~va])
    best_c, best_ll = CS[0], None
    for c in CS:
        m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(A[~va]), y[~va])
        p = np.clip(m.predict_proba(sc.transform(A[va]))[:, 1], 1e-6, 1 - 1e-6)
        ll = -np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
        if best_ll is None or ll < best_ll:
            best_ll, best_c = ll, c
    sc = StandardScaler().fit(A)
    m = LogisticRegression(C=best_c, max_iter=2000).fit(sc.transform(A), y)
    return lambda X: m.predict_proba(sc.transform(X))[:, 1]


def fit_ridge(A, y):
    sc = StandardScaler().fit(A)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), y)
    return lambda X: m.predict(sc.transform(X))


def run(profile_name="fanduel_test_slate_1", min_hist=3):
    f = build_frame(profile_name).filter(pl.col("n_games") >= min_hist)
    seasons = sorted(f["season"].unique().to_list())
    tests = [s for s in seasons if s >= FIRST_TEST]

    out = []
    for ts in tests:
        tr, te = f.filter(pl.col("season") < ts), f.filter(pl.col("season") == ts)
        if len(tr) < 5000 or len(te) < 500:
            continue

        v3 = np.zeros(len(te)); hurdle = np.zeros(len(te))
        pplay = np.zeros(len(te)); cond = np.zeros(len(te))

        for pos in POSITIONS:
            a = tr.filter(pl.col("position") == pos)
            b = te.filter(pl.col("position") == pos)
            if len(a) < 500 or len(b) < 50:
                continue

            A, ref = prep(a.select(FEATURES).to_numpy())
            B, _ = prep(b.select(FEATURES).to_numpy(), ref)
            y = a["fpts"].to_numpy().astype(float)
            ymax = float(a["fpts"].max())
            idx = (te["position"] == pos).to_numpy().nonzero()[0]

            # --- frozen V3, refit identically on the same rows
            v3[idx] = np.clip(fit_ridge(A, y)(B), 0.0, ymax)

            # --- hurdle part 1: availability
            played = (a["opps"].to_numpy() > 0).astype(int)
            p_hat = fit_classifier(A, played, a["season"].to_numpy())(B)
            pplay[idx] = p_hat

            # --- hurdle part 2: production GIVEN an opportunity
            m = played == 1
            if m.sum() < 300:
                cond[idx] = v3[idx]
                hurdle[idx] = v3[idx]
                continue
            Ap, refp = prep(a.filter(pl.col("opps") > 0).select(FEATURES).to_numpy())
            Bp, _ = prep(b.select(FEATURES).to_numpy(), refp)
            yp = a.filter(pl.col("opps") > 0)["fpts"].to_numpy().astype(float)
            c_hat = np.clip(fit_ridge(Ap, yp)(Bp), 0.0, ymax)
            cond[idx] = c_hat
            hurdle[idx] = p_hat * c_hat

        out.append(te.select(["season", "week", "pid", "position", "team", "opps"])
                     .with_columns([
                         pl.Series("y", te["fpts"].cast(float)),
                         pl.Series("naive", te["fpts__ewm"].cast(float)),
                         pl.Series("v3", v3),
                         pl.Series("hurdle", hurdle),
                         pl.Series("p_play", pplay),
                         pl.Series("cond", cond)]))

    R = pl.concat(out)
    R.write_parquet(f"tables/hurdle_{profile_name}.parquet")
    report(R)
    return R


def boot_pair(R, a, b, metric=RMSE, B=1500, seed=77):
    """Block bootstrap of metric(a) - metric(b); positive means b is better."""
    rng = np.random.default_rng(seed)
    d = R.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk")).drop_nulls(["y", a, b])
    blks = d["blk"].unique().to_list()
    by = {k: d.filter(pl.col("blk") == k) for k in blks}
    o = []
    for _ in range(B):
        s = pl.concat([by[k] for k in rng.choice(blks, len(blks), replace=True)])
        o.append(metric(s, a) - metric(s, b))
    o = np.array(o)
    return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o > 0).mean()


def brier(d):
    y = (d["opps"] > 0).to_numpy().astype(float)
    return float(np.mean((d["p_play"].to_numpy() - y) ** 2))


def block(R, label):
    print(f"\n{'='*76}\n{label}  n={len(R)}\n{'='*76}")
    print(f"  {'arm':8s}{'RMSE':>9s}{'MAE':>9s}{'Spearman':>10s}{'bias':>9s}"
          f"{'calib slope':>13s}{'intercept':>11s}")
    for c in ["naive", "v3", "hurdle", "cond"]:
        _, sl, ic, _ = calib(R, c)
        print(f"  {c:8s}{RMSE(R,c):9.4f}{MAE(R,c):9.4f}{spearman(R,c):10.4f}"
              f"{BIAS(R,c):+9.4f}{sl:13.3f}{ic:+11.3f}")

    print("\n  block-bootstrap RMSE, positive = second arm better")
    for a, b in [("naive", "v3"), ("naive", "hurdle"), ("v3", "hurdle")]:
        mu, lo, hi, p = boot_pair(R, a, b)
        print(f"    {a:6s} -> {b:6s}  {mu:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P>0={p:.3f}")
    print("  block-bootstrap MAE")
    for a, b in [("v3", "hurdle")]:
        mu, lo, hi, p = boot_pair(R, a, b, metric=MAE)
        print(f"    {a:6s} -> {b:6s}  {mu:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P>0={p:.3f}")

    print(f"\n  availability classifier: Brier {brier(R):.4f}  "
          f"mean P(play) {float(R['p_play'].mean()):.3f}  "
          f"actual play rate {float((R['opps']>0).mean()):.3f}")

    print("\n  played vs zero-opportunity")
    for lab, d in [("played", R.filter(pl.col("opps") > 0)),
                   ("zero opps", R.filter(pl.col("opps") == 0))]:
        mu, lo, hi, p = boot_pair(d, "v3", "hurdle", B=800)
        print(f"    {lab:10s} n={len(d):6d}  RMSE naive {RMSE(d,'naive'):7.4f} "
              f"v3 {RMSE(d,'v3'):7.4f} hurdle {RMSE(d,'hurdle'):7.4f} cond {RMSE(d,'cond'):7.4f}"
              f"   v3->hurdle {mu:+.4f} CI [{lo:+.4f}, {hi:+.4f}]")

    print("\n  by position (RMSE)")
    for pos in POSITIONS:
        d = R.filter(pl.col("position") == pos)
        if len(d) < 100:
            continue
        mu, lo, hi, p = boot_pair(d, "v3", "hurdle", B=800)
        print(f"    {pos:4s} n={len(d):6d}  naive {RMSE(d,'naive'):7.4f} v3 {RMSE(d,'v3'):7.4f} "
              f"hurdle {RMSE(d,'hurdle'):7.4f}   v3->hurdle {mu:+.4f} CI [{lo:+.4f}, {hi:+.4f}]")


def report(R):
    block(R.filter((pl.col("season") >= 2011) & (pl.col("season") <= 2017)),
          "FRESH BLOCK 2011-2017 (primary: fresh for V3 and for the hurdle arm)")
    block(R.filter(pl.col("season") >= 2018),
          "2018-2025 (V3 development seasons, biased toward V3)")

    print("\n  by season (RMSE)")
    print(f"    {'season':7s}{'n':>7s}{'naive':>9s}{'v3':>9s}{'hurdle':>9s}{'v3->hurdle':>12s}")
    for s in sorted(R["season"].unique().to_list()):
        d = R.filter(pl.col("season") == s)
        print(f"    {s:<7d}{len(d):7d}{RMSE(d,'naive'):9.4f}{RMSE(d,'v3'):9.4f}"
              f"{RMSE(d,'hurdle'):9.4f}{RMSE(d,'v3')-RMSE(d,'hurdle'):+12.4f}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "fanduel_test_slate_1")
