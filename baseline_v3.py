"""
Baseline V3, committed and reproducible.

Single direct ridge on player-game fantasy points. No product of role and
efficiency, no scheme, no charting. Scoring comes from a configurable profile.

Validation is rolling origin: for each test season, train on every prior
season only.

IMPORTANT VALIDATION NOTE:
The current 2018-2023 block is no longer a pristine confirmatory holdout.
During this research pass, the 2022 TE results exposed a Taysom Hill
extrapolation failure and motivated the training-support winsorization below.
That fix is sensible, but once a test season changes the estimator it becomes
development evidence. A new untouched historical block is required before V3
can be called confirmed.

Availability is represented with `played_rate`, the exponentially weighted
share of recent games in which the player recorded an opportunity.
"""
import numpy as np, polars as pl
import nflreadpy as nfl
from scipy import stats as sps
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import scoring

HALF_LIFE = 8
PBP_LAG = 1
ALPHAS = np.logspace(-2, 4, 25)
FIRST_TEST = 2018

FEATURES = [
    "fpts__ewm", "opps__ewm", "targets__ewm", "carries__ewm", "pass_att__ewm",
    "played_rate__ewm", "play_share__ewm", "team_plays__ewm",
    "opp_allowed__ewm", "implied_total", "opp_implied_total", "team_spread",
    "total_line", "n_games", "is_home", "rest_days",
]


def _ewm_asof(df, keys, cols, lag=PBP_LAG):
    df = df.sort(keys + ["tindex"])
    e = df.with_columns(
        [pl.col(c).ewm_mean(half_life=HALF_LIFE).over(keys).alias(f"{c}__ewm") for c in cols]
        + [pl.col(cols[0]).cum_count().over(keys).alias("n_games")]
    ).select(keys + ["tindex", "n_games"] + [f"{c}__ewm" for c in cols])
    return e.with_columns((pl.col("tindex") + lag).alias("visible_at")).sort("visible_at")


def build_frame(profile_name):
    prof = scoring.get(profile_name)
    pg = pl.read_parquet(f"tables/pg_{prof.name}.parquet").with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("tindex"))

    pg = pg.with_columns([
        (pl.col("opps") > 0).cast(pl.Float64).alias("played"),
        (pl.col("opps") / pl.col("team_plays")).alias("play_share"),
    ])

    hist = _ewm_asof(pg, ["pid"],
                     ["fpts", "opps", "targets", "carries", "pass_att",
                      "played", "play_share"])
    hist = hist.rename({"played__ewm": "played_rate__ewm"})

    team = _ewm_asof(pg.select(["season", "week", "tindex", "team", "team_plays"])
                       .unique(subset=["season", "week", "team"]),
                     ["team"], ["team_plays"]).drop("n_games")

    opp = (pg.group_by(["season", "week", "tindex", "opp", "position"])
             .agg(pl.col("fpts").mean().alias("opp_allowed")))
    opp = _ewm_asof(opp, ["opp", "position"], ["opp_allowed"]).drop("n_games")

    f = pg.sort("tindex")
    f = f.join_asof(hist, left_on="tindex", right_on="visible_at",
                    by=["pid"], strategy="backward")
    f = f.join_asof(team, left_on="tindex", right_on="visible_at",
                    by=["team"], strategy="backward", suffix="_t")
    f = f.join_asof(opp, left_on="tindex", right_on="visible_at",
                    by=["opp", "position"], strategy="backward", suffix="_o")

    sched = nfl.load_schedules().select(
        ["game_id", "home_team", "spread_line", "total_line", "home_rest", "away_rest"])
    f = f.join(sched, on="game_id", how="left").with_columns([
        (pl.col("team") == pl.col("home_team")).cast(pl.Int8).alias("is_home"),
    ]).with_columns([
        pl.when(pl.col("is_home") == 1).then(pl.col("spread_line"))
          .otherwise(-pl.col("spread_line")).alias("team_spread"),
        pl.when(pl.col("is_home") == 1).then(pl.col("home_rest"))
          .otherwise(pl.col("away_rest")).alias("rest_days"),
    ]).with_columns([
        ((pl.col("total_line") + pl.col("team_spread")) / 2).alias("implied_total"),
        ((pl.col("total_line") - pl.col("team_spread")) / 2).alias("opp_implied_total"),
    ])
    return f


def prep(X, ref=None):
    """
    Column-wise median imputation plus clipping to the TRAINING feature range.

    This guard was added after a 2022 TE evaluation exposed a pathological
    extrapolation: Taysom Hill's QB-like pass/carry profile sat far outside the
    historical TE support and a linear model projected an absurd score.

    Because that observation changed the estimator, 2018-2023 is development
    evidence rather than an untouched final holdout.
    """
    X = np.asarray(X, float)
    if ref is None:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        lo = np.nanpercentile(X, 0.5, axis=0)
        hi = np.nanpercentile(X, 99.5, axis=0)
        ref = (med, lo, hi)
    med, lo, hi = ref
    i = np.where(np.isnan(X))
    X[i] = np.take(med, i[1])
    X = np.clip(X, lo, hi)
    return X, ref


def run(profile_name="fanduel_test_slate_1", min_hist=3):
    f = build_frame(profile_name)
    f = f.filter(pl.col("n_games") >= min_hist)
    seasons = sorted(f["season"].unique().to_list())
    tests = [s for s in seasons if s >= FIRST_TEST]

    out = []
    for ts in tests:
        tr, te = f.filter(pl.col("season") < ts), f.filter(pl.col("season") == ts)
        if len(tr) < 5000 or len(te) < 500:
            continue
        preds = np.zeros(len(te))
        for pos in ["QB", "RB", "WR", "TE"]:
            a, b = tr.filter(pl.col("position") == pos), te.filter(pl.col("position") == pos)
            if len(a) < 500 or len(b) < 50:
                continue
            A, ref = prep(a.select(FEATURES).to_numpy())
            B, _ = prep(b.select(FEATURES).to_numpy(), ref)
            sc = StandardScaler().fit(A)
            m = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), a["fpts"].to_numpy().astype(float))
            ymax = float(a["fpts"].max())
            pred = np.clip(m.predict(sc.transform(B)), 0.0, ymax)
            idx = (te["position"] == pos).to_numpy().nonzero()[0]
            preds[idx] = pred
        out.append(te.select(["season", "week", "pid", "position", "team", "opps"])
                     .with_columns([
                         pl.Series("y", te["fpts"].cast(float)),
                         pl.Series("naive", te["fpts__ewm"].cast(float)),
                         pl.Series("v3", preds)]))

    R = pl.concat(out)
    R.write_parquet(f"tables/v3_{profile_name}.parquet")
    report(R, profile_name)
    return R


def block_boot(R, metric, B=1500, seed=41):
    rng = np.random.default_rng(seed)
    d = R.with_columns((pl.col("season") * 100 + pl.col("week")).alias("blk")).drop_nulls(["y", "v3", "naive"])
    blks = d["blk"].unique().to_list()
    by = {k: d.filter(pl.col("blk") == k) for k in blks}
    o = []
    for _ in range(B):
        s = pl.concat([by[k] for k in rng.choice(blks, len(blks), replace=True)])
        o.append(metric(s, "naive") - metric(s, "v3"))
    o = np.array(o)
    return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o > 0).mean()


RMSE = lambda d, c: float((((d["y"] - d[c]) ** 2).mean()) ** 0.5)
MAE = lambda d, c: float((d["y"] - d[c]).abs().mean())
BIAS = lambda d, c: float((d[c] - d["y"]).mean())


def spearman(d, c):
    return float(sps.spearmanr(d["y"].to_numpy(), d[c].to_numpy()).statistic)


def calib(d, c, bins=10):
    """Regress actual on predicted across predicted deciles."""
    q = d.drop_nulls(["y", c]).with_columns(
        pl.col(c).qcut(bins, labels=[str(i) for i in range(bins)],
                       allow_duplicates=True).alias("b"))
    t = q.group_by("b").agg([pl.col(c).mean().alias("p"),
                             pl.col("y").mean().alias("a"),
                             pl.len().alias("n")]).sort("p")
    r = sps.linregress(t["p"].to_numpy(), t["a"].to_numpy())
    return t, r.slope, r.intercept, r.rvalue


def report(R, profile_name):
    prof = scoring.get(profile_name)
    print("=" * 74)
    print(f"BASELINE V3, rolling origin, scoring profile {prof.name} "
          f"(verified={prof.verified}, complete={prof.complete})")
    print(f"player-game grain, zero-opportunity active-roster rows included, n={len(R)}")
    print("=" * 74)

    for lab, sub in [("ALL 2018-2025 DEVELOPMENT EVIDENCE", R),
                     ("2018-2023 DEVELOPMENT BLOCK", R.filter(pl.col("season") <= 2023)),
                     ("reused 2024-2025", R.filter(pl.col("season") >= 2024))]:
        if not len(sub):
            continue
        mu, lo, hi, p = block_boot(sub, RMSE)
        mm, ml, mh, mp = block_boot(sub, MAE)
        print(f"\n{lab}  n={len(sub)}")
        print(f"  RMSE naive {RMSE(sub,'naive'):.4f}  v3 {RMSE(sub,'v3'):.4f}  "
              f"delta {mu:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P>0={p:.3f}")
        print(f"  MAE  naive {MAE(sub,'naive'):.4f}  v3 {MAE(sub,'v3'):.4f}  "
              f"delta {mm:+.4f}  CI [{ml:+.4f}, {mh:+.4f}]  P>0={mp:.3f}")
        print(f"  Spearman  naive {spearman(sub,'naive'):.4f}  v3 {spearman(sub,'v3'):.4f}")
        print(f"  bias      naive {BIAS(sub,'naive'):+.4f}  v3 {BIAS(sub,'v3'):+.4f}")
        for c in ("naive", "v3"):
            _, sl, ic, rv = calib(sub, c)
            print(f"  calib {c:5s} slope {sl:6.3f}  intercept {ic:+6.3f}  r {rv:.4f}")

    print("\nby season (RMSE)")
    print(f"  {'season':7s}{'n':>7s}{'naive':>9s}{'v3':>9s}{'delta':>9s}")
    for s in sorted(R["season"].unique().to_list()):
        d = R.filter(pl.col("season") == s)
        print(f"  {s:<7d}{len(d):7d}{RMSE(d,'naive'):9.4f}{RMSE(d,'v3'):9.4f}"
              f"{RMSE(d,'naive')-RMSE(d,'v3'):+9.4f}")

    print("\nby position (RMSE)")
    for pos in ["QB", "RB", "WR", "TE"]:
        d = R.filter(pl.col("position") == pos)
        if len(d):
            print(f"  {pos:4s} n={len(d):6d}  naive {RMSE(d,'naive'):7.4f}  "
                  f"v3 {RMSE(d,'v3'):7.4f}  delta {RMSE(d,'naive')-RMSE(d,'v3'):+.4f}")

    print("\nzero-opportunity rows vs played rows")
    for lab, d in [("played", R.filter(pl.col("opps") > 0)),
                   ("zero opps", R.filter(pl.col("opps") == 0))]:
        print(f"  {lab:10s} n={len(d):6d}  RMSE naive {RMSE(d,'naive'):7.4f} "
              f"v3 {RMSE(d,'v3'):7.4f} delta {RMSE(d,'naive')-RMSE(d,'v3'):+.4f}   "
              f"MAE naive {MAE(d,'naive'):6.4f} v3 {MAE(d,'v3'):6.4f}   "
              f"bias v3 {BIAS(d,'v3'):+.3f}")

    print("\ncalibration deciles, v3, 2018-2023 development block")
    t, sl, ic, rv = calib(R.filter(pl.col("season") <= 2023), "v3")
    print(f"  {'pred':>7s} {'actual':>7s} {'n':>7s}")
    for row in t.iter_rows(named=True):
        print(f"  {row['p']:7.2f} {row['a']:7.2f} {row['n']:7d}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "fanduel_test_slate_1")
