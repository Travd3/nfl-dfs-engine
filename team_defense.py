"""
Issue #4: FanDuel DEF baseline.

Deliberately simple. Transparent features, one linear model, compared against
a naive carry-forward. No scheme, coverage or personnel inputs, because the
issue asks for a defensible number for the first live test and nothing here
has yet earned more.

Targets are built at final team-game grain from nflverse team stats, with
safeties and blocked kicks recovered from play-by-play because the team-stats
release does not carry them cleanly. FanDuel points allowed is NOT simply the
opponent final score: the contest rules count offensive scoring only. The
target therefore subtracts opponent defensive/special-teams TDs, safeties and
defensive conversion returns from the opponent's final score.

All features respect the repo's as-of rule: a row in week w sees data stamped
week w-1 or earlier. For the live slate, training additionally stops at the
last fully completed week, so the already-played Week 2 Thursday game cannot
enter either training or features.
"""
from __future__ import annotations

import argparse, datetime as dt, gc, json, os

import numpy as np
import polars as pl
import nflreadpy as nfl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy import stats as sps

import def_scoring
from live_projection import latest_complete_week, norm_team

SEASONS = list(range(2009, 2027))
HALF_LIFE = 8
LAG = 1
ALPHAS = np.logspace(-2, 4, 25)
FIRST_TEST = 2011
MIN_HIST = 3

FEATURES = [
    "opp_implied_total", "team_spread", "total_line", "is_home",
    "d_fpts__ewm", "d_sacks__ewm", "d_int__ewm", "d_fum__ewm", "d_pa__ewm",
    "o_sacks_allowed__ewm", "o_int_thrown__ewm", "o_fum_lost__ewm", "o_points__ewm",
    "n_games",
]


def safeties_and_blocks(seasons=SEASONS, cache="tables/def_extras"):
    os.makedirs(cache, exist_ok=True)
    out = []
    for s in seasons:
        f = f"{cache}/{s}.parquet"
        if not os.path.exists(f):
            p = nfl.load_pbp([s])
            cols = p.columns
            blocked = pl.lit(0)
            if "field_goal_result" in cols:
                blocked = blocked + (pl.col("field_goal_result") == "blocked").cast(pl.Int32)
            if "extra_point_result" in cols:
                blocked = blocked + (pl.col("extra_point_result") == "blocked").cast(pl.Int32)
            if "punt_blocked" in cols:
                blocked = blocked + pl.col("punt_blocked").fill_null(0).cast(pl.Int32)
            (p.filter(pl.col("defteam").is_not_null())
               .with_columns([pl.col("safety").fill_null(0).cast(pl.Int32).alias("sf"),
                              blocked.alias("bk")])
               .group_by(["season", "week", pl.col("defteam").alias("team")])
               .agg([pl.col("sf").sum().alias("safeties"),
                     pl.col("bk").sum().alias("blocked_kicks")])
               .write_parquet(f))
            del p
            gc.collect()
        out.append(pl.read_parquet(f))
    return pl.concat(out, how="diagonal_relaxed")


def build_targets(seasons=SEASONS, profile="fanduel_def_test_slate_1"):
    p = def_scoring.get(profile)
    ts = pl.concat([nfl.load_team_stats([s]) for s in seasons], how="diagonal_relaxed")
    ts = ts.filter(pl.col("season_type") == "REG")

    num = lambda c: pl.col(c).cast(pl.Float64).fill_null(0.0)
    d = ts.select([
        "season", "week", "team", "opponent_team",
        num("def_sacks").alias("sacks"),
        num("def_interceptions").alias("interceptions"),
        num("fumble_recovery_opp").alias("fumble_recoveries"),
        num("def_tds").alias("defensive_tds"),
        num("special_teams_tds").alias("return_tds"),
        num("def_2pt_made").alias("extra_point_returns"),
        num("def_safeties").alias("team_def_safeties"),
        num("sacks_suffered").alias("o_sacks_allowed"),
        num("passing_interceptions").alias("o_int_thrown"),
        (num("sack_fumbles_lost") + num("rushing_fumbles_lost")
         + num("receiving_fumbles_lost")).alias("o_fum_lost"),
    ])

    extras = safeties_and_blocks(seasons)
    d = d.join(extras, on=["season", "week", "team"], how="left").with_columns([
        pl.col("safeties").cast(pl.Float64).fill_null(0.0),
        pl.col("blocked_kicks").cast(pl.Float64).fill_null(0.0),
    ])

    # FanDuel's contest note defines points allowed as offensive scoring only.
    # Opponent D/ST touchdowns, safeties and defensive conversion returns are
    # excluded from the tier.
    opp_dst = d.select([
        "season", "week", pl.col("team").alias("opponent_team"),
        pl.col("defensive_tds").alias("opp_def_tds"),
        pl.col("return_tds").alias("opp_return_tds"),
        pl.col("team_def_safeties").alias("opp_def_safeties"),
        pl.col("extra_point_returns").alias("opp_def_2pt"),
    ])

    sched = nfl.load_schedules().filter(pl.col("season").is_in(seasons))
    scores = pl.concat([
        sched.select(["season", "week", "game_id",
                      pl.col("home_team").alias("team"),
                      pl.col("away_team").alias("opponent_team"),
                      pl.col("away_score").alias("opponent_total_score"),
                      pl.col("home_score").alias("points_scored"),
                      pl.lit(1).alias("is_home"), "spread_line", "total_line"]),
        sched.select(["season", "week", "game_id",
                      pl.col("away_team").alias("team"),
                      pl.col("home_team").alias("opponent_team"),
                      pl.col("home_score").alias("opponent_total_score"),
                      pl.col("away_score").alias("points_scored"),
                      pl.lit(0).alias("is_home"),
                      (-pl.col("spread_line")).alias("spread_line"), "total_line"]),
    ])

    scores = scores.join(opp_dst, on=["season", "week", "opponent_team"], how="left")
    scores = scores.with_columns(def_scoring.points_allowed_expr())

    d = scores.join(d, on=["season", "week", "team", "opponent_team"], how="left")
    counting = [c for c in def_scoring.STAT_COLS if c != "points_allowed"]
    d = d.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0)
                        for c in counting if c in d.columns])
    d = d.with_columns([
        pl.col("points_allowed").cast(pl.Float64),
        pl.col("points_scored").cast(pl.Float64),
        (pl.col("season") * 100 + pl.col("week")).alias("tindex"),
        pl.col("spread_line").alias("team_spread"),
    ]).with_columns([
        ((pl.col("total_line") - pl.col("team_spread")) / 2).alias("opp_implied_total"),
        ((pl.col("total_line") + pl.col("team_spread")) / 2).alias("team_implied_total"),
    ])
    d = d.with_columns(
        pl.when(pl.col("points_allowed").is_not_null())
          .then(def_scoring.score_expr(p)).otherwise(None).alias("def_fpts"))
    return d


def build_frame(seasons=SEASONS, profile="fanduel_def_test_slate_1"):
    d = build_targets(seasons, profile)

    def ewm(df, keys, cols, prefix):
        df = df.sort(keys + ["tindex"])
        e = df.with_columns(
            [pl.col(c).ewm_mean(half_life=HALF_LIFE).over(keys).alias(f"{prefix}{c}__ewm")
             for c in cols]
            + [pl.col(cols[0]).cum_count().over(keys).alias("n_games")]
        ).select(keys + ["tindex", "n_games"] + [f"{prefix}{c}__ewm" for c in cols])
        return e.with_columns((pl.col("tindex") + LAG).alias("visible_at")).sort("visible_at")

    played = d.filter(pl.col("def_fpts").is_not_null())

    dh = ewm(played.select(["team", "tindex", "def_fpts", "sacks", "interceptions",
                            "fumble_recoveries", "points_allowed"]),
             ["team"],
             ["def_fpts", "sacks", "interceptions", "fumble_recoveries", "points_allowed"],
             "d_").rename({
        "d_def_fpts__ewm": "d_fpts__ewm",
        "d_interceptions__ewm": "d_int__ewm",
        "d_fumble_recoveries__ewm": "d_fum__ewm",
        "d_points_allowed__ewm": "d_pa__ewm"})

    oh = ewm(played.select(["team", "tindex", "o_sacks_allowed", "o_int_thrown",
                            "o_fum_lost", "points_scored"]),
             ["team"],
             ["o_sacks_allowed", "o_int_thrown", "o_fum_lost", "points_scored"],
             "o_").rename({"o_points_scored__ewm": "o_points__ewm",
                           "o_o_sacks_allowed__ewm": "o_sacks_allowed__ewm",
                           "o_o_int_thrown__ewm": "o_int_thrown__ewm",
                           "o_o_fum_lost__ewm": "o_fum_lost__ewm"}).drop("n_games")

    f = d.sort("tindex")
    f = f.join_asof(dh, left_on="tindex", right_on="visible_at", by="team",
                    strategy="backward")
    oh = oh.rename({"team": "opponent_team"})
    f = f.join_asof(oh, left_on="tindex", right_on="visible_at",
                    by="opponent_team", strategy="backward", suffix="_o")
    return f


def prep(X, ref=None):
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
    return np.clip(X, lo, hi), ref


RMSE = lambda d, c: float((((d["y"] - d[c]) ** 2).mean()) ** 0.5)
MAE = lambda d, c: float((d["y"] - d[c]).abs().mean())
BIAS = lambda d, c: float((d[c] - d["y"]).mean())


def spearman(d, c):
    return float(sps.spearmanr(d["y"].to_numpy(), d[c].to_numpy()).statistic)


def calib(d, c, bins=10):
    q = d.drop_nulls(["y", c]).with_columns(
        pl.col(c).qcut(bins, labels=[str(i) for i in range(bins)],
                       allow_duplicates=True).alias("b"))
    t = q.group_by("b").agg([pl.col(c).mean().alias("p"), pl.col("y").mean().alias("a"),
                             pl.len().alias("n")]).sort("p")
    r = sps.linregress(t["p"].to_numpy(), t["a"].to_numpy())
    return t, r.slope, r.intercept


def block_boot(R, a, b, metric=RMSE, B=1500, seed=91):
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


def backtest(profile="fanduel_def_test_slate_1"):
    f = build_frame(profile=profile).filter(
        pl.col("def_fpts").is_not_null() & (pl.col("n_games") >= MIN_HIST))
    out = []
    for ts in sorted(x for x in f["season"].unique().to_list() if x >= FIRST_TEST):
        tr, te = f.filter(pl.col("season") < ts), f.filter(pl.col("season") == ts)
        if len(tr) < 1000 or len(te) < 100:
            continue
        A, ref = prep(tr.select(FEATURES).to_numpy())
        B, _ = prep(te.select(FEATURES).to_numpy(), ref)
        y = tr["def_fpts"].to_numpy().astype(float)
        sc = StandardScaler().fit(A)
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), y)
        out.append(te.select(["season", "week", "team", "opponent_team"]).with_columns([
            pl.Series("y", te["def_fpts"].cast(float)),
            pl.Series("naive", te["d_fpts__ewm"].cast(float)),
            pl.Series("model", np.clip(m.predict(sc.transform(B)),
                                       float(y.min()), float(y.max()))),
        ]))
    R = pl.concat(out)
    R.write_parquet("tables/def_backtest.parquet")
    report(R)
    return R


def report(R):
    print("=" * 74)
    print(f"DEF BACKTEST, rolling origin, n={len(R)}")
    print("=" * 74)
    print(f"  {'arm':8s}{'RMSE':>9s}{'MAE':>9s}{'Spearman':>10s}{'bias':>9s}"
          f"{'calib slope':>13s}{'intercept':>11s}")
    for c in ["naive", "model"]:
        _, sl, ic = calib(R, c)
        print(f"  {c:8s}{RMSE(R,c):9.4f}{MAE(R,c):9.4f}{spearman(R,c):10.4f}"
              f"{BIAS(R,c):+9.4f}{sl:13.3f}{ic:+11.3f}")
    for metric, lab in [(RMSE, "RMSE"), (MAE, "MAE")]:
        mu, lo, hi, p = block_boot(R, "naive", "model", metric=metric)
        print(f"\n  {lab} naive -> model  {mu:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P>0={p:.3f}")
    print(f"\n  target: mean {float(R['y'].mean()):.2f}  sd {float(R['y'].std()):.2f}")
    print("\n  by season")
    print(f"    {'season':7s}{'n':>7s}{'naive':>9s}{'model':>9s}{'delta':>9s}")
    for s in sorted(R["season"].unique().to_list()):
        d = R.filter(pl.col("season") == s)
        print(f"    {s:<7d}{len(d):7d}{RMSE(d,'naive'):9.4f}{RMSE(d,'model'):9.4f}"
              f"{RMSE(d,'naive')-RMSE(d,'model'):+9.4f}")


def live(season, week, slate_csv, profile="fanduel_def_test_slate_1"):
    lcw = latest_complete_week(season)
    print(f"latest fully completed week of {season}: {lcw}")
    if week <= lcw:
        raise SystemExit("slate week already complete")

    f = build_frame(profile=profile)
    train = f.filter(
        pl.col("def_fpts").is_not_null() & (pl.col("n_games") >= MIN_HIST)
        & ((pl.col("season") < season)
           | ((pl.col("season") == season) & (pl.col("week") <= lcw))))
    slate = f.filter((pl.col("season") == season) & (pl.col("week") == week))
    print(f"training team-games {len(train)} (through {season} wk {lcw})")

    A, ref = prep(train.select(FEATURES).to_numpy())
    B, _ = prep(slate.select(FEATURES).to_numpy(), ref)
    y = train["def_fpts"].to_numpy().astype(float)
    sc = StandardScaler().fit(A)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), y)
    pred = np.clip(m.predict(sc.transform(B)), float(y.min()), float(y.max()))

    proj = slate.select(["season", "week", "team", "opponent_team", "is_home",
                         "opp_implied_total", "team_spread"]).with_columns(
        pl.Series("def_projection", pred))

    import ingest_fanduel
    pool = ingest_fanduel.parse_fanduel_template(slate_csv).filter(pl.col("Position") == "D")
    pool = pool.with_columns(
        pl.col("Team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"))
    proj = proj.with_columns(
        pl.col("team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"))

    t = pool.join(proj, on="tkey", how="left").with_columns(
        (pl.col("def_projection") / (pl.col("Salary") / 1000.0)).alias("pts_per_1k"))
    t = t.select([
        pl.col("Id").alias("fanduel_id"),
        pl.col("Nickname").alias("defense"),
        pl.col("Team").alias("team"),
        pl.col("Opponent").alias("opponent"),
        pl.col("Game").alias("game"),
        pl.col("Salary").alias("salary"),
        pl.col("def_projection").round(2),
        pl.col("pts_per_1k").round(3),
        pl.col("opp_implied_total").round(2),
        pl.col("team_spread"),
        pl.col("Injury Indicator").alias("injury_indicator"),
        pl.when(pl.col("def_projection").is_null()).then(pl.lit("unmapped"))
          .otherwise(pl.lit("ok")).alias("mapping_status"),
    ]).sort("def_projection", descending=True, nulls_last=True)

    os.makedirs("tables", exist_ok=True)
    t.write_csv("tables/live_def_projections.csv")
    t.write_parquet("tables/live_def_projections.parquet")
    p = def_scoring.get(profile)
    with open("tables/live_def_vintage.json", "w") as fh:
        json.dump({"generated_utc": dt.datetime.now(dt.UTC).isoformat(),
                   "nflreadpy": getattr(nfl, "__version__", "unknown"),
                   "season": season, "week": week,
                   "trained_through_week": lcw,
                   "def_profile": p.name, "def_profile_verified": p.verified,
                   "slate_file": os.path.basename(slate_csv),
                   "defenses": len(t)}, fh, indent=2)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--slate")
    a = ap.parse_args()
    os.makedirs("tables", exist_ok=True)
    if a.backtest:
        backtest()
    if a.slate:
        t = live(a.season, a.week, a.slate)
        print(f"\ndefenses: {len(t)}  games: {t['game'].n_unique()}")
        print(t)
