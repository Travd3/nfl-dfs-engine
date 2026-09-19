"""
Issue #3: live V3 inference for a FanDuel slate.

Frozen V3 is not modified. The estimator, feature list, winsorization
percentiles, alpha grid and minimum-history filter are imported from
baseline_v3 and used exactly as committed, the same way hurdle.py did in
Issue #2. This module adds only the live plumbing: a completed-week guard,
slate-week feature construction, ID mapping, and an output table.

No optimizer logic lives here. This produces projections, nothing else.

--------------------------------------------------------------------------
COMPLETED-WEEK GUARD, which this slate specifically needs

The frozen historical pipeline has no notion of an in-progress week. It would
happily treat a partially played week as history. That is dangerous right now
because the current week is already partly played:

    week 1   Sep 9-14    16/16 final     <- latest FULLY completed week
    week 2   Sep 17 Thu, Sep 20 Sun, Sep 21 Mon   1/16 final
    week 3   Sep 24-28   0/16 final

A week counts as complete only when every scheduled game has a result.
Training stops at the last complete week. The roster universe creates a row
for every active skill player in the slate week with zero stats, because those
games have not been played; if such a week reached training it would teach the
model that every player scored zero.

The as-of feature rules already protect the feature side. PBP_LAG is 1, so a
row in week w draws on data stamped week w-1 or earlier, and the Thursday game
of the slate week cannot enter the Sunday slate's features. The guard here
protects the training side, which the lag does not cover.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
import polars as pl
import nflreadpy as nfl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import scoring
import player_games
from baseline_v3 import build_frame, prep, FEATURES, ALPHAS

POSITIONS = ["QB", "RB", "WR", "TE"]

# FanDuel and nflverse disagree on a handful of club abbreviations.
TEAM_ALIASES = {
    "JAC": "JAX", "JAX": "JAX",
    "WSH": "WAS", "WAS": "WAS",
    "LAR": "LA", "LA": "LA",
    "LVR": "LV", "LV": "LV", "OAK": "LV",
    "SD": "LAC", "LAC": "LAC",
    "STL": "LA", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "NWE": "NE", "NOR": "NO", "GNB": "GB", "KAN": "KC", "SFO": "SF", "TAM": "TB",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ------------------------------------------------------------------ vintage
def vintage(slate_season: int, slate_week: int) -> dict:
    """Source vintage for the live inputs. Written next to the projections."""
    v = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "nflreadpy": getattr(nfl, "__version__", "unknown"),
        "slate_season": slate_season,
        "slate_week": slate_week,
    }
    try:
        v["pbp_max_week"] = int(nfl.load_pbp([slate_season])["week"].max())
    except Exception:
        v["pbp_max_week"] = None
    try:
        v["weekly_stats_max_week"] = int(nfl.load_player_stats([slate_season])["week"].max())
    except Exception:
        v["weekly_stats_max_week"] = None
    try:
        v["rosters_max_week"] = int(nfl.load_rosters_weekly([slate_season])["week"].max())
    except Exception:
        v["rosters_max_week"] = None
    return v


# --------------------------------------------------------- completed weeks
def week_completion(season: int) -> pl.DataFrame:
    s = nfl.load_schedules().filter(pl.col("season") == season)
    return (s.group_by("week")
             .agg([pl.len().alias("games"),
                   pl.col("result").is_not_null().sum().alias("final")])
             .with_columns((pl.col("games") == pl.col("final")).alias("complete"))
             .sort("week"))


def latest_complete_week(season: int) -> int:
    wc = week_completion(season)
    done = wc.filter(pl.col("complete"))["week"].to_list()
    if not done:
        return 0
    # the last week such that every week up to it is complete
    last = 0
    for w in sorted(done):
        if w == last + 1:
            last = w
        else:
            break
    return last


# -------------------------------------------------------------- projection
def project(slate_season: int, slate_week: int, profile="fanduel_test_slate_1",
            min_hist=3, rebuild=True):
    if rebuild:
        player_games.build(profile, seasons=list(range(2009, slate_season + 1)))

    lcw = latest_complete_week(slate_season)
    print(f"latest fully completed week of {slate_season}: {lcw}")
    if slate_week <= lcw:
        raise SystemExit(
            f"slate week {slate_week} is already complete; this is a backtest, "
            "not a live projection")

    f = build_frame(profile)

    # training: every prior season in full, plus complete weeks of this season
    train_mask = ((pl.col("season") < slate_season)
                  | ((pl.col("season") == slate_season) & (pl.col("week") <= lcw)))
    tr = f.filter(train_mask & (pl.col("n_games") >= min_hist))
    te_all = f.filter((pl.col("season") == slate_season) & (pl.col("week") == slate_week))
    te = te_all.filter(pl.col("n_games") >= min_hist)

    # null n_games means no prior game at all (debut). Catch it explicitly:
    # a filter on "< min_hist" alone drops nulls and loses players silently.
    thin = te_all.filter(pl.col("n_games").is_null() | (pl.col("n_games") < min_hist))
    print(f"training rows {len(tr)} (through {slate_season} wk {lcw})")
    assert len(te) + len(thin) == len(te_all), "slate rows lost between buckets"
    print(f"slate rows {len(te_all)}; projectable {len(te)}; "
          f"insufficient history {len(thin)}")

    preds = np.full(len(te), np.nan)
    for pos in POSITIONS:
        a = tr.filter(pl.col("position") == pos)
        b = te.filter(pl.col("position") == pos)
        if len(a) < 500 or len(b) == 0:
            continue
        A, ref = prep(a.select(FEATURES).to_numpy())
        B, _ = prep(b.select(FEATURES).to_numpy(), ref)
        sc = StandardScaler().fit(A)
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), a["fpts"].to_numpy().astype(float))
        ymax = float(a["fpts"].max())
        p = np.clip(m.predict(sc.transform(B)), 0.0, ymax)
        preds[(te["position"] == pos).to_numpy().nonzero()[0]] = p

    out = te.select(["season", "week", "pid", "position", "team", "opp"]).with_columns(
        pl.Series("v3_projection", preds))
    thin_out = thin.select(["season", "week", "pid", "position", "team", "opp"]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("v3_projection"))
    return pl.concat([out, thin_out], how="vertical"), lcw


# ------------------------------------------------------------------ mapping
def norm_name(s: str) -> str:
    s = (s or "").lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    parts = [p for p in s.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def norm_team(t: str) -> str:
    t = (t or "").strip().upper()
    return TEAM_ALIASES.get(t, t)


def nflverse_names(season: int, week: int) -> pl.DataFrame:
    ros = nfl.load_rosters_weekly([season]).filter(pl.col("week") <= week)
    ros = (ros.sort("week", descending=True)
              .unique(subset=["gsis_id"], keep="first")
              .select([pl.col("gsis_id").alias("pid"), "full_name", "position", "team"]))
    return ros.with_columns([
        pl.col("full_name").map_elements(norm_name, return_dtype=pl.Utf8).alias("nkey"),
        pl.col("team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"),
    ])


def map_to_fanduel(proj: pl.DataFrame, pool: pl.DataFrame,
                   season: int, week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    names = nflverse_names(season, week)
    proj = proj.join(names.select(["pid", "full_name", "nkey", "tkey"]), on="pid", how="left")

    fd = pool.filter(pl.col("Position").is_in(POSITIONS)).with_columns([
        (pl.col("First Name") + " " + pl.col("Last Name")).alias("fd_name"),
    ])
    fd = fd.with_columns([
        pl.col("fd_name").map_elements(norm_name, return_dtype=pl.Utf8).alias("nkey"),
        pl.col("Team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"),
    ])

    # pass 1: name + team + position. pass 2: name + position only.
    p1 = fd.join(proj.select(["nkey", "tkey", "position", "pid", "v3_projection"]),
                 left_on=["nkey", "tkey", "Position"],
                 right_on=["nkey", "tkey", "position"], how="left")
    need = p1.filter(pl.col("pid").is_null()).drop(["pid", "v3_projection"])
    p2 = need.join(proj.select(["nkey", "position", "pid", "v3_projection"]).unique(subset=["nkey", "position"]),
                   left_on=["nkey", "Position"], right_on=["nkey", "position"], how="left")

    merged = pl.concat([p1.filter(pl.col("pid").is_not_null()),
                        p2], how="diagonal_relaxed")
    merged = merged.with_columns([
        pl.when(pl.col("pid").is_null()).then(pl.lit("unmapped"))
          .when(pl.col("v3_projection").is_null()).then(pl.lit("mapped_no_projection"))
          .otherwise(pl.lit("ok")).alias("mapping_status"),
    ]).with_columns([
        (pl.col("v3_projection") / (pl.col("Salary") / 1000.0)).alias("pts_per_1k"),
    ])

    table = merged.select([
        pl.col("Id").alias("fanduel_id"),
        pl.col("fd_name").alias("player"),
        pl.col("Position").alias("position"),
        pl.col("Salary").alias("salary"),
        pl.col("Team").alias("team"),
        pl.col("Opponent").alias("opponent"),
        pl.col("Roster Position").alias("roster_position"),
        pl.col("v3_projection").round(2),
        pl.col("pts_per_1k").round(3),
        pl.col("Injury Indicator").alias("injury_indicator"),
        pl.col("Injury Details").alias("injury_details"),
        pl.col("pid").alias("gsis_id"),
        "mapping_status",
    ]).sort(["position", "v3_projection"], descending=[False, True], nulls_last=True)

    # slate players the model could not reach, and model rows no FanDuel row claimed
    unmapped = table.filter(pl.col("mapping_status") != "ok")
    return table, unmapped


# ---------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="Live V3 projections for a FanDuel slate")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--slate", help="FanDuel entries-upload-template CSV")
    ap.add_argument("--profile", default="fanduel_test_slate_1")
    ap.add_argument("--no-rebuild", action="store_true")
    a = ap.parse_args()

    prof = scoring.get(a.profile)
    if not (prof.verified and prof.complete):
        print(f"WARNING: scoring profile {prof.name} is not verified+complete")

    os.makedirs("tables", exist_ok=True)
    v = vintage(a.season, a.week)
    print("week completion:")
    print(week_completion(a.season))

    proj, lcw = project(a.season, a.week, a.profile, rebuild=not a.no_rebuild)
    v["trained_through_week"] = lcw
    proj.write_parquet("tables/live_projection_model_side.parquet")

    if not a.slate:
        import json
        with open("tables/live_vintage.json", "w") as f:
            json.dump(v, f, indent=2)
        print("\nNo --slate supplied. Model-side projections written; "
              "FanDuel mapping skipped.")
        print(json.dumps(v, indent=2))
        return

    import ingest_fanduel
    pool = ingest_fanduel.parse_fanduel_template(a.slate)
    v["slate_file"] = os.path.basename(a.slate)
    v["slate_players"] = len(pool)

    table, unmapped = map_to_fanduel(proj, pool, a.season, a.week)
    table.write_parquet("tables/live_projections.parquet")
    table.write_csv("tables/live_projections.csv")

    import json
    with open("tables/live_vintage.json", "w") as f:
        json.dump(v, f, indent=2)

    ok = table.filter(pl.col("mapping_status") == "ok")
    print(f"\nslate skill players: {len(table)}   mapped+projected: {len(ok)} "
          f"({len(ok)/max(len(table),1):.1%})")
    print(table.group_by(["position", "mapping_status"]).len()
               .sort(["position", "mapping_status"]))
    if len(unmapped):
        print(f"\nUNMAPPED / UNPROJECTED ({len(unmapped)}):")
        print(unmapped.select(["player", "position", "team", "salary",
                               "injury_indicator", "mapping_status"]))
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()