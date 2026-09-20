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


def resolve_table(season: int, week: int) -> pl.DataFrame:
    """
    Every nflverse player on a roster this season, ANY status, with the
    normalized keys used for matching. Resolution is deliberately separated
    from projection eligibility: a name we can identify but cannot project is
    a different problem from a name we cannot identify at all, and collapsing
    the two into "unmapped" hides which one occurred.
    """
    ros = nfl.load_rosters_weekly([season]).filter(pl.col("week") <= week)
    ros = (ros.sort("week", descending=True)
              .unique(subset=["gsis_id"], keep="first")
              .select([pl.col("gsis_id").alias("pid"), "full_name",
                       pl.col("position").alias("nfl_position"),
                       "team", "status"])
              .drop_nulls("pid"))
    return ros.with_columns([
        pl.col("full_name").map_elements(norm_name, return_dtype=pl.Utf8).alias("nkey"),
        pl.col("team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"),
    ])


def map_to_fanduel(proj: pl.DataFrame, pool: pl.DataFrame,
                   season: int, week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    res = resolve_table(season, week)
    res_nt = res.unique(subset=["nkey", "tkey"], keep="first")
    res_n = res.unique(subset=["nkey"], keep="first")

    fd = pool.filter(pl.col("Position").is_in(POSITIONS)).with_columns([
        pl.when(pl.col("Nickname").str.len_chars() > 0).then(pl.col("Nickname"))
          .otherwise(pl.col("First Name") + " " + pl.col("Last Name")).alias("fd_name"),
    ])
    # Two name keys per row. FanDuel's Nickname is the display name and can be
    # a genuine nickname: "Hollywood Brown" where nflverse carries "Marquise
    # Brown". First+Last recovers those.
    fd = fd.with_columns([
        pl.col("fd_name").map_elements(norm_name, return_dtype=pl.Utf8).alias("nkey"),
        (pl.col("First Name") + " " + pl.col("Last Name"))
            .map_elements(norm_name, return_dtype=pl.Utf8).alias("nkey2"),
        pl.col("Team").map_elements(norm_team, return_dtype=pl.Utf8).alias("tkey"),
    ])

    RES = ["pid", "nfl_position", "status"]
    passes = [
        (res_nt.select(["nkey", "tkey"] + RES), ["nkey", "tkey"], ["nkey", "tkey"]),
        (res_nt.select(["nkey", "tkey"] + RES).rename({"nkey": "nkey2"}),
         ["nkey2", "tkey"], ["nkey2", "tkey"]),
        (res_n.select(["nkey"] + RES), ["nkey"], ["nkey"]),
        (res_n.select(["nkey"] + RES).rename({"nkey": "nkey2"}), ["nkey2"], ["nkey2"]),
    ]
    resolved, pending = None, fd
    for src, left_on, right_on in passes:
        if pending is None or len(pending) == 0:
            break
        j = pending.join(src, left_on=left_on, right_on=right_on, how="left")
        got = j.filter(pl.col("pid").is_not_null())
        resolved = got if resolved is None else pl.concat([resolved, got], how="diagonal_relaxed")
        pending = j.filter(pl.col("pid").is_null()).drop(RES)
    if pending is not None and len(pending):
        pending = pending.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in RES])
        resolved = pl.concat([resolved, pending], how="diagonal_relaxed")
    fd = resolved

    fd = fd.join(proj.select(["pid", "v3_projection"]), on="pid", how="left")

    fd = fd.with_columns([
        pl.when(pl.col("pid").is_null())
          .then(pl.lit("unresolved"))
          .when(pl.col("v3_projection").is_not_null())
          .then(pl.lit("ok"))
          .when(~pl.col("nfl_position").is_in(POSITIONS))
          .then(pl.lit("position_mismatch"))
          .when(pl.col("status") != "ACT")
          .then(pl.lit("inactive_or_practice_squad"))
          .otherwise(pl.lit("no_history"))
          .alias("mapping_status"),
    ]).with_columns([
        (pl.col("v3_projection") / (pl.col("Salary") / 1000.0)).alias("pts_per_1k"),
    ])

    table = fd.select([
        pl.col("Id").alias("fanduel_id"),
        pl.col("fd_name").alias("player"),
        pl.col("Position").alias("position"),
        pl.col("Salary").alias("salary"),
        pl.col("Team").alias("team"),
        pl.col("Opponent").alias("opponent"),
        pl.col("Game").alias("game"),
        pl.col("Roster Position").alias("roster_position"),
        pl.col("v3_projection").round(2),
        pl.col("pts_per_1k").round(3),
        pl.col("Injury Indicator").alias("injury_indicator"),
        pl.col("Injury Details").alias("injury_details"),
        pl.col("pid").alias("gsis_id"),
        pl.col("nfl_position").alias("nflverse_position"),
        pl.col("status").alias("nflverse_status"),
        "mapping_status",
    ]).sort(["position", "v3_projection"], descending=[False, True], nulls_last=True)

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
    print("\nmapping status totals:")
    print(table.group_by("mapping_status").len().sort("len", descending=True))
    print("\nby position:")
    print(table.pivot(values="fanduel_id", index="position",
                      on="mapping_status", aggregate_function="len").fill_null(0))
    print("\ngames represented:", table["game"].n_unique())
    if len(unmapped):
        print(f"\nUNMAPPED / UNPROJECTED ({len(unmapped)}):")
        print(unmapped.select(["player", "position", "team", "salary",
                               "injury_indicator", "mapping_status"]))
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
