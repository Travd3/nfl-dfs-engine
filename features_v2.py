"""
Baseline V2 features.

Adds role information the first baseline ignored entirely:

  snap share        offense_pct, the strongest available proxy for whether a
                    player is on the field. Joined via the gsis/pfr crosswalk.
  red zone share    inside the 20, where touchdowns come from
  team pace         plays per game, which sets the opportunity pool
  rest and venue    known from the schedule before lock

All of it flows through the same as-of machinery with PBP_LAG, so nothing
here can see the game it is predicting.
"""
import polars as pl
import nflreadpy as nfl
from features import _ewm_asof, _asof, PBP_LAG, TINDEX


def snap_history():
    xw = (nfl.load_ff_playerids().select(["gsis_id", "pfr_id"])
            .drop_nulls().unique(subset=["pfr_id"]))
    sc = pl.concat([nfl.load_snap_counts([s]) for s in [2022, 2023, 2024, 2025]],
                   how="diagonal_relaxed")
    sc = (sc.join(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
            .rename({"gsis_id": "pid"})
            .select(["season", "week", "pid", "offense_pct", "offense_snaps"])
            .drop_nulls("offense_pct")
            .with_columns(TINDEX.alias("tindex"))
            .unique(subset=["season", "week", "pid"]))
    return _ewm_asof(sc, ["pid"], ["offense_pct", "offense_snaps"])


def redzone_history(opps):
    rz = (opps.filter(pl.col("yardline_100") <= 20)
              .group_by(["season", "week", "tindex", "pid", "channel"])
              .agg(pl.len().alias("rz_opps")))
    team_rz = (opps.filter(pl.col("yardline_100") <= 20)
                   .group_by(["season", "week", "posteam", "channel"])
                   .agg(pl.len().alias("team_rz")))
    rz = rz.join(opps.select(["season", "week", "pid", "channel", "posteam"]).unique(),
                 on=["season", "week", "pid", "channel"], how="left")
    rz = rz.join(team_rz, on=["season", "week", "posteam", "channel"], how="left")
    rz = rz.with_columns((pl.col("rz_opps") / pl.col("team_rz")).alias("rz_share"))
    return _ewm_asof(rz, ["pid", "channel"], ["rz_opps", "rz_share"])


def pace_history(opps):
    tp = (opps.group_by(["season", "week", "tindex", "posteam"])
              .agg(pl.len().alias("team_plays")))
    return _ewm_asof(tp, ["posteam"], ["team_plays"])


def build():
    frame = pl.read_parquet("tables/frame.parquet")
    opps = pl.read_parquet("tables/opportunities.parquet").with_columns(TINDEX.alias("tindex"))

    frame = _asof(frame, snap_history().rename({"n_games": "sn_ngames"}),
                  by=["pid"], lag=PBP_LAG, suffix="_sn")
    frame = _asof(frame, redzone_history(opps).rename({"n_games": "rz_ngames"}),
                  by=["pid", "channel"], lag=PBP_LAG, suffix="_rz")
    frame = _asof(frame, pace_history(opps).rename({"n_games": "pc_ngames"}),
                  by=["posteam"], lag=PBP_LAG, suffix="_pc")

    sched = nfl.load_schedules().select(
        ["game_id", "home_team", "away_team", "home_rest", "away_rest"])
    frame = frame.join(sched.drop(["home_team", "away_team"]), on="game_id", how="left")
    frame = frame.join(sched.select(["game_id", "home_team"]), on="game_id", how="left",
                       suffix="_hr")
    frame = frame.with_columns([
        (pl.col("posteam") == pl.col("home_team_hr")).cast(pl.Int8).alias("is_home"),
    ]).with_columns([
        pl.when(pl.col("is_home") == 1).then(pl.col("home_rest"))
          .otherwise(pl.col("away_rest")).alias("rest_days"),
    ])

    frame.write_parquet("tables/frame_v2.parquet")
    print("frame_v2:", frame.shape)
    for c in ["offense_pct__ewm", "rz_share__ewm", "team_plays__ewm", "rest_days", "is_home"]:
        print(f"  {c:22s} null {frame[c].null_count()/len(frame):.3f}")
    return frame


NEW_ROLE = ["offense_pct__ewm", "offense_snaps__ewm", "rz_opps__ewm", "rz_share__ewm",
            "team_plays__ewm", "rest_days", "is_home"]
NEW_EFF = ["offense_pct__ewm", "rz_share__ewm", "team_plays__ewm", "is_home"]


if __name__ == "__main__":
    build()
