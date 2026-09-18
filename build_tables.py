"""
Stage 1: clean analytical tables.

Five channels, never pooled:
  qb_pass, qb_rush, rb_rush, rb_rec, wrte_rec

Every row is one opportunity. Conditions attached are ONLY those available
from free sources with a known publication lag. No participation/personnel.

DraftKings Classic, full PPR. Yardage bonuses are game-level and are applied
in the player-game table, never per opportunity.
"""
import polars as pl
import nflreadpy as nfl

SEASONS = [2022, 2023, 2024, 2025]      # FTN charting begins 2022
OUT = "tables"

# ---------------------------------------------------------------- positions
def position_map(seasons):
    ros = pl.concat([nfl.load_rosters_weekly([s]) for s in seasons], how="diagonal_relaxed")
    return (ros.select(["season", "gsis_id", "position"])
               .drop_nulls()
               .group_by(["season", "gsis_id"])
               .agg(pl.col("position").mode().first().alias("pos")))

# ---------------------------------------------------------------- charting
CONDITION_COLS = ["is_motion", "is_play_action", "is_rpo", "is_screen_pass",
                  "is_no_huddle", "qb_location", "n_defense_box",
                  "n_blitzers", "n_pass_rushers"]

def load_charting(seasons):
    f = pl.concat([nfl.load_ftn_charting([s]) for s in seasons], how="diagonal_relaxed")
    return f.select(
        [pl.col("nflverse_game_id").alias("game_id"),
         pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id")] +
        [pl.col(c) for c in CONDITION_COLS]
    ).with_columns([
        # explicit three-way alignment, never collapsed to a binary
        (pl.col("qb_location") == "U").cast(pl.Int8).alias("align_under"),
        (pl.col("qb_location") == "S").cast(pl.Int8).alias("align_shotgun"),
        (pl.col("qb_location") == "P").cast(pl.Int8).alias("align_pistol"),
    ]).with_columns([
        pl.col(c).cast(pl.Int8) for c in
        ["is_motion", "is_play_action", "is_rpo", "is_screen_pass", "is_no_huddle"]
    ])

# ---------------------------------------------------------------- main build
def build():
    pbp = (pl.concat([nfl.load_pbp([s]) for s in SEASONS], how="diagonal_relaxed")
             .with_columns(pl.col("play_id").cast(pl.Float64)))
    ftn = load_charting(SEASONS)
    pos = position_map(SEASONS)

    sched = (nfl.load_schedules()
               .filter(pl.col("season").is_in(SEASONS))
               .select(["game_id", "season", "week", "home_team", "away_team",
                        "spread_line", "total_line", "roof", "gameday"]))

    # live plays only; charting joined LEFT so uncharted plays stay visible
    plays = (pbp.filter(pl.col("play_type").is_in(["pass", "run"]))
                .filter(pl.col("posteam").is_not_null())
                .join(ftn, on=["game_id", "play_id"], how="left")
                .with_columns(pl.col("is_motion").is_not_null().alias("charted")))

    base = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "down", "ydstogo", "yardline_100", "score_differential", "wp",
            "half_seconds_remaining", "qtr", "charted"] +            ["is_motion", "is_play_action", "is_rpo", "is_screen_pass", "is_no_huddle",
            "align_under", "align_shotgun", "align_pistol",
            "n_defense_box", "n_blitzers", "n_pass_rushers"]

    def chan(df, pid, nm, pts, label):
        return df.select(base + [
            pl.col(pid).alias("pid"), pl.col(nm).alias("name"),
            pts.alias("pts"), pl.lit(label).alias("channel")])

    # --- passing: opportunity is a pass attempt, sacks excluded (no DK scoring)
    qb_pass = chan(
        plays.filter((pl.col("play_type") == "pass") & (pl.col("sack") == 0)
                     & pl.col("passer_player_id").is_not_null()),
        "passer_player_id", "passer_player_name",
        (pl.col("passing_yards").fill_null(0) * 0.04
         + pl.col("pass_touchdown").fill_null(0) * 4.0
         + pl.col("interception").fill_null(0) * -1.0),
        "qb_pass")

    rush_all = plays.filter(pl.col("rusher_player_id").is_not_null())
    rush_pts = (pl.col("rushing_yards").fill_null(0) * 0.1
                + pl.col("rush_touchdown").fill_null(0) * 6.0)
    rush = chan(rush_all, "rusher_player_id", "rusher_player_name", rush_pts, "rush")

    rec_all = plays.filter(pl.col("receiver_player_id").is_not_null())
    rec_pts = (pl.col("complete_pass").fill_null(0) * 1.0
               + pl.col("receiving_yards").fill_null(0) * 0.1
               + pl.col("pass_touchdown").fill_null(0) * 6.0)
    rec = chan(rec_all, "receiver_player_id", "receiver_player_name", rec_pts, "rec")

    # attach position, then split by it. RB carries and RB targets never mix.
    def with_pos(df):
        return df.join(pos, left_on=["season", "pid"], right_on=["season", "gsis_id"],
                       how="left").filter(pl.col("pos").is_not_null())

    rush, rec = with_pos(rush), with_pos(rec)
    qb_pass = with_pos(qb_pass)

    opps = pl.concat([
        qb_pass.filter(pl.col("pos") == "QB").with_columns(pl.lit("qb_pass").alias("channel")),
        rush.filter(pl.col("pos") == "QB").with_columns(pl.lit("qb_rush").alias("channel")),
        rush.filter(pl.col("pos") == "RB").with_columns(pl.lit("rb_rush").alias("channel")),
        rec.filter(pl.col("pos") == "RB").with_columns(pl.lit("rb_rec").alias("channel")),
        rec.filter(pl.col("pos").is_in(["WR", "TE"])).with_columns(pl.lit("wrte_rec").alias("channel")),
    ], how="diagonal_relaxed").join(sched.drop(["season", "week"]), on="game_id", how="left")

    opps.write_parquet(f"{OUT}/opportunities.parquet")

    # --- player-game rollup. Yardage bonuses belong here, not per play.
    pg = (opps.group_by(["season", "week", "game_id", "pid", "name", "pos",
                         "channel", "posteam", "defteam"])
              .agg([pl.len().alias("opps"),
                    pl.col("pts").sum().alias("pts"),
                    pl.col("charted").mean().alias("charted_share")]))
    pg.write_parquet(f"{OUT}/player_game.parquet")

    # --- team-game volume, for the role model
    tg = (opps.group_by(["season", "week", "game_id", "posteam", "channel"])
              .agg(pl.len().alias("team_opps")))
    tg.write_parquet(f"{OUT}/team_game.parquet")

    print("opportunities:", opps.shape)
    print(opps.group_by("channel").agg([
        pl.len().alias("n"),
        pl.col("pts").mean().round(3).alias("pts_per_opp"),
        pl.col("charted").mean().round(3).alias("charted_share")]).sort("channel"))
    print("player-games:", pg.shape)
    return opps


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    build()
