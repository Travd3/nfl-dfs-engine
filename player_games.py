"""
Player-game stats, sourced primarily from the nflverse weekly player-stat
release.

Issue #1 item 2: prefer weekly box-score-aligned fields over reconstructing
common scoring categories from play-by-play. The weekly table carries:

  passing_2pt_conversions / rushing_2pt_conversions / receiving_2pt_conversions
      credits the passer and scorer natively
  sack_fumbles_lost / rushing_fumbles_lost / receiving_fumbles_lost
      splits common offensive fumbles by how they happened
  special_teams_tds / fumble_recovery_tds
      rare categories that PBP reconstruction previously omitted

Play-by-play is still used for team play counts.

TWO DEFINITIONAL GAPS ARE DELIBERATELY LEFT VISIBLE:

  special_teams_tds is broader than "kickoff or punt return touchdown".
  In the 2024 audit, PBP showed 14 return TDs while the weekly field totaled
  19 league-wide, 13 at skill positions. The extra events can include other
  special-teams touchdowns.

  fumble_recovery_tds cannot be split into OWN versus opponent recoveries in
  this weekly table. The FanDuel rule specifically pays own-fumble recovery TDs.

Those gaps are rare but prevent the historical FanDuel target from being
labeled exact/complete.

The zero-opportunity universe uses weekly roster status ACT. That means active
roster, not proof that the player dressed on gameday and not a historical
FanDuel slate membership file. Live slates use the official FanDuel player pool.
"""
import polars as pl
import nflreadpy as nfl
import scoring

SEASONS = list(range(2009, 2026))  # 2003-2008 targets unusable
POS = ["QB", "RB", "WR", "TE"]


def weekly_stats(seasons=SEASONS):
    w = pl.concat([nfl.load_player_stats([s]) for s in seasons],
                  how="diagonal_relaxed")
    w = w.filter(pl.col("season_type") == "REG")

    num = lambda c: pl.col(c).cast(pl.Float64).fill_null(0.0)
    return w.select([
        "season", "week",
        pl.col("player_id").alias("pid"),
        pl.col("team").alias("team"),
        pl.col("opponent_team").alias("opp"),
        num("passing_yards").alias("pass_yds"),
        num("passing_tds").alias("pass_tds"),
        num("passing_interceptions").alias("ints"),
        num("rushing_yards").alias("rush_yds"),
        num("rushing_tds").alias("rush_tds"),
        num("receptions").alias("receptions"),
        num("receiving_yards").alias("rec_yds"),
        num("receiving_tds").alias("rec_tds"),
        (num("sack_fumbles_lost") + num("rushing_fumbles_lost")
         + num("receiving_fumbles_lost")).alias("fumbles_lost"),
        (num("passing_2pt_conversions") + num("rushing_2pt_conversions")
         + num("receiving_2pt_conversions")).alias("two_pts"),
        num("special_teams_tds").alias("return_tds"),
        num("fumble_recovery_tds").alias("fumble_rec_tds"),
        num("attempts").alias("pass_att"),
        num("carries").alias("carries"),
        num("targets").alias("targets"),
    ])


def team_play_counts(seasons=SEASONS):
    """Distinct offensive pass/run plays per team-week."""
    import gc, os
    out = []
    for s in seasons:
        # cached per season: holding many seasons of play-by-play at once
        # exhausts memory on a 4 GB container. Pure I/O change, same counts.
        cache = f"tables/tp/{s}.parquet"
        if os.path.exists(cache):
            out.append(pl.read_parquet(cache))
            continue
        p = nfl.load_pbp([s])
        out.append(p.filter(pl.col("play_type").is_in(["pass", "run"])
                            & pl.col("posteam").is_not_null())
                    .select(["season", "week", "posteam", "play_id", "game_id"])
                    .unique(subset=["game_id", "play_id"])
                    .group_by(["season", "week", pl.col("posteam").alias("team")])
                    .agg(pl.len().alias("team_plays")))
        # release the season's play-by-play before loading the next one;
        # 17 seasons held simultaneously exhausts memory. Aggregates only.
        del p
        gc.collect()
    return pl.concat(out, how="diagonal_relaxed")


def universe(seasons=SEASONS):
    ros = pl.concat([nfl.load_rosters_weekly([s]) for s in seasons],
                    how="diagonal_relaxed")
    ros = (ros.filter(pl.col("position").is_in(POS) & (pl.col("status") == "ACT"))
              .select(["season", "week", "team", "position",
                       pl.col("gsis_id").alias("pid")])
              .drop_nulls("pid").unique())

    sched = nfl.load_schedules().filter(pl.col("season").is_in(seasons))
    games = pl.concat([
        sched.select(["season", "week", "game_id",
                      pl.col("home_team").alias("team"),
                      pl.col("away_team").alias("opp")]),
        sched.select(["season", "week", "game_id",
                      pl.col("away_team").alias("team"),
                      pl.col("home_team").alias("opp")]),
    ])
    return ros.join(games, on=["season", "week", "team"], how="inner")


def build(profile_name="fanduel_test_slate_1", seasons=SEASONS):
    prof = scoring.get(profile_name)
    w = weekly_stats(seasons)
    u = universe(seasons)
    tp = team_play_counts(seasons)

    full = u.join(w.drop("opp"), on=["season", "week", "pid", "team"], how="left")
    fill = scoring.STAT_COLS + ["pass_att", "carries", "targets"]
    full = full.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in fill])
    full = full.with_columns(scoring.score_expr(prof))
    full = full.join(tp, on=["season", "week", "team"], how="left")
    full = full.with_columns(
        (pl.col("pass_att") + pl.col("carries") + pl.col("targets")).alias("opps"))

    full.write_parquet(f"tables/pg_{prof.name}.parquet")

    played = full.filter(pl.col("opps") > 0)
    rare = full.filter((pl.col("return_tds") > 0) | (pl.col("fumble_rec_tds") > 0)
                       | (pl.col("two_pts") > 0))
    bonus_rows = full.filter((pl.col("pass_yds") >= 300) | (pl.col("rush_yds") >= 100)
                             | (pl.col("rec_yds") >= 100))
    print(f"profile {prof.name}  verified={prof.verified}  complete={prof.complete}")
    print(f"  rows {len(full)}  played {len(played)} ({len(played)/len(full):.1%})  "
          f"zero-opportunity {len(full)-len(played)}")
    print(f"  mean fpts all {full['fpts'].mean():.3f}  played {played['fpts'].mean():.3f}")
    print(f"  rows earning a yardage bonus: {len(bonus_rows)} ({len(bonus_rows)/len(full):.2%})")
    print(f"  rows with a rare category (2pt / return TD / fumble-rec TD): {len(rare)}")
    print(full.group_by("position").agg([
        pl.len().alias("n"),
        (pl.col("opps") == 0).mean().round(3).alias("zero_share"),
        pl.col("fpts").mean().round(2).alias("mean_fpts"),
        pl.col("fpts").max().round(1).alias("max_fpts")]).sort("position"))
    return full


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "fanduel_test_slate_1")
