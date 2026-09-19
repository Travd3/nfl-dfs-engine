"""
Player-game stats at the grain used for model evaluation.

Corrections from Issue #1:
1. Scoring is applied from counting stats under a selected profile.
2. Passing/rushing/receiving channels are recombined to one player-game.
3. Active-roster zero-opportunity rows are retained.
4. Team plays are counted as distinct (game_id, play_id), not rows in the
   concatenated opportunity table.
5. Successful two-point passes now credit both the passer and the player who
   scores the conversion.

Important limitation: roster status ACT means active roster, not confirmed
gameday-active. That is intentional for availability modeling, but it must not
be described as proof that the player dressed.

The historical builder still needs player KR/PR return TDs and own-fumble-
recovery TDs for exact FanDuel scoring. Until those are added the FanDuel
profile remains complete=False.
"""
import polars as pl
import nflreadpy as nfl
import scoring

SEASONS = list(range(2016, 2026))
POS = ["QB", "RB", "WR", "TE"]


def player_game_stats(seasons=SEASONS):
    frames = []
    team_plays = []
    for s in seasons:
        p = nfl.load_pbp([s]).filter(pl.col("play_type").is_in(["pass", "run"]))

        tp = (p.filter(pl.col("posteam").is_not_null())
                .select(["season", "week", "game_id", "posteam", "play_id"])
                .unique(subset=["game_id", "play_id"])
                .group_by(["season", "week", "game_id", "posteam"])
                .agg(pl.len().alias("team_plays")))
        team_plays.append(tp)

        def agg(idcol, exprs):
            return (p.filter(pl.col(idcol).is_not_null())
                      .group_by(["season", "week", "game_id",
                                 pl.col(idcol).alias("pid"),
                                 pl.col("posteam").alias("team"),
                                 pl.col("defteam").alias("opp")])
                      .agg(exprs))

        passing = agg("passer_player_id", [
            pl.col("passing_yards").fill_null(0).sum().alias("pass_yds"),
            pl.col("pass_touchdown").fill_null(0).sum().alias("pass_tds"),
            pl.col("interception").fill_null(0).sum().alias("ints"),
            ((pl.col("play_type") == "pass") & (pl.col("sack") == 0))
                .sum().alias("pass_att"),
        ])
        rushing = agg("rusher_player_id", [
            pl.col("rushing_yards").fill_null(0).sum().alias("rush_yds"),
            pl.col("rush_touchdown").fill_null(0).sum().alias("rush_tds"),
            pl.len().alias("carries"),
        ])
        receiving = agg("receiver_player_id", [
            pl.col("receiving_yards").fill_null(0).sum().alias("rec_yds"),
            pl.col("pass_touchdown").fill_null(0).sum().alias("rec_tds"),
            pl.col("complete_pass").fill_null(0).sum().alias("receptions"),
            pl.len().alias("targets"),
        ])
        fum = (p.filter(pl.col("fumbled_1_player_id").is_not_null())
                 .group_by(["season", "week", "game_id",
                            pl.col("fumbled_1_player_id").alias("pid")])
                 .agg(pl.col("fumble_lost").fill_null(0).sum().alias("fumbles_lost")))

        success_2pt = p.filter(
            (pl.col("two_point_attempt") == 1)
            & (pl.col("two_point_conv_result") == "success")
        )

        # The scorer receives +2 on a successful rush/receive conversion.
        twp_scored = (success_2pt
            .select(["season", "week", "game_id",
                     pl.coalesce(["rusher_player_id", "receiver_player_id"]).alias("pid")])
            .drop_nulls("pid"))

        # The passer also receives +2 on a successful passing conversion.
        twp_pass = (success_2pt
            .filter(pl.col("passer_player_id").is_not_null()
                    & pl.col("receiver_player_id").is_not_null())
            .select(["season", "week", "game_id",
                     pl.col("passer_player_id").alias("pid")]))

        twp = (pl.concat([twp_scored, twp_pass], how="diagonal_relaxed")
                 .group_by(["season", "week", "game_id", "pid"])
                 .agg(pl.len().cast(pl.Float64).alias("two_pts")))

        j = passing.join(rushing, on=["season", "week", "game_id", "pid", "team", "opp"],
                         how="full", coalesce=True)
        j = j.join(receiving, on=["season", "week", "game_id", "pid", "team", "opp"],
                   how="full", coalesce=True)
        j = j.join(fum, on=["season", "week", "game_id", "pid"], how="left")
        j = j.join(twp, on=["season", "week", "game_id", "pid"], how="left")
        frames.append(j)

    stats = pl.concat(frames, how="diagonal_relaxed")
    tp = pl.concat(team_plays, how="diagonal_relaxed")

    num = scoring.STAT_COLS + ["pass_att", "carries", "targets"]
    stats = stats.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0)
                                for c in num if c in stats.columns])
    return stats, tp


def add_zero_rows(stats, seasons=SEASONS):
    """Every active-roster skill player gets a row, scored zero if absent."""
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
    universe = ros.join(games, on=["season", "week", "team"], how="inner")

    full = universe.join(
        stats.drop(["opp"]), on=["season", "week", "game_id", "pid", "team"],
        how="left")
    full = full.with_columns([pl.col(c).fill_null(0.0)
                              for c in scoring.STAT_COLS + ["pass_att", "carries", "targets"]])
    return full


def build(profile_name="fanduel_test_slate_1", seasons=SEASONS):
    prof = scoring.get(profile_name)
    stats, tp = player_game_stats(seasons)
    full = add_zero_rows(stats, seasons)
    full = full.with_columns(scoring.score_expr(prof))
    full = full.join(tp.drop("game_id").rename({"posteam": "team"}),
                     on=["season", "week", "team"], how="left")

    full = full.with_columns(
        (pl.col("pass_att") + pl.col("carries") + pl.col("targets")).alias("opps"))
    full.write_parquet(f"tables/pg_{prof.name}.parquet")

    played = full.filter(pl.col("opps") > 0)
    print(f"profile {prof.name}  verified={prof.verified} complete={prof.complete}")
    print(f"  rows {len(full)},  with an opportunity {len(played)} "
          f"({len(played)/len(full):.1%}),  zero-opportunity {len(full)-len(played)}")
    print(f"  mean fpts all rows {full['fpts'].mean():.3f}, "
          f"played only {played['fpts'].mean():.3f}")
    print(full.group_by("position").agg([
        pl.len().alias("n"),
        (pl.col("opps") == 0).mean().round(3).alias("zero_share"),
        pl.col("fpts").mean().round(2).alias("mean_fpts")]).sort("position"))
    return full


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "fanduel_test_slate_1")
