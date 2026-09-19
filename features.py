"""
Stage 2: as-of-lock-time features.

Two publication lags, kept separate on purpose:

  PBP_LAG   = 1 week. Play-by-play for week w-1 is posted before week w locks.
  CHART_LAG = 1 week as of 2026-09-19. The live canary observed PBP and FTN
              both through week 2 before the week 3 Sunday lock. This value is
              source-vintage dependent and must keep being checked.

Every feature is stamped with the week it becomes usable and joined backward.
Nothing in this module can see the row it is predicting.

Role and efficiency are built as separate targets throughout:
  role       = opportunities in the channel
  efficiency = points per opportunity
"""
import polars as pl
import numpy as np

PBP_LAG = 1
CHART_LAG = 1
HALF_LIFE = 8          # games, for recency weighting
TINDEX = pl.col("season") * 100 + pl.col("week")

CONDITIONS = ["is_motion", "is_play_action", "is_rpo", "is_screen_pass",
              "is_no_huddle", "align_under", "align_shotgun", "align_pistol"]


def _ewm_asof(df, keys, value_cols, stamp="tindex"):
    """Cumulative EWMA per key, stamped at the game it becomes known."""
    df = df.sort(keys + [stamp])
    out = df.with_columns(
        [pl.col(c).ewm_mean(half_life=HALF_LIFE).over(keys).alias(f"{c}__ewm")
         for c in value_cols] +
        [pl.col(value_cols[0]).cum_count().over(keys).alias("n_games")]
    )
    return out.select(keys + [stamp, "n_games"] + [f"{c}__ewm" for c in value_cols])


def _asof(target, source, by, lag, suffix):
    """Backward as-of join: source rows stamped at tindex t are visible to
    target rows with tindex >= t + lag."""
    src = source.with_columns((pl.col("tindex") + lag).alias("visible_at")).sort("visible_at")
    tgt = target.sort("tindex")
    joined = tgt.join_asof(src, left_on="tindex", right_on="visible_at",
                           by=by, strategy="backward", suffix=suffix)
    return joined


def build_frame():
    opps = pl.read_parquet("tables/opportunities.parquet")
    pg = pl.read_parquet("tables/player_game.parquet").with_columns(TINDEX.alias("tindex"))
    tg = pl.read_parquet("tables/team_game.parquet").with_columns(TINDEX.alias("tindex"))
    opps = opps.with_columns(TINDEX.alias("tindex"))

    # ---------------- targets ----------------
    frame = pg.with_columns([
        (pl.col("pts") / pl.col("opps")).alias("y_ppo"),
        pl.col("opps").alias("y_opps"),
        pl.col("pts").alias("y_pts"),
    ])

    # ---------------- player history (pbp lag) ----------------
    ph = pg.join(tg, on=["season", "week", "game_id", "posteam", "channel"], how="left") \
           .with_columns((pl.col("opps") / pl.col("team_opps")).alias("share")) \
           .with_columns((pl.col("pts") / pl.col("opps")).alias("ppo"))
    ph_e = _ewm_asof(ph, ["pid", "channel"], ["opps", "share", "ppo", "pts"])
    frame = _asof(frame, ph_e.rename({"n_games": "p_ngames"}),
                  by=["pid", "channel"], lag=PBP_LAG, suffix="_p")

    # ---------------- team volume history (pbp lag) ----------------
    tg_e = _ewm_asof(tg, ["posteam", "channel"], ["team_opps"])
    frame = _asof(frame, tg_e.rename({"n_games": "t_ngames"}),
                  by=["posteam", "channel"], lag=PBP_LAG, suffix="_t")

    # ---------------- opponent efficiency allowed (pbp lag) ----------------
    opp_g = (pg.group_by(["season", "week", "tindex", "defteam", "channel"])
               .agg([(pl.col("pts").sum() / pl.col("opps").sum()).alias("ppo_allowed"),
                     pl.col("opps").sum().alias("opps_allowed")]))
    opp_e = _ewm_asof(opp_g, ["defteam", "channel"], ["ppo_allowed", "opps_allowed"])
    frame = _asof(frame, opp_e.rename({"n_games": "d_ngames"}),
                  by=["defteam", "channel"], lag=PBP_LAG, suffix="_d")

    # ---------------- offensive scheme rates (CHART lag) ----------------
    ch = opps.filter(pl.col("charted"))
    team_sch = ch.group_by(["season", "week", "tindex", "posteam"]).agg(
        [pl.col(c).mean().alias(f"off_{c}") for c in CONDITIONS] +
        [pl.col("n_defense_box").mean().alias("off_box_faced")])
    ts_e = _ewm_asof(team_sch, ["posteam"],
                     [f"off_{c}" for c in CONDITIONS] + ["off_box_faced"])
    frame = _asof(frame, ts_e.rename({"n_games": "sch_ngames"}),
                  by=["posteam"], lag=CHART_LAG, suffix="_os")

    # ---------------- defensive tendencies (CHART lag) ----------------
    def_sch = ch.group_by(["season", "week", "tindex", "defteam"]).agg([
        pl.col("n_defense_box").mean().alias("def_box"),
        pl.col("n_blitzers").mean().alias("def_blitz"),
        pl.col("n_pass_rushers").mean().alias("def_rushers")])
    ds_e = _ewm_asof(def_sch, ["defteam"], ["def_box", "def_blitz", "def_rushers"])
    frame = _asof(frame, ds_e.rename({"n_games": "dsch_ngames"}),
                  by=["defteam"], lag=CHART_LAG, suffix="_ds")

    # ---------------- vegas (known pre-lock) ----------------
    import nflreadpy as nfl
    sched = nfl.load_schedules().select(
        ["game_id", "home_team", "away_team", "spread_line", "total_line", "roof"])
    frame = frame.join(sched, on="game_id", how="left").with_columns([
        pl.when(pl.col("posteam") == pl.col("home_team"))
          .then(pl.col("spread_line")).otherwise(-pl.col("spread_line")).alias("team_spread"),
    ]).with_columns([
        ((pl.col("total_line") + pl.col("team_spread")) / 2).alias("implied_total"),
        ((pl.col("total_line") - pl.col("team_spread")) / 2).alias("opp_implied_total"),
        (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Int8).alias("indoors"),
    ])

    return frame


FEATURES_ROLE = ["opps__ewm", "share__ewm", "team_opps__ewm",
                 "implied_total", "opp_implied_total", "team_spread", "total_line",
                 "p_ngames", "t_ngames", "indoors"]

FEATURES_EFF = ["ppo__ewm", "pts__ewm", "ppo_allowed__ewm", "opps_allowed__ewm",
                "implied_total", "opp_implied_total", "team_spread",
                "p_ngames", "d_ngames", "indoors"]

FEATURES_SCHEME_TEAM = [f"off_{c}__ewm" for c in CONDITIONS] + \
                       ["off_box_faced__ewm", "def_box__ewm", "def_blitz__ewm",
                        "def_rushers__ewm"]


if __name__ == "__main__":
    f = build_frame()
    f.write_parquet("tables/frame.parquet")
    print("frame:", f.shape)
    print(f.select(["season", "week", "channel", "y_opps", "y_ppo",
                    "opps__ewm", "share__ewm", "implied_total",
                    "off_is_play_action__ewm", "sch_ngames"]).tail(4))
    miss = {c: f[c].null_count() / len(f) for c in
            FEATURES_ROLE + FEATURES_EFF + FEATURES_SCHEME_TEAM if c in f.columns}
    print("\nnull share by feature:")
    for k, v in sorted(miss.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:28s} {v:.3f}")
