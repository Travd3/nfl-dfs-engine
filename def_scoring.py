"""
FanDuel defense/special-teams scoring.

Kept separate from scoring.py, which holds the skill-player profiles, because
the DEF unit is a different scoring object with a tiered component rather than
a linear one.

PROVENANCE. The values below were read directly from the Test Slate #1
FanDuel contest Rules screenshots supplied by the user.

The screenshots show:
  blocked punts/kicks +2
  blocked-kick return TD +6
  fumble return TD +6
  return TD +6
  safety +2
  extra-point return +2
  fumble recovery +2
  interception +2
  sack +1
  points allowed: 0:+10, 1-6:+7, 7-13:+4, 14-20:+1,
                  21-27:0, 28-34:-1, 35+:-4

The app omits the zero-point 21-27 tier from the visible list, so that tier is
represented as 0 between the adjacent +1 and -1 tiers.

POINTS-ALLOWED DEFINITION. The contest note explicitly defines points allowed
as offensive scoring only:

  6 * (Rushing TD + Receiving TD + Own fumbles recovered for TD)
  + 2 * (Two point conversions) + Extra points + 3 * Field Goals

Opponent defensive/special-teams TDs, safeties, and defensive conversion
returns therefore do not count toward the points-allowed tier.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DefScoring:
    name: str
    sack: float = 1.0
    interception: float = 2.0
    fumble_recovery: float = 2.0
    defensive_td: float = 6.0
    return_td: float = 6.0
    safety: float = 2.0
    blocked_kick: float = 2.0
    extra_point_return: float = 2.0
    # (upper_bound_inclusive, points). Last entry is the catch-all.
    points_allowed_tiers: tuple = (
        (0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0),
        (27, 0.0), (34, -1.0), (10_000, -4.0),
    )
    verified: bool = False
    source: str = ""

    def as_dict(self):
        return asdict(self)

    def tier_points(self, pa: float) -> float:
        for bound, pts in self.points_allowed_tiers:
            if pa <= bound:
                return pts
        return self.points_allowed_tiers[-1][1]


FANDUEL_DEF = DefScoring(
    name="fanduel_def_test_slate_1",
    verified=True,
    source="Actual FanDuel Test Slate #1 contest Rules screenshots supplied "
           "2026-09-18. The 21-27 points-allowed tier is the omitted zero-point "
           "tier between the visible 14-20 (+1) and 28-34 (-1) tiers.",
)

DEF_PROFILES = {p.name: p for p in (FANDUEL_DEF,)}

STAT_COLS = ["sacks", "interceptions", "fumble_recoveries", "defensive_tds",
             "return_tds", "safeties", "blocked_kicks", "extra_point_returns",
             "points_allowed"]

DEF_CATEGORIES = {
    "sack":                  ("sack", "sacks"),
    "interception":          ("interception", "interceptions"),
    "fumble recovery":       ("fumble_recovery", "fumble_recoveries"),
    "defensive touchdown":   ("defensive_td", "defensive_tds"),
    "return touchdown":      ("return_td", "return_tds"),
    "safety":                ("safety", "safeties"),
    "blocked kick":          ("blocked_kick", "blocked_kicks"),
    "extra point return":    ("extra_point_return", "extra_point_returns"),
    "points allowed tiers":  ("points_allowed_tiers", "points_allowed"),
}


def get(name="fanduel_def_test_slate_1"):
    if name not in DEF_PROFILES:
        raise KeyError(f"unknown DEF profile {name!r}")
    return DEF_PROFILES[name]


def points_allowed_expr(total_col="opponent_total_score",
                        def_tds_col="opp_def_tds",
                        return_tds_col="opp_return_tds",
                        safeties_col="opp_def_safeties",
                        xpr_col="opp_def_2pt"):
    """FanDuel contest definition of defensive points allowed."""
    import polars as pl
    return (
        pl.when(pl.col(total_col).is_not_null())
          .then(
              pl.col(total_col)
              - 6.0 * (pl.col(def_tds_col).fill_null(0.0)
                       + pl.col(return_tds_col).fill_null(0.0))
              - 2.0 * pl.col(safeties_col).fill_null(0.0)
              - 2.0 * pl.col(xpr_col).fill_null(0.0)
          )
          .otherwise(None)
          .alias("points_allowed")
    )


def score_expr(p: DefScoring):
    """Polars expression scoring one team-game defensive stat row."""
    import polars as pl
    base = (pl.col("sacks") * p.sack
            + pl.col("interceptions") * p.interception
            + pl.col("fumble_recoveries") * p.fumble_recovery
            + pl.col("defensive_tds") * p.defensive_td
            + pl.col("return_tds") * p.return_td
            + pl.col("safeties") * p.safety
            + pl.col("blocked_kicks") * p.blocked_kick
            + pl.col("extra_point_returns") * p.extra_point_return)

    tier = pl.lit(p.points_allowed_tiers[-1][1])
    for bound, pts in reversed(p.points_allowed_tiers[:-1]):
        tier = pl.when(pl.col("points_allowed") <= bound).then(pl.lit(pts)).otherwise(tier)

    return (base + tier).alias("def_fpts")
