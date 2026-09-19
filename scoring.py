"""
Configurable fantasy scoring.

Scoring is a pure function of counting stats, so a platform is a config entry
rather than a table rebuild.

FanDuel Test Slate #1 rules are VERIFIED from the contest rules screenshots:
  300+ passing yards: +3
  100+ rushing yards: +3
  100+ receiving yards: +3
  receptions: +0.5
  fumbles lost: -2
  successful two-point conversion pass: +2
  successful two-point conversion scored: +2
  kickoff return touchdown: +6
  punt return touchdown: +6
  own fumble recovered touchdown: +6

Important distinction:
  verified=True means the contest rule values themselves are verified.
  complete=True means the historical target builder represents every listed
  category with sufficiently exact source semantics.

The FanDuel profile remains complete=False because the current weekly-stat
proxies for rare return TDs and fumble-recovery TDs are broader than the exact
contest definitions. Defense/special-teams UNIT scoring is separate.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Scoring:
    name: str
    pass_yd: float
    pass_td: float
    interception: float
    rush_yd: float
    rush_td: float
    reception: float
    rec_yd: float
    rec_td: float
    fumble_lost: float
    two_pt: float
    return_td: float = 0.0
    fumble_recovery_td: float = 0.0
    bonus_pass_yd: float = 0.0
    bonus_pass_threshold: float = 300.0
    bonus_rush_yd: float = 0.0
    bonus_rush_threshold: float = 100.0
    bonus_rec_yd: float = 0.0
    bonus_rec_threshold: float = 100.0
    verified: bool = False
    complete: bool = False
    source: str = ""

    def as_dict(self):
        return asdict(self)


DK_CLASSIC = Scoring(
    name="dk_classic",
    pass_yd=0.04, pass_td=4.0, interception=-1.0,
    rush_yd=0.1, rush_td=6.0,
    reception=1.0, rec_yd=0.1, rec_td=6.0,
    fumble_lost=-1.0, two_pt=2.0,
    return_td=6.0, fumble_recovery_td=6.0,
    bonus_pass_yd=3.0, bonus_rush_yd=3.0, bonus_rec_yd=3.0,
    verified=False, complete=False,
    source="Comparison profile only; not yet directly verified against the "
           "current DraftKings rules page.",
)

FANDUEL_TEST_SLATE_1 = Scoring(
    name="fanduel_test_slate_1",
    pass_yd=0.04, pass_td=4.0, interception=-1.0,
    rush_yd=0.1, rush_td=6.0,
    reception=0.5, rec_yd=0.1, rec_td=6.0,
    fumble_lost=-2.0, two_pt=2.0,
    return_td=6.0, fumble_recovery_td=6.0,
    bonus_pass_yd=3.0, bonus_rush_yd=3.0, bonus_rec_yd=3.0,
    verified=True, complete=False,
    source="Actual FanDuel Test Slate #1 contest Rules screenshots supplied "
           "2026-09-18. Rule values are verified. Historical target mapping "
           "for rare return-TD and own-fumble-recovery-TD categories remains "
           "approximate; DEF unit scoring is separate.",
)

PROFILES = {p.name: p for p in (DK_CLASSIC, FANDUEL_TEST_SLATE_1)}

FANDUEL_SKILL_CATEGORIES = {
    "passing yards":                 ("pass_yd", "pass_yds"),
    "passing touchdown":             ("pass_td", "pass_tds"),
    "interception thrown":           ("interception", "ints"),
    "rushing yards":                 ("rush_yd", "rush_yds"),
    "rushing touchdown":             ("rush_td", "rush_tds"),
    "reception":                     ("reception", "receptions"),
    "receiving yards":               ("rec_yd", "rec_yds"),
    "receiving touchdown":           ("rec_td", "rec_tds"),
    "fumble lost":                   ("fumble_lost", "fumbles_lost"),
    "two point conversion":          ("two_pt", "two_pts"),
    "kickoff/punt return touchdown": ("return_td", "return_tds"),
    "own fumble recovery touchdown": ("fumble_recovery_td", "fumble_rec_tds"),
    "300 passing yard bonus":        ("bonus_pass_yd", "pass_yds"),
    "100 rushing yard bonus":        ("bonus_rush_yd", "rush_yds"),
    "100 receiving yard bonus":      ("bonus_rec_yd", "rec_yds"),
}

STAT_COLS = ["pass_yds", "pass_tds", "ints", "rush_yds", "rush_tds",
             "receptions", "rec_yds", "rec_tds", "fumbles_lost", "two_pts",
             "return_tds", "fumble_rec_tds"]


def get(name):
    if name not in PROFILES:
        raise KeyError(f"unknown scoring profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


def score_expr(profile):
    """Polars expression scoring one player-game stat row under `profile`."""
    import polars as pl
    p = profile
    base = (
        pl.col("pass_yds") * p.pass_yd
        + pl.col("pass_tds") * p.pass_td
        + pl.col("ints") * p.interception
        + pl.col("rush_yds") * p.rush_yd
        + pl.col("rush_tds") * p.rush_td
        + pl.col("receptions") * p.reception
        + pl.col("rec_yds") * p.rec_yd
        + pl.col("rec_tds") * p.rec_td
        + pl.col("fumbles_lost") * p.fumble_lost
        + pl.col("two_pts") * p.two_pt
        + pl.col("return_tds") * p.return_td
        + pl.col("fumble_rec_tds") * p.fumble_recovery_td
    )
    bonus = (
        pl.when(pl.col("pass_yds") >= p.bonus_pass_threshold)
          .then(p.bonus_pass_yd).otherwise(0.0)
        + pl.when(pl.col("rush_yds") >= p.bonus_rush_threshold)
            .then(p.bonus_rush_yd).otherwise(0.0)
        + pl.when(pl.col("rec_yds") >= p.bonus_rec_threshold)
            .then(p.bonus_rec_yd).otherwise(0.0)
    )
    return (base + bonus).alias("fpts")
