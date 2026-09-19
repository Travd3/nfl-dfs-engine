"""
Scoring guards.

The enumeration in scoring.FANDUEL_SKILL_CATEGORIES is the rules contract.
These tests ensure every listed FanDuel skill-player category has a coefficient
and a feeding stat column. They do NOT declare the historical source semantics
exact; scoring.complete remains False until the rare return/fumble-recovery
mappings are exact enough for production labeling.
"""
import os, sys
import pytest
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scoring


def test_profiles_registered():
    assert {"dk_classic", "fanduel_test_slate_1"} <= set(scoring.PROFILES)


def test_fanduel_verified_values():
    fd = scoring.get("fanduel_test_slate_1")
    assert fd.reception == 0.5
    assert fd.fumble_lost == -2.0
    assert fd.two_pt == 2.0
    assert fd.return_td == 6.0
    assert fd.fumble_recovery_td == 6.0
    assert (fd.bonus_pass_yd, fd.bonus_pass_threshold) == (3.0, 300.0)
    assert (fd.bonus_rush_yd, fd.bonus_rush_threshold) == (3.0, 100.0)
    assert (fd.bonus_rec_yd, fd.bonus_rec_threshold) == (3.0, 100.0)
    assert fd.verified


def test_every_category_has_a_coefficient_and_a_column():
    fd = scoring.get("fanduel_test_slate_1")
    for label, (coef, col) in scoring.FANDUEL_SKILL_CATEGORIES.items():
        assert hasattr(fd, coef), f"{label}: profile has no field {coef}"
        assert getattr(fd, coef) != 0.0, f"{label}: coefficient {coef} is zero"
        assert col in scoring.STAT_COLS, f"{label}: {col} is not a built stat column"


def test_fanduel_historical_target_not_marked_exact_yet():
    fd = scoring.get("fanduel_test_slate_1")
    assert not fd.complete


def test_verified_profile_records_a_source():
    for p in scoring.PROFILES.values():
        if p.verified or p.complete:
            assert p.source, f"{p.name} claims verified/complete with no source"


@pytest.mark.parametrize("name", ["dk_classic", "fanduel_test_slate_1"])
def test_scoring_matches_hand_calculation(name):
    p = scoring.get(name)
    row = pl.DataFrame({c: [0.0] for c in scoring.STAT_COLS}).with_columns([
        pl.lit(342.0).alias("pass_yds"), pl.lit(3.0).alias("pass_tds"),
        pl.lit(82.0).alias("rush_yds"), pl.lit(3.0).alias("rush_tds"),
    ])
    got = float(row.select(scoring.score_expr(p))["fpts"][0])
    want = (342 * p.pass_yd + 3 * p.pass_td + 82 * p.rush_yd + 3 * p.rush_td
            + p.bonus_pass_yd)
    assert abs(got - want) < 1e-9


def test_bonus_thresholds_are_inclusive():
    p = scoring.get("fanduel_test_slate_1")
    row = pl.DataFrame({c: [0.0] for c in scoring.STAT_COLS}).with_columns(
        pl.lit(100.0).alias("rec_yds"))
    got = float(row.select(scoring.score_expr(p))["fpts"][0])
    assert abs(got - (100 * p.rec_yd + p.bonus_rec_yd)) < 1e-9


def test_two_point_credits_passer_and_scorer_in_weekly_source():
    import nflreadpy as nfl
    w = nfl.load_player_stats([2024]).filter(pl.col("season_type") == "REG")
    passer = float(w["passing_2pt_conversions"].sum())
    receiver = float(w["receiving_2pt_conversions"].sum())
    assert passer == receiver > 0, "passing and receiving 2pt totals must match"
