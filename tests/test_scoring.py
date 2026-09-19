"""Scoring-profile guards."""
import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scoring


def test_profiles_registered():
    assert {"dk_classic", "fanduel_test_slate_1"} <= set(scoring.PROFILES)


def test_fanduel_matches_actual_contest_core_rules():
    fd = scoring.get("fanduel_test_slate_1")
    assert fd.reception == 0.5
    assert fd.fumble_lost == -2.0
    assert fd.two_pt == 2.0
    assert fd.bonus_pass_threshold == 300.0 and fd.bonus_pass_yd == 3.0
    assert fd.bonus_rush_threshold == 100.0 and fd.bonus_rush_yd == 3.0
    assert fd.bonus_rec_threshold == 100.0 and fd.bonus_rec_yd == 3.0
    assert fd.verified


def test_fanduel_not_complete_until_rare_player_scoring_is_added():
    fd = scoring.get("fanduel_test_slate_1")
    assert not fd.complete


@pytest.mark.parametrize("name", ["dk_classic", "fanduel_test_slate_1"])
def test_scoring_is_a_pure_function_of_stats(name):
    import polars as pl
    p = scoring.get(name)
    row = pl.DataFrame({c: [0.0] for c in scoring.STAT_COLS}).with_columns([
        pl.lit(300.0).alias("pass_yds"), pl.lit(2.0).alias("pass_tds"),
        pl.lit(105.0).alias("rush_yds"), pl.lit(1.0).alias("rush_tds"),
        pl.lit(6.0).alias("receptions"), pl.lit(101.0).alias("rec_yds"),
    ])
    got = float(row.select(scoring.score_expr(p))["fpts"][0])
    want = (300 * p.pass_yd + 2 * p.pass_td + 105 * p.rush_yd + 1 * p.rush_td
            + 6 * p.reception + 101 * p.rec_yd
            + p.bonus_pass_yd + p.bonus_rush_yd + p.bonus_rec_yd)
    assert abs(got - want) < 1e-9


def test_complete_profile_requires_provenance():
    for p in scoring.PROFILES.values():
        if p.complete:
            assert p.verified
            assert p.source
