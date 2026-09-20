"""DEF scoring guards for the verified Test Slate #1 contest rules."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import def_scoring


def test_profile_registered():
    assert "fanduel_def_test_slate_1" in def_scoring.DEF_PROFILES


def test_profile_is_verified_from_actual_contest_rules():
    p = def_scoring.get()
    assert p.verified
    assert p.source
    assert p.sack == 1.0
    assert p.interception == 2.0
    assert p.fumble_recovery == 2.0
    assert p.defensive_td == 6.0
    assert p.return_td == 6.0
    assert p.safety == 2.0
    assert p.blocked_kick == 2.0
    assert p.extra_point_return == 2.0


def test_every_category_has_coefficient_and_column():
    p = def_scoring.get()
    for label, (coef, col) in def_scoring.DEF_CATEGORIES.items():
        assert hasattr(p, coef), f"{label}: no field {coef}"
        assert col in def_scoring.STAT_COLS, f"{label}: {col} is not a built column"


def test_tiers_are_monotone_and_cover_everything():
    p = def_scoring.get()
    bounds = [b for b, _ in p.points_allowed_tiers]
    pts = [v for _, v in p.points_allowed_tiers]
    assert bounds == sorted(bounds), "tier bounds out of order"
    assert pts == sorted(pts, reverse=True), "tier payouts must not increase with points allowed"
    assert p.tier_points(0) == 10.0 and p.tier_points(99) == pts[-1]


def test_tier_boundaries_are_inclusive():
    p = def_scoring.get()
    assert p.tier_points(6) == 7.0 and p.tier_points(7) == 4.0
    assert p.tier_points(13) == 4.0 and p.tier_points(14) == 1.0
    assert p.tier_points(20) == 1.0 and p.tier_points(21) == 0.0
    assert p.tier_points(27) == 0.0 and p.tier_points(28) == -1.0


def test_score_expr_matches_hand_calculation():
    p = def_scoring.get()
    row = pl.DataFrame({c: [0.0] for c in def_scoring.STAT_COLS}).with_columns([
        pl.lit(4.0).alias("sacks"), pl.lit(2.0).alias("interceptions"),
        pl.lit(1.0).alias("fumble_recoveries"), pl.lit(1.0).alias("defensive_tds"),
        pl.lit(1.0).alias("extra_point_returns"),
        pl.lit(10.0).alias("points_allowed"),
    ])
    got = float(row.select(def_scoring.score_expr(p))["def_fpts"][0])
    assert abs(got - (4 * 1 + 2 * 2 + 1 * 2 + 1 * 6 + 1 * 2 + 4.0)) < 1e-9


def test_points_allowed_excludes_opponent_dst_scoring():
    row = pl.DataFrame({
        "opponent_total_score": [31.0],
        "opp_def_tds": [1.0],
        "opp_return_tds": [0.0],
        "opp_def_safeties": [1.0],
        "opp_def_2pt": [0.0],
    })
    got = float(row.select(def_scoring.points_allowed_expr())["points_allowed"][0])
    assert got == 23.0


def test_points_allowed_excludes_special_teams_and_defensive_xpr():
    row = pl.DataFrame({
        "opponent_total_score": [30.0],
        "opp_def_tds": [0.0],
        "opp_return_tds": [1.0],
        "opp_def_safeties": [0.0],
        "opp_def_2pt": [1.0],
    })
    got = float(row.select(def_scoring.points_allowed_expr())["points_allowed"][0])
    assert got == 22.0


def test_points_allowed_is_never_null_filled():
    import team_defense
    d = team_defense.build_targets([2026])
    future = d.filter(pl.col("week") >= 3)
    assert len(future) > 0
    assert future["points_allowed"].null_count() == len(future)
    assert future["def_fpts"].null_count() == len(future)
