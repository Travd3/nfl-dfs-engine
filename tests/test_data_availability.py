"""
Data availability canaries.

Every claim about what the free stack provides is asserted here rather than
written in prose. These tests encode reality as of the last run. When an
upstream source changes, a test fails and the spec gets corrected instead of
quietly drifting.

Some tests assert that something is UNAVAILABLE. Those are intentional. If
participation data returns for the current season, that test fails, and the
failure is the notification.

Run:  pytest tests/test_data_availability.py -v
"""
import os, sys
import pytest
import polars as pl
import nflreadpy as nfl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import CHART_LAG

SEASON = nfl.get_current_season()
PRIOR = SEASON - 1


# --------------------------------------------------------------- injuries
def test_injuries_load_for_current_season():
    """DATA_SOURCES.md flagged this as unverified. It resolves: data flows."""
    inj = nfl.load_injuries([SEASON])
    assert len(inj) > 0, "no current-season injury rows"
    assert inj["team"].n_unique() == 32, "injury feed is missing teams"


def test_injuries_have_no_timestamp_column():
    """
    The spec and DATA_SOURCES both require an as-of timestamp on live
    features. The injury table does not carry one. DATA_SOURCES lists
    `date_modified`, which does not exist.

    Finest granularity is (season, week), so Wednesday practice and Friday
    game status are indistinguishable. Stamp at pull time yourself.

    If nflverse ever adds a timestamp, this test fails and the workaround
    can be removed.
    """
    cols = nfl.load_injuries([SEASON]).columns
    for candidate in ("date_modified", "last_modified", "updated_at", "timestamp"):
        assert candidate not in cols, f"{candidate} now exists; revisit the archive workaround"


def test_injury_game_status_coverage_is_low():
    """
    What is live is the PRACTICE report, not the game-status report.
    `report_status` (Out/Doubtful/Questionable) is mostly null, while
    `practice_status` is populated. The Q/D/O designation is the higher
    value field for DFS and it is the one that is missing.

    Fails upward: if coverage improves past 50%, revisit the plan to source
    game status elsewhere.
    """
    inj = nfl.load_injuries([SEASON])
    cov = 1 - inj["report_status"].null_count() / len(inj)
    prac = 1 - inj["practice_status"].null_count() / len(inj)
    print(f"\n  report_status coverage {cov:.1%}, practice_status {prac:.1%}")
    assert prac > 0.80, "practice status coverage collapsed"
    assert cov < 0.50, "game-status coverage improved; revisit sourcing"


# ---------------------------------------------------- participation / personnel
def test_participation_unavailable_for_current_season():
    """
    Personnel groupings (11/12/21) cannot be a live feature. The loader
    range ends at the prior season. This test failing is GOOD NEWS and means
    personnel can be reconsidered as a live input.
    """
    with pytest.raises(Exception):
        nfl.load_participation([SEASON])


def test_participation_available_historically():
    p = nfl.load_participation([PRIOR])
    assert len(p) > 0 and "offense_personnel" in p.columns


# --------------------------------------------------------------- charting lag
def test_ftn_charting_lag_matches_live_freshness():
    """
    Convert the observed live source gap into the lag the feature builder
    should declare for the next slate.

    If PBP and FTN are both through week w before week w+1 locks, chart data
    can use lag=1. If FTN trails PBP by one completed week, lag=2 is required.
    Any mismatch fails so the source-vintage assumption gets reviewed.
    """
    pbp_wk = nfl.load_pbp([SEASON])["week"].max()
    ftn_wk = nfl.load_ftn_charting([SEASON])["week"].max()
    print(f"\n  pbp through week {pbp_wk}, ftn through week {ftn_wk}, declared lag {CHART_LAG}")
    assert ftn_wk <= pbp_wk, "charting ahead of play-by-play, which should be impossible"
    observed_required_lag = int(pbp_wk - ftn_wk) + 1
    assert CHART_LAG == observed_required_lag, (
        f"live source gap implies CHART_LAG={observed_required_lag}, "
        f"but features.py declares {CHART_LAG}"
    )


def test_ftn_condition_fields_present():
    cols = nfl.load_ftn_charting([PRIOR]).columns
    for c in ("is_motion", "is_play_action", "is_rpo", "is_screen_pass",
              "qb_location", "n_defense_box", "n_blitzers"):
        assert c in cols, f"FTN dropped {c}"


def test_qb_location_is_three_way():
    """Alignment must stay U/S/P. Collapsing pistol into shotgun was defect C9."""
    v = set(nfl.load_ftn_charting([PRIOR])["qb_location"].unique().to_list())
    assert {"U", "S", "P"} <= v, f"alignment codes changed: {v}"


# --------------------------------------------------------------- vegas / schedule
def test_vegas_lines_published_ahead_of_kickoff():
    s = nfl.load_schedules().filter(pl.col("season") == SEASON)
    played = s.filter(pl.col("result").is_not_null())["week"].max() or 0
    future = s.filter((pl.col("week") > played) & pl.col("total_line").is_not_null())
    assert len(future) > 0, "no forward-week lines; Vegas features are not live"


def test_post_game_columns_are_distinguishable():
    """
    `total` is the realized combined score and sits one column from
    `total_line`. Any feature list that grabs `total` leaks the outcome.
    """
    cols = nfl.load_schedules().columns
    assert "total" in cols and "total_line" in cols
    assert "result" in cols


# --------------------------------------------------------------- reproducibility
def test_record_data_vintage():
    """
    Section 7 wants the backtest reproducible. nflverse restates history, so
    results drift unless the vintage is recorded. This writes it out rather
    than asserting anything.
    """
    import json, datetime
    vintage = {
        "pulled_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "nflreadpy": nflreadpy_version(),
        "season": int(SEASON),
        "pbp_max_week": int(nfl.load_pbp([SEASON])["week"].max()),
        "ftn_max_week": int(nfl.load_ftn_charting([SEASON])["week"].max()),
    }
    print("\n  vintage:", json.dumps(vintage))
    with open("data_vintage.json", "w") as f:
        json.dump(vintage, f, indent=2)


def nflreadpy_version():
    import nflreadpy as m
    return getattr(m, "__version__", "unknown")
