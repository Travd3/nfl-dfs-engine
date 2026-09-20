"""
Optimizer integrity tests.

Every legality rule in Issue #5 is asserted here against the real Test Slate #1
pool, not a fixture, so the tests fail if the live projection tables or the
slate change shape.
"""
import os, sys
import pytest
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optimizer as opt

SKILL = "tables/live_projections.csv"
DEF = "tables/live_def_projections.csv"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(SKILL) and os.path.exists(DEF)),
    reason="run live_projection.py and team_defense.py first")


@pytest.fixture(scope="module")
def solved():
    pool, elig, excl = opt.load_pool(SKILL, DEF)
    lineups = opt.optimize(elig, n_lineups=10)
    return pool, elig, excl, lineups


def test_ten_distinct_lineups(solved):
    _, _, _, L = solved
    assert len(L) == 10
    sigs = {frozenset(r["fanduel_id"] for r in x["players"].to_dicts()) for x in L}
    assert len(sigs) == 10, "alternate lineups are not distinct"


def test_all_lineups_validate(solved):
    _, elig, _, L = solved
    ids = set(elig["fanduel_id"].to_list())
    for x in L:
        assert opt.validate(x, ids)


def test_salary_cap(solved):
    _, _, _, L = solved
    for x in L:
        assert x["total_salary"] <= opt.CAP
        assert x["salary_remaining"] == opt.CAP - x["total_salary"]


def test_exactly_nine_spots(solved):
    _, _, _, L = solved
    for x in L:
        assert len(x["players"]) == 9


def test_position_counts_with_single_flex(solved):
    _, _, _, L = solved
    for x in L:
        c = {r["position"]: r["len"] for r in
             x["players"].group_by("position").len().to_dicts()}
        flex = x["flex_position"]
        assert c.get("QB", 0) == 1
        assert c.get("DEF", 0) == 1
        assert c.get("RB", 0) == 2 + (flex == "RB")
        assert c.get("WR", 0) == 3 + (flex == "WR")
        assert c.get("TE", 0) == 1 + (flex == "TE")
        assert flex in opt.FLEX_POSITIONS
        extra = sum(c.get(p, 0) for p in opt.FLEX_POSITIONS) - 6
        assert extra == 1, "there must be exactly one flex"


def test_no_duplicate_player(solved):
    _, _, _, L = solved
    for x in L:
        ids = x["players"]["fanduel_id"].to_list()
        assert len(set(ids)) == len(ids)


def test_no_excluded_status_selected(solved):
    _, _, _, L = solved
    for x in L:
        for r in x["players"].to_dicts():
            assert r["status"] not in opt.EXCLUDED_STATUS


def test_only_slate_players_selected(solved):
    pool, _, _, L = solved
    slate_ids = set(pool["fanduel_id"].to_list())
    for x in L:
        for r in x["players"].to_dicts():
            assert r["fanduel_id"] in slate_ids


def test_only_validated_projections_are_eligible(solved):
    _, elig, _, _ = solved
    assert (elig["mapping_status"] == "ok").all()
    assert elig["projection"].null_count() == 0


def test_excluded_rows_carry_a_reason(solved):
    _, _, excl, _ = solved
    assert len(excl) > 0
    assert excl["eligibility"].null_count() == 0
    assert set(excl["eligibility"].unique().to_list()) <= {
        "no validated projection", "game status excluded",
        "official inactive", "no projection"}


def test_questionable_players_stay_eligible_but_flagged(solved):
    _, elig, _, _ = solved
    q = elig.filter(pl.col("status") == "Q")
    if len(q):
        assert q["flagged"].all()
        assert q["projection"].null_count() == 0


def test_optimal_is_the_best_of_the_alternates(solved):
    _, _, _, L = solved
    tot = [x["projected_points"] for x in L]
    assert tot == sorted(tot, reverse=True)


def test_validate_rejects_an_illegal_lineup(solved):
    _, elig, _, L = solved
    ids = set(elig["fanduel_id"].to_list())
    broken = dict(L[0])
    broken["players"] = L[0]["players"].head(8)
    with pytest.raises(AssertionError):
        opt.validate(broken, ids)
