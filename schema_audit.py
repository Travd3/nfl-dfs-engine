"""
Historical schema audit.

Run BEFORE extending the evaluation window. Checks that every field the frozen
V3 pipeline depends on is present AND actually populated in each season. A
column that exists but is entirely null or entirely zero is worse than a
missing one, because it degrades the model silently instead of raising.

Reports only. Changes nothing.
"""
import polars as pl
import nflreadpy as nfl

WEEKLY_REQUIRED = [
    "season", "week", "player_id", "team", "opponent_team", "season_type",
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
    "special_teams_tds", "fumble_recovery_tds",
    "attempts", "carries", "targets",
]
ROSTER_REQUIRED = ["season", "week", "team", "position", "status", "gsis_id"]
SCHED_REQUIRED = ["game_id", "home_team", "away_team", "spread_line",
                  "total_line", "home_rest", "away_rest"]
PBP_REQUIRED = ["season", "week", "play_type", "posteam", "play_id", "game_id"]


def _status(df, cols, label):
    out = {}
    for c in cols:
        if c not in df.columns:
            out[c] = "MISSING"
            continue
        s = df[c]
        if s.null_count() == len(s):
            out[c] = "ALL NULL"
        elif s.dtype.is_numeric() and float(s.fill_null(0).abs().sum()) == 0.0:
            out[c] = "ALL ZERO"
        else:
            nn = 1 - s.null_count() / len(s)
            out[c] = f"{nn:.2f}"
    return out


def audit(seasons):
    problems = []
    print(f"{'season':7s} {'weekly':>8s} {'roster':>8s} {'sched':>8s} {'pbp':>8s}   issues")
    for s in seasons:
        issues = []
        try:
            w = nfl.load_player_stats([s]).filter(pl.col("season_type") == "REG")
            wst = _status(w, WEEKLY_REQUIRED, "weekly")
            nw = len(w)
        except Exception as e:
            wst, nw = {c: "LOAD FAIL" for c in WEEKLY_REQUIRED}, 0
            issues.append(f"weekly load: {str(e)[:40]}")
        try:
            r = nfl.load_rosters_weekly([s])
            rst, nr = _status(r, ROSTER_REQUIRED, "roster"), len(r)
        except Exception as e:
            rst, nr = {c: "LOAD FAIL" for c in ROSTER_REQUIRED}, 0
            issues.append(f"roster load: {str(e)[:40]}")
        sc = nfl.load_schedules().filter(pl.col("season") == s)
        sst, ns = _status(sc, SCHED_REQUIRED, "sched"), len(sc)
        try:
            p = nfl.load_pbp([s])
            pst, npb = _status(p, PBP_REQUIRED, "pbp"), len(p)
        except Exception as e:
            pst, npb = {c: "LOAD FAIL" for c in PBP_REQUIRED}, 0
            issues.append(f"pbp load: {str(e)[:40]}")

        for label, st in (("weekly", wst), ("roster", rst), ("sched", sst), ("pbp", pst)):
            for c, v in st.items():
                if v in ("MISSING", "ALL NULL", "ALL ZERO", "LOAD FAIL"):
                    issues.append(f"{label}.{c}={v}")
                elif v not in ("MISSING",) and not v.startswith("LOAD"):
                    try:
                        if float(v) < 0.50:
                            issues.append(f"{label}.{c} only {v} populated")
                    except ValueError:
                        pass

        if issues:
            problems.append((s, issues))
        print(f"{s:<7d} {nw:8d} {nr:8d} {ns:8d} {npb:8d}   "
              f"{'; '.join(issues[:3]) if issues else 'ok'}")
        if len(issues) > 3:
            for extra in issues[3:]:
                print(f"{'':39s}{extra}")

    print()
    if problems:
        print("SEASONS WITH ISSUES:", [s for s, _ in problems])
    else:
        print("no schema issues across the requested window")
    return problems


if __name__ == "__main__":
    import sys
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1999
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 2017
    audit(range(lo, hi + 1))
