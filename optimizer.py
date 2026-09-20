"""
Issue #5: FanDuel lineup optimizer.

Strictly separated from projection code. This module reads the two projection
tables that live_projection.py and team_defense.py already wrote and does
nothing but select players. It never computes or adjusts a projection.

Roster, FanDuel NFL Classic:
    1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DEF, 9 spots, $60,000 cap.

Objective for v1: maximize the sum of mean projections subject to the legal
roster and cap constraints. Nothing else. No ownership, no correlation, no
ceiling weighting.

ELIGIBILITY, per the Issue #5 policy:
  required   mapping_status == "ok"
  excluded   game status O (Out) and D (Doubtful)
  eligible   Q (Questionable), but flagged for pre-lock review
  excluded   no_history, inactive_or_practice_squad, position_mismatch,
             unresolved, because none of them carries a validated projection
  excluded   anyone on an official inactive list supplied via --inactives

No injury haircut is applied. Eligibility and mean projection are kept as
separate concerns, so a Questionable player carries his full projection and a
visible flag rather than an invented discount.

Alternate lineups come from no-good cuts: after a solution is found, a
constraint forbidding that exact set of nine is added and the problem is
re-solved. Each lineup therefore differs from every earlier one by at least
one player.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import polars as pl
from scipy.optimize import milp, LinearConstraint, Bounds

CAP = 60_000
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DEF": 1}
FLEX_POSITIONS = ("RB", "WR", "TE")
ROSTER_SIZE = 9
EXCLUDED_STATUS = {"O", "D", "IR", "NA"}
FLAGGED_STATUS = {"Q", "GTD"}


def load_pool(skill_csv="tables/live_projections.csv",
              def_csv="tables/live_def_projections.csv",
              inactives=None):
    """Combine projection tables; return excluded rows so nothing disappears."""
    s = pl.read_csv(skill_csv, infer_schema_length=10_000)
    d = pl.read_csv(def_csv, infer_schema_length=10_000)

    s = s.select([
        pl.col("fanduel_id").cast(pl.Utf8), "player", "position", "salary",
        "team", "opponent", "game",
        pl.col("v3_projection").alias("projection"),
        pl.col("injury_indicator").cast(pl.Utf8).fill_null("").alias("status"),
        pl.col("injury_details").cast(pl.Utf8).fill_null("").alias("status_detail"),
        "mapping_status",
    ])
    d = d.select([
        pl.col("fanduel_id").cast(pl.Utf8),
        pl.col("defense").alias("player"),
        pl.lit("DEF").alias("position"), "salary", "team", "opponent", "game",
        pl.col("def_projection").alias("projection"),
        pl.col("injury_indicator").cast(pl.Utf8).fill_null("").alias("status"),
        pl.lit("").alias("status_detail"),
        "mapping_status",
    ])
    pool = pl.concat([s, d], how="vertical")

    inactive_ids = set()
    if inactives and os.path.exists(inactives):
        inactive_ids = {ln.strip() for ln in open(inactives) if ln.strip()}

    pool = pool.with_columns([
        pl.when(pl.col("mapping_status") != "ok")
          .then(pl.lit("no validated projection"))
          .when(pl.col("fanduel_id").is_in(list(inactive_ids)) if inactive_ids else pl.lit(False))
          .then(pl.lit("official inactive"))
          .when(pl.col("status").is_in(list(EXCLUDED_STATUS)))
          .then(pl.lit("game status excluded"))
          .when(pl.col("projection").is_null())
          .then(pl.lit("no projection"))
          .otherwise(pl.lit("eligible"))
          .alias("eligibility"),
    ]).with_columns([
        pl.col("status").is_in(list(FLAGGED_STATUS)).alias("flagged"),
        (pl.col("projection") / (pl.col("salary") / 1000.0)).round(3).alias("pts_per_1k"),
    ])

    eligible = pool.filter(pl.col("eligibility") == "eligible")
    excluded = pool.filter(pl.col("eligibility") != "eligible")
    return pool, eligible, excluded


def _solve(elig: pl.DataFrame, banned: list[set[int]]):
    """One MILP solve with player binaries plus RB/WR/TE flex indicators."""
    n = len(elig)
    pos = elig["position"].to_list()
    sal = np.array(elig["salary"].to_list(), float)
    proj = np.array(elig["projection"].to_list(), float)

    nf = len(FLEX_POSITIONS)
    N = n + nf
    c = np.zeros(N)
    c[:n] = -proj

    A, lo, hi = [], [], []

    r = np.zeros(N); r[:n] = sal
    A.append(r); lo.append(0.0); hi.append(float(CAP))

    r = np.zeros(N); r[:n] = 1.0
    A.append(r); lo.append(float(ROSTER_SIZE)); hi.append(float(ROSTER_SIZE))

    r = np.zeros(N); r[n:] = 1.0
    A.append(r); lo.append(1.0); hi.append(1.0)

    for p in ("QB", "RB", "WR", "TE", "DEF"):
        r = np.zeros(N)
        for i in range(n):
            if pos[i] == p:
                r[i] = 1.0
        if p in FLEX_POSITIONS:
            r[n + FLEX_POSITIONS.index(p)] = -1.0
        A.append(r); lo.append(float(SLOTS[p])); hi.append(float(SLOTS[p]))

    for prev in banned:
        r = np.zeros(N)
        for i in prev:
            r[i] = 1.0
        A.append(r); lo.append(-np.inf); hi.append(float(ROSTER_SIZE - 1))

    res = milp(c=c,
               constraints=LinearConstraint(np.array(A), np.array(lo), np.array(hi)),
               integrality=np.ones(N),
               bounds=Bounds(0, 1))
    if not res.success:
        return None
    x = np.round(res.x[:n]).astype(int)
    chosen = set(np.nonzero(x)[0].tolist())
    flex = FLEX_POSITIONS[int(np.argmax(np.round(res.x[n:])))]
    return chosen, flex


def optimize(elig: pl.DataFrame, n_lineups=10):
    banned, out = [], []
    for k in range(n_lineups):
        sol = _solve(elig, banned)
        if sol is None:
            break
        idx, flex = sol
        banned.append(idx)
        rows = elig[sorted(idx)]
        out.append({
            "rank": k + 1,
            "flex_position": flex,
            "total_salary": int(rows["salary"].sum()),
            "salary_remaining": int(CAP - rows["salary"].sum()),
            "projected_points": round(float(rows["projection"].sum()), 2),
            "players": rows.sort(
                ["position", "projection"], descending=[False, True]),
            "flagged": rows.filter(pl.col("flagged"))["player"].to_list(),
        })
    return out


def validate(lineup, elig_ids: set[str]):
    """Every legality assertion from Issue #5. Raises on any violation."""
    p = lineup["players"]
    errs = []
    if lineup["total_salary"] > CAP:
        errs.append(f"salary {lineup['total_salary']} exceeds cap")
    if len(p) != ROSTER_SIZE:
        errs.append(f"{len(p)} roster spots, expected {ROSTER_SIZE}")
    if p["fanduel_id"].n_unique() != len(p):
        errs.append("duplicate player")
    counts = {r["position"]: r["len"] for r in p.group_by("position").len().to_dicts()}
    flex = lineup["flex_position"]
    for pos, base in SLOTS.items():
        want = base + (1 if pos == flex else 0)
        if counts.get(pos, 0) != want:
            errs.append(f"{pos}: {counts.get(pos, 0)} selected, expected {want}")
    if flex not in FLEX_POSITIONS:
        errs.append(f"illegal flex position {flex}")
    if sum(counts.get(x, 0) for x in FLEX_POSITIONS) != sum(
            SLOTS[x] for x in FLEX_POSITIONS) + 1:
        errs.append("flex count is not exactly one")
    bad = p.filter(pl.col("status").is_in(list(EXCLUDED_STATUS)))
    if len(bad):
        errs.append(f"excluded status selected: {bad['player'].to_list()}")
    off = [i for i in p["fanduel_id"].to_list() if i not in elig_ids]
    if off:
        errs.append(f"players not in the eligible slate pool: {off}")
    if errs:
        raise AssertionError("; ".join(errs))
    return True


def show(L):
    print(f"\nOPTIMAL LINEUP   flex = {L['flex_position']}")
    print(f"  {'pos':4s}{'player':26s}{'team':5s}{'opp':5s}{'salary':>7s}"
          f"{'proj':>7s}{'/$1k':>6s}  status")
    for r in L["players"].to_dicts():
        st = r["status"] or ""
        print(f"  {r['position']:4s}{r['player'][:25]:26s}{r['team']:5s}{r['opponent']:5s}"
              f"{r['salary']:7d}{r['projection']:7.2f}{r['pts_per_1k']:6.2f}  {st}")
    print(f"  {'':4s}{'TOTAL':26s}{'':10s}{L['total_salary']:7d}{L['projected_points']:7.2f}")
    print("  salary remaining: $" + f"{L['salary_remaining']:,}")
    if L["flagged"]:
        print(f"  FLAGGED for pre-lock review: {', '.join(L['flagged'])}")


def main():
    ap = argparse.ArgumentParser(description="FanDuel lineup optimizer (v1)")
    ap.add_argument("--skill", default="tables/live_projections.csv")
    ap.add_argument("--def", dest="def_csv", default="tables/live_def_projections.csv")
    ap.add_argument("--inactives", help="file of FanDuel ids, one per line")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()

    pool, elig, excl = load_pool(a.skill, a.def_csv, a.inactives)
    print(f"pool {len(pool)}   eligible {len(elig)}   excluded {len(excl)}")
    print(elig.group_by("position").agg([
        pl.len().alias("n"), pl.col("flagged").sum().alias("flagged")]).sort("position"))

    lineups = optimize(elig, a.n)
    ids = set(elig["fanduel_id"].to_list())
    for L in lineups:
        validate(L, ids)
    print(f"\n{len(lineups)} distinct legal lineups, all validated")

    os.makedirs(a.out, exist_ok=True)
    payload = {
        "cap": CAP,
        "lineups": [{k: (v.to_dicts() if isinstance(v, pl.DataFrame) else v)
                     for k, v in L.items()} for L in lineups],
        "pool": pool.to_dicts(),
    }
    with open(f"{a.out}/optimizer_output.json", "w") as f:
        json.dump(payload, f, indent=2)

    show(lineups[0])
    print("\nALTERNATES")
    print(f"  {'#':>2s} {'salary':>7s} {'left':>6s} {'proj':>7s}  flex  differs from optimal by")
    base = {r["fanduel_id"] for r in lineups[0]["players"].to_dicts()}
    for L in lineups[1:]:
        ids2 = {r["fanduel_id"] for r in L["players"].to_dicts()}
        swapped = len(base - ids2)
        print(f"  {L['rank']:2d} {L['total_salary']:7d} {L['salary_remaining']:6d} "
              f"{L['projected_points']:7.2f}  {L['flex_position']:4s}  {swapped} player(s)")


if __name__ == "__main__":
    main()
