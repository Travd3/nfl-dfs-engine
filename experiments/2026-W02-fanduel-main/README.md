# Test Slate #1: FanDuel NFL Main Slate, 2026-09-20

Status: **PRE-LOCK SKILL + DEF SNAPSHOT**

Contest slate:
- NFL Week 2
- 13 FanDuel main-slate games
- excludes DET@BUF (Thursday), IND@KC (Sunday night), NYG@LA (Monday night)
- skill positions in this snapshot: QB/RB/WR/TE
- DEF projections are included below and Issue #4 is complete

Generation:
- generated UTC: 2026-09-20T00:38:15.968936+00:00
- nflreadpy: 0.1.5
- V3 training cutoff: 2026 Week 1, the latest fully completed week
- PBP max week visible at generation: 2
- weekly stats max week visible at generation: 2
- roster max week visible at generation: 2
- same-week Week 2 data is excluded from both training and as-of features

Mapping summary:
- FanDuel QB/RB/WR/TE rows: 626
- mapped + projected: 332
- mapped, insufficient history: 73
- identified inactive/reserve/practice-squad: 201
- FanDuel/nflverse position mismatch: 10
- unresolved identity: 10
- games represented: 13

The 53% mapped+projected rate is **not** an identity-mapping accuracy measure. Many FanDuel rows are reserve/practice-squad players or active rookies below V3's frozen minimum-history threshold. Only 10 of 626 rows remained unresolved.

Top V3 mean projections at snapshot time:

| Pos | Player | Salary | Projection | Status |
| --- | --- | ---: | ---: | --- |
| QB | Joe Burrow | 8200 | 19.17 | Q |
| QB | Trevor Lawrence | 7800 | 18.87 | |
| QB | Dak Prescott | 8000 | 18.12 | |
| RB | Bijan Robinson | 8900 | 19.96 | |
| RB | Derrick Henry | 8700 | 16.61 | |
| RB | Christian McCaffrey | 9000 | 15.71 | |
| WR | Ja'Marr Chase | 8800 | 15.15 | |
| WR | Chris Olave | 7900 | 14.07 | Q |
| WR | Zay Flowers | 7600 | 12.66 | D |
| TE | Trey McBride | 7500 | 12.05 | |
| TE | Brock Bowers | 6700 | 11.48 | D |
| TE | George Kittle | 5900 | 9.28 | |

Output integrity:
- zero GSIS IDs mapped to multiple FanDuel rows
- zero mapped rows had a team disagreement between FanDuel and nflverse
- FanDuel FPPG is not in V3's feature list and was not used as a projection input

Artifacts from the run were returned outside the public repository:
- `live_projections.csv`
- `live_vintage.json`

SHA-256:
- `live_projections.csv`: `8ebf12feb8723adf388cc586f96f75f486aec4b64b74653b5f5c671c8d53f990`
- `live_vintage.json`: `9ee6a18177e37032d5bd956fafb98aa3e31983aa16bc878cb4585a257c0722e8`
- submitted `live_projection.py`: `b77f41a3cd2e272bfe071888087254a49bd71ce3db6e01a91471f64d3f6e8ee7`

Important optimizer policy is not encoded in this snapshot. Injury/game-status eligibility is applied downstream so the projection model remains a pure historical mean model.


## DEF baseline snapshot

The corrected FanDuel DEF model was rerun after fixing the points-allowed target to match the contest rule note and adding Extra Point Return +2.

Rolling-origin historical result, n=7,649 team-games:

```text
arm        RMSE      MAE   Spearman     bias   calib slope   intercept
naive    5.7547   4.4452     0.1388   +0.0359        0.491      +2.977
model    5.5008   4.2347     0.3127   -0.0882        0.957      +0.336

RMSE naive -> model +0.2540
95% CI [+0.2136, +0.2939]
P=1.000

MAE naive -> model +0.2114
95% CI [+0.1746, +0.2488]
P=1.000
```

The model improved RMSE in all 14 reported test seasons, 2012-2025.

Live Week 2 generation:
- generated UTC: 2026-09-20T00:58:52.964061+00:00
- trained through: 2026 Week 1 only
- FanDuel defenses: 26
- mapped: 26
- unmapped: 0

Top live DEF mean projections:

| Defense | Salary | Projection | Pts/$1k |
| --- | ---: | ---: | ---: |
| Tampa Bay | 4800 | 8.93 | 1.86 |
| Baltimore | 4600 | 7.54 | 1.64 |
| LA Chargers | 4500 | 7.41 | 1.65 |
| Philadelphia | 4900 | 7.24 | 1.48 |
| Chicago | 3700 | 7.21 | 1.95 |
| Carolina | 3300 | 6.44 | 1.95 |
| Arizona | 3200 | 6.04 | 1.89 |

Archived:
- `live_def_projections.csv`
- `live_def_vintage.json`

Scoring provenance note:
- all non-zero DEF categories and the FanDuel points-allowed formula are visible in the actual contest screenshots
- the UI omits the zero-valued 21-27 points-allowed row; the model uses 0 for that band, consistent with long-standing FanDuel scoring references
