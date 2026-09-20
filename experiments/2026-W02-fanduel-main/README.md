# Test Slate #1: FanDuel NFL Main Slate, 2026-09-20

Status: **PRE-LOCK SKILL-PLAYER SNAPSHOT**

Contest slate:
- NFL Week 2
- 13 FanDuel main-slate games
- excludes DET@BUF (Thursday), IND@KC (Sunday night), NYG@LA (Monday night)
- skill positions in this snapshot: QB/RB/WR/TE
- DEF is tracked separately in Issue #4

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
