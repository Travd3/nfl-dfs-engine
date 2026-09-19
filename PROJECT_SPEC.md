# NFL DFS Engine Project Specification

## 1. Product

### Product 1
NFL salary-cap DFS projection, simulation, and lineup engine.

Initial platform targets:
- DraftKings Classic
- FanDuel NFL salary-cap contests

The football projection layer must be platform-agnostic. Platform scoring, salary cap, roster slots, and contest rules belong in explicit contest/scoring configuration rather than being hardcoded into the model.

### Scoring shape
The current historical research tables were originally built with full-PPR skill-player scoring and therefore do **not** yet represent the FanDuel Test Slate #1 scoring shown in the live contest.

Before a projection is used for the FanDuel test contest, scoring must be rebuilt from a FanDuel configuration, including 0.5 PPR, yardage bonuses, turnovers, two-point conversions, and all applicable defense/special-teams scoring. Bonuses and other game-level rules belong at player-game or simulation level rather than per-opportunity features.

### Long-term system

```text
RAW NFL DATA
    ->
FEATURE ENGINE
    ->
PLAYER / ROLE / TEAM / OPPONENT FEATURES
    ->
PROJECTION MODEL
    ->
UNCERTAINTY DISTRIBUTION
    ->
GAME SIMULATION
    ->
CONTEST SIMULATION
    ->
OWNERSHIP / LEVERAGE
    ->
PORTFOLIO OPTIMIZER
    ->
EXPLANATION LAYER
```

The natural-language layer explains model output. It does not generate the model output.

---

## 2. Current data policy

### Confirmed core sources

- nflverse / nflreadpy play-by-play
- nflverse schedules and Vegas lines
- FTN charting subset through nflverse
- Next Gen Stats where useful and available
- player/team weekly stats
- snap counts
- rosters and depth-chart data when verified
- DraftKings official lobby CSV for salaries and player IDs when live slate ingestion is implemented

### Live DFS input decisions

#### Injuries
**Status: PARTIALLY VERIFIED. PRODUCTION FOR PRACTICE STATUS, NOT SUFFICIENT ALONE FOR GAME STATUS.**

A direct 2026 `nflreadpy==0.1.5` pull on 2026-09-18 returned current-season injury rows for Weeks 1 and 2 across all 32 teams. This proves the live release is flowing even though the nflverse update-schedule prose still says the old injury source died after 2024.

The important limitation is coverage:
- `practice_status` is populated broadly enough to use as a live practice-participation feature.
- `report_status` (Out/Doubtful/Questionable) was populated on only about 16% of the current rows in the verified pull, so nflverse cannot be treated as the sole Sunday game-status source.
- the actual Python table contained no usable source-update timestamp. The current nflverse R dictionary still documents `date_modified`, so schema reality and documentation conflict.

Production rule:
- stamp every injury pull with our own retrieval timestamp and archive snapshots append-only
- use nflverse for structured practice status
- use roster status for IR/PUP/NFI
- add a separate verified source for final Out/Doubtful/Questionable designations and official inactives
- exclude TNF injury features from historical backtests until pre-lock snapshots exist, because a weekly row may reflect information added after Thursday kickoff

These claims are guarded by `tests/test_data_availability.py`.

Sleeper may be useful as a same-day overlay, but its public API documentation says free use is for non-commercial purposes and commercial use requires contacting Sleeper. Do not make Sleeper a commercial product dependency without permission.

#### Platform salaries and IDs
**Status: PRODUCTION INPUT WHEN FROM THE OFFICIAL PLATFORM EXPORT.**

DraftKings:
- use the official lobby/lineup-template CSV supplied to the logged-in user
- do not make the undocumented draftables API a production dependency

FanDuel:
- use the official entries upload-template CSV for the target slate
- `ingest_fanduel.py` extracts the player pool while dropping user-specific entry/contest identifiers
- raw upload templates remain gitignored

The official platform file is the source of truth for slate membership, salary, roster position, platform player ID, and team/game metadata.

#### Weather
**Status: EXPERIMENTAL FOR MODEL WEIGHT; LIVE NWS FEED APPROVED FOR CURRENT-SLATE DISPLAY.**

Use:
- nflverse schedules for roof/surface/game metadata and Vegas lines
- NWS / NOAA `api.weather.gov` for current outdoor-stadium forecast data

NWS is a clean live source, but its forecast endpoint is not a historical archive of what a forecast said before a past slate locked. Therefore weather cannot receive production projection weight under the project's leakage rule until we either:
- accumulate our own timestamped forecast archive, or
- build a defensible historical forecast/reanalysis pipeline and prove it represents pre-lock information.

Observed final weather must not be substituted for historical pre-lock forecasts. Dome/closed-roof games receive no weather adjustment.

#### Ownership
**Status: MODEL INTERNALLY. NO OFFICIAL FREE PRE-LOCK FEED.**

Pre-lock ownership should be estimated internally from features such as salary, projection, value, implied team total, stack context, and injury-driven role changes.

Post-lock DraftKings GameCenter CSV exports provide realized `% Drafted` and can be used as clean training/calibration data. DraftKings documentation states these exports can be downloaded for contests the user entered and also for viewable contests the user did not enter. Downloads remain available for a limited period after contests end, so archive selected contest CSVs weekly.

Public article ownership percentages may be stored as timestamped calibration snapshots, but paid or paywalled ownership tables must not be scraped into the product.

### Historical-only or restricted-use sources

Participation/personnel data from recent seasons may be useful for historical research or priors, but it is not a dependable live current-season input.

### Unsupported free-data claims

Do not present the following as measured live features unless a verified data source is later added:

- routes run
- target per route run
- man/zone splits
- Cover-1/Cover-3/etc. performance
- slot/wide/inline alignment
- shadow coverage
- OL/DL grades
- true called zone/gap blocking concept
- proprietary pressure rate
- true receiver first-read share
- live 11/12/21 personnel

---

## 3. Analytical channels

The model must keep fundamentally different opportunity types separate.

Current channels:

1. `qb_pass`
2. `qb_rush`
3. `rb_rush`
4. `rb_rec`
5. `wrte_rec`

No model is allowed to pool RB carries with RB targets for scheme sensitivity or points-per-opportunity calculations.

---

## 4. Scheme status

### Team tendencies
Team tendencies remain useful descriptive football features and may be tested as predictive inputs.

Examples include:

- motion
- play action
- RPO
- screen rate
- no huddle
- under center
- shotgun
- pistol
- box count
- blitz count
- pass-rusher count

### Player-level scheme sensitivity
Current status: **DESCRIPTIVE / REJECTED FOR PROJECTION WEIGHT**

Current walk-forward testing found player-level scheme sensitivity worse than the baseline out of sample.

Therefore:

- projection weight = 0
- may remain on research cards if clearly labeled
- may be revisited only after a materially different estimator, better data, or new evidence
- no UI may imply that the model has proven player-specific scheme fit to be predictive

### Clustering
K-means or named scheme clusters may be used for visualization only.

They are not predictive features.

---

## 5. Current modeling standard

Every proposed feature must be classified as one of:

### PRODUCTION
Validated enough to affect projections.

### EXPERIMENTAL
Worth testing, but not trusted enough to materially affect projections.

### DESCRIPTIVE
Useful for explaining football, but not demonstrated to improve prediction.

### REJECTED
Unsupported, redundant, misleading, unavailable, or harmful.

A feature must earn production weight through held-out testing.

---

## 6. Leakage rules

Historical backtests must reproduce what was knowable before each slate locked.

### Current declared source lags in code

- play-by-play history: 1 week
- FTN charting history: 1 week as of the 2026-09-19 source vintage

The FTN lag is not a permanent constant. On 2026-09-18 FTN trailed PBP by one completed week, while on 2026-09-19 the live canary observed both sources through Week 2 before the Week 3 Sunday lock. The canary now converts the observed source gap into the required declared lag and fails on any mismatch.

### Forbidden leakage

Do not allow post-game columns or same-game outcomes into feature construction.

Examples include:

- realized score
- realized result
- post-game totals
- same-week player production
- future injury outcomes
- final-season aggregates that would not have existed at lock

All as-of joins should preserve enough source timing information to be audited.

---

## 7. Current backtest result

### Scheme pass
The first walk-forward pass used prior seasons to predict held-out 2024 and 2025 player-game-channel outcomes:

```text
arm                MAE
naive           4.1260
base            4.1254
team rates      4.1294
sensitivity     4.1532
both            4.1701
noise control   4.1216
```

Interpretation:
- Baseline V1 was effectively tied with naive persistence.
- Team scheme rates did not provide reliable improvement.
- Player sensitivity was harmful out of sample.
- Scheme therefore has zero projection weight.

### Baseline V2 research pass
Claude's second walk-forward pass tested structural alternatives:

```text
arm     MAE      RMSE
naive   4.1260   5.5998
v1      4.1254   5.5862
v2a     4.1231   5.5774
v2b     4.1318   5.5706
v2c     4.1421   5.5653
v2d     4.1855   5.6394
v2e     3.9751   5.6873
```

The promising structural finding is that a **direct ridge model on fantasy points using the original feature set** reportedly reached RMSE 5.5615 versus 5.5998 for naive persistence, with a block-bootstrap delta of +0.0379 and reported 95% CI [+0.0133, +0.0624].

This is **PROMISING / NOT YET PRODUCTION**.

Reasons it is not yet production:
- the reported V3/direct-old model is not yet a standalone reproducible production file
- model choice was informed by performance on the same 2024-2025 evaluation seasons, so a confirmatory backtest is required
- evaluation is currently on player-game-channel rows, not final player-game fantasy totals
- the frame excludes player-games with zero opportunities, so availability/zero outcomes are not represented
- current historical targets still use the original scoring implementation, which does not match FanDuel Test Slate #1 and omits some full platform scoring details
- the direct model must be re-evaluated after scoring is made configurable

The main structural conclusion is still useful: independently fitting opportunities and points per opportunity and multiplying them appears to compound error relative to a direct points model.

The exact results must remain reproducible from committed code and should be confirmed on a broader rolling historical sample.

### Baseline V3 corrected-scoring development pass

The player-game builder now uses nflverse weekly player stats for common box-score categories, applies the actual FanDuel +3 yardage bonuses, includes two-point conversions, and carries proxies for the rare return/fumble-recovery TD categories.

Claude reported:

```text
FanDuel Test Slate #1 scoring, player-game grain, n=59,773

2018-2023 development block
  RMSE  naive 5.9541   v3 5.9251   delta +0.0288
  95% CI [+0.0008, +0.0623]
  MAE   naive 3.9132   v3 3.9365
  Spearman     0.6438      0.6413
  bias        +0.0741     -0.1098
  calibration slope/intercept:
      naive 0.922 / +0.352
      v3    1.043 / -0.115

2024-2025 reused/model-selection seasons
  RMSE  naive 5.7843   v3 5.7409   delta +0.0442
  95% CI [+0.0059, +0.0888]

ALL 2018-2025
  RMSE  naive 5.9120   v3 5.8794   delta +0.0330
  95% CI [+0.0086, +0.0598]
```

Interpretation:
- V3 still improves RMSE under the corrected FanDuel scoring target, but the margin is small.
- MAE and rank correlation favor naive persistence in the 2018-2023 block.
- V3 calibration is materially better than naive persistence.
- V3 improves zero-opportunity rows but is worse on played rows under the corrected target.
- The previous pre-bonus FanDuel metrics are void.

**2018-2023 is not a pristine confirmatory holdout anymore.** During this pass, the 2022 TE results exposed an extreme Taysom Hill extrapolation error and directly motivated the 0.5/99.5 training-support winsorization now used by V3. Once a test season changes the estimator, it becomes development evidence.

Therefore V3 remains **PREFERRED RESEARCH BASELINE / NOT YET CONFIRMED FOR PRODUCTION**.

A new untouched historical block is required before promotion. nflverse player stats and PBP are available back to 1999, while weekly rosters are available back to 2002, so V3 can be frozen and evaluated on an earlier rolling-origin block without using FTN charting.
---

## 8. Baseline model status

### Baseline V1
Role and efficiency were modeled separately, then multiplied. Held-out testing indicated that independently estimating the two components compounded variance.

### Baseline V2
A single direct ridge on fantasy points was better than the role × efficiency product. Added snap share, red-zone role, pace, rest, and venue did not improve it and remain **REJECTED** for projection weight.

### Baseline V3
`baseline_v3.py` implements the direct player-game ridge with:
- rolling-origin evaluation
- player-game targets
- zero-opportunity active-roster rows
- configurable platform scoring
- calibration and rank diagnostics
- clipping to training feature support for extreme role/position mismatches

Current status: **PREFERRED RESEARCH BASELINE / NOT YET CONFIRMED FOR PRODUCTION.**

The corrected FanDuel run still favors V3 on RMSE and calibration, but not on MAE or rank correlation. More importantly, 2022 influenced the estimator through the Taysom Hill clipping fix, so the 2018-2023 block can no longer be called untouched confirmation.

### Next validation step
Freeze V3 as currently implemented and extend the historical source window. Use an earlier untouched rolling-origin block, preferably 2010-2017 with prior seasons reserved only for training, if all required fields are available and semantically consistent.

Do not change V3 after seeing that fresh block and still call the same block confirmatory. Any estimator change triggered by those results resets the validation process.

### Availability hurdle model
Issue #2 remains a justified experiment, but is blocked until the fresh V3 confirmation is complete. The corrected target flipped the played/zero-row direction from the earlier pass, so the hurdle model should be tested rather than assumed superior.

---

## 9. Evaluation requirements

Any model change should report:

1. held-out RMSE as the primary interim point-projection metric
2. held-out MAE as a secondary diagnostic
3. rank correlation
4. calibration
5. bootstrap confidence interval for improvement
6. position/channel slices
7. early-season vs late-season performance
8. evidence of leakage
9. comparison against naive persistence
10. comparison against a noise-control feature set where relevant

Bootstrap should resample slate-weeks or another appropriately correlated block rather than treating every player-row as independent.

RMSE is only an interim metric for point projections. It is not the final tournament objective. Once outcome distributions exist, use proper distributional calibration/scoring and contest-level simulation metrics rather than selecting models on RMSE alone.

---

## 10. DFS roadmap

### Immediate
1. Freeze current V3 and run a genuinely fresh historical confirmation block using earlier seasons.
2. Keep FanDuel scoring rules verified but historical target completeness false until rare return/fumble-recovery semantics are exact or explicitly accepted as approximations.
3. Keep Issue #2 hurdle-model work blocked until V3's fresh confirmation is known.
4. Use the official FanDuel upload template already ingested for Test Slate #1.
5. Complete reliable final game-status/inactive sourcing.
6. Add a separate FanDuel DEF scoring/projection path before a complete 9-player lineup can be generated.

### After baseline improvement
4. Add defensive contextual features only if they pass held-out tests.
5. Build player outcome distributions, not only point estimates.
6. Build game-level simulation.
7. Add DraftKings bonuses inside simulation.
8. Build internal ownership model.
9. Build contest simulation.
10. Build portfolio/GPP optimizer.

### Tournament outputs eventually desired

- median projection
- floor / ceiling
- boom probability
- bust probability
- top 10% rate
- top 1% rate
- top 0.1% rate
- first-place rate
- cash rate
- expected ROI
- duplication estimate
- stack correlation
- bring-back correlation
- lineup exposure

Fixed score thresholds such as 142 or 155 may be shown as reference lines, but they are not true contest cash or smash probabilities.

---

## 11. Simulation direction

Long-term simulation should occur at the team/game/opportunity level rather than by adding arbitrary independent Gaussian shocks to player projections.

Preferred conceptual order:

```text
game scoring environment
    ->
team plays / pass-run mix
    ->
player opportunity allocation
    ->
player efficiency / scoring events
    ->
DraftKings scoring
    ->
lineup scores
    ->
contest ranks
```

Correlation should emerge from shared football events where possible.

Examples:

- QB and targeted WR are positively linked
- QB and opposing pass catcher can be positively linked through game environment
- RB and own DST can share positive game-script correlation
- RB1 and RB2 can compete for opportunities
- DST and opposing QB are negatively linked

---

## 12. Ownership

No official free ownership feed is assumed.

Preferred long-term path:

- train an internal ownership model
- use public snapshots only as calibration targets when legally available
- do not depend on scraping paid providers
- ownership model inputs may include salary, projection, value, implied team total, stack context, and public narrative proxies

---

## 13. AI collaboration workflow

### ChatGPT
Role: lead architect and synthesizer.

Responsibilities:

- maintain the project specification
- reconcile conflicting recommendations
- manage architecture and implementation order
- distinguish verified results from suggestions
- ensure code changes remain aligned with the product

### Claude
Role: quantitative and code red team.

Responsibilities:

- attack statistical assumptions
- find leakage
- identify false precision
- review code behavior
- propose and run held-out tests
- challenge features that sound plausible but lack predictive evidence

### Grok
Role: external data researcher and source auditor.

Responsibilities:

- verify current public/free data availability
- inspect field definitions and update cadence
- identify alternative public sources
- flag licensing and ToS risks
- distinguish live, historical-only, paid, and unsupported features

### Coordination rule
One source of truth lives in this repository.

Before proposing architectural changes, every AI should be given or pointed to this specification.

No AI should silently create a separate competing architecture.

---

## 14. Current known gaps

- `dfs_week2.py` is not currently present in the repository.
- No production DraftKings optimizer is currently committed here.
- No contest simulator is currently committed here.
- No ownership model is currently committed here.
- Injury practice-status data is verified live, but final game-status/inactive sourcing and timestamped snapshot archiving are not yet implemented.
- No frontend is currently committed here.
- FTN publication lag is source-vintage dependent; the live canary currently supports a 1-week lag and will fail when the observed gap changes.
- Current route-level information is unavailable in the free stack.
- Direct-ridge Baseline V3 still improves RMSE under corrected FanDuel scoring, but 2018-2023 is development evidence rather than a pristine confirmatory holdout because the 2022 Taysom Hill error changed the estimator.
- V3 currently improves zero-opportunity rows but loses RMSE on played rows under corrected scoring; the direction is target-sensitive, so hurdle modeling remains experimental.
- FanDuel rule values are verified, but historical target completeness remains false because `special_teams_tds` is broader than kickoff/punt return TDs and `fumble_recovery_tds` cannot isolate own recoveries.
- FanDuel DEF scoring/projection is not yet implemented.
- The official FanDuel Test Slate #1 player pool is available through the upload-template ingest path.

---

## 15. Decision rule

The project should prefer being honest over sounding sophisticated.

A feature does not earn projection weight because it has a compelling football story.

It earns projection weight because it improves future predictions when evaluated using only information that would have been available before lineup lock.
