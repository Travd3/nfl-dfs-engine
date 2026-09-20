# NFL DFS Engine Project Specification

## 1. Product

### Product 1
NFL salary-cap DFS projection, simulation, and lineup engine.

Initial platform targets:
- DraftKings Classic
- FanDuel NFL salary-cap contests

The football projection layer must be platform-agnostic. Platform scoring, salary cap, roster slots, and contest rules belong in explicit contest/scoring configuration rather than being hardcoded into the model.

### Scoring shape
Scoring is now platform-configurable and applied at player-game level.

FanDuel Test Slate #1 is the **2026-09-20 Week 2, 13-game Sunday main slate** from the official FanDuel upload template. It uses 0.5 PPR, the verified yardage bonuses and turnover/two-point rules already recorded in `scoring.py`.

The historical skill-player target is usable for projection research but remains `complete=False` because the rare `special_teams_tds` and `fumble_recovery_tds` source fields are not semantically identical to FanDuel's narrower return-TD / own-fumble-recovery-TD wording. DEF scoring is a separate model path.

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

Test Slate #1 optimizer policy:
- Out = auto-exclude
- Doubtful = auto-exclude for the first single-entry test
- Questionable = keep eligible but visibly flag for final pre-lock review
- newer official team/NFL game-status information can override a stale FanDuel flag when timestamped
- official inactives auto-exclude
- do not apply an unvalidated injury-based projection haircut; status eligibility is downstream from the pure V3 mean projection

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

The FTN lag is not a permanent constant. On 2026-09-18 FTN trailed PBP by one completed week. On 2026-09-19 both sources contained Week 2 rows before the Week 2 Sunday lock, but that includes a partially played week and does **not** prove that full Week 2 charting will be available for Week 3. The current Week 2 live projection does not use FTN scheme inputs. Re-verify full prior-week coverage before using a one-week charting lag in a future live scheme feature.

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

### Baseline V3 fresh historical confirmation

The frozen V3 estimator was extended backward without changing the feature set,
winsorization thresholds, scoring logic, alpha grid, or evaluation metrics.

A schema audit found that `targets` is effectively unusable in 2003-2008 even
though receptions remain populated. Because `targets__ewm`, `opps`,
`played_rate__ewm`, and `play_share__ewm` depend on target attribution, the
training floor is 2009. The requested 2010 test season was then skipped by the
already-frozen minimum-training-row guard, leaving 2011-2017 as the untouched
confirmation block.

Reported fresh confirmation:

```text
FanDuel Test Slate #1 scoring, player-game grain
Fresh confirmation 2011-2017, n=44,857

RMSE
  naive 5.6916
  V3    5.6566
  delta +0.0348
  95% CI [+0.0102, +0.0606]
  P>0 = 0.999

MAE
  naive 3.5670
  V3    3.6004
  delta -0.0335
  95% CI [-0.0478, -0.0183]

Spearman
  naive 0.6533
  V3    0.6505

Bias
  naive +0.0343
  V3    -0.1246

Calibration
  naive slope 0.913 / intercept +0.385
  V3    slope 1.029 / intercept -0.010
```

By season, V3 lost 2011 and 2012 and improved in each season from 2013 through
2017. By position, the aggregate direction was positive for QB, RB, WR, and TE,
but only RB's position-specific bootstrap interval excluded zero.

Most importantly, the aggregate RMSE advantage is not uniform across player
states:

```text
played rows
  n=25,361
  RMSE delta -0.0570
  V3 bias -1.840

zero-opportunity rows
  n=19,496
  RMSE delta +0.2567
  V3 bias +2.106
```

Interpretation:
- V3 has now cleared the project's broad mean-projection RMSE gate on an untouched historical block.
- V3 calibration is materially better than naive persistence.
- V3 remains slightly worse on MAE and rank correlation.
- The total RMSE gain is driven by better handling of zero-opportunity rows.
- V3 is reliably worse than naive persistence on rows where the player records an opportunity.

Therefore V3 is promoted to **INTERIM PRODUCTION MEAN BASELINE**, not to a final
DFS ranking or tournament model. Its weaknesses on played-player production and
ordering are explicit requirements for the next modeling pass.

The historical FanDuel scoring rules are verified, but the target remains
`complete=False` because `special_teams_tds` is broader than only kickoff/punt
return touchdowns and `fumble_recovery_tds` cannot isolate own recoveries.
Those rare semantic gaps are documented rather than hidden.

### Availability hurdle experiment

Issue #2 tested:

```text
E[points] = P(record an opportunity) × E[points | opportunity]
```

The experiment imported frozen V3's frame, features, winsorization, minimum-history
filter, rolling-origin splits, and scoring profile unchanged. Only the estimator
differed.

Fresh 2011-2017 block:

```text
arm          RMSE      MAE   Spearman      bias
naive      5.6916   3.5670     0.6533   +0.0343
v3         5.6566   3.6004     0.6505   -0.1246
hurdle     5.6765   3.6315     0.6541   -0.0800

V3 -> hurdle RMSE delta -0.0199
95% CI [-0.0288, -0.0113]
```

The hurdle model is **REJECTED**. It is significantly worse than V3 on both RMSE
and MAE and does not cleanly beat naive persistence on RMSE.

The useful diagnostic is the conditional production component trained only on
played rows. On played rows it reported RMSE 6.4876 versus 6.8188 for V3, a
+0.3319 improvement with 95% CI [+0.3002, +0.3649], while largely removing V3's
negative played-row bias.

That conditional result is **EXPERIMENTAL / DIAGNOSTIC ONLY**. It does not by
itself produce a full-slate mean projection, and the 2011-2017 block has now
been consumed by model experimentation. Do not keep testing new arms on that
block and call them independently confirmed.

Current lesson:
- forced multiplication of separately estimated components has failed twice
  in this project
- availability information may still be useful as an input to a direct points
  model rather than as a multiplicative discount
- any next variant must be pre-registered and evaluated without reusing the
  2011-2017 block as a fresh confirmation set


### FanDuel DEF baseline

`def_scoring.py` and `team_defense.py` implement the first transparent
FanDuel DEF mean model.

Test Slate #1 scoring provenance:
- all non-zero DEF categories are visible in the actual contest Rules screenshots
- Extra Point Return is +2
- the contest note defines defensive points allowed from offensive scoring only,
  so opponent defensive/special-teams TDs, defensive safeties, and defensive
  conversion returns do not count toward the PA tier
- the FanDuel UI omits the zero-valued 21-27 PA row; the model uses 0 for that
  band, consistent with established FanDuel scoring references

Corrected rolling-origin backtest reported by Claude, n=7,649 team-games:

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

The reported RMSE direction is positive in all 14 test seasons from 2012-2025.
The corrected target was sanity-checked for double subtraction: 2023-2025 had
zero negative points-allowed rows and 11.6% of team-games were changed by the
D/ST-score removal.

Live Test Slate #1:
- trained through fully completed 2026 Week 1 only
- 26 defenses in the exact 13-game FanDuel slate
- 26 mapped, 0 unmapped
- highest mean projection: Tampa Bay 8.93
- highest value tier includes Chicago 7.21 at $3,700, Carolina 6.44 at $3,300,
  and Arizona 6.04 at $3,200

Current status: **PRODUCTION MEAN BASELINE FOR TEST SLATE #1**.

This remains a mean model, not a defense outcome distribution. The historical
DEF target standard deviation is about 5.79 while the live means span only
3.25-8.93, so tail outcomes such as shutout-plus-return-TD remain future
simulation work.

---

## 8. Baseline model status

### Baseline V1
Role and efficiency were modeled separately, then multiplied. Held-out testing
showed that independently estimating the two components compounded variance.

### Baseline V2
A direct ridge on fantasy points outperformed the role × efficiency product.
Added snap share, red-zone role, pace, rest, and venue did not improve it and
remain **REJECTED** for projection weight.

### Baseline V3
`baseline_v3.py` is now the **INTERIM PRODUCTION MEAN BASELINE**.

Why it earned that status:
- final player-game grain
- platform-configurable scoring
- zero-opportunity active-roster rows included
- rolling-origin validation
- fresh untouched 2011-2017 confirmation
- block-bootstrap RMSE improvement vs naive persistence
- materially improved calibration

Why it is still only interim:
- MAE remains worse than naive persistence
- Spearman rank correlation remains slightly worse
- its RMSE edge is driven by zero-opportunity rows
- it is worse than naive persistence on played-player rows
- it is not an outcome-distribution, ceiling, or contest-win model

### Issue #2 result
The multiplicative availability hurdle is **REJECTED** and frozen V3 remains the
interim production mean baseline.

The played-only conditional ridge is a strong research signal, but it is not a
replacement model. Future work may test availability probability as a feature
inside a direct points model or another non-multiplicative structure.

That research is **non-blocking for Test Slate #1**. The immediate product path is
live inference, DEF, optimizer, and the minimal test interface.

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
1. Build the first salary-cap optimizer for the 9-player, $60,000 FanDuel roster using the archived skill-player and DEF projection snapshots.
2. Build a minimal test interface that shows projections, salary, value, injury/status flags, and the recommended lineup.
3. Complete reliable final game-status/inactive sourcing before Sunday lock.
4. Archive the final pre-lock lineup, salary used, projections, source vintage, and commit SHA.

Conditional played-player modeling remains a parallel research task and must not
delay the first live Sunday test.

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
- Direct-ridge Baseline V3 has passed a fresh 2011-2017 RMSE confirmation and is the interim production mean baseline.
- V3's gain is concentrated in zero-opportunity rows; it remains worse than naive on played rows and slightly worse on rank correlation.
- The multiplicative availability hurdle was tested and rejected. A played-only conditional ridge showed a large diagnostic improvement, but no validated full-slate replacement model exists yet.
- FanDuel rule values are verified, but historical target completeness remains false because `special_teams_tds` is broader than kickoff/punt return TDs and `fumble_recovery_tds` cannot isolate own recoveries.
- FanDuel DEF scoring/model has passed the corrected rolling-origin rerun and is the production mean baseline for Test Slate #1.
- The official FanDuel Test Slate #1 player pool is available through the upload-template ingest path.
- `live_projection.py` is complete for Week 2 live QB/RB/WR/TE inference, including the fully-completed-week training guard and explicit mapping-status diagnostics.
- Test Slate #1 skill-player mapping/inference is archived under `experiments/2026-W02-fanduel-main/`.

---

## 15. Decision rule

The project should prefer being honest over sounding sophisticated.

A feature does not earn projection weight because it has a compelling football story.

It earns projection weight because it improves future predictions when evaluated using only information that would have been available before lineup lock.
