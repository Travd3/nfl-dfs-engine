# NFL DFS Engine Project Specification

## 1. Product

### Product 1
DraftKings NFL Classic DFS optimizer and slate simulator.

### Scoring shape
Full PPR with DraftKings-specific scoring handled at the appropriate level, including yardage bonuses inside game or player-game simulation rather than per-play features.

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

#### DraftKings salaries and IDs
**Status: PRODUCTION INPUT.**

Use the official DraftKings lobby/lineup-template CSV supplied to the logged-in user. It is the source of truth for:
- slate membership
- salary
- roster position
- DraftKings player ID
- team/game metadata

Do not make the undocumented DraftKings draftables API a production dependency.

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
- FTN charting history: 2 weeks

The FTN two-week assumption is currently conservative and based on limited live observation. It must be measured across the season rather than treated as permanent truth.

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

Current walk-forward evaluation uses prior seasons to predict held-out 2024 and 2025 player-game-channel outcomes.

Reported result from the current research pass:

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

- Baseline V1 is effectively tied with naive persistence.
- Team scheme rates do not provide reliable improvement.
- Player sensitivity is harmful out of sample.
- Scheme therefore has zero projection weight in the current DFS model.

The exact results should be reproducible from the committed pipeline before they are treated as canonical.

---

## 8. Baseline model status

### Current Baseline V1
Role and efficiency are modeled separately, then multiplied.

Conceptually:

```text
expected opportunities
    x
expected points per opportunity
    =
expected fantasy points
```

This structure is interpretable, but current results suggest that independently estimating the two pieces may compound variance.

### Current highest-priority modeling question

Can a stronger production projection materially beat naive player scoring persistence?

Candidate next approaches:

- direct player-points model with role features as inputs
- two-part role/efficiency model with estimated dependence between components
- hierarchical or empirical-Bayes role priors
- more explicit team-volume modeling
- better opportunity-state updating across early-season weeks

Do not add more advanced DFS layers until the baseline projection is meaningfully measurable.

---

## 9. Evaluation requirements

Any model change should report:

1. held-out MAE
2. held-out RMSE
3. rank correlation
4. calibration
5. bootstrap confidence interval for improvement
6. position/channel slices
7. early-season vs late-season performance
8. evidence of leakage
9. comparison against naive persistence
10. comparison against a noise-control feature set where relevant

Bootstrap should resample slate-weeks or another appropriately correlated block rather than treating every player-row as independent.

---

## 10. DFS roadmap

### Immediate
1. Improve the baseline projection model.
2. Add official DraftKings salary/player-ID CSV ingest.
3. Verify and add a reliable live injury/status source.

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
- Current FTN publication lag assumption needs ongoing measurement; data-availability canaries now monitor it.
- Current route-level information is unavailable in the free stack.
- Current baseline projection is not yet demonstrably better than naive persistence.

---

## 15. Decision rule

The project should prefer being honest over sounding sophisticated.

A feature does not earn projection weight because it has a compelling football story.

It earns projection weight because it improves future predictions when evaluated using only information that would have been available before lineup lock.
