# Live Data Source Notes

This file tracks the live-data sources intended for the NFL DFS engine and the caveats that matter for production use.

## 1. Injuries

### nflverse
Potentially useful fields include:
- practice_status
- report_status
- report_primary_injury
- date_modified
- gsis_id

Important caveat: nflverse's current release metadata exposes an injuries release and `load_injuries()` accepts current seasons, but the official nflverse update-schedule page still says the source died after 2024 and that 2025 data is unavailable. Treat current-season injury coverage as unverified until a direct pull confirms current 2026 records.

### Sleeper
The public player endpoint includes fields such as:
- injury_status
- practice_participation
- status
- news_updated

Sleeper documents the player map as a once-per-day style endpoint and says the API is free for non-commercial use. Commercial use requires contacting Sleeper. Therefore it is not an automatic commercial fallback.

### Production rule
Do not mark a player Out, Doubtful, Questionable, IR, PUP, or NFI from a stale cached source without recording the source timestamp.

Near kickoff, official inactive status should supersede midweek practice/report information.

## 2. DraftKings salary and player IDs

Use the official DraftKings CSV template/export supplied through the logged-in product.

This is the production source of truth for:
- DK player ID
- salary
- slate membership
- roster position
- team/game mapping

Classic, Showdown, and other contest formats may use different draft groups and IDs.

Avoid depending on undocumented DraftKings JSON endpoints for the production system.

## 3. Weather

### Stadium/game metadata
Use nflverse schedules for:
- roof
- surface
- game time
- stadium context
- Vegas spread/total

### Forecast
Use NWS / NOAA `api.weather.gov` for U.S. outdoor-stadium forecasts.

Recommended flow:
1. static stadium latitude/longitude
2. `/points/{lat},{lon}`
3. follow the returned `forecastHourly` grid endpoint
4. retain forecast timestamp and kickoff-relative forecast row

Use a descriptive User-Agent as required by NWS.

Weather should contribute little or nothing when the roof is dome/closed.

## 4. Ownership

There is no official free pre-lock DraftKings ownership feed.

### Pre-lock
Build an internal ownership model from:
- salary
- projection
- projection per dollar
- implied team total
- position
- stack context
- injury-driven role
- likely chalk/narrative variables only when they can be represented reproducibly

### Post-lock ground truth
DraftKings GameCenter CSV exports include `% Drafted` after the contest locks.

DraftKings documentation says users can export lineups for contests they entered and also for viewable contests they did not enter. These files should be archived for selected contest archetypes each week.

Suggested archive targets:
- one large-field main GPP
- one smaller-field GPP
- one double-up or similar cash contest if available

This allows ownership models to be trained against actual realized field ownership rather than another site's projection.

Publicly visible ownership predictions from articles can be stored as timestamped calibration snapshots, but paid/paywalled datasets should not be scraped.

## 5. Current free production stack

```text
Official DK lobby CSV
    -> salary / DK ID / slate

nflverse schedules
    -> Vegas / roof / game metadata

nflverse PBP + player stats + snaps + NGS + FTN
    -> football / role / scheme features

verified injury source(s)
    -> practice / game status / availability

NWS api.weather.gov
    -> kickoff weather for outdoor games

DK GameCenter post-lock CSVs
    -> realized ownership training data
```

## 6. Rule

Every live feature must carry an as-of timestamp or source vintage when practical. If historical backtesting cannot reproduce what was knowable before slate lock, the feature is not eligible for production evaluation.
