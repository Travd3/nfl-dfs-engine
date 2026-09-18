# Live Data Source Notes

This file tracks the live-data sources intended for the NFL DFS engine and the caveats that matter for production use.

## 1. Injuries

### nflverse
Verified useful fields in the current Python pull include:
- practice_status
- report_status
- report_primary_injury
- gsis_id

A direct `nflreadpy==0.1.5` pull on 2026-09-18 returned 412 current-season rows covering Weeks 1 and 2 and all 32 teams. In that pull, `practice_status` was fully populated while `report_status` coverage was only about 16%.

Important schema caveat: the current nflverse R injury dictionary documents a `date_modified` field, but the verified Python table did not contain `date_modified` or another update timestamp. Treat the actual runtime schema as authoritative and let the canary test alert us when this changes.

Because the table has week-level rows without a reliable source timestamp, every live pull must be stamped by our pipeline and archived. Do not use final weekly injury rows as if they were guaranteed pre-lock snapshots for TNF historical backtests.

### Sleeper
The public player endpoint includes fields such as:
- injury_status
- practice_participation
- status
- news_updated

Sleeper documents the player map as a once-per-day style endpoint and says the API is free for non-commercial use. Commercial use requires contacting Sleeper. Therefore it is not an automatic commercial fallback.

### Production rule
Use nflverse as a structured practice-status source, not as the sole final game-designation source.

Do not mark a player Out, Doubtful, Questionable, IR, PUP, or NFI from a stale cached source. Record our own retrieval timestamp and retain append-only snapshots.

Near kickoff, a verified final game-status source and official inactive status should supersede midweek practice information.

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
Use NWS / NOAA `api.weather.gov` for current U.S. outdoor-stadium forecasts.

Current-slate display is approved. Projection weight remains experimental until we possess timestamped historical pre-lock forecasts or an equivalent defensible archive.

Recommended flow:
1. static stadium latitude/longitude
2. `/points/{lat},{lon}`
3. follow the returned `forecastHourly` grid endpoint
4. retain forecast timestamp and kickoff-relative forecast row

Use a descriptive User-Agent as required by NWS.

Weather should contribute nothing when the roof is dome/closed.

Do not backfill a historical forecast feature with observed final weather. That would use realized information rather than what was knowable before lock.

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


## 7. Executable source canaries

Data-source claims that can be checked mechanically live in `tests/test_data_availability.py`.

Current canaries cover:
- current-season injury availability
- missing injury timestamps in the actual Python schema
- injury game-status coverage
- live participation unavailability
- historical participation availability
- FTN versus PBP freshness
- required FTN fields
- U/S/P quarterback-location codes
- forward Vegas line availability
- separation of pre-game `total_line` from post-game `total` / `result`
- data-vintage recording

When a canary fails because a source improves or changes, update the implementation and this document together.
