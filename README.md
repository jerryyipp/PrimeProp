# PrimeProp

Sports betting tool that helps optimize NBA player props by comparing your projections to live market lines, identifying +EV bets, and tracking performance over time.

## Features

- **Live odds ingestion** — Pulls pre-game player props from The Odds API (per-event endpoint) for upcoming NBA games
- **Best-odds aggregation** — Merges lines from multiple books into a single snapshot with best over/under odds per provider
- **Fuzzy name matching** — Links player names across providers (e.g. "Steph Curry" ↔ "Stephen Curry")
- **Projection engine** — Historical stats from NBA.com API; weighted/simple/exponential methods; optional blend/ensemble strategies
- **Matchup adjustment** — Adjusts PTS/REB/AST projections based on opponent defensive metrics (LeagueDashOpponentTeamStats)
- **+EV optimizer** — Computes EV from American odds + Normal(mean, stdev) model; optional calibration; ranks by best_ev; recommends Over/Under/Pass
- **Real-time alerting** — Telegram and/or Discord alerts when |edge| exceeds threshold (default 5%)
- **Persistence** — SQLite logs only the final "best available" picks shown (display = save). Grading uses exact game date only; DNP/ruled-out marked void. Win rate includes pushes as a separate count.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```
   ODDS_API_KEY=your_api_key_here
   TELEGRAM_BOT_TOKEN=optional
   TELEGRAM_CHAT_ID=optional
   DISCORD_WEBHOOK_URL=optional
   # Context adjustment (home/away, rest): optional
   HOME_BONUS_PCT=0.02
   B2B_PENALTY_PCT=0.01
   CONTEXT_ADJUST_CLAMP=0.05
   # Projection strategy: single | blend | ensemble | season_blend
   PROJECTION_STRATEGY=single
   SEASON_BLEND_ALPHA=0.7
   # Confidence interval for projection display (console + alerts)
   PROJECTION_INTERVAL_LEVEL=0.80
   # Ingest resilience: timeout (s), retries, circuit breaker max failures per bookmaker
   INGEST_TIMEOUT_S=12
   INGEST_RETRIES=3
   INGEST_CB_MAX_FAILURES=3
   # Optional: persist market snapshots for replay/audit (0=off, 1=on)
   PERSIST_SNAPSHOTS=0
  # Optional: only alert on props where line moved in our favor (Over: line down; Under: line up)
  REQUIRE_LINE_MOVEMENT_WITH_US=0
  # Betting day rollover: 1 = 1 AM local (America/Indiana/Indianapolis); no DST drift
  BETTING_DAY_ROLLOVER_HOUR=1
   ```

3. Get an API key from [The Odds API](https://the-odds-api.com/).

## Usage

- **Run the main pipeline** (fetch upcoming games, ingest props, build projections, rank, alert, persist):
  ```bash
  python main.py
  ```

- **Run tests** (no API calls, uses mock data):
  ```bash
  python test_projection.py
  python test_optimizer.py
  python test_optimizer_prob.py
  python test_database.py
  python test_alerts.py   # requires Telegram/Discord env vars for real alerts
  ```

- **Ping live players** (verify API and see who has odds posted):
  ```bash
  python radar.py
  ```

- **Grade picks** (update database with actual results; exact game date only, DNP → void):
  ```bash
  python -m src.grade_picks
  ```
  Or from project root: `python grade_picks.py` (if run from `src/`, path is adjusted).

- **Backtest** (hit rate on graded picks; Brier/log loss/calibration temporarily disabled until `p_over_model` is populated):
  ```bash
  python -m src.backtest
  ```

- **Repair bad grades** (e.g. picks graded from wrong date; resets or marks void, optional cache clear):
  ```bash
  python -m src.repair_bad_grades --void --clear-cache
  ```
  Use `--pick-id ID` to target a specific pick; default finds Giannis 2026-03-08 Points.

- **Import manual picks** (one-off insert into DB for grading/backtest):
  ```bash
  python import_manual_picks.py
  ```

- **Replay a snapshot** (re-run ranking without calling Odds API; requires a saved snapshot):
  ```bash
  python replay_snapshot.py <id>
  ```
  Run with no args to list recent snapshot ids. Enable saving with `PERSIST_SNAPSHOTS=1` in `.env`.

## Architecture

- `src/models.py` — Pydantic models (Player, PropLine, MarketSnapshot, Game)
- `src/ingest.py` — Async multi-source ingestor, FuzzyNameMatcher, OddsApiIngestor, aggregate_snapshot_by_best_odds
- `src/stats.py` — NBA.com API client: resolve_nba_player_id, fetch_last_n_games, get_stat_series; fetch_game_result (exact-date only, DNP→void) for grading; game_result_cache
- `src/projection.py` — Historical stat projection (weighted/simple/exponential); winsorize; blend/ensemble; ProjectionResult; adjust_for_matchup; apply_context_adjustments
- `src/matchup.py` — Team defensive metrics cache, league averages, get_player_team; used for matchup adjustment
- `src/optimizer.py` — EV calculation, calibration, rank_props_by_edge (best_ev first), PropEdge
- `src/alerting.py` — Telegram/Discord notifications for high-value props
- `src/database.py` — SQLite persistence (log_pick, get_win_rate 5-tuple, update_pick_result, reset_pick_to_ungraded, snapshots/snapshot_lines for replay)
- `src/main.py` — Canonical ranking: run_ranking_for_snapshot, _env_* helpers; used by root main and replay
- `src/run_ranking.py` — Re-exports run_ranking_for_snapshot from src.main
- `src/grade_picks.py` — Grade picks (exact game date; DNP→void); uses fetch_game_result(status, value)
- `src/backtest.py` — Hit rate on graded picks (Brier/log loss/calibration/EV sections temporarily disabled)
- `src/repair_bad_grades.py` — One-time repair for wrongly graded picks (e.g. closest-date bug); --void, --clear-cache
- `main.py` — Thin entrypoint: load_dotenv; timezone-aware betting day (BETTING_DAY_ROLLOVER_HOUR, America/Indiana/Indianapolis); upcoming events → ingest → run_ranking_for_snapshot → best available picks (no fixed Top N) → alerts → save final_displayed_picks only
- `replay_snapshot.py` — CLI to load a saved snapshot by id and re-run ranking without API calls
- `import_manual_picks.py` — One-off script to insert manual picks into primeprop.db

## License

MIT
