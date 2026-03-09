"""
Auto-grade stored picks: load ungraded picks (game in the past), fetch actual stat
from nba_api, compare to market_line, and update won/actual_result.
Uses pick timestamp date as game date. Includes logging and safe retry behavior.

Run from project root: python -m src.grade_picks
Or from src/: python grade_picks.py (adds project root to path).
"""

import sys
from pathlib import Path

# Allow running as script from src/ (e.g. python grade_picks.py)
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.database import DatabaseManager
from src.stats import fetch_game_result, resolve_nba_player_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Only consider picks at least this old (game assumed finished)
GRADE_CUTOFF_HOURS = 4
# Retry settings for API/transient failures
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0


def _game_date_from_pick_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse pick timestamp (ISO) to date (UTC)."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt
    except (ValueError, TypeError):
        return None


def _compute_won(actual: float, market_line: float, recommended_side: str) -> int:
    """Over: win if actual > line; Under: win if actual < line; push (actual == line) = -1 (void); else loss = 0."""
    side = (recommended_side or "").strip()
    if side == "Over":
        if actual > market_line:
            return 1
        if actual == market_line:
            return -1
        return 0
    if side == "Under":
        if actual < market_line:
            return 1
        if actual == market_line:
            return -1
        return 0
    return 0


def grade_picks(
    db_path: Optional[Path] = None,
    before_datetime: Optional[datetime] = None,
) -> tuple[int, int, int]:
    """
    Load ungraded picks with timestamp < before_datetime, fetch actual stat per pick,
    update actual_result and won. Uses pick timestamp date as game date.

    Returns:
        (graded_count, skipped_count, error_count).
    """
    db = DatabaseManager(db_path=db_path)
    if before_datetime is None:
        before_datetime = datetime.now(timezone.utc) - timedelta(hours=GRADE_CUTOFF_HOURS)
    picks = db.get_ungraded_picks(before_datetime)
    logger.info("Ungraded picks (timestamp < %s): %d", before_datetime.isoformat(), len(picks))

    graded = 0
    skipped = 0
    errors = 0

    for pick in picks:
        pick_id = pick["id"]
        player_name = pick["player_name"]
        stat_type = pick["stat_type"]
        market_line = float(pick["market_line"])
        recommended_side = pick["recommended_side"] or ""

        ts = pick["timestamp"]
        dt = _game_date_from_pick_timestamp(ts)
        if dt is None:
            logger.warning("Pick id=%s: invalid timestamp %r", pick_id, ts)
            errors += 1
            continue
        game_date = dt.date()

        nba_id = pick["nba_player_id"] if "nba_player_id" in pick.keys() else None
        if nba_id is not None:
            try:
                nba_id = int(nba_id)
            except (TypeError, ValueError):
                nba_id = None
        if nba_id is None:
            nba_id = resolve_nba_player_id(player_name)
        if nba_id is None:
            logger.warning("Pick id=%s: could not resolve player %r", pick_id, player_name)
            skipped += 1
            continue

        status: str = "no_match"
        actual: Optional[float] = None
        for attempt in range(MAX_RETRIES):
            status, actual = fetch_game_result(nba_id, game_date, stat_type)
            if status != "no_match" or attempt == MAX_RETRIES - 1:
                break
            if attempt < MAX_RETRIES - 1:
                logger.debug("Pick id=%s: no exact-date match, retry %s/%s", pick_id, attempt + 1, MAX_RETRIES)
                time.sleep(RETRY_DELAY_SEC)

        if status == "no_match":
            logger.warning(
                "Pick id=%s: no exact game on %s for %s (%s); skipped (do not grade from another date).",
                pick_id, game_date, player_name, stat_type,
            )
            skipped += 1
            continue

        if status == "dnp":
            try:
                db.update_pick_result(pick_id, None, -1)
                graded += 1
                logger.info(
                    "Pick id=%s: %s %s on %s — ruled out/DNP; marked void (won=-1).",
                    pick_id, player_name, stat_type, game_date,
                )
            except Exception as e:
                logger.exception("Pick id=%s: update_pick_result (void) failed: %s", pick_id, e)
                errors += 1
            continue

        assert status == "ok" and actual is not None
        won = _compute_won(actual, market_line, recommended_side)
        try:
            db.update_pick_result(pick_id, actual, won)
            graded += 1
            outcome = "WIN" if won == 1 else ("PUSH" if won == -1 else "LOSS")
            logger.info("Graded pick id=%s: %s %s %s line=%.1f actual=%.1f -> %s", pick_id, player_name, stat_type, recommended_side, market_line, actual, outcome)
        except Exception as e:
            logger.exception("Pick id=%s: update_pick_result failed: %s", pick_id, e)
            errors += 1

    db.close()
    logger.info("Grade run complete: graded=%d skipped=%d errors=%d", graded, skipped, errors)
    return (graded, skipped, errors)


def main() -> None:
    """CLI entrypoint for auto-grading."""
    grade_picks()


if __name__ == "__main__":
    main()
