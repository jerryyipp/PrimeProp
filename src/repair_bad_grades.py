"""
One-time repair for picks incorrectly graded using the closest-date fallback.
Resets affected rows to ungraded or marks them void, and optionally clears bad cache entries.

Run from project root:
  python -m src.repair_bad_grades                    # repair known Giannis 2026-03-08 Points pick
  python -m src.repair_bad_grades --pick-id 123     # repair by pick id
  python -m src.repair_bad_grades --void            # mark as void (won=-1) instead of ungraded
  python -m src.repair_bad_grades --clear-cache     # also remove corresponding game_result_cache entry
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.database import DatabaseManager
from src.stats import delete_cached_game_result, resolve_nba_player_id


def _game_date_from_timestamp(timestamp_str: str):
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.date()
    except (ValueError, TypeError):
        return None


def repair_pick(
    pick_id: int = None,
    *,
    db_path: Path = None,
    mark_void: bool = False,
    clear_cache: bool = False,
) -> None:
    """
    Find the pick (by id or the known bad Giannis pick), reset to ungraded or mark void, optionally clear cache.
    """
    db = DatabaseManager(db_path=db_path)
    conn = db._conn
    if conn is None:
        print("DB not connected")
        return

    if pick_id is not None:
        cur = conn.execute(
            "SELECT id, player_name, timestamp, stat_type, nba_player_id, actual_result, won FROM picks WHERE id = ?",
            (pick_id,),
        )
        row = cur.fetchone()
        if not row:
            print(f"No pick with id={pick_id}")
            db.close()
            return
        picks = [dict(row)]
    else:
        cur = conn.execute(
            """
            SELECT id, player_name, timestamp, stat_type, nba_player_id, actual_result, won
            FROM picks
            WHERE (player_name LIKE '%Giannis%' AND player_name LIKE '%Antetokounmpo%')
            AND stat_type = 'Points' AND timestamp LIKE '2026-03-08%' AND won = 0
            ORDER BY id DESC
            LIMIT 5
            """
        )
        picks = [dict(r) for r in cur.fetchall()]
        if not picks:
            print("No matching Giannis pick found (Points, 2026-03-08, won=0). Use --pick-id ID.")
            db.close()
            return

    for pick in picks:
        pid = pick["id"]
        player_name = pick["player_name"]
        ts = pick["timestamp"]
        stat_type = pick["stat_type"]
        nba_id = pick["nba_player_id"]
        game_date = _game_date_from_timestamp(ts)
        if game_date is None:
            print(f"Pick id={pid}: could not parse timestamp {ts}")
            continue

        if mark_void:
            db.update_pick_result(pid, None, -1)
            print(f"Pick id={pid} ({player_name} {stat_type}): set to void (won=-1, actual_result=NULL).")
        else:
            db.reset_pick_to_ungraded(pid)
            print(f"Pick id={pid} ({player_name} {stat_type}): reset to ungraded (actual_result=NULL, won=NULL).")

        if clear_cache:
            if nba_id is None:
                nba_id = resolve_nba_player_id(player_name)
            if nba_id is not None:
                delete_cached_game_result(int(nba_id), game_date, stat_type)
                print(f"Cleared game_result_cache for player_nba_id={nba_id}, date={game_date}, stat={stat_type}.")
            else:
                print("Could not resolve nba_id; cache not cleared.")

    db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair picks incorrectly graded via closest-date fallback.")
    ap.add_argument("--pick-id", type=int, default=None, help="Specific pick id to repair (default: find Giannis 2026-03-08 Points)")
    ap.add_argument("--void", action="store_true", help="Mark as void (won=-1) instead of resetting to ungraded")
    ap.add_argument("--clear-cache", action="store_true", help="Remove corresponding game_result_cache entry")
    ap.add_argument("--db", type=Path, default=None, help="Path to primeprop.db (default: project root)")
    args = ap.parse_args()
    repair_pick(
        pick_id=args.pick_id,
        db_path=args.db,
        mark_void=args.void,
        clear_cache=args.clear_cache,
    )


if __name__ == "__main__":
    main()
