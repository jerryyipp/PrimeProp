"""
Replay a persisted market snapshot: load by id (or snapshot_id) and re-run ranking
without calling Odds API. Usage: python replay_snapshot.py <snapshot_id_or_int>
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv()

from src.database import DatabaseManager
from src.run_ranking import run_ranking_for_snapshot


def _parse_ref(raw: str):
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python replay_snapshot.py <snapshot_id_or_int>")
        print("  Example: python replay_snapshot.py 1")
        print("  Example: python replay_snapshot.py abc123-game-id")
        db = DatabaseManager()
        try:
            rows = db.list_snapshots()
            if rows:
                print("\nRecent snapshots (id, snapshot_id, game_id, created_at):")
                for r in rows[:10]:
                    print(f"  {r[0]}  {r[1]}  {r[2]}  {r[3]}")
            else:
                print("No snapshots in database. Run main.py with PERSIST_SNAPSHOTS=1 first.")
        finally:
            db.close()
        sys.exit(1)

    ref = _parse_ref(sys.argv[1])
    db = DatabaseManager()
    try:
        snapshot, players_by_id = db.load_snapshot(ref)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        db.close()

    print(f"Loaded snapshot id={ref}: {len(snapshot.lines)} lines, game_id={snapshot.game_id}")
    ranked, id_to_canonical, _, _ = asyncio.run(run_ranking_for_snapshot(snapshot, players_by_id))
    print(f"Ranked {len(ranked)} props (no alerts or log_pick in replay mode).")


if __name__ == "__main__":
    main()
