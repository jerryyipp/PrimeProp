"""
Persistence layer for +EV picks and performance tracking.

Logs pre-game picks to SQLite; outcome columns (actual_result, won) remain NULL
until graded manually or updated via a future stats API.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default DB path: project root / primeprop.db (resolve relative to this file)
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "primeprop.db"


class DatabaseManager:
    """
    Manages SQLite connection and schema for logging picks and computing win rates.
    Creates primeprop.db if it does not exist.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._connect_and_init()

    def _connect_and_init(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                player_name TEXT NOT NULL,
                stat_type TEXT NOT NULL,
                market_line REAL NOT NULL,
                projected REAL NOT NULL,
                edge REAL NOT NULL,
                recommended_side TEXT NOT NULL,
                actual_result REAL,
                won INTEGER
            )
            """
        )
        self._conn.commit()

    def log_pick(
        self,
        player_name: str,
        stat_type: str,
        market_line: float,
        projected: float,
        edge: float,
        recommended_side: str,
    ) -> None:
        """
        Insert a pre-game pick with current UTC timestamp.
        actual_result and won remain NULL for manual grading later.
        """
        assert self._conn is not None
        timestamp = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO picks (
                timestamp, player_name, stat_type, market_line,
                projected, edge, recommended_side, actual_result, won
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (timestamp, player_name, stat_type, market_line, projected, edge, recommended_side),
        )
        self._conn.commit()

    def get_win_rate(self) -> tuple[int, int, int, float]:
        """
        Compute stats over graded picks only (rows where won IS NOT NULL).

        Returns:
            (total_graded, wins, losses, win_pct).
            win_pct is 0.0 when there are no graded picks (avoids ZeroDivisionError).
        """
        assert self._conn is not None
        cur = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) AS losses
            FROM picks
            WHERE won IS NOT NULL
            """
        )
        row = cur.fetchone()
        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        if total == 0:
            return (0, 0, 0, 0.0)
        return (total, wins, losses, round(wins / total * 100.0, 2))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
