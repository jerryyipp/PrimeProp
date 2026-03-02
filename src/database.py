"""
Persistence layer for +EV picks and performance tracking.

Logs pre-game picks to SQLite; outcome columns (actual_result, won) remain NULL
until graded manually or updated via auto-grading (grade_picks).

log_pick supports optional fields from main: over_odds, under_odds, over_provider,
under_provider, player_id, nba_player_id, game_start. Schema and migration add
these columns to picks when missing (ALTER TABLE for existing DBs).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Type alias for pick rows (sqlite3.Row)
PickRow = sqlite3.Row

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
                player_id TEXT,
                nba_player_id INTEGER,
                game_start TEXT,
                stat_type TEXT NOT NULL,
                market_line REAL NOT NULL,
                line REAL,
                projected REAL NOT NULL,
                edge REAL NOT NULL,
                recommended_side TEXT NOT NULL,
                p_over_model REAL,
                stdev REAL,
                odds REAL,
                over_odds REAL,
                under_odds REAL,
                over_provider TEXT,
                under_provider TEXT,
                actual_result REAL,
                won INTEGER
            )
            """
        )
        self._conn.commit()
        self._migrate_picks_columns()

    def _migrate_picks_columns(self) -> None:
        """Add missing columns via ALTER TABLE so existing databases migrate safely."""
        assert self._conn is not None
        cur = self._conn.execute("PRAGMA table_info(picks)")
        names = {row[1] for row in cur.fetchall()}
        if "p_over_model" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN p_over_model REAL")
        if "stdev" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN stdev REAL")
        if "odds" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN odds REAL")
        if "over_odds" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN over_odds REAL")
        if "under_odds" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN under_odds REAL")
        if "over_provider" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN over_provider TEXT")
        if "under_provider" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN under_provider TEXT")
        if "line" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN line REAL")
        if "player_id" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN player_id TEXT")
        if "nba_player_id" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN nba_player_id INTEGER")
        if "game_start" not in names:
            self._conn.execute("ALTER TABLE picks ADD COLUMN game_start TEXT")
        self._conn.commit()

    def log_pick(
        self,
        player_name: str,
        stat_type: str,
        market_line: float,
        projected: float,
        edge: float,
        recommended_side: str,
        *,
        p_over_model: Optional[float] = None,
        stdev: Optional[float] = None,
        odds: Optional[float] = None,
        over_odds: Optional[float] = None,
        under_odds: Optional[float] = None,
        over_provider: Optional[str] = None,
        under_provider: Optional[str] = None,
        player_id: Optional[str] = None,
        nba_player_id: Optional[int] = None,
        game_start: Optional[str] = None,
    ) -> None:
        """
        Insert a pre-game pick. Optional: over_odds, under_odds, over_provider, under_provider,
        player_id, nba_player_id, game_start (main.py passes these). Stores line = market_line.
        actual_result and won remain NULL until grading.
        """
        assert self._conn is not None
        timestamp = datetime.now(timezone.utc).isoformat()
        line_val = market_line
        self._conn.execute(
            """
            INSERT INTO picks (
                timestamp, player_name, player_id, nba_player_id, game_start,
                stat_type, market_line, line, projected, edge, recommended_side,
                p_over_model, stdev, odds, over_odds, under_odds, over_provider, under_provider,
                actual_result, won
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                timestamp, player_name, player_id, nba_player_id, game_start,
                stat_type, market_line, line_val, projected, edge, recommended_side,
                p_over_model, stdev, odds,
                over_odds, under_odds, over_provider, under_provider,
            ),
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

    def get_graded_picks(self) -> list[sqlite3.Row]:
        """Return all picks with won IS NOT NULL (for backtest and calibration)."""
        assert self._conn is not None
        cur = self._conn.execute(
            """
            SELECT id, timestamp, player_name, player_id, nba_player_id, game_start,
                   stat_type, market_line, line, projected, edge, recommended_side,
                   p_over_model, stdev, odds, over_odds, under_odds, over_provider, under_provider,
                   actual_result, won
            FROM picks
            WHERE won IS NOT NULL
            ORDER BY id
            """
        )
        return cur.fetchall()

    def get_ungraded_picks(self, before_datetime: datetime) -> list[PickRow]:
        """
        Return picks where actual_result IS NULL and timestamp < before_datetime.
        Used by auto-grading: only consider picks old enough that the game has been played.
        """
        assert self._conn is not None
        before_str = before_datetime.isoformat()
        cur = self._conn.execute(
            """
            SELECT id, timestamp, player_name, player_id, nba_player_id, game_start,
                   stat_type, market_line, line, projected, edge, recommended_side,
                   p_over_model, stdev, odds, over_odds, under_odds, over_provider, under_provider,
                   actual_result, won
            FROM picks
            WHERE actual_result IS NULL AND won IS NULL AND timestamp < ?
            ORDER BY id
            """,
            (before_str,),
        )
        return cur.fetchall()

    def update_pick_result(self, pick_id: int, actual_result: float, won: int) -> None:
        """
        Set actual_result and won for a pick (won: 1 = win, 0 = loss).
        actual_result is the actual stat value (PTS/REB/AST) from the game.
        """
        assert self._conn is not None
        self._conn.execute(
            "UPDATE picks SET actual_result = ?, won = ? WHERE id = ?",
            (actual_result, won, pick_id),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
