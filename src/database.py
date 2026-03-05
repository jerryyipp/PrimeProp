"""
Persistence layer for +EV picks and performance tracking.

Logs pre-game picks to SQLite; outcome columns (actual_result, won) remain NULL
until graded manually or updated via auto-grading (grade_picks).

log_pick supports optional fields from main: over_odds, under_odds, over_provider,
under_provider, player_id, nba_player_id, game_start. Schema and migration add
these columns to picks when missing (ALTER TABLE for existing DBs).
"""

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import MarketSnapshot, Player, PropLine

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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                game_id TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_lines (
                snapshot_id INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                stat_type TEXT NOT NULL,
                line REAL NOT NULL,
                over_odds REAL,
                under_odds REAL,
                over_provider TEXT,
                under_provider TEXT,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
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

    def save_snapshot(self, snapshot: MarketSnapshot, players_by_id: Dict[str, Player]) -> int:
        """
        Insert a snapshot and its lines for replay/audit.
        Returns the snapshots.id (integer PK).
        """
        assert self._conn is not None
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO snapshots (created_at, snapshot_id, game_id)
            VALUES (?, ?, ?)
            """,
            (created_at, snapshot.snapshot_id, snapshot.game_id),
        )
        pk = cur.lastrowid
        if pk is None:
            self._conn.commit()
            raise RuntimeError("save_snapshot: failed to get lastrowid")
        for line in snapshot.lines:
            player = players_by_id.get(line.player_id)
            player_name = player.canonical_name if player is not None else line.player_id
            self._conn.execute(
                """
                INSERT INTO snapshot_lines (snapshot_id, player_id, player_name, stat_type, line,
                                           over_odds, under_odds, over_provider, under_provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pk,
                    line.player_id,
                    player_name,
                    line.stat_type,
                    line.threshold,
                    line.over_odds,
                    line.under_odds,
                    line.over_provider,
                    line.under_provider,
                ),
            )
        self._conn.commit()
        return pk

    def load_snapshot(
        self, snapshot_ref: int | str
    ) -> Tuple[MarketSnapshot, Dict[str, Player]]:
        """
        Load a snapshot by integer id (snapshots.id) or by logical snapshot_id string.
        Returns (MarketSnapshot, players_by_id) for replay.
        """
        assert self._conn is not None
        if isinstance(snapshot_ref, int):
            row = self._conn.execute(
                "SELECT id, snapshot_id, game_id, created_at FROM snapshots WHERE id = ?",
                (snapshot_ref,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, snapshot_id, game_id, created_at FROM snapshots WHERE snapshot_id = ? ORDER BY id DESC LIMIT 1",
                (str(snapshot_ref),),
            ).fetchone()
        if row is None:
            raise ValueError(f"No snapshot found for id={snapshot_ref!r}")
        pk = row["id"]
        snapshot_id_str = row["snapshot_id"]
        game_id_str = row["game_id"]
        cur = self._conn.execute(
            """
            SELECT player_id, player_name, stat_type, line, over_odds, under_odds, over_provider, under_provider
            FROM snapshot_lines WHERE snapshot_id = ?
            """,
            (pk,),
        )
        lines: List[PropLine] = []
        players_by_id: Dict[str, Player] = {}
        for r in cur.fetchall():
            player_id = r["player_id"]
            player_name = r["player_name"] or player_id
            if player_id not in players_by_id:
                players_by_id[player_id] = Player(
                    id=player_id,
                    provider_name=player_name,
                    canonical_name=player_name,
                    nba_player_id=None,
                    team="UNK",
                    aliases=[],
                )
            provider = r["over_provider"] or r["under_provider"] or "saved"
            lines.append(
                PropLine(
                    player_id=player_id,
                    provider=provider,
                    stat_type=r["stat_type"],
                    threshold=float(r["line"]),
                    over_odds=float(r["over_odds"]) if r["over_odds"] is not None else None,
                    under_odds=float(r["under_odds"]) if r["under_odds"] is not None else None,
                    over_provider=r["over_provider"],
                    under_provider=r["under_provider"],
                    home_team=None,
                    away_team=None,
                )
            )
        snapshot = MarketSnapshot(
            snapshot_id=snapshot_id_str,
            game_id=game_id_str,
            lines=lines,
        )
        return snapshot, players_by_id

    def list_snapshots(self) -> List[Tuple[int, str, str, str]]:
        """Return list of (id, snapshot_id, game_id, created_at) for replay selection."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, snapshot_id, game_id, created_at FROM snapshots ORDER BY id DESC LIMIT 100"
        )
        return [(r["id"], r["snapshot_id"], r["game_id"], r["created_at"]) for r in cur.fetchall()]

    def get_earliest_snapshot_lines_for_game(
        self, game_id: str, for_date: Optional[date] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Return (player_id, stat_type) -> line from the earliest snapshot for this game on the given day.
        Used for line movement: compare current line to earliest line of the day.
        for_date: default today UTC. Returns {} if no snapshot for that game on that day.
        """
        assert self._conn is not None
        if for_date is None:
            for_date = datetime.now(timezone.utc).date()
        date_str = for_date.isoformat()
        row = self._conn.execute(
            """
            SELECT id FROM snapshots
            WHERE game_id = ? AND date(created_at) = ?
            ORDER BY created_at ASC LIMIT 1
            """,
            (game_id, date_str),
        ).fetchone()
        if row is None:
            return {}
        snap_id = row["id"]
        cur = self._conn.execute(
            "SELECT player_id, stat_type, line FROM snapshot_lines WHERE snapshot_id = ?",
            (snap_id,),
        )
        out: Dict[Tuple[str, str], float] = {}
        for r in cur.fetchall():
            out[(r["player_id"], r["stat_type"])] = float(r["line"])
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
