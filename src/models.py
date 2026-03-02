from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class Player(BaseModel):
    """Canonical NBA player identity. id is a stable slug+hash; use canonical_name for display."""
    id: str = Field(..., description="Stable internal id (slug + short hash of canonical_name)")
    provider_name: str = Field(..., description="Raw name from provider (e.g. Odds API)")
    canonical_name: str = Field(..., description="Best matched canonical display name")
    nba_player_id: Optional[int] = Field(None, description="NBA.com player id when resolved")
    team: str = Field(default="UNK", min_length=2, max_length=3)
    aliases: List[str] = Field(default_factory=list)


class PropLine(BaseModel):
    player_id: str = Field(...)
    provider: str = Field(..., description="Primary provider; use over_provider/under_provider when aggregated")
    stat_type: Literal["Points", "Rebounds", "Assists", "PRA", "Threes"] = Field(...)
    threshold: float = Field(..., gt=0)
    over_odds: Optional[float] = Field(None)
    under_odds: Optional[float] = Field(None)
    over_provider: Optional[str] = Field(None, description="Book offering best over odds (when aggregated)")
    under_provider: Optional[str] = Field(None, description="Book offering best under odds (when aggregated)")
    home_team: Optional[str] = Field(None, description="Event home team (from Odds API)")
    away_team: Optional[str] = Field(None, description="Event away team (from Odds API)")

    # Enforces the domain rule that all prop thresholds must be strictly positive.
    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive.")
        return v


class MarketSnapshot(BaseModel):
    snapshot_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    game_id: str = Field(...)
    lines: List[PropLine] = Field(default_factory=list)

class Game(BaseModel):
    game_id: str = Field(..., description="Unique identifier for the game")
    home_team: str = Field(..., description="Name or code for the home team")
    away_team: str = Field(..., description="Name or code for the away team")
    start_time: datetime = Field(..., description="Scheduled start time of the game (UTC)")