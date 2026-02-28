from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class Player(BaseModel):
    id: str = Field(..., description="Unique internal identifier")
    standardized_name: str = Field(..., description="Canonical name")
    team: str = Field(..., min_length=2, max_length=3)
    aliases: List[str] = Field(default_factory=list)


class PropLine(BaseModel):
    player_id: str = Field(...)
    provider: str = Field(...)
    # Restricts stat types to a shared, canonical set understood across all providers.
    stat_type: Literal["Points", "Rebounds", "Assists", "PRA", "Threes"] = Field(...)
    threshold: float = Field(..., gt=0)
    over_odds: Optional[float] = Field(None)
    under_odds: Optional[float] = Field(None)

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