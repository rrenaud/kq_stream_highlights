"""
Data models for the highlight rating tool.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Path to chapters directory (relative to main project)
CHAPTERS_DIR = Path(__file__).parent.parent / "chapters"


@dataclass
class Event:
    """A player event that can be rated."""
    id: int
    game_id: int
    time: float
    event_type: str
    delta: float
    positions: list[int] = field(default_factory=list)
    values: list = field(default_factory=list)

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'game_id': self.game_id,
            'time': self.time,
            'type': self.event_type,
            'delta': self.delta,
            'abs_delta': self.abs_delta,
            'positions': self.positions,
            'values': self.values,
        }


@dataclass
class GameContext:
    """Context information about a game."""
    game_id: int
    title: str
    map_name: str
    winner: str
    win_condition: str
    start_time: float
    end_time: float
    hivemind_url: str
    users: dict = field(default_factory=dict)  # position -> user_id

    def to_dict(self) -> dict:
        return {
            'game_id': self.game_id,
            'title': self.title,
            'map': self.map_name,
            'winner': self.winner,
            'win_condition': self.win_condition,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'hivemind_url': self.hivemind_url,
            'users': self.users,
        }


@dataclass
class ChaptersData:
    """Loaded chapters file data."""
    filename: str
    video_id: str
    video_start_utc: str
    events: list[Event] = field(default_factory=list)
    games: dict[int, GameContext] = field(default_factory=dict)
    users: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, chapters_file: str) -> 'ChaptersData':
        """Load chapters data from a file."""
        # Handle different path formats
        if '/' in chapters_file:
            path = CHAPTERS_DIR / chapters_file
        else:
            # Try league_nights first, then tournaments
            path = CHAPTERS_DIR / "league_nights" / chapters_file
            if not path.exists():
                path = CHAPTERS_DIR / "tournaments" / chapters_file

        if not path.exists():
            raise FileNotFoundError(f"Chapters file not found: {chapters_file}")

        with open(path) as f:
            data = json.load(f)

        events = []
        games = {}

        for chapter in data.get('chapters', []):
            game_id = chapter.get('game_id')

            # Create game context
            games[game_id] = GameContext(
                game_id=game_id,
                title=chapter.get('title', f'Game {game_id}'),
                map_name=chapter.get('map', 'Unknown'),
                winner=chapter.get('winner', 'unknown'),
                win_condition=chapter.get('win_condition', 'unknown'),
                start_time=chapter.get('start_time', 0),
                end_time=chapter.get('end_time', 0),
                hivemind_url=chapter.get('hivemind_url', ''),
                users=chapter.get('users', {}),
            )

            # Extract events
            for evt in chapter.get('player_events', []):
                events.append(Event(
                    id=evt['id'],
                    game_id=game_id,
                    time=evt.get('time', 0),
                    event_type=evt.get('type', 'unknown'),
                    delta=evt.get('delta', 0),
                    positions=evt.get('positions', []),
                    values=evt.get('values', []),
                ))

        return cls(
            filename=chapters_file,
            video_id=data.get('video_id', ''),
            video_start_utc=data.get('video_start_utc', ''),
            events=events,
            games=games,
            users=data.get('users', {}),
        )

    def get_events_sorted_by_delta(self) -> list[Event]:
        """Get all events sorted by absolute delta (highest first)."""
        return sorted(self.events, key=lambda e: e.abs_delta, reverse=True)

    def get_events_for_game(self, game_id: int) -> list[Event]:
        """Get events for a specific game, sorted by delta."""
        game_events = [e for e in self.events if e.game_id == game_id]
        return sorted(game_events, key=lambda e: e.abs_delta, reverse=True)

    def get_unrated_events(self, rated_ids: set[int]) -> list[Event]:
        """Get events not yet rated, sorted by delta."""
        unrated = [e for e in self.events if e.id not in rated_ids]
        return sorted(unrated, key=lambda e: e.abs_delta, reverse=True)


def list_chapters_files() -> list[str]:
    """List available chapters files."""
    files = []

    # League nights
    league_dir = CHAPTERS_DIR / "league_nights"
    if league_dir.exists():
        for f in league_dir.glob("*.json"):
            files.append(f"league_nights/{f.name}")

    # Tournaments
    tourney_dir = CHAPTERS_DIR / "tournaments"
    if tourney_dir.exists():
        for f in tourney_dir.glob("*.json"):
            files.append(f"tournaments/{f.name}")

    return sorted(files)


# Rating scale labels
RATING_LABELS = {
    0: "Not a highlight",
    1: "Minor",
    2: "Good",
    3: "Excellent",
}

# Position names
POSITION_NAMES = {
    1: 'Gold Queen', 2: 'Blue Queen',
    3: 'Gold Stripes', 4: 'Gold Skull', 5: 'Gold Abs', 6: 'Gold Checkers',
    7: 'Blue Stripes', 8: 'Blue Skull', 9: 'Blue Abs', 10: 'Blue Checkers',
}


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def get_position_display(positions: list[int]) -> str:
    """Get display string for positions."""
    if not positions:
        return ""
    names = [POSITION_NAMES.get(p, f"P{p}") for p in positions]
    return ", ".join(names)
