"""
Track player statistics from HiveMind game events.
"""

from dataclasses import dataclass
from hivemind_api import fetch_game_events, GameEvent


@dataclass
class PlayerStats:
    kills: int = 0
    deaths: int = 0
    queen_kills: int = 0

    @property
    def kd_ratio(self) -> float | None:
        if self.deaths == 0:
            return float('inf') if self.kills > 0 else None
        return self.kills / self.deaths


class StatTracker:
    """Track kill/death stats for all positions in a game."""

    def __init__(self, events: list[GameEvent]):
        self.stats: dict[int, PlayerStats] = {pos: PlayerStats() for pos in range(1, 11)}
        self._process_events(events)

    def _process_events(self, events: list[GameEvent]):
        for event in events:
            if event.event_type == 'playerKill':
                self._process_kill(event)
            elif event.event_type == 'snailEat':
                self._process_snail_eat(event)

    def _process_kill(self, event: GameEvent):
        values = event.values
        if len(values) < 4:
            return

        try:
            killer = int(values[2])
            victim = int(values[3])
        except (ValueError, TypeError):
            return

        if 1 <= killer <= 10:
            self.stats[killer].kills += 1
            # Queen kill if victim is position 1 (gold queen) or 2 (blue queen)
            if victim in (1, 2):
                self.stats[killer].queen_kills += 1

        if 1 <= victim <= 10:
            self.stats[victim].deaths += 1

    def _process_snail_eat(self, event: GameEvent):
        values = event.values
        if len(values) < 4:
            return

        try:
            rider = int(values[2])  # snail rider is the killer
            victim = int(values[3])
        except (ValueError, TypeError):
            return

        if 1 <= rider <= 10:
            self.stats[rider].kills += 1

        if 1 <= victim <= 10:
            self.stats[victim].deaths += 1

    def get_stats(self, position: int) -> PlayerStats:
        return self.stats.get(position, PlayerStats())

    @classmethod
    def from_game_id(cls, game_id: int) -> 'StatTracker':
        """Fetch events for a game and create a StatTracker."""
        events = fetch_game_events(game_id, verbose=False)
        return cls(events)
