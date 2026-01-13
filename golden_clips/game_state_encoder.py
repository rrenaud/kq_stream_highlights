"""
Game state encoder for Killer Queen clips.

Ported from https://github.com/rrenaud/kquity
Tracks game state from events and produces feature vectors.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


SCREEN_WIDTH = 1920


class Team(Enum):
    BLUE = 0
    GOLD = 1


class ContestableState(Enum):
    NEUTRAL = 0
    BLUE = 1
    GOLD = 2


class Map(Enum):
    DAY = "map_day"
    NIGHT = "map_night"
    DUSK = "map_dusk"
    TWILIGHT = "map_twilight"


# Position ID mappings (from kquity)
# Blue team: 1=queen, 2-5=workers
# Gold team: 6=queen, 7-10=workers
def position_id_to_team(position_id: int) -> Team:
    return Team.BLUE if position_id <= 5 else Team.GOLD


def position_id_to_worker_index(position_id: int) -> int:
    if position_id <= 5:
        return position_id - 2  # 2-5 -> 0-3
    return position_id - 7  # 7-10 -> 0-3


def is_queen(position_id: int) -> bool:
    return position_id == 1 or position_id == 6


@dataclass
class WorkerState:
    """State of a single worker bee."""
    has_speed: bool = False
    has_wings: bool = False
    has_food: bool = False
    is_bot: bool = False

    def power(self) -> float:
        """Power level for sorting workers."""
        return self.has_wings + self.has_speed * 0.5 + self.has_food * 0.25

    def reset(self):
        """Reset worker state (after death)."""
        self.has_speed = False
        self.has_wings = False
        self.has_food = False


@dataclass
class TeamState:
    """State of one team."""
    eggs: int = 2
    food_deposited: list = field(default_factory=lambda: [False] * 12)
    workers: list = field(default_factory=lambda: [WorkerState() for _ in range(4)])

    def num_food_deposited(self) -> int:
        return sum(self.food_deposited)

    def num_warriors(self) -> int:
        return sum(1 for w in self.workers if w.has_wings)

    def num_speed_warriors(self) -> int:
        return sum(1 for w in self.workers if w.has_wings and w.has_speed)


@dataclass
class SnailState:
    """Inferred snail position and movement."""
    VANILLA_SPEED = 20.896  # pixels per second
    SPEED_SNAIL = 28.21  # pixels per second

    position: float = SCREEN_WIDTH / 2
    velocity: float = 0.0
    last_timestamp: float = 0.0
    rider_position_id: Optional[int] = None

    def inferred_position(self, timestamp: float) -> float:
        """Get snail position at given timestamp."""
        return self.position + (timestamp - self.last_timestamp) * self.velocity

    def normalized_position(self, timestamp: float) -> float:
        """Get snail position normalized to [-0.5, 0.5]."""
        return self.inferred_position(timestamp) / SCREEN_WIDTH - 0.5


@dataclass
class GameState:
    """Full game state."""
    map_name: str = "map_day"
    gold_on_left: bool = True
    teams: list = field(default_factory=lambda: [TeamState(), TeamState()])
    maiden_states: list = field(default_factory=lambda: [ContestableState.NEUTRAL] * 5)
    snail: SnailState = field(default_factory=SnailState)
    berries_available: int = 66
    game_start_timestamp: float = 0.0

    def get_team(self, team: Team) -> TeamState:
        return self.teams[team.value]

    def get_worker(self, position_id: int) -> Optional[WorkerState]:
        if is_queen(position_id):
            return None
        team = position_id_to_team(position_id)
        idx = position_id_to_worker_index(position_id)
        if 0 <= idx < 4:
            return self.get_team(team).workers[idx]
        return None


def apply_event(state: GameState, event: dict) -> None:
    """Apply an event to modify game state."""
    event_type = event.get("event_type", "")
    values = event.get("values", [])

    if event_type == "gamestart":
        if len(values) >= 2:
            state.map_name = values[0]
            state.gold_on_left = values[1] == "True"
        state.game_start_timestamp = _parse_event_timestamp(event)

    elif event_type == "spawn":
        if len(values) >= 2:
            position_id = int(values[0])
            is_bot = values[1] == "True"
            worker = state.get_worker(position_id)
            if worker:
                worker.is_bot = is_bot

    elif event_type == "carryFood":
        if len(values) >= 1:
            position_id = int(values[0])
            worker = state.get_worker(position_id)
            if worker:
                worker.has_food = True

    elif event_type == "berryDeposit":
        if len(values) >= 3:
            position_id = int(values[2])
            team = position_id_to_team(position_id)
            team_state = state.get_team(team)
            # Find first empty deposit slot
            for i, deposited in enumerate(team_state.food_deposited):
                if not deposited:
                    team_state.food_deposited[i] = True
                    break
            worker = state.get_worker(position_id)
            if worker:
                worker.has_food = False
            state.berries_available -= 1

    elif event_type == "berryKickIn":
        if len(values) >= 3:
            position_id = int(values[2])
            team = position_id_to_team(position_id)
            team_state = state.get_team(team)
            for i, deposited in enumerate(team_state.food_deposited):
                if not deposited:
                    team_state.food_deposited[i] = True
                    break
            state.berries_available -= 1

    elif event_type == "useMaiden":
        if len(values) >= 4:
            maiden_type = values[2]
            position_id = int(values[3])
            worker = state.get_worker(position_id)
            if worker:
                if maiden_type == "maiden_speed":
                    worker.has_speed = True
                elif maiden_type == "maiden_wings":
                    worker.has_wings = True
                worker.has_food = False

    elif event_type == "blessMaiden":
        if len(values) >= 3:
            x, y = int(values[0]), int(values[1])
            color = values[2]
            maiden_idx = _get_maiden_index(x, y)
            if maiden_idx is not None:
                if color == "Blue":
                    state.maiden_states[maiden_idx] = ContestableState.BLUE
                elif color in ("Red", "Gold"):
                    state.maiden_states[maiden_idx] = ContestableState.GOLD
                else:
                    state.maiden_states[maiden_idx] = ContestableState.NEUTRAL

    elif event_type == "playerKill":
        if len(values) >= 5:
            killed_id = int(values[3])
            unit_type = values[4]
            team = position_id_to_team(killed_id)
            if unit_type == "Queen":
                state.get_team(team).eggs -= 1
            else:
                worker = state.get_worker(killed_id)
                if worker:
                    worker.reset()

    elif event_type == "getOnSnail":
        if len(values) >= 3:
            x = int(values[0])
            position_id = int(values[2])
            ts = _parse_event_timestamp(event)
            state.snail.position = x
            state.snail.last_timestamp = ts
            state.snail.rider_position_id = position_id
            # Calculate velocity
            worker = state.get_worker(position_id)
            base_speed = SnailState.SPEED_SNAIL if (worker and worker.has_speed) else SnailState.VANILLA_SPEED
            # Direction depends on team and gold_on_left
            team = position_id_to_team(position_id)
            direction = _snail_direction(state.gold_on_left, team)
            state.snail.velocity = base_speed * direction

    elif event_type == "getOffSnail":
        if len(values) >= 1:
            x = int(values[0]) if values[0] else state.snail.inferred_position(_parse_event_timestamp(event))
            ts = _parse_event_timestamp(event)
            state.snail.position = x
            state.snail.last_timestamp = ts
            state.snail.velocity = 0
            state.snail.rider_position_id = None

    elif event_type == "snailEat":
        if len(values) >= 3:
            eaten_id = int(values[2])
            worker = state.get_worker(eaten_id)
            if worker:
                worker.reset()

    elif event_type == "snailEscape":
        ts = _parse_event_timestamp(event)
        state.snail.position = state.snail.inferred_position(ts)
        state.snail.last_timestamp = ts
        state.snail.velocity = 0
        state.snail.rider_position_id = None


def _parse_event_timestamp(event: dict) -> float:
    """Parse timestamp from event."""
    ts_str = event.get("timestamp", "")
    if not ts_str:
        return 0.0
    from datetime import datetime, timezone
    if ts_str.endswith("+00:00"):
        ts_str = ts_str.replace("+00:00", "")
    try:
        dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _snail_direction(gold_on_left: bool, team: Team) -> int:
    """Get snail movement direction multiplier."""
    if gold_on_left:
        return -1 if team == Team.GOLD else 1
    else:
        return 1 if team == Team.GOLD else -1


def _get_maiden_index(x: int, y: int) -> Optional[int]:
    """Get maiden index from coordinates (approximate mapping)."""
    # Simplified mapping based on typical positions
    # Maidens are roughly at: (170, 740), (700, 260), (960, 700), (1220, 260), (1750, 740)
    if y > 600:  # Bottom row
        if x < 400:
            return 0
        elif x < 1100:
            return 2
        else:
            return 4
    else:  # Top row
        if x < 1000:
            return 1
        else:
            return 3


def vectorize_worker(worker: WorkerState) -> np.ndarray:
    """Convert worker state to feature vector."""
    return np.array([
        float(worker.is_bot),
        float(worker.has_food),
        float(worker.has_speed),
        float(worker.has_wings),
    ])


def vectorize_team(team: TeamState) -> np.ndarray:
    """Convert team state to feature vector."""
    eggs = float(team.eggs)
    num_food = float(team.num_food_deposited())
    num_vanilla_warriors = float(sum(1 for w in team.workers if w.has_wings and not w.has_speed))
    num_speed_warriors = float(sum(1 for w in team.workers if w.has_wings and w.has_speed))

    parts = [np.array([eggs, num_food, num_vanilla_warriors, num_speed_warriors])]
    for worker in sorted(team.workers, key=WorkerState.power):
        parts.append(vectorize_worker(worker))

    return np.concatenate(parts)


def vectorize_maidens(maidens: list) -> np.ndarray:
    """Convert maiden states to feature vector."""
    def encode(state: ContestableState) -> float:
        if state == ContestableState.NEUTRAL:
            return 0.0
        elif state == ContestableState.BLUE:
            return 1.0
        else:
            return -1.0

    return np.array([encode(m) for m in maidens])


def vectorize_map(map_name: str) -> np.ndarray:
    """One-hot encode the map."""
    maps = ["map_day", "map_night", "map_dusk", "map_twilight"]
    return np.array([1.0 if map_name == m else 0.0 for m in maps])


def vectorize_snail(state: GameState, timestamp: float) -> np.ndarray:
    """Convert snail state to feature vector."""
    symmetry = 1.0 if state.gold_on_left else -1.0
    pos = state.snail.normalized_position(timestamp) * symmetry
    vel = (state.snail.velocity / SnailState.SPEED_SNAIL) * symmetry
    return np.array([pos, vel])


def vectorize_game_state(state: GameState, timestamp: float) -> np.ndarray:
    """Convert full game state to feature vector.

    Returns ~35 dimensional feature vector:
    - Blue team: 4 + 4*4 = 20 features
    - Gold team: 20 features
    - Maidens: 5 features
    - Map: 4 features
    - Snail: 2 features
    - Berries: 1 feature
    Total: 52 features
    """
    blue = vectorize_team(state.get_team(Team.BLUE))
    gold = vectorize_team(state.get_team(Team.GOLD))

    return np.concatenate([
        blue,
        gold,
        vectorize_maidens(state.maiden_states),
        vectorize_map(state.map_name),
        vectorize_snail(state, timestamp),
        np.array([state.berries_available / 70.0]),
    ])


def build_game_state(events: list) -> tuple[GameState, float]:
    """Build game state from a list of events.

    Returns (final_state, last_timestamp).
    """
    state = GameState()
    last_ts = 0.0

    for event in events:
        apply_event(state, event)
        ts = _parse_event_timestamp(event)
        if ts > last_ts:
            last_ts = ts

    return state, last_ts


def get_state_at_time(events: list, target_utc: float) -> tuple[GameState, float]:
    """Build game state up to a specific UTC timestamp.

    Returns (state_at_time, closest_event_timestamp).
    """
    state = GameState()
    closest_ts = 0.0

    for event in events:
        event_ts = _parse_event_timestamp(event)
        if event_ts > target_utc:
            break
        apply_event(state, event)
        closest_ts = event_ts

    return state, closest_ts


# Feature vector size for reference
FEATURE_VECTOR_SIZE = 52
