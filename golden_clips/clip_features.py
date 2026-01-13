"""
Feature extraction for highlight clips.

Extracts ML features from clips using:
1. Win probability changes
2. Event counts
3. Game state vectors (from game_state_encoder)
4. Duration
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import game_state_encoder


BASE_DIR = Path(__file__).parent.parent
CLIP_INFO_FILE = Path(__file__).parent / "clip_info.jsonl"
ALIGNMENTS_FILE = Path(__file__).parent / "tournament_alignments.jsonl"
RATINGS_FILE = Path(__file__).parent / "highlight_ratings.jsonl"
CANDIDATE_RATINGS_FILE = Path(__file__).parent / "candidate_ratings.jsonl"
CANDIDATES_FILE = Path(__file__).parent / "clip_candidates.jsonl"
GAME_EVENTS_DIR = BASE_DIR / "cache" / "game_events"
GAME_DETAIL_DIR = BASE_DIR / "cache" / "game_detail"


@dataclass
class ClipFeatures:
    """Features extracted from a clip."""
    clip_id: str

    # Duration
    duration_seconds: float

    # Win probability features
    win_prob_start: Optional[float]
    win_prob_end: Optional[float]
    win_prob_delta: Optional[float]
    win_prob_max_swing: Optional[float]
    win_prob_volatility: Optional[float]
    win_prob_min: Optional[float]
    win_prob_max: Optional[float]

    # Event counts
    queen_kills: int
    worker_kills: int
    total_kills: int
    snail_eats: int
    snail_escapes: int
    berry_deposits: int
    berry_kick_ins: int
    maiden_uses: int
    victory_in_clip: bool

    # Event density
    events_per_second: float
    kills_per_second: float

    # Game state features (from kquity encoding)
    state_vector_start: Optional[np.ndarray]
    state_vector_end: Optional[np.ndarray]
    state_vector_delta: Optional[np.ndarray]

    # Derived game state features
    blue_eggs_start: Optional[int]
    gold_eggs_start: Optional[int]
    blue_eggs_end: Optional[int]
    gold_eggs_end: Optional[int]
    blue_warriors_start: int
    gold_warriors_start: int
    blue_warriors_end: int
    gold_warriors_end: int

    def to_feature_vector(self) -> np.ndarray:
        """Convert to flat feature vector for ML.

        Always returns same-sized vector (27 + 52 = 79 features).
        """
        features = [
            self.duration_seconds,
            self.win_prob_start or 0.5,
            self.win_prob_end or 0.5,
            self.win_prob_delta or 0.0,
            self.win_prob_max_swing or 0.0,
            self.win_prob_volatility or 0.0,
            self.win_prob_min or 0.5,
            self.win_prob_max or 0.5,
            float(self.queen_kills),
            float(self.worker_kills),
            float(self.total_kills),
            float(self.snail_eats),
            float(self.snail_escapes),
            float(self.berry_deposits),
            float(self.berry_kick_ins),
            float(self.maiden_uses),
            float(self.victory_in_clip),
            self.events_per_second,
            self.kills_per_second,
            float(self.blue_eggs_start or 2),
            float(self.gold_eggs_start or 2),
            float(self.blue_eggs_end or 2),
            float(self.gold_eggs_end or 2),
            float(self.blue_warriors_start),
            float(self.gold_warriors_start),
            float(self.blue_warriors_end),
            float(self.gold_warriors_end),
        ]

        # Add state vector delta (zeros if not available)
        if self.state_vector_delta is not None:
            features.extend(self.state_vector_delta.tolist())
        else:
            features.extend([0.0] * game_state_encoder.FEATURE_VECTOR_SIZE)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        """Get names of features in the feature vector."""
        names = [
            "duration_seconds",
            "win_prob_start",
            "win_prob_end",
            "win_prob_delta",
            "win_prob_max_swing",
            "win_prob_volatility",
            "win_prob_min",
            "win_prob_max",
            "queen_kills",
            "worker_kills",
            "total_kills",
            "snail_eats",
            "snail_escapes",
            "berry_deposits",
            "berry_kick_ins",
            "maiden_uses",
            "victory_in_clip",
            "events_per_second",
            "kills_per_second",
            "blue_eggs_start",
            "gold_eggs_start",
            "blue_eggs_end",
            "gold_eggs_end",
            "blue_warriors_start",
            "gold_warriors_start",
            "blue_warriors_end",
            "gold_warriors_end",
        ]
        # Add state delta feature names
        for i in range(game_state_encoder.FEATURE_VECTOR_SIZE):
            names.append(f"state_delta_{i}")
        return names


def load_clips() -> list[dict]:
    """Load clips from clip_info.jsonl."""
    clips = []
    if CLIP_INFO_FILE.exists():
        with open(CLIP_INFO_FILE) as f:
            for line in f:
                if line.strip():
                    clips.append(json.loads(line))
    return clips


def load_alignments() -> dict[str, dict]:
    """Load tournament alignments."""
    alignments = {}
    if ALIGNMENTS_FILE.exists():
        with open(ALIGNMENTS_FILE) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    video_id = data.get("video_id")
                    if video_id:
                        alignments[video_id] = data
    return alignments


def load_ratings() -> dict[str, dict]:
    """Load existing ratings."""
    ratings = {}
    if RATINGS_FILE.exists():
        with open(RATINGS_FILE) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    clip_id = data.get("clip_id")
                    if clip_id:
                        ratings[clip_id] = data
    return ratings


def load_candidate_ratings() -> dict[str, dict]:
    """Load candidate ratings."""
    ratings = {}
    if CANDIDATE_RATINGS_FILE.exists():
        with open(CANDIDATE_RATINGS_FILE) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    candidate_id = data.get("candidate_id")
                    if candidate_id:
                        ratings[candidate_id] = data
    return ratings


def load_candidates() -> list[dict]:
    """Load candidates from clip_candidates.jsonl."""
    candidates = []
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE) as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    return candidates


def load_game_details() -> dict[int, dict]:
    """Load all game details."""
    details = {}
    for f in GAME_DETAIL_DIR.glob("*.json"):
        game_id = int(f.stem)
        with open(f) as fp:
            details[game_id] = json.load(fp)
    return details


def load_game_events(game_id: int) -> list[dict]:
    """Load events for a game."""
    events_file = GAME_EVENTS_DIR / f"{game_id}.jsonl"
    if not events_file.exists():
        return []

    events = []
    with open(events_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    if ts_str.endswith("+00:00"):
        ts_str = ts_str.replace("+00:00", "")
    if "." in ts_str:
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def get_events_for_clip(
    clip: dict,
    alignment: dict,
    game_details: dict[int, dict]
) -> list[dict]:
    """Get all events that occurred during a clip's time range."""
    # Parse alignment data
    gamestart_utc = parse_timestamp(alignment["gamestart_utc"])
    video_offset = alignment["video_timestamp_seconds"]

    # Convert clip timestamps to UTC
    clip_start_offset = clip["start_seconds"] - video_offset
    clip_end_offset = clip["end_seconds"] - video_offset

    clip_utc_start = gamestart_utc + timedelta(seconds=clip_start_offset)
    clip_utc_end = gamestart_utc + timedelta(seconds=clip_end_offset)

    matching_events = []

    # Find games near this time
    for game_id, game in game_details.items():
        game_start = game.get("start_time", "")
        if not game_start:
            continue

        try:
            game_start_dt = parse_timestamp(game_start)
        except (ValueError, TypeError):
            continue

        # Check if game is within ~2 hours of clip time
        time_diff = (game_start_dt - clip_utc_start).total_seconds()
        if abs(time_diff) > 7200:
            continue

        # Load events for this game
        events = load_game_events(game_id)
        for event in events:
            event_ts = event.get("timestamp", "")
            if not event_ts:
                continue

            try:
                event_dt = parse_timestamp(event_ts)
            except (ValueError, TypeError):
                continue

            if clip_utc_start <= event_dt <= clip_utc_end:
                event_with_time = {
                    **event,
                    "utc_datetime": event_dt,
                    "game_id": game_id,
                }
                matching_events.append(event_with_time)

    matching_events.sort(key=lambda e: e.get("timestamp", ""))
    return matching_events


def extract_win_prob_features(events: list[dict]) -> dict:
    """Extract win probability features from events."""
    win_probs = []
    for event in events:
        wp = event.get("win_probability")
        if wp is not None:
            try:
                win_probs.append(float(wp))
            except (ValueError, TypeError):
                pass

    if not win_probs:
        return {
            "win_prob_start": None,
            "win_prob_end": None,
            "win_prob_delta": None,
            "win_prob_max_swing": None,
            "win_prob_volatility": None,
            "win_prob_min": None,
            "win_prob_max": None,
        }

    start = win_probs[0]
    end = win_probs[-1]
    delta = end - start

    # Max swing: largest change from any point to any later point
    max_swing = 0.0
    for i, wp1 in enumerate(win_probs):
        for wp2 in win_probs[i + 1:]:
            swing = abs(wp2 - wp1)
            if swing > max_swing:
                max_swing = swing

    # Volatility: standard deviation
    volatility = float(np.std(win_probs)) if len(win_probs) > 1 else 0.0

    return {
        "win_prob_start": start,
        "win_prob_end": end,
        "win_prob_delta": delta,
        "win_prob_max_swing": max_swing,
        "win_prob_volatility": volatility,
        "win_prob_min": min(win_probs),
        "win_prob_max": max(win_probs),
    }


def count_events(events: list[dict]) -> dict:
    """Count different event types."""
    counts = {
        "queen_kills": 0,
        "worker_kills": 0,
        "total_kills": 0,
        "snail_eats": 0,
        "snail_escapes": 0,
        "berry_deposits": 0,
        "berry_kick_ins": 0,
        "maiden_uses": 0,
        "victory_in_clip": False,
    }

    for event in events:
        event_type = event.get("event_type", "")
        values = event.get("values", [])

        if event_type == "playerKill":
            counts["total_kills"] += 1
            if len(values) >= 5 and values[4] == "Queen":
                counts["queen_kills"] += 1
            else:
                counts["worker_kills"] += 1
        elif event_type == "snailEat":
            counts["snail_eats"] += 1
        elif event_type == "snailEscape":
            counts["snail_escapes"] += 1
        elif event_type == "berryDeposit":
            counts["berry_deposits"] += 1
        elif event_type == "berryKickIn":
            counts["berry_kick_ins"] += 1
        elif event_type == "useMaiden":
            counts["maiden_uses"] += 1
        elif event_type == "victory":
            counts["victory_in_clip"] = True

    return counts


def extract_game_state_features(events: list[dict]) -> dict:
    """Extract game state features at start and end of clip."""
    if not events:
        return {
            "state_vector_start": None,
            "state_vector_end": None,
            "state_vector_delta": None,
            "blue_eggs_start": None,
            "gold_eggs_start": None,
            "blue_eggs_end": None,
            "gold_eggs_end": None,
            "blue_warriors_start": 0,
            "gold_warriors_start": 0,
            "blue_warriors_end": 0,
            "gold_warriors_end": 0,
        }

    # Group events by game
    events_by_game = {}
    for event in events:
        game_id = event.get("game_id")
        if game_id:
            if game_id not in events_by_game:
                events_by_game[game_id] = []
            events_by_game[game_id].append(event)

    # Use the game with most events
    if not events_by_game:
        return {
            "state_vector_start": None,
            "state_vector_end": None,
            "state_vector_delta": None,
            "blue_eggs_start": None,
            "gold_eggs_start": None,
            "blue_eggs_end": None,
            "gold_eggs_end": None,
            "blue_warriors_start": 0,
            "gold_warriors_start": 0,
            "blue_warriors_end": 0,
            "gold_warriors_end": 0,
        }

    main_game_id = max(events_by_game.keys(), key=lambda g: len(events_by_game[g]))
    game_events = events_by_game[main_game_id]

    # Load all events for this game to build proper state
    all_game_events = load_game_events(main_game_id)

    if not all_game_events:
        return {
            "state_vector_start": None,
            "state_vector_end": None,
            "state_vector_delta": None,
            "blue_eggs_start": None,
            "gold_eggs_start": None,
            "blue_eggs_end": None,
            "gold_eggs_end": None,
            "blue_warriors_start": 0,
            "gold_warriors_start": 0,
            "blue_warriors_end": 0,
            "gold_warriors_end": 0,
        }

    # Get timestamps for start and end of clip events
    first_event_ts = game_events[0].get("utc_datetime")
    last_event_ts = game_events[-1].get("utc_datetime")

    if not first_event_ts or not last_event_ts:
        return {
            "state_vector_start": None,
            "state_vector_end": None,
            "state_vector_delta": None,
            "blue_eggs_start": None,
            "gold_eggs_start": None,
            "blue_eggs_end": None,
            "gold_eggs_end": None,
            "blue_warriors_start": 0,
            "gold_warriors_start": 0,
            "blue_warriors_end": 0,
            "gold_warriors_end": 0,
        }

    # Build state at start of clip
    start_ts_utc = first_event_ts.timestamp()
    state_start, _ = game_state_encoder.get_state_at_time(all_game_events, start_ts_utc)
    vec_start = game_state_encoder.vectorize_game_state(state_start, start_ts_utc)

    # Build state at end of clip
    end_ts_utc = last_event_ts.timestamp()
    state_end, _ = game_state_encoder.get_state_at_time(all_game_events, end_ts_utc)
    vec_end = game_state_encoder.vectorize_game_state(state_end, end_ts_utc)

    vec_delta = vec_end - vec_start

    return {
        "state_vector_start": vec_start,
        "state_vector_end": vec_end,
        "state_vector_delta": vec_delta,
        "blue_eggs_start": state_start.get_team(game_state_encoder.Team.BLUE).eggs,
        "gold_eggs_start": state_start.get_team(game_state_encoder.Team.GOLD).eggs,
        "blue_eggs_end": state_end.get_team(game_state_encoder.Team.BLUE).eggs,
        "gold_eggs_end": state_end.get_team(game_state_encoder.Team.GOLD).eggs,
        "blue_warriors_start": state_start.get_team(game_state_encoder.Team.BLUE).num_warriors(),
        "gold_warriors_start": state_start.get_team(game_state_encoder.Team.GOLD).num_warriors(),
        "blue_warriors_end": state_end.get_team(game_state_encoder.Team.BLUE).num_warriors(),
        "gold_warriors_end": state_end.get_team(game_state_encoder.Team.GOLD).num_warriors(),
    }


def extract_clip_features(
    clip: dict,
    alignments: dict[str, dict],
    game_details: dict[int, dict]
) -> Optional[ClipFeatures]:
    """Extract all features for a clip."""
    clip_id = clip.get("clip_id", "")
    video_id = clip.get("source_video_id")
    alignment = alignments.get(video_id)

    if not alignment:
        return None

    duration = clip.get("duration", 0) or (clip.get("end_seconds", 0) - clip.get("start_seconds", 0))

    # Get events for clip
    events = get_events_for_clip(clip, alignment, game_details)

    # Extract features
    wp_features = extract_win_prob_features(events)
    event_counts = count_events(events)
    state_features = extract_game_state_features(events)

    # Calculate density metrics
    events_per_second = len(events) / duration if duration > 0 else 0
    kills_per_second = event_counts["total_kills"] / duration if duration > 0 else 0

    return ClipFeatures(
        clip_id=clip_id,
        duration_seconds=duration,
        win_prob_start=wp_features["win_prob_start"],
        win_prob_end=wp_features["win_prob_end"],
        win_prob_delta=wp_features["win_prob_delta"],
        win_prob_max_swing=wp_features["win_prob_max_swing"],
        win_prob_volatility=wp_features["win_prob_volatility"],
        win_prob_min=wp_features["win_prob_min"],
        win_prob_max=wp_features["win_prob_max"],
        queen_kills=event_counts["queen_kills"],
        worker_kills=event_counts["worker_kills"],
        total_kills=event_counts["total_kills"],
        snail_eats=event_counts["snail_eats"],
        snail_escapes=event_counts["snail_escapes"],
        berry_deposits=event_counts["berry_deposits"],
        berry_kick_ins=event_counts["berry_kick_ins"],
        maiden_uses=event_counts["maiden_uses"],
        victory_in_clip=event_counts["victory_in_clip"],
        events_per_second=events_per_second,
        kills_per_second=kills_per_second,
        state_vector_start=state_features["state_vector_start"],
        state_vector_end=state_features["state_vector_end"],
        state_vector_delta=state_features["state_vector_delta"],
        blue_eggs_start=state_features["blue_eggs_start"],
        gold_eggs_start=state_features["gold_eggs_start"],
        blue_eggs_end=state_features["blue_eggs_end"],
        gold_eggs_end=state_features["gold_eggs_end"],
        blue_warriors_start=state_features["blue_warriors_start"],
        gold_warriors_start=state_features["gold_warriors_start"],
        blue_warriors_end=state_features["blue_warriors_end"],
        gold_warriors_end=state_features["gold_warriors_end"],
    )


def extract_all_clip_features() -> list[ClipFeatures]:
    """Extract features for all clips with alignments."""
    clips = load_clips()
    alignments = load_alignments()
    game_details = load_game_details()

    features = []
    for clip in clips:
        clip_features = extract_clip_features(clip, alignments, game_details)
        if clip_features:
            features.append(clip_features)

    return features


def extract_candidate_features(
    candidate: dict,
    game_details: dict[int, dict],
) -> Optional[ClipFeatures]:
    """Extract features from a candidate (similar to extract_clip_features)."""
    candidate_id = candidate.get("candidate_id", "")
    game_id = candidate.get("game_id")
    start_utc_str = candidate.get("start_utc")
    end_utc_str = candidate.get("end_utc")

    if not all([game_id, start_utc_str, end_utc_str]):
        return None

    # Parse timestamps
    try:
        start_utc = parse_timestamp(start_utc_str)
        end_utc = parse_timestamp(end_utc_str)
    except (ValueError, TypeError):
        return None

    duration = (end_utc - start_utc).total_seconds()

    # Load events for this game
    events = load_game_events(game_id)
    if not events:
        return None

    # Filter events within candidate window
    clip_events = []
    for event in events:
        ts_str = event.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = parse_timestamp(ts_str)
        except (ValueError, TypeError):
            continue

        if start_utc <= ts <= end_utc:
            event["utc_datetime"] = ts
            event["game_id"] = game_id
            clip_events.append(event)

    if not clip_events:
        return None

    # Extract features
    wp_features = extract_win_prob_features(clip_events)
    event_counts = count_events(clip_events)
    state_features = extract_game_state_features(clip_events)

    # Calculate density metrics
    events_per_second = len(clip_events) / duration if duration > 0 else 0
    kills_per_second = event_counts["total_kills"] / duration if duration > 0 else 0

    return ClipFeatures(
        clip_id=candidate_id,
        duration_seconds=duration,
        win_prob_start=wp_features["win_prob_start"],
        win_prob_end=wp_features["win_prob_end"],
        win_prob_delta=wp_features["win_prob_delta"],
        win_prob_max_swing=wp_features["win_prob_max_swing"],
        win_prob_volatility=wp_features["win_prob_volatility"],
        win_prob_min=wp_features["win_prob_min"],
        win_prob_max=wp_features["win_prob_max"],
        queen_kills=event_counts["queen_kills"],
        worker_kills=event_counts["worker_kills"],
        total_kills=event_counts["total_kills"],
        snail_eats=event_counts["snail_eats"],
        snail_escapes=event_counts["snail_escapes"],
        berry_deposits=event_counts["berry_deposits"],
        berry_kick_ins=event_counts["berry_kick_ins"],
        maiden_uses=event_counts["maiden_uses"],
        victory_in_clip=event_counts["victory_in_clip"],
        events_per_second=events_per_second,
        kills_per_second=kills_per_second,
        state_vector_start=state_features["state_vector_start"],
        state_vector_end=state_features["state_vector_end"],
        state_vector_delta=state_features["state_vector_delta"],
        blue_eggs_start=state_features["blue_eggs_start"],
        gold_eggs_start=state_features["gold_eggs_start"],
        blue_eggs_end=state_features["blue_eggs_end"],
        gold_eggs_end=state_features["gold_eggs_end"],
        blue_warriors_start=state_features["blue_warriors_start"],
        gold_warriors_start=state_features["gold_warriors_start"],
        blue_warriors_end=state_features["blue_warriors_end"],
        gold_warriors_end=state_features["gold_warriors_end"],
    )


def get_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Get training data matrix (X, y, ids) for rated clips and candidates."""
    clips = load_clips()
    alignments = load_alignments()
    game_details = load_game_details()
    clip_ratings = load_ratings()
    candidate_ratings = load_candidate_ratings()

    X_list = []
    y_list = []
    ids = []

    # Process clip ratings
    for clip in clips:
        clip_id = clip.get("clip_id", "")
        rating_data = clip_ratings.get(clip_id)
        if not rating_data:
            continue

        rating = rating_data.get("rating")
        if rating is None:
            continue

        features = extract_clip_features(clip, alignments, game_details)
        if features is None:
            continue

        X_list.append(features.to_feature_vector())
        y_list.append(float(rating))
        ids.append(clip_id)

    # Process candidate ratings (requires stored metadata)
    for candidate_id, rating_data in candidate_ratings.items():
        rating = rating_data.get("rating")
        if rating is None:
            continue

        # Require stored metadata (game_id, start_utc, end_utc)
        if "game_id" not in rating_data or "start_utc" not in rating_data or "end_utc" not in rating_data:
            continue

        candidate = {
            "candidate_id": candidate_id,
            "game_id": rating_data["game_id"],
            "start_utc": rating_data["start_utc"],
            "end_utc": rating_data["end_utc"],
        }

        features = extract_candidate_features(candidate, game_details)
        if features is None:
            continue

        X_list.append(features.to_feature_vector())
        y_list.append(float(rating))
        ids.append(candidate_id)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.vstack(X_list), np.array(y_list), ids


if __name__ == "__main__":
    print("Loading data...")
    features = extract_all_clip_features()
    print(f"Extracted features for {len(features)} clips")

    print("\nGetting training data...")
    X, y, clip_ids = get_training_data()
    print(f"Training samples: {len(y)}")

    if len(y) > 0:
        print(f"Feature vector size: {X.shape[1]}")
        print(f"Rating distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        print("\nSample features for first clip:")
        print(f"  Clip ID: {clip_ids[0]}")
        for i, name in enumerate(ClipFeatures.feature_names()[:27]):
            print(f"  {name}: {X[0, i]:.4f}")
