#!/usr/bin/env python3
"""
Generate 100 high-quality diverse clip candidates from game event data.

Scans all games for dramatic moments, scores them with the ML model,
and selects 100 diverse candidates for human rating.

Usage:
    python -m golden_clips.generate_candidates
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import clip_features, train_rater


BASE_DIR = Path(__file__).parent.parent
GAME_EVENTS_DIR = BASE_DIR / "cache" / "game_events"
GAME_DETAIL_DIR = BASE_DIR / "cache" / "game_detail"
ALIGNMENTS_FILE = Path(__file__).parent / "tournament_alignments.jsonl"
CANDIDATES_FILE = Path(__file__).parent / "clip_candidates.jsonl"


# Trigger window configuration (seconds)
TRIGGER_CONFIG = {
    "victory": {"pre": 3.0, "post": 3.0},
    "queen_kill": {"pre": 5.0, "post": 8.0},
    "snail_eat": {"pre": 5.0, "post": 5.0},
    "snail_escape": {"pre": 5.0, "post": 5.0},
    "combat_burst": {"pre": 2.0, "post": 3.0},  # Added to detected window
    "win_prob_spike": {"pre": 3.0, "post": 5.0},
}

# Diversity limits
MAX_PER_TRIGGER_TYPE = 25
MAX_PER_TOURNAMENT = 15
MAX_PER_GAME = 3
TARGET_CANDIDATES = 100

# Detection thresholds
COMBAT_BURST_MIN_KILLS = 3
COMBAT_BURST_WINDOW_SECONDS = 5.0
WIN_PROB_SPIKE_THRESHOLD = 0.20


@dataclass
class Candidate:
    """A clip candidate."""
    candidate_id: str
    game_id: int
    tournament_id: Optional[int]
    start_utc: datetime
    end_utc: datetime
    trigger_type: str
    trigger_event_id: Optional[int] = None
    predicted_score: float = 0.0
    has_video: bool = False
    video_id: Optional[str] = None
    video_start_seconds: Optional[float] = None
    video_end_seconds: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "game_id": int(self.game_id),
            "tournament_id": int(self.tournament_id) if self.tournament_id else None,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "duration_seconds": float(self.duration_seconds),
            "trigger_type": self.trigger_type,
            "predicted_score": float(round(self.predicted_score, 3)),
            "has_video": self.has_video,
            "video_id": self.video_id,
            "video_start_seconds": float(self.video_start_seconds) if self.video_start_seconds else None,
            "video_end_seconds": float(self.video_end_seconds) if self.video_end_seconds else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    if ts_str.endswith("+00:00"):
        ts_str = ts_str.replace("+00:00", "")
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def load_game_details() -> dict[int, dict]:
    """Load all game details."""
    details = {}
    for f in GAME_DETAIL_DIR.glob("*.json"):
        game_id = int(f.stem)
        with open(f) as fp:
            details[game_id] = json.load(fp)
    return details


def load_alignments() -> dict[int, dict]:
    """Load tournament alignments indexed by tournament_id."""
    alignments = {}
    if ALIGNMENTS_FILE.exists():
        with open(ALIGNMENTS_FILE) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    tid = data.get("tournament_id")
                    if tid:
                        alignments[tid] = data
    return alignments


def load_match_to_tournament() -> dict[int, int]:
    """Build match_id -> tournament_id lookup from matches data."""
    matches_dir = BASE_DIR / "cache" / "hivemind" / "matches_by_tournament"
    lookup = {}
    for f in matches_dir.glob("*.json"):
        with open(f) as fp:
            matches = json.load(fp)
        for match in matches:
            match_id = match.get("id")
            tournament = match.get("tournament", {})
            tournament_id = tournament.get("id") if isinstance(tournament, dict) else None
            if match_id and tournament_id:
                lookup[match_id] = tournament_id
    return lookup


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


def detect_triggers(events: list[dict], game_id: int) -> list[tuple[str, datetime, int]]:
    """Detect trigger events in a game.

    Returns list of (trigger_type, timestamp, event_id) tuples.
    """
    triggers = []

    # Parse all event timestamps
    parsed_events = []
    for e in events:
        ts_str = e.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = parse_timestamp(ts_str)
            parsed_events.append((ts, e))
        except (ValueError, TypeError):
            continue

    if not parsed_events:
        return triggers

    # Sort by timestamp
    parsed_events.sort(key=lambda x: x[0])

    # Track for combat burst detection
    recent_kills = []  # (timestamp, event_id)

    # Track for win prob spike detection
    last_win_prob = None

    for ts, event in parsed_events:
        event_type = event.get("event_type", "")
        values = event.get("values", [])
        event_id = event.get("id", 0)
        win_prob = event.get("win_probability")

        # Victory
        if event_type == "victory":
            triggers.append(("victory", ts, event_id))

        # Queen kill
        elif event_type == "playerKill":
            if len(values) >= 5 and values[4] == "Queen":
                triggers.append(("queen_kill", ts, event_id))
            # Track all kills for combat burst
            recent_kills.append((ts, event_id))
            # Remove kills older than window
            cutoff = ts - timedelta(seconds=COMBAT_BURST_WINDOW_SECONDS)
            recent_kills = [(t, eid) for t, eid in recent_kills if t >= cutoff]
            # Check for combat burst
            if len(recent_kills) >= COMBAT_BURST_MIN_KILLS:
                # Use middle kill as trigger
                mid_idx = len(recent_kills) // 2
                triggers.append(("combat_burst", recent_kills[mid_idx][0], recent_kills[mid_idx][1]))
                recent_kills = []  # Reset to avoid duplicate triggers

        # Snail events
        elif event_type == "snailEat":
            triggers.append(("snail_eat", ts, event_id))
        elif event_type == "snailEscape":
            triggers.append(("snail_escape", ts, event_id))

        # Win probability spike
        if win_prob is not None:
            try:
                wp = float(win_prob)
                if last_win_prob is not None:
                    delta = abs(wp - last_win_prob)
                    if delta >= WIN_PROB_SPIKE_THRESHOLD:
                        triggers.append(("win_prob_spike", ts, event_id))
                last_win_prob = wp
            except (ValueError, TypeError):
                pass

    return triggers


def create_candidate_window(
    trigger_type: str,
    trigger_ts: datetime,
    event_id: int,
    game_id: int,
    tournament_id: Optional[int],
) -> Candidate:
    """Create a candidate window around a trigger event."""
    config = TRIGGER_CONFIG[trigger_type]
    start_utc = trigger_ts - timedelta(seconds=config["pre"])
    end_utc = trigger_ts + timedelta(seconds=config["post"])

    candidate_id = f"game_{game_id}_{trigger_type}_{event_id}"

    return Candidate(
        candidate_id=candidate_id,
        game_id=game_id,
        tournament_id=tournament_id,
        start_utc=start_utc,
        end_utc=end_utc,
        trigger_type=trigger_type,
        trigger_event_id=event_id,
    )


def merge_overlapping_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Merge overlapping candidates from the same game."""
    if not candidates:
        return []

    # Group by game
    by_game = defaultdict(list)
    for c in candidates:
        by_game[c.game_id].append(c)

    merged = []
    for game_id, game_candidates in by_game.items():
        # Sort by start time
        game_candidates.sort(key=lambda c: c.start_utc)

        current = game_candidates[0]
        for next_c in game_candidates[1:]:
            # Check for overlap
            if next_c.start_utc <= current.end_utc:
                # Merge: extend end time, keep higher-priority trigger type
                current.end_utc = max(current.end_utc, next_c.end_utc)
                # Prefer certain trigger types
                priority = ["victory", "queen_kill", "snail_eat", "snail_escape", "combat_burst", "win_prob_spike"]
                if priority.index(next_c.trigger_type) < priority.index(current.trigger_type):
                    current.trigger_type = next_c.trigger_type
                    current.trigger_event_id = next_c.trigger_event_id
                    current.candidate_id = next_c.candidate_id
            else:
                merged.append(current)
                current = next_c
        merged.append(current)

    return merged


def score_candidate(
    candidate: Candidate,
    game_details: dict[int, dict],
    alignments: dict[int, dict],
) -> Optional[float]:
    """Score a candidate using the ML model.

    Returns predicted score or None if scoring fails.
    """
    # We need to create a synthetic clip-like structure for feature extraction
    # The clip_features module expects clips with source_video_id for alignment lookup
    # We'll work around this by directly extracting features from events

    # Load events for this game
    events = load_game_events(candidate.game_id)
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

        if candidate.start_utc <= ts <= candidate.end_utc:
            event["utc_datetime"] = ts
            event["game_id"] = candidate.game_id
            clip_events.append(event)

    if not clip_events:
        return None

    # Extract features using the existing functions
    wp_features = clip_features.extract_win_prob_features(clip_events)
    event_counts = clip_features.count_events(clip_events)
    state_features = clip_features.extract_game_state_features(clip_events)

    duration = candidate.duration_seconds
    events_per_second = len(clip_events) / duration if duration > 0 else 0
    kills_per_second = event_counts["total_kills"] / duration if duration > 0 else 0

    # Build ClipFeatures object
    features = clip_features.ClipFeatures(
        clip_id=candidate.candidate_id,
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

    return train_rater.predict_rating(features)


def add_video_alignment(
    candidate: Candidate,
    game_details: dict[int, dict],
    alignments: dict[int, dict],
    match_to_tournament: dict[int, int],
) -> None:
    """Add video alignment info to candidate if available."""
    game = game_details.get(candidate.game_id)
    if not game:
        return

    # Get tournament_id via tournament_match -> match_to_tournament lookup
    tournament_match = game.get("tournament_match")
    if not tournament_match:
        return

    tournament_id = match_to_tournament.get(tournament_match)
    if not tournament_id:
        return

    candidate.tournament_id = tournament_id
    alignment = alignments.get(tournament_id)
    if not alignment:
        return

    # Get alignment reference point
    gamestart_utc_str = alignment.get("gamestart_utc")
    video_offset = alignment.get("video_timestamp_seconds")
    video_id = alignment.get("video_id")

    if not all([gamestart_utc_str, video_offset is not None, video_id]):
        return

    try:
        gamestart_utc = parse_timestamp(gamestart_utc_str)
    except (ValueError, TypeError):
        return

    # Convert candidate UTC times to video timestamps
    start_offset = (candidate.start_utc - gamestart_utc).total_seconds()
    end_offset = (candidate.end_utc - gamestart_utc).total_seconds()

    video_start = video_offset + start_offset
    video_end = video_offset + end_offset

    # Only add if video timestamps are valid
    if video_start >= 0 and video_end > video_start:
        candidate.has_video = True
        candidate.video_id = video_id
        candidate.video_start_seconds = round(video_start, 2)
        candidate.video_end_seconds = round(video_end, 2)


def select_diverse_candidates(
    candidates: list[Candidate],
    target: int = TARGET_CANDIDATES,
) -> list[Candidate]:
    """Select diverse candidates respecting limits. Only selects candidates with video."""
    # Filter to only candidates with video
    candidates = [c for c in candidates if c.has_video]

    # Sort by predicted score descending
    candidates.sort(key=lambda c: -c.predicted_score)

    selected = []
    counts_by_type = defaultdict(int)
    counts_by_tournament = defaultdict(int)
    counts_by_game = defaultdict(int)

    for candidate in candidates:
        # Check limits
        if counts_by_type[candidate.trigger_type] >= MAX_PER_TRIGGER_TYPE:
            continue
        if candidate.tournament_id and counts_by_tournament[candidate.tournament_id] >= MAX_PER_TOURNAMENT:
            continue
        if counts_by_game[candidate.game_id] >= MAX_PER_GAME:
            continue

        # Select this candidate
        selected.append(candidate)
        counts_by_type[candidate.trigger_type] += 1
        if candidate.tournament_id:
            counts_by_tournament[candidate.tournament_id] += 1
        counts_by_game[candidate.game_id] += 1

        if len(selected) >= target:
            break

    return selected


def generate_candidates(aligned_only: bool = True, tournament_ids: Optional[list[int]] = None):
    """Main function to generate clip candidates.

    Args:
        aligned_only: If True (default), only scan games from aligned tournaments
        tournament_ids: If provided, only scan games from these specific tournaments
    """
    print("Loading game details...")
    game_details = load_game_details()
    print(f"Loaded {len(game_details)} games")

    print("Loading alignments...")
    alignments = load_alignments()
    print(f"Loaded {len(alignments)} tournament alignments")

    print("Loading match -> tournament lookup...")
    match_to_tournament = load_match_to_tournament()
    print(f"Loaded {len(match_to_tournament)} match -> tournament mappings")

    # Build reverse lookup: tournament_id -> set of game_ids
    tournament_to_games = defaultdict(set)
    for game_id, game in game_details.items():
        match_id = game.get("tournament_match")
        if match_id:
            tid = match_to_tournament.get(match_id)
            if tid:
                tournament_to_games[tid].add(game_id)

    # Determine which game IDs to scan
    all_game_ids = set(int(f.stem) for f in GAME_EVENTS_DIR.glob("*.jsonl"))

    if tournament_ids:
        # Only specific tournaments
        game_ids = []
        for tid in tournament_ids:
            game_ids.extend(tournament_to_games.get(tid, []))
        game_ids = [g for g in game_ids if g in all_game_ids]
        print(f"Filtering to {len(game_ids)} games from tournaments {tournament_ids}")
    elif aligned_only:
        # Only aligned tournaments
        game_ids = []
        for tid in alignments.keys():
            game_ids.extend(tournament_to_games.get(tid, []))
        game_ids = [g for g in game_ids if g in all_game_ids]
        print(f"Filtering to {len(game_ids)} games from {len(alignments)} aligned tournaments")
    else:
        game_ids = list(all_game_ids)
        print(f"Found {len(game_ids)} games with events")

    # Phase 1: Detect triggers
    print("\nPhase 1: Detecting trigger events...")
    all_candidates = []
    trigger_counts = defaultdict(int)

    for i, game_id in enumerate(game_ids):
        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{len(game_ids)} games...")

        events = load_game_events(game_id)
        if not events:
            continue

        # Get tournament ID from game details
        game = game_details.get(game_id, {})
        tournament_id = game.get("tournament")

        triggers = detect_triggers(events, game_id)
        for trigger_type, ts, event_id in triggers:
            candidate = create_candidate_window(
                trigger_type, ts, event_id, game_id, tournament_id
            )
            all_candidates.append(candidate)
            trigger_counts[trigger_type] += 1

    print(f"\nDetected {len(all_candidates)} trigger events:")
    for tt, count in sorted(trigger_counts.items()):
        print(f"  {tt}: {count}")

    # Merge overlapping
    print("\nMerging overlapping candidates...")
    all_candidates = merge_overlapping_candidates(all_candidates)
    print(f"After merge: {len(all_candidates)} candidates")

    # Phase 2: Score candidates
    print("\nPhase 2: Scoring candidates with ML model...")
    scored_candidates = []
    score_failures = 0

    for i, candidate in enumerate(all_candidates):
        if (i + 1) % 1000 == 0:
            print(f"  Scored {i + 1}/{len(all_candidates)} candidates...")

        score = score_candidate(candidate, game_details, alignments)
        if score is not None and score >= 3.0:
            candidate.predicted_score = score
            scored_candidates.append(candidate)
        elif score is None:
            score_failures += 1

    print(f"\nCandidates with score >= 3.0: {len(scored_candidates)}")
    print(f"Score failures: {score_failures}")

    # Phase 3: Add video alignment
    print("\nPhase 3: Adding video alignment...")
    for candidate in scored_candidates:
        add_video_alignment(candidate, game_details, alignments, match_to_tournament)

    with_video = sum(1 for c in scored_candidates if c.has_video)
    print(f"Candidates with video alignment: {with_video}")

    # Phase 4: Select diverse candidates
    print("\nPhase 4: Selecting diverse candidates...")
    final_candidates = select_diverse_candidates(scored_candidates, TARGET_CANDIDATES)

    # Stats
    print(f"\nFinal selection: {len(final_candidates)} candidates")
    type_counts = defaultdict(int)
    for c in final_candidates:
        type_counts[c.trigger_type] += 1
    print("By trigger type:")
    for tt, count in sorted(type_counts.items()):
        print(f"  {tt}: {count}")

    video_count = sum(1 for c in final_candidates if c.has_video)
    print(f"With video: {video_count}")

    scores = [c.predicted_score for c in final_candidates]
    if scores:
        print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")

    # Save
    print(f"\nSaving to {CANDIDATES_FILE}...")
    with open(CANDIDATES_FILE, "w") as f:
        for candidate in final_candidates:
            f.write(json.dumps(candidate.to_dict()) + "\n")

    print("Done!")
    return final_candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate clip candidates")
    parser.add_argument("--all-games", action="store_true",
                       help="Scan all games (default: only aligned tournaments)")
    parser.add_argument("--tournament", type=int, action="append",
                       help="Only scan specific tournament ID(s)")
    args = parser.parse_args()

    generate_candidates(
        aligned_only=not args.all_games,
        tournament_ids=args.tournament
    )
