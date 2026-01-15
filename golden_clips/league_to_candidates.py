#!/usr/bin/env python3
"""
Generate golden clip candidates from league night chapters.

Ports the auto-highlighter algorithm from player.html to Python.
"""

import json
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class HighlightEvent:
    """An event that could be a highlight."""
    time: float  # Video timestamp in seconds
    delta: float  # Win probability change
    event_type: str
    game_id: int
    set_number: int
    event_id: int
    values: list
    position: Optional[int]
    ml_score: Optional[float]
    score: float = 0.0  # Computed highlight score
    window_size: Optional[int] = None


@dataclass
class Candidate:
    """A highlight candidate for the rating system."""
    candidate_id: str
    game_id: int
    set_number: int
    start_seconds: float  # Video timestamp
    end_seconds: float
    duration_seconds: float
    trigger_type: str
    score: float
    event_id: int
    video_id: str


# Algorithm parameters (matching player.html)
MIN_HIGHLIGHT_GAP = 10  # Minimum seconds between highlights
TARGET_PER_SET = 4  # Max highlights per set
CLUSTER_WINDOW = 5  # Seconds for clustering bonus
ML_THRESHOLD = 0.05  # Score threshold for ML-scored events
DELTA_THRESHOLD = 0.15  # Score threshold for delta-based events

# Window parameters (matching generate_chapters.py)
HIGHLIGHT_SEEK_BUFFER = 4.5  # Seconds before event to start clip
HIGHLIGHT_PLAY_DURATION = 6.0  # Seconds after event to end clip

CANDIDATE_RATINGS_FILE = Path(__file__).parent / "candidate_ratings.jsonl"


def load_rated_ranges() -> list[tuple[int, float, float]]:
    """Load (game_id, start, end) from existing ratings."""
    ranges = []
    if not CANDIDATE_RATINGS_FILE.exists():
        return ranges

    with open(CANDIDATE_RATINGS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r.get("candidate_id", "")
            # Extract game_id from candidate_id (e.g., "game_1718162_..." or "league_1718162_...")
            if cid.startswith("game_") or cid.startswith("league_"):
                parts = cid.split("_")
                try:
                    game_id = int(parts[1])
                except (IndexError, ValueError):
                    continue
                # Use adjusted times if available, fall back to original
                start = r.get("adjusted_video_start") or r.get("original_video_start") or r.get("video_start_seconds", 0)
                end = r.get("adjusted_video_end") or r.get("original_video_end") or r.get("video_end_seconds", 0)
                if start and end:
                    ranges.append((game_id, float(start), float(end)))
    return ranges


def overlaps_rated(candidate: dict, rated_ranges: list[tuple[int, float, float]]) -> bool:
    """Check if candidate overlaps with any rated clip."""
    game_id = candidate["game_id"]
    start = candidate["video_start_seconds"]
    end = candidate["video_end_seconds"]
    for r_game, r_start, r_end in rated_ranges:
        if game_id == r_game and start < r_end and end > r_start:
            return True
    return False


def collect_events(chapters: list) -> list[HighlightEvent]:
    """
    Collect all high-impact events from chapters.

    Port of the JavaScript collection logic from updatePlayerHighlights().
    """
    events = []

    for ch in chapters:
        if not ch.get('player_events'):
            continue

        for evt in ch['player_events']:
            # Include if has ML score OR high delta (|delta| >= 0.10)
            ml_score = evt.get('ml_score')
            delta = evt.get('delta', 0)

            if ml_score is not None or abs(delta) >= 0.10:
                events.append(HighlightEvent(
                    time=evt['time'],
                    delta=delta,
                    event_type=evt.get('type', 'unknown'),
                    game_id=ch['game_id'],
                    set_number=ch.get('set_number', 0),
                    event_id=evt.get('id', 0),
                    values=evt.get('values', []),
                    position=evt['positions'][0] if evt.get('positions') else None,
                    ml_score=ml_score,
                    window_size=evt.get('window_size'),
                ))

    return events


def compute_scores(events: list[HighlightEvent]) -> None:
    """
    Compute highlight scores for all events.

    Uses ML score if available, otherwise delta + clustering bonus.
    """
    # Sort by time for clustering calculation
    events.sort(key=lambda e: e.time)

    for i, evt in enumerate(events):
        if evt.ml_score is not None:
            # ML score 1-4 -> normalized 0-0.75
            evt.score = (evt.ml_score - 1) / 4
        else:
            # Delta with clustering bonus
            cluster_score = abs(evt.delta)
            for j, other in enumerate(events):
                if i != j and abs(evt.time - other.time) < CLUSTER_WINDOW:
                    cluster_score += abs(other.delta) * 0.3
            evt.score = cluster_score


def select_highlights(events: list[HighlightEvent]) -> list[HighlightEvent]:
    """
    Select top highlights per set with deduplication.

    Port of the JavaScript selection logic.
    """
    # Group by set
    by_set = defaultdict(list)
    for evt in events:
        # Apply threshold
        threshold = ML_THRESHOLD if evt.ml_score is not None else DELTA_THRESHOLD
        if evt.score >= threshold:
            by_set[evt.set_number].append(evt)

    # Select top N per set with minimum gap
    selected = []
    for set_num, set_events in by_set.items():
        # Sort by score descending
        set_events.sort(key=lambda e: e.score, reverse=True)

        set_selected = []
        for evt in set_events:
            # Check if too close to already-selected event
            too_close = any(
                abs(s.time - evt.time) < MIN_HIGHLIGHT_GAP
                for s in set_selected
            )
            if not too_close:
                set_selected.append(evt)
                if len(set_selected) >= TARGET_PER_SET:
                    break

        selected.extend(set_selected)

    # Sort final list by time
    selected.sort(key=lambda e: e.time)
    return selected


def event_to_candidate(evt: HighlightEvent, video_id: str) -> Candidate:
    """Convert a highlight event to a candidate."""
    # Use window_size if available, otherwise default
    if evt.window_size:
        pre_buffer = evt.window_size / 2
        post_buffer = evt.window_size / 2
    else:
        pre_buffer = HIGHLIGHT_SEEK_BUFFER
        post_buffer = HIGHLIGHT_PLAY_DURATION

    start_seconds = evt.time - pre_buffer
    end_seconds = evt.time + post_buffer

    return Candidate(
        candidate_id=f"league_{evt.game_id}_{evt.event_type}_{evt.event_id}",
        game_id=evt.game_id,
        set_number=evt.set_number,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
        trigger_type=evt.event_type,
        score=evt.ml_score if evt.ml_score else evt.score,
        event_id=evt.event_id,
        video_id=video_id,
    )


def generate_candidates(chapters_path: Path) -> list[dict]:
    """Generate candidates from a league night chapters file."""
    with open(chapters_path) as f:
        data = json.load(f)

    video_id = data.get('video_id')
    if not video_id:
        raise ValueError(f"No video_id in {chapters_path}")

    # Get video start UTC for timestamp conversion
    video_start_utc_str = data.get('video_start_utc')
    if video_start_utc_str:
        video_start_utc = datetime.fromisoformat(video_start_utc_str.replace('Z', '+00:00'))
    else:
        video_start_utc = None

    chapters = data.get('chapters', [])

    # Run the highlight algorithm
    events = collect_events(chapters)
    print(f"Collected {len(events)} high-impact events")

    compute_scores(events)

    highlights = select_highlights(events)
    print(f"Selected {len(highlights)} highlights")

    # Convert to candidates
    candidates = []
    for evt in highlights:
        candidate = event_to_candidate(evt, video_id)

        # Calculate UTC timestamps from video timestamps
        if video_start_utc:
            start_utc = video_start_utc + timedelta(seconds=candidate.start_seconds)
            end_utc = video_start_utc + timedelta(seconds=candidate.end_seconds)
            start_utc_str = start_utc.isoformat()
            end_utc_str = end_utc.isoformat()
        else:
            start_utc_str = None
            end_utc_str = None

        candidates.append({
            'candidate_id': candidate.candidate_id,
            'game_id': candidate.game_id,
            'set_number': candidate.set_number,
            'trigger_type': candidate.trigger_type,
            'predicted_score': round(candidate.score, 2),
            'event_id': candidate.event_id,
            'has_video': True,
            'video_id': candidate.video_id,
            'video_start_seconds': round(candidate.start_seconds, 2),
            'video_end_seconds': round(candidate.end_seconds, 2),
            'start_utc': start_utc_str,
            'end_utc': end_utc_str,
            'duration_seconds': round(candidate.duration_seconds, 2),
            'generated_at': datetime.utcnow().isoformat(),
        })

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description='Generate golden clip candidates from league night chapters'
    )
    parser.add_argument(
        'chapters_file',
        type=Path,
        help='Path to chapters JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output JSONL file (default: print to stdout)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to output file instead of overwriting'
    )
    parser.add_argument(
        '--exclude-rated',
        action='store_true',
        help='Exclude candidates that overlap with already-rated clips'
    )

    args = parser.parse_args()

    candidates = generate_candidates(args.chapters_file)

    # Filter out already-rated candidates if requested
    if args.exclude_rated:
        rated_ranges = load_rated_ranges()
        original_count = len(candidates)
        candidates = [c for c in candidates if not overlaps_rated(c, rated_ranges)]
        excluded = original_count - len(candidates)
        print(f"Excluded {excluded} candidates overlapping with {len(rated_ranges)} rated clips")

    # Print summary
    by_type = defaultdict(int)
    for c in candidates:
        by_type[c['trigger_type']] += 1

    print(f"\nCandidates by trigger type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Output
    if args.output:
        mode = 'a' if args.append else 'w'
        with open(args.output, mode) as f:
            for c in candidates:
                f.write(json.dumps(c) + '\n')
        print(f"\nWrote {len(candidates)} candidates to {args.output}")
    else:
        print(f"\n--- Candidates ---")
        for c in candidates:
            print(json.dumps(c))


if __name__ == '__main__':
    main()
