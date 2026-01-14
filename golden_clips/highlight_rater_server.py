#!/usr/bin/env python3
"""
Browser-based tool for rating highlight clips with corresponding game events.

Shows golden clips alongside the game events that occurred during that clip,
allowing users to rate highlight quality.

Usage:
    python golden_clips/highlight_rater_server.py
    # Open http://localhost:5002 in browser
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_file

# Import ML prediction modules
try:
    from .scorer import get_scorer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
CLIP_INFO_FILE = Path(__file__).parent / "clip_info.jsonl"
ALIGNMENTS_FILE = Path(__file__).parent / "tournament_alignments.jsonl"
RATINGS_FILE = Path(__file__).parent / "highlight_ratings.jsonl"
CANDIDATES_FILE = Path(__file__).parent / "clip_candidates.jsonl"
CANDIDATE_RATINGS_FILE = Path(__file__).parent / "candidate_ratings.jsonl"
GAME_EVENTS_DIR = BASE_DIR / "cache" / "game_events"
GAME_DETAIL_DIR = BASE_DIR / "cache" / "game_detail"
MATCHES_DIR = BASE_DIR / "cache" / "hivemind" / "matches_by_tournament"
UI_FILE = Path(__file__).parent / "highlight_rater_ui.html"

# Global data
clips = []  # List of clips from clip_info.jsonl
candidates = []  # List of candidates from clip_candidates.jsonl
alignments = {}  # video_id -> alignment data
ratings = {}  # clip_id -> rating data
candidate_ratings = {}  # candidate_id -> rating data
game_events_cache = {}  # game_id -> list of events
game_details = {}  # game_id -> game detail
matches_by_id = {}  # match_id -> match data


def load_clips():
    """Load clips from clip_info.jsonl."""
    global clips
    if not CLIP_INFO_FILE.exists():
        print(f"Warning: {CLIP_INFO_FILE} not found")
        return

    with open(CLIP_INFO_FILE) as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))
    print(f"Loaded {len(clips)} clips")


def load_alignments():
    """Load tournament alignments."""
    global alignments
    if not ALIGNMENTS_FILE.exists():
        print(f"Warning: {ALIGNMENTS_FILE} not found")
        return

    with open(ALIGNMENTS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                video_id = data.get("video_id")
                if video_id:
                    alignments[video_id] = data
    print(f"Loaded {len(alignments)} alignments")


def load_ratings():
    """Load existing ratings."""
    global ratings
    if not RATINGS_FILE.exists():
        return

    with open(RATINGS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                clip_id = data.get("clip_id")
                if clip_id:
                    ratings[clip_id] = data
    print(f"Loaded {len(ratings)} ratings")


def load_candidates():
    """Load clip candidates."""
    global candidates
    if not CANDIDATES_FILE.exists():
        print(f"Warning: {CANDIDATES_FILE} not found")
        return

    with open(CANDIDATES_FILE) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")


def load_candidate_ratings():
    """Load existing candidate ratings."""
    global candidate_ratings
    if not CANDIDATE_RATINGS_FILE.exists():
        return

    with open(CANDIDATE_RATINGS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cid = data.get("candidate_id")
                if cid:
                    candidate_ratings[cid] = data
    print(f"Loaded {len(candidate_ratings)} candidate ratings")


def save_candidate_rating(candidate_id: str, rating_data: dict):
    """Save a candidate rating."""
    candidate_ratings[candidate_id] = rating_data
    with open(CANDIDATE_RATINGS_FILE, "w") as f:
        for data in candidate_ratings.values():
            f.write(json.dumps(data) + "\n")


def load_game_details():
    """Load all game details."""
    global game_details
    for f in GAME_DETAIL_DIR.glob("*.json"):
        game_id = int(f.stem)
        with open(f) as fp:
            game_details[game_id] = json.load(fp)
    print(f"Loaded {len(game_details)} game details")


def load_matches():
    """Load all matches and index by match_id."""
    global matches_by_id
    for f in MATCHES_DIR.glob("*.json"):
        with open(f) as fp:
            matches = json.load(fp)
        for match in matches:
            matches_by_id[match["id"]] = match
    print(f"Loaded {len(matches_by_id)} matches")


def get_team_names_for_game(game_id: int) -> tuple[str, str]:
    """Get blue and gold team names for a game."""
    game = game_details.get(game_id)
    if not game:
        return "Blue Team", "Gold Team"

    # Try tournament_match first, then match
    match_id = game.get("tournament_match") or game.get("match")
    if not match_id:
        return "Blue Team", "Gold Team"

    match = matches_by_id.get(match_id)
    if not match:
        return "Blue Team", "Gold Team"

    blue_info = match.get("blue_team_info") or {}
    gold_info = match.get("gold_team_info") or {}
    return blue_info.get("name", "Blue Team"), gold_info.get("name", "Gold Team")


def save_rating(clip_id: str, rating_data: dict):
    """Save a rating."""
    ratings[clip_id] = rating_data
    with open(RATINGS_FILE, "w") as f:
        for data in ratings.values():
            f.write(json.dumps(data) + "\n")


def get_predicted_rating(clip: dict) -> float | None:
    """Get ML-predicted rating for a clip."""
    if not ML_AVAILABLE:
        return None

    result = get_scorer().score_clip(clip, alignments, game_details)
    return result.score if result.success else None


def get_game_events(game_id: int) -> list:
    """Load events for a game (cached)."""
    if game_id in game_events_cache:
        return game_events_cache[game_id]

    events_file = GAME_EVENTS_DIR / f"{game_id}.jsonl"
    if not events_file.exists():
        return []

    events = []
    with open(events_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    game_events_cache[game_id] = events
    return events


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    # Handle various formats
    if ts_str.endswith("+00:00"):
        ts_str = ts_str.replace("+00:00", "")
    if "." in ts_str:
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def get_clip_alignment(clip: dict) -> dict | None:
    """Get alignment for a clip's source video."""
    video_id = clip.get("source_video_id")
    return alignments.get(video_id)


def get_events_for_clip(clip: dict) -> tuple[list, dict]:
    """Get all game events that occurred during a clip's time range.

    Returns (events, debug_info) tuple.
    """
    alignment = get_clip_alignment(clip)
    if not alignment:
        return [], {"error": "no_alignment"}

    # Parse alignment data
    gamestart_utc = parse_timestamp(alignment["gamestart_utc"])
    video_offset = alignment["video_timestamp_seconds"]

    # Convert clip timestamps to UTC
    clip_start_offset = clip["start_seconds"] - video_offset
    clip_end_offset = clip["end_seconds"] - video_offset

    clip_utc_start = gamestart_utc + timedelta(seconds=clip_start_offset)
    clip_utc_end = gamestart_utc + timedelta(seconds=clip_end_offset)

    # Debug info
    debug_info = {
        "clip_utc_start": clip_utc_start.isoformat(),
        "clip_utc_end": clip_utc_end.isoformat(),
        "games_checked": 0,
        "games_in_window": 0,
        "nearest_game_before": None,
        "nearest_game_after": None,
    }

    # Find games that might overlap with this time range
    matching_events = []
    nearest_before = None
    nearest_after = None

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
        if abs(time_diff) > 7200:  # 2 hours
            continue

        debug_info["games_checked"] += 1

        # Track nearest games for debug
        if time_diff < 0:  # Game is before clip
            if nearest_before is None or time_diff > nearest_before[1]:
                nearest_before = (game_id, time_diff, game_start)
        else:  # Game is after clip
            if nearest_after is None or time_diff < nearest_after[1]:
                nearest_after = (game_id, time_diff, game_start)

        # Load events for this game
        events = get_game_events(game_id)
        for event in events:
            event_ts = event.get("timestamp", "")
            if not event_ts:
                continue

            try:
                event_dt = parse_timestamp(event_ts)
            except (ValueError, TypeError):
                continue

            # Check if event falls within clip time range
            if clip_utc_start <= event_dt <= clip_utc_end:
                debug_info["games_in_window"] += 1
                # Add relative time within clip
                relative_seconds = (event_dt - clip_utc_start).total_seconds()
                event_with_time = {
                    **event,
                    "relative_seconds": relative_seconds,
                    "game_id": game_id,
                }
                matching_events.append(event_with_time)

    # Add nearest game info to debug
    if nearest_before:
        debug_info["nearest_game_before"] = {
            "game_id": nearest_before[0],
            "seconds_before": abs(nearest_before[1]),
            "start_time": nearest_before[2],
        }
    if nearest_after:
        debug_info["nearest_game_after"] = {
            "game_id": nearest_after[0],
            "seconds_after": nearest_after[1],
            "start_time": nearest_after[2],
        }

    # Sort by timestamp
    matching_events.sort(key=lambda e: e.get("timestamp", ""))
    return matching_events, debug_info


def format_event_display(event: dict) -> dict:
    """Format an event for display in the UI."""
    event_type = event.get("event_type", "unknown")
    values = event.get("values", [])
    relative_seconds = event.get("relative_seconds", 0)

    # Format time as MM:SS.s
    minutes = int(relative_seconds // 60)
    seconds = relative_seconds % 60
    time_str = f"{minutes}:{seconds:05.2f}"

    # Format event details based on type
    details = ""
    if event_type == "playerKill":
        if len(values) >= 5:
            unit_type = values[4] if len(values) > 4 else "?"
            killer = values[2] if len(values) > 2 else "?"
            killed = values[3] if len(values) > 3 else "?"
            details = f"{unit_type} kill: P{killer} -> P{killed}"
    elif event_type == "snailEat":
        if len(values) >= 4:
            eaten = values[2] if len(values) > 2 else "?"
            owner = values[3] if len(values) > 3 else "?"
            details = f"Snail (P{owner}) ate P{eaten}"
    elif event_type == "snailEscape":
        details = "Snail escaped!"
    elif event_type == "victory":
        if len(values) >= 2:
            team = values[0] if values else "?"
            victory_type = values[1] if len(values) > 1 else "?"
            details = f"{team} wins by {victory_type}"
    elif event_type == "berryDeposit":
        if len(values) >= 3:
            player = values[2] if len(values) > 2 else "?"
            details = f"Berry deposited by P{player}"
    elif event_type == "berryKickIn":
        if len(values) >= 3:
            player = values[2] if len(values) > 2 else "?"
            details = f"Berry kicked in by P{player}"
    elif event_type == "gamestart":
        if values:
            map_name = values[0] if values else "?"
            details = f"Game start: {map_name}"
    elif event_type == "gameend":
        details = "Game end"
    elif event_type == "useMaiden":
        if len(values) >= 4:
            ability = values[2] if len(values) > 2 else "?"
            player = values[3] if len(values) > 3 else "?"
            details = f"P{player} used {ability}"
    elif event_type == "getOnSnail":
        if len(values) >= 3:
            player = values[2] if len(values) > 2 else "?"
            details = f"P{player} got on snail"
    elif event_type == "getOffSnail":
        if len(values) >= 4:
            player = values[3] if len(values) > 3 else "?"
            details = f"P{player} got off snail"
    else:
        details = str(values)[:50] if values else ""

    return {
        "event_type": event_type,
        "time_str": time_str,
        "relative_seconds": relative_seconds,
        "details": details,
        "win_probability": event.get("win_probability"),
        "game_id": event.get("game_id"),
    }


# Routes

@app.route("/")
def index():
    return send_file(UI_FILE)


@app.route("/api/clips")
def get_clips():
    """Get all clips with alignment and rating status."""
    result = []
    for clip in clips:
        clip_id = clip.get("clip_id", "")
        alignment = get_clip_alignment(clip)
        rating = ratings.get(clip_id)

        # Skip unaligned clips
        if not alignment:
            continue

        # Get ML prediction
        predicted = get_predicted_rating(clip)

        result.append({
            "clip_id": clip_id,
            "title": clip.get("title", ""),
            "source_video_id": clip.get("source_video_id", ""),
            "start_seconds": clip.get("start_seconds", 0),
            "end_seconds": clip.get("end_seconds", 0),
            "duration": clip.get("duration", 0),
            "has_alignment": alignment is not None,
            "has_rating": rating is not None,
            "rating": rating.get("rating") if rating else None,
            "predicted_rating": round(predicted, 2) if predicted else None,
        })
    return jsonify(result)


@app.route("/api/clip/<clip_id>")
def get_clip_detail(clip_id: str):
    """Get clip details with aligned events."""
    # Find clip
    clip = next((c for c in clips if c.get("clip_id") == clip_id), None)
    if not clip:
        return jsonify({"error": "Clip not found"}), 404

    alignment = get_clip_alignment(clip)
    if not alignment:
        return jsonify({"error": "No alignment for this clip"}), 404

    # Get events
    events, debug_info = get_events_for_clip(clip)
    formatted_events = [format_event_display(e) for e in events]

    # Get team names from the first game found in events
    blue_team, gold_team = "Blue Team", "Gold Team"
    game_ids = set(e.get("game_id") for e in events if e.get("game_id"))
    if game_ids:
        # Use the first game_id to get team names
        blue_team, gold_team = get_team_names_for_game(list(game_ids)[0])

    # Get ML prediction
    predicted = get_predicted_rating(clip)

    return jsonify({
        "clip_id": clip_id,
        "clip_url": clip.get("clip_url", ""),
        "title": clip.get("title", ""),
        "source_video_id": clip.get("source_video_id", ""),
        "source_video_url": clip.get("source_video_url", ""),
        "start_seconds": clip.get("start_seconds", 0),
        "end_seconds": clip.get("end_seconds", 0),
        "duration": clip.get("duration", 0),
        "alignment": {
            "tournament_id": alignment.get("tournament_id"),
            "gamestart_utc": alignment.get("gamestart_utc"),
            "video_timestamp_seconds": alignment.get("video_timestamp_seconds"),
        },
        "events": formatted_events,
        "event_count": len(formatted_events),
        "rating": ratings.get(clip_id, {}).get("rating"),
        "predicted_rating": round(predicted, 2) if predicted else None,
        "blue_team": blue_team,
        "gold_team": gold_team,
        "debug": debug_info,
    })


@app.route("/api/clip/<clip_id>/rate", methods=["POST"])
def rate_clip(clip_id: str):
    """Save rating for a clip."""
    data = request.json
    rating = data.get("rating")

    rating_data = {
        "clip_id": clip_id,
        "rating": rating,
        "rated_at": datetime.now().isoformat(),
    }

    save_rating(clip_id, rating_data)
    return jsonify({"success": True, "rating": rating_data})


# Candidate routes

@app.route("/api/candidates")
def get_candidates():
    """Get all candidates with rating status."""
    result = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        rating = candidate_ratings.get(cid)

        result.append({
            "candidate_id": cid,
            "game_id": c.get("game_id"),
            "tournament_id": c.get("tournament_id"),
            "trigger_type": c.get("trigger_type"),
            "duration": c.get("duration_seconds", 0),
            "predicted_score": c.get("predicted_score"),
            "has_video": c.get("has_video", False),
            "video_id": c.get("video_id"),
            "video_start_seconds": c.get("video_start_seconds"),
            "video_end_seconds": c.get("video_end_seconds"),
            "has_rating": rating is not None,
            "rating": rating.get("rating") if rating else None,
        })
    return jsonify(result)


@app.route("/api/candidate/<candidate_id>")
def get_candidate_detail(candidate_id: str):
    """Get candidate details with events."""
    # Find candidate
    candidate = next((c for c in candidates if c.get("candidate_id") == candidate_id), None)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404

    game_id = candidate.get("game_id")
    start_utc_str = candidate.get("start_utc")
    end_utc_str = candidate.get("end_utc")

    # Parse timestamps
    start_utc = parse_timestamp(start_utc_str)
    end_utc = parse_timestamp(end_utc_str)

    # Load events for this game
    events = get_game_events(game_id)
    clip_events = []

    for event in events:
        event_ts = event.get("timestamp", "")
        if not event_ts:
            continue
        try:
            event_dt = parse_timestamp(event_ts)
        except (ValueError, TypeError):
            continue

        if start_utc <= event_dt <= end_utc:
            relative_seconds = (event_dt - start_utc).total_seconds()
            event_with_time = {
                **event,
                "relative_seconds": relative_seconds,
                "game_id": game_id,
            }
            clip_events.append(event_with_time)

    clip_events.sort(key=lambda e: e.get("timestamp", ""))
    formatted_events = [format_event_display(e) for e in clip_events]

    # Get team names
    blue_team, gold_team = get_team_names_for_game(game_id)

    return jsonify({
        "candidate_id": candidate_id,
        "game_id": game_id,
        "tournament_id": candidate.get("tournament_id"),
        "trigger_type": candidate.get("trigger_type"),
        "start_utc": start_utc_str,
        "end_utc": end_utc_str,
        "duration": candidate.get("duration_seconds", 0),
        "predicted_score": candidate.get("predicted_score"),
        "has_video": candidate.get("has_video", False),
        "video_id": candidate.get("video_id"),
        "video_start_seconds": candidate.get("video_start_seconds"),
        "video_end_seconds": candidate.get("video_end_seconds"),
        "events": formatted_events,
        "event_count": len(formatted_events),
        "rating": candidate_ratings.get(candidate_id, {}).get("rating"),
        "blue_team": blue_team,
        "gold_team": gold_team,
    })


@app.route("/api/candidate/<candidate_id>/rate", methods=["POST"])
def rate_candidate(candidate_id: str):
    """Save rating for a candidate, optionally with time adjustments."""
    data = request.json
    rating = data.get("rating")

    # Look up candidate to get metadata (handle _adjusted suffix)
    base_id = candidate_id.replace('_adjusted', '')
    candidate = next((c for c in candidates if c.get("candidate_id") == base_id), None)

    rating_data = {
        "candidate_id": candidate_id,
        "rating": rating,
        "rated_at": datetime.now().isoformat(),
    }

    # Add metadata from candidate so training works after candidates regenerate
    if candidate:
        rating_data["game_id"] = candidate.get("game_id")
        rating_data["start_utc"] = candidate.get("start_utc")
        rating_data["end_utc"] = candidate.get("end_utc")

    # Include time adjustments if provided
    if "adjusted_video_start" in data:
        rating_data["adjusted_video_start"] = data["adjusted_video_start"]
        rating_data["adjusted_video_end"] = data["adjusted_video_end"]
        rating_data["original_video_start"] = data["original_video_start"]
        rating_data["original_video_end"] = data["original_video_end"]

    save_candidate_rating(candidate_id, rating_data)
    return jsonify({"success": True, "rating": rating_data})


if __name__ == "__main__":
    load_clips()
    load_alignments()
    load_ratings()
    load_candidates()
    load_candidate_ratings()
    load_game_details()
    load_matches()

    # Count aligned clips
    aligned_count = sum(1 for c in clips if get_clip_alignment(c))
    print(f"\nClips with alignments: {aligned_count}/{len(clips)}")
    print(f"Candidates: {len(candidates)} ({sum(1 for c in candidates if c.get('has_video'))} with video)")

    print(f"\nStarting highlight rater server...")
    print(f"Open http://localhost:5002 in your browser\n")
    app.run(debug=True, port=5002)
