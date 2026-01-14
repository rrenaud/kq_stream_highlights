#!/usr/bin/env python3
"""
Browser-based tool for aligning YouTube tournament videos with Hivemind game timestamps.

The user finds the video timestamp of the first game's "gamestart" event.
This establishes a mapping between video time and absolute UTC time.

Usage:
    python golden_clips/tournament_align_server.py
    # Open http://localhost:5000 in browser
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_file

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_chapters import (
    CABINET_TO_YOUTUBE,
    CABINET_TO_URL,
    fetch_most_recent_video,
)
from fetch_games import fetch_games_for_day
from hivemind_api import fetch_game_events

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
TOURNAMENT_VIDEOS_FILE = BASE_DIR / "cache" / "tournament_videos" / "all_videos.jsonl"
CLIP_INFO_FILE = Path(__file__).parent / "clip_info.jsonl"
MATCHES_DIR = BASE_DIR / "cache" / "hivemind" / "matches_by_tournament"
GAME_DETAIL_DIR = BASE_DIR / "cache" / "game_detail"
GAME_EVENTS_DIR = BASE_DIR / "cache" / "game_events"
ALIGNMENTS_FILE = Path(__file__).parent / "tournament_alignments.jsonl"
LEAGUE_ALIGNMENTS_FILE = Path(__file__).parent / "league_alignments.jsonl"
PENDING_LEAGUE_NIGHTS_FILE = Path(__file__).parent / "pending_league_nights.jsonl"
UI_FILE = Path(__file__).parent / "tournament_align_ui.html"

# Global data
tournament_videos = []  # List of videos from clips (deduplicated by source_video_id)
video_to_tournament = {}  # video_id -> tournament_id lookup from all_videos.jsonl
tournaments_by_date = {}  # date string -> list of tournament_ids
matches_by_tournament = {}  # tournament_id -> list of matches
game_details = {}  # game_id -> game detail dict
alignments = {}  # tournament_id -> alignment data
league_alignments = {}  # cabinet_id_date -> alignment data
pending_league_nights = {}  # cabinet_id_date -> pending league night data


def extract_video_id(video_id_or_url: str) -> str:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    if "youtube.com" in video_id_or_url or "youtu.be" in video_id_or_url:
        # Extract from URL
        match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', video_id_or_url)
        if match:
            return match.group(1)
    return video_id_or_url


def get_video_duration(video_id: str) -> float | None:
    """Get video duration in seconds using yt-dlp."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info.get('duration', 0))
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return None


def load_tournament_videos():
    """Load videos from clip_info.jsonl, deduplicated by source_video_id."""
    global tournament_videos, video_to_tournament

    # Build video_id -> tournament_id lookup from all_videos.jsonl
    if TOURNAMENT_VIDEOS_FILE.exists():
        with open(TOURNAMENT_VIDEOS_FILE) as f:
            for line in f:
                if line.strip():
                    v = json.loads(line)
                    vid = extract_video_id(v.get("video_id", ""))
                    if vid and v.get("tournament_id"):
                        video_to_tournament[vid] = v["tournament_id"]
        print(f"Built lookup for {len(video_to_tournament)} videos -> tournaments")

    # Load clips and deduplicate by source_video_id
    if not CLIP_INFO_FILE.exists():
        print(f"Warning: {CLIP_INFO_FILE} not found")
        return

    seen_videos = {}  # source_video_id -> clip info (keep first occurrence)
    with open(CLIP_INFO_FILE) as f:
        for line in f:
            if line.strip():
                clip = json.loads(line)
                vid = clip.get("source_video_id", "")
                if vid and vid not in seen_videos:
                    seen_videos[vid] = clip

    # Convert to list sorted by upload_date (most recent first)
    tournament_videos = list(seen_videos.values())
    tournament_videos.sort(key=lambda c: c.get("upload_date", ""), reverse=True)
    print(f"Loaded {len(tournament_videos)} unique videos from clips")


def load_matches():
    """Load tournament match data and build date lookup."""
    global matches_by_tournament, tournaments_by_date
    for f in MATCHES_DIR.glob("*.json"):
        tournament_id = int(f.stem)
        with open(f) as fp:
            matches = json.load(fp)
        matches_by_tournament[tournament_id] = matches

        # Extract tournament date for date-based matching
        if matches and matches[0].get("tournament"):
            date_str = matches[0]["tournament"].get("date", "")
            if date_str:
                if date_str not in tournaments_by_date:
                    tournaments_by_date[date_str] = []
                tournaments_by_date[date_str].append(tournament_id)

    print(f"Loaded matches for {len(matches_by_tournament)} tournaments")
    print(f"Built date lookup for {len(tournaments_by_date)} dates")


def load_game_details():
    """Load all game details."""
    global game_details
    for f in GAME_DETAIL_DIR.glob("*.json"):
        game_id = int(f.stem)
        with open(f) as fp:
            game_details[game_id] = json.load(fp)
    print(f"Loaded {len(game_details)} game details")


def load_alignments():
    """Load saved alignments."""
    global alignments
    if not ALIGNMENTS_FILE.exists():
        return
    with open(ALIGNMENTS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                alignments[data["tournament_id"]] = data
    print(f"Loaded {len(alignments)} alignments")


def save_alignment(tournament_id: int, data: dict):
    """Save an alignment."""
    alignments[tournament_id] = data
    with open(ALIGNMENTS_FILE, "w") as f:
        for d in alignments.values():
            f.write(json.dumps(d) + "\n")


def load_league_alignments():
    """Load saved league night alignments."""
    global league_alignments
    if not LEAGUE_ALIGNMENTS_FILE.exists():
        return
    with open(LEAGUE_ALIGNMENTS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                key = f"{data['cabinet_id']}_{data['date']}"
                league_alignments[key] = data
    print(f"Loaded {len(league_alignments)} league alignments")


def save_league_alignment(cabinet_id: str, date: str, data: dict):
    """Save a league night alignment."""
    key = f"{cabinet_id}_{date}"
    league_alignments[key] = data
    with open(LEAGUE_ALIGNMENTS_FILE, "w") as f:
        for d in league_alignments.values():
            f.write(json.dumps(d) + "\n")


def load_pending_league_nights():
    """Load pending league nights from file."""
    global pending_league_nights
    if not PENDING_LEAGUE_NIGHTS_FILE.exists():
        return
    with open(PENDING_LEAGUE_NIGHTS_FILE) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                key = f"{data['cabinet_id']}_{data['date']}"
                pending_league_nights[key] = data
    print(f"Loaded {len(pending_league_nights)} pending league nights")


def get_first_game_for_tournament(tournament_id: int) -> dict | None:
    """Find the first game of a tournament by start time."""
    matches = matches_by_tournament.get(tournament_id, [])
    if not matches:
        return None

    # Get all match IDs for this tournament
    match_ids = {m["id"] for m in matches}

    # Find games that belong to these matches
    tournament_games = []
    for game_id, game in game_details.items():
        if game.get("tournament_match") in match_ids:
            tournament_games.append(game)

    if not tournament_games:
        return None

    # Sort by start_time, return first
    tournament_games.sort(key=lambda g: g.get("start_time", ""))
    return tournament_games[0]


def get_first_games_by_cabinet(tournament_id: int) -> dict[int, dict]:
    """Find the first game per cabinet for a tournament.

    Returns dict mapping cabinet_id -> first game on that cabinet.
    """
    matches = matches_by_tournament.get(tournament_id, [])
    if not matches:
        return {}

    match_ids = {m["id"] for m in matches}

    # Group games by cabinet
    games_by_cab = defaultdict(list)
    for game_id, game in game_details.items():
        if game.get("tournament_match") in match_ids:
            cab_id = game.get("cabinet", {}).get("id")
            if cab_id:
                games_by_cab[cab_id].append(game)

    # Get first game per cabinet (by start_time)
    first_games = {}
    for cab_id, games in games_by_cab.items():
        games.sort(key=lambda g: g.get("start_time", ""))
        first_games[cab_id] = games[0]

    return first_games


def get_gamestart_event(game_id: int) -> dict | None:
    """Get the gamestart event for a game."""
    events_file = GAME_EVENTS_DIR / f"{game_id}.jsonl"
    if not events_file.exists():
        return None

    with open(events_file) as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                if event.get("event_type") == "gamestart":
                    return event
    return None


def get_team_names_for_match(match_id: int, tournament_id: int) -> tuple[str, str]:
    """Get blue and gold team names for a match."""
    matches = matches_by_tournament.get(tournament_id, [])
    for m in matches:
        if m["id"] == match_id:
            blue = m.get("blue_team_info") or {}
            gold = m.get("gold_team_info") or {}
            return blue.get("name", "Unknown"), gold.get("name", "Unknown")
    return "Unknown", "Unknown"


def get_games_around_date(target_date: str) -> dict:
    """Find the nearest Wonderville games before and after a given date."""
    # Filter to Wonderville games only (cabinet_id: 29)
    wonderville_games = [
        g for g in game_details.values()
        if g.get("cabinet", {}).get("id") == 29
    ]

    if not wonderville_games:
        return {"last_before": None, "first_after": None}

    # Sort by start_time
    sorted_games = sorted(wonderville_games, key=lambda g: g.get("start_time", ""))

    last_before = None
    first_after = None

    for game in sorted_games:
        game_time = game.get("start_time", "")
        if game_time < target_date:
            last_before = game
        elif game_time >= target_date and first_after is None:
            first_after = game
            break

    def format_game(g):
        if not g:
            return None
        return {
            "game_id": g["id"],
            "start_time": g.get("start_time"),
            "map_name": g.get("map_name", ""),
        }

    return {
        "last_before": format_game(last_before),
        "first_after": format_game(first_after),
    }


def find_tournament_for_video(video: dict) -> int | None:
    """Find tournament_id for a video using two-tier matching.

    Primary: Look up video_id in video_to_tournament mapping.
    Fallback: Match upload_date to tournament date (try same day and day before,
              since videos are often uploaded the day after the tournament).
    """
    vid = video.get("source_video_id", "")

    # Primary: direct video_id lookup
    if vid in video_to_tournament:
        return video_to_tournament[vid]

    # Fallback: date matching
    upload_date = video.get("upload_date", "")
    if len(upload_date) == 8:
        # Parse upload date
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d")
        except ValueError:
            return None

        # Try upload date and day before (videos often uploaded day after tournament)
        dates_to_try = [
            dt.strftime("%Y-%m-%d"),
            (dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        ]

        for date_str in dates_to_try:
            tournament_ids = tournaments_by_date.get(date_str, [])
            if len(tournament_ids) == 1:
                return tournament_ids[0]
            # If multiple tournaments on same date, can't disambiguate

    return None


def get_league_night_detail(league_id: str):
    """Get league night details for alignment UI.

    Args:
        league_id: String like "league_sf_2026-01-13"

    Returns:
        JSON response with league night details
    """
    # Parse the league ID
    parts = league_id.split("_", 2)
    if len(parts) != 3 or parts[0] != "league":
        return jsonify({"error": "Invalid league night ID"}), 400

    cabinet_id = parts[1]
    date = parts[2]
    key = f"{cabinet_id}_{date}"

    # Find the pending league night
    league_night = pending_league_nights.get(key)
    if not league_night:
        return jsonify({"error": "League night not found"}), 404

    # Get the first game's gamestart event and map name
    first_game_id = league_night.get("first_game_id")
    gamestart_utc = None
    map_name = None

    if first_game_id:
        events = fetch_game_events(first_game_id, verbose=False)
        gamestart_event = next(
            (e for e in events if e.event_type == "gamestart"),
            None
        )
        if gamestart_event:
            gamestart_utc = gamestart_event.timestamp.isoformat()
            # Extract map name from gamestart event values (first value is map name)
            if gamestart_event.values:
                raw_map = gamestart_event.values[0]
                # Clean up map name (e.g., "map_day" -> "Day")
                map_name = raw_map.replace("map_", "").title() if raw_map else None

    # Check for existing alignment
    existing_alignment = league_alignments.get(key)

    return jsonify({
        "tournament_id": league_id,
        "tournament_name": f"League Night: {cabinet_id.upper()} - {date}",
        "video_id": league_night.get("video_id", ""),
        "video_title": league_night.get("title", ""),
        "type": "league_night",
        "cabinet_id": cabinet_id,
        "date": date,
        "cabinet_options": [{
            "cabinet_id": cabinet_id,
            "cabinet_name": cabinet_id.upper(),
            "game_id": first_game_id,
            "map_name": map_name,
            "gamestart_utc": gamestart_utc,
            "hivemind_url": f"https://kqhivemind.com/game/{first_game_id}" if first_game_id else None,
        }],
        "alignment": existing_alignment,
        "games_count": league_night.get("games_count", 0),
    })


# Routes

@app.route("/")
def index():
    return send_file(UI_FILE)


@app.route("/api/tournaments")
def get_tournaments():
    """Get all videos with alignment status (tournaments + league nights)."""
    result = []

    # Add tournament videos
    for video in tournament_videos:
        tournament_id = find_tournament_for_video(video)
        alignment = alignments.get(tournament_id) if tournament_id else None
        result.append({
            "tournament_id": tournament_id,
            "video_id": video.get("source_video_id", ""),
            "title": video.get("title", ""),
            "upload_date": video.get("upload_date", ""),
            "aligned": alignment is not None,
            "type": "tournament",
        })

    # Add pending league nights
    for key, league_night in pending_league_nights.items():
        # Check if already aligned
        alignment = league_alignments.get(key)
        # Use league_cabinet_date as the ID format
        league_id = f"league_{league_night['cabinet_id']}_{league_night['date']}"
        result.append({
            "tournament_id": league_id,
            "video_id": league_night.get("video_id", ""),
            "title": league_night.get("title", ""),
            "upload_date": league_night.get("upload_date", ""),
            "aligned": alignment is not None,
            "type": "league_night",
            "cabinet_id": league_night.get("cabinet_id", ""),
            "date": league_night.get("date", ""),
        })

    # Sort by upload_date (most recent first)
    result.sort(key=lambda x: x.get("upload_date", ""), reverse=True)

    return jsonify(result)


@app.route("/api/tournament/<tournament_id>")
def get_tournament_detail(tournament_id):
    """Get tournament/league night details including first game info."""
    # Check if this is a league night ID
    if isinstance(tournament_id, str) and tournament_id.startswith("league_"):
        return get_league_night_detail(tournament_id)

    # Convert to int for tournament lookup
    try:
        tournament_id = int(tournament_id)
    except ValueError:
        return jsonify({"error": "Invalid tournament ID"}), 400

    # Find the video for this tournament
    video = next(
        (v for v in tournament_videos if find_tournament_for_video(v) == tournament_id),
        None
    )
    if not video:
        return jsonify({"error": "Tournament not found"}), 404

    # Get tournament name from matches
    matches = matches_by_tournament.get(tournament_id, [])
    tournament_name = ""
    if matches and matches[0].get("tournament"):
        tournament_name = matches[0]["tournament"].get("name", "")

    # Get first game per cabinet
    first_games_by_cab = get_first_games_by_cabinet(tournament_id)
    cabinet_options = []

    for cab_id, game in first_games_by_cab.items():
        game_id = game["id"]
        gamestart_event = get_gamestart_event(game_id)
        match_id = game.get("tournament_match")
        blue_team, gold_team = get_team_names_for_match(match_id, tournament_id)

        cabinet_options.append({
            "cabinet_id": cab_id,
            "cabinet_name": game.get("cabinet", {}).get("display_name", "Unknown"),
            "game_id": game_id,
            "map_name": game.get("map_name", ""),
            "start_time": game.get("start_time"),
            "blue_team": blue_team,
            "gold_team": gold_team,
            "gamestart_utc": gamestart_event.get("timestamp") if gamestart_event else None,
            "hivemind_url": f"https://kqhivemind.com/game/{game_id}",
        })

    # Sort by cabinet name for consistent ordering
    cabinet_options.sort(key=lambda c: c["cabinet_name"])

    # Get cache coverage info for debugging
    # Convert upload_date "20251023" to "2025-10-23" for date matching
    upload_date = video.get("upload_date", "")
    if len(upload_date) == 8:
        video_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    else:
        video_date = upload_date
    cache_info = get_games_around_date(video_date)

    return jsonify({
        "tournament_id": tournament_id,
        "tournament_name": tournament_name,
        "video_id": video.get("source_video_id", ""),
        "video_title": video.get("title", ""),
        "video_start_time": video_date,
        "cabinet_options": cabinet_options,
        "alignment": alignments.get(tournament_id),
        "cache_info": cache_info,
    })


@app.route("/api/tournament/<tournament_id>/align", methods=["POST"])
def align_tournament(tournament_id):
    """Save alignment for a tournament or league night."""
    # Check if this is a league night
    if isinstance(tournament_id, str) and tournament_id.startswith("league_"):
        return align_league_night(tournament_id)

    # Convert to int for tournament
    try:
        tournament_id = int(tournament_id)
    except ValueError:
        return jsonify({"error": "Invalid tournament ID"}), 400

    data = request.json
    video_timestamp_seconds = float(data["video_timestamp_seconds"])
    game_id = int(data["game_id"])
    cabinet_id = int(data["cabinet_id"])

    # Get gamestart event for selected game
    gamestart_event = get_gamestart_event(game_id)
    if not gamestart_event:
        return jsonify({"error": "No gamestart event found"}), 400

    # Find video info
    video = next(
        (v for v in tournament_videos if find_tournament_for_video(v) == tournament_id),
        None
    )

    video_id = video.get("source_video_id", "") if video else ""

    # Fetch video duration from YouTube
    video_duration = get_video_duration(video_id) if video_id else None

    alignment_data = {
        "tournament_id": tournament_id,
        "video_id": video_id,
        "cabinet_id": cabinet_id,
        "first_game_id": game_id,
        "gamestart_utc": gamestart_event["timestamp"],
        "video_timestamp_seconds": video_timestamp_seconds,
        "video_duration_seconds": video_duration,
        "aligned_at": datetime.now().isoformat(),
    }

    save_alignment(tournament_id, alignment_data)

    return jsonify({"success": True, "alignment": alignment_data})


def align_league_night(league_id: str):
    """Save alignment for a league night."""
    # Parse the league ID
    parts = league_id.split("_", 2)
    if len(parts) != 3 or parts[0] != "league":
        return jsonify({"error": "Invalid league night ID"}), 400

    cabinet_id = parts[1]
    date = parts[2]
    key = f"{cabinet_id}_{date}"

    # Find the pending league night
    league_night = pending_league_nights.get(key)
    if not league_night:
        return jsonify({"error": "League night not found"}), 404

    data = request.json
    video_timestamp_seconds = float(data["video_timestamp_seconds"])
    game_id = int(data["game_id"])

    # Fetch gamestart event
    events = fetch_game_events(game_id, verbose=False)
    gamestart_event = next(
        (e for e in events if e.event_type == "gamestart"),
        None
    )

    if not gamestart_event:
        return jsonify({"error": "No gamestart event found for game"}), 400

    video_id = league_night.get("video_id", "")

    # Fetch video duration
    video_duration = get_video_duration(video_id) if video_id else None

    alignment_data = {
        "cabinet_id": cabinet_id,
        "date": date,
        "video_id": video_id,
        "first_game_id": game_id,
        "gamestart_utc": gamestart_event.timestamp.isoformat(),
        "video_timestamp_seconds": video_timestamp_seconds,
        "video_duration_seconds": video_duration,
        "aligned_at": datetime.now().isoformat(),
    }

    save_league_alignment(cabinet_id, date, alignment_data)

    return jsonify({"success": True, "alignment": alignment_data})


# League night endpoints

@app.route("/api/league/<cabinet_id>/recent")
def get_league_recent(cabinet_id: str):
    """Get the most recent video and games for a cabinet."""
    if cabinet_id not in CABINET_TO_YOUTUBE:
        return jsonify({
            "error": f"Unknown cabinet: {cabinet_id}",
            "known_cabinets": list(CABINET_TO_YOUTUBE.keys())
        }), 404

    try:
        # Fetch most recent video
        channel = CABINET_TO_YOUTUBE[cabinet_id]
        video_info = fetch_most_recent_video(channel, verbose=False)

        # Parse upload date
        upload_date_str = video_info['upload_date']
        if len(upload_date_str) == 8:
            target_date = datetime.strptime(upload_date_str, "%Y%m%d")
            target_date = target_date.replace(tzinfo=timezone.utc)
        else:
            return jsonify({"error": f"Invalid upload_date: {upload_date_str}"}), 400

        # Fetch games for that day
        cabinet_url = CABINET_TO_URL[cabinet_id]
        games = fetch_games_for_day(cabinet_url, target_date, verbose=False)

        date_str = target_date.strftime("%Y-%m-%d")
        original_date = target_date

        if not games:
            # Try next day (evening Pacific = next day UTC)
            next_date = original_date + timedelta(days=1)
            games = fetch_games_for_day(cabinet_url, next_date, verbose=False)
            if games:
                date_str = next_date.strftime("%Y-%m-%d")

        if not games:
            # Try day before
            prev_date = original_date - timedelta(days=1)
            games = fetch_games_for_day(cabinet_url, prev_date, verbose=False)
            if games:
                date_str = prev_date.strftime("%Y-%m-%d")

        if not games:
            return jsonify({
                "error": f"No games found for {cabinet_id} around {upload_date_str}",
                "video_info": video_info,
            }), 404

        # Get first game info
        first_game = games[0]

        # Fetch gamestart event for the first game
        events = fetch_game_events(first_game.id, verbose=False)
        gamestart_event = next(
            (e for e in events if e.event_type == "gamestart"),
            None
        )
        gamestart_utc = gamestart_event.timestamp.isoformat() if gamestart_event else None

        # Check for existing alignment
        alignment_key = f"{cabinet_id}_{date_str}"
        existing_alignment = league_alignments.get(alignment_key)

        return jsonify({
            "cabinet_id": cabinet_id,
            "video_info": video_info,
            "date": date_str,
            "games_count": len(games),
            "first_game": {
                "game_id": first_game.id,
                "map_name": first_game.map_name,
                "start_time": first_game.start_time.isoformat(),
                "hivemind_url": f"https://kqhivemind.com/game/{first_game.id}",
                "gamestart_utc": gamestart_utc,
            },
            "alignment": existing_alignment,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/league/<cabinet_id>/align", methods=["POST"])
def align_league(cabinet_id: str):
    """Save alignment for a league night video."""
    if cabinet_id not in CABINET_TO_YOUTUBE:
        return jsonify({
            "error": f"Unknown cabinet: {cabinet_id}",
            "known_cabinets": list(CABINET_TO_YOUTUBE.keys())
        }), 404

    data = request.json
    video_timestamp_seconds = float(data["video_timestamp_seconds"])
    game_id = int(data["game_id"])
    date = data["date"]
    video_id = data["video_id"]

    # Fetch gamestart event
    events = fetch_game_events(game_id, verbose=False)
    gamestart_event = next(
        (e for e in events if e.event_type == "gamestart"),
        None
    )

    if not gamestart_event:
        return jsonify({"error": "No gamestart event found for game"}), 400

    # Fetch video duration
    video_duration = get_video_duration(video_id) if video_id else None

    alignment_data = {
        "cabinet_id": cabinet_id,
        "date": date,
        "video_id": video_id,
        "first_game_id": game_id,
        "gamestart_utc": gamestart_event.timestamp.isoformat(),
        "video_timestamp_seconds": video_timestamp_seconds,
        "video_duration_seconds": video_duration,
        "aligned_at": datetime.now().isoformat(),
    }

    save_league_alignment(cabinet_id, date, alignment_data)

    return jsonify({"success": True, "alignment": alignment_data})


if __name__ == "__main__":
    load_tournament_videos()
    load_matches()
    load_game_details()
    load_alignments()
    load_league_alignments()
    load_pending_league_nights()

    print(f"\nStarting tournament alignment server...")
    print(f"Open http://localhost:5001 in your browser\n")
    app.run(debug=True, port=5001)
