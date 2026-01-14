#!/usr/bin/env python3
"""
Generate chapter data for a video based on HiveMind game data.
"""

import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fetch_games import fetch_games_for_day, Game
from hivemind_api import fetch_game_events
from video_align import find_first_qr, calculate_video_offset, VideoOffset
from chapter_utils import collect_users_for_games, build_output_data

# ML scoring imports (optional - gracefully handle if not available)
try:
    from golden_clips.scorer import get_scorer
    ML_SCORING_AVAILABLE = True
except ImportError:
    ML_SCORING_AVAILABLE = False

# Cabinet ID to YouTube channel mapping
CABINET_TO_YOUTUBE = {
    "sf": "@kqsf",
}

# Cabinet ID to HiveMind URL mapping
CABINET_TO_URL = {
    "sf": "https://kqhivemind.com/cabinet/sf/sf",
}

# File for pending league nights awaiting alignment
PENDING_LEAGUE_NIGHTS_FILE = Path(__file__).parent / "golden_clips" / "pending_league_nights.jsonl"

# File for completed league night alignments
LEAGUE_ALIGNMENTS_FILE = Path(__file__).parent / "golden_clips" / "league_alignments.jsonl"

# Output directory for chapter files
CHAPTERS_OUTPUT_DIR = Path(__file__).parent / "chapters" / "league_nights"


def save_pending_league_night(data: dict):
    """
    Save a pending league night to the file.

    Replaces existing entry if same cabinet_id + date already exists.
    """
    pending = {}

    # Load existing entries
    if PENDING_LEAGUE_NIGHTS_FILE.exists():
        with open(PENDING_LEAGUE_NIGHTS_FILE) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = f"{entry['cabinet_id']}_{entry['date']}"
                    pending[key] = entry

    # Add/update this entry
    key = f"{data['cabinet_id']}_{data['date']}"
    pending[key] = data

    # Write all entries back
    with open(PENDING_LEAGUE_NIGHTS_FILE, "w") as f:
        for entry in pending.values():
            f.write(json.dumps(entry) + "\n")


def load_alignment(cabinet_id: str, date: str) -> dict | None:
    """
    Load alignment data for a cabinet/date combination.

    Returns None if not found.
    """
    if not LEAGUE_ALIGNMENTS_FILE.exists():
        return None

    with open(LEAGUE_ALIGNMENTS_FILE) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry['cabinet_id'] == cabinet_id and entry['date'] == date:
                    return entry
    return None


def build_chapters_from_games(
    games: list,
    video_start_utc: datetime,
    verbose: bool = True
) -> list[dict]:
    """
    Build chapter data for a list of games.

    Args:
        games: List of Game objects
        video_start_utc: Start time of the video in UTC
        verbose: Print progress

    Returns:
        List of chapter dicts
    """
    chapters = []
    set_number = 1
    game_in_set = 0

    for i, game in enumerate(games):
        # Calculate video timestamps
        start_seconds = (game.start_time - video_start_utc).total_seconds()
        end_seconds = (game.end_time - video_start_utc).total_seconds()

        # Skip games that are before or after the video
        if end_seconds < 0:
            continue

        # Detect set boundaries
        is_set_start = False

        if len(chapters) == 0:
            is_set_start = True
        else:
            prev_chapter = chapters[-1]
            prev_game = games[i - 1]
            gap_seconds = (game.start_time - prev_game.end_time).total_seconds()

            # Heuristics for new set:
            # 1. Long gap (> 5 minutes) - definite set break
            # 2. Previous was Twilight AND current is Day AND gap > 90s
            # 3. Current is Day and gap > 2 minutes
            if gap_seconds > 300:
                is_set_start = True
            elif prev_chapter['map'] == 'Twilight' and game.map_name == 'Day' and gap_seconds > 90:
                is_set_start = True
            elif game.map_name == 'Day' and gap_seconds > 120:
                is_set_start = True

        if is_set_start:
            set_number += 1 if len(chapters) > 0 else 0
            game_in_set = 1
        else:
            game_in_set += 1

        adjusted_start = max(0, start_seconds - 1)

        # Fetch events for this game
        if verbose:
            print(f"  Fetching events for game {game.id}...", end='\r')

        game_events = fetch_game_events(game.id, verbose=False)
        queen_kills = extract_queen_kills(game_events, video_start_utc)
        player_events = extract_player_events(game_events, video_start_utc)
        kill_events = extract_kill_events(game_events, video_start_utc)
        win_prob_timeline = extract_win_prob_timeline(game_events, video_start_utc)

        # Add ML scores to significant events
        if ML_SCORING_AVAILABLE:
            scorer = get_scorer()
            WINDOW_SIZES = [6, 8, 10, 12, 15]  # seconds to try
            for evt in player_events:
                if abs(evt.get('delta', 0)) >= 0.10:
                    # Find the original event to get its timestamp
                    for raw_evt in game_events:
                        if raw_evt.id == evt.get('id'):
                            # Try multiple window sizes, keep best score
                            best_result = None
                            best_window = None
                            for window_seconds in WINDOW_SIZES:
                                result = scorer.score_around_event(game_events, raw_evt.timestamp, window_seconds)
                                if result.success and (best_result is None or result.score > best_result.score):
                                    best_result = result
                                    best_window = window_seconds
                            if best_result and best_result.success:
                                evt['ml_score'] = best_result.score
                                evt['window_size'] = best_window
                            break

        chapter = {
            "game_id": game.id,
            "title": f"Game {game.id}: {game.map_name}",
            "map": game.map_name,
            "winner": game.winning_team,
            "win_condition": game.win_condition,
            "start_time": adjusted_start,
            "end_time": end_seconds,
            "duration": end_seconds - adjusted_start,
            "hivemind_url": f"https://kqhivemind.com/game/{game.id}",
            "set_number": set_number,
            "game_in_set": game_in_set,
            "is_set_start": is_set_start,
            "queen_kills": queen_kills,
            "player_events": player_events,
            "kill_events": kill_events,
            "win_timeline": win_prob_timeline,
        }
        chapters.append(chapter)

    if verbose:
        print(f"  Generated {len(chapters)} chapters" + " " * 30)

    return chapters


def generate_chapters_from_alignment(
    cabinet_id: str,
    alignment: dict,
    games: list,
    video_info: dict,
    verbose: bool = True
) -> str:
    """
    Generate chapter data using manual alignment.

    Args:
        cabinet_id: Cabinet ID (e.g., "sf")
        alignment: Alignment data from league_alignments.jsonl
        games: List of Game objects for the day
        video_info: Video metadata from yt-dlp
        verbose: Print progress

    Returns:
        Path to the saved chapter file
    """
    # Calculate video start UTC from alignment
    gamestart_utc = datetime.fromisoformat(alignment['gamestart_utc'])
    video_timestamp_seconds = alignment['video_timestamp_seconds']
    video_start_utc = gamestart_utc - timedelta(seconds=video_timestamp_seconds)

    if verbose:
        print(f"\nGenerating chapters from alignment:")
        print(f"  Aligned game: {alignment['first_game_id']}")
        print(f"  Game start UTC: {gamestart_utc}")
        print(f"  Video timestamp: {video_timestamp_seconds:.1f}s")
        print(f"  Video start UTC: {video_start_utc}")

    # Generate chapters
    chapters = build_chapters_from_games(games, video_start_utc, verbose=verbose)

    # Fetch user data
    if verbose:
        print(f"  Fetching user sign-ins...")
    game_ids = [g.id for g in games]
    users_dict, game_users_map = collect_users_for_games(game_ids, verbose=False)
    if verbose:
        print(f"  Found {len(users_dict)} unique users")

    # Add user mappings to chapters
    for chapter in chapters:
        if chapter['game_id'] in game_users_map:
            chapter['users'] = game_users_map[chapter['game_id']]

    # Build output data
    output_data = {
        'video_id': video_info['video_id'],
        'video_title': video_info['title'],
        'video_start_utc': video_start_utc.isoformat(),
        'cabinet_id': cabinet_id,
        'alignment': {
            'game_id': alignment['first_game_id'],
            'video_timestamp_seconds': video_timestamp_seconds,
            'aligned_at': alignment['aligned_at'],
        },
        'chapters': chapters,
        'users': users_dict,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

    # Ensure output directory exists
    CHAPTERS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save to file
    date_str = alignment['date']
    output_path = CHAPTERS_OUTPUT_DIR / f"{cabinet_id}-{date_str}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nSaved chapters to: {output_path}")

    return str(output_path)


def fetch_most_recent_video(channel_handle: str, verbose: bool = True) -> dict:
    """
    Fetch the most recent video from a YouTube channel using yt-dlp.

    Args:
        channel_handle: YouTube channel handle (e.g., "@kqsf")
        verbose: Print progress

    Returns:
        Dict with keys: video_id, upload_date, title, duration
    """
    import re

    url = f"https://www.youtube.com/{channel_handle}/videos"

    if verbose:
        print(f"Fetching most recent video from {channel_handle}...")

    # Use --flat-playlist to avoid bot detection issues with full metadata
    result = subprocess.run(
        ['yt-dlp', '--dump-json', '--flat-playlist', '--playlist-items', '1', url],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Try to parse JSON even if return code is non-zero (warnings may cause this)
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"yt-dlp failed to return valid JSON: {result.stderr}")

    # Extract upload_date - try from metadata first, then parse from title
    upload_date = info.get('upload_date', '')

    if not upload_date:
        # Try to parse date from title (e.g., "Monday Mixer 1/12/2026")
        title = info.get('title', '')
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', title)
        if date_match:
            month, day, year = date_match.groups()
            upload_date = f"{year}{int(month):02d}{int(day):02d}"

    return {
        'video_id': info['id'],
        'upload_date': upload_date,
        'title': info.get('title', ''),
        'duration': info.get('duration', 0),
    }


def extract_queen_kills(events, video_start_utc) -> list[dict]:
    """Extract queen kill events and convert to video timestamps."""
    queen_kills = []
    for event in events:
        if event.event_type != 'playerKill':
            continue
        values = event.values
        if len(values) >= 5 and values[4] == 'Queen':
            video_time = (event.timestamp - video_start_utc).total_seconds()
            # Position 1 = Gold Queen, Position 2 = Blue Queen
            victim_pos = int(values[3])
            victim_team = 'gold' if victim_pos == 1 else 'blue'
            queen_kills.append({
                'time': video_time,
                'victim': victim_team,
            })
    return queen_kills


def extract_positions_from_event(event) -> list[int]:
    """Extract position IDs involved in an event."""
    values = event.values
    positions = []

    try:
        if event.event_type == 'playerKill':
            # values: [x, y, killer_pos, victim_pos, unit_type]
            if len(values) >= 4:
                positions = [int(values[2]), int(values[3])]
        elif event.event_type == 'snailEat':
            # values: [x, snail_id, rider_pos, victim_pos]
            if len(values) >= 4:
                positions = [int(values[2]), int(values[3])]
        elif event.event_type == 'berryDeposit':
            # values: [x, y, position_id]
            if len(values) >= 3:
                positions = [int(values[2])]
        elif event.event_type == 'useMaiden':
            # values: [x, y, maiden_type, position_id]
            if len(values) >= 4:
                positions = [int(values[3])]
        elif event.event_type in ('getOnSnail', 'getOffSnail', 'snailEscape'):
            # values: [x, snail_id, position_id, ...]
            if len(values) >= 3:
                positions = [int(values[2])]
        elif event.event_type == 'carryFood':
            # values: [position_id]
            if len(values) >= 1:
                positions = [int(values[0])]
    except (ValueError, IndexError):
        pass

    return positions


def extract_player_events(events, video_start_utc) -> list[dict]:
    """Extract events with win probability changes per position."""
    player_events = []
    prev_prob = 0.5

    for event in events:
        if event.win_probability is None:
            continue

        try:
            curr_prob = float(event.win_probability)
        except (ValueError, TypeError):
            continue

        delta = curr_prob - prev_prob  # Keep sign for direction

        # Get positions involved in this event
        positions = extract_positions_from_event(event)

        if positions and abs(delta) > 0.01:  # Threshold for significance
            video_time = (event.timestamp - video_start_utc).total_seconds()
            player_events.append({
                'id': event.id,
                'time': video_time,
                'type': event.event_type,
                'positions': positions,
                'delta': round(delta, 4),
                'values': event.values,
            })

        prev_prob = curr_prob

    return player_events


def extract_kill_events(events, video_start_utc) -> list[dict]:
    """Extract all kill events (playerKill and snailEat) for K/D tracking."""
    kill_events = []

    for event in events:
        if event.event_type not in ('playerKill', 'snailEat'):
            continue

        values = event.values
        if len(values) < 4:
            continue

        try:
            if event.event_type == 'playerKill':
                # values: [x, y, killer_pos, victim_pos, unit_type]
                killer = int(values[2])
                victim = int(values[3])
            else:  # snailEat
                # values: [x, snail_id, rider_pos, victim_pos]
                killer = int(values[2])
                victim = int(values[3])
        except (ValueError, TypeError):
            continue

        video_time = (event.timestamp - video_start_utc).total_seconds()
        kill_events.append({
            'time': round(video_time, 2),
            'type': event.event_type,
            'killer': killer,
            'victim': victim,
        })

    return kill_events


def extract_win_prob_timeline(events, video_start_utc) -> list[dict]:
    """Extract win probability timeline for plotting."""
    timeline = []

    for event in events:
        if event.win_probability is None:
            continue

        try:
            prob = float(event.win_probability)
        except (ValueError, TypeError):
            continue

        video_time = (event.timestamp - video_start_utc).total_seconds()
        positions = extract_positions_from_event(event)

        timeline.append({
            't': round(video_time, 2),
            'p': round(prob, 3),
            'pos': positions if positions else [],
        })

    return timeline


def generate_chapters(
    video_path: str,
    cabinet_url: str,
    output_path: str | None = None,
    cab_id: str | None = None,
    video_id: str | None = None,
    include_users: bool = True,
    verbose: bool = True
) -> list[dict]:
    """
    Generate chapter data for a video.

    Args:
        video_path: Path to the video file
        cabinet_url: Cabinet URL (e.g., https://kqhivemind.com/cabinet/sf/sf)
        output_path: Path to save JSON output (optional, auto-generated if None and cab_id provided)
        cab_id: Cabinet ID for auto-generating output path (e.g., "sf")
        video_id: YouTube video ID for the stream
        include_users: Fetch HiveMind user sign-ins (default True)
        verbose: Print progress

    Returns:
        List of chapter dicts with video timing info
    """
    if verbose:
        print("Step 1: Finding QR code in video...")

    qr = find_first_qr(video_path, verbose=verbose)
    if qr is None:
        raise ValueError("No QR code found in video")

    if verbose:
        print(f"\nStep 2: Fetching game events for game {qr.game_id}...")

    events = fetch_game_events(qr.game_id, verbose=verbose)
    victory_events = [e for e in events if e.event_type == "victory"]
    if not victory_events:
        raise ValueError(f"No victory event found for game {qr.game_id}")

    hivemind_victory_utc = victory_events[0].timestamp

    if verbose:
        print(f"\nStep 3: Calculating video offset...")

    offset = calculate_video_offset(video_path, hivemind_victory_utc, qr, verbose=verbose)

    # Determine the target date from the game
    target_date = hivemind_victory_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    if verbose:
        print(f"\nStep 4: Fetching all games for {target_date.date()}...")

    games = fetch_games_for_day(cabinet_url, target_date, verbose=verbose)

    # Step 5: Fetch user data if requested
    users_dict = None
    game_users_map = {}
    if include_users:
        if verbose:
            print(f"\nStep 5: Fetching user sign-ins...")
        game_ids = [g.id for g in games]
        users_dict, game_users_map = collect_users_for_games(game_ids, verbose=verbose)
        if verbose:
            print(f"  Found {len(users_dict)} unique users")

    if verbose:
        print(f"\nStep 6: Generating chapters for {len(games)} games...")

    chapters = build_chapters_from_games(games, offset.video_start_utc, verbose=verbose)

    # Add user mappings to chapters
    for chapter in chapters:
        if chapter['game_id'] in game_users_map:
            chapter['users'] = game_users_map[chapter['game_id']]

    # Auto-generate output path if cab_id provided but no output_path
    if output_path is None and cab_id:
        date_str = target_date.strftime("%Y-%m-%d")
        output_path = str(Path(__file__).parent / "chapters" / "league_nights" / f"{cab_id}-{date_str}.json")
        if verbose:
            print(f"  Auto-generated output path: {output_path}")

    # Save to file if we have a path
    if output_path:
        output_data = build_output_data(
            video_id=video_id,
            video_start_utc=offset.video_start_utc,
            chapters=chapters,
            users=users_dict,
            video_path=video_path,
            cabinet_url=cabinet_url,
            fps=offset.fps,
        )
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nSaved to: {output_path}")

    return chapters


def generate_chapters_from_cabinet(
    cabinet_id: str,
    verbose: bool = True
) -> dict:
    """
    Fetch most recent video for a cabinet and prepare for alignment.

    Args:
        cabinet_id: Cabinet ID (e.g., "sf")
        verbose: Print progress

    Returns:
        Dict with video_info, games, and first_game for alignment
    """
    # 1. Look up YouTube channel
    if cabinet_id not in CABINET_TO_YOUTUBE:
        raise ValueError(
            f"Unknown cabinet: {cabinet_id}. "
            f"Known cabinets: {list(CABINET_TO_YOUTUBE.keys())}"
        )

    channel = CABINET_TO_YOUTUBE[cabinet_id]
    cabinet_url = CABINET_TO_URL[cabinet_id]

    # 2. Fetch most recent video
    video_info = fetch_most_recent_video(channel, verbose=verbose)

    if verbose:
        print(f"  Video: {video_info['title']}")
        print(f"  Video ID: {video_info['video_id']}")
        print(f"  Upload date: {video_info['upload_date']}")

    # 3. Parse upload date to get target day for games
    upload_date_str = video_info['upload_date']
    if len(upload_date_str) == 8:
        target_date = datetime.strptime(upload_date_str, "%Y%m%d")
        target_date = target_date.replace(tzinfo=timezone.utc)
    else:
        raise ValueError(f"Invalid upload_date format: {upload_date_str}")

    # 4. Fetch games for that day
    if verbose:
        print(f"\nFetching games for {target_date.date()}...")

    games = fetch_games_for_day(cabinet_url, target_date, verbose=verbose)

    original_date = target_date

    if not games:
        # Try next day (evening Pacific = next day UTC)
        next_date = original_date + timedelta(days=1)
        if verbose:
            print(f"No games found, trying {next_date.date()}...")
        games = fetch_games_for_day(cabinet_url, next_date, verbose=verbose)
        if games:
            target_date = next_date

    if not games:
        # Try day before (video might be uploaded next morning)
        prev_date = original_date - timedelta(days=1)
        if verbose:
            print(f"No games found, trying {prev_date.date()}...")
        games = fetch_games_for_day(cabinet_url, prev_date, verbose=verbose)
        if games:
            target_date = prev_date

    if not games:
        raise ValueError(f"No games found for {cabinet_id} around {upload_date_str}")

    if verbose:
        print(f"  Found {len(games)} games for the day")

    # 5. Calculate which games are actually in the stream
    # Use video duration to estimate stream start time
    duration = video_info.get('duration') or 0
    last_game = games[-1]

    if duration > 0:
        # Estimate stream start: last game end time minus video duration
        stream_start_approx = last_game.end_time - timedelta(seconds=duration)

        # Find games that started after the stream began
        games_in_stream = [g for g in games if g.start_time >= stream_start_approx]

        if games_in_stream:
            first_game = games_in_stream[0]
        else:
            # Fallback if calculation seems off
            first_game = games[0]
            games_in_stream = games

        if verbose:
            print(f"  Video duration: {duration:.0f}s ({duration/60:.1f} min)")
            print(f"  Estimated stream start: {stream_start_approx}")
            print(f"  Games in stream: {len(games_in_stream)} (skipped {len(games) - len(games_in_stream)} pre-stream games)")
    else:
        # No duration info, use all games
        first_game = games[0]
        games_in_stream = games
        stream_start_approx = first_game.start_time

    if verbose:
        print(f"\nFirst game for alignment:")
        print(f"  Game ID: {first_game.id}")
        print(f"  Map: {first_game.map_name}")
        print(f"  Start time: {first_game.start_time}")
        print(f"  HiveMind: https://kqhivemind.com/game/{first_game.id}")

    # 6. Check if alignment already exists
    date_str = target_date.strftime("%Y-%m-%d")
    alignment = load_alignment(cabinet_id, date_str)

    if alignment and alignment.get('video_id') == video_info['video_id']:
        # Alignment exists - generate chapters
        if verbose:
            print(f"\nFound existing alignment for {cabinet_id} {date_str}")

        output_path = generate_chapters_from_alignment(
            cabinet_id=cabinet_id,
            alignment=alignment,
            games=games_in_stream,
            video_info=video_info,
            verbose=verbose,
        )

        return {
            'cabinet_id': cabinet_id,
            'video_info': video_info,
            'target_date': date_str,
            'games': games_in_stream,
            'first_game': first_game,
            'chapters_file': output_path,
        }

    # 7. No alignment - save to pending league nights for alignment server
    pending_data = {
        'cabinet_id': cabinet_id,
        'video_id': video_info['video_id'],
        'title': video_info['title'],
        'upload_date': video_info['upload_date'],
        'date': date_str,
        'duration': duration,
        'stream_start_utc': stream_start_approx.isoformat() if duration > 0 else None,
        'first_game_id': first_game.id,
        'games_count': len(games_in_stream),
        'scraped_at': datetime.now(timezone.utc).isoformat(),
    }
    save_pending_league_night(pending_data)

    if verbose:
        print(f"\nSaved to pending league nights for alignment.")

    return {
        'cabinet_id': cabinet_id,
        'video_info': video_info,
        'target_date': date_str,
        'games': games_in_stream,
        'first_game': first_game,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chapter data for Killer Queen league night videos"
    )
    parser.add_argument(
        "--cabinet", "-c",
        required=True,
        help=f"Cabinet ID (one of: {', '.join(CABINET_TO_YOUTUBE.keys())})"
    )

    args = parser.parse_args()

    result = generate_chapters_from_cabinet(args.cabinet)

    print(f"\n" + "=" * 50)
    if 'chapters_file' in result:
        print(f"Chapters generated: {result['chapters_file']}")
    else:
        print("To complete alignment, start the alignment server:")
        print("  python golden_clips/tournament_align_server.py")
        print(f"\nThen open: http://localhost:5001")
    print("=" * 50)
