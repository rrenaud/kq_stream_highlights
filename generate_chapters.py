"""
Generate chapter data for a video based on HiveMind game data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fetch_games import fetch_games_for_day, Game
from hivemind_api import fetch_game_events
from video_align import find_first_qr, calculate_video_offset, VideoOffset
from chapter_utils import (
    collect_users_for_games, build_output_data,
    extract_queen_kills, extract_player_events,
    extract_kill_events, extract_win_prob_timeline,
)


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
        print("  Fetching events for each game...")

    chapters = []
    set_number = 1
    game_in_set = 0

    for i, game in enumerate(games):
        # Calculate video timestamps
        start_seconds = (game.start_time - offset.video_start_utc).total_seconds()
        end_seconds = (game.end_time - offset.video_start_utc).total_seconds()

        # Skip games that are before or after the video
        if end_seconds < 0:
            continue

        # Detect set boundaries
        is_set_start = False

        if len(chapters) == 0:
            # First game always starts a set
            is_set_start = True
        else:
            prev_chapter = chapters[-1]
            prev_game = games[i - 1]

            # Calculate gap between games
            gap_seconds = (game.start_time - prev_game.end_time).total_seconds()

            # Heuristics for new set:
            # 1. Long gap (> 5 minutes) between games - definite set break
            # 2. Previous game was Twilight AND current is Day AND gap > 90s
            #    (need some gap to avoid false positives from quick restarts)
            # 3. Current is Day and gap > 2 minutes (weaker signal)

            if gap_seconds > 300:  # > 5 minutes gap - definite break
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

        # Start 1 second earlier to not miss the beginning
        adjusted_start = max(0, start_seconds - 1)

        # Fetch events for this game
        game_events = fetch_game_events(game.id, verbose=False)
        queen_kills = extract_queen_kills(game_events, offset.video_start_utc)
        player_events = extract_player_events(game_events, offset.video_start_utc)
        kill_events = extract_kill_events(game_events, offset.video_start_utc)
        win_prob_timeline = extract_win_prob_timeline(game_events, offset.video_start_utc)

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
        # Add user mapping if available
        if game.id in game_users_map:
            chapter["users"] = game_users_map[game.id]
        chapters.append(chapter)

    if verbose:
        print(f"  Generated {len(chapters)} chapters")

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


if __name__ == "__main__":
    import sys
    import re

    if len(sys.argv) < 4:
        print("Usage: python generate_chapters.py <video_file> <cabinet_url> <video_id> [output.json]")
        print("Example: python generate_chapters.py sf_12_15_2025.mkv https://kqhivemind.com/cabinet/sf/sf UmxLa4CW9cY")
        print("Output defaults to: chapters/league_nights/{cab_id}-{YYYY-MM-DD}.json")
        sys.exit(1)

    video_path = sys.argv[1]
    cabinet_url = sys.argv[2]
    video_id = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else None

    # Extract cabinet ID from URL (e.g., "sf" from "https://kqhivemind.com/cabinet/sf/sf")
    cab_match = re.search(r'/cabinet/([^/]+)', cabinet_url)
    cab_id = cab_match.group(1) if cab_match else "unknown"

    chapters = generate_chapters(video_path, cabinet_url, output_path, cab_id=cab_id, video_id=video_id)

    print(f"\nChapters:")
    for ch in chapters[:10]:
        start_min = int(ch['start_time'] // 60)
        start_sec = ch['start_time'] % 60
        print(f"  {start_min}:{start_sec:05.2f} - {ch['title']} ({ch['winner']} {ch['win_condition']})")
    if len(chapters) > 10:
        print(f"  ... and {len(chapters) - 10} more")
