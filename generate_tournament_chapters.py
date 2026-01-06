"""
Generate chapter data for a tournament video.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from tournament_api import (
    fetch_tournament_info,
    fetch_tournament_matches,
    fetch_video_timestamps,
    fetch_all_tournament_games,
)
from tournament_video import fetch_tournament_videos, TournamentVideo
from hivemind_api import fetch_game_events, fetch_game_detail, fetch_user_public_data
from generate_chapters import (
    extract_queen_kills,
    extract_kill_events,
    extract_win_prob_timeline,
    extract_player_events,
)


def load_tournament_offset(tournament_id: int) -> float:
    """Load video offset for a tournament from config."""
    config_path = Path(__file__).parent / "tournament_offsets.json"
    if config_path.exists():
        with open(config_path) as f:
            offsets = json.load(f)
            return offsets.get(str(tournament_id), 0.0)
    return 0.0


def generate_tournament_chapters(
    tournament_id: int,
    output_path: str | None = None,
    verbose: bool = True
) -> dict:
    """
    Generate chapter data for a tournament.

    Args:
        tournament_id: The tournament ID
        output_path: Path to save JSON output (optional)
        verbose: Print progress

    Returns:
        Dict with tournament info, players, and chapters
    """
    if verbose:
        print(f"Generating chapters for tournament {tournament_id}...")

    # Load calibration offset for this tournament
    offset_seconds = load_tournament_offset(tournament_id)
    if verbose and offset_seconds != 0:
        print(f"  Using video offset: {offset_seconds}s")

    # Step 1: Get tournament info
    if verbose:
        print("\nStep 1: Fetching tournament info...")
    tournament_info = fetch_tournament_info(tournament_id, verbose=verbose)

    # Step 2: Get tournament video
    if verbose:
        print("\nStep 2: Fetching tournament videos...")
    videos = fetch_tournament_videos(tournament_id, verbose=verbose)
    if not videos:
        raise ValueError(f"No videos found for tournament {tournament_id}")

    # For now, use the first video
    video = videos[0]
    if verbose:
        print(f"  Using video: {video.video_id} ({video.cabinet_name})")

    # Step 3: Get match timestamps
    if verbose:
        print("\nStep 3: Fetching match timestamps...")
    timestamps = fetch_video_timestamps(video.id, verbose=verbose)

    # Step 4: Get all games from tournament structure
    if verbose:
        print("\nStep 4: Fetching games from tournament structure...")
    games = fetch_all_tournament_games(tournament_id, verbose=verbose)

    # Parse start_time strings to datetime for video timestamp calculation
    from dateutil import parser as date_parser
    for game in games:
        if isinstance(game.get('start_time'), str):
            game['start_time'] = date_parser.isoparse(game['start_time'])

    # Step 5: Fetch game details and collect user IDs
    if verbose:
        print(f"\nStep 5: Fetching game details for {len(games)} games...")

    all_user_ids = set()
    game_details = {}

    for i, game in enumerate(games):
        game_id = game['id']
        if verbose:
            print(f"  Fetching game detail {i+1}/{len(games)}: {game_id}")

        detail = fetch_game_detail(game_id, verbose=False)
        game_details[game_id] = detail

        # Collect user IDs from this game
        for user_entry in detail.get('users', []):
            user_id = user_entry.get('user')
            if user_id:
                all_user_ids.add(user_id)

    # Step 6: Fetch user names
    if verbose:
        print(f"\nStep 6: Fetching {len(all_user_ids)} user profiles...")

    users = {}
    for i, user_id in enumerate(sorted(all_user_ids)):
        if verbose:
            print(f"  Fetching user {i+1}/{len(all_user_ids)}: {user_id}")
        user_data = fetch_user_public_data(user_id, verbose=False)
        users[str(user_id)] = {
            'name': user_data.get('name', f'User {user_id}'),
            'scene': user_data.get('scene'),
        }

    # Step 7: Process each game into chapters
    if verbose:
        print(f"\nStep 7: Processing {len(games)} games into chapters...")

    chapters = []

    for i, game in enumerate(games):
        game_id = game['id']
        if verbose:
            print(f"  Processing game {i+1}/{len(games)}: {game_id}")

        # Fetch game events
        events = fetch_game_events(game_id, verbose=False)

        # Build user mapping for this game (position -> user_id)
        detail = game_details[game_id]
        game_users = {}
        for user_entry in detail.get('users', []):
            pos = user_entry.get('player_id')
            user_id = user_entry.get('user')
            if pos and user_id:
                game_users[pos] = user_id

        # Calculate video timestamps (apply calibration offset)
        video_start = (game['start_time'] - video.start_time).total_seconds() - offset_seconds

        # Get end time from game detail
        end_time_str = game_details[game_id].get('end_time') or game_details[game_id].get('endTime')
        if end_time_str:
            game_end = date_parser.isoparse(end_time_str)
            video_end = (game_end - video.start_time).total_seconds() - offset_seconds
            duration = video_end - video_start
        else:
            # Fallback: estimate from events
            duration = 120  # default 2 minutes
            video_end = video_start + duration

        # Create adjusted reference time for extract_* functions
        adjusted_video_start = video.start_time + timedelta(seconds=offset_seconds)

        # Extract events for chapter
        queen_kills = extract_queen_kills(events, adjusted_video_start)
        kill_events = extract_kill_events(events, adjusted_video_start)
        win_timeline = extract_win_prob_timeline(events, adjusted_video_start)
        player_events = extract_player_events(events, adjusted_video_start)

        # Match info comes from tournament structure
        match_info = {
            'blue': game.get('blue_team', ''),
            'gold': game.get('gold_team', ''),
        }

        # Determine if this is start of a new match (set)
        current_match = game.get('match_id')
        is_set_start = (i == 0) or (games[i-1].get('match_id') != current_match)

        chapter = {
            'game_id': game_id,
            'title': f"Game {game_id}: {game['map_name']}",
            'map': game['map_name'],
            'winner': game['winning_team'],
            'win_condition': game['win_condition'],
            'start_time': video_start,
            'end_time': video_end,
            'duration': duration,
            'is_set_start': is_set_start,
            'set_number': current_match,  # Use match_id as set number
            'users': game_users,  # position -> user_id mapping
            'match_info': match_info,
            'queen_kills': queen_kills,
            'kill_events': kill_events,
            'win_timeline': win_timeline,
            'player_events': player_events,
            'hivemind_url': f"https://kqhivemind.com/game/{game_id}",
        }
        chapters.append(chapter)

    # Build output data
    # Use adjusted video start time in output
    adjusted_video_start_utc = video.start_time + timedelta(seconds=offset_seconds)
    output_data = {
        'tournament_id': tournament_id,
        'tournament_name': tournament_info.get('name', ''),
        'video_id': video.video_id,
        'video_start_utc': adjusted_video_start_utc.isoformat(),
        'offset_seconds': offset_seconds,  # Include for reference
        'users': users,  # user_id -> {name, scene}
        'chapters': chapters,
    }

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nSaved to: {output_path}")

    if verbose:
        print(f"\nGenerated {len(chapters)} chapters")
        print(f"Found {len(users)} unique users:")
        for user_id, user_info in sorted(users.items(), key=lambda x: x[1]['name'].lower()):
            print(f"  {user_id}: {user_info['name']}")

    return output_data


if __name__ == "__main__":
    import sys

    tournament_id = int(sys.argv[1]) if len(sys.argv) > 1 else 842
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tournament_chapters.json"

    data = generate_tournament_chapters(tournament_id, output_path)

    print(f"\nChapters:")
    for ch in data['chapters'][:10]:
        start_min = int(ch['start_time'] // 60)
        start_sec = ch['start_time'] % 60
        # Get user names for this game
        user_names = []
        for pos, user_id in ch.get('users', {}).items():
            user_info = data['users'].get(str(user_id), {})
            name = user_info.get('name', f'User {user_id}')
            user_names.append(f"P{pos}:{name}")
        print(f"  {start_min}:{start_sec:05.2f} - {ch['title']} ({ch['winner']} {ch['win_condition']})")
        if user_names:
            print(f"    Players: {', '.join(user_names[:5])}")
    if len(data['chapters']) > 10:
        print(f"  ... and {len(data['chapters']) - 10} more")
