"""
Generate chapter data for a video based on HiveMind game data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fetch_games import fetch_games_for_day, Game
from hivemind_api import fetch_game_events
from video_align import find_first_qr, calculate_video_offset, VideoOffset


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
    verbose: bool = True
) -> list[dict]:
    """
    Generate chapter data for a video.

    Args:
        video_path: Path to the video file
        cabinet_url: Cabinet URL (e.g., https://kqhivemind.com/cabinet/sf/sf)
        output_path: Path to save JSON output (optional)
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

    if verbose:
        print(f"\nStep 5: Generating chapters for {len(games)} games...")
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
        chapters.append(chapter)

    if verbose:
        print(f"  Generated {len(chapters)} chapters")

    # Save to file if requested
    if output_path:
        output_data = {
            "video_path": video_path,
            "cabinet_url": cabinet_url,
            "video_start_utc": offset.video_start_utc.isoformat(),
            "fps": offset.fps,
            "chapters": chapters,
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nSaved to: {output_path}")

    return chapters


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate_chapters.py <video_file> <cabinet_url> [output.json]")
        print("Example: python generate_chapters.py sf_12_15_2025.mkv https://kqhivemind.com/cabinet/sf/sf chapters.json")
        sys.exit(1)

    video_path = sys.argv[1]
    cabinet_url = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "chapters.json"

    chapters = generate_chapters(video_path, cabinet_url, output_path)

    print(f"\nChapters:")
    for ch in chapters[:10]:
        start_min = int(ch['start_time'] // 60)
        start_sec = ch['start_time'] % 60
        print(f"  {start_min}:{start_sec:05.2f} - {ch['title']} ({ch['winner']} {ch['win_condition']})")
    if len(chapters) > 10:
        print(f"  ... and {len(chapters) - 10} more")
