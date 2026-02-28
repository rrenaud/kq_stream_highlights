"""
Generate chapter data for a multi-cab league night.

Merges games from two cabinets into a single chapter file with
per-chapter video_source tags. Chapter timestamps are seconds
into each cab's video stream.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hivemind_api import fetch_game_events
from chapter_utils import (
    collect_users_for_games, build_output_data,
    extract_queen_kills, extract_player_events,
    extract_kill_events, extract_win_prob_timeline,
)


def load_games_from_cache(cache_path: Path) -> list[dict]:
    """Load games from a JSONL cache file, handling both flat and nested formats."""
    games = []
    with open(cache_path) as f:
        for line in f:
            item = json.loads(line)
            # Normalize: the kqsf0 cache uses nested format from the API
            if 'cabinet' in item and isinstance(item['cabinet'], dict):
                # Nested API format
                start_time = item['start_time'].replace('Z', '+00:00')
                end_time = item['end_time'].replace('Z', '+00:00')
                games.append({
                    'id': item['id'],
                    'cabinet_id': item['cabinet']['id'],
                    'cabinet_name': item['cabinet']['name'],
                    'map_name': item.get('map_name', ''),
                    'start_time': start_time,
                    'end_time': end_time,
                    'win_condition': item.get('win_condition', ''),
                    'winning_team': item.get('winning_team', ''),
                    'player_count': item.get('player_count', 0),
                })
            else:
                # Flat format (from save_games_to_cache)
                games.append(item)
    return games


def generate_multi_cab_chapters(
    cab_configs: list[dict],
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Generate a multi-cab chapter file.

    Args:
        cab_configs: List of dicts with keys:
            - cache_path: Path to cabinet game cache JSONL
            - video_source: Key for this cab in the videos dict (e.g., "cab1")
            - video_id: YouTube video ID (or None)
            - label: Display label (e.g., "Cab 1")
            - video_start_utc: (optional) UTC datetime when video time 00:00 occurs.
              If provided, chapter timestamps are video-relative. Otherwise,
              timestamps are relative to the first game on this cab.
        output_path: Where to write the output JSON
        verbose: Print progress
    """
    all_sets = []  # list of (utc_start, [chapters])
    all_game_ids = []
    videos = {}

    for config in cab_configs:
        cache_path = Path(config['cache_path'])
        source_key = config['video_source']
        video_id = config.get('video_id')
        label = config['label']

        videos[source_key] = {
            'video_id': video_id,
            'label': label,
        }

        if verbose:
            print(f"\n=== Loading {label} from {cache_path} ===")

        games = load_games_from_cache(cache_path)
        if verbose:
            print(f"  {len(games)} games loaded")

        if not games:
            continue

        # Parse timestamps and sort by start_time
        for g in games:
            if isinstance(g['start_time'], str):
                g['start_time'] = datetime.fromisoformat(g['start_time'])
            if isinstance(g['end_time'], str):
                g['end_time'] = datetime.fromisoformat(g['end_time'])
        games.sort(key=lambda g: g['start_time'])

        # Reference time: video start if provided, else first game start
        if config.get('video_start_utc'):
            ref_utc = config['video_start_utc']
            if verbose:
                print(f"  Video 00:00 at {ref_utc}")
                offset = (games[0]['start_time'] - ref_utc).total_seconds()
                print(f"  First game at {games[0]['start_time']} ({offset:+.0f}s into video)")
        else:
            ref_utc = games[0]['start_time']
            if verbose:
                print(f"  First game at {ref_utc} (no video alignment)")

        # Build chapters for this cab, grouped into sets
        current_set = []
        set_utc_start = None
        prev_chapter = None
        prev_game = None

        skipped = 0
        for i, game in enumerate(games):
            start_seconds = (game['start_time'] - ref_utc).total_seconds()
            end_seconds = (game['end_time'] - ref_utc).total_seconds()

            # Skip games that ended before the video started
            if end_seconds < 0:
                skipped += 1
                continue

            game_id = game['id']
            all_game_ids.append(game_id)

            # Detect set boundaries
            is_set_start = False
            if prev_chapter is None:
                is_set_start = True
            else:
                gap_seconds = (game['start_time'] - prev_game['end_time']).total_seconds()
                if gap_seconds > 300:
                    is_set_start = True
                elif prev_chapter['map'] == 'Twilight' and game['map_name'] == 'Day' and gap_seconds > 90:
                    is_set_start = True
                elif game['map_name'] == 'Day' and gap_seconds > 120:
                    is_set_start = True

            if is_set_start and current_set:
                # Flush previous set
                all_sets.append((set_utc_start, current_set))
                current_set = []

            if is_set_start:
                set_utc_start = game['start_time']

            adjusted_start = max(0, start_seconds - 1)

            # Fetch events
            if verbose:
                print(f"  [{i+1}/{len(games)}] Game {game_id} ({game['map_name']})...", end=" ", flush=True)

            game_events = fetch_game_events(game_id, verbose=False)
            queen_kills = extract_queen_kills(game_events, ref_utc)
            player_events = extract_player_events(game_events, ref_utc)
            kill_events = extract_kill_events(game_events, ref_utc)
            win_timeline = extract_win_prob_timeline(game_events, ref_utc)

            if verbose:
                print(f"{len(game_events)} events")

            chapter = {
                'game_id': game_id,
                'title': f"Game {game_id}: {game['map_name']}",
                'map': game['map_name'],
                'winner': game['winning_team'],
                'win_condition': game['win_condition'],
                'start_time': adjusted_start,
                'end_time': end_seconds,
                'duration': end_seconds - adjusted_start,
                'hivemind_url': f"https://kqhivemind.com/game/{game_id}",
                'is_set_start': is_set_start,
                'queen_kills': queen_kills,
                'player_events': player_events,
                'kill_events': kill_events,
                'win_timeline': win_timeline,
                'video_source': source_key,
            }
            current_set.append(chapter)
            prev_chapter = chapter
            prev_game = game

        # Flush last set
        if current_set:
            all_sets.append((set_utc_start, current_set))

        if verbose and skipped:
            print(f"  Skipped {skipped} pre-stream games")

    # Sort sets by their start time, then flatten with global set numbering
    all_sets.sort(key=lambda s: s[0])

    all_chapters = []
    for set_number, (_utc, set_chapters) in enumerate(all_sets, start=1):
        for game_in_set, ch in enumerate(set_chapters, start=1):
            ch['set_number'] = set_number
            ch['game_in_set'] = game_in_set
            # is_set_start is already correct for the first game;
            # ensure only the first game in each set is marked
            ch['is_set_start'] = (game_in_set == 1)
            all_chapters.append(ch)

    if verbose:
        print(f"\n  {len(all_sets)} sets across {len(cab_configs)} cabs")

    if verbose:
        print(f"\n=== Collecting user data for {len(all_game_ids)} games ===")

    users_dict, game_users_map = collect_users_for_games(all_game_ids, verbose=verbose)

    # Attach user mappings to chapters
    for ch in all_chapters:
        if ch['game_id'] in game_users_map:
            ch['users'] = game_users_map[ch['game_id']]

    if verbose:
        print(f"\n  {len(users_dict)} unique users across both cabs")
        for key, vs in videos.items():
            count = sum(1 for c in all_chapters if c['video_source'] == key)
            print(f"  {vs['label']}: {count} games")

    # Build output
    output = build_output_data(
        video_id=None,  # No default video — use videos dict
        video_start_utc=None,
        chapters=all_chapters,
        users=users_dict,
        videos=videos,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nSaved {len(all_chapters)} chapters to {output_path}")


if __name__ == '__main__':
    import sys

    # Default: SF league night 2026-02-24
    base = Path(__file__).parent
    output = str(base / 'chapters' / 'league_nights' / 'sf-2026-02-24.json')

    # Video alignment: stream time 00:00 in local PST (UTC-8)
    pst = timezone(timedelta(hours=-8))
    cab1_video_start = datetime(2026, 2, 23, 20, 3, 50, tzinfo=pst)
    # Derived from game 1744898 gamestart at 1:06:58 in video
    cab2_video_start = datetime(2026, 2, 23, 19, 13, 32, tzinfo=pst)

    cab_configs = [
        {
            'cache_path': str(base / 'cache' / 'cabinets' / 'sf_2026-02-24.jsonl'),
            'video_source': 'cab1',
            'video_id': 'anblmlk4SNo',
            'label': 'Cab 1',
            'video_start_utc': cab1_video_start.astimezone(timezone.utc),
        },
        {
            'cache_path': str(base / 'cache' / 'cabinets' / 'kqsf0_2026-02-24.jsonl'),
            'video_source': 'cab2',
            'video_id': '-TulQI6RKJA',
            'label': 'Cab 2',
            'video_start_utc': cab2_video_start.astimezone(timezone.utc),
        },
    ]

    generate_multi_cab_chapters(cab_configs, output)
