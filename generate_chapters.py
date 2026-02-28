#!/usr/bin/env python3
"""
Generate chapter data for a video based on HiveMind game data.

Supports both single-cab (games from live API) and multi-cab
(games from JSONL cache files) workflows. Single-cab is just
multi-cab with one cab config.
"""

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chapter_utils import (
    collect_users_for_games, build_output_data, build_chapters_for_games,
)


def load_games_from_cache(cache_path: Path) -> list[dict]:
    """Load games from a JSONL cache file, handling both flat and nested formats."""
    games = []
    with open(cache_path) as f:
        for line in f:
            item = json.loads(line)
            # Normalize: the kqsf0 cache uses nested format from the API
            if 'cabinet' in item and isinstance(item['cabinet'], dict):
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
                games.append(item)
    return games



def generate_chapters(
    cab_configs: list[dict],
    output_path: str,
    include_users: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate chapter data from one or more cabinet configs.

    Args:
        cab_configs: List of dicts, each with:
            - games: list[dict] — pre-loaded games (from cache or API)
            - ref_utc: datetime — reference time for timestamps (video start)
            - video_source: str | None — key for multi-video (None for single-cab)
            - video_id: str | None — YouTube video ID
            - label: str | None — display label (e.g., "Cab 1")
        output_path: Where to write the output JSON
        include_users: Fetch HiveMind user sign-ins (default True)
        verbose: Print progress

    Returns:
        List of chapter dicts with video timing info
    """
    is_multi = len(cab_configs) > 1 or any(c.get('video_source') for c in cab_configs)

    all_sets = []  # list of (utc_start, [chapters])
    all_game_ids = []
    videos = {}

    for config in cab_configs:
        games = config['games']
        ref_utc = config['ref_utc']
        source_key = config.get('video_source')
        video_id = config.get('video_id')
        label = config.get('label')

        if is_multi and source_key:
            videos[source_key] = {
                'video_id': video_id,
                'label': label,
            }

        # Parse string timestamps if needed
        for g in games:
            if isinstance(g['start_time'], str):
                g['start_time'] = datetime.fromisoformat(g['start_time'])
            if isinstance(g['end_time'], str):
                g['end_time'] = datetime.fromisoformat(g['end_time'])

        if verbose:
            cab_label = label or 'default'
            print(f"\n=== Building chapters for {cab_label} ({len(games)} games) ===")
            if games:
                offset = (games[0]['start_time'] - ref_utc).total_seconds()
                print(f"  Video 00:00 at {ref_utc}")
                print(f"  First game at {games[0]['start_time']} ({offset:+.0f}s into video)")

        sets = build_chapters_for_games(
            games, ref_utc, video_source=source_key, verbose=verbose,
        )

        # Collect game IDs and convert sets to (utc_start, chapters) tuples
        for s in sets:
            if s:
                # Use the first game's start_time as the set's UTC start
                first_game_start = None
                for g in sorted(games, key=lambda g: g['start_time']):
                    if g['id'] == s[0]['game_id']:
                        first_game_start = g['start_time']
                        break
                all_sets.append((first_game_start or ref_utc, s))
            for ch in s:
                all_game_ids.append(ch['game_id'])

    # Sort sets by their start time, then flatten with global set numbering
    all_sets.sort(key=lambda s: s[0])

    all_chapters = []
    for set_number, (_utc, set_chapters) in enumerate(all_sets, start=1):
        for game_in_set, ch in enumerate(set_chapters, start=1):
            ch['set_number'] = set_number
            ch['game_in_set'] = game_in_set
            ch['is_set_start'] = (game_in_set == 1)
            all_chapters.append(ch)

    if verbose:
        print(f"\n  {len(all_sets)} sets, {len(all_chapters)} chapters across {len(cab_configs)} cab(s)")

    # Collect user data
    users_dict = None
    if include_users:
        if verbose:
            print(f"\n=== Collecting user data for {len(all_game_ids)} games ===")
        users_dict, game_users_map, _ = collect_users_for_games(all_game_ids, verbose=verbose)
        for ch in all_chapters:
            if ch['game_id'] in game_users_map:
                ch['users'] = game_users_map[ch['game_id']]
        if verbose:
            print(f"  {len(users_dict)} unique users")

    # Build output
    if is_multi:
        output = build_output_data(
            video_id=None,
            video_start_utc=None,
            chapters=all_chapters,
            users=users_dict,
            videos=videos,
        )
    else:
        config = cab_configs[0]
        output = build_output_data(
            video_id=config.get('video_id'),
            video_start_utc=config['ref_utc'],
            chapters=all_chapters,
            users=users_dict,
        )

    out = Path(output_path)
    if out.suffix != '.gz':
        out = out.with_suffix('.json.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt') as f:
        json.dump(output, f)

    if verbose:
        print(f"\nSaved {len(all_chapters)} chapters to {out}")

    return all_chapters


if __name__ == '__main__':
    base = Path(__file__).parent
    output = str(base / 'chapters' / 'league_nights' / 'sf-2026-02-24.json')

    # Video alignment: stream time 00:00 in local PST (UTC-8)
    pst = timezone(timedelta(hours=-8))
    cab1_video_start = datetime(2026, 2, 23, 20, 3, 50, tzinfo=pst)
    # Derived from game 1744898 gamestart at 1:06:58 in video
    cab2_video_start = datetime(2026, 2, 23, 19, 13, 32, tzinfo=pst)

    cab_configs = [
        {
            'games': load_games_from_cache(base / 'cache' / 'cabinets' / 'sf_2026-02-24.jsonl'),
            'ref_utc': cab1_video_start.astimezone(timezone.utc),
            'video_source': 'cab1',
            'video_id': 'anblmlk4SNo',
            'label': 'Cab 1',
        },
        {
            'games': load_games_from_cache(base / 'cache' / 'cabinets' / 'kqsf0_2026-02-24.jsonl'),
            'ref_utc': cab2_video_start.astimezone(timezone.utc),
            'video_source': 'cab2',
            'video_id': '-TulQI6RKJA',
            'label': 'Cab 2',
        },
    ]

    generate_chapters(cab_configs, output)
