"""
Fetch games for a specific cabinet from KQHiveMind.
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


BASE_URL = "https://kqhivemind.com/api"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}

CACHE_DIR = Path(__file__).parent / "cache" / "cabinets"


@dataclass
class Game:
    """Represents a game from the HiveMind API."""
    id: int
    cabinet_id: int
    cabinet_name: str
    scene_name: str
    map_name: str
    start_time: datetime
    end_time: datetime
    win_condition: str
    winning_team: str
    player_count: int


def parse_cabinet_url(url: str) -> tuple[str, str]:
    """
    Parse a cabinet URL to extract scene and cabinet name.

    Args:
        url: URL like https://kqhivemind.com/cabinet/sf/sf

    Returns:
        Tuple of (scene_name, cabinet_name)
    """
    match = re.search(r'/cabinet/([^/]+)/([^/]+)', url)
    if not match:
        raise ValueError(f"Invalid cabinet URL: {url}")
    return match.group(1), match.group(2)


def get_cabinet_id(scene_name: str, cabinet_name: str) -> int:
    """
    Look up the cabinet ID from scene and cabinet names.

    Args:
        scene_name: Scene name (e.g., 'sf')
        cabinet_name: Cabinet name (e.g., 'sf')

    Returns:
        Cabinet ID
    """
    headers = {**HEADERS, 'Referer': f'https://kqhivemind.com/cabinet/{scene_name}/{cabinet_name}'}

    url = f"{BASE_URL}/game/cabinet?rows_per_page=200"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    for cab in data['results']:
        if cab['name'] == cabinet_name:
            return cab['id']

    raise ValueError(f"Cabinet not found: {cabinet_name}")


def fetch_games_for_day(
    cabinet_url: str,
    target_date: datetime | None = None,
    rows_per_page: int = 50,
    delay_between_requests: float = 0.3,
    verbose: bool = True
) -> list[Game]:
    """
    Fetch all games for a specific cabinet on a given day.

    Args:
        cabinet_url: URL like https://kqhivemind.com/cabinet/sf/sf
        target_date: Date to fetch games for (default: most recent day with games)
        rows_per_page: Number of games per API request
        delay_between_requests: Seconds to wait between requests
        verbose: Print progress

    Returns:
        List of Game objects for that day, sorted by start_time
    """
    scene_name, cabinet_name = parse_cabinet_url(cabinet_url)

    if verbose:
        print(f"Fetching games for cabinet: {scene_name}/{cabinet_name}")

    cabinet_id = get_cabinet_id(scene_name, cabinet_name)
    if verbose:
        print(f"  Cabinet ID: {cabinet_id}")

    headers = {**HEADERS, 'Referer': cabinet_url}

    games: list[Game] = []
    page = 1
    target_day_start: datetime | None = None
    target_day_end: datetime | None = None

    while True:
        url = f"{BASE_URL}/game/game/recent/?page={page}&rows_per_page={rows_per_page}&cabinet_id={cabinet_id}"

        if verbose:
            print(f"  Fetching page {page}...")

        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data['results']:
            break

        for item in data['results']:
            start_time = datetime.fromisoformat(item['start_time'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(item['end_time'].replace('Z', '+00:00'))

            # If no target date specified, use the first game's date
            if target_day_start is None:
                if target_date is not None:
                    # Use specified date (ensure UTC timezone)
                    target_day_start = target_date.replace(
                        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                    )
                    target_day_end = target_day_start + timedelta(days=1)
                else:
                    # Use the date of the first (most recent) game
                    target_day_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    target_day_end = target_day_start + timedelta(days=1)

                if verbose:
                    print(f"  Target day: {target_day_start.date()}")

            # Check if game is within target day
            if target_day_start <= start_time < target_day_end:
                game = Game(
                    id=item['id'],
                    cabinet_id=cabinet_id,
                    cabinet_name=cabinet_name,
                    scene_name=scene_name,
                    map_name=item.get('map_name', ''),
                    start_time=start_time,
                    end_time=end_time,
                    win_condition=item.get('win_condition', ''),
                    winning_team=item.get('winning_team', ''),
                    player_count=item.get('player_count', 0),
                )
                games.append(game)
            elif start_time < target_day_start:
                # We've gone past the target day, stop fetching
                if verbose:
                    print(f"  Reached games before target day, stopping")
                break
        else:
            # Continue to next page if we didn't break
            if data.get('next') is None:
                break
            page += 1
            time.sleep(delay_between_requests)
            continue

        # If we broke out of the for loop, break out of while loop too
        break

    # Sort by start time (oldest first)
    games.sort(key=lambda g: g.start_time)

    if verbose:
        print(f"  Found {len(games)} games on {target_day_start.date() if target_day_start else 'N/A'}")

    return games


def save_games_to_cache(games: list[Game], cabinet_name: str, date: datetime) -> Path:
    """Save games to a cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime('%Y-%m-%d')
    cache_path = CACHE_DIR / f"{cabinet_name}_{date_str}.jsonl"

    with open(cache_path, 'w') as f:
        for game in games:
            item = {
                'id': game.id,
                'cabinet_id': game.cabinet_id,
                'cabinet_name': game.cabinet_name,
                'scene_name': game.scene_name,
                'map_name': game.map_name,
                'start_time': game.start_time.isoformat(),
                'end_time': game.end_time.isoformat(),
                'win_condition': game.win_condition,
                'winning_team': game.winning_team,
                'player_count': game.player_count,
            }
            f.write(json.dumps(item) + '\n')

    return cache_path


def fetch_all_events_for_day(
    cabinet_url: str,
    target_date: datetime | None = None,
    verbose: bool = True
) -> dict[int, list]:
    """
    Fetch all game events for all games on a specific day.

    Args:
        cabinet_url: URL like https://kqhivemind.com/cabinet/sf/sf
        target_date: Date to fetch games for (default: most recent day with games)
        verbose: Print progress

    Returns:
        Dict mapping game_id -> list of GameEvent objects
    """
    from hivemind_api import fetch_game_events

    # Get all games for the day
    games = fetch_games_for_day(cabinet_url, target_date, verbose=verbose)

    if not games:
        if verbose:
            print("No games found")
        return {}

    if verbose:
        print(f"\nFetching events for {len(games)} games...")

    all_events: dict[int, list] = {}

    for i, game in enumerate(games):
        if verbose:
            print(f"  [{i+1}/{len(games)}] Game {game.id}...", end=" ", flush=True)

        events = fetch_game_events(game.id, verbose=False)
        all_events[game.id] = events

        if verbose:
            print(f"{len(events)} events")

    if verbose:
        total_events = sum(len(e) for e in all_events.values())
        print(f"\nTotal: {total_events} events across {len(games)} games")

    return all_events


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fetch_games.py <cabinet_url> [YYYY-MM-DD] [--events]")
        print("Example: python fetch_games.py https://kqhivemind.com/cabinet/sf/sf 2025-12-16")
        print("         python fetch_games.py https://kqhivemind.com/cabinet/sf/sf 2025-12-16 --events")
        sys.exit(1)

    cabinet_url = sys.argv[1]
    target_date = None
    fetch_events = "--events" in sys.argv

    # Parse date if provided
    for arg in sys.argv[2:]:
        if arg != "--events":
            target_date = datetime.fromisoformat(arg)
            break

    if fetch_events:
        # Fetch all events for the day
        all_events = fetch_all_events_for_day(cabinet_url, target_date)
    else:
        # Just fetch game list
        games = fetch_games_for_day(cabinet_url, target_date)

        if games:
            print(f"\nGames on {games[0].start_time.date()}:")
            for game in games:
                duration = (game.end_time - game.start_time).total_seconds()
                print(f"  {game.id}: {game.start_time.strftime('%H:%M:%S')} - {game.map_name} - {game.winning_team} {game.win_condition} ({duration:.0f}s)")

            # Save to cache
            cache_path = save_games_to_cache(games, games[0].cabinet_name, games[0].start_time)
            print(f"\nSaved to: {cache_path}")
