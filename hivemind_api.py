"""
KQHiveMind API client for fetching game data.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import requests


BASE_URL = "https://kqhivemind.com/api"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Cache directory for storing game events
CACHE_DIR = Path(__file__).parent / "cache" / "game_events"


@dataclass
class GameEvent:
    """Represents a single game event."""
    id: int
    game_id: int
    game_uuid: str
    timestamp: datetime
    event_type: str
    values: list[str]
    win_probability: float | None


def _get_cache_path(game_id: int) -> Path:
    """Get the cache file path for a game."""
    return CACHE_DIR / f"{game_id}.jsonl"


def _load_from_cache(game_id: int, verbose: bool = True) -> list[GameEvent] | None:
    """Load game events from cache if available."""
    cache_path = _get_cache_path(game_id)
    if not cache_path.exists():
        return None

    if verbose:
        print(f"  Loading from cache: {cache_path}")

    events = []
    with open(cache_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            event = GameEvent(
                id=item['id'],
                game_id=item['game_id'],
                game_uuid=item['game_uuid'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                event_type=item['event_type'],
                values=item['values'],
                win_probability=item['win_probability'],
            )
            events.append(event)

    if verbose:
        print(f"  Loaded {len(events)} events from cache")

    return events


def _save_to_cache(game_id: int, events: list[GameEvent], verbose: bool = True) -> None:
    """Save game events to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(game_id)

    with open(cache_path, 'w') as f:
        for event in events:
            item = {
                'id': event.id,
                'game_id': event.game_id,
                'game_uuid': event.game_uuid,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'values': event.values,
                'win_probability': event.win_probability,
            }
            f.write(json.dumps(item) + '\n')

    if verbose:
        print(f"  Saved {len(events)} events to cache: {cache_path}")


def fetch_game_events(
    game_id: int,
    rows_per_page: int = 50,
    delay_between_requests: float = 0.5,
    use_cache: bool = True,
    verbose: bool = True
) -> list[GameEvent]:
    """
    Fetch all game events for a given game ID.

    Args:
        game_id: The HiveMind game ID
        rows_per_page: Number of events per page (default 50)
        delay_between_requests: Seconds to wait between paginated requests
        use_cache: Whether to use cached data if available
        verbose: Print progress

    Returns:
        List of GameEvent objects, sorted by timestamp
    """
    # Check cache first
    if use_cache:
        cached = _load_from_cache(game_id, verbose=verbose)
        if cached is not None:
            return cached

    headers = {
        **HEADERS,
        'Referer': f'https://kqhivemind.com/game/{game_id}',
    }

    events: list[GameEvent] = []
    page = 1
    total_count = None

    while True:
        url = f"{BASE_URL}/game/game-event?page={page}&rows_per_page={rows_per_page}&game_id={game_id}"

        if verbose:
            if total_count:
                print(f"  Fetching page {page} ({len(events)}/{total_count} events)...")
            else:
                print(f"  Fetching page {page}...")

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch game events: {e}")

        if total_count is None:
            total_count = data.get('count', 0)
            if verbose:
                print(f"  Total events: {total_count}")

        results = data.get('results', [])
        if not results:
            break

        for item in results:
            event = GameEvent(
                id=item['id'],
                game_id=item['game'],
                game_uuid=item['game_uuid'],
                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                event_type=item['event_type'],
                values=item.get('values', []),
                win_probability=item.get('win_probability'),
            )
            events.append(event)

        # Check if there are more pages
        if data.get('next') is None:
            break

        page += 1
        time.sleep(delay_between_requests)

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    if verbose:
        print(f"  Fetched {len(events)} events")

    # Save to cache
    if use_cache and events:
        _save_to_cache(game_id, events, verbose=verbose)

    return events


# Additional cache directories
GAME_DETAIL_CACHE_DIR = Path(__file__).parent / "cache" / "game_detail"
USER_CACHE_DIR = Path(__file__).parent / "cache" / "user"


def fetch_game_detail(
    game_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> dict:
    """
    Fetch game detail including user sign-ins.

    Args:
        game_id: The HiveMind game ID
        use_cache: Whether to use cached data if available
        verbose: Print progress

    Returns:
        Dict with game details including 'users' array with position mappings
    """
    cache_path = GAME_DETAIL_CACHE_DIR / f"{game_id}.json"

    # Check cache first
    if use_cache and cache_path.exists():
        if verbose:
            print(f"  Loading game detail from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    url = f"{BASE_URL}/game/game/{game_id}/"

    if verbose:
        print(f"  Fetching game detail for {game_id}...")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch game detail: {e}")

    # Save to cache
    if use_cache:
        GAME_DETAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"  Saved game detail to cache: {cache_path}")

    return data


def fetch_user_public_data(
    user_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> dict:
    """
    Fetch public user data (name, scene, etc).

    Args:
        user_id: The HiveMind user ID
        use_cache: Whether to use cached data if available
        verbose: Print progress

    Returns:
        Dict with user info including 'name' field
    """
    cache_path = USER_CACHE_DIR / f"{user_id}.json"

    # Check cache first
    if use_cache and cache_path.exists():
        if verbose:
            print(f"  Loading user from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    url = f"{BASE_URL}/user/user/{user_id}/public-data/"

    if verbose:
        print(f"  Fetching user {user_id}...")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch user data: {e}")

    # Save to cache
    if use_cache:
        USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"  Saved user to cache: {cache_path}")

    return data


def get_game_duration(events: list[GameEvent]) -> tuple[datetime, datetime, float]:
    """
    Get the start time, end time, and duration of a game from its events.

    Returns:
        Tuple of (start_time, end_time, duration_seconds)
    """
    if not events:
        raise ValueError("No events provided")

    start_time = events[0].timestamp
    end_time = events[-1].timestamp
    duration = (end_time - start_time).total_seconds()

    return start_time, end_time, duration


def summarize_game_events(events: list[GameEvent]) -> dict:
    """
    Create a summary of game events by type.

    Returns:
        Dict with event_type -> count
    """
    summary: dict[str, int] = {}
    for event in events:
        summary[event.event_type] = summary.get(event.event_type, 0) + 1
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hivemind_api.py <game_id>")
        sys.exit(1)

    game_id = int(sys.argv[1])
    print(f"Fetching events for game {game_id}...")

    events = fetch_game_events(game_id)

    if events:
        start, end, duration = get_game_duration(events)
        print(f"\nGame duration: {duration:.1f} seconds")
        print(f"Start: {start}")
        print(f"End: {end}")

        print("\nEvent summary:")
        summary = summarize_game_events(events)
        for event_type, count in sorted(summary.items(), key=lambda x: -x[1]):
            print(f"  {event_type}: {count}")

        print(f"\nFirst 5 events:")
        for event in events[:5]:
            print(f"  {event.timestamp} - {event.event_type}: {event.values}")

        print(f"\nLast 5 events:")
        for event in events[-5:]:
            print(f"  {event.timestamp} - {event.event_type}: {event.values}")
