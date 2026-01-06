"""
KQHiveMind Tournament API client.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests


BASE_URL = "https://kqhivemind.com/api/tournament"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

CACHE_DIR = Path(__file__).parent / "cache" / "tournament"


@dataclass
class TournamentMatch:
    """Tournament match info."""
    id: int
    blue_team: str
    gold_team: str
    blue_score: int
    gold_score: int
    bracket_name: str
    round_num: int
    video_timestamp: float | None = None  # seconds into video


@dataclass
class VideoTimestamp:
    """Match timestamp within a video."""
    match_id: int
    blue_team: str
    gold_team: str
    timestamp_seconds: float


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path."""
    return CACHE_DIR / f"{cache_key}.json"


def _load_from_cache(cache_key: str, verbose: bool = True) -> dict | None:
    """Load from cache if available."""
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None
    if verbose:
        print(f"  Loading from cache: {cache_path}")
    with open(cache_path, 'r') as f:
        return json.load(f)


def _save_to_cache(cache_key: str, data: dict, verbose: bool = True) -> None:
    """Save to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(cache_key)
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"  Saved to cache: {cache_path}")


def fetch_tournament_info(tournament_id: int, use_cache: bool = True, verbose: bool = True) -> dict:
    """Fetch tournament details."""
    cache_key = f"tournament_{tournament_id}_info"
    if use_cache:
        cached = _load_from_cache(cache_key, verbose)
        if cached:
            return cached

    if verbose:
        print(f"Fetching tournament {tournament_id} info...")

    url = f"{BASE_URL}/tournament/{tournament_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if use_cache:
        _save_to_cache(cache_key, data, verbose)

    return data


def fetch_tournament_matches(
    tournament_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> list[TournamentMatch]:
    """Fetch all matches for a tournament."""
    cache_key = f"tournament_{tournament_id}_matches"
    if use_cache:
        cached = _load_from_cache(cache_key, verbose)
        if cached:
            return [TournamentMatch(**m) for m in cached]

    if verbose:
        print(f"Fetching matches for tournament {tournament_id}...")

    matches = []
    page = 1

    while True:
        url = f"{BASE_URL}/match/?tournament_id={tournament_id}&page={page}"
        if verbose:
            print(f"  Fetching page {page}...")

        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get('results', [])
        if not results:
            break

        for item in results:
            # Team names are in nested info objects
            blue_info = item.get('blue_team_info') or {}
            gold_info = item.get('gold_team_info') or {}
            match = TournamentMatch(
                id=item['id'],
                blue_team=blue_info.get('name', ''),
                gold_team=gold_info.get('name', ''),
                blue_score=item.get('blue_score', 0),
                gold_score=item.get('gold_score', 0),
                bracket_name=item.get('stage_name', ''),
                round_num=item.get('round', 0),
            )
            matches.append(match)

        if data.get('next') is None:
            break
        page += 1
        time.sleep(0.5)

    if verbose:
        print(f"  Fetched {len(matches)} matches")

    if use_cache and matches:
        cache_data = [
            {
                'id': m.id,
                'blue_team': m.blue_team,
                'gold_team': m.gold_team,
                'blue_score': m.blue_score,
                'gold_score': m.gold_score,
                'bracket_name': m.bracket_name,
                'round_num': m.round_num,
                'video_timestamp': m.video_timestamp,
            }
            for m in matches
        ]
        _save_to_cache(cache_key, cache_data, verbose)

    return matches


def _parse_timestamp_string(time_str: str) -> float:
    """Parse timestamp string like '0:06:16' or '1:23:45' to seconds."""
    parts = time_str.split(':')
    try:
        if len(parts) == 3:
            hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + mins * 60 + secs
        elif len(parts) == 2:
            mins, secs = int(parts[0]), int(parts[1])
            return mins * 60 + secs
        else:
            return 0
    except ValueError:
        return 0


def fetch_video_timestamps(
    video_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> list[VideoTimestamp]:
    """Fetch match timestamps for a tournament video."""
    cache_key = f"video_{video_id}_timestamps"
    if use_cache:
        cached = _load_from_cache(cache_key, verbose)
        if cached:
            return [VideoTimestamp(**t) for t in cached]

    if verbose:
        print(f"Fetching timestamps for video {video_id}...")

    url = f"{BASE_URL}/video/{video_id}/timestamps/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    timestamps = []
    # Timestamps come as a single string with newline-separated entries
    # Format: "0:06:16 - Round 1: GGs Nuts vs. We don't got em\n..."
    timestamps_str = data.get('timestamps', '')
    if isinstance(timestamps_str, str):
        for line in timestamps_str.strip().split('\n'):
            if ' - ' not in line:
                continue
            time_part, match_part = line.split(' - ', 1)
            total_secs = _parse_timestamp_string(time_part.strip())

            # Parse "Round 1: Team A vs. Team B" or "Semifinals: Team A vs. Team B"
            blue_team = ''
            gold_team = ''
            if ' vs. ' in match_part:
                # Remove round/stage prefix if present
                if ': ' in match_part:
                    match_part = match_part.split(': ', 1)[1]
                teams = match_part.split(' vs. ')
                if len(teams) == 2:
                    blue_team = teams[0].strip()
                    gold_team = teams[1].strip()

            ts = VideoTimestamp(
                match_id=0,  # Not available in this format
                blue_team=blue_team,
                gold_team=gold_team,
                timestamp_seconds=total_secs,
            )
            timestamps.append(ts)

    if verbose:
        print(f"  Found {len(timestamps)} match timestamps")

    if use_cache and timestamps:
        cache_data = [
            {
                'match_id': t.match_id,
                'blue_team': t.blue_team,
                'gold_team': t.gold_team,
                'timestamp_seconds': t.timestamp_seconds,
            }
            for t in timestamps
        ]
        _save_to_cache(cache_key, cache_data, verbose)

    return timestamps


def fetch_match_games(
    match_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> list[dict]:
    """Fetch all games for a tournament match."""
    cache_key = f"match_{match_id}_games"
    if use_cache:
        cached = _load_from_cache(cache_key, verbose)
        if cached:
            return cached

    if verbose:
        print(f"  Fetching games for match {match_id}...")

    url = "https://kqhivemind.com/api/game/game/"
    games = []
    page = 1

    while True:
        params = {
            'tournament_match_id': match_id,
            'page': page,
            'rows_per_page': 50,
        }
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get('results', [])
        if not results:
            break

        for g in results:
            games.append({
                'id': g['id'],
                'map_name': g.get('map_name', ''),
                'start_time': g.get('start_time', ''),
                'end_time': g.get('end_time', ''),
                'winning_team': g.get('winning_team', ''),
                'win_condition': g.get('win_condition', ''),
            })

        if data.get('next') is None:
            break
        page += 1

    if use_cache and games:
        _save_to_cache(cache_key, games, verbose=False)

    return games


def fetch_all_tournament_games(
    tournament_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> list[dict]:
    """Fetch all games for a tournament via its matches."""
    cache_key = f"tournament_{tournament_id}_games"
    if use_cache:
        cached = _load_from_cache(cache_key, verbose)
        if cached:
            return cached

    if verbose:
        print(f"Fetching all games for tournament {tournament_id}...")

    matches = fetch_tournament_matches(tournament_id, use_cache=use_cache, verbose=verbose)

    all_games = []
    for match in matches:
        games = fetch_match_games(match.id, use_cache=use_cache, verbose=verbose)
        for g in games:
            g['match_id'] = match.id
            g['blue_team'] = match.blue_team
            g['gold_team'] = match.gold_team
        all_games.extend(games)

    # Sort by start_time
    all_games.sort(key=lambda g: g.get('start_time', ''))

    if verbose:
        print(f"  Found {len(all_games)} total games")

    if use_cache and all_games:
        _save_to_cache(cache_key, all_games, verbose)

    return all_games


if __name__ == "__main__":
    import sys

    tournament_id = int(sys.argv[1]) if len(sys.argv) > 1 else 842

    # Fetch tournament info
    info = fetch_tournament_info(tournament_id)
    print(f"\nTournament: {info.get('name')}")
    print(f"Scene: {info.get('scene_name')}")
    print(f"Date: {info.get('date')}")

    # Fetch matches
    matches = fetch_tournament_matches(tournament_id)
    print(f"\nMatches ({len(matches)}):")
    for m in matches[:5]:
        print(f"  {m.blue_team} vs {m.gold_team} ({m.blue_score}-{m.gold_score})")
    if len(matches) > 5:
        print(f"  ... and {len(matches) - 5} more")

    # Fetch all games
    games = fetch_all_tournament_games(tournament_id)
    print(f"\nGames ({len(games)}):")
    for g in games[:10]:
        print(f"  {g['id']}: {g['map_name']} - {g['blue_team']} vs {g['gold_team']}")
    if len(games) > 10:
        print(f"  ... and {len(games) - 10} more")
