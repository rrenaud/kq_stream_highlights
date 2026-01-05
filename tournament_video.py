"""
Fetch and cache tournament video data from KQHiveMind.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests


BASE_URL = "https://kqhivemind.com/api/tournament"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Cache directory for storing tournament videos
CACHE_DIR = Path(__file__).parent / "cache" / "tournament_videos"


def _parse_duration(duration_str: str) -> timedelta:
    """Parse duration string to timedelta. Handles HH:MM:SS or space-separated formats."""
    # Replace spaces with colons for consistency
    duration_str = duration_str.replace(' ', ':')
    parts = duration_str.split(':')
    try:
        if len(parts) >= 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        elif len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return timedelta(minutes=minutes, seconds=seconds)
        else:
            return timedelta(seconds=int(parts[0]))
    except ValueError:
        return timedelta(seconds=0)


@dataclass
class TournamentVideo:
    """Represents a tournament video recording."""
    id: int
    cabinet_name: str
    service: str  # "youtube"
    video_id: str  # YouTube video ID
    start_time: datetime
    length: timedelta
    tournament_id: int | None
    cabinet_id: int


def _get_cache_path(tournament_id: int | None) -> Path:
    """Get the cache file path for a tournament's videos."""
    key = f"tournament_{tournament_id}" if tournament_id else "all_videos"
    return CACHE_DIR / f"{key}.jsonl"


def _load_from_cache(tournament_id: int | None, verbose: bool = True) -> list[TournamentVideo] | None:
    """Load videos from cache if available."""
    cache_path = _get_cache_path(tournament_id)
    if not cache_path.exists():
        return None

    if verbose:
        print(f"  Loading from cache: {cache_path}")

    videos = []
    with open(cache_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            video = TournamentVideo(
                id=item['id'],
                cabinet_name=item['cabinet_name'],
                service=item['service'],
                video_id=item['video_id'],
                start_time=datetime.fromisoformat(item['start_time']),
                length=timedelta(seconds=item['length_seconds']),
                tournament_id=item['tournament_id'],
                cabinet_id=item['cabinet_id'],
            )
            videos.append(video)

    if verbose:
        print(f"  Loaded {len(videos)} videos from cache")

    return videos


def _save_to_cache(tournament_id: int | None, videos: list[TournamentVideo], verbose: bool = True) -> None:
    """Save videos to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(tournament_id)

    with open(cache_path, 'w') as f:
        for video in videos:
            item = {
                'id': video.id,
                'cabinet_name': video.cabinet_name,
                'service': video.service,
                'video_id': video.video_id,
                'start_time': video.start_time.isoformat(),
                'length_seconds': video.length.total_seconds(),
                'tournament_id': video.tournament_id,
                'cabinet_id': video.cabinet_id,
            }
            f.write(json.dumps(item) + '\n')

    if verbose:
        print(f"  Saved {len(videos)} videos to cache: {cache_path}")


def _fetch_videos_paginated(
    tournament_id: int | None = None,
    delay_between_requests: float = 0.5,
    verbose: bool = True
) -> list[TournamentVideo]:
    """Fetch videos with pagination."""
    videos: list[TournamentVideo] = []
    page = 1
    total_count = None

    while True:
        url = f"{BASE_URL}/video/?page={page}"
        if tournament_id is not None:
            url += f"&tournament_id={tournament_id}"

        if verbose:
            if total_count:
                print(f"  Fetching page {page} ({len(videos)}/{total_count} videos)...")
            else:
                print(f"  Fetching page {page}...")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch tournament videos: {e}")

        if total_count is None:
            total_count = data.get('count', 0)
            if verbose:
                print(f"  Total videos: {total_count}")

        results = data.get('results', [])
        if not results:
            break

        for item in results:
            video = TournamentVideo(
                id=item['id'],
                cabinet_name=item.get('cabinet_name', ''),
                service=item.get('service', 'youtube'),
                video_id=item['video_id'],
                start_time=datetime.fromisoformat(item['start_time'].replace('Z', '+00:00')),
                length=_parse_duration(item['length']),
                tournament_id=item.get('tournament'),
                cabinet_id=item['cabinet'],
            )
            videos.append(video)

        # Check if there are more pages
        if data.get('next') is None:
            break

        page += 1
        time.sleep(delay_between_requests)

    if verbose:
        print(f"  Fetched {len(videos)} videos")

    return videos


def fetch_tournament_videos(
    tournament_id: int,
    use_cache: bool = True,
    verbose: bool = True
) -> list[TournamentVideo]:
    """
    Fetch all videos for a specific tournament.

    Args:
        tournament_id: The tournament ID
        use_cache: Whether to use cached data if available
        verbose: Print progress

    Returns:
        List of TournamentVideo objects
    """
    if use_cache:
        cached = _load_from_cache(tournament_id, verbose=verbose)
        if cached is not None:
            return cached

    if verbose:
        print(f"Fetching videos for tournament {tournament_id}...")

    videos = _fetch_videos_paginated(tournament_id, verbose=verbose)

    if use_cache and videos:
        _save_to_cache(tournament_id, videos, verbose=verbose)

    return videos


def fetch_all_videos(
    use_cache: bool = True,
    verbose: bool = True
) -> list[TournamentVideo]:
    """
    Fetch all videos in the system.

    Args:
        use_cache: Whether to use cached data if available
        verbose: Print progress

    Returns:
        List of TournamentVideo objects
    """
    if use_cache:
        cached = _load_from_cache(None, verbose=verbose)
        if cached is not None:
            return cached

    if verbose:
        print("Fetching all tournament videos...")

    videos = _fetch_videos_paginated(tournament_id=None, verbose=verbose)

    if use_cache and videos:
        _save_to_cache(None, videos, verbose=verbose)

    return videos


def get_youtube_url(video: TournamentVideo, timestamp_seconds: float = 0) -> str:
    """Get YouTube URL for a video, optionally at a specific timestamp."""
    base_url = f"https://www.youtube.com/watch?v={video.video_id}"
    if timestamp_seconds > 0:
        return f"{base_url}&t={int(timestamp_seconds)}s"
    return base_url


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        tournament_id = int(sys.argv[1])
        print(f"Fetching videos for tournament {tournament_id}...")
        videos = fetch_tournament_videos(tournament_id)
    else:
        print("Fetching all videos...")
        videos = fetch_all_videos()

    if videos:
        print(f"\nFound {len(videos)} videos:")
        for video in videos[:10]:
            print(f"  [{video.id}] {video.cabinet_name}")
            print(f"      YouTube: {video.video_id}")
            print(f"      Start: {video.start_time}")
            print(f"      Length: {video.length}")
            print(f"      Tournament: {video.tournament_id}")
            print()
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")

        # Group by tournament
        by_tournament: dict[int | None, list[TournamentVideo]] = {}
        for v in videos:
            by_tournament.setdefault(v.tournament_id, []).append(v)

        print(f"\nTournaments with videos:")
        for tid, vids in sorted(by_tournament.items(), key=lambda x: (x[0] is None, x[0])):
            print(f"  Tournament {tid}: {len(vids)} videos")
