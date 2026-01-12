#!/usr/bin/env python3
"""
Scrape and cache tournament and match data from kqhivemind.com API.

Usage:
    python hivemind_scraper.py --scene 15  # NYC/Wonderville
    python hivemind_scraper.py --tournament-id 835
    python hivemind_scraper.py --all-scenes
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


CACHE_DIR = Path("cache/hivemind")
BASE_URL = "https://kqhivemind.com/api/tournament"
GAME_API_URL = "https://kqhivemind.com/api/game"


def fetch_json(url: str) -> dict:
    """Fetch JSON from URL with rate limiting."""
    time.sleep(0.1)  # Be nice to the API
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def fetch_all_pages(endpoint: str, params: str = "") -> list:
    """Fetch all pages from a paginated endpoint."""
    results = []
    page = 1
    while True:
        sep = "&" if params else ""
        url = f"{BASE_URL}/{endpoint}/?{params}{sep}page={page}&limit=100"
        data = fetch_json(url)
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return results


def cache_path(category: str, item_id: int) -> Path:
    """Get cache file path for an item."""
    return CACHE_DIR / category / f"{item_id}.json"


def load_cached(category: str, item_id: int) -> dict | None:
    """Load item from cache if it exists."""
    path = cache_path(category, item_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(category: str, item_id: int, data: dict):
    """Save item to cache."""
    path = cache_path(category, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fetch_tournament(tournament_id: int, force: bool = False) -> dict:
    """Fetch and cache a tournament."""
    if not force:
        cached = load_cached("tournament", tournament_id)
        if cached:
            return cached

    url = f"{BASE_URL}/tournament/{tournament_id}/"
    data = fetch_json(url)
    save_cache("tournament", tournament_id, data)
    return data


def fetch_matches_for_tournament(tournament_id: int, force: bool = False) -> list:
    """Fetch and cache all matches for a tournament."""
    cache_file = CACHE_DIR / "matches_by_tournament" / f"{tournament_id}.json"

    if not force and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    matches = fetch_all_pages("match", f"tournament_id={tournament_id}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(matches, f, indent=2)

    return matches


def fetch_tournaments_for_scene(scene_id: int, force: bool = False) -> list:
    """Fetch all tournaments for a scene."""
    cache_file = CACHE_DIR / "tournaments_by_scene" / f"{scene_id}.json"

    if not force and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    all_tournaments = fetch_all_pages("tournament")
    scene_tournaments = [t for t in all_tournaments if t.get("scene") == scene_id]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(scene_tournaments, f, indent=2)

    return scene_tournaments


def fetch_games_for_match(match_id: int, force: bool = False) -> list:
    """Fetch and cache all games for a tournament match."""
    cache_file = CACHE_DIR / "games_by_match" / f"{match_id}.json"

    if not force and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # Use game API with tournament_match_id filter
    games = []
    page = 1
    while True:
        url = f"{GAME_API_URL}/game/?tournament_match_id={match_id}&page={page}&limit=100"
        data = fetch_json(url)
        games.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(games, f, indent=2)

    return games


def fetch_all_scenes() -> list:
    """Fetch all scenes."""
    cache_file = CACHE_DIR / "scenes.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    scenes = fetch_all_pages("scene")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(scenes, f, indent=2)

    return scenes


def scrape_scene(scene_id: int, force: bool = False, include_games: bool = False):
    """Scrape all tournaments and matches for a scene."""
    print(f"Fetching tournaments for scene {scene_id}...")
    tournaments = fetch_tournaments_for_scene(scene_id, force)
    print(f"Found {len(tournaments)} tournaments")

    total_matches = 0
    total_games = 0

    for t in tournaments:
        tid = t["id"]
        name = t.get("name", "Unknown")
        print(f"  Tournament {tid}: {name}...")
        matches = fetch_matches_for_tournament(tid, force)
        print(f"    {len(matches)} matches")
        total_matches += len(matches)

        if include_games:
            match_games = 0
            for m in matches:
                mid = m["id"]
                games = fetch_games_for_match(mid, force)
                match_games += len(games)
                total_games += len(games)
            print(f"    {match_games} games")

    print(f"\nTotal: {len(tournaments)} tournaments, {total_matches} matches", end="")
    if include_games:
        print(f", {total_games} games")
    else:
        print()


def scrape_tournament(tournament_id: int, force: bool = False):
    """Scrape a single tournament and its matches."""
    print(f"Fetching tournament {tournament_id}...")
    tournament = fetch_tournament(tournament_id, force)
    print(f"  Name: {tournament.get('name', 'Unknown')}")

    print(f"Fetching matches...")
    matches = fetch_matches_for_tournament(tournament_id, force)
    print(f"  Found {len(matches)} matches")

    return tournament, matches


def scrape_game_details_and_events(scene_id: int, days: int | None = None, year: int | None = None, force: bool = False):
    """Scrape game details and events for tournament games in a scene.

    Args:
        scene_id: Scene ID (e.g., 15 for NYC/Wonderville)
        days: Filter to tournaments in last N days
        year: Filter to tournaments in specific year (e.g., 2024)
        force: Force refresh cached data
    """
    from datetime import datetime, timedelta
    from hivemind_api import fetch_game_detail, fetch_game_events

    # Load tournaments for this scene
    tournaments_file = CACHE_DIR / "tournaments_by_scene" / f"{scene_id}.json"
    if not tournaments_file.exists():
        print("No cached tournaments found. Run with --scene first.")
        return

    with open(tournaments_file) as f:
        tournaments = json.load(f)

    # Filter tournaments by date
    if year:
        print(f"Fetching game details/events for scene {scene_id} tournaments in {year}...")
        filtered_tournaments = [
            t for t in tournaments
            if t.get("date", "").startswith(str(year))
        ]
        print(f"Found {len(filtered_tournaments)} tournaments in {year}")
    elif days:
        cutoff = datetime.now() - timedelta(days=days)
        print(f"Fetching game details/events for scene {scene_id} tournaments since {cutoff.date()}...")
        filtered_tournaments = []
        for t in tournaments:
            date_str = t.get("date", "")
            if date_str:
                try:
                    t_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if t_date >= cutoff:
                        filtered_tournaments.append(t)
                except ValueError:
                    pass
        print(f"Found {len(filtered_tournaments)} tournaments in last {days} days")
    else:
        print("Fetching game details/events for all tournaments...")
        filtered_tournaments = tournaments
        print(f"Found {len(filtered_tournaments)} tournaments")

    # Process tournaments, fetching matches, games, and details incrementally
    total_games = 0
    for ti, t in enumerate(filtered_tournaments):
        tid = t["id"]
        tname = t.get("name", "Unknown")
        tdate = t.get("date", "")
        print(f"\n[{ti+1}/{len(filtered_tournaments)}] {tname} ({tdate})")

        # Fetch matches if not cached
        matches = fetch_matches_for_tournament(tid, force=force)
        print(f"  {len(matches)} matches")

        for m in matches:
            mid = m["id"]
            # Fetch games if not cached
            games = fetch_games_for_match(mid, force=force)

            for g in games:
                game_id = g["id"]
                total_games += 1
                print(f"    Game {game_id}...", end=" ", flush=True)

                try:
                    # Use cache by default, skip cache only when --force is specified
                    use_cache = not force

                    # Check if already cached (use same paths as hivemind_api.py)
                    base_dir = Path(__file__).parent
                    detail_cache_path = base_dir / "cache" / "game_detail" / f"{game_id}.json"
                    events_cache_path = base_dir / "cache" / "game_events" / f"{game_id}.jsonl"
                    already_cached = detail_cache_path.exists() and events_cache_path.exists()

                    if already_cached and use_cache:
                        # Skip fetching, just load to get event count
                        events = fetch_game_events(game_id, use_cache=True, verbose=False)
                        print(f"{len(events)} events (cached)")
                    else:
                        fetch_game_detail(game_id, use_cache=use_cache, verbose=False)
                        events = fetch_game_events(game_id, use_cache=use_cache, verbose=False)
                        print(f"{len(events)} events (fetched)")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\nDone! Processed {total_games} games from {len(filtered_tournaments)} tournaments.")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and cache tournament/match data from kqhivemind.com"
    )
    parser.add_argument("--scene", type=int, help="Scene ID to scrape (e.g., 15 for NYC)")
    parser.add_argument("--tournament-id", type=int, help="Single tournament ID to scrape")
    parser.add_argument("--list-scenes", action="store_true", help="List all scenes")
    parser.add_argument("--games", action="store_true", help="Also scrape games for each match")
    parser.add_argument("--game-details", action="store_true", help="Scrape game details and events")
    parser.add_argument("--days", type=int, help="Days of history for --game-details")
    parser.add_argument("--year", type=int, help="Year to filter for --game-details (e.g., 2024)")
    parser.add_argument("--force", action="store_true", help="Force refresh cached data")

    args = parser.parse_args()

    if args.list_scenes:
        scenes = fetch_all_scenes()
        for s in scenes:
            print(f"{s['id']}: {s['name']}")
        return

    if args.tournament_id:
        scrape_tournament(args.tournament_id, args.force)
    elif args.game_details and args.scene:
        scrape_game_details_and_events(args.scene, days=args.days, year=args.year, force=args.force)
    elif args.scene:
        scrape_scene(args.scene, args.force, include_games=args.games)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
