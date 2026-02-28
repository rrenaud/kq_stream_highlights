"""
Shared utilities for chapter generation.
"""

from datetime import datetime

from hivemind_api import fetch_game_detail, fetch_user_public_data


def extract_positions_from_event(event) -> list[int]:
    """Extract position IDs involved in an event."""
    values = event.values
    positions = []

    try:
        if event.event_type == 'playerKill':
            if len(values) >= 4:
                positions = [int(values[2]), int(values[3])]
        elif event.event_type == 'snailEat':
            if len(values) >= 4:
                positions = [int(values[2]), int(values[3])]
        elif event.event_type == 'berryDeposit':
            if len(values) >= 3:
                positions = [int(values[2])]
        elif event.event_type == 'useMaiden':
            if len(values) >= 4:
                positions = [int(values[3])]
        elif event.event_type in ('getOnSnail', 'getOffSnail', 'snailEscape'):
            if len(values) >= 3:
                positions = [int(values[2])]
        elif event.event_type == 'carryFood':
            if len(values) >= 1:
                positions = [int(values[0])]
    except (ValueError, IndexError):
        pass

    return positions


def extract_queen_kills(events, reference_utc: datetime) -> list[dict]:
    """Extract queen kill events and convert to timestamps relative to reference_utc."""
    queen_kills = []
    for event in events:
        if event.event_type != 'playerKill':
            continue
        values = event.values
        if len(values) >= 5 and values[4] == 'Queen':
            t = (event.timestamp - reference_utc).total_seconds()
            victim_pos = int(values[3])
            victim_team = 'gold' if victim_pos == 1 else 'blue'
            queen_kills.append({'time': t, 'victim': victim_team})
    return queen_kills


def extract_player_events(events, reference_utc: datetime) -> list[dict]:
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

        delta = curr_prob - prev_prob
        positions = extract_positions_from_event(event)

        if positions and abs(delta) > 0.01:
            t = (event.timestamp - reference_utc).total_seconds()
            player_events.append({
                'id': event.id,
                'time': t,
                'type': event.event_type,
                'positions': positions,
                'delta': round(delta, 4),
                'values': event.values,
            })

        prev_prob = curr_prob

    return player_events


def extract_kill_events(events, reference_utc: datetime) -> list[dict]:
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
                killer = int(values[2])
                victim = int(values[3])
            else:
                killer = int(values[2])
                victim = int(values[3])
        except (ValueError, TypeError):
            continue

        t = (event.timestamp - reference_utc).total_seconds()
        kill_events.append({
            'time': round(t, 2),
            'type': event.event_type,
            'killer': killer,
            'victim': victim,
        })

    return kill_events


def extract_win_prob_timeline(events, reference_utc: datetime) -> list[dict]:
    """Extract win probability timeline for plotting."""
    timeline = []

    for event in events:
        if event.win_probability is None:
            continue
        try:
            prob = float(event.win_probability)
        except (ValueError, TypeError):
            continue

        t = (event.timestamp - reference_utc).total_seconds()
        positions = extract_positions_from_event(event)

        timeline.append({
            't': round(t, 2),
            'p': round(prob, 3),
            'pos': positions if positions else [],
        })

    return timeline


def collect_users_for_games(
    game_ids: list[int],
    verbose: bool = True,
    return_game_details: bool = False
) -> tuple[dict, dict] | tuple[dict, dict, dict]:
    """
    Fetch user data for a list of games.

    Args:
        game_ids: List of HiveMind game IDs
        verbose: Print progress
        return_game_details: Also return full game details dict

    Returns:
        Tuple of (users_dict, game_users_map) or (users_dict, game_users_map, game_details):
        - users_dict: {user_id_str: {name, scene}} for root-level output
        - game_users_map: {game_id: {position: user_id}} for chapter-level
        - game_details: {game_id: detail_dict} (only if return_game_details=True)
    """
    if verbose:
        print(f"  Fetching game details for {len(game_ids)} games...")

    all_user_ids = set()
    game_users_map = {}  # game_id -> {position: user_id}
    game_details = {}  # game_id -> detail dict

    for i, game_id in enumerate(game_ids):
        if verbose:
            print(f"    Game detail {i+1}/{len(game_ids)}: {game_id}")

        detail = fetch_game_detail(game_id, verbose=False)
        game_details[game_id] = detail

        # Build position -> user_id mapping for this game
        game_users = {}
        for user_entry in detail.get('users', []):
            pos = user_entry.get('player_id')
            user_id = user_entry.get('user')
            if pos and user_id:
                game_users[pos] = user_id
                all_user_ids.add(user_id)

        game_users_map[game_id] = game_users

    # Fetch user profiles
    if verbose:
        print(f"  Fetching {len(all_user_ids)} user profiles...")

    users_dict = {}
    for i, user_id in enumerate(sorted(all_user_ids)):
        if verbose:
            print(f"    User {i+1}/{len(all_user_ids)}: {user_id}")
        user_data = fetch_user_public_data(user_id, verbose=False)
        users_dict[str(user_id)] = {
            'name': user_data.get('name', f'User {user_id}'),
            'scene': user_data.get('scene'),
        }

    if return_game_details:
        return users_dict, game_users_map, game_details
    return users_dict, game_users_map


def build_output_data(
    video_id: str | None,
    video_start_utc,
    chapters: list[dict],
    users: dict | None = None,
    videos: dict | None = None,
    **kwargs
) -> dict:
    """
    Build consistent output structure for chapter data.

    Args:
        video_id: YouTube video ID (default/fallback video)
        video_start_utc: Video start time as datetime or ISO string
        chapters: List of chapter dicts
        users: Optional user dict {user_id: {name, scene}}
        videos: Optional multi-video dict {key: {video_id, label}}
        **kwargs: Additional fields to include (e.g., video_path, cabinet_url, fps)

    Returns:
        Dict with standardized output structure
    """
    # Convert datetime to ISO string if needed
    if hasattr(video_start_utc, 'isoformat'):
        video_start_utc = video_start_utc.isoformat()

    output = {
        'video_id': video_id,
        'video_start_utc': video_start_utc,
        'chapters': chapters,
    }

    if videos is not None:
        output['videos'] = videos

    if users is not None:
        output['users'] = users

    # Add any additional fields
    output.update(kwargs)

    return output
