"""
Shared utilities for chapter generation.
"""

from hivemind_api import fetch_game_detail, fetch_user_public_data


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
    **kwargs
) -> dict:
    """
    Build consistent output structure for chapter data.

    Args:
        video_id: YouTube video ID
        video_start_utc: Video start time as datetime or ISO string
        chapters: List of chapter dicts
        users: Optional user dict {user_id: {name, scene}}
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

    if users is not None:
        output['users'] = users

    # Add any additional fields
    output.update(kwargs)

    return output
