#!/usr/bin/env python3
"""Run KQuity predictions on chapter games using cached HiveMind events.

Converts cached events to the format expected by export_predictions,
runs the model, and assembles into the chapter file.

Accepts .json or .json.gz input; always writes .json.gz output.

Usage:
    python run_kquity.py chapters/league_nights/sf-2026-02-24.json
    python run_kquity.py chapters/league_nights/sf-2026-02-24.json.gz
"""

import argparse
import datetime
import gzip
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

# Add KQuity to path
KQUITY_DIR = Path.home() / 'KQuity'
sys.path.insert(0, str(KQUITY_DIR))

from export_predictions import _vectorize_game_wrapper, _predict_and_assemble
import lightgbm as lgb


CACHE_DIR = Path(__file__).parent / 'cache' / 'game_events'
DEFAULT_MODEL = KQUITY_DIR / 'model_experiments' / 'new_data_model' / 'model.mdl'


def load_cached_events(game_id: int) -> list[tuple[datetime.datetime, str, str]] | None:
    """Load cached events and convert to _vectorize_game format.

    Returns list of (datetime, event_type, values_str) where values_str
    is in {val1,val2,...} format expected by KQuity.
    """
    cache_path = CACHE_DIR / f'{game_id}.jsonl'
    if not cache_path.exists():
        return None

    events = []
    with open(cache_path) as f:
        for line in f:
            item = json.loads(line)
            dt = datetime.datetime.fromisoformat(item['timestamp'])
            event_type = item['event_type']
            # Convert ["val1", "val2"] -> "{val1,val2}"
            values = item.get('values', [])
            values_str = '{' + ','.join(str(v) for v in values) + '}'
            events.append((dt, event_type, values_str))

    return events if events else None


def main():
    parser = argparse.ArgumentParser(
        description='Run KQuity predictions on chapter games')
    parser.add_argument('chapters', help='Path to chapter JSON file')
    parser.add_argument('--model', default=str(DEFAULT_MODEL),
                        help='Path to LightGBM model')
    parser.add_argument('--name', default='model',
                        help='Model name for the timeline key')
    parser.add_argument('--no-counterfactuals', dest='counterfactuals',
                        action='store_false', default=True,
                        help='Disable counterfactual computation')
    args = parser.parse_args()

    # Load chapter data (support both .json and .json.gz)
    chapters_path = Path(args.chapters)
    if chapters_path.suffix == '.gz':
        with gzip.open(chapters_path, 'rt') as f:
            chapter_data: dict[str, Any] = json.load(f)
    else:
        with open(chapters_path) as f:
            chapter_data = json.load(f)

    game_ids = [ch['game_id'] for ch in chapter_data['chapters']]
    unique_ids = sorted(set(game_ids))
    print(f"Chapter file has {len(game_ids)} chapters, {len(unique_ids)} unique games")

    # Load model
    print(f"Loading model from {args.model}...")
    model = lgb.Booster(model_file=args.model)

    # Phase 1: Load events and vectorize in parallel (CPU-bound)
    work_items = []
    missing = 0
    for game_id in unique_ids:
        events = load_cached_events(game_id)
        if events is None:
            print(f"  Game {game_id}: no cached events, skipping")
            missing += 1
            continue
        work_items.append((game_id, events, args.counterfactuals))

    n_workers = min(len(work_items), multiprocessing.cpu_count() or 1)
    print(f"Vectorizing {len(work_items)} games with {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        vec_results = pool.map(_vectorize_game_wrapper, work_items)

    # Phase 2: Predict sequentially (LightGBM uses threads internally)
    results: dict[str, list[dict[str, Any]]] = {}
    gold_on_left_map: dict[str, bool] = {}

    for game_id, vec_data in vec_results:
        if vec_data is None:
            print(f"  Game {game_id}: vectorization failed, skipping")
            missing += 1
            continue

        preds, gol = _predict_and_assemble(vec_data, model)
        results[str(game_id)] = preds
        gold_on_left_map[str(game_id)] = gol
        n_cf = sum(1 for p in preds if 'c' in p) if args.counterfactuals else 0
        print(f"  Game {game_id}: {len(preds)} predictions" + (f", {n_cf} with counterfactuals" if n_cf else ""))

    print(f"\nPredictions for {len(results)}/{len(unique_ids)} games"
          f" ({missing} missing/failed)")

    # Now augment the chapter data directly (inline assemble)
    augmented = 0
    for ch in chapter_data['chapters']:
        game_id_str = str(ch['game_id'])
        game_preds = results.get(game_id_str)
        if game_preds is None:
            continue

        # video_offset: start_time = max(0, game_start_seconds - 1), so
        # game-relative time 0 ≈ start_time + 1. When game_start_seconds < 1,
        # start_time is clamped to 0, introducing up to ~1s offset error.
        video_offset: float = ch['start_time'] + 1

        timeline = []
        for pt in game_preds:
            entry: dict[str, Any] = {
                't': round(pt['t'] + video_offset, 2),
                'p': pt['p'],
            }
            for key in ('c', 'sx', 'eg', 'ee', 'bg', 'bc', 'sp', 'sc', 'st'):
                if key in pt:
                    entry[key] = pt[key]
            timeline.append(entry)

        ch['model_timelines'] = {args.name: timeline}
        if game_id_str in gold_on_left_map:
            ch['gold_on_left'] = gold_on_left_map[game_id_str]
        augmented += 1

    print(f"Augmented {augmented}/{len(chapter_data['chapters'])} chapters")

    # Write back as .json.gz
    if chapters_path.suffix == '.gz':
        out_path = chapters_path
    else:
        out_path = chapters_path.with_suffix('.json.gz')
    with gzip.open(out_path, 'wt') as f:
        json.dump(chapter_data, f)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
