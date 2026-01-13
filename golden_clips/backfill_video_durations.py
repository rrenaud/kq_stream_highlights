#!/usr/bin/env python3
"""
Backfill video_duration_seconds for existing tournament alignments.

Usage:
    python -m golden_clips.backfill_video_durations
"""

import json
import subprocess
from pathlib import Path

ALIGNMENTS_FILE = Path(__file__).parent / "tournament_alignments.jsonl"


def get_video_duration(video_id: str) -> float | None:
    """Get video duration in seconds using yt-dlp."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info.get('duration', 0))
    except Exception as e:
        print(f"  Error: {e}")
    return None


def main():
    if not ALIGNMENTS_FILE.exists():
        print("No alignments file found")
        return

    # Load existing alignments
    alignments = []
    with open(ALIGNMENTS_FILE) as f:
        for line in f:
            if line.strip():
                alignments.append(json.loads(line))

    print(f"Found {len(alignments)} alignments")

    updated = 0
    for alignment in alignments:
        video_id = alignment.get("video_id")
        existing_duration = alignment.get("video_duration_seconds")

        if existing_duration:
            print(f"Tournament {alignment['tournament_id']}: already has duration {existing_duration:.0f}s")
            continue

        if not video_id:
            print(f"Tournament {alignment['tournament_id']}: no video_id")
            continue

        print(f"Tournament {alignment['tournament_id']}: fetching duration for {video_id}...", end=" ")
        duration = get_video_duration(video_id)
        if duration:
            alignment["video_duration_seconds"] = duration
            print(f"{duration:.0f}s ({duration/3600:.2f}h)")
            updated += 1
        else:
            print("failed")

    # Save updated alignments
    if updated > 0:
        with open(ALIGNMENTS_FILE, "w") as f:
            for alignment in alignments:
                f.write(json.dumps(alignment) + "\n")
        print(f"\nUpdated {updated} alignments")
    else:
        print("\nNo updates needed")


if __name__ == "__main__":
    main()
