#!/usr/bin/env python3
"""
Extract source video URL and timestamps from YouTube clip URLs.

Usage:
    python clip_to_video.py <clip_url>
    python clip_to_video.py --input clips.txt --output results.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict


@dataclass
class ClipInfo:
    """Information extracted from a YouTube clip."""
    clip_url: str
    clip_id: str
    title: str
    source_video_id: str
    source_video_url: str
    start_seconds: float
    end_seconds: float
    duration: float


def extract_clip_info(clip_url: str) -> ClipInfo:
    """Extract source video and timestamps from a YouTube clip URL."""
    # Use subprocess to call yt-dlp to avoid format resolution issues
    result = subprocess.run(
        ['yt-dlp', '--dump-json', '--no-download', clip_url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    info = json.loads(result.stdout)

    # Find video ID in storyboard/thumbnail URLs
    # YouTube embeds the source video ID in URLs like /vi/XXXXX/ or /sb/XXXXX/
    video_id = None

    def search(obj):
        nonlocal video_id
        if video_id:
            return
        if isinstance(obj, str):
            match = re.search(r'/(?:vi|sb)/([a-zA-Z0-9_-]{11})/', obj)
            if match:
                video_id = match.group(1)
        elif isinstance(obj, dict):
            for v in obj.values():
                search(v)
        elif isinstance(obj, list):
            for v in obj:
                search(v)

    search(info)

    start = info.get('section_start', 0)
    end = info.get('section_end', 0)

    return ClipInfo(
        clip_url=clip_url,
        clip_id=info.get('id', ''),
        title=info.get('title', ''),
        source_video_id=video_id or '',
        source_video_url=f'https://www.youtube.com/watch?v={video_id}' if video_id else '',
        start_seconds=start,
        end_seconds=end,
        duration=end - start,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Extract source video URL and timestamps from YouTube clip URLs'
    )
    parser.add_argument('url', nargs='?', help='Single clip URL to process')
    parser.add_argument('--input', '-i', help='Input file with clip URLs (one per line)')
    parser.add_argument('--output', '-o', help='Output JSONL file (default: stdout)')

    args = parser.parse_args()

    if not args.url and not args.input:
        parser.error('Provide either a clip URL or --input file')

    # Collect URLs to process
    urls = []
    if args.url:
        urls.append(args.url)
    if args.input:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)

    # Open output file or use stdout
    out_file = open(args.output, 'w') if args.output else sys.stdout

    try:
        for url in urls:
            try:
                info = extract_clip_info(url)
                out_file.write(json.dumps(asdict(info)) + '\n')
                out_file.flush()
            except Exception as e:
                print(f"Error processing {url}: {e}", file=sys.stderr)
    finally:
        if args.output:
            out_file.close()


if __name__ == '__main__':
    main()
