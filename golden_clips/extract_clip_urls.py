#!/usr/bin/env python3
"""
Extract YouTube clip URLs from text files.

Usage:
    python extract_clip_urls.py <input_file> [--output <output_file>]
"""

import argparse
import re
import sys


def extract_clip_urls(text: str) -> list[str]:
    """Extract all YouTube clip URLs from text."""
    # Match youtube.com/clip/... URLs
    pattern = r'https?://(?:www\.)?youtube\.com/clip/[a-zA-Z0-9_-]+'
    return re.findall(pattern, text)


def main():
    parser = argparse.ArgumentParser(
        description='Extract YouTube clip URLs from text files'
    )
    parser.add_argument('input', help='Input file to extract URLs from')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    urls = extract_clip_urls(text)

    out = open(args.output, 'w') if args.output else sys.stdout
    try:
        for url in urls:
            out.write(url + '\n')
    finally:
        if args.output:
            out.close()

    print(f"Extracted {len(urls)} clip URLs from {args.input}", file=sys.stderr)


if __name__ == '__main__':
    main()
