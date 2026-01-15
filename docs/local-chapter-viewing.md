# Local Chapter Viewing Guide

View generated chapter data with the built-in player.

## Quick Start

1. Start a local server from the project root:
   ```bash
   python -m http.server 8000
   ```

2. Open the player in your browser:
   ```
   http://localhost:8000/player.html?chapters=<path-to-chapters-file>
   ```

## Examples

**League night chapters:**
```
http://localhost:8000/player.html?chapters=chapters/league_nights/sf-2026-01-13.json
```

**Tournament chapters:**
```
http://localhost:8000/player.html?chapters=chapters/tournaments/842.json
```

## Chapter File Locations

| Type | Directory | Naming Convention |
|------|-----------|-------------------|
| League nights | `chapters/league_nights/` | `{cabinet}-{date}.json` (e.g., `sf-2026-01-13.json`) |
| Tournaments | `chapters/tournaments/` | `{tournament_id}.json` (e.g., `842.json`) |

## Player Features

- YouTube video playback synced to HiveMind game data
- Chapter navigation sidebar
- Win probability timeline
- Queen kill markers
- Set detection and grouping
- Click anywhere on timeline to seek

## Generating Chapters

**For league nights:**
```bash
# First run saves to pending (requires alignment)
python generate_chapters.py --cabinet sf

# After alignment, re-run to generate chapters
python generate_chapters.py --cabinet sf
```

**For tournaments:**
```bash
python generate_chapters.py <video_path> <cabinet_url> --output chapters/tournaments/842.json
```
