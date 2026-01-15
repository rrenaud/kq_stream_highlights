# Video Event UI Guide

## Overview

This codebase has 3 distinct UIs for viewing KQ game footage with event data:

| UI | File | Purpose |
|----|------|---------|
| Tournament Viewer | `player.html` | Full tournament analysis & navigation |
| Highlight Rater | `golden_clips/highlight_rater_ui.html` | Rate clips to train ML model |
| Alignment Tool | `golden_clips/tournament_align_ui.html` | Map video timestamps to game data |

---

## 1. Tournament Viewer (`player.html`)

**Purpose**: Main viewer for analyzing tournament/league night footage

**Features**:
- Chapter-based navigation (sets, games)
- Player/position filtering with K/D stats
- Win probability timeline graphs (clickable)
- Queen kill navigation
- ML-scored highlight navigation
- Auto-play highlight mode
- Full keyboard controls (G/S/E/H/A keys)
- Mobile-optimized layout

**Event Data Shown**:
- Win probability over time (SVG graph)
- Queen kills
- Player events with delta values
- ML scores per event
- K/D per game for selected player

**Tradeoffs**:
- Most comprehensive analysis view
- Great for following individual players
- Rich keyboard navigation
- Requires pre-generated chapters JSON
- Complex UI, steeper learning curve
- No rating/feedback mechanism

---

## 2. Highlight Rater (`golden_clips/highlight_rater_ui.html`)

**Purpose**: Human rating interface to train the ML highlight model

**Features**:
- Two modes: Curated clips vs ML candidates
- 4-point rating scale (Great/Good/Meh/Bad)
- Time adjustment controls for candidates (+/-1s, +/-5s)
- Auto-loop within clip bounds
- Event panels (kills left, other events right)
- Predicted ML score display
- Progress tracking

**Event Data Shown**:
- Kills (queen, warrior) in left panel
- Other events (snail, berry, victory) in right panel
- Events clickable to seek
- Current event highlighted during playback

**Tradeoffs**:
- Purpose-built for training data collection
- Clip adjustment for refining boundaries
- Clear rating workflow
- Only shows one clip at a time
- No cross-game navigation
- Limited to pre-generated candidates

---

## 3. Alignment Tool (`golden_clips/tournament_align_ui.html`)

**Purpose**: One-time setup to map video timestamps to HiveMind game data

**Features**:
- Cabinet selector (for multi-cab tournaments)
- First game info display
- Single "Aligned" button
- Auto-detects cabinet from video title
- League night support

**Event Data Shown**:
- First game metadata only (ID, map, start time)
- Team names if available

**Tradeoffs**:
- Simple, focused workflow
- Enables all downstream features
- Manual process per video
- No event visualization
- Requires finding exact game start moment

---

## When to Use Each

| Task | Use This UI |
|------|-------------|
| Watch & analyze a tournament | Tournament Viewer |
| Follow a specific player's performance | Tournament Viewer |
| Train the ML model with ratings | Highlight Rater |
| Set up a new video for the system | Alignment Tool |
| Find best highlights quickly | Tournament Viewer (Auto mode) |
| Adjust clip boundaries | Highlight Rater |

---

## Data Flow

```
Alignment Tool -> alignment.jsonl
                      |
              generate_chapters.py
                      |
               chapters/*.json
                      |
             Tournament Viewer

generate_candidates.py -> clip_candidates.jsonl
                              |
                       Highlight Rater
                              |
                        ratings.jsonl
                              |
                        train_rater.py
                              |
                       rating_model.pkl
```

---

## File Locations

- `player.html` - Tournament viewer (root)
- `golden_clips/highlight_rater_ui.html` - Rater UI
- `golden_clips/tournament_align_ui.html` - Alignment UI
- `golden_clips/highlight_rater_server.py` - Serves rater data
- `golden_clips/tournament_align_server.py` - Serves alignment data
