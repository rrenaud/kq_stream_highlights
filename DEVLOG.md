# Development Log

## [001] - 2026-01-14 - Variable-length highlight windows
Approach: Brute-force - try multiple window sizes (6, 8, 10, 12, 15 seconds) for each significant event, keep the highest ML score.

Modified `generate_chapters.py` to loop through window sizes instead of fixed 8s window.

Results show model prefers shorter windows (6s: 121 events, avg 2.41) and 12s windows (115 events, avg 2.08).

## [002] - 2026-01-14 - Highlight-only auto-play mode
Added auto-play mode to player.html that automatically advances through highlights.

Features:
- "Auto HL" button (desktop) / "Auto" button (mobile) / 'A' keyboard shortcut
- Plays each highlight for 6 seconds then advances to next
- Pulsing red button when active
- Auto-stops at end of highlights

Modified files: player.html (CSS, state variables, toggleHighlightMode(), checkHighlightAutoAdvance())

## [003] - 2026-01-14 - Fix clustered highlight deduplication
Problem: Game 1718181 had 7 highlights within 32 seconds, making auto-play feel like continuous watching.

Fix: Added MIN_HIGHLIGHT_GAP (10 seconds) - when selecting top highlights per set, skip events that are within 10 seconds of an already-selected higher-scoring event.

Example: Events at 2067.9s (score 3.74), 2074.8s (1.46), 2085.7s (3.02) now becomes just 2067.9s and 2085.7s.
