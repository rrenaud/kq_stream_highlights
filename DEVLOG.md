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

## [004] - 2026-01-14 - Document video event UIs
Created docs/video-ui-guide.md documenting the 3 video UIs:
1. Tournament Viewer (player.html) - full analysis
2. Highlight Rater (highlight_rater_ui.html) - ML training
3. Alignment Tool (tournament_align_ui.html) - timestamp mapping

Includes tradeoffs, when to use each, and data flow diagram.

## [005] - 2026-01-14 - League night to candidates script
Created golden_clips/league_to_candidates.py - ports the auto-highlighter algorithm from player.html to Python.

Algorithm:
1. Collect events with ml_score OR |delta| >= 0.10
2. Compute score: ml_score/4 if available, else delta + clustering bonus
3. Group by set, filter by threshold (0.05 for ML, 0.15 for delta)
4. Take top 4 per set with 10s minimum gap deduplication

Usage: `python -m golden_clips.league_to_candidates chapters/league_nights/sf-2026-01-13.json`
Result: 36 candidates from 320 events (23 playerKill, 7 carryFood, 4 berryDeposit, 2 useMaiden)

## [006] - 2026-01-14 - Simple scorer experiment
Created golden_clips/simple_scorer.py as an interpretable alternative to 79-feature ML model.

Approach: Use ML's top 2 features (win_prob_volatility 27%, kills_per_second 17%) in simple formula.

Results on 29 rated clips:
- Simple: Corr 0.40, MAE 1.06
- ML:     Corr 0.53, MAE 0.66

Conclusion: ML model captures non-linear interactions that simple linear formula misses.
The 79-feature GradientBoosting model is genuinely better, not just overfitting.

## [007] - 2026-01-14 - Lasso feature selection with team symmetry

Added `--analyze-features` flag to train_rater.py to run LassoCV feature selection.

Explored different alpha values to find optimal feature count vs performance trade-off:

| Alpha | Features | MAE   | Corr  |
|-------|----------|-------|-------|
| 0.01  | 42       | 0.899 | 0.408 |
| 0.05  | 26       | 0.901 | 0.411 |
| 0.10  | 13       | 0.887 | 0.421 |
| 0.20  | 8        | 0.822 | 0.483 |
| 0.30  | 2        | 0.898 | 0.362 |

Lasso selected 8 features at alpha=0.20. Added team symmetry enforcement: if a blue feature is selected, also include gold counterpart (and vice versa). This added 4 features:

Symmetric feature set (12 features):
- win_prob_start, win_prob_volatility (team-agnostic)
- victory_in_clip, kills_per_second (team-agnostic)
- state_delta_2/22 (blue/gold vanilla warriors change)
- state_delta_3/23 (blue/gold speed warriors change)
- state_delta_5/25 (blue/gold worker wings change)
- state_delta_18/38 (blue/gold worker food change)

Final comparison (65 samples):
- Full model (79 features): MAE 0.880, Corr 0.401
- Lasso model (8 features): MAE 0.822, Corr 0.483
- Symmetric model (12 features): MAE 0.837, Corr 0.469

Conclusion: Symmetric model is slightly worse than raw Lasso but still much better than full model, and ensures team-fair predictions.

## [008] - 2026-01-14 - Deploy reduced feature model

Trained and saved Ridge model with 12 symmetric features to rating_model.pkl.

Model metadata now includes:
- `reduced_features: True` flag
- `feature_indices: [1, 5, 16, 18, 29, 30, 32, 45, 49, 50, 52, 65]`

Updated scorer.py `_predict()` to check for reduced_features flag and select only the specified feature indices before prediction.

Results: Ridge on 12 features - MAE 0.802, Correlation 0.503 (best so far)

## [009] - 2026-01-14 - Exclude rated clips from candidate generation

Added `--exclude-rated` flag to league_to_candidates.py to filter out candidates that overlap with already-rated clips.

Implementation:
- `load_rated_ranges()`: Extracts (game_id, start, end) tuples from candidate_ratings.jsonl
- `overlaps_rated()`: Checks if candidate's time window intersects any rated clip

Usage: `python -m golden_clips.league_to_candidates chapters/... --output ... --exclude-rated`

This avoids duplicate rating work when regenerating candidates.

## [010] - 2026-01-14 - No logged in users message

Modified player.html to show "No logged in users" message instead of empty player dropdown when chapter data has no user sign-ins.

Changes:
- Desktop sidebar: Replaces dropdown with gray italic text
- Mobile: Hides player selector entirely
- Position selector remains available for filtering by cabinet position
