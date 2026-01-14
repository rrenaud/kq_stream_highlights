"""
Clean API for highlight clip scoring.

This module provides a unified entry point for scoring clip candidates
using the trained ML model.
"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from . import clip_features
from .clip_features import (
    extract_win_prob_features,
    count_events,
    extract_game_state_features,
    load_game_events,
)


DEFAULT_MODEL_PATH = Path(__file__).parent / "rating_model.pkl"


@dataclass
class ScoreResult:
    """Result of scoring a clip."""
    score: Optional[float] = None  # Rating 1-4
    features: dict = field(default_factory=dict)  # Features used for scoring
    error: Optional[str] = None  # Error message if scoring failed

    @property
    def success(self) -> bool:
        """Whether scoring succeeded."""
        return self.score is not None and self.error is None


class HighlightScorer:
    """
    Clean API for highlight clip scoring.

    Usage:
        scorer = HighlightScorer()

        # Score with events in memory
        result = scorer.score_events(events, start_time, end_time)

        # Score by game ID (fetches events automatically)
        result = scorer.score_game_window(game_id, start_utc, end_utc)

        # Batch scoring
        results = scorer.score_batch([(game_id, start, end), ...])
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize scorer.

        Args:
            model_path: Path to model pickle file. Uses default if not provided.
        """
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._model_data = None

    def _load_model(self):
        """Lazy-load and cache the model."""
        if self._model_data is None:
            if not self._model_path.exists():
                return None
            with open(self._model_path, "rb") as f:
                self._model_data = pickle.load(f)
        return self._model_data

    def _predict(self, features: clip_features.ClipFeatures) -> float:
        """Run prediction on features."""
        model_data = self._load_model()
        if model_data is None:
            return 2.5  # Default middle rating

        model = model_data["model"]
        X = features.to_feature_vector().reshape(1, -1)
        prediction = model.predict(X)[0]

        # Clip to valid range [1, 4]
        return max(1.0, min(4.0, float(prediction)))

    def _events_to_dicts(self, events: list, start_time: datetime, end_time: datetime) -> list[dict]:
        """Convert events to dict format, filtering by time window."""
        event_dicts = []
        for evt in events:
            # Handle both Event namedtuples and dicts
            if hasattr(evt, 'timestamp'):
                ts = evt.timestamp
                event_type = evt.event_type
                values = evt.values
                win_prob = evt.win_probability
            else:
                ts_str = evt.get("timestamp", "")
                if not ts_str:
                    continue
                ts = clip_features.parse_timestamp(ts_str) if isinstance(ts_str, str) else ts_str
                event_type = evt.get("event_type", "")
                values = evt.get("values", [])
                win_prob = evt.get("win_probability")

            if start_time <= ts <= end_time:
                event_dicts.append({
                    "event_type": event_type,
                    "values": values,
                    "win_probability": win_prob,
                    "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else ts,
                    "utc_datetime": ts,
                })

        return event_dicts

    def _build_features(
        self,
        event_dicts: list[dict],
        duration: float,
        clip_id: str = "temp"
    ) -> Optional[clip_features.ClipFeatures]:
        """Build ClipFeatures from event dicts."""
        if len(event_dicts) < 2:
            return None

        # Extract features using existing helpers
        wp_features = extract_win_prob_features(event_dicts)
        event_counts = count_events(event_dicts)
        state_features = extract_game_state_features(event_dicts)

        # Calculate density metrics
        events_per_second = len(event_dicts) / duration if duration > 0 else 0
        kills_per_second = event_counts["total_kills"] / duration if duration > 0 else 0

        return clip_features.ClipFeatures(
            clip_id=clip_id,
            duration_seconds=duration,
            win_prob_start=wp_features["win_prob_start"],
            win_prob_end=wp_features["win_prob_end"],
            win_prob_delta=wp_features["win_prob_delta"],
            win_prob_max_swing=wp_features["win_prob_max_swing"],
            win_prob_volatility=wp_features["win_prob_volatility"],
            win_prob_min=wp_features["win_prob_min"],
            win_prob_max=wp_features["win_prob_max"],
            queen_kills=event_counts["queen_kills"],
            worker_kills=event_counts["worker_kills"],
            total_kills=event_counts["total_kills"],
            snail_eats=event_counts["snail_eats"],
            snail_escapes=event_counts["snail_escapes"],
            berry_deposits=event_counts["berry_deposits"],
            berry_kick_ins=event_counts["berry_kick_ins"],
            maiden_uses=event_counts["maiden_uses"],
            victory_in_clip=event_counts["victory_in_clip"],
            events_per_second=events_per_second,
            kills_per_second=kills_per_second,
            state_vector_start=state_features.get("state_vector_start", np.zeros(52)),
            state_vector_end=state_features.get("state_vector_end", np.zeros(52)),
            state_vector_delta=state_features.get("state_vector_delta", np.zeros(52)),
            blue_eggs_start=state_features.get("blue_eggs_start", 0),
            gold_eggs_start=state_features.get("gold_eggs_start", 0),
            blue_eggs_end=state_features.get("blue_eggs_end", 0),
            gold_eggs_end=state_features.get("gold_eggs_end", 0),
            blue_warriors_start=state_features.get("blue_warriors_start", 0),
            gold_warriors_start=state_features.get("gold_warriors_start", 0),
            blue_warriors_end=state_features.get("blue_warriors_end", 0),
            gold_warriors_end=state_features.get("gold_warriors_end", 0),
        )

    def score_events(
        self,
        events: list,
        start_time: datetime,
        end_time: datetime,
    ) -> ScoreResult:
        """
        Score a time window given events in memory.

        Args:
            events: List of Event namedtuples or event dicts
            start_time: Window start (UTC)
            end_time: Window end (UTC)

        Returns:
            ScoreResult with score, features, or error
        """
        try:
            duration = (end_time - start_time).total_seconds()
            event_dicts = self._events_to_dicts(events, start_time, end_time)

            if len(event_dicts) < 2:
                return ScoreResult(error="Not enough events in window")

            features = self._build_features(event_dicts, duration)
            if features is None:
                return ScoreResult(error="Failed to build features")

            score = self._predict(features)

            return ScoreResult(
                score=round(score, 2),
                features={
                    "duration": duration,
                    "event_count": len(event_dicts),
                    "kills_per_second": features.kills_per_second,
                    "win_prob_volatility": features.win_prob_volatility,
                    "queen_kills": features.queen_kills,
                },
            )
        except Exception as e:
            return ScoreResult(error=str(e))

    def score_game_window(
        self,
        game_id: int,
        start_utc: datetime,
        end_utc: datetime,
    ) -> ScoreResult:
        """
        Score a time window, fetching events from cache.

        Args:
            game_id: HiveMind game ID
            start_utc: Window start (UTC)
            end_utc: Window end (UTC)

        Returns:
            ScoreResult with score, features, or error
        """
        try:
            # Load events from cache
            events = load_game_events(game_id)
            if not events:
                return ScoreResult(error=f"No cached events for game {game_id}")

            return self.score_events(events, start_utc, end_utc)
        except Exception as e:
            return ScoreResult(error=str(e))

    def score_around_event(
        self,
        events: list,
        center_time: datetime,
        window_seconds: float = 8.0,
    ) -> ScoreResult:
        """
        Score a window centered around an event time.

        Args:
            events: List of Event namedtuples or event dicts
            center_time: Center of the window (UTC)
            window_seconds: Total window duration

        Returns:
            ScoreResult with score, features, or error
        """
        half = timedelta(seconds=window_seconds / 2)
        return self.score_events(events, center_time - half, center_time + half)

    def score_clip(
        self,
        clip: dict,
        alignments: dict,
        game_details: dict,
    ) -> ScoreResult:
        """
        Score a clip using its video timestamps and alignment data.

        This method handles the complex clip format used by highlight_rater_server.

        Args:
            clip: Clip dict with source_video_id, start_seconds, end_seconds
            alignments: Dict mapping video_id -> alignment data
            game_details: Dict mapping game_id -> game detail

        Returns:
            ScoreResult with score, features, or error
        """
        try:
            features = clip_features.extract_clip_features(clip, alignments, game_details)
            if features is None:
                return ScoreResult(error="Failed to extract clip features")

            score = self._predict(features)
            return ScoreResult(
                score=round(score, 2),
                features={
                    "duration": features.duration_seconds,
                    "kills_per_second": features.kills_per_second,
                    "win_prob_volatility": features.win_prob_volatility,
                    "queen_kills": features.queen_kills,
                },
            )
        except Exception as e:
            return ScoreResult(error=str(e))

    def score_batch(
        self,
        windows: list[tuple[int, datetime, datetime]],
    ) -> list[ScoreResult]:
        """
        Score multiple windows efficiently.

        Args:
            windows: List of (game_id, start_utc, end_utc) tuples

        Returns:
            List of ScoreResults in same order as input
        """
        # Group by game_id to minimize cache loads
        from collections import defaultdict
        game_windows = defaultdict(list)
        for i, (game_id, start, end) in enumerate(windows):
            game_windows[game_id].append((i, start, end))

        results = [None] * len(windows)

        for game_id, items in game_windows.items():
            events = load_game_events(game_id)
            for idx, start, end in items:
                if events:
                    results[idx] = self.score_events(events, start, end)
                else:
                    results[idx] = ScoreResult(error=f"No cached events for game {game_id}")

        return results


# Module-level convenience instance
_default_scorer = None


def get_scorer() -> HighlightScorer:
    """Get or create the default scorer instance."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = HighlightScorer()
    return _default_scorer


def score_events(events: list, start_time: datetime, end_time: datetime) -> ScoreResult:
    """Convenience function using default scorer."""
    return get_scorer().score_events(events, start_time, end_time)


def score_game_window(game_id: int, start_utc: datetime, end_utc: datetime) -> ScoreResult:
    """Convenience function using default scorer."""
    return get_scorer().score_game_window(game_id, start_utc, end_utc)
