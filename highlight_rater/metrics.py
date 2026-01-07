"""
Agreement metrics for highlight rating evaluation.

Implements:
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K
- Spearman Correlation
- Cohen's Kappa (inter-rater agreement)
"""

import math
from collections import defaultdict
from typing import Optional

from . import db


def dcg(relevances: list[float], k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain.

    DCG = sum(rel_i / log2(i + 1)) for i in 1..k
    """
    if k is not None:
        relevances = relevances[:k]

    dcg_sum = 0.0
    for i, rel in enumerate(relevances):
        # i is 0-indexed, formula uses 1-indexed positions
        dcg_sum += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg_sum


def ndcg(predicted_relevances: list[float], k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where IDCG is DCG with ideal (sorted) ordering.

    Args:
        predicted_relevances: Relevance scores in predicted order
        k: Only consider top k items (default: all)

    Returns:
        NDCG score between 0 and 1
    """
    # Calculate DCG of predicted order
    predicted_dcg = dcg(predicted_relevances, k)

    # Calculate ideal DCG (sorted descending)
    ideal_relevances = sorted(predicted_relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return predicted_dcg / ideal_dcg


def precision_at_k(predicted_relevances: list[float], threshold: float = 2.0, k: int = 10) -> float:
    """
    Calculate Precision@K.

    What fraction of the top K predicted items are relevant?

    Args:
        predicted_relevances: Relevance scores in predicted order
        threshold: Minimum relevance to be considered "relevant"
        k: Number of top items to consider

    Returns:
        Precision score between 0 and 1
    """
    top_k = predicted_relevances[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(1 for r in top_k if r >= threshold)
    return relevant_count / len(top_k)


def spearman_correlation(rankings1: list[float], rankings2: list[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        rankings1: First set of rankings/scores
        rankings2: Second set of rankings/scores (same length)

    Returns:
        Correlation coefficient between -1 and 1
    """
    if len(rankings1) != len(rankings2):
        raise ValueError("Rankings must have same length")

    n = len(rankings1)
    if n < 2:
        return 0.0

    # Convert to ranks
    def to_ranks(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        ranks = [0] * len(values)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    ranks1 = to_ranks(rankings1)
    ranks2 = to_ranks(rankings2)

    # Calculate Spearman's rho
    d_squared_sum = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

    return rho


def cohens_kappa(ratings1: list[int], ratings2: list[int]) -> float:
    """
    Calculate Cohen's Kappa for inter-rater agreement.

    Args:
        ratings1: Ratings from rater 1
        ratings2: Ratings from rater 2 (same items, same order)

    Returns:
        Kappa coefficient: <0 = worse than chance, 0 = chance, 1 = perfect
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have same length")

    n = len(ratings1)
    if n == 0:
        return 0.0

    # Get unique categories
    categories = sorted(set(ratings1) | set(ratings2))

    # Build confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for r1, r2 in zip(ratings1, ratings2):
        matrix[r1][r2] += 1

    # Calculate observed agreement
    observed = sum(matrix[c][c] for c in categories) / n

    # Calculate expected agreement by chance
    expected = 0.0
    for c in categories:
        p1 = sum(matrix[c][c2] for c2 in categories) / n  # P(rater1 = c)
        p2 = sum(matrix[c1][c] for c1 in categories) / n  # P(rater2 = c)
        expected += p1 * p2

    # Calculate kappa
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0

    kappa = (observed - expected) / (1 - expected)
    return kappa


# ============== Database-backed metric calculations ==============

def calculate_algorithm_agreement(session_id: int) -> dict:
    """
    Calculate agreement between human ratings and algorithm (delta-based) ranking.

    Returns dict with NDCG, Precision@K, and Spearman correlation.
    """
    session = db.get_session(session_id)
    if not session:
        return {'error': 'Session not found'}

    ratings = db.get_ratings_for_session(session_id)
    if not ratings:
        return {'error': 'No ratings in session'}

    # Get algorithm scores for these events
    algo_scores = {
        s['event_id']: s['abs_delta']
        for s in db.get_algorithm_scores(session['chapters_file'])
    }

    # Build parallel lists: algorithm order -> human relevance
    # Sort events by algorithm score (abs_delta) descending
    rated_events = [
        (r['event_id'], algo_scores.get(r['event_id'], 0), r['rating'])
        for r in ratings
        if r['event_id'] in algo_scores
    ]
    rated_events.sort(key=lambda x: x[1], reverse=True)

    if not rated_events:
        return {'error': 'No matching events found'}

    # Extract human ratings in algorithm-predicted order
    human_ratings_in_algo_order = [e[2] for e in rated_events]
    algo_scores_list = [e[1] for e in rated_events]
    human_ratings_list = [e[2] for e in rated_events]

    return {
        'ndcg': ndcg(human_ratings_in_algo_order),
        'ndcg_at_5': ndcg(human_ratings_in_algo_order, k=5),
        'ndcg_at_10': ndcg(human_ratings_in_algo_order, k=10),
        'precision_at_5': precision_at_k(human_ratings_in_algo_order, threshold=2.0, k=5),
        'precision_at_10': precision_at_k(human_ratings_in_algo_order, threshold=2.0, k=10),
        'spearman': spearman_correlation(algo_scores_list, human_ratings_list),
        'num_events': len(rated_events),
        'avg_rating': sum(human_ratings_list) / len(human_ratings_list),
    }


def calculate_inter_rater_agreement(session_id1: int, session_id2: int) -> dict:
    """
    Calculate inter-rater agreement between two sessions rating the same events.

    Returns dict with Cohen's Kappa and other metrics.
    """
    ratings1 = {r['event_id']: r['rating'] for r in db.get_ratings_for_session(session_id1)}
    ratings2 = {r['event_id']: r['rating'] for r in db.get_ratings_for_session(session_id2)}

    # Find common events
    common_events = set(ratings1.keys()) & set(ratings2.keys())

    if not common_events:
        return {'error': 'No common events rated by both raters'}

    # Build parallel lists
    r1_list = [ratings1[e] for e in sorted(common_events)]
    r2_list = [ratings2[e] for e in sorted(common_events)]

    # Calculate agreement
    exact_matches = sum(1 for r1, r2 in zip(r1_list, r2_list) if r1 == r2)
    within_one = sum(1 for r1, r2 in zip(r1_list, r2_list) if abs(r1 - r2) <= 1)

    return {
        'cohens_kappa': cohens_kappa(r1_list, r2_list),
        'exact_agreement': exact_matches / len(common_events),
        'within_one_agreement': within_one / len(common_events),
        'num_common_events': len(common_events),
        'spearman': spearman_correlation(
            [float(r) for r in r1_list],
            [float(r) for r in r2_list]
        ),
    }


def get_rating_distribution(session_id: int) -> dict[int, int]:
    """Get distribution of ratings for a session."""
    ratings = db.get_ratings_for_session(session_id)
    distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in ratings:
        distribution[r['rating']] = distribution.get(r['rating'], 0) + 1
    return distribution
