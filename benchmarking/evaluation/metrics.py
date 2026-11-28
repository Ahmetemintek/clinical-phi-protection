"""Strict exact-span matching and precision/recall/F1.

A span key is (start, end, label). A prediction matches a gold entity only when
all three are equal. Matching is one-to-one: each gold entity can be matched by
at most one prediction, so duplicate predictions of the same key count as FPs.
"""

from __future__ import annotations


def match_exact(gold: list[tuple], predicted: list[tuple]) -> tuple[set[int], set[int]]:
    """Match predictions to gold one-to-one on identical (start, end, label) keys.

    Returns (matched_gold_indices, matched_prediction_indices). Predictions are
    matched in list order, so the result is deterministic.
    """
    unmatched_gold: dict[tuple, list[int]] = {}
    for i, key in enumerate(gold):
        unmatched_gold.setdefault(key, []).append(i)

    matched_gold = set()
    matched_predicted = set()
    for j, key in enumerate(predicted):
        candidates = unmatched_gold.get(key)
        if candidates:
            matched_gold.add(candidates.pop(0))
            matched_predicted.add(j)
    return matched_gold, matched_predicted


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """TP/FP/FN with precision, recall and F1; a metric is None if its denominator is 0."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if half-open spans [a_start, a_end) and [b_start, b_end) share a character."""
    return a_start < b_end and b_start < a_end
