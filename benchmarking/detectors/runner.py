"""Shared detector runner: timing, span validation and label standardisation.

The runner receives a detector and query text only (no gold annotations).
Latency covers just the `detector.detect(text)` call, measured here with
time.perf_counter(); detector construction/model loading is not included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .base import DetectedEntity
from .label_mapping import DETECTOR_LABEL_MAPPINGS, standardize_entities


class InvalidDetectionError(ValueError):
    """Raised when a detector returns an invalid entity or result."""


@dataclass
class DetectionResult:
    entities: list[DetectedEntity]
    latency_ms: float


def validate_entities(detector_name: str, text: str, entities: object) -> None:
    """Raise InvalidDetectionError unless every entity is a valid span of `text`."""
    if not isinstance(entities, list):
        raise InvalidDetectionError(
            f"{detector_name}: detect() must return a list, got {type(entities).__name__}")
    for i, e in enumerate(entities):
        where = f"{detector_name}: entity {i}"
        if not isinstance(e, DetectedEntity):
            raise InvalidDetectionError(f"{where}: expected DetectedEntity, got {type(e).__name__}")
        if not isinstance(e.original_label, str) or not e.original_label:
            raise InvalidDetectionError(f"{where}: original_label must be a non-empty string")
        # type(...) is int rejects bools and floats, which isinstance(..., int) would not.
        if type(e.start) is not int or type(e.end) is not int:
            raise InvalidDetectionError(f"{where}: start/end must be integers, got {e.start!r}, {e.end!r}")
        if not 0 <= e.start < e.end <= len(text):
            raise InvalidDetectionError(
                f"{where}: invalid offsets [{e.start}, {e.end}) for text of length {len(text)}")
        if text[e.start:e.end] != e.text:
            raise InvalidDetectionError(
                f"{where}: text {e.text!r} does not match input_text[{e.start}:{e.end}] = {text[e.start:e.end]!r}")
        if e.score is not None and (isinstance(e.score, bool) or not isinstance(e.score, (int, float))):
            raise InvalidDetectionError(f"{where}: score must be a number or None, got {e.score!r}")


def run_detector(detector, text: str,
                 label_mappings: dict[str, dict[str, str]] | None = None) -> DetectionResult:
    """Run `detector` on `text`, validate its spans and apply its label mapping.

    `detector` is any object with a `name` attribute and a `detect(text)` method.
    `label_mappings` maps detector name -> {original_label: ASQ-PHI category};
    it defaults to DETECTOR_LABEL_MAPPINGS. A detector with no registered mapping
    gets `standardized_label=None` on every entity.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if label_mappings is None:
        label_mappings = DETECTOR_LABEL_MAPPINGS

    start = time.perf_counter()
    entities = detector.detect(text)
    latency_ms = (time.perf_counter() - start) * 1000.0

    validate_entities(detector.name, text, entities)
    mapping = label_mappings.get(detector.name, {})
    entities = standardize_entities(entities, mapping)
    return DetectionResult(entities=entities, latency_ms=latency_ms)
