"""Common detector interface.

A detector is any object with:
  - `name`: a unique string, used to look up its label mapping;
  - `detect(text) -> list[DetectedEntity]`: finds entities in `text`.
Model loading belongs in `__init__`, so it is not counted in inference latency.

Conventions: `start` inclusive, `end` exclusive, Python string (code point)
positions in the text passed to `detect`; `text == input_text[start:end]`.
Detectors receive only the query text, never gold labels or offsets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectedEntity:
    text: str
    original_label: str                    # label exactly as emitted by the detector
    start: int
    end: int
    standardized_label: str | None = None  # ASQ-PHI category; None if unmapped
    score: float | None = None
    source: str | None = None
