"""Common detector interface.

Conventions: `start` inclusive, `end` exclusive, Python string (code point)
positions in the text passed to `detect`; `text == input_text[start:end]`.
Detectors receive only the query text, never gold labels or offsets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectedEntity:
    text: str
    original_label: str          # label exactly as emitted by the detector
    standardized_label: str | None = None  # ASQ-PHI category; None if unmapped
    start: int = field(kw_only=True)  # required; keyword-only so it can follow defaulted fields
    end: int = field(kw_only=True)
    score: float | None = None
    source: str | None = None


class BaseDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique detector name (used to look up its label mapping)."""

    @abstractmethod
    def detect(self, text: str) -> list[DetectedEntity]:
        """Return entities found in `text`. Model loading belongs in __init__."""
