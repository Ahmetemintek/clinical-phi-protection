"""Presidio Analyzer detector (default recognizers, English).

Returns Presidio's own entity labels in `original_label`; mapping to ASQ-PHI
categories is done by the runner via DETECTOR_LABEL_MAPPINGS["presidio"].
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine

from .base import DetectedEntity


class PresidioDetector:
    name = "presidio"

    def __init__(self):
        self.analyzer = AnalyzerEngine()

    def detect(self, text: str) -> list[DetectedEntity]:
        results = self.analyzer.analyze(text=text, language="en")
        entities = []
        for result in results:
            entities.append(DetectedEntity(
                text=text[result.start:result.end],
                original_label=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
                source="presidio",
            ))
        return entities
