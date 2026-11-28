import pytest
from presidio_analyzer import RecognizerResult

from benchmarking.detectors import presidio
from benchmarking.detectors.base import DetectedEntity
from benchmarking.detectors.label_mapping import DETECTOR_LABEL_MAPPINGS
from benchmarking.detectors.presidio import PresidioDetector
from benchmarking.detectors.runner import run_detector

TEXT = "Patient John Smith was admitted on 2023-04-12."


class FakeAnalyzer:
    """Stands in for AnalyzerEngine so tests don't load the spaCy model."""

    results = []

    def __init__(self):
        self.calls = []

    def analyze(self, text, language):
        self.calls.append((text, language))
        return list(self.results)


@pytest.fixture
def make_detector(monkeypatch):
    def make(results):
        FakeAnalyzer.results = results
        monkeypatch.setattr(presidio, "AnalyzerEngine", FakeAnalyzer)
        return PresidioDetector()
    return make


def test_converts_presidio_result_to_detected_entity(make_detector):
    det = make_detector([RecognizerResult("PERSON", 8, 18, 0.85)])
    entities = det.detect(TEXT)

    assert det.name == "presidio"
    assert det.analyzer.calls == [(TEXT, "en")]
    assert entities == [DetectedEntity(text="John Smith", original_label="PERSON", start=8, end=18,
                                       standardized_label=None, score=0.85, source="presidio")]


def test_text_is_extracted_from_input_using_offsets(make_detector):
    s = TEXT.index("2023-04-12")
    det = make_detector([RecognizerResult("DATE_TIME", s, s + 10, 0.6)])
    (e,) = det.detect(TEXT)
    assert e.text == "2023-04-12" == TEXT[e.start:e.end]


def test_preserves_label_offsets_score_and_source_for_each_result(make_detector):
    results = [RecognizerResult("PERSON", 8, 18, 0.85), RecognizerResult("UK_NHS", 0, 7, 0.3)]
    entities = make_detector(results).detect(TEXT)
    assert [(e.original_label, e.start, e.end, e.score, e.source) for e in entities] == [
        ("PERSON", 8, 18, 0.85, "presidio"),
        ("UK_NHS", 0, 7, 0.3, "presidio"),
    ]


def test_empty_result(make_detector):
    assert make_detector([]).detect(TEXT) == []
    assert run_detector(make_detector([]), TEXT).entities == []


def test_presidio_mapping_contains_only_agreed_labels():
    assert DETECTOR_LABEL_MAPPINGS["presidio"] == {
        "PERSON": "NAME",
        "LOCATION": "GEOGRAPHIC_LOCATION",
        "DATE_TIME": "DATE",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "US_SSN": "SOCIAL_SECURITY_NUMBER",
        "IP_ADDRESS": "IP_ADDRESS",
    }


def test_runner_applies_presidio_label_mapping(make_detector):
    s = TEXT.index("2023-04-12")
    results = [
        RecognizerResult("PERSON", 8, 18, 0.85),
        RecognizerResult("DATE_TIME", s, s + 10, 0.6),
        RecognizerResult("UK_NHS", 0, 7, 0.3),
        RecognizerResult("MEDICAL_LICENSE", 19, 22, 0.3),
    ]
    result = run_detector(make_detector(results), TEXT)  # default DETECTOR_LABEL_MAPPINGS

    assert [(e.original_label, e.standardized_label) for e in result.entities] == [
        ("PERSON", "NAME"),
        ("DATE_TIME", "DATE"),
        ("UK_NHS", None),            # unmapped labels are kept, not guessed
        ("MEDICAL_LICENSE", None),
    ]
    assert all(e.source == "presidio" for e in result.entities)
