import pytest

from benchmarking.detectors.base import BaseDetector, DetectedEntity
from benchmarking.detectors.label_mapping import standardize_entities
from benchmarking.detectors.runner import (
    DetectionResult,
    InvalidDetectionError,
    run_detector,
)

TEXT = "Patient John Smith was admitted on 2023-04-12."


class DummyDetector(BaseDetector):
    """Returns a fixed list of entities, whatever the text."""

    def __init__(self, entities, name="dummy"):
        self._entities = entities
        self._name = name
        self.received = []

    @property
    def name(self):
        return self._name

    def detect(self, text):
        self.received.append(text)
        return list(self._entities)


def name_entity(**kw):
    args = dict(text="John Smith", original_label="PERSON", start=8, end=18)
    args.update(kw)
    return DetectedEntity(**args)


MAPPINGS = {"dummy": {"PERSON": "NAME", "DATE_TIME": "DATE"}}


def test_base_detector_is_abstract():
    with pytest.raises(TypeError):
        BaseDetector()


def test_dummy_detector_and_runner_return_entities():
    det = DummyDetector([name_entity(score=0.9, source="dummy")])
    result = run_detector(det, TEXT, label_mappings={})
    assert isinstance(result, DetectionResult)
    assert det.name == "dummy"
    assert det.received == [TEXT]  # detector gets only the text
    assert len(result.entities) == 1
    e = result.entities[0]
    assert (e.text, e.start, e.end, e.score, e.source) == ("John Smith", 8, 18, 0.9, "dummy")


def test_latency_is_non_negative_float():
    result = run_detector(DummyDetector([]), TEXT)
    assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
    assert result.entities == []


def test_latency_measures_detect_call_only():
    import time

    class Slow(DummyDetector):
        def __init__(self):
            time.sleep(0.05)  # "model loading": must not count
            super().__init__([])

        def detect(self, text):
            time.sleep(0.02)
            return []

    det = Slow()
    latency = run_detector(det, TEXT).latency_ms
    assert 15 <= latency < 45


def test_optional_score_defaults_to_none():
    e = name_entity()
    assert e.score is None and e.source is None and e.standardized_label is None
    assert run_detector(DummyDetector([e]), TEXT).entities[0].score is None


def test_valid_offsets_accepted_including_boundaries():
    whole = DetectedEntity(text=TEXT, original_label="X", start=0, end=len(TEXT))
    last = DetectedEntity(text=".", original_label="X", start=len(TEXT) - 1, end=len(TEXT))
    assert len(run_detector(DummyDetector([whole, last]), TEXT).entities) == 2


def test_unicode_offsets_are_code_points():
    text = "Patient José Ñandú was seen."
    e = DetectedEntity(text="José Ñandú", original_label="PERSON", start=8, end=18)
    assert len(run_detector(DummyDetector([e]), text).entities) == 1


@pytest.mark.parametrize("start,end", [(-1, 5), (5, 5), (10, 5), (0, 9999), (len(TEXT), len(TEXT) + 1)])
def test_invalid_offsets_rejected(start, end):
    e = DetectedEntity(text="x", original_label="X", start=start, end=end)
    with pytest.raises(InvalidDetectionError, match="invalid offsets"):
        run_detector(DummyDetector([e]), TEXT)


def test_non_integer_offsets_rejected():
    e = DetectedEntity(text="John", original_label="X", start=8.0, end=12)
    with pytest.raises(InvalidDetectionError, match="must be integers"):
        run_detector(DummyDetector([e]), TEXT)


def test_text_span_mismatch_rejected():
    e = name_entity(text="Jane Smith")
    with pytest.raises(InvalidDetectionError, match="does not match"):
        run_detector(DummyDetector([e]), TEXT)


def test_non_entity_and_non_list_results_rejected():
    with pytest.raises(InvalidDetectionError, match="expected DetectedEntity"):
        run_detector(DummyDetector(["John Smith"]), TEXT)

    class BadReturn(DummyDetector):
        def detect(self, text):
            return None

    with pytest.raises(InvalidDetectionError, match="must return a list"):
        run_detector(BadReturn([]), TEXT)


def test_label_mapping_applied_per_detector():
    s = TEXT.index("2023-04-12")
    date = DetectedEntity(text="2023-04-12", original_label="DATE_TIME", start=s, end=s + 10)
    result = run_detector(DummyDetector([name_entity(), date]), TEXT, label_mappings=MAPPINGS)
    assert [e.standardized_label for e in result.entities] == ["NAME", "DATE"]

    # A different detector name uses its own mapping, not "dummy"'s.
    other = DummyDetector([name_entity()], name="other")
    mappings = {**MAPPINGS, "other": {"PERSON": "NAME"}}
    assert run_detector(other, TEXT, label_mappings=mappings).entities[0].standardized_label == "NAME"
    unmapped = DummyDetector([name_entity()], name="unregistered")
    assert run_detector(unmapped, TEXT, label_mappings=mappings).entities[0].standardized_label is None


def test_unmapped_labels_kept_with_none_and_original_label_preserved():
    s = TEXT.index("admitted")
    entities = [name_entity(), DetectedEntity(text="admitted", original_label="VERB_X", start=s, end=s + 8)]
    result = run_detector(DummyDetector(entities), TEXT, label_mappings=MAPPINGS)
    assert len(result.entities) == 2  # nothing discarded
    mapped, unmapped = result.entities
    assert (mapped.original_label, mapped.standardized_label) == ("PERSON", "NAME")
    assert (unmapped.original_label, unmapped.standardized_label) == ("VERB_X", None)


def test_default_registry_has_no_invented_mappings():
    result = run_detector(DummyDetector([name_entity()], name="presidio"), TEXT)
    assert result.entities[0].standardized_label is None


def test_standardize_does_not_mutate_inputs():
    original = name_entity()
    (out,) = standardize_entities([original], {"PERSON": "NAME"})
    assert original.standardized_label is None and out.standardized_label == "NAME"


def test_mapping_to_unknown_gold_category_rejected():
    with pytest.raises(ValueError, match="unknown ASQ-PHI categories"):
        standardize_entities([name_entity()], {"PERSON": "PERSON_NAME"})


def test_non_str_text_rejected():
    with pytest.raises(TypeError):
        run_detector(DummyDetector([]), None)
