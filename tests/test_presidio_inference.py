import json

import pytest

from benchmarking import run_presidio
from benchmarking.data.loader import DatasetRecord, GoldEntity
from benchmarking.detectors.base import DetectedEntity
from benchmarking.detectors.runner import InvalidDetectionError
from benchmarking.run_presidio import (
    check_output,
    run_inference,
    select_samples,
    summarize,
)

POSITIVE = "Patient John Smith was seen on 2023-04-12 in Boston."
NEGATIVE = "What is the usual dose of metformin?"
EMPTY = "Any advice on managing asthma?"


def entity(text, query, label, score=0.8):
    start = query.index(text)
    return DetectedEntity(text=text, original_label=label, start=start, end=start + len(text),
                          score=score, source="presidio")


class FakePresidio:
    """Returns fixed entities per input text; records which texts it received."""

    name = "presidio"

    def __init__(self, outputs):
        self.outputs = outputs
        self.received = []

    def detect(self, text):
        self.received.append(text)
        return list(self.outputs.get(text, []))


RECORDS = [
    DatasetRecord(query_id=0, text=POSITIVE, has_phi=True,
                  entities=[GoldEntity("John Smith", "NAME", 8, 18, "matched")]),
    DatasetRecord(query_id=1, text=NEGATIVE, has_phi=False, entities=[]),
    DatasetRecord(query_id=2, text=EMPTY, has_phi=False, entities=[]),
]

OUTPUTS = {
    # Deliberately out of offset order, to check output sorting.
    POSITIVE: [
        entity("Boston", POSITIVE, "LOCATION", 0.85),
        entity("John Smith", POSITIVE, "PERSON", 0.85),
        entity("2023-04-12", POSITIVE, "DATE_TIME", 0.6),
    ],
    NEGATIVE: [entity("metformin", NEGATIVE, "NRP", 0.4)],
    EMPTY: [],
}


@pytest.fixture
def rows():
    return run_inference(RECORDS, FakePresidio(OUTPUTS))


def test_detector_receives_only_query_text():
    detector = FakePresidio(OUTPUTS)
    run_inference(RECORDS, detector)
    assert detector.received == [POSITIVE, NEGATIVE, EMPTY]


def test_output_row_structure(rows):
    assert [r["query_id"] for r in rows] == [0, 1, 2]
    for row in rows:
        assert set(row) == {"query_id", "text", "has_phi", "latency_ms", "predictions"}
        assert isinstance(row["latency_ms"], float) and row["latency_ms"] >= 0
        for p in row["predictions"]:
            assert set(p) == {"text", "original_label", "standardized_label", "start", "end", "score", "source"}
            assert row["text"][p["start"]:p["end"]] == p["text"]

    first = rows[0]
    assert (first["text"], first["has_phi"]) == (POSITIVE, True)
    assert "entities" not in first  # no gold annotations in the output
    assert [p["text"] for p in first["predictions"]] == ["John Smith", "2023-04-12", "Boston"]
    assert first["predictions"][0] == {
        "text": "John Smith", "original_label": "PERSON", "standardized_label": "NAME",
        "start": 8, "end": 18, "score": 0.85, "source": "presidio",
    }
    assert rows[1]["predictions"][0]["standardized_label"] is None  # NRP is unmapped


def test_empty_predictions(rows):
    assert rows[2]["predictions"] == []


def test_invalid_prediction_raises_instead_of_being_dropped():
    bad = DetectedEntity(text="Jane", original_label="PERSON", start=8, end=12, source="presidio")
    with pytest.raises(InvalidDetectionError, match="does not match"):
        run_inference(RECORDS[:1], FakePresidio({POSITIVE: [bad]}))

    out_of_bounds = DetectedEntity(text="x", original_label="PERSON", start=0, end=999, source="presidio")
    with pytest.raises(InvalidDetectionError, match="invalid offsets"):
        run_inference(RECORDS[:1], FakePresidio({POSITIVE: [out_of_bounds]}))


def test_check_output(rows):
    check_output(RECORDS, rows)
    with pytest.raises(ValueError, match="Processed 2 rows but the dataset has 3"):
        check_output(RECORDS, rows[:2])
    with pytest.raises(ValueError, match=r"Duplicate query_id\(s\) in output: \[0\]"):
        check_output(RECORDS, [rows[0], rows[0], rows[2]])
    with pytest.raises(ValueError, match="not in dataset order"):
        check_output(RECORDS, [rows[1], rows[0], rows[2]])


def test_summary_counts(rows):
    summary = summarize(rows)
    assert summary["dataset"] == {
        "total_records": 3,
        "phi_positive_records": 1,
        "hard_negative_records": 2,
        "records_with_predictions": 2,
        "records_without_predictions": 1,
    }
    preds = summary["predictions"]
    assert preds["total"] == 4
    assert preds["by_original_label"] == {"DATE_TIME": 1, "LOCATION": 1, "NRP": 1, "PERSON": 1}
    assert preds["by_standardized_label"] == {"DATE": 1, "GEOGRAPHIC_LOCATION": 1, "NAME": 1, "UNMAPPED": 1}
    assert (preds["mapped"], preds["unmapped"]) == (3, 1)
    assert preds["on_hard_negative_records"] == 1


def test_label_counts_are_ordered_by_frequency(rows):
    rows[0]["predictions"].append(dict(rows[0]["predictions"][0]))  # second PERSON
    preds = summarize(rows)["predictions"]
    assert list(preds["by_original_label"]) == ["PERSON", "DATE_TIME", "LOCATION", "NRP"]
    assert list(preds["by_standardized_label"])[0] == "NAME"


def test_latency_summary(rows):
    for row, latency in zip(rows, [10.0, 20.0, 60.0]):
        row["latency_ms"] = latency
    latency = summarize(rows)["latency_ms"]
    assert latency["mean"] == pytest.approx(30.0)
    assert latency["p50"] == pytest.approx(20.0)
    assert latency["p95"] == pytest.approx(56.0)  # 20 + 0.9 * (60 - 20)

    single = summarize(rows[:1])["latency_ms"]
    assert single == {"mean": 10.0, "p50": 10.0, "p95": 10.0}


def test_summary_of_empty_run_raises():
    with pytest.raises(ValueError, match="empty run"):
        summarize([])


def test_samples_grouped_by_original_label(rows):
    samples = select_samples(rows)
    assert list(samples) == ["DATE_TIME", "LOCATION", "NRP", "PERSON"]
    assert samples["PERSON"] == [{
        "query_id": 0, "query": POSITIVE, "detected_text": "John Smith",
        "original_label": "PERSON", "standardized_label": "NAME",
        "start": 8, "end": 18, "score": 0.85,
    }]
    assert samples["NRP"][0]["standardized_label"] is None  # unmapped labels included


def test_samples_skip_repeated_texts_and_spread_scores():
    text = "Smith smith Jones Brown Lee Diaz"
    names = ["Smith", "smith", "Jones", "Brown", "Lee", "Diaz"]
    predictions = []
    for i, name in enumerate(names):
        start = text.index(name) if name != "smith" else 6
        predictions.append({"text": name, "original_label": "PERSON", "standardized_label": "NAME",
                            "start": start, "end": start + len(name), "score": i / 10, "source": "presidio"})
    rows = [{"query_id": 0, "text": text, "has_phi": True, "latency_ms": 1.0, "predictions": predictions}]

    examples = select_samples(rows, per_label=3)["PERSON"]
    # "smith" repeats "Smith"; 5 distinct texts remain, spread from lowest to highest score.
    assert [e["detected_text"] for e in examples] == ["Smith", "Brown", "Diaz"]


def test_main_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_presidio, "load_dataset", lambda: RECORDS)
    monkeypatch.setattr(run_presidio, "PresidioDetector", lambda: FakePresidio(OUTPUTS))

    summary = run_presidio.main(output_dir=tmp_path / "presidio")

    out = tmp_path / "presidio"
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["query_id"] for line in lines] == [0, 1, 2]
    assert json.loads(lines[1])["predictions"][0]["standardized_label"] is None  # null in JSON
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    assert set(json.loads((out / "samples.json").read_text(encoding="utf-8"))) == {
        "DATE_TIME", "LOCATION", "NRP", "PERSON"}
