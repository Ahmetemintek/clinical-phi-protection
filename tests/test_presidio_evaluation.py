import json

import pytest

from benchmarking import evaluate_presidio
from benchmarking.data.loader import DatasetRecord, GoldEntity
from benchmarking.evaluate_presidio import evaluate, load_predictions
from benchmarking.evaluation.metrics import match_exact, precision_recall_f1


def gold(query, text, label):
    start = query.index(text)
    return GoldEntity(text, label, start, start + len(text), "matched")


def pred(query, text, original_label, standardized_label, score=0.8, start=None):
    if start is None:
        start = query.index(text)
    return {"text": text, "original_label": original_label, "standardized_label": standardized_label,
            "start": start, "end": start + len(text), "score": score, "source": "presidio"}


def make(query_id, query, gold_entities, predictions, has_phi=None):
    if has_phi is None:
        has_phi = bool(gold_entities)
    record = DatasetRecord(query_id=query_id, text=query, has_phi=has_phi, entities=gold_entities)
    row = {"query_id": query_id, "text": query, "has_phi": has_phi, "latency_ms": 1.0,
           "predictions": predictions}
    return record, row


def run(*pairs):
    records = [r for r, _ in pairs]
    rows = [row for _, row in pairs]
    return evaluate(records, rows)


# ---- metrics.py ----

def test_match_exact_is_one_to_one():
    gold_keys = [(0, 4, "NAME"), (10, 14, "DATE")]
    pred_keys = [(0, 4, "NAME"), (0, 4, "NAME"), (10, 14, "NAME"), (10, 15, "DATE")]
    matched_gold, matched_pred = match_exact(gold_keys, pred_keys)
    assert matched_gold == {0} and matched_pred == {0}


def test_precision_recall_f1_and_zero_denominators():
    m = precision_recall_f1(tp=3, fp=1, fn=2)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))

    assert precision_recall_f1(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0,
                                            "precision": None, "recall": None, "f1": None}
    assert precision_recall_f1(0, 0, 2)["precision"] is None   # no predictions
    assert precision_recall_f1(0, 2, 0)["recall"] is None      # no gold
    assert precision_recall_f1(0, 2, 3)["f1"] == 0.0


# ---- evaluate ----

Q = "Patient John Smith seen on 2023-04-12 in Boston."


def test_exact_match_is_true_positive():
    evaluation, fps, fns = run(make(0, Q, [gold(Q, "John Smith", "NAME")],
                                    [pred(Q, "John Smith", "PERSON", "NAME")]))
    assert evaluation["overall"]["tp"] == 1
    assert (evaluation["overall"]["fp"], evaluation["overall"]["fn"]) == (0, 0)
    assert fps == [] and fns == []


def test_wrong_label_and_wrong_boundary_are_fp_and_fn():
    evaluation, fps, fns = run(make(0, Q, [gold(Q, "John Smith", "NAME"), gold(Q, "Boston", "GEOGRAPHIC_LOCATION")],
                                    [pred(Q, "John Smith", "LOCATION", "GEOGRAPHIC_LOCATION"),
                                     pred(Q, "Bosto", "LOCATION", "GEOGRAPHIC_LOCATION")]))
    overall = evaluation["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 2, 2)
    assert evaluation["per_category"]["NAME"]["fn"] == 1
    assert evaluation["per_category"]["GEOGRAPHIC_LOCATION"]["fp"] == 2
    assert evaluation["errors"]["fp_by_gold_relation"] == {
        "overlapping_gold_different_span": 1, "same_span_different_label": 1}
    assert evaluation["errors"]["fn_by_prediction_relation"] == {
        "overlapping_prediction_different_span": 1, "same_span_different_label": 1}

    fp = next(f for f in fps if f["text"] == "Bosto")
    assert fp["query_id"] == 0 and fp["query"] == Q and fp["original_label"] == "LOCATION"
    assert fp["score"] == 0.8 and fp["overlapping_gold"][0]["text"] == "Boston"
    fn = next(f for f in fns if f["text"] == "John Smith")
    assert (fn["label"], fn["start"], fn["end"]) == ("NAME", 8, 18)
    assert fn["overlapping_predictions"][0]["standardized_label"] == "GEOGRAPHIC_LOCATION"


def test_unmatched_gold_is_false_negative():
    evaluation, fps, fns = run(make(0, Q, [gold(Q, "2023-04-12", "DATE")], []))
    assert evaluation["per_category"]["DATE"] == {"tp": 0, "fp": 0, "fn": 1,
                                                  "precision": None, "recall": 0.0, "f1": 0.0}
    assert evaluation["errors"]["fn_by_prediction_relation"] == {"no_overlapping_prediction": 1}
    assert fns[0]["text"] == "2023-04-12" and fps == []


def test_duplicate_predictions_are_kept_and_extra_is_fp():
    p = pred(Q, "John Smith", "PERSON", "NAME")
    evaluation, fps, _ = run(make(0, Q, [gold(Q, "John Smith", "NAME")], [p, dict(p, score=0.5)]))
    assert (evaluation["overall"]["tp"], evaluation["overall"]["fp"]) == (1, 1)
    assert evaluation["errors"]["fp_by_gold_relation"] == {"duplicate_of_matched_prediction": 1}
    assert fps[0]["score"] == 0.5


def test_overlapping_and_conflicting_predictions_are_reported():
    evaluation, _, _ = run(make(0, Q, [gold(Q, "John Smith", "NAME")], [
        pred(Q, "John Smith", "PERSON", "NAME"),
        pred(Q, "John Smith", "UK_NHS", None),
        pred(Q, "Smith", "LOCATION", "GEOGRAPHIC_LOCATION"),
    ]))
    ov = evaluation["overlaps"]
    assert (ov["same_span_groups"], ov["same_span_groups_with_conflicting_labels"]) == (1, 1)
    assert ov["same_span_label_combinations"] == {"PERSON + UK_NHS": {"count": 1, "query_ids": [0]}}
    assert ov["overlapping_pairs_different_spans"] == 2  # "Smith" overlaps both "John Smith" predictions
    assert ov["overlapping_label_combinations"]["LOCATION / PERSON"] == {"count": 1, "query_ids": [0]}
    assert evaluation["overall"]["tp"] == 1 and evaluation["overall"]["fp"] == 1


def test_unmapped_predictions_excluded_from_metrics_but_reported():
    evaluation, fps, fns = run(make(0, Q, [gold(Q, "2023-04-12", "DATE")], [
        pred(Q, "2023-04-12", "US_BANK_NUMBER", None),
        pred(Q, "Patient", "NRP", None),
    ]))
    assert (evaluation["overall"]["tp"], evaluation["overall"]["fp"], evaluation["overall"]["fn"]) == (0, 0, 1)
    assert fps == []
    assert evaluation["counts"]["unmapped_predictions"] == 2
    assert evaluation["counts"]["mapped_predictions"] == 0
    assert evaluation["unmapped_predictions"]["by_original_label"] == {"NRP": 1, "US_BANK_NUMBER": 1}
    assert evaluation["unmapped_predictions"]["exact_gold_span_by_label"] == {"US_BANK_NUMBER -> DATE": 1}
    assert evaluation["errors"]["fn_by_prediction_relation"] == {"same_span_unmapped_only": 1}
    assert fns[0]["overlapping_predictions"][0]["original_label"] == "US_BANK_NUMBER"


def test_hard_negative_predictions():
    neg = "What is the usual dose of metformin in Boston?"
    evaluation, fps, _ = run(make(0, neg, [], [
        pred(neg, "Boston", "LOCATION", "GEOGRAPHIC_LOCATION"),
        pred(neg, "metformin", "NRP", None),
    ]), make(1, "No identifiers here.", [], []))
    h = evaluation["hard_negatives"]
    assert (h["records"], h["records_with_predictions"], h["predictions"]) == (2, 1, 2)
    assert h["mapped_predictions_counted_as_fp"] == 1
    assert h["by_category"] == {"GEOGRAPHIC_LOCATION": 1, "UNMAPPED": 1}
    assert h["by_original_label"] == {"LOCATION": 1, "NRP": 1}
    assert evaluation["overall"]["fp"] == 1
    assert fps[0]["has_phi"] is False and fps[0]["overlapping_gold"] == []
    assert evaluation["errors"]["fp_by_original_label"] == {"LOCATION": 1}


def test_records_with_unresolved_gold_are_excluded():
    unresolved_q = "Seen at UCSF and UCSF again by Dr Lee."
    unresolved = make(1, unresolved_q,
                      [GoldEntity("UCSF", "GEOGRAPHIC_LOCATION", None, None, "ambiguous"),
                       gold(unresolved_q, "Lee", "NAME")],
                      [pred(unresolved_q, "Lee", "PERSON", "NAME"),
                       pred(unresolved_q, "UCSF", "LOCATION", "GEOGRAPHIC_LOCATION")])
    evaluation, fps, fns = run(make(0, Q, [gold(Q, "John Smith", "NAME")], []), unresolved)
    c = evaluation["counts"]
    assert (c["dataset_records"], c["eligible_records"], c["excluded_records"]) == (2, 1, 1)
    assert c["excluded_query_ids"] == [1]
    assert (c["excluded_gold_annotations"], c["excluded_unresolved_gold_annotations"]) == (2, 1)
    assert (c["excluded_predictions"], c["eligible_gold_annotations"], c["eligible_predictions"]) == (2, 1, 0)
    assert (evaluation["overall"]["tp"], evaluation["overall"]["fp"], evaluation["overall"]["fn"]) == (0, 0, 1)
    assert all(e["query_id"] == 0 for e in fps + fns)


def test_all_categories_reported_including_zero_predictions():
    evaluation, _, _ = run(make(0, Q, [gold(Q, "John Smith", "NAME")], [pred(Q, "John Smith", "PERSON", "NAME")]))
    per_category = evaluation["per_category"]
    assert len(per_category) == 13
    assert per_category["NAME"]["f1"] == 1.0
    assert per_category["FAX_NUMBER"] == {"tp": 0, "fp": 0, "fn": 0,
                                          "precision": None, "recall": None, "f1": None}


# ---- loading and outputs ----

def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_load_predictions_validates_against_dataset(tmp_path):
    record, row = make(0, Q, [gold(Q, "John Smith", "NAME")], [pred(Q, "John Smith", "PERSON", "NAME")])
    path = tmp_path / "predictions.jsonl"

    _write_rows(path, [row])
    assert load_predictions(path, [record]) == [row]

    bad_cases = [
        ({**row, "query_id": 5}, "expected query_id 0"),
        ({**row, "text": "other"}, "query text differs"),
        ({**row, "predictions": [pred(Q, "John Smith", "PERSON", "NAME", start=9)]}, "does not match"),
        ({**row, "predictions": [dict(pred(Q, "John Smith", "PERSON", "NAME"), end=999)]}, "invalid offsets"),
        ({**row, "predictions": [pred(Q, "John Smith", "UK_NHS", "NAME")]}, "current Presidio mapping"),
    ]
    for bad_row, message in bad_cases:
        _write_rows(path, [bad_row])
        with pytest.raises(ValueError, match=message):
            load_predictions(path, [record])

    _write_rows(path, [])
    with pytest.raises(ValueError, match="0 rows but the dataset has 1"):
        load_predictions(path, [record])


def test_main_writes_outputs_without_touching_predictions(tmp_path, monkeypatch):
    neg = "What is the dose of metformin in Boston?"
    pairs = [
        make(0, Q, [gold(Q, "John Smith", "NAME"), gold(Q, "2023-04-12", "DATE")],
             [pred(Q, "John Smith", "PERSON", "NAME")]),
        make(1, neg, [], [pred(neg, "Boston", "LOCATION", "GEOGRAPHIC_LOCATION")]),
    ]
    records = [r for r, _ in pairs]
    _write_rows(tmp_path / "predictions.jsonl", [row for _, row in pairs])
    before = (tmp_path / "predictions.jsonl").read_bytes()
    monkeypatch.setattr(evaluate_presidio, "load_dataset", lambda: records)

    evaluation = evaluate_presidio.main(results_dir=tmp_path)

    assert (tmp_path / "predictions.jsonl").read_bytes() == before
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8")) == evaluation
    fps = [json.loads(line) for line in (tmp_path / "false_positives.jsonl").read_text().splitlines()]
    fns = [json.loads(line) for line in (tmp_path / "false_negatives.jsonl").read_text().splitlines()]
    assert [(f["query_id"], f["text"]) for f in fps] == [(1, "Boston")]
    assert [(f["query_id"], f["text"]) for f in fns] == [(0, "2023-04-12")]
    report = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert "## Methodology" in report and "| NAME | 1 | 0 | 0 |" in report
