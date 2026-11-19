import json
from collections import Counter

import pytest

from benchmarking.data.loader import (
    DatasetFormatError,
    DatasetRecord,
    GoldEntity,
    iter_dataset,
    load_dataset,
)


@pytest.fixture(scope="module")
def records():
    return load_dataset()


def _write(tmp_path, *raw_records):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in raw_records) + "\n",
                    encoding="utf-8")
    return path


def _record(**overrides):
    rec = {
        "query_id": 0,
        "query": "Patient John Smith was admitted.",
        "has_phi": True,
        "entities": [{"text": "John Smith", "label": "NAME", "start": 8, "end": 18,
                      "alignment_status": "matched"}],
    }
    rec.update(overrides)
    return rec


# ---- real dataset ----

def test_loads_expected_counts(records):
    assert len(records) == 1051
    assert sum(r.has_phi for r in records) == 832
    assert sum(not r.has_phi for r in records) == 219
    assert sum(len(r.entities) for r in records) == 2973
    assert all(isinstance(r, DatasetRecord) for r in records)


def test_preserves_order_and_ids(records):
    assert [r.query_id for r in records] == list(range(1051))
    assert [r.query_id for r in iter_dataset()] == list(range(1051))


def test_preserves_first_record(records):
    r = records[0]
    assert r.text.startswith("What is the latest treatment protocol for a 34-year-old female")
    assert r.entities[0] == GoldEntity("Anna S.", "NAME", 86, 93, "matched")
    assert [e.label for e in r.entities] == ["NAME", "GEOGRAPHIC_LOCATION", "DATE"]


def test_hard_negatives_have_no_entities(records):
    negatives = [r for r in records if not r.has_phi]
    assert negatives and all(r.entities == [] for r in negatives)


def test_alignment_statuses_and_labels(records):
    ents = [e for r in records for e in r.entities]
    assert Counter(e.alignment_status for e in ents) == {"matched": 2969, "ambiguous": 3, "missing": 1}
    assert len({e.label for e in ents}) == 13


def test_unresolved_annotations_retained(records):
    unresolved = {(r.query_id, e.text, e.label, e.alignment_status)
                  for r in records for e in r.entities if e.alignment_status != "matched"}
    assert unresolved == {
        (22, "UCSF", "GEOGRAPHIC_LOCATION", "ambiguous"),
        (134, "UCSF", "GEOGRAPHIC_LOCATION", "ambiguous"),
        (568, "UPMC", "GEOGRAPHIC_LOCATION", "ambiguous"),
        (149, "Children's Clinic", "GEOGRAPHIC_LOCATION", "missing"),
    }
    for r in records:
        for e in r.entities:
            if e.alignment_status != "matched":
                assert e.start is None and e.end is None


def test_matched_offsets_valid(records):
    n = 0
    for r in records:
        for e in r.entities:
            if e.alignment_status == "matched":
                assert r.text[e.start:e.end] == e.text
                n += 1
    assert n == 2969


# ---- malformed input ----

def test_valid_synthetic_record_loads(tmp_path):
    (rec,) = load_dataset(_write(tmp_path, _record()))
    assert rec.entities[0].text == "John Smith"


def test_invalid_json(tmp_path):
    with pytest.raises(DatasetFormatError, match=r"data\.jsonl:1: invalid JSON"):
        load_dataset(_write(tmp_path, "{not json"))


def test_blank_line(tmp_path):
    with pytest.raises(DatasetFormatError, match="blank line"):
        load_dataset(_write(tmp_path, _record(), "", _record(query_id=1)))


@pytest.mark.parametrize("field", ["query_id", "query", "has_phi", "entities"])
def test_missing_required_field(tmp_path, field):
    rec = _record()
    del rec[field]
    with pytest.raises(DatasetFormatError, match=f"missing required field '{field}'"):
        load_dataset(_write(tmp_path, rec))


def test_wrong_field_types(tmp_path):
    with pytest.raises(DatasetFormatError, match="'has_phi' must be a boolean"):
        load_dataset(_write(tmp_path, _record(has_phi=1)))
    with pytest.raises(DatasetFormatError, match="'query_id' must be an integer"):
        load_dataset(_write(tmp_path, _record(query_id="0")))


def test_duplicate_query_id(tmp_path):
    with pytest.raises(DatasetFormatError, match="duplicate query_id 0"):
        load_dataset(_write(tmp_path, _record(), _record()))


def test_matched_offset_mismatch(tmp_path):
    bad = _record(entities=[{"text": "John Smith", "label": "NAME", "start": 9, "end": 19,
                             "alignment_status": "matched"}])
    with pytest.raises(DatasetFormatError, match="expected 'John Smith'"):
        load_dataset(_write(tmp_path, bad))


def test_matched_offset_out_of_bounds(tmp_path):
    bad = _record(entities=[{"text": "x", "label": "NAME", "start": 30, "end": 99,
                             "alignment_status": "matched"}])
    with pytest.raises(DatasetFormatError, match="out-of-bounds"):
        load_dataset(_write(tmp_path, bad))


def test_matched_without_offsets(tmp_path):
    bad = _record(entities=[{"text": "John Smith", "label": "NAME", "start": None, "end": None,
                             "alignment_status": "matched"}])
    with pytest.raises(DatasetFormatError, match="needs integer start/end"):
        load_dataset(_write(tmp_path, bad))


def test_unresolved_with_offsets(tmp_path):
    bad = _record(entities=[{"text": "John Smith", "label": "NAME", "start": 8, "end": 18,
                             "alignment_status": "ambiguous"}])
    with pytest.raises(DatasetFormatError, match="must have null start/end"):
        load_dataset(_write(tmp_path, bad))


def test_unknown_alignment_status(tmp_path):
    bad = _record(entities=[{"text": "John Smith", "label": "NAME", "start": None, "end": None,
                             "alignment_status": "guessed"}])
    with pytest.raises(DatasetFormatError, match="unknown alignment_status 'guessed'"):
        load_dataset(_write(tmp_path, bad))


def test_has_phi_inconsistent_with_entities(tmp_path):
    with pytest.raises(DatasetFormatError, match="has_phi=False inconsistent"):
        load_dataset(_write(tmp_path, _record(has_phi=False)))


def test_entity_missing_field(tmp_path):
    bad = _record(entities=[{"text": "John Smith", "start": 8, "end": 18, "alignment_status": "matched"}])
    with pytest.raises(DatasetFormatError, match="missing required field 'label'"):
        load_dataset(_write(tmp_path, bad))
