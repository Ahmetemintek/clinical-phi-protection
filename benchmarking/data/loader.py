"""Loader for the processed ASQ-PHI dataset (data/processed/asq_phi_processed.jsonl).

Gold annotations live only in DatasetRecord/GoldEntity. Detectors must be given
`record.text` alone, never `record.entities`.

Offsets: `start` inclusive, `end` exclusive, Python string (code point) indices
into `DatasetRecord.text`. Unresolved entities (`ambiguous`/`missing`) have
`start = end = None` and are kept, not dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "asq_phi_processed.jsonl"
)

ALIGNMENT_STATUSES = {"matched", "ambiguous", "missing"}


class DatasetFormatError(ValueError):
    """Raised when a record in the processed dataset is malformed."""


@dataclass
class GoldEntity:
    text: str
    label: str
    start: int | None
    end: int | None
    alignment_status: str


@dataclass
class DatasetRecord:
    query_id: int
    text: str
    has_phi: bool
    entities: list[GoldEntity]


def _parse_entity(raw: object, query: str, where: str) -> GoldEntity:
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"{where}: entity must be an object, got {type(raw).__name__}")
    for key in ("text", "label", "start", "end", "alignment_status"):
        if key not in raw:
            raise DatasetFormatError(f"{where}: entity missing required field '{key}'")

    text = raw["text"]
    label = raw["label"]
    start = raw["start"]
    end = raw["end"]
    status = raw["alignment_status"]

    if not isinstance(text, str) or not text:
        raise DatasetFormatError(f"{where}: entity field 'text' must be a non-empty string")
    if not isinstance(label, str) or not label:
        raise DatasetFormatError(f"{where}: entity field 'label' must be a non-empty string")
    if status not in ALIGNMENT_STATUSES:
        raise DatasetFormatError(
            f"{where}: unknown alignment_status {status!r} (expected one of {sorted(ALIGNMENT_STATUSES)})")

    if status == "matched":
        # type(...) is int rejects bools, which isinstance(..., int) would accept.
        if type(start) is not int or type(end) is not int:
            raise DatasetFormatError(f"{where}: matched entity {text!r} needs integer start/end, got {start!r}, {end!r}")
        if not 0 <= start < end <= len(query):
            raise DatasetFormatError(
                f"{where}: matched entity {text!r} has out-of-bounds span [{start}, {end}) for query of length {len(query)}")
        if query[start:end] != text:
            raise DatasetFormatError(
                f"{where}: matched entity span [{start}, {end}) is {query[start:end]!r}, expected {text!r}")
    elif start is not None or end is not None:
        raise DatasetFormatError(f"{where}: {status} entity {text!r} must have null start/end, got {start!r}, {end!r}")

    return GoldEntity(text=text, label=label, start=start, end=end, alignment_status=status)


def _parse_record(raw: object, where: str) -> DatasetRecord:
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"{where}: record must be an object, got {type(raw).__name__}")
    for key in ("query_id", "query", "has_phi", "entities"):
        if key not in raw:
            raise DatasetFormatError(f"{where}: missing required field '{key}'")

    query_id = raw["query_id"]
    query = raw["query"]
    has_phi = raw["has_phi"]

    if type(query_id) is not int:
        raise DatasetFormatError(f"{where}: 'query_id' must be an integer")
    if not isinstance(query, str) or not query:
        raise DatasetFormatError(f"{where}: 'query' must be a non-empty string")
    if not isinstance(has_phi, bool):
        raise DatasetFormatError(f"{where}: 'has_phi' must be a boolean")
    if not isinstance(raw["entities"], list):
        raise DatasetFormatError(f"{where}: 'entities' must be a list")

    where = f"{where} (query_id={query_id})"
    entities = []
    for i, raw_entity in enumerate(raw["entities"]):
        entities.append(_parse_entity(raw_entity, query, f"{where}, entity {i}"))

    if has_phi != (len(entities) > 0):
        raise DatasetFormatError(f"{where}: has_phi={has_phi} inconsistent with {len(entities)} entities")
    return DatasetRecord(query_id=query_id, text=query, has_phi=has_phi, entities=entities)


def load_dataset(path: str | Path | None = None) -> list[DatasetRecord]:
    """Load and validate all records, in file order. Raises DatasetFormatError on bad data."""
    path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    records = []
    seen_ids = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            where = f"{path.name}:{line_no}"
            if not line.strip():
                raise DatasetFormatError(f"{where}: blank line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetFormatError(f"{where}: invalid JSON ({e})") from e

            record = _parse_record(raw, where)
            if record.query_id in seen_ids:
                raise DatasetFormatError(f"{where}: duplicate query_id {record.query_id}")
            seen_ids.add(record.query_id)
            records.append(record)
    return records
