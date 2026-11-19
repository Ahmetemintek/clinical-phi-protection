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
from typing import Iterator, Union

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "asq_phi_processed.jsonl"
)

MATCHED = "matched"
ALIGNMENT_STATUSES = frozenset({MATCHED, "ambiguous", "missing"})


class DatasetFormatError(ValueError):
    """Raised when a record in the processed dataset is malformed."""


@dataclass(frozen=True)
class GoldEntity:
    text: str
    label: str
    start: int | None
    end: int | None
    alignment_status: str


@dataclass(frozen=True)
class DatasetRecord:
    query_id: int
    text: str
    has_phi: bool
    entities: list[GoldEntity]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_entity(raw: object, query: str, where: str) -> GoldEntity:
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"{where}: entity must be an object, got {type(raw).__name__}")
    for key, typ in (("text", str), ("label", str), ("alignment_status", str)):
        if key not in raw:
            raise DatasetFormatError(f"{where}: entity missing required field '{key}'")
        if not isinstance(raw[key], typ):
            raise DatasetFormatError(f"{where}: entity field '{key}' must be {typ.__name__}")
    for key in ("start", "end"):
        if key not in raw:
            raise DatasetFormatError(f"{where}: entity missing required field '{key}'")

    text, label, status = raw["text"], raw["label"], raw["alignment_status"]
    start, end = raw["start"], raw["end"]
    if not text or not label:
        raise DatasetFormatError(f"{where}: entity has empty text or label: {raw!r}")
    if status not in ALIGNMENT_STATUSES:
        raise DatasetFormatError(
            f"{where}: unknown alignment_status {status!r} (expected one of {sorted(ALIGNMENT_STATUSES)})")

    if status == MATCHED:
        if not (_is_int(start) and _is_int(end)):
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
    if not _is_int(raw["query_id"]):
        raise DatasetFormatError(f"{where}: 'query_id' must be an integer")
    if not isinstance(raw["query"], str) or not raw["query"]:
        raise DatasetFormatError(f"{where}: 'query' must be a non-empty string")
    if not isinstance(raw["has_phi"], bool):
        raise DatasetFormatError(f"{where}: 'has_phi' must be a boolean")
    if not isinstance(raw["entities"], list):
        raise DatasetFormatError(f"{where}: 'entities' must be a list")

    where = f"{where} (query_id={raw['query_id']})"
    entities = [_parse_entity(e, raw["query"], f"{where}, entity {i}")
                for i, e in enumerate(raw["entities"])]
    if raw["has_phi"] != bool(entities):
        raise DatasetFormatError(f"{where}: has_phi={raw['has_phi']} inconsistent with {len(entities)} entities")
    return DatasetRecord(query_id=raw["query_id"], text=raw["query"],
                         has_phi=raw["has_phi"], entities=entities)


def iter_dataset(path: Union[str, Path, None] = None) -> Iterator[DatasetRecord]:
    """Yield validated records in file order. Raises DatasetFormatError on bad data."""
    path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    seen_ids: set[int] = set()
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
            yield record


def load_dataset(path: Union[str, Path, None] = None) -> list[DatasetRecord]:
    """Load the whole dataset into a list (file order preserved)."""
    return list(iter_dataset(path))
