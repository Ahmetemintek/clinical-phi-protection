"""Stage 2C: run Presidio over the full ASQ-PHI dataset and save its predictions.

Usage (from the project root):
    python -m benchmarking.run_presidio

Writes to results/presidio/:
  - predictions.jsonl: one line per query, in dataset order (no gold annotations)
  - summary.json:      prediction counts and latency (no gold-based metrics)
  - samples.json:      up to 5 example predictions per original Presidio label

Each prediction's offsets and text are validated by `run_detector`, which raises
InvalidDetectionError instead of dropping invalid predictions.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarking.data.loader import DatasetRecord, load_dataset
from benchmarking.detectors.presidio import PresidioDetector
from benchmarking.detectors.runner import run_detector

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "presidio"
SAMPLES_PER_LABEL = 5
UNMAPPED = "UNMAPPED"


def run_inference(records: list[DatasetRecord], detector) -> list[dict]:
    """Run `detector` on each record's text and return one output row per record."""
    rows = []
    for record in records:
        result = run_detector(detector, record.text)  # text only, never record.entities
        entities = sorted(result.entities, key=lambda e: (e.start, e.end, e.original_label))
        predictions = []
        for e in entities:
            predictions.append({
                "text": e.text,
                "original_label": e.original_label,
                "standardized_label": e.standardized_label,
                "start": e.start,
                "end": e.end,
                "score": e.score,
                "source": e.source,
            })
        rows.append({
            "query_id": record.query_id,
            "text": record.text,
            "has_phi": record.has_phi,
            "latency_ms": result.latency_ms,
            "predictions": predictions,
        })
    return rows


def check_output(records: list[DatasetRecord], rows: list[dict]) -> None:
    """Raise ValueError unless there is exactly one row per record, in dataset order."""
    if len(rows) != len(records):
        raise ValueError(f"Processed {len(rows)} rows but the dataset has {len(records)} records")
    row_ids = [row["query_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        duplicates = sorted(qid for qid, n in Counter(row_ids).items() if n > 1)
        raise ValueError(f"Duplicate query_id(s) in output: {duplicates}")
    if row_ids != [record.query_id for record in records]:
        raise ValueError("Output rows are not in dataset order")


def percentile(values: list[float], p: int) -> float:
    """p-th percentile with linear interpolation between data points."""
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def count_sorted(counter: Counter) -> dict[str, int]:
    """Counter as a dict ordered by count (descending), then label."""
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarise an empty run")

    predictions = [p for row in rows for p in row["predictions"]]
    by_original = Counter(p["original_label"] for p in predictions)
    by_standardized = Counter(p["standardized_label"] or UNMAPPED for p in predictions)
    mapped = sum(1 for p in predictions if p["standardized_label"] is not None)
    on_hard_negatives = sum(len(row["predictions"]) for row in rows if not row["has_phi"])
    with_predictions = sum(1 for row in rows if row["predictions"])
    latencies = [row["latency_ms"] for row in rows]

    return {
        "detector": "presidio",
        "dataset": {
            "total_records": len(rows),
            "phi_positive_records": sum(1 for row in rows if row["has_phi"]),
            "hard_negative_records": sum(1 for row in rows if not row["has_phi"]),
            "records_with_predictions": with_predictions,
            "records_without_predictions": len(rows) - with_predictions,
        },
        "predictions": {
            "total": len(predictions),
            "by_original_label": count_sorted(by_original),
            "by_standardized_label": count_sorted(by_standardized),
            "mapped": mapped,
            "unmapped": len(predictions) - mapped,
            "on_hard_negative_records": on_hard_negatives,
        },
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 95),
        },
    }


def select_samples(rows: list[dict], per_label: int = SAMPLES_PER_LABEL) -> dict[str, list[dict]]:
    """Up to `per_label` examples per original label, for manual inspection.

    To avoid near-identical examples, only the first prediction of each distinct
    (case-insensitive) detected text is considered; the chosen examples are then
    spread evenly from the lowest to the highest score.
    """
    candidates: dict[str, list[dict]] = {}
    seen_texts: dict[str, set[str]] = {}
    for row in rows:
        for p in row["predictions"]:
            label = p["original_label"]
            key = p["text"].lower()
            if key in seen_texts.setdefault(label, set()):
                continue
            seen_texts[label].add(key)
            candidates.setdefault(label, []).append({
                "query_id": row["query_id"],
                "query": row["text"],
                "detected_text": p["text"],
                "original_label": label,
                "standardized_label": p["standardized_label"],
                "start": p["start"],
                "end": p["end"],
                "score": p["score"],
            })

    samples = {}
    for label in sorted(candidates):
        examples = sorted(candidates[label], key=lambda s: (s["score"], s["query_id"], s["start"]))
        if len(examples) > per_label:
            step = (len(examples) - 1) / (per_label - 1)
            examples = [examples[round(i * step)] for i in range(per_label)]
        samples[label] = examples
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main(output_dir: Path = OUTPUT_DIR) -> dict:
    records = load_dataset()
    print(f"Loaded {len(records)} records")
    detector = PresidioDetector()

    rows = run_inference(records, detector)
    check_output(records, rows)
    summary = summarize(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "samples.json", select_samples(rows))
    print(f"Wrote predictions, summary and samples to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
