"""Stage 2D: evaluate the saved Presidio predictions against ASQ-PHI gold annotations.

Usage (from the project root, after `python -m benchmarking.run_presidio`):
    python -m benchmarking.evaluate_presidio

Reads data/processed/asq_phi_processed.jsonl and results/presidio/predictions.jsonl
(neither is modified) and writes to results/presidio/:
  - evaluation.json:        counts, overall and per-category metrics, error summaries
  - false_positives.jsonl:  one line per false-positive prediction
  - false_negatives.jsonl:  one line per false-negative gold entity
  - evaluation_report.md:   short human-readable report

Matching is strict and one-to-one on (start, end, standardized_label); see
benchmarking/evaluation/metrics.py. No inference is run here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarking.data.loader import DatasetRecord, load_dataset
from benchmarking.detectors.label_mapping import ASQ_PHI_LABELS, DETECTOR_LABEL_MAPPINGS
from benchmarking.evaluation.metrics import match_exact, precision_recall_f1, spans_overlap
from benchmarking.run_presidio import UNMAPPED, count_sorted, write_json, write_jsonl

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "presidio"
PREDICTION_FIELDS = ("text", "original_label", "standardized_label", "start", "end", "score", "source")

METHODOLOGY = {
    "matching": "strict one-to-one exact match on (start, end, standardized_label)",
    "offsets": "start inclusive, end exclusive, Python string (code point) indices",
    "duplicates": "a gold entity can be matched once; further predictions with the same key are FPs",
    "unmapped_predictions": "excluded from all category metrics (not counted as FP); reported separately",
    "hard_negatives": "mapped predictions on hard-negative records are FPs",
    "unresolved_gold": "records containing an ambiguous/missing gold annotation are excluded entirely",
    "thresholds": "none; every saved prediction is used regardless of score",
    "zero_denominators": "a metric is null when its denominator is 0; F1 = 2TP / (2TP + FP + FN)",
}


def load_predictions(path: Path, records: list[DatasetRecord]) -> list[dict]:
    """Load Stage 2C predictions and check they belong to `records`, in the same order.

    Raises ValueError if any row or prediction is inconsistent with the dataset or
    with the current Presidio label mapping.
    """
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) != len(records):
        raise ValueError(f"{path.name}: {len(rows)} rows but the dataset has {len(records)} records")
    mapping = DETECTOR_LABEL_MAPPINGS["presidio"]

    for row, record in zip(rows, records):
        where = f"{path.name} (query_id={row.get('query_id')!r})"
        if row.get("query_id") != record.query_id:
            raise ValueError(f"{where}: expected query_id {record.query_id} (rows must follow dataset order)")
        if row.get("text") != record.text:
            raise ValueError(f"{where}: query text differs from the dataset")
        if not isinstance(row.get("predictions"), list):
            raise ValueError(f"{where}: 'predictions' must be a list")

        for i, p in enumerate(row["predictions"]):
            where_p = f"{where}, prediction {i}"
            missing = [k for k in PREDICTION_FIELDS if k not in p]
            if missing:
                raise ValueError(f"{where_p}: missing field(s) {missing}")
            start, end = p["start"], p["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(record.text):
                raise ValueError(f"{where_p}: invalid offsets [{start!r}, {end!r})")
            if record.text[start:end] != p["text"]:
                raise ValueError(f"{where_p}: text {p['text']!r} does not match query[{start}:{end}]")
            if p["standardized_label"] != mapping.get(p["original_label"]):
                raise ValueError(
                    f"{where_p}: standardized_label {p['standardized_label']!r} for {p['original_label']!r} "
                    f"does not match the current Presidio mapping")
    return rows


def _gold_summary(e) -> dict:
    return {"text": e.text, "label": e.label, "start": e.start, "end": e.end}


def _prediction_summary(p: dict) -> dict:
    return {"text": p["text"], "original_label": p["original_label"],
            "standardized_label": p["standardized_label"],
            "start": p["start"], "end": p["end"], "score": p["score"]}


def evaluate(records: list[DatasetRecord], rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Return (evaluation, false_positives, false_negatives) for aligned records and rows."""
    categories = sorted(ASQ_PHI_LABELS)
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in categories}
    false_positives: list[dict] = []
    false_negatives: list[dict] = []

    excluded_ids = []
    excluded_annotations = 0
    excluded_unresolved = 0
    excluded_predictions = 0
    eligible_records = 0
    eligible_gold = 0
    eligible_predictions = 0

    unmapped_by_label = Counter()
    unmapped_on_gold_span = Counter()     # "UNMAPPED_LABEL -> GOLD_LABEL" for exact span matches
    fp_by_original_label = Counter()
    fp_by_gold_relation = Counter()
    fn_by_prediction_relation = Counter()

    hard_negative_records = 0
    hard_negative_records_with_predictions = 0
    hard_negative_by_category = Counter()
    hard_negative_by_original_label = Counter()

    same_span_groups = 0
    same_span_conflicting_groups = 0
    same_span_combinations: dict[str, list[int]] = {}
    overlap_pairs = 0
    overlap_combinations: dict[str, list[int]] = {}

    for record, row in zip(records, rows):
        predictions = row["predictions"]

        if any(e.alignment_status != "matched" for e in record.entities):
            excluded_ids.append(record.query_id)
            excluded_annotations += len(record.entities)
            excluded_unresolved += sum(1 for e in record.entities if e.alignment_status != "matched")
            excluded_predictions += len(predictions)
            continue

        eligible_records += 1
        eligible_gold += len(record.entities)
        eligible_predictions += len(predictions)

        gold = record.entities
        gold_keys = [(e.start, e.end, e.label) for e in gold]
        gold_spans = {(e.start, e.end): e.label for e in gold}
        mapped = [p for p in predictions if p["standardized_label"] is not None]
        mapped_keys = [(p["start"], p["end"], p["standardized_label"]) for p in mapped]
        matched_gold, matched_predictions = match_exact(gold_keys, mapped_keys)

        for i, e in enumerate(gold):
            if i in matched_gold:
                counts[e.label]["tp"] += 1
                continue
            counts[e.label]["fn"] += 1
            overlapping = [p for p in predictions if spans_overlap(p["start"], p["end"], e.start, e.end)]
            same_span = [p for p in overlapping if (p["start"], p["end"]) == (e.start, e.end)]
            if any(p["standardized_label"] is not None for p in same_span):
                fn_by_prediction_relation["same_span_different_label"] += 1
            elif same_span:
                fn_by_prediction_relation["same_span_unmapped_only"] += 1
            elif overlapping:
                fn_by_prediction_relation["overlapping_prediction_different_span"] += 1
            else:
                fn_by_prediction_relation["no_overlapping_prediction"] += 1
            false_negatives.append({
                "query_id": record.query_id, "query": record.text,
                "text": e.text, "label": e.label, "start": e.start, "end": e.end,
                "overlapping_predictions": [_prediction_summary(p) for p in overlapping],
            })

        for j, p in enumerate(mapped):
            if j in matched_predictions:
                continue
            label = p["standardized_label"]
            counts[label]["fp"] += 1
            fp_by_original_label[p["original_label"]] += 1
            overlapping = [e for e in gold if spans_overlap(p["start"], p["end"], e.start, e.end)]
            if mapped_keys[j] in gold_keys:
                fp_by_gold_relation["duplicate_of_matched_prediction"] += 1
            elif (p["start"], p["end"]) in gold_spans:
                fp_by_gold_relation["same_span_different_label"] += 1
            elif overlapping:
                fp_by_gold_relation["overlapping_gold_different_span"] += 1
            else:
                fp_by_gold_relation["no_overlapping_gold"] += 1
            false_positives.append({
                "query_id": record.query_id, "query": record.text, "has_phi": record.has_phi,
                "text": p["text"], "standardized_label": label, "original_label": p["original_label"],
                "start": p["start"], "end": p["end"], "score": p["score"],
                "overlapping_gold": [_gold_summary(e) for e in overlapping],
            })

        for p in predictions:
            if p["standardized_label"] is None:
                unmapped_by_label[p["original_label"]] += 1
                gold_label = gold_spans.get((p["start"], p["end"]))
                if gold_label is not None:
                    unmapped_on_gold_span[f"{p['original_label']} -> {gold_label}"] += 1

        if not record.has_phi:
            hard_negative_records += 1
            if predictions:
                hard_negative_records_with_predictions += 1
            for p in predictions:
                hard_negative_by_category[p["standardized_label"] or UNMAPPED] += 1
                hard_negative_by_original_label[p["original_label"]] += 1

        by_span: dict[tuple, list[dict]] = {}
        for p in predictions:
            by_span.setdefault((p["start"], p["end"]), []).append(p)
        for group in by_span.values():
            if len(group) > 1:
                same_span_groups += 1
                if len({p["standardized_label"] or UNMAPPED for p in group}) > 1:
                    same_span_conflicting_groups += 1
                combination = " + ".join(sorted(p["original_label"] for p in group))
                same_span_combinations.setdefault(combination, []).append(record.query_id)
        for a in range(len(predictions)):
            for b in range(a + 1, len(predictions)):
                pa, pb = predictions[a], predictions[b]
                if (pa["start"], pa["end"]) != (pb["start"], pb["end"]) and \
                        spans_overlap(pa["start"], pa["end"], pb["start"], pb["end"]):
                    overlap_pairs += 1
                    combination = " / ".join(sorted([pa["original_label"], pb["original_label"]]))
                    overlap_combinations.setdefault(combination, []).append(record.query_id)

    total_tp = sum(c["tp"] for c in counts.values())
    total_fp = sum(c["fp"] for c in counts.values())
    total_fn = sum(c["fn"] for c in counts.values())
    mapped_total = sum(c["tp"] + c["fp"] for c in counts.values())
    unmapped_total = sum(unmapped_by_label.values())

    def with_query_ids(combinations: dict[str, list[int]]) -> dict:
        ordered = sorted(combinations.items(), key=lambda item: (-len(item[1]), item[0]))
        return {k: {"count": len(ids), "query_ids": sorted(set(ids))} for k, ids in ordered}

    evaluation = {
        "detector": "presidio",
        "methodology": METHODOLOGY,
        "counts": {
            "dataset_records": len(records),
            "eligible_records": eligible_records,
            "excluded_records": len(excluded_ids),
            "excluded_query_ids": excluded_ids,
            "excluded_gold_annotations": excluded_annotations,
            "excluded_unresolved_gold_annotations": excluded_unresolved,
            "excluded_predictions": excluded_predictions,
            "eligible_gold_annotations": eligible_gold,
            "eligible_predictions": eligible_predictions,
            "mapped_predictions": mapped_total,
            "unmapped_predictions": unmapped_total,
        },
        "overall": precision_recall_f1(total_tp, total_fp, total_fn),
        "per_category": {label: precision_recall_f1(**counts[label]) for label in categories},
        "unmapped_predictions": {
            "total": unmapped_total,
            "by_original_label": count_sorted(unmapped_by_label),
            "exact_gold_span_by_label": count_sorted(unmapped_on_gold_span),
        },
        "hard_negatives": {
            "records": hard_negative_records,
            "records_with_predictions": hard_negative_records_with_predictions,
            "predictions": sum(hard_negative_by_category.values()),
            "mapped_predictions_counted_as_fp": sum(
                n for label, n in hard_negative_by_category.items() if label != UNMAPPED),
            "by_category": count_sorted(hard_negative_by_category),
            "by_original_label": count_sorted(hard_negative_by_original_label),
        },
        "errors": {
            "fp_by_category": {label: counts[label]["fp"] for label in categories},
            "fn_by_category": {label: counts[label]["fn"] for label in categories},
            "fp_by_original_label": count_sorted(fp_by_original_label),
            "fp_by_gold_relation": count_sorted(fp_by_gold_relation),
            "fn_by_prediction_relation": count_sorted(fn_by_prediction_relation),
        },
        "overlaps": {
            "same_span_groups": same_span_groups,
            "same_span_groups_with_conflicting_labels": same_span_conflicting_groups,
            "same_span_label_combinations": with_query_ids(same_span_combinations),
            "overlapping_pairs_different_spans": overlap_pairs,
            "overlapping_label_combinations": with_query_ids(overlap_combinations),
        },
    }
    return evaluation, false_positives, false_negatives


def _fmt(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def build_report(evaluation: dict) -> str:
    c = evaluation["counts"]
    o = evaluation["overall"]
    u = evaluation["unmapped_predictions"]
    h = evaluation["hard_negatives"]
    err = evaluation["errors"]
    ov = evaluation["overlaps"]
    per_category = evaluation["per_category"]

    lines = [
        "# Presidio evaluation on ASQ-PHI (Stage 2D)",
        "",
        "Baseline evaluation of saved Presidio predictions (default recognizers, no score "
        "threshold) against ASQ-PHI gold annotations. Generated by `benchmarking/evaluate_presidio.py`.",
        "",
        "## Methodology",
        "",
    ]
    lines += [f"- **{key.replace('_', ' ').capitalize()}:** {value}" for key, value in evaluation["methodology"].items()]
    lines += [
        "",
        "## Data",
        "",
        f"- Records: {c['dataset_records']} total, {c['eligible_records']} eligible, "
        f"{c['excluded_records']} excluded (query IDs {c['excluded_query_ids']}).",
        f"- Excluded: {c['excluded_gold_annotations']} gold annotations "
        f"({c['excluded_unresolved_gold_annotations']} unresolved) and {c['excluded_predictions']} predictions.",
        f"- Eligible gold annotations: {c['eligible_gold_annotations']}.",
        f"- Eligible predictions: {c['eligible_predictions']} "
        f"({c['mapped_predictions']} mapped, {c['unmapped_predictions']} unmapped).",
        "",
        "## Overall (micro-averaged, mapped predictions only)",
        "",
        "| TP | FP | FN | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {o['tp']} | {o['fp']} | {o['fn']} | {_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} |",
        "",
        "## Per category",
        "",
        "`–` means the metric is undefined (zero denominator).",
        "",
        "| Category | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in per_category.items():
        lines.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | "
                     f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} |")

    no_tp = [label for label, m in per_category.items() if m["tp"] == 0 and m["fn"] > 0]
    top_fp_labels = list(err["fp_by_original_label"].items())[:3]
    lines += [
        "",
        "## Key findings",
        "",
        f"- Overall: precision {_fmt(o['precision'])}, recall {_fmt(o['recall'])}, F1 {_fmt(o['f1'])} "
        f"over {c['eligible_gold_annotations']} eligible gold annotations.",
        f"- Gold categories with no true positives: {', '.join(no_tp) if no_tp else 'none'}.",
        f"- Largest FP sources by Presidio label: "
        f"{', '.join(f'{k} ({v})' for k, v in top_fp_labels) if top_fp_labels else 'none'}.",
        f"- {h['records_with_predictions']} of {h['records']} hard-negative records have predictions; "
        f"{h['mapped_predictions_counted_as_fp']} mapped predictions on them count as FPs.",
        f"- {u['total']} unmapped predictions are excluded from the metrics; "
        f"{sum(u['exact_gold_span_by_label'].values())} of them have exactly the span of a gold entity.",
        "",
        "## Errors",
        "",
        "Individual errors, with the original query, are in `false_positives.jsonl` and `false_negatives.jsonl`.",
        "",
        f"- FP by Presidio label: {err['fp_by_original_label']}",
        f"- FP by relation to gold spans: {err['fp_by_gold_relation']}",
        f"- FN by relation to predictions: {err['fn_by_prediction_relation']}",
        "",
        "## Unmapped predictions",
        "",
        f"- By Presidio label: {u['by_original_label']}",
        f"- Exactly on a gold span (Presidio label -> gold label): {u['exact_gold_span_by_label']}",
        "",
        "## Hard negatives",
        "",
        f"- {h['predictions']} predictions on {h['records_with_predictions']} of {h['records']} records.",
        f"- By category: {h['by_category']}",
        f"- By Presidio label: {h['by_original_label']}",
        "",
        "## Overlapping predictions",
        "",
        f"- Groups of predictions sharing an identical span: {ov['same_span_groups']} "
        f"({ov['same_span_groups_with_conflicting_labels']} with conflicting standardised labels).",
        f"- Overlapping prediction pairs with different spans: {ov['overlapping_pairs_different_spans']}.",
        "- Label combinations and query IDs are listed in `evaluation.json` under `overlaps`.",
        "",
        "## Limitations",
        "",
        "- Exact matching counts a span that differs from gold by even one character as both an FP and an FN.",
        "- Unmapped predictions are not penalised, so precision covers only the 7 mapped Presidio labels; "
        "Presidio has no mapped label for several ASQ-PHI categories.",
        "- Gold labels follow ASQ-PHI annotation conventions, which may differ from Presidio's notion of each entity.",
        "- Records with unresolved gold annotations are excluded, so a few real entities are not evaluated.",
        "- Default Presidio configuration and a single run; results are a baseline, not a tuned system.",
        "- The `fp_by_gold_relation` / `fn_by_prediction_relation` groups describe span overlap only; "
        "they are not semantic error types.",
        "",
    ]
    return "\n".join(lines)


def main(results_dir: Path = RESULTS_DIR) -> dict:
    records = load_dataset()
    rows = load_predictions(results_dir / "predictions.jsonl", records)
    evaluation, false_positives, false_negatives = evaluate(records, rows)

    write_json(results_dir / "evaluation.json", evaluation)
    write_jsonl(results_dir / "false_positives.jsonl", false_positives)
    write_jsonl(results_dir / "false_negatives.jsonl", false_negatives)
    (results_dir / "evaluation_report.md").write_text(build_report(evaluation), encoding="utf-8")
    print(f"Wrote evaluation outputs to {results_dir}")
    return evaluation


if __name__ == "__main__":
    main()
