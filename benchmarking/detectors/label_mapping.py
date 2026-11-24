"""Detector-label -> ASQ-PHI-category mapping.

Each detector has a plain dict {original_label: asq_phi_category}, registered
under the detector's `name` in DETECTOR_LABEL_MAPPINGS. Labels without an entry
get `standardized_label=None`; they are kept, never dropped, and
`original_label` is never changed.

The registry is intentionally empty: add an entry when a real detector is
integrated, after inspecting the labels it actually emits.
"""

from __future__ import annotations

from dataclasses import replace

from .base import DetectedEntity

# The 13 gold categories used in ASQ-PHI (labels are not renamed).
ASQ_PHI_LABELS = {
    "GEOGRAPHIC_LOCATION", "NAME", "DATE", "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUMBER", "PHONE_NUMBER", "SOCIAL_SECURITY_NUMBER",
    "EMAIL_ADDRESS", "UNIQUE_IDENTIFIER", "ACCOUNT_NUMBER", "FAX_NUMBER",
    "CERTIFICATE_LICENSE_NUMBER", "IP_ADDRESS",
}

DETECTOR_LABEL_MAPPINGS: dict[str, dict[str, str]] = {}


def standardize_entities(entities: list[DetectedEntity],
                         mapping: dict[str, str]) -> list[DetectedEntity]:
    """Return copies of `entities` with `standardized_label` set from `mapping`.

    Matching on `original_label` is exact (case-sensitive). Any value already in
    `standardized_label` is overwritten. The input entities are not modified.
    Raises ValueError if the mapping targets a label that is not an ASQ-PHI category.
    """
    invalid = {v for v in mapping.values() if v not in ASQ_PHI_LABELS}
    if invalid:
        raise ValueError(f"Label mapping targets unknown ASQ-PHI categories: {sorted(invalid)}")

    standardized = []
    for entity in entities:
        asq_phi_label = mapping.get(entity.original_label)  # None if unmapped
        standardized.append(replace(entity, standardized_label=asq_phi_label))
    return standardized
