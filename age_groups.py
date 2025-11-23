"""
Age grouping utilities.

Provides a 7-bucket age mapping via `age_to_group`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AgeGroup:
    """Represents an inclusive age range.

    max_inclusive=None means "no upper bound" (e.g. 60+).
    """
    name: str           # machine-friendly name (e.g. "young_adult")
    label: str          # human-friendly label (e.g. "Young adult (18–24)")
    min_inclusive: int
    max_inclusive: Optional[int] = None


# 7 reasonably detailed, but not hyper-granular, groups
AGE_GROUPS: List[AgeGroup] = [
    AgeGroup(
        name="early_child",
        label="Early childhood (0–5)",
        min_inclusive=0,
        max_inclusive=5,
    ),
    AgeGroup(
        name="child",
        label="Child (6–12)",
        min_inclusive=6,
        max_inclusive=12,
    ),
    AgeGroup(
        name="teen",
        label="Teen (13–17)",
        min_inclusive=13,
        max_inclusive=17,
    ),
    AgeGroup(
        name="young_adult",
        label="Young adult (18–24)",
        min_inclusive=18,
        max_inclusive=24,
    ),
    AgeGroup(
        name="adult",
        label="Adult (25–39)",
        min_inclusive=25,
        max_inclusive=39,
    ),
    AgeGroup(
        name="middle_aged",
        label="Middle-aged adult (40–59)",
        min_inclusive=40,
        max_inclusive=59,
    ),
    AgeGroup(
        name="senior",
        label="Older adult (60+)",
        min_inclusive=60,
        max_inclusive=None,
    ),
]


def age_to_group(age: int) -> AgeGroup:
    """Map a numeric age to a 7-bucket AgeGroup.

    Raises:
        ValueError: if age is negative or unrealistically large.
    """
    if age < 0:
        raise ValueError(f"Age cannot be negative (got {age}).")

    # Optional sanity cap – adjust or remove if you like
    if age > 120:
        raise ValueError(f"Age {age} is out of supported range (0–120).")

    for group in AGE_GROUPS:
        if group.max_inclusive is None:
            if age >= group.min_inclusive:
                return group
        elif group.min_inclusive <= age <= group.max_inclusive:
            return group

    # In practice we should never get here because of the 0–120 coverage.
    raise RuntimeError(f"No age group found for age {age} – check AGE_GROUPS config.")
