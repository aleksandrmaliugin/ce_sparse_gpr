from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Sequence

import numpy as np


def normalize_mask(
    mask: Sequence[bool] | np.ndarray | None,
) -> tuple[bool, ...] | None:
    """
    Return a validated immutable descriptor mask.
    """
    if mask is None:
        return None

    mask_arr = np.asarray(mask, dtype=bool).ravel()

    if mask_arr.size == 0:
        raise ValueError("descriptor_mask must not be empty.")

    if not bool(np.any(mask_arr)):
        raise ValueError("descriptor_mask removes all descriptors.")

    return tuple(bool(x) for x in mask_arr)


def _normalize_elements(elements: Sequence[str]) -> tuple[str, ...]:
    elements = tuple(str(e) for e in elements)

    if len(elements) == 0:
        raise ValueError("elements must contain at least one chemical symbol.")

    duplicates = sorted({e for e in elements if elements.count(e) > 1})
    if duplicates:
        raise ValueError(f"elements must be unique, duplicates: {duplicates}")

    if any(e == "" for e in elements):
        raise ValueError("elements must not contain empty symbols.")

    return elements


def _normalize_positive_float(value: Real, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")

    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")

    return value


def _normalize_shells(shells: Sequence[Real]) -> tuple[float, ...]:
    shells = tuple(float(x) for x in shells)

    if len(shells) < 2:
        raise ValueError("shells must contain at least two values.")

    if not np.all(np.isfinite(shells)):
        raise ValueError(f"shells must be finite, got {shells}.")

    if shells[0] < 0.0:
        raise ValueError(f"shells[0] must be non-negative, got {shells[0]}.")

    for left, right in zip(shells[:-1], shells[1:]):
        if right <= left:
            raise ValueError(
                "shells must be strictly increasing; "
                f"got adjacent values {left} and {right}."
            )

    return shells


@dataclass
class CEConfig:
    """Configuration for the discrete CE/cluster-count descriptors.

    shells are dimensionless multipliers of mindist.  For example, if
    mindist=1.50 and shells=(1.0, 1.6, 2.1), the shell bounds are
    [1.50, 2.40) and [2.40, 3.15) Angstrom.
    """

    elements: Sequence[str]
    mindist: float
    shells: Sequence[float]
    max_order: int = 2
    descriptor_mask: Sequence[bool] | np.ndarray | None = None

    shells_dict: dict[str, tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.elements = _normalize_elements(self.elements)
        self.mindist = _normalize_positive_float(self.mindist, "mindist")
        self.shells = _normalize_shells(self.shells)
        self.max_order = int(self.max_order)

        if self.max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3.")

        self.descriptor_mask = normalize_mask(self.descriptor_mask)
        self._build_shells_dict()

    def _build_shells_dict(self) -> None:
        shells_dict: dict[str, tuple[float, float]] = {}

        for r1, r2 in zip(self.shells[:-1], self.shells[1:]):
            rmin = r1 * self.mindist
            rmax = r2 * self.mindist
            shell_name = f"{rmin:.2f}"

            if shell_name in shells_dict:
                raise ValueError(
                    "Different shell lower bounds map to the same formatted name "
                    f"'{shell_name}'. Use less closely spaced shell boundaries."
                )

            shells_dict[shell_name] = (rmin, rmax)

        self.shells_dict = shells_dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "elements": list(self.elements),
            "mindist": self.mindist,
            "shells": list(self.shells),
            "max_order": self.max_order,
            "descriptor_mask": (
                list(self.descriptor_mask)
                if self.descriptor_mask is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CEConfig":
        if d is None:
            raise ValueError("Cannot build CEConfig from None.")

        return cls(
            elements=d["elements"],
            mindist=d["mindist"],
            shells=d["shells"],
            max_order=d.get("max_order", 2),
            descriptor_mask=d.get("descriptor_mask"),
        )
