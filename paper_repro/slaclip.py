"""Small, dependency-free helpers for a fixed-target SlaClip-Q controller.

The formulas are independently expressed from the camera-ready SlaClip paper
published through OpenReview and the authors' official repository at commit
``d48b8e07aef33c58a3595ee18b4dccf9c75fa1f3``.  This module implements the
fixed-target SlaClip-Q ablation, not the full SlaClip controller whose target
is itself dynamic.

For one protected contribution with gradient norm ``n`` and threshold ``C``,
the K-slot construction augments the clipped gradient with slack coordinates
while retaining the same joint L2 bound ``C``.  The slack coordinates must be
noised as part of the same Gaussian release as the clipped gradient.  Merely
calling these helpers does not establish a privacy claim or an accountant.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from numbers import Real
from typing import Any


Z_0995 = 2.5758293035489004
_MAX_LOG_UPDATE = 50.0


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(name: str, value: Real) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_slots(num_slots: int) -> int:
    if isinstance(num_slots, bool):
        raise TypeError("num_slots must be an integer")
    try:
        result = operator.index(num_slots)
    except TypeError as error:
        raise TypeError("num_slots must be an integer") from error
    if result <= 0:
        raise ValueError("num_slots must be positive")
    return result


def automatic_num_slots(
    expected_records: Real,
    noise_multiplier: Real,
) -> int:
    """Return the largest integer K allowed by SlaClip's monotonicity rule.

    ``expected_records`` is the public, fixed normalization count for the
    release; it is not a randomly realized batch size.  Small DP-LoRA rounds
    can legitimately select ``K=1``.
    """

    records = _positive_real("expected_records", expected_records)
    sigma = _positive_real("noise_multiplier", noise_multiplier)
    ratio = records / (2.0 * Z_0995 * sigma)
    if not math.isfinite(ratio):
        raise FloatingPointError("automatic slot ratio is not finite")
    upper_bound = ratio ** (2.0 / 3.0)
    if not math.isfinite(upper_bound):
        raise FloatingPointError("automatic slot bound is not finite")
    return max(1, int(math.floor(upper_bound)))


def build_slack_vector(
    raw_norm: Real,
    clip_norm: Real,
    num_slots: int,
) -> tuple[float, ...]:
    """Build the official K-slot slack encoding for one scalar raw norm.

    With ``lambda = C/sqrt(K)``, the non-negative budget
    ``sqrt(K) * max(C-n, 0)`` is filled from the first slot onward, with every
    slot bounded by ``lambda``.  The resulting vector is deterministic and
    contains no noise.
    """

    norm = _finite_real("raw_norm", raw_norm)
    if norm < 0.0:
        raise ValueError("raw_norm must be non-negative")
    threshold = _positive_real("clip_norm", clip_norm)
    slots = _positive_slots(num_slots)
    sqrt_slots = math.sqrt(slots)
    coordinate_bound = threshold / sqrt_slots
    slack_budget = max(threshold - norm, 0.0) * sqrt_slots
    if (
        not math.isfinite(coordinate_bound)
        or coordinate_bound <= 0.0
        or not math.isfinite(slack_budget)
    ):
        raise FloatingPointError("slack construction is not finite")

    result: list[float] = []
    remaining = slack_budget
    for _ in range(slots):
        coordinate = min(coordinate_bound, max(remaining, 0.0))
        result.append(float(coordinate))
        remaining = max(0.0, remaining - coordinate)

    encoded = tuple(result)
    check_joint_release_bound(norm, threshold, slots, slack_vector=encoded)
    return encoded


def check_joint_release_bound(
    raw_norm: Real,
    clip_norm: Real,
    num_slots: int,
    *,
    slack_vector: Iterable[Real] | None = None,
) -> dict[str, float | bool]:
    """Validate and summarize the clipped-gradient-plus-slack L2 bound.

    A caller may supply a slack vector to validate an externally retained
    encoding.  Otherwise the canonical vector is reconstructed locally.
    """

    norm = _finite_real("raw_norm", raw_norm)
    if norm < 0.0:
        raise ValueError("raw_norm must be non-negative")
    threshold = _positive_real("clip_norm", clip_norm)
    slots = _positive_slots(num_slots)

    if slack_vector is None:
        # Inline the construction to avoid recursive validation.
        sqrt_slots = math.sqrt(slots)
        coordinate_bound = threshold / sqrt_slots
        remaining = max(threshold - norm, 0.0) * sqrt_slots
        values = []
        for _ in range(slots):
            coordinate = min(coordinate_bound, max(remaining, 0.0))
            values.append(float(coordinate))
            remaining = max(0.0, remaining - coordinate)
    else:
        try:
            values = [
                _finite_real(f"slack_vector[{index}]", value)
                for index, value in enumerate(slack_vector)
            ]
        except TypeError as error:
            raise TypeError("slack_vector must be an iterable of real numbers") from error
        if len(values) != slots:
            raise ValueError(
                f"slack_vector must contain exactly {slots} coordinates"
            )
        if any(value < 0.0 for value in values):
            raise ValueError("slack_vector coordinates must be non-negative")

    clipped_gradient_norm = min(norm, threshold)
    slack_squared_norm = math.fsum(value * value for value in values)
    joint_squared_norm = clipped_gradient_norm**2 + slack_squared_norm
    bound_squared = threshold**2
    if not all(
        math.isfinite(value)
        for value in (slack_squared_norm, joint_squared_norm, bound_squared)
    ):
        raise FloatingPointError("joint release bound calculation is not finite")
    tolerance = max(64.0 * math.ulp(bound_squared), 1e-12 * bound_squared)
    within_bound = joint_squared_norm <= bound_squared + tolerance
    if not within_bound:
        raise RuntimeError(
            "clipped gradient plus slack exceeds the declared joint L2 bound"
        )
    return {
        "clipped_gradient_norm": float(clipped_gradient_norm),
        "slack_l2_norm": float(math.sqrt(slack_squared_norm)),
        "joint_l2_norm": float(math.sqrt(joint_squared_norm)),
        "clip_norm_bound": float(threshold),
        "within_bound": True,
    }


def normalize_noisy_slack(
    sum_noisy_slack: Iterable[Real],
    clip_norm: Real,
    num_slots: int,
    expected_records: Real,
) -> tuple[float, ...]:
    """Normalize a summed, already-noised slack release into CDF proxies.

    The caller is responsible for adding Gaussian noise with coordinate scale
    appropriate to its release model.  This function divides by
    ``(C/sqrt(K)) * expected_records`` and never uses a realized batch count.
    """

    threshold = _positive_real("clip_norm", clip_norm)
    slots = _positive_slots(num_slots)
    records = _positive_real("expected_records", expected_records)
    try:
        values = tuple(
            _finite_real(f"sum_noisy_slack[{index}]", value)
            for index, value in enumerate(sum_noisy_slack)
        )
    except TypeError as error:
        raise TypeError(
            "sum_noisy_slack must be an iterable of real numbers"
        ) from error
    if len(values) != slots:
        raise ValueError(
            f"sum_noisy_slack must contain exactly {slots} coordinates"
        )
    denominator = (threshold / math.sqrt(slots)) * records
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("slack normalization denominator is not finite")
    normalized = tuple(value / denominator for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise FloatingPointError("normalized noisy slack is not finite")
    return normalized


def slaclip_q_update(
    current_clip_norm: Real,
    target_clipped_fraction: Real,
    noisy_unclipped_proxy: Real,
    eta: Real,
    min_clip_norm: Real,
    max_clip_norm: Real,
) -> dict[str, Any]:
    """Apply one bounded fixed-target SlaClip-Q threshold update.

    SlaClip-Q controls the first noisy slack coordinate, a smoothed *unclipped*
    CDF proxy.  Therefore a requested clipped fraction ``q`` is complemented
    to ``gamma = 1-q`` before applying the exponential feedback controller.
    The noisy proxy is intentionally not clamped because Gaussian outputs are
    unbounded.  Only the log update and the public threshold range are bounded.
    """

    current = _positive_real("current_clip_norm", current_clip_norm)
    target_clipped = _finite_real(
        "target_clipped_fraction", target_clipped_fraction
    )
    proxy = _finite_real("noisy_unclipped_proxy", noisy_unclipped_proxy)
    gain = _finite_real("eta", eta)
    lower = _positive_real("min_clip_norm", min_clip_norm)
    upper = _positive_real("max_clip_norm", max_clip_norm)
    if not 0.0 <= target_clipped <= 1.0:
        raise ValueError("target_clipped_fraction must be in [0, 1]")
    if gain < 0.0:
        raise ValueError("eta must be non-negative")
    if upper < lower:
        raise ValueError("max_clip_norm must be at least min_clip_norm")
    if not lower <= current <= upper:
        raise ValueError("current_clip_norm must lie within the declared bounds")

    target_unclipped = 1.0 - target_clipped
    controller_error = target_unclipped - proxy
    raw_log_update = gain * controller_error
    if not math.isfinite(raw_log_update):
        raise FloatingPointError("SlaClip-Q log update is not finite")
    bounded_log_update = max(
        -_MAX_LOG_UPDATE,
        min(_MAX_LOG_UPDATE, raw_log_update),
    )
    candidate = current * math.exp(bounded_log_update)
    if not math.isfinite(candidate) or candidate <= 0.0:
        raise FloatingPointError("SlaClip-Q candidate threshold is not finite")
    next_clip = max(lower, min(upper, candidate))

    return {
        "current_clip_norm": float(current),
        "target_clipped_fraction": float(target_clipped),
        "target_unclipped_proxy": float(target_unclipped),
        "noisy_unclipped_proxy": float(proxy),
        "controller_error": float(controller_error),
        "eta": float(gain),
        "raw_log_update": float(raw_log_update),
        "bounded_log_update": float(bounded_log_update),
        "log_update_was_clamped": bool(raw_log_update != bounded_log_update),
        "unbounded_next_clip_norm": float(candidate),
        "next_clip_norm": float(next_clip),
        "min_clip_norm": float(lower),
        "max_clip_norm": float(upper),
        "hit_lower_bound": bool(candidate < lower),
        "hit_upper_bound": bool(candidate > upper),
    }


__all__ = [
    "Z_0995",
    "automatic_num_slots",
    "build_slack_vector",
    "check_joint_release_bound",
    "normalize_noisy_slack",
    "slaclip_q_update",
]
