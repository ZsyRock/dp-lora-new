"""Small, dependency-free helpers for SlaClip's CDF endpoint controller.

The formulas are independently expressed from the camera-ready SlaClip paper
published through OpenReview and the authors' official repository at commit
``d48b8e07aef33c58a3595ee18b4dccf9c75fa1f3``.  The active controller uses
the first and last entries of the noisy binned CDF to derive its target each
round.  The obsolete fixed-target ablation is intentionally not part of this
active code.

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
_ENDPOINT_DENOMINATOR_EPSILON = 1e-6
MAX_ABS_LOG_STEP = 50.0
DEFAULT_BASE_TARGET_CLIPPED_FRACTION = 0.5


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


def resolve_base_target_clipped_fraction(
    *,
    base_target_clipped_fraction: Real | None = None,
    beta: Real | None = None,
) -> float:
    """Resolve full SlaClip's canonical base target and legacy ``beta`` alias.

    ``beta`` is retained solely as a backwards-compatible name.  Supplying
    both names with different values is rejected so an experiment can never
    silently choose one controller setting over another.
    """

    canonical = (
        None
        if base_target_clipped_fraction is None
        else _finite_real(
            "base_target_clipped_fraction",
            base_target_clipped_fraction,
        )
    )
    alias = None if beta is None else _finite_real("beta", beta)
    if canonical is not None and alias is not None and canonical != alias:
        raise ValueError(
            "base_target_clipped_fraction and beta specify different values"
        )
    value = (
        canonical
        if canonical is not None
        else alias
        if alias is not None
        else DEFAULT_BASE_TARGET_CLIPPED_FRACTION
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError("base_target_clipped_fraction must be in [0, 1]")
    return float(value)


def full_slaclip_update(
    current_clip_norm: Real,
    noisy_cdf_near_threshold: Real,
    noisy_cdf_near_zero: Real,
    *,
    base_target_clipped_fraction: Real | None = None,
    beta: Real | None = None,
    eta: Real,
    min_clip_norm: Real,
    max_clip_norm: Real,
    epsilon: Real = _ENDPOINT_DENOMINATOR_EPSILON,
) -> dict[str, Any]:
    """Apply one full SlaClip update from the two noisy CDF endpoints.

    Following the paper and pinned reference implementation, let ``q_hat`` be
    the first normalized noisy slack coordinate (the endpoint nearest the
    current threshold) and ``r_hat`` the final coordinate (the endpoint nearest
    zero).  The dynamic target and threshold update are

    ``z_hat = r_hat / (C + 1e-6)``,
    ``remaining_t = 1 - z_hat``,
    ``target_clip_t = clip_[0,1](base_target_clip * remaining_t)``,
    ``gamma_t = 1 - target_clip_t``, and
    ``C_next = clip_[Cmin,Cmax](C * exp(eta * (gamma_t - q_hat)))``.

    The paper/reference name ``beta`` is accepted as a compatibility alias for
    ``base_target_clipped_fraction``.  It is not a fixed clipping target:
    every round still modulates it with the noisy near-zero CDF endpoint.
    Both endpoint inputs intentionally remain unclamped because a Gaussian
    release is unbounded.  Only the derived target, numerical log step, and
    public threshold range are bounded.  Exact, non-noisy CDF values must not
    be supplied when this controller is intended to consume the DP release.
    """

    current = _positive_real("current_clip_norm", current_clip_norm)
    q_hat = _finite_real(
        "noisy_cdf_near_threshold", noisy_cdf_near_threshold
    )
    r_hat = _finite_real("noisy_cdf_near_zero", noisy_cdf_near_zero)
    base_target = resolve_base_target_clipped_fraction(
        base_target_clipped_fraction=base_target_clipped_fraction,
        beta=beta,
    )
    gain = _finite_real("eta", eta)
    lower = _positive_real("min_clip_norm", min_clip_norm)
    upper = _positive_real("max_clip_norm", max_clip_norm)
    denominator_epsilon = _positive_real("epsilon", epsilon)
    if gain < 0.0:
        raise ValueError("eta must be non-negative")
    if upper < lower:
        raise ValueError("max_clip_norm must be at least min_clip_norm")
    if not lower <= current <= upper:
        raise ValueError("current_clip_norm must lie within the declared bounds")

    endpoint_denominator = current + denominator_epsilon
    if not math.isfinite(endpoint_denominator) or endpoint_denominator <= 0.0:
        raise FloatingPointError("SlaClip endpoint denominator is not finite")
    z_hat = r_hat / endpoint_denominator
    if not math.isfinite(z_hat):
        raise FloatingPointError("SlaClip normalized near-zero endpoint is not finite")
    remaining_non_small_gradient_fraction = 1.0 - z_hat
    if not math.isfinite(remaining_non_small_gradient_fraction):
        raise FloatingPointError(
            "SlaClip remaining non-small-gradient fraction is not finite"
        )
    raw_dynamic_target_clipped = (
        base_target * remaining_non_small_gradient_fraction
    )
    if not math.isfinite(raw_dynamic_target_clipped):
        raise FloatingPointError("SlaClip raw clipped target is not finite")
    clamped_dynamic_target_clipped = max(
        0.0,
        min(1.0, raw_dynamic_target_clipped),
    )
    raw_gamma = 1.0 - raw_dynamic_target_clipped
    if not math.isfinite(raw_gamma):
        raise FloatingPointError("SlaClip dynamic target is not finite")
    gamma_t = 1.0 - clamped_dynamic_target_clipped

    controller_error = gamma_t - q_hat
    raw_log_update = gain * controller_error
    if not math.isfinite(raw_log_update):
        raise FloatingPointError("SlaClip log update is not finite")
    bounded_log_update = max(
        -MAX_ABS_LOG_STEP,
        min(MAX_ABS_LOG_STEP, raw_log_update),
    )
    candidate = current * math.exp(bounded_log_update)
    if not math.isfinite(candidate) or candidate <= 0.0:
        raise FloatingPointError("SlaClip candidate threshold is not finite")
    next_clip = max(lower, min(upper, candidate))

    return {
        "current_clip_norm": float(current),
        "near_threshold_proxy": float(q_hat),
        "near_zero_proxy": float(r_hat),
        "base_target_clipped_fraction": float(base_target),
        "beta": float(base_target),
        "epsilon": float(denominator_epsilon),
        "endpoint_denominator": float(endpoint_denominator),
        "near_zero_adjusted": float(z_hat),
        "remaining_non_small_gradient_fraction": float(
            remaining_non_small_gradient_fraction
        ),
        "raw_dynamic_target_clipped": float(raw_dynamic_target_clipped),
        "clamped_dynamic_target_clipped": float(
            clamped_dynamic_target_clipped
        ),
        "raw_dynamic_target_unclipped": float(raw_gamma),
        "dynamic_target_unclipped": float(gamma_t),
        "dynamic_target_clipped": float(clamped_dynamic_target_clipped),
        "gamma_clamped_low": bool(raw_gamma < 0.0),
        "gamma_clamped_high": bool(raw_gamma > 1.0),
        "controller_error": float(controller_error),
        "eta": float(gain),
        "raw_log_step": float(raw_log_update),
        "bounded_log_step": float(bounded_log_update),
        "log_step_was_bounded": bool(raw_log_update != bounded_log_update),
        "unbounded_next_clip_norm": float(candidate),
        "next_clip_norm": float(next_clip),
        "min_clip_norm": float(lower),
        "max_clip_norm": float(upper),
        "hit_min_clip_norm": bool(candidate < lower),
        "hit_max_clip_norm": bool(candidate > upper),
    }


__all__ = [
    "Z_0995",
    "MAX_ABS_LOG_STEP",
    "DEFAULT_BASE_TARGET_CLIPPED_FRACTION",
    "automatic_num_slots",
    "build_slack_vector",
    "check_joint_release_bound",
    "normalize_noisy_slack",
    "resolve_base_target_clipped_fraction",
    "full_slaclip_update",
]
