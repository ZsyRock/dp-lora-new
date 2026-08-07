from __future__ import annotations

import math
import unittest

from paper_repro.slaclip import (
    DEFAULT_BASE_TARGET_CLIPPED_FRACTION,
    Z_0995,
    automatic_num_slots,
    build_slack_vector,
    check_joint_release_bound,
    full_slaclip_update,
    normalize_noisy_slack,
    resolve_base_target_clipped_fraction,
    stationary_beta_from_exact_endpoints,
)


class SlaClipConstructionTests(unittest.TestCase):
    def test_automatic_slots_uses_paper_rule(self) -> None:
        expected = math.floor((64.0 / (2.0 * Z_0995)) ** (2.0 / 3.0))
        self.assertEqual(automatic_num_slots(64, 1.0), expected)

    def test_small_dp_lora_round_selects_one_slot(self) -> None:
        self.assertEqual(automatic_num_slots(5, 2.0), 1)
        self.assertEqual(automatic_num_slots(8, 2.0), 1)

    def test_one_slot_encoding(self) -> None:
        self.assertEqual(build_slack_vector(0.0, 10.0, 1), (10.0,))
        self.assertEqual(build_slack_vector(4.0, 10.0, 1), (6.0,))
        self.assertEqual(build_slack_vector(10.0, 10.0, 1), (0.0,))
        self.assertEqual(build_slack_vector(20.0, 10.0, 1), (0.0,))

    def test_joint_release_bound_across_norms_and_slot_counts(self) -> None:
        for slots in (1, 2, 3, 8, 13):
            for norm in (0.0, 0.01, 0.5, 1.0, 1.999999, 2.0, 4.0):
                with self.subTest(slots=slots, norm=norm):
                    slack = build_slack_vector(norm, 2.0, slots)
                    result = check_joint_release_bound(
                        norm,
                        2.0,
                        slots,
                        slack_vector=slack,
                    )
                    self.assertTrue(result["within_bound"])
                    self.assertLessEqual(result["joint_l2_norm"], 2.0 + 1e-10)

    def test_external_slack_that_breaks_bound_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            check_joint_release_bound(2.0, 2.0, 1, slack_vector=(1.0,))

    def test_noisy_slack_normalization(self) -> None:
        self.assertEqual(normalize_noisy_slack((25.0,), 10.0, 1, 5), (0.5,))
        self.assertEqual(
            normalize_noisy_slack((2.0, 4.0, 6.0, 8.0), 2.0, 4, 2),
            (1.0, 2.0, 3.0, 4.0),
        )


class FullSlaClipControllerTests(unittest.TestCase):
    def _update(
        self,
        near_threshold: float = 0.2,
        near_zero: float = 5.0,
        **overrides: float,
    ) -> dict:
        values = {
            "base_target_clipped_fraction": 0.5,
            "eta": 0.2,
            "min_clip_norm": 0.1,
            "max_clip_norm": 100.0,
        }
        values.update(overrides)
        return full_slaclip_update(
            10.0,
            near_threshold,
            near_zero,
            **values,
        )

    def test_matches_pinned_two_endpoint_equations(self) -> None:
        result = self._update()
        expected_adjusted = 5.0 / (10.0 + 1e-6)
        expected_target = 1.0 - 0.5 * (1.0 - expected_adjusted)
        expected_remaining = 1.0 - expected_adjusted
        expected_clipped_target = 0.5 * expected_remaining
        expected_step = 0.2 * (expected_target - 0.2)
        self.assertAlmostEqual(result["near_zero_adjusted"], expected_adjusted)
        self.assertEqual(result["base_target_clipped_fraction"], 0.5)
        self.assertEqual(result["beta"], 0.5)
        self.assertAlmostEqual(
            result["remaining_non_small_gradient_fraction"],
            expected_remaining,
        )
        self.assertAlmostEqual(
            result["raw_dynamic_target_clipped"],
            expected_clipped_target,
        )
        self.assertAlmostEqual(
            result["clamped_dynamic_target_clipped"],
            expected_clipped_target,
        )
        self.assertAlmostEqual(
            result["raw_dynamic_target_unclipped"], expected_target
        )
        self.assertAlmostEqual(result["dynamic_target_unclipped"], expected_target)
        self.assertAlmostEqual(result["dynamic_target_clipped"], 1.0 - expected_target)
        self.assertAlmostEqual(result["controller_error"], expected_target - 0.2)
        self.assertAlmostEqual(result["raw_log_step"], expected_step)
        self.assertAlmostEqual(result["bounded_log_step"], expected_step)
        self.assertAlmostEqual(
            result["next_clip_norm"],
            10.0 * math.exp(expected_step),
        )
        self.assertFalse(result["gamma_clamped_low"])
        self.assertFalse(result["gamma_clamped_high"])

    def test_beta_is_a_compatible_alias_for_the_canonical_base_target(self) -> None:
        canonical = self._update()
        alias = full_slaclip_update(
            10.0,
            0.2,
            5.0,
            beta=0.5,
            eta=0.2,
            min_clip_norm=0.1,
            max_clip_norm=100.0,
        )
        both = full_slaclip_update(
            10.0,
            0.2,
            5.0,
            base_target_clipped_fraction=0.5,
            beta=0.5,
            eta=0.2,
            min_clip_norm=0.1,
            max_clip_norm=100.0,
        )
        self.assertEqual(alias, canonical)
        self.assertEqual(both, canonical)
        with self.assertRaisesRegex(ValueError, "different values"):
            full_slaclip_update(
                10.0,
                0.2,
                5.0,
                base_target_clipped_fraction=0.25,
                beta=0.5,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
            )

    def test_unspecified_base_target_uses_the_documented_default(self) -> None:
        self.assertEqual(
            resolve_base_target_clipped_fraction(),
            DEFAULT_BASE_TARGET_CLIPPED_FRACTION,
        )
        result = full_slaclip_update(
            10.0,
            0.2,
            5.0,
            eta=0.2,
            min_clip_norm=0.1,
            max_clip_norm=100.0,
        )
        self.assertEqual(
            result["base_target_clipped_fraction"],
            DEFAULT_BASE_TARGET_CLIPPED_FRACTION,
        )

    def test_noisy_endpoints_are_not_preclamped(self) -> None:
        low = self._update(near_threshold=-2.0, near_zero=-100.0)
        high = self._update(near_threshold=3.0, near_zero=100.0)
        self.assertEqual(low["near_threshold_proxy"], -2.0)
        self.assertEqual(low["near_zero_proxy"], -100.0)
        self.assertLess(low["raw_dynamic_target_unclipped"], 0.0)
        self.assertEqual(low["dynamic_target_unclipped"], 0.0)
        self.assertGreater(low["raw_dynamic_target_clipped"], 1.0)
        self.assertEqual(low["clamped_dynamic_target_clipped"], 1.0)
        self.assertEqual(low["dynamic_target_clipped"], 1.0)
        self.assertTrue(low["gamma_clamped_low"])
        self.assertEqual(high["near_threshold_proxy"], 3.0)
        self.assertEqual(high["near_zero_proxy"], 100.0)
        self.assertGreater(high["raw_dynamic_target_unclipped"], 1.0)
        self.assertEqual(high["dynamic_target_unclipped"], 1.0)
        self.assertLess(high["raw_dynamic_target_clipped"], 0.0)
        self.assertEqual(high["clamped_dynamic_target_clipped"], 0.0)
        self.assertEqual(high["dynamic_target_clipped"], 0.0)
        self.assertTrue(high["gamma_clamped_high"])

    def test_one_slot_can_supply_the_same_value_to_both_endpoints(self) -> None:
        result = self._update(near_threshold=0.4, near_zero=0.4)
        expected_target = 1.0 - 0.5 * (1.0 - 0.4 / (10.0 + 1e-6))
        self.assertEqual(result["near_threshold_proxy"], 0.4)
        self.assertEqual(result["near_zero_proxy"], 0.4)
        self.assertAlmostEqual(result["dynamic_target_unclipped"], expected_target)
        self.assertAlmostEqual(result["dynamic_target_clipped"], 1.0 - expected_target)

    def test_public_threshold_and_numerical_step_are_bounded(self) -> None:
        upper = self._update(
            near_threshold=-100.0,
            eta=10.0,
            max_clip_norm=15.0,
        )
        lower = self._update(
            near_threshold=100.0,
            eta=10.0,
            min_clip_norm=5.0,
        )
        self.assertEqual(upper["next_clip_norm"], 15.0)
        self.assertTrue(upper["log_step_was_bounded"])
        self.assertTrue(upper["hit_max_clip_norm"])
        self.assertEqual(lower["next_clip_norm"], 5.0)
        self.assertTrue(lower["log_step_was_bounded"])
        self.assertTrue(lower["hit_min_clip_norm"])

    def test_zero_gain_is_a_valid_no_op(self) -> None:
        result = self._update(near_threshold=-100.0, eta=0.0)
        self.assertEqual(result["next_clip_norm"], 10.0)
        self.assertEqual(result["bounded_log_step"], 0.0)

    def test_explicit_epsilon_is_applied_and_recorded(self) -> None:
        result = self._update(epsilon=0.5)
        self.assertEqual(result["epsilon"], 0.5)
        self.assertAlmostEqual(result["near_zero_adjusted"], 5.0 / 10.5)

    def test_stationary_beta_inversion_uses_the_first_endpoint_not_hard_p(self) -> None:
        calibration = stationary_beta_from_exact_endpoints(10.0, 0.25, 0.0)
        self.assertEqual(calibration["stationary_beta"], 0.75)
        update = full_slaclip_update(
            10.0,
            0.25,
            0.0,
            beta=calibration["stationary_beta"],
            eta=0.2,
            min_clip_norm=0.1,
            max_clip_norm=50.0,
        )
        self.assertEqual(update["controller_error"], 0.0)
        self.assertEqual(update["next_clip_norm"], 10.0)

    def test_stationary_beta_rejects_non_monotone_exact_endpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 <= r_exact"):
            stationary_beta_from_exact_endpoints(10.0, 0.2, 0.3)


class SlaClipFailClosedTests(unittest.TestCase):
    def test_non_finite_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(function="slots", value=value):
                with self.assertRaises(ValueError):
                    automatic_num_slots(value, 2.0)
            with self.subTest(function="slack", value=value):
                with self.assertRaises(ValueError):
                    build_slack_vector(value, 10.0, 1)
            with self.subTest(function="normalize", value=value):
                with self.assertRaises(ValueError):
                    normalize_noisy_slack((value,), 10.0, 1, 5)
            with self.subTest(function="full_controller_q", value=value):
                with self.assertRaises(ValueError):
                    full_slaclip_update(
                        10.0,
                        value,
                        0.5,
                        beta=0.5,
                        eta=0.2,
                        min_clip_norm=0.1,
                        max_clip_norm=100.0,
                    )
            with self.subTest(function="full_controller_r", value=value):
                with self.assertRaises(ValueError):
                    full_slaclip_update(
                        10.0,
                        0.5,
                        value,
                        beta=0.5,
                        eta=0.2,
                        min_clip_norm=0.1,
                        max_clip_norm=100.0,
                    )

    def test_invalid_domains_are_rejected(self) -> None:
        invalid_calls = (
            lambda: automatic_num_slots(0, 2.0),
            lambda: automatic_num_slots(5, 0.0),
            lambda: build_slack_vector(-0.1, 10.0, 1),
            lambda: build_slack_vector(1.0, 0.0, 1),
            lambda: build_slack_vector(1.0, 10.0, 0),
            lambda: normalize_noisy_slack((1.0, 2.0), 10.0, 1, 5),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=-0.1,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
            ),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=1.1,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
            ),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=0.5,
                eta=-0.1,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
            ),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=0.5,
                eta=0.2,
                min_clip_norm=20.0,
                max_clip_norm=100.0,
            ),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=0.5,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=5.0,
            ),
            lambda: full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=0.5,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
                epsilon=0.0,
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_boolean_inputs_are_not_accepted_as_numbers(self) -> None:
        with self.assertRaises(TypeError):
            automatic_num_slots(True, 2.0)
        with self.assertRaises(TypeError):
            build_slack_vector(1.0, 10.0, True)
        with self.assertRaises(TypeError):
            full_slaclip_update(
                10.0,
                0.2,
                0.5,
                beta=True,
                eta=0.2,
                min_clip_norm=0.1,
                max_clip_norm=100.0,
            )


if __name__ == "__main__":
    unittest.main()
