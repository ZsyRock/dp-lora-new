from __future__ import annotations

import math
import unittest

from paper_repro.slaclip import (
    Z_0995,
    automatic_num_slots,
    build_slack_vector,
    check_joint_release_bound,
    normalize_noisy_slack,
    slaclip_q_update,
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


class SlaClipQControllerTests(unittest.TestCase):
    def _update(self, proxy: float, **overrides: float) -> dict:
        values = {
            "current_clip_norm": 10.0,
            "target_clipped_fraction": 0.2,
            "noisy_unclipped_proxy": proxy,
            "eta": 0.2,
            "min_clip_norm": 0.1,
            "max_clip_norm": 100.0,
        }
        values.update(overrides)
        return slaclip_q_update(**values)

    def test_target_is_complemented_to_unclipped_proxy(self) -> None:
        result = self._update(
            0.01,
            target_clipped_fraction=0.99,
        )
        self.assertAlmostEqual(result["target_unclipped_proxy"], 0.01)
        self.assertAlmostEqual(result["controller_error"], 0.0)
        self.assertAlmostEqual(result["next_clip_norm"], 10.0)

    def test_controller_direction_and_fixed_point(self) -> None:
        increase = self._update(0.3)
        fixed = self._update(0.8)
        decrease = self._update(0.95)
        self.assertGreater(increase["next_clip_norm"], 10.0)
        self.assertAlmostEqual(fixed["next_clip_norm"], 10.0)
        self.assertLess(decrease["next_clip_norm"], 10.0)

    def test_controller_reports_both_threshold_bounds(self) -> None:
        upper = self._update(
            -100.0,
            eta=10.0,
            max_clip_norm=15.0,
        )
        lower = self._update(
            100.0,
            eta=10.0,
            min_clip_norm=5.0,
        )
        self.assertEqual(upper["next_clip_norm"], 15.0)
        self.assertTrue(upper["hit_upper_bound"])
        self.assertFalse(upper["hit_lower_bound"])
        self.assertEqual(lower["next_clip_norm"], 5.0)
        self.assertTrue(lower["hit_lower_bound"])
        self.assertFalse(lower["hit_upper_bound"])

    def test_gaussian_proxy_is_not_artificially_clamped(self) -> None:
        result = self._update(-2.0)
        self.assertAlmostEqual(result["noisy_unclipped_proxy"], -2.0)
        self.assertGreater(result["controller_error"], 1.0)

    def test_zero_gain_is_a_valid_no_op(self) -> None:
        result = self._update(-100.0, eta=0.0)
        self.assertEqual(result["next_clip_norm"], 10.0)
        self.assertEqual(result["bounded_log_update"], 0.0)


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
            with self.subTest(function="controller", value=value):
                with self.assertRaises(ValueError):
                    slaclip_q_update(10.0, 0.2, value, 0.2, 0.1, 100.0)

    def test_invalid_domains_are_rejected(self) -> None:
        invalid_calls = (
            lambda: automatic_num_slots(0, 2.0),
            lambda: automatic_num_slots(5, 0.0),
            lambda: build_slack_vector(-0.1, 10.0, 1),
            lambda: build_slack_vector(1.0, 0.0, 1),
            lambda: build_slack_vector(1.0, 10.0, 0),
            lambda: normalize_noisy_slack((1.0, 2.0), 10.0, 1, 5),
            lambda: slaclip_q_update(10.0, -0.1, 0.5, 0.2, 0.1, 100.0),
            lambda: slaclip_q_update(10.0, 1.1, 0.5, 0.2, 0.1, 100.0),
            lambda: slaclip_q_update(10.0, 0.2, 0.5, -0.1, 0.1, 100.0),
            lambda: slaclip_q_update(10.0, 0.2, 0.5, 0.2, 20.0, 100.0),
            lambda: slaclip_q_update(10.0, 0.2, 0.5, 0.2, 0.1, 5.0),
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


if __name__ == "__main__":
    unittest.main()
