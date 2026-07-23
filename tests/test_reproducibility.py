from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from paper_repro.reproducibility import (
    METHOD_SPECS,
    PrivateKeyError,
    canonical_json_fingerprint,
    derive_seed,
    int64_index_digest,
    load_or_create_private_key,
    load_private_key,
    private_key_fingerprint,
    safe_quantiles,
    safe_ratio,
)


class MethodContractTests(unittest.TestCase):
    def test_method_specs_make_controls_and_paper_arm_explicit(self) -> None:
        self.assertEqual(
            set(METHOD_SPECS),
            {"no_dp_lora_control", "clip_only_control", "paper_dp_lora"},
        )
        self.assertFalse(METHOD_SPECS["no_dp_lora_control"].clipping_enabled)
        self.assertTrue(METHOD_SPECS["clip_only_control"].clipping_enabled)
        self.assertFalse(METHOD_SPECS["clip_only_control"].gaussian_noise_enabled)
        self.assertTrue(METHOD_SPECS["paper_dp_lora"].gaussian_noise_enabled)
        self.assertTrue(METHOD_SPECS["no_dp_lora_control"].is_control)
        self.assertFalse(METHOD_SPECS["paper_dp_lora"].independently_accounted)


class PrivateKeyTests(unittest.TestCase):
    def _private_directory(self, root: str) -> Path:
        directory = Path(root) / "private"
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        return directory

    def test_load_or_create_is_stable_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._private_directory(root) / "run.key"
            first = load_or_create_private_key(path)
            second = load_or_create_private_key(path)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 32)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_broad_key_permissions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._private_directory(root) / "run.key"
            path.write_bytes(b"x" * 32)
            os.chmod(path, 0o644)
            with self.assertRaises(PrivateKeyError):
                load_private_key(path)

    def test_wrong_key_length_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._private_directory(root) / "run.key"
            path.write_bytes(b"too short")
            os.chmod(path, 0o600)
            with self.assertRaises(PrivateKeyError):
                load_private_key(path)

    def test_broad_parent_permissions_fail_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "broad"
            directory.mkdir(mode=0o750)
            os.chmod(directory, 0o750)
            path = directory / "run.key"
            with self.assertRaises(PrivateKeyError):
                load_or_create_private_key(path)
            self.assertFalse(path.exists())


class StatelessSeedTests(unittest.TestCase):
    KEY = bytes(range(32))

    def _seed(self, **overrides: object) -> int:
        context = {
            "key": self.KEY,
            "domain": "paper-repro-v1",
            "purpose": "client-sampling",
            "model": "roberta-base",
            "round": 4,
            "client": 2,
            "method_scope": None,
        }
        context.update(overrides)
        return derive_seed(**context)  # type: ignore[arg-type]

    def test_same_key_and_context_are_stable(self) -> None:
        self.assertEqual(self._seed(), self._seed())
        self.assertGreaterEqual(self._seed(), 0)
        self.assertLess(self._seed(), 1 << 63)

    def test_private_key_fingerprint_is_stable_and_non_secret(self) -> None:
        fingerprint = private_key_fingerprint(self.KEY)
        self.assertEqual(fingerprint, private_key_fingerprint(self.KEY))
        self.assertEqual(len(fingerprint), 64)
        self.assertNotIn(self.KEY.hex(), fingerprint)

    def test_purpose_separates_random_streams(self) -> None:
        self.assertNotEqual(self._seed(), self._seed(purpose="gradient-noise"))

    def test_method_scope_controls_sharing(self) -> None:
        shared_a = self._seed(method_scope=None)
        shared_b = self._seed(method_scope=None)
        paper = self._seed(method_scope="paper_dp_lora")
        control = self._seed(method_scope="clip_only_control")
        self.assertEqual(shared_a, shared_b)
        self.assertNotEqual(paper, control)
        self.assertNotEqual(shared_a, paper)


class FingerprintAndStatisticsTests(unittest.TestCase):
    def test_canonical_fingerprint_ignores_object_key_order(self) -> None:
        left = {"model": "bert", "config": {"rank": 8, "alpha": 16}}
        right = {"config": {"alpha": 16, "rank": 8}, "model": "bert"}
        self.assertEqual(
            canonical_json_fingerprint(left), canonical_json_fingerprint(right)
        )
        self.assertNotEqual(
            canonical_json_fingerprint(left),
            canonical_json_fingerprint({"model": "bert", "config": [8, 16]}),
        )

    def test_int64_index_digest_is_stable_and_order_sensitive(self) -> None:
        self.assertEqual(int64_index_digest([1, 7, 9]), int64_index_digest((1, 7, 9)))
        self.assertNotEqual(int64_index_digest([1, 7, 9]), int64_index_digest([9, 7, 1]))
        with self.assertRaises(OverflowError):
            int64_index_digest([1 << 63])

    def test_safe_ratio_never_returns_non_finite_json_numbers(self) -> None:
        self.assertEqual(safe_ratio(3, 2), 1.5)
        self.assertIsNone(safe_ratio(1, 0))
        self.assertIsNone(safe_ratio(float("inf"), 2))

    def test_safe_quantiles_count_and_exclude_invalid_values(self) -> None:
        summary = safe_quantiles([1, 2, 3, float("nan"), "bad"], (0, 0.5, 1))
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["finite_count"], 3)
        self.assertEqual(summary["non_finite_count"], 2)
        self.assertEqual(summary["quantiles"], {"0": 1.0, "0.5": 2.0, "1": 3.0})

        empty = safe_quantiles([], (0.5,))
        self.assertEqual(empty["quantiles"], {"0.5": None})


if __name__ == "__main__":
    unittest.main()
