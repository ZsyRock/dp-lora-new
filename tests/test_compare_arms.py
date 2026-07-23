from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from paper_repro.compare_arms import EXPECTED_METHODS, build_comparison
from paper_repro.reproducibility import METHOD_SPECS, canonical_json_fingerprint


class CompareArmsTests(unittest.TestCase):
    def _write_arm(self, root: Path, method: str, schedule: str = "shared") -> Path:
        directory = root / method
        directory.mkdir()
        contract = {
            "schema_version": 2,
            "repository_sha": "a" * 40,
            "method": asdict(METHOD_SPECS[method]),
            "effective_config": {"method": method, "rounds": 50, "seed": 42},
            "private_key_commitment": "b" * 64,
            "rng_domain": "pair-1",
        }
        run_config = {
            "method": method,
            "scientific_contract": contract,
            "run_config_fingerprint": canonical_json_fingerprint(contract),
        }
        adapter_path = directory / "bert" / "final_adapter" / "adapter_model.safetensors"
        adapter_path.parent.mkdir(parents=True)
        adapter_path.write_bytes(f"synthetic-{method}".encode())
        adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        adapter_config_path = adapter_path.with_name("adapter_config.json")
        adapter_config_path.write_text("{}", encoding="utf-8")
        adapter_config_sha256 = hashlib.sha256(
            adapter_config_path.read_bytes()
        ).hexdigest()
        behavior = {
            "sample_schedule_sha256": schedule,
            "supervision_schedule_sha256": "same-masks",
        }
        model = {
            "status": "COMPLETED",
            "method": method,
            "adapter_sha256": adapter_sha256,
            "adapter_config_sha256": adapter_config_sha256,
            "adapter_integrity": {"all_finite": True},
            "client_partition_sha256": ["client-0", "client-1"],
            "client_steps": 100,
            "clipping": {
                "any_group": {
                    "count": 0 if method == "no_dp_lora_control" else 3,
                    "fraction": 0.0 if method == "no_dp_lora_control" else 0.03,
                }
            },
            "behavior_summary": behavior,
            "evaluations": [
                {"round": 0, "loss": 4.0},
                {"round": 50, "loss": 3.5},
            ],
            "privacy_accounting": {"status": "NOT_CERTIFIED", "epsilon": None},
        }
        final = {
            "status": "COMPLETED",
            "method": method,
            "paper_result_reproduced": False,
            "run_config_fingerprint": run_config["run_config_fingerprint"],
            "models": {"bert": model},
        }
        (directory / "run_config.json").write_text(
            json.dumps(run_config), encoding="utf-8"
        )
        (directory / "final_summary.json").write_text(
            json.dumps(final), encoding="utf-8"
        )
        (directory / "bert" / "final_summary.json").write_text(
            json.dumps(model), encoding="utf-8"
        )
        return directory

    def test_valid_matched_comparison_is_level_one_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            arms = {
                method: self._write_arm(root, method) for method in EXPECTED_METHODS
            }
            result = build_comparison(arms)
            self.assertEqual(
                result["status"], "VALID_MATCHED_THREE_ARM_LEVEL1_COMPARISON"
            )
            self.assertFalse(result["paper_result_reproduced"])
            self.assertFalse(result["paper_benchmarks_evaluated"])
            self.assertTrue(result["privacy_notice"]["controls_are_non_private"])
            self.assertEqual(
                result["models"]["bert"]["matched_sample_schedule_sha256"],
                "shared",
            )

    def test_schedule_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            arms = {
                method: self._write_arm(
                    root,
                    method,
                    schedule="different" if method == "paper_dp_lora" else "shared",
                )
                for method in EXPECTED_METHODS
            }
            with self.assertRaisesRegex(RuntimeError, "sample schedules do not match"):
                build_comparison(arms)


if __name__ == "__main__":
    unittest.main()
