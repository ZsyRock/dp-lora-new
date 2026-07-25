from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from paper_repro.calibrate_slaclip import (
    CALIBRATION_PRIVACY_CLASS,
    CALIBRATION_REDUCER,
    atomic_create_calibration,
    build_calibration,
    load_and_validate_calibration,
    main,
    round_shard_prefix_sha256,
)
from paper_repro.reproducibility import canonical_json_fingerprint


MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    return path


def private_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def build_baseline_fixture(
    parent: Path,
    counts: dict[str, dict[str, list[int]]],
    *,
    clients_per_round: int = 5,
) -> Path:
    baseline = private_directory(parent / "baseline")
    rounds = len(counts["bert"]["A"])
    if any(
        len(counts[model][group]) != rounds
        for model in MODELS
        for group in GROUPS
    ):
        raise ValueError("fixture count vectors must have equal length")
    effective = {
        "method": "paper_dp_lora",
        "rounds": rounds,
        "num_clients": clients_per_round,
    }
    contract = {
        "schema_version": 2,
        "repository_sha": "a" * 40,
        "method": {"name": "paper_dp_lora"},
        "effective_config": effective,
    }
    run_fingerprint = canonical_json_fingerprint(contract)
    run_config = {
        "schema_version": 2,
        "method": "paper_dp_lora",
        "run_config_fingerprint": run_fingerprint,
        "scientific_contract": contract,
        "effective_config": effective,
        "models": list(MODELS),
    }
    private_json(baseline / "run_config.json", run_config)

    root_models: dict[str, object] = {}
    for model in MODELS:
        model_dir = private_directory(baseline / model)
        diagnostics = private_directory(model_dir / "private_diagnostics")
        rounds_dir = private_directory(diagnostics / "rounds")
        total_counts = {group: 0 for group in GROUPS}
        for round_index in range(1, rounds + 1):
            group_counts = {
                group: counts[model][group][round_index - 1] for group in GROUPS
            }
            if any(
                count < 0 or count > clients_per_round
                for count in group_counts.values()
            ):
                raise ValueError("fixture clipped count is outside client count")
            records = []
            for client in range(clients_per_round):
                records.append(
                    {
                        "method": "paper_dp_lora",
                        "model": model,
                        "round": round_index,
                        "client": client,
                        "gradient_groups": {
                            group: {"clipped": client < group_counts[group]}
                            for group in GROUPS
                        },
                    }
                )
            shard = {
                "schema_version": 2,
                "method": "paper_dp_lora",
                "model": model,
                "round": round_index,
                "client_records": records,
                "round_summary": {
                    "round": round_index,
                    "clients": clients_per_round,
                    **{
                        group: {
                            "clipped_count": group_counts[group],
                            "clipped_fraction": group_counts[group]
                            / clients_per_round,
                        }
                        for group in GROUPS
                    },
                },
            }
            private_json(
                rounds_dir / f"round-{round_index:05d}.json",
                shard,
            )
            for group in GROUPS:
                total_counts[group] += group_counts[group]
        prefix = round_shard_prefix_sha256(rounds_dir, rounds)
        steps = rounds * clients_per_round
        model_summary = {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": "paper_dp_lora",
            "model": model,
            "run_config_fingerprint": run_fingerprint,
            "client_steps": steps,
            "round_shard_prefix_sha256": prefix,
            "clipping": {
                group: {
                    "count": total_counts[group],
                    "fraction": total_counts[group] / steps,
                }
                for group in GROUPS
            },
            "behavior_summary": {
                "groups": {
                    group: {
                        "actual_clipped_count": total_counts[group],
                        "actual_clipped_fraction": total_counts[group] / steps,
                    }
                    for group in GROUPS
                }
            },
        }
        private_json(model_dir / "final_summary.json", model_summary)
        root_models[model] = model_summary

    root_summary = {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": "paper_dp_lora",
        "contains_slaclip": False,
        "run_config_fingerprint": run_fingerprint,
        "models": root_models,
    }
    private_json(baseline / "final_summary.json", root_summary)
    return baseline


class CalibrateSlaClipTests(unittest.TestCase):
    def test_odd_and_even_medians_are_per_model_and_group(self) -> None:
        cases = (
            (
                {
                    "bert": {"A": [0, 2, 4], "B": [0, 0, 0]},
                    "gpt2": {"A": [5, 3, 1], "B": [1, 2, 3]},
                },
                {
                    "bert": {"A": 0.4, "B": 0.0},
                    "gpt2": {"A": 0.6, "B": 0.4},
                },
            ),
            (
                {
                    "bert": {"A": [0, 1, 2, 3], "B": [0, 0, 0, 0]},
                    "gpt2": {"A": [4, 3, 2, 1], "B": [5, 4, 3, 2]},
                },
                {
                    "bert": {"A": 0.3, "B": 0.0},
                    "gpt2": {"A": 0.5, "B": 0.7},
                },
            ),
        )
        for case_index, (counts, expected) in enumerate(cases):
            with self.subTest(round_parity=case_index):
                with tempfile.TemporaryDirectory() as raw_root:
                    parent = Path(raw_root)
                    baseline = build_baseline_fixture(parent, counts)
                    calibration = build_calibration(baseline)
                    self.assertEqual(
                        calibration["privacy_class"], CALIBRATION_PRIVACY_CLASS
                    )
                    self.assertEqual(calibration["reducer"], CALIBRATION_REDUCER)
                    for model in MODELS:
                        for group in GROUPS:
                            group_value = calibration["models"][model]["groups"][group]
                            self.assertAlmostEqual(
                                group_value["target_clip_fraction"],
                                expected[model][group],
                            )
                            self.assertEqual(
                                group_value["round_count"],
                                len(counts[model][group]),
                            )
                    self.assertNotEqual(
                        calibration["models"]["bert"]["groups"]["A"][
                            "round_actual_clipped_fractions_sha256"
                        ],
                        calibration["models"]["gpt2"]["groups"]["A"][
                            "round_actual_clipped_fractions_sha256"
                        ],
                    )

    def test_cli_atomically_creates_private_file_and_verifies_it(self) -> None:
        counts = {
            "bert": {"A": [0, 1, 0], "B": [0, 0, 0]},
            "gpt2": {"A": [1, 1, 1], "B": [2, 2, 2]},
        }
        with tempfile.TemporaryDirectory() as raw_root:
            parent = Path(raw_root)
            baseline = build_baseline_fixture(parent, counts)
            output_parent = private_directory(parent / "calibration")
            output = output_parent / "median.json"
            main(["--baseline-dir", str(baseline), "--output", str(output)])
            self.assertEqual(stat_mode(output), 0o600)
            self.assertEqual(output.stat().st_nlink, 1)
            loaded = load_and_validate_calibration(
                output,
                expected_models=MODELS,
                expected_source_dir=baseline,
            )
            self.assertEqual(loaded["source"]["method"], "paper_dp_lora")
            main(
                [
                    "--baseline-dir",
                    str(baseline),
                    "--output",
                    str(output),
                    "--verify-existing",
                ]
            )
            with self.assertRaises(SystemExit):
                main(["--baseline-dir", str(baseline), "--output", str(output)])

    def test_tampered_or_missing_round_shard_fails_closed(self) -> None:
        counts = {
            "bert": {"A": [0, 1, 2], "B": [2, 1, 0]},
            "gpt2": {"A": [0, 0, 0], "B": [1, 1, 1]},
        }
        for case in ("tampered", "missing"):
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as raw_root:
                    parent = Path(raw_root)
                    baseline = build_baseline_fixture(parent, counts)
                    shard = (
                        baseline
                        / "bert"
                        / "private_diagnostics"
                        / "rounds"
                        / "round-00002.json"
                    )
                    if case == "missing":
                        shard.unlink()
                        expected = "round shard set mismatch"
                    else:
                        payload = json.loads(shard.read_text(encoding="utf-8"))
                        payload["client_records"][0]["gradient_groups"]["A"][
                            "clipped"
                        ] = not payload["client_records"][0]["gradient_groups"]["A"][
                            "clipped"
                        ]
                        private_json(shard, payload)
                        expected = "clipped count mismatch"
                    with self.assertRaisesRegex(RuntimeError, expected):
                        build_calibration(baseline)

    def test_symlinked_source_paths_fail_closed(self) -> None:
        counts = {
            "bert": {"A": [0], "B": [0]},
            "gpt2": {"A": [0], "B": [0]},
        }
        with tempfile.TemporaryDirectory() as raw_root:
            parent = Path(raw_root)
            baseline = build_baseline_fixture(parent, counts)
            linked_baseline = parent / "linked-baseline"
            linked_baseline.symlink_to(baseline, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "not a real directory"):
                build_calibration(linked_baseline)

            shard = (
                baseline
                / "bert"
                / "private_diagnostics"
                / "rounds"
                / "round-00001.json"
            )
            target = parent / "target.json"
            target.write_bytes(shard.read_bytes())
            os.chmod(target, 0o600)
            shard.unlink()
            shard.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "not a real regular file"):
                build_calibration(baseline)

    def test_canonical_storage_alias_is_accepted_but_leaf_symlink_is_not(self) -> None:
        counts = {
            "bert": {"A": [0], "B": [0]},
            "gpt2": {"A": [0], "B": [0]},
        }
        with tempfile.TemporaryDirectory() as raw_root:
            parent = Path(raw_root)
            canonical_parent = private_directory(parent / "canonical")
            baseline = build_baseline_fixture(canonical_parent, counts)
            output_parent = private_directory(canonical_parent / "calibration")
            alias = parent / "storage-alias"
            alias.symlink_to(canonical_parent, target_is_directory=True)

            calibration = build_calibration(alias / baseline.name)
            output = alias / output_parent.name / "median.json"
            atomic_create_calibration(output, calibration)
            loaded = load_and_validate_calibration(
                output,
                expected_source_dir=alias / baseline.name,
            )
            self.assertEqual(
                loaded["source"]["baseline_dir"], str(baseline.resolve())
            )

            leaf_alias = output_parent / "leaf-alias.json"
            leaf_alias.symlink_to(output)
            with self.assertRaisesRegex(RuntimeError, "not a real regular file"):
                load_and_validate_calibration(leaf_alias)

    def test_loader_rejects_source_mismatch_and_calibration_tampering(self) -> None:
        counts = {
            "bert": {"A": [0, 1, 2], "B": [0, 0, 0]},
            "gpt2": {"A": [1, 2, 3], "B": [3, 2, 1]},
        }
        with tempfile.TemporaryDirectory() as raw_root:
            parent = Path(raw_root)
            baseline = build_baseline_fixture(parent, counts)
            output_parent = private_directory(parent / "calibration")
            output = output_parent / "median.json"
            calibration = build_calibration(baseline)
            atomic_create_calibration(output, calibration)

            other_source = private_directory(parent / "other-source")
            with self.assertRaisesRegex(RuntimeError, "source directory mismatch"):
                load_and_validate_calibration(
                    output, expected_source_dir=other_source
                )

            tampered = copy.deepcopy(calibration)
            tampered["models"]["bert"]["groups"]["A"][
                "target_clip_fraction"
            ] = 0.9
            private_json(output, tampered)
            with self.assertRaisesRegex(RuntimeError, "median mismatch"):
                load_and_validate_calibration(output)

            timestamp_tampered = copy.deepcopy(calibration)
            timestamp_tampered["created_at_utc"] = "2099-01-01T00:00:00+00:00"
            private_json(output, timestamp_tampered)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_and_validate_calibration(output)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
