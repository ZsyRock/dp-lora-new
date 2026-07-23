from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from paper_repro.checkpointing import (
    archive_round_shards_after,
    load_latest_checkpoint,
    write_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def _private_rounds(root: str) -> Path:
        diagnostics = Path(root) / "private_diagnostics"
        diagnostics.mkdir(mode=0o700)
        rounds = diagnostics / "rounds"
        rounds.mkdir(mode=0o700)
        return rounds

    @staticmethod
    def _write_private_shard(rounds: Path, round_index: int) -> Path:
        shard = rounds / f"round-{round_index:05d}.json"
        shard.write_text("{}\n", encoding="utf-8")
        shard.chmod(0o600)
        return shard

    def test_round_trip_is_pickle_free_and_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_root = Path(root) / "checkpoints"
            tensors = {
                "layer.lora_A.default.weight": torch.arange(6).reshape(2, 3).float(),
                "layer.lora_B.default.weight": torch.ones(3, 2),
            }
            destination = write_checkpoint(
                checkpoint_root,
                completed_round=10,
                tensors=tensors,
                trainer_state={"client_steps": 50},
                config_fingerprint="a" * 64,
            )
            self.assertTrue((destination / "adapter_state.safetensors").is_file())
            self.assertTrue((destination / "trainer_state.json").is_file())
            self.assertFalse(any(destination.glob("*.pt")))

            loaded = load_latest_checkpoint(
                checkpoint_root, expected_config_fingerprint="a" * 64
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.completed_round, 10)
            self.assertEqual(loaded.trainer_state["client_steps"], 50)
            for name, expected in tensors.items():
                self.assertTrue(torch.equal(loaded.tensors[name], expected))

    def test_fingerprint_mismatch_and_duplicate_round_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_root = Path(root) / "checkpoints"
            arguments = {
                "checkpoint_root": checkpoint_root,
                "completed_round": 1,
                "tensors": {"x": torch.ones(1)},
                "trainer_state": {},
                "config_fingerprint": "b" * 64,
            }
            write_checkpoint(**arguments)
            with self.assertRaises(FileExistsError):
                write_checkpoint(**arguments)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_latest_checkpoint(
                    checkpoint_root, expected_config_fingerprint="c" * 64
                )

    def test_payload_corruption_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_root = Path(root) / "checkpoints"
            destination = write_checkpoint(
                checkpoint_root,
                completed_round=2,
                tensors={"x": torch.ones(1)},
                trainer_state={},
                config_fingerprint="d" * 64,
            )
            state_path = destination / "trainer_state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["completed_round"] = 999
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                load_latest_checkpoint(
                    checkpoint_root, expected_config_fingerprint="d" * 64
                )

    def test_uncheckpointed_round_shards_are_archived_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            for round_index in range(1, 5):
                self._write_private_shard(rounds, round_index)
            moved = archive_round_shards_after(rounds, completed_round=2)
            self.assertEqual([path.name for path in moved], [
                "round-00003.json",
                "round-00004.json",
            ])
            self.assertTrue((rounds / "round-00001.json").is_file())
            self.assertTrue((rounds / "round-00002.json").is_file())
            self.assertFalse((rounds / "round-00003.json").exists())
            self.assertTrue(all(path.is_file() for path in moved))
            self.assertTrue(
                all(path.read_text(encoding="utf-8") == "{}\n" for path in moved)
            )
            archive = moved[0].parent
            superseded = archive.parent
            self.assertEqual(stat.S_IMODE(superseded.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o700)
            self.assertTrue(
                all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in moved)
            )

    def test_archive_rejects_symlinked_or_broad_private_directories(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            real_rounds = Path(root) / "real-rounds"
            real_rounds.mkdir(mode=0o700)
            linked_rounds = Path(root) / "linked-rounds"
            linked_rounds.symlink_to(real_rounds, target_is_directory=True)
            with self.assertRaisesRegex(
                RuntimeError, "symlink component|real directory"
            ):
                archive_round_shards_after(linked_rounds, completed_round=0)

        with tempfile.TemporaryDirectory() as root:
            real_diagnostics = Path(root) / "real-diagnostics"
            real_diagnostics.mkdir(mode=0o700)
            rounds = real_diagnostics / "rounds"
            rounds.mkdir(mode=0o700)
            linked_diagnostics = Path(root) / "linked-diagnostics"
            linked_diagnostics.symlink_to(
                real_diagnostics, target_is_directory=True
            )
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                archive_round_shards_after(
                    linked_diagnostics / "rounds", completed_round=0
                )

        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            rounds.chmod(0o750)
            with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
                archive_round_shards_after(rounds, completed_round=0)

        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            self._write_private_shard(rounds, 1)
            real_superseded = Path(root) / "real-superseded"
            real_superseded.mkdir(mode=0o700)
            (rounds.parent / "superseded").symlink_to(
                real_superseded, target_is_directory=True
            )
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                archive_round_shards_after(rounds, completed_round=0)

        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            self._write_private_shard(rounds, 1)
            superseded = rounds.parent / "superseded"
            superseded.mkdir(mode=0o700)
            superseded.chmod(0o750)
            with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
                archive_round_shards_after(rounds, completed_round=0)

    def test_archive_is_durable_before_source_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            shard = self._write_private_shard(rounds, 1)
            real_unlink = os.unlink

            def fail_source_cleanup(path: str, *, dir_fd: int | None = None) -> None:
                if path == shard.name and dir_fd is not None:
                    raise OSError("simulated source cleanup failure")
                real_unlink(path, dir_fd=dir_fd)

            with mock.patch(
                "paper_repro.checkpointing.os.unlink",
                side_effect=fail_source_cleanup,
            ):
                with self.assertRaisesRegex(OSError, "simulated source cleanup failure"):
                    archive_round_shards_after(rounds, completed_round=0)

            self.assertTrue(shard.is_file())
            archives = list((rounds.parent / "superseded").glob("after-round-*"))
            self.assertEqual(len(archives), 1)
            archived = archives[0] / shard.name
            self.assertEqual(archived.read_text(encoding="utf-8"), "{}\n")
            self.assertEqual(stat.S_IMODE(archived.stat().st_mode), 0o600)

    def test_archive_rejects_symlinked_or_broad_round_shards(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            target = Path(root) / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            target.chmod(0o600)
            (rounds / "round-00001.json").symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "real regular file"):
                archive_round_shards_after(rounds, completed_round=0)

        with tempfile.TemporaryDirectory() as root:
            rounds = self._private_rounds(root)
            shard = self._write_private_shard(rounds, 1)
            shard.chmod(0o640)
            with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
                archive_round_shards_after(rounds, completed_round=0)

    def test_durable_orphan_is_promoted_past_stale_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_root = Path(root) / "checkpoints"
            write_checkpoint(
                checkpoint_root,
                completed_round=1,
                tensors={"x": torch.tensor([1.0])},
                trainer_state={"marker": "one"},
                config_fingerprint="e" * 64,
            )
            stale_pointer = (checkpoint_root / "latest.json").read_bytes()
            write_checkpoint(
                checkpoint_root,
                completed_round=2,
                tensors={"x": torch.tensor([2.0])},
                trainer_state={"marker": "two"},
                config_fingerprint="e" * 64,
            )
            # Simulate a crash window in which round 2 is durable but the
            # latest pointer still names round 1.
            (checkpoint_root / "latest.json").write_bytes(stale_pointer)
            loaded = load_latest_checkpoint(
                checkpoint_root, expected_config_fingerprint="e" * 64
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.completed_round, 2)
            self.assertEqual(loaded.trainer_state["marker"], "two")
            self.assertTrue(torch.equal(loaded.tensors["x"], torch.tensor([2.0])))
            pointer = json.loads(
                (checkpoint_root / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["completed_round"], 2)

    def test_broad_or_symlinked_private_checkpoint_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_root = Path(root) / "checkpoints"
            write_checkpoint(
                checkpoint_root,
                completed_round=1,
                tensors={"x": torch.ones(1)},
                trainer_state={},
                config_fingerprint="f" * 64,
            )
            checkpoint_root.chmod(0o750)
            with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
                load_latest_checkpoint(
                    checkpoint_root, expected_config_fingerprint="f" * 64
                )


if __name__ == "__main__":
    unittest.main()
