from __future__ import annotations

import errno
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper_repro import durable_io
from paper_repro.full_slaclip_campaign import atomic_csv
from paper_repro.train_federated import atomic_json, atomic_jsonl


class DurableIOTests(unittest.TestCase):
    def test_atomic_write_retries_transient_file_fsync_with_fresh_temporary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            real_fsync = os.fsync
            temporary_names: list[str] = []
            injected = False

            def flaky_fsync(descriptor: int) -> None:
                nonlocal injected
                metadata = os.fstat(descriptor)
                if stat.S_ISREG(metadata.st_mode):
                    temporary_names.append(
                        Path(os.readlink(f"/proc/self/fd/{descriptor}")).name
                    )
                    if not injected:
                        injected = True
                        raise FileNotFoundError(errno.ENOENT, "simulated GPFS miss")
                real_fsync(descriptor)

            with mock.patch(
                "paper_repro.durable_io.os.fsync", side_effect=flaky_fsync
            ), mock.patch("paper_repro.durable_io.time.sleep") as sleep:
                durable_io.atomic_write_bytes(destination, b'{"ok": true}\n')

            self.assertEqual(destination.read_bytes(), b'{"ok": true}\n')
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(len(temporary_names), 2)
            self.assertNotEqual(temporary_names[0], temporary_names[1])
            self.assertEqual(list(Path(root).glob(f".{destination.name}.*.tmp")), [])

    def test_atomic_write_retries_directory_fsync_after_replace(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            real_fsync = os.fsync
            injected = False

            def flaky_fsync(descriptor: int) -> None:
                nonlocal injected
                if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not injected:
                    injected = True
                    raise FileNotFoundError(
                        errno.ENOENT, "simulated directory fsync miss"
                    )
                real_fsync(descriptor)

            with mock.patch(
                "paper_repro.durable_io.os.fsync", side_effect=flaky_fsync
            ), mock.patch("paper_repro.durable_io.time.sleep") as sleep:
                durable_io.atomic_write_bytes(destination, b"durable\n")

            self.assertEqual(destination.read_bytes(), b"durable\n")
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(list(Path(root).glob(f".{destination.name}.*.tmp")), [])

    def test_atomic_write_restarts_when_temporary_disappears_after_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            real_fsync = os.fsync
            removed_temporary: Path | None = None

            def unlink_after_first_file_fsync(descriptor: int) -> None:
                nonlocal removed_temporary
                real_fsync(descriptor)
                if (
                    removed_temporary is None
                    and stat.S_ISREG(os.fstat(descriptor).st_mode)
                ):
                    removed_temporary = Path(
                        os.readlink(f"/proc/self/fd/{descriptor}")
                    )
                    removed_temporary.unlink()

            with mock.patch(
                "paper_repro.durable_io.os.fsync",
                side_effect=unlink_after_first_file_fsync,
            ), mock.patch("paper_repro.durable_io.time.sleep") as sleep:
                durable_io.atomic_write_bytes(destination, b"recreated\n")

            self.assertIsNotNone(removed_temporary)
            self.assertEqual(destination.read_bytes(), b"recreated\n")
            self.assertEqual(sleep.call_count, 1)
            assert removed_temporary is not None
            self.assertNotEqual(destination.name, removed_temporary.name)
            self.assertEqual(list(Path(root).glob(f".{destination.name}.*.tmp")), [])

    def test_directory_fsync_exhaustion_propagates_without_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            real_fsync = os.fsync

            def fail_directory_fsync(descriptor: int) -> None:
                if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise FileNotFoundError(
                        errno.ENOENT, "persistent directory fsync miss"
                    )
                real_fsync(descriptor)

            with mock.patch(
                "paper_repro.durable_io.os.fsync", side_effect=fail_directory_fsync
            ), mock.patch("paper_repro.durable_io.time.sleep") as sleep:
                with self.assertRaises(FileNotFoundError):
                    durable_io.atomic_write_bytes(destination, b"visible-not-acked\n")

            self.assertEqual(sleep.call_count, durable_io.RETRY_ATTEMPTS - 1)
            self.assertEqual(destination.read_bytes(), b"visible-not-acked\n")
            self.assertEqual(list(Path(root).glob(f".{destination.name}.*.tmp")), [])

    def test_transient_failure_exhausts_exactly_four_attempts_and_preserves_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            destination.write_bytes(b"old\n")
            destination.chmod(0o600)
            failure = FileNotFoundError(errno.ENOENT, "persistent GPFS miss")

            with mock.patch(
                "paper_repro.durable_io.os.fsync", side_effect=failure
            ) as fsync, mock.patch(
                "paper_repro.durable_io.time.sleep"
            ) as sleep:
                with self.assertRaises(FileNotFoundError):
                    durable_io.atomic_write_bytes(destination, b"new\n")

            self.assertEqual(fsync.call_count, durable_io.RETRY_ATTEMPTS)
            self.assertEqual(sleep.call_count, durable_io.RETRY_ATTEMPTS - 1)
            self.assertEqual(destination.read_bytes(), b"old\n")
            self.assertEqual(list(Path(root).glob(f".{destination.name}.*.tmp")), [])

    def test_non_retryable_durability_error_fails_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "result.json"
            destination.write_bytes(b"old\n")
            destination.chmod(0o600)
            failure = OSError(errno.EIO, "simulated permanent I/O failure")

            with mock.patch(
                "paper_repro.durable_io.os.fsync", side_effect=failure
            ) as fsync, mock.patch(
                "paper_repro.durable_io.time.sleep"
            ) as sleep:
                with self.assertRaises(OSError) as raised:
                    durable_io.atomic_write_bytes(destination, b"new\n")

            self.assertEqual(raised.exception.errno, errno.EIO)
            self.assertEqual(fsync.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"old\n")

    def test_temporary_fd_and_path_identity_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            temporary = Path(root) / ".result.tmp"
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            try:
                os.fchmod(descriptor, 0o600)
                os.write(descriptor, b"original")
                temporary.unlink()
                temporary.write_bytes(b"replacement")
                temporary.chmod(0o600)
                with self.assertRaisesRegex(RuntimeError, "changed identity"):
                    durable_io._validate_temporary_identity(
                        descriptor,
                        temporary,
                        mode=0o600,
                        expected_size=len(b"original"),
                    )
            finally:
                os.close(descriptor)

    def test_training_json_and_campaign_csv_serialization_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            json_path = directory / "value.json"
            jsonl_path = directory / "values.jsonl"
            csv_path = directory / "values.csv"

            atomic_json(json_path, {"b": 2, "a": 1})
            atomic_jsonl(jsonl_path, [{"b": 2, "a": 1}, {"a": 3}])
            atomic_csv(csv_path, [{"a": "x", "b": 1}], ("a", "b"))

            self.assertEqual(
                json_path.read_text(encoding="utf-8"),
                '{\n  "a": 1,\n  "b": 2\n}\n',
            )
            self.assertEqual(
                jsonl_path.read_text(encoding="utf-8"),
                '{"a": 1, "b": 2}\n{"a": 3}\n',
            )
            self.assertEqual(
                csv_path.read_bytes(),
                b"a,b\r\nx,1\r\n",
            )


if __name__ == "__main__":
    unittest.main()
