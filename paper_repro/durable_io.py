"""Durable, fail-closed filesystem writes for shared HPC filesystems.

The experiment outputs live on a distributed filesystem.  A small set of
metadata/cache errors can be transient there, so durability operations receive
four bounded attempts.  Every atomic-file retry restarts the complete
create-write-fsync-publish-directory-fsync transaction; no failed ``fsync`` is
ever treated as success.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
import time
import uuid
from pathlib import Path


RETRY_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (0.02, 0.10, 0.50)
RETRYABLE_ERRNOS = frozenset(
    {errno.ENOENT, errno.ESTALE, errno.EAGAIN, errno.EINTR}
)


def _should_retry(error: OSError, attempt: int) -> bool:
    return error.errno in RETRYABLE_ERRNOS and attempt < RETRY_ATTEMPTS


def _report_retry(
    *, operation: str, path: Path | None, error: OSError, attempt: int
) -> None:
    target = os.fspath(path) if path is not None else "<descriptor>"
    print(
        "durable_io_retry "
        f"operation={operation} path={target} errno={error.errno} "
        f"attempt={attempt}/{RETRY_ATTEMPTS}",
        file=sys.stderr,
        flush=True,
    )


def _wait_before_retry(attempt: int) -> None:
    time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])


def fsync_fd(
    descriptor: int, *, path: Path | None = None, operation: str = "fsync"
) -> None:
    """Synchronize an existing descriptor with bounded transient retries."""

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            os.fsync(descriptor)
            return
        except OSError as error:
            if not _should_retry(error, attempt):
                raise
            _report_retry(
                operation=operation, path=path, error=error, attempt=attempt
            )
            _wait_before_retry(attempt)


def _fsync_directory_once(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Synchronize a directory entry with bounded transient retries."""

    path = Path(path)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            _fsync_directory_once(path)
            return
        except OSError as error:
            if not _should_retry(error, attempt):
                raise
            _report_retry(
                operation="fsync_directory",
                path=path,
                error=error,
                attempt=attempt,
            )
            _wait_before_retry(attempt)


def _write_all(descriptor: int, value: bytes) -> None:
    pending = memoryview(value)
    while pending:
        written = os.write(descriptor, pending)
        if written <= 0:
            raise OSError("atomic write made no forward progress")
        pending = pending[written:]


def _validate_temporary_identity(
    descriptor: int, path: Path, *, mode: int, expected_size: int
) -> os.stat_result:
    opened = os.fstat(descriptor)
    linked = path.lstat()
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        raise RuntimeError(f"atomic-write temporary path changed identity: {path}")
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or opened.st_uid != os.getuid()
        or linked.st_uid != os.getuid()
        or opened.st_nlink != 1
        or linked.st_nlink != 1
        or opened.st_size != expected_size
        or linked.st_size != expected_size
        or stat.S_IMODE(opened.st_mode) != mode
        or stat.S_IMODE(linked.st_mode) != mode
    ):
        raise RuntimeError(f"unsafe atomic-write temporary file: {path}")
    return opened


def atomic_write_bytes(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    """Atomically publish bytes after making file and directory state durable.

    Only a narrow set of transient errors is retried.  Retrying the whole
    transaction also handles the ambiguous window where ``replace`` completed
    but the following directory ``fsync`` failed; the same immutable payload is
    simply published again by the process that already owns the run lock.
    """

    path = Path(path)
    payload = bytes(value)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        descriptor: int | None = None
        phase = "create_temporary"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
            )
            os.fchmod(descriptor, mode)
            phase = "write_temporary"
            _write_all(descriptor, payload)
            phase = "fsync_temporary"
            os.fsync(descriptor)
            phase = "validate_temporary"
            temporary_metadata = _validate_temporary_identity(
                descriptor,
                temporary,
                mode=mode,
                expected_size=len(payload),
            )
            closing_descriptor = descriptor
            descriptor = None
            os.close(closing_descriptor)

            phase = "replace_destination"
            os.replace(temporary, path)
            phase = "validate_destination"
            published = path.lstat()
            if (
                not stat.S_ISREG(published.st_mode)
                or published.st_uid != os.getuid()
                or published.st_nlink != 1
                or published.st_size != len(payload)
                or stat.S_IMODE(published.st_mode) != mode
                or (published.st_dev, published.st_ino)
                != (temporary_metadata.st_dev, temporary_metadata.st_ino)
            ):
                raise RuntimeError(f"atomic-write destination changed identity: {path}")
            phase = "fsync_parent_directory"
            _fsync_directory_once(path.parent)
            return
        except OSError as error:
            if not _should_retry(error, attempt):
                raise
            _report_retry(
                operation=phase, path=path, error=error, attempt=attempt
            )
            _wait_before_retry(attempt)
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: Path, value: str, *, mode: int = 0o600, encoding: str = "utf-8"
) -> None:
    atomic_write_bytes(path, value.encode(encoding), mode=mode)
