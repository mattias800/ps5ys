#!/usr/bin/python3 -I
# prosper-perf-managed-v1
"""Restricted host scheduler recording; installed by install.py, never sudo the repo copy.

Usage: sudo -n /usr/local/libexec/prosper-perf-capture scheduler SECONDS > capture.data
SECONDS is a canonical integer from 1 through 60. Stdout is binary perf pipe-format data.
Nonzero exit means the output is incomplete/unusable. No output paths or extra perf arguments.
"""
import fcntl
import os
from pathlib import Path
import re
import resource
import select
import signal
import stat
import subprocess
import sys
import time

ALLOWED_UID = None  # installer replaces this
LOCK = Path("/run/prosper-perf-capture.lock")
MAX_BYTES = 256 * 1024 * 1024
EVENTS = ("sched:sched_switch", "sched:sched_wakeup", "sched:sched_wakeup_new",
          "sched:sched_migrate_task")
ENV = {"PATH": "/usr/bin:/usr/sbin", "LANG": "C", "LC_ALL": "C",
       "PERF_CONFIG": "/dev/null", "PERF_CONFIG_NOSYSTEM": "1",
       "PERF_CONFIG_NOGLOBAL": "1", "DEBUGINFOD_URLS": ""}


def seconds_from(args):
    if len(args) != 2 or args[0] != "scheduler" or not re.fullmatch(r"[1-9][0-9]?", args[1]):
        raise ValueError("expected: scheduler SECONDS (1..60); no other options are allowed")
    seconds = int(args[1])
    if seconds > 60:
        raise ValueError("capture duration exceeds 60 seconds")
    return seconds


def authorize(euid, sudo_uid, allowed_uid):
    if euid != 0 or not isinstance(allowed_uid, int) or allowed_uid <= 0:
        raise PermissionError("use the installed helper through its restricted sudo rule")
    if sudo_uid != str(allowed_uid):
        raise PermissionError("caller is not the account authorized at installation")


def trusted_path(path, directory=False):
    """Allow system symlinks such as /usr/local -> /var/usrlocal; trust every resolved ancestor."""
    resolved = Path(path).resolve(strict=True)
    for item in [resolved, *resolved.parents]:
        st = item.stat()
        if st.st_uid != 0 or st.st_mode & 0o022:
            raise PermissionError(f"not root-owned and protected against writes: {item}")
    st = resolved.stat()
    if directory:
        if not stat.S_ISDIR(st.st_mode):
            raise PermissionError(f"not a directory: {resolved}")
    elif not stat.S_ISREG(st.st_mode) or not st.st_mode & 0o111:
        raise PermissionError(f"not a regular executable: {resolved}")
    return resolved


def command_for(seconds):
    # Outer timeout also bounds perf if the supervisor dies. No caller-provided executable.
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=2s", str(seconds + 5),
               "/usr/bin/perf", "record", "-a", "-c", "1", "-m", "256",
               "--no-buildid", "--no-buildid-cache", "-o", "-"]
    for event in EVENTS:
        command += ["-e", event]
    return command + ["--", "/usr/bin/sleep", str(seconds)]


def acquire_lock(path=LOCK):
    trusted_path(path.parent, directory=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1 or st.st_mode & 0o077:
            raise PermissionError("unsafe profiling lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)


def pump(source_fd, output_fd, deadline, max_bytes=MAX_BYTES):
    """Bound memory, bytes and wall time even when a pipe consumer stops reading."""
    was_blocking = os.get_blocking(output_fd)
    os.set_blocking(output_fd, False)
    pending = b""
    total = 0
    eof = False
    try:
        while not eof or pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("capture/output deadline exceeded")
            readable, writable, _ = select.select(
                [] if eof or pending else [source_fd], [output_fd] if pending else [], [],
                min(remaining, 0.2))
            if readable:
                pending = os.read(source_fd, 65536)
                eof = not pending
                total += len(pending)
                if total > max_bytes:
                    raise ValueError("capture exceeds the 256 MiB output limit")
            if writable:
                try:
                    pending = pending[os.write(output_fd, pending):]
                except BlockingIOError:
                    pass
        return total
    finally:
        os.set_blocking(output_fd, was_blocking)


def record(seconds, output_fd):
    proc = subprocess.Popen(command_for(seconds), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=None, cwd="/", env=ENV, close_fds=True,
                            start_new_session=True, preexec_fn=child_limits)
    try:
        size = pump(proc.stdout.fileno(), output_fd, time.monotonic() + seconds + 10)
        code = proc.wait(timeout=2)
        if code != 0 or size == 0:
            raise RuntimeError(f"perf failed (exit={code}, bytes={size}); discard partial output")
        return size
    finally:
        # Keep an unreaped child PID until signaling, so its process-group ID cannot be reused.
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        proc.stdout.close()


def interrupted(signum, frame):
    raise InterruptedError(f"capture interrupted by signal {signum}")


def notice(fd, message):
    """Best-effort, one bounded write: a diagnostic consumer must never hold the lock."""
    blocking = None
    try:
        blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        os.write(fd, (message + "\n").encode()[:4096])
    except OSError:
        pass  # Closed or backpressured diagnostics are deliberately disposable.
    finally:
        if blocking is not None:
            try:
                os.set_blocking(fd, blocking)
            except OSError:
                pass


def main():
    if sys.argv[1:] == ["--help"]:
        notice(1, __doc__)
        return 0
    lock = None
    stderr_blocking = None
    try:
        # perf/timeout inherit stderr too. Make it nonblocking for their entire lifetime,
        # not only for our final diagnostic, and never use buffered Python stderr writes.
        stderr_blocking = os.get_blocking(2)
        os.set_blocking(2, False)
        seconds = seconds_from(sys.argv[1:])
        authorize(os.geteuid(), os.environ.get("SUDO_UID"), ALLOWED_UID)
        if sys.stdout.isatty():
            raise ValueError("redirect binary stdout to a file or pipe before capturing")
        for executable in ("/usr/bin/timeout", "/usr/bin/perf", "/usr/bin/sleep"):
            trusted_path(executable)
        lock = acquire_lock()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        size = record(seconds, sys.stdout.fileno())
        notice(2, f"prosper-perf: completed ({size} bytes); decode and check events before attribution")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        notice(2, f"prosper-perf: {exc}")
        return 1
    finally:
        if lock is not None:
            os.close(lock)
        if stderr_blocking is not None:
            try:
                os.set_blocking(2, stderr_blocking)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
