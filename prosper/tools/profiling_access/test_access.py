#!/usr/bin/env python3
"""Unprivileged policy/IO/installation tests. Real sudo access requires profile.py --verify-access."""
import io
import fcntl
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import capture
import install
import profile as client


class PolicyTests(unittest.TestCase):
    def test_allowed_arguments(self):
        for seconds in (1, 2, 10, 60):
            self.assertEqual(capture.seconds_from(["scheduler", str(seconds)]), seconds)

    def test_hostile_arguments(self):
        for args in ([], ["scheduler"], ["scheduler", "0"], ["scheduler", "61"],
                     ["scheduler", "-1"], ["scheduler", "1.5"], ["scheduler", "01"],
                     ["scheduler", "１"], ["scheduler", "1\n"], ["scheduler", "1;id"],
                     ["scheduler", "1", "--", "/bin/sh"], ["scheduler", "1", "-o", "/etc/passwd"],
                     ["stat", "1"], ["record", "1"], ["/bin/sh", "1"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                capture.seconds_from(args)

    def test_identity(self):
        capture.authorize(0, "1000", 1000)
        for values in ((1000, "1000", 1000), (0, "1001", 1000), (0, None, 1000),
                       (0, "0", 0), (0, "1000", None), (0, "01000", 1000)):
            with self.subTest(values=values), self.assertRaises(PermissionError):
                capture.authorize(*values)

    def test_command_is_fixed(self):
        cmd = capture.command_for(2)
        self.assertEqual(cmd[:5], ["/usr/bin/timeout", "--signal=TERM", "--kill-after=2s", "7", "/usr/bin/perf"])
        self.assertEqual(cmd[-3:], ["--", "/usr/bin/sleep", "2"])
        self.assertEqual(cmd[cmd.index("-o") + 1], "-")
        self.assertEqual([cmd[i+1] for i, word in enumerate(cmd) if word == "-e"], list(capture.EVENTS))
        self.assertIn("--no-buildid", cmd)
        self.assertIn("--no-buildid-cache", cmd)

    def test_environment_does_not_inherit_injection(self):
        self.assertNotIn("PYTHONPATH", capture.ENV)
        self.assertNotIn("LD_PRELOAD", capture.ENV)
        self.assertNotIn("HOME", capture.ENV)
        self.assertEqual(capture.ENV["PERF_CONFIG"], "/dev/null")
        self.assertEqual(capture.ENV["PATH"], "/usr/bin:/usr/sbin")

    def test_shebang_is_isolated_python(self):
        self.assertEqual(Path(capture.__file__).read_text().splitlines()[0], "#!/usr/bin/python3 -I")

    def test_trusted_system_binary(self):
        capture.trusted_path("/usr/bin/sleep")

    def test_refusal_needs_correct_cause(self):
        expected = "expected: scheduler SECONDS"
        self.assertTrue(client.rejection_ok({"returncode": 1}, "prosper-perf: " + expected, expected))
        for code, stderr in ((0, "prosper-perf: " + expected), (None, "prosper-perf: " + expected),
                             (1, "sudo: a password is required"), (1, "No such file")):
            self.assertFalse(client.rejection_ok({"returncode": code}, stderr, expected))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_reject_user_controlled_directory(self):
        self.root.chmod(0o777)
        with self.assertRaises(PermissionError):
            capture.trusted_path(self.root, directory=True)
        with self.assertRaises(PermissionError):
            install.protected_directory(self.root)

    def test_lock_symlink_is_refused(self):
        target = self.root / "target"
        target.write_text("unchanged")
        lock = self.root / "lock"
        lock.symlink_to(target)
        with patch.object(capture, "trusted_path"), self.assertRaises(OSError):
            capture.acquire_lock(lock)
        self.assertEqual(target.read_text(), "unchanged")

    def test_lock_contention_and_release(self):
        original = os.fstat
        def as_root(fd):
            st = original(fd)
            return SimpleNamespace(st_uid=0, st_mode=st.st_mode, st_nlink=st.st_nlink)
        with patch.object(capture, "trusted_path"), patch.object(capture.os, "fstat", side_effect=as_root):
            fd = capture.acquire_lock(self.root / "lock")
            try:
                with self.assertRaises(BlockingIOError):
                    capture.acquire_lock(self.root / "lock")
            finally:
                os.close(fd)
            os.close(capture.acquire_lock(self.root / "lock"))

    def test_lock_rejects_permissive_mode_and_hardlinks(self):
        original = os.fstat
        def as_root(fd):
            st = original(fd)
            return SimpleNamespace(st_uid=0, st_mode=st.st_mode, st_nlink=st.st_nlink)
        lock = self.root / "lock"
        lock.touch(mode=0o600)
        with patch.object(capture, "trusted_path"), patch.object(capture.os, "fstat", side_effect=as_root):
            lock.chmod(0o644)
            with self.assertRaises(PermissionError):
                capture.acquire_lock(lock)
            lock.chmod(0o600)
            os.link(lock, self.root / "second-link")
            with self.assertRaises(PermissionError):
                capture.acquire_lock(lock)

    def test_event_counts_reject_diagnostics(self):
        path = self.root / "events"
        path.write_text("sched:sched_switch: \nsched:sched_wakeup: \nerror: sched:sched_wakeup\n")
        self.assertEqual(client.event_counts(path), {"sched:sched_switch": 1, "sched:sched_wakeup": 1})


class StreamTests(unittest.TestCase):
    def test_actual_child_has_clean_environment_and_cwd(self):
        command = [sys.executable, "-c", "import os; print(os.environ['PERF_CONFIG']); print('PYTHONPATH' in os.environ); print(os.getcwd())"]
        with tempfile.TemporaryFile(dir=".") as output, \
                patch.dict(os.environ, {"PERF_CONFIG": "/untrusted", "PYTHONPATH": "/untrusted"}), \
                patch.object(capture, "command_for", return_value=command):
            capture.record(1, output.fileno())
            output.seek(0)
            self.assertEqual(output.read(), b"/dev/null\nFalse\n/\n")

    def test_complete_capture(self):
        with tempfile.TemporaryFile(dir=".") as output, \
                patch.object(capture, "command_for", return_value=[sys.executable, "-c", "print('payload')"]):
            self.assertEqual(capture.record(1, output.fileno()), 8)
            output.seek(0)
            self.assertEqual(output.read(), b"payload\n")

    def test_nonzero_recorder_is_not_success(self):
        with tempfile.TemporaryFile(dir=".") as output, \
                patch.object(capture, "command_for", return_value=[sys.executable, "-c", "print('partial');exit(7)"]):
            with self.assertRaisesRegex(RuntimeError, "exit=7"):
                capture.record(1, output.fileno())

    def test_empty_recorder_is_not_success(self):
        with tempfile.TemporaryFile(dir=".") as output, \
                patch.object(capture, "command_for", return_value=[sys.executable, "-c", "pass"]):
            with self.assertRaisesRegex(RuntimeError, "bytes=0"):
                capture.record(1, output.fileno())

    def test_timeout_cleans_up_recorder(self):
        with tempfile.TemporaryFile(dir=".") as output, \
                patch.object(capture, "command_for", return_value=[sys.executable, "-c", "import time;time.sleep(30)"]), \
                patch.object(capture, "pump", side_effect=TimeoutError("control timeout")):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                capture.record(1, output.fileno())
            self.assertLess(time.monotonic() - start, 3)

    def test_byte_limit(self):
        with tempfile.TemporaryFile(dir=".") as source, tempfile.TemporaryFile(dir=".") as output:
            source.write(b"123456")
            source.seek(0)
            with self.assertRaisesRegex(ValueError, "output limit"):
                capture.pump(source.fileno(), output.fileno(), time.monotonic() + 2, max_bytes=5)
            self.assertTrue(os.get_blocking(output.fileno()))

    def test_blocked_consumer_has_deadline(self):
        reader, writer = os.pipe()
        try:
            with tempfile.TemporaryFile(dir=".") as source:
                source.write(b"x" * 1024 * 1024)
                source.seek(0)
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    capture.pump(source.fileno(), writer, start + 0.1)
                self.assertLess(time.monotonic() - start, 2)
                self.assertTrue(os.get_blocking(writer))
        finally:
            os.close(reader)
            os.close(writer)


class DiagnosticTests(unittest.TestCase):
    def full_pipe(self):
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        self.addCleanup(os.close, writer)
        os.set_blocking(writer, False)
        try:
            while True:
                os.write(writer, b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(writer, True)
        return reader, writer

    def invoke_main(self, arguments, *, stderr, stdout=subprocess.PIPE, lock_path=None):
        # Exercise the actual main/record paths in a separate ordinary-user process. Only
        # privilege checks, the protected lock location and fixed recorder are substituted.
        # The outer fake-backend timeout prevents an old helper leaving an orphan on failure.
        backend = ("import os\ntry: os.write(2, b'backend diagnostic\\n')\n"
                   "except BlockingIOError: pass\nos.write(1, b'payload\\n')")
        code = f"""
import fcntl, os, sys
sys.path.insert(0, {str(Path(capture.__file__).resolve().parent)!r})
import capture
capture.authorize = lambda *args: None
capture.trusted_path = lambda *args, **kwargs: None
def acquire():
    fd = os.open({str(lock_path)!r}, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd
capture.acquire_lock = acquire
capture.command_for = lambda seconds: ['/usr/bin/timeout', '--kill-after=1s', '2',
                                      sys.executable, '-c', {backend!r}]
sys.argv = ['capture.py', *{arguments!r}]
sys.exit(capture.main())
"""
        return subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                              stdout=stdout, stderr=stderr, timeout=4)

    def test_full_stderr_cannot_hold_capture_or_lock(self):
        _, writer = self.full_pipe()
        with tempfile.TemporaryDirectory(dir=".") as directory:
            lock_path = Path(directory).resolve() / "lock"
            result = self.invoke_main(["scheduler", "1"], stderr=writer, lock_path=lock_path)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"payload\n")
            self.assertTrue(os.get_blocking(writer), "restore the caller's shared FD flags")
            with lock_path.open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_full_stderr_does_not_block_refusal(self):
        _, writer = self.full_pipe()
        result = self.invoke_main(["scheduler", "61"], stderr=writer)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertTrue(os.get_blocking(writer))

    def test_full_stdout_does_not_block_help(self):
        _, writer = self.full_pipe()
        result = self.invoke_main(["--help"], stdout=writer, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.get_blocking(writer))

    def test_notice_is_bounded_and_preserves_flags(self):
        with tempfile.TemporaryFile(dir=".") as output:
            capture.notice(output.fileno(), "x" * 10000)
            output.seek(0)
            self.assertEqual(len(output.read()), 4096)
            self.assertTrue(os.get_blocking(output.fileno()))

    def test_broken_diagnostic_pipe_is_disposable(self):
        reader, writer = os.pipe()
        os.close(reader)
        try:
            capture.notice(writer, "discard this diagnostic")
            self.assertTrue(os.get_blocking(writer))
        finally:
            os.close(writer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.helper = self.root / "libexec" / "helper"
        self.rule = self.root / "sudoers" / "rule"
        self.rule.parent.mkdir()
        self.source = Path(capture.__file__)

    def invoke(self, validation=None):
        with patch.object(install, "protected_directory"), \
                patch.object(install.subprocess, "run", side_effect=validation) as check:
            install.install("developer", 1000, self.source, "/usr/sbin/visudo", self.helper, self.rule)
        return check

    def test_login_validation(self):
        lookup = lambda _: SimpleNamespace(pw_uid=1000)
        self.assertEqual(install.validate_username("developer", lookup), 1000)
        for value in ("root ALL=(ALL)", "a\nb", "-bad", "a,b", "ALL", ""):
            with self.assertRaises(ValueError):
                install.validate_username(value, lookup)
        with self.assertRaises(ValueError):
            install.validate_username("root", lambda _: SimpleNamespace(pw_uid=0))

    def test_install_success_and_modes(self):
        check = self.invoke()
        self.assertEqual(check.call_count, 3)
        self.assertIn("ALLOWED_UID = 1000", self.helper.read_text())
        self.assertIn("NOPASSWD: NOSETENV:", self.rule.read_text())
        self.assertEqual(stat.S_IMODE(self.helper.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(self.rule.stat().st_mode), 0o440)
        self.assertEqual(self.helper.stat().st_nlink, 1)
        self.assertEqual(list(self.rule.parent.glob(".prosper-perf-*")), [])

    def test_real_sudoers_parser_positive_and_negative(self):
        visudo = next((p for p in ("/usr/sbin/visudo", "/usr/bin/visudo") if Path(p).is_file()), None)
        self.assertIsNotNone(visudo, "visudo is required for the sudoers syntax control")
        self.rule.write_bytes(install.sudoers_payload("developer"))
        good = subprocess.run([visudo, "-c", "-f", str(self.rule)], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.rule.write_text("this is invalid sudoers (((\n")
        bad = subprocess.run([visudo, "-c", "-f", str(self.rule)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_validation_failure_rolls_back(self):
        for failing_call in (1, 2, 3):
            count = 0
            def validation(*args, **kwargs):
                nonlocal count
                count += 1
                if count == failing_call:
                    raise subprocess.CalledProcessError(1, args[0])
            with self.subTest(failing_call=failing_call), self.assertRaises(subprocess.CalledProcessError):
                self.invoke(validation)
            self.assertFalse(self.helper.exists())
            self.assertFalse(self.rule.exists())
            self.assertEqual(list(self.root.rglob(".prosper-perf-*")), [])

    def test_existing_rule_is_not_overwritten(self):
        self.rule.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.rule.read_text(), "existing")

    def test_dangling_destination_symlink_is_not_followed(self):
        self.rule.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertFalse((self.root / "absent").exists())

    def test_source_must_have_installation_marker(self):
        source = self.root / "bad.py"
        source.write_text("print('not the helper')")
        with self.assertRaises(ValueError):
            install.helper_payload(source, 1000)

    def test_uninstall_revokes_rule_first_and_preserves_unrelated_files(self):
        self.invoke()
        unrelated = self.rule.parent / "unrelated"
        unrelated.write_text("keep")
        with patch.object(install, "managed_file", return_value=True):
            self.assertEqual(install.uninstall(self.helper, self.rule), [self.rule, self.helper])
        self.assertFalse(self.rule.exists())
        self.assertFalse(self.helper.exists())
        self.assertEqual(unrelated.read_text(), "keep")

    def test_unknown_installation_is_refused(self):
        self.rule.write_text("unrecognized")
        with patch.object(install, "protected_directory"), self.assertRaises(PermissionError):
            install.managed_file(self.rule)
        self.assertEqual(self.rule.read_text(), "unrecognized")

    def test_uninstall_validates_both_before_deleting(self):
        self.invoke()
        def validate(path):
            if path == self.helper:
                raise PermissionError("unrecognized helper")
            return True
        with patch.object(install, "managed_file", side_effect=validate), self.assertRaises(PermissionError):
            install.uninstall(self.helper, self.rule)
        self.assertTrue(self.rule.exists())
        self.assertTrue(self.helper.exists())


class ClientTests(unittest.TestCase):
    def invoke(self, capture_rc=0, decode_rc=0, events=True, verify=False, bad_refusal=False, stale=False):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            output = Path(directory).resolve() / "output"
            if stale:
                output.mkdir()
            def run(command, destination, name, timeout):
                code, stderr, data = 0, "", b""
                if name == "capture":
                    code, data = capture_rc, b"capture fixture"
                elif name == "events":
                    code = decode_rc
                    if events:
                        data = b"sched:sched_switch: \nsched:sched_wakeup: \n"
                else:
                    code = 1
                    reason = "capture duration exceeds 60 seconds" if name == "reject-duration" else "expected: scheduler SECONDS"
                    stderr = "sudo: a password is required" if bad_refusal else "prosper-perf: " + reason
                (destination / (name + ".out")).write_bytes(data)
                (destination / (name + ".err")).write_text(stderr)
                return {"returncode": code, "command": command}
            argv = ["profile.py", "--seconds", "2", "--output", str(output)]
            if verify:
                argv.append("--verify-access")
            old_mask = os.umask(0o077)
            try:
                with patch("sys.argv", argv), patch.object(client.os, "geteuid", return_value=1000), \
                        patch.object(client, "run", side_effect=run) as execute, \
                        patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                    try:
                        status = client.main()
                    except SystemExit as exc:
                        status = exc.code
                return status, execute.call_count
            finally:
                os.umask(old_mask)

    def test_positive_capture_and_refusals(self):
        self.assertEqual(self.invoke(verify=True), (0, 5))

    def test_unavailable_sudo_does_not_decode(self):
        self.assertEqual(self.invoke(capture_rc=1, verify=True), (1, 1))

    def test_empty_decode_fails(self):
        self.assertEqual(self.invoke(events=False), (1, 2))

    def test_failed_decode_cannot_certify_samples(self):
        self.assertEqual(self.invoke(decode_rc=1), (1, 2))

    def test_unavailable_helper_does_not_certify_refusals(self):
        self.assertEqual(self.invoke(verify=True, bad_refusal=True), (1, 5))

    def test_stale_directory_rejected_before_capture(self):
        self.assertEqual(self.invoke(stale=True), (2, 0))


if __name__ == "__main__":
    unittest.main()
