#!/usr/bin/env python3
"""A zero-exit recorder must not certify an empty, failed, or unrelated decode."""
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import doctor
import renderdoc_inspect


class SampleVerdictTests(unittest.TestCase):
    def verdict(self, text, record_rc=0, decode_rc=0):
        return doctor.sample_verdict(
            {"returncode": record_rc}, {"returncode": decode_rc, "stdout": text},
            ["sched:sched_switch", "sched:sched_wakeup"])

    def test_both_events_required(self):
        result = self.verdict("sched:sched_switch:\nsched:sched_wakeup:\nsched:sched_switch:\n")
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["samples"], {"sched:sched_switch": 2, "sched:sched_wakeup": 1})
        self.assertEqual(self.verdict("sched:sched_switch:\n")["status"], "UNAVAILABLE")

    def test_empty_success_is_unavailable(self):
        self.assertEqual(self.verdict("")["status"], "UNAVAILABLE")

    def test_error_mentions_are_not_samples(self):
        self.assertEqual(self.verdict(
            "event syntax error: sched:sched_switch\nmissing sched:sched_wakeup\n"
        )["status"], "UNAVAILABLE")

    def test_failed_commands_do_not_certify_stale_samples(self):
        text = "sched:sched_switch:\nsched:sched_wakeup:\n"
        for rec, dec in ((1, 0), (0, 1), (None, 0)):
            self.assertEqual(self.verdict(text, rec, dec)["status"], "UNAVAILABLE")

    def test_user_and_kernel_events_stay_distinct(self):
        self.assertEqual(doctor.events_in("cpu-clock:u:\ncpu-clock:k:\n"),
                         {"cpu-clock:u": 1, "cpu-clock:k": 1})

    def test_recorded_scheduler_populations(self):
        # Counts/format from the privileged child-only and system-wide controls.
        self.assertEqual(self.verdict("sched:sched_switch: \n" * 308)["status"], "UNAVAILABLE")
        result = self.verdict("sched:sched_switch: \n" * 6908 + "sched:sched_wakeup: \n" * 3434)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["samples"], {"sched:sched_switch": 6908, "sched:sched_wakeup": 3434})


class ProbeScopeTests(unittest.TestCase):
    def test_scheduler_command_is_system_wide_and_bounded(self):
        response = {"returncode": 0, "stdout": "sched:sched_switch:\nsched:sched_wakeup:\n"}
        with patch.object(doctor.shutil, "which", return_value="perf"), \
                patch.object(doctor, "run", return_value=response) as run:
            result = doctor.perf_probe(Path("evidence"), "scheduler")
        self.assertEqual(run.call_args_list[0].args[0], [
            "perf", "record", "-q", "-o", str(Path("evidence/scheduler.data")),
            "-a", "-c", "1", "-e", "sched:sched_switch", "-e", "sched:sched_wakeup",
            "--", "sleep", "2"])
        self.assertEqual(result["status"], "READY")
        self.assertIn("including other processes", result["scope"])

    def test_cpu_recording_does_not_expand_scope(self):
        for mode, event in (("perf", "cpu-clock:u"), ("perf-kernel", "cpu-clock:k")):
            response = {"returncode": 0, "stdout": event + ":\n"}
            with patch.object(doctor.shutil, "which", return_value="perf"), \
                    patch.object(doctor, "run", return_value=response) as run:
                result = doctor.perf_probe(Path("evidence"), mode)
            command = run.call_args_list[0].args[0]
            self.assertNotIn("-a", command)
            self.assertIn(event, command)
            self.assertEqual(command[command.index("--") + 1:],
                             [doctor.sys.executable, "-c", doctor.CPU_CONTROL])
            self.assertEqual(result["status"], "READY")

    def test_kernel_denial_does_not_condemn_user_space_sampling(self):
        # perf_event_paranoid=2 denies cpu-clock:k and permits cpu-clock:u. Each probe answers
        # for the capability it names; a shared verdict reported "perf is unavailable" on a host
        # where the sampling these workflows actually run was fine.
        def denied_kernel(command, directory, name, timeout=15):
            if any("kernel" in part for part in command):
                return {"returncode": 1, "stdout": "",
                        "stderr": "perf_event_open: Permission denied"}
            return {"returncode": 0, "stdout": "cpu-clock:u:\n"}

        with patch.object(doctor.shutil, "which", return_value="perf"), \
                patch.object(doctor, "run", side_effect=denied_kernel):
            user = doctor.perf_probe(Path("evidence"), "perf")
            kernel = doctor.perf_probe(Path("evidence"), "perf-kernel")
        self.assertEqual(user["status"], "READY")
        self.assertEqual(kernel["status"], "UNAVAILABLE")
        self.assertIn("separate capability", user["scope"])

    def invoke(self, probes):
        with tempfile.TemporaryDirectory(dir=".") as temp:
            output = Path(temp) / "report"
            argv = ["doctor.py", *probes, "--output", str(output), "--json"]
            with patch("sys.argv", argv), patch.object(doctor, "inventory", return_value={}), \
                    patch.object(doctor, "perf_probe", return_value={"status": "READY"}) as probe, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    status = doctor.main()
                except SystemExit as exc:
                    status = exc.code
            self.modes = [call.args[1] for call in probe.call_args_list]
            return status, probe.call_count, output.exists()

    def test_scheduler_requires_explicit_scope_consent(self):
        self.assertEqual(self.invoke(["--probe", "scheduler"]), (2, 0, False))

    def test_unused_scope_consent_is_rejected(self):
        self.assertEqual(self.invoke(["--probe", "perf", "--system-wide-scheduler"]), (2, 0, False))

    def test_scheduler_with_consent_runs(self):
        self.assertEqual(self.invoke(["--probe", "scheduler", "--system-wide-scheduler"]), (0, 1, True))
        # The scope argument IS the consent mechanism, so pin which probe reaches the system-wide
        # path. Asserting only call_count stays green with the mapping inverted, which would put
        # -a behind --probe perf and the child-only control behind the consent flag.
        self.assertEqual(self.modes, ["scheduler"])

    def test_each_requested_capability_reaches_its_own_probe(self):
        self.assertEqual(self.invoke(["--probe", "perf", "--probe", "perf-kernel"]), (0, 2, True))
        self.assertEqual(self.modes, ["perf", "perf-kernel"])


class ReplayWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.capture = self.root / "control.rdc"
        self.capture.write_bytes(b"fixture, not a real RDC")
        self.output = self.root / "replay"

    def invoke(self, report=None, returncode=0):
        def replay(*args, **kwargs):
            if report is not None:
                (self.output / "replay.json").write_text(report)
            return subprocess.CompletedProcess(args[0], returncode, "", "")

        argv = ["renderdoc_inspect.py", str(self.capture), "--output", str(self.output)]
        with patch("sys.argv", argv), patch("shutil.which", return_value="qrenderdoc"), \
                patch("subprocess.run", side_effect=replay) as process, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                status = renderdoc_inspect.main()
            except SystemExit as exc:
                status = exc.code
        return status, process.call_count

    def test_success_requires_report(self):
        self.assertEqual(self.invoke(json.dumps({"status": "REPLAYED"})), (0, 1))

    def test_empty_success_fails(self):
        self.assertEqual(self.invoke(), (1, 1))

    def test_failed_process_cannot_certify_report(self):
        self.assertEqual(self.invoke(json.dumps({"status": "REPLAYED"}), 1), (1, 1))

    def test_malformed_report_fails(self):
        self.assertEqual(self.invoke("not JSON"), (1, 1))

    def test_failed_report_fails(self):
        self.assertEqual(self.invoke(json.dumps({"status": "FAILED"})), (1, 1))

    def test_stale_output_is_rejected_before_replay(self):
        self.output.mkdir()
        (self.output / "replay.json").write_text('{"status":"REPLAYED"}')
        self.assertEqual(self.invoke(), (2, 0))

    def test_missing_capture_is_rejected_before_replay(self):
        self.capture.unlink()
        self.assertEqual(self.invoke(), (2, 0))


class IndexControlTests(unittest.TestCase):
    def check(self, data, draws=5, copies=1):
        buffers = [SimpleNamespace(resourceId=i, length=len(data)) for i in range(copies)]
        return renderdoc_inspect.index_control_matches([None] * draws, buffers,
                                                       lambda *_: data)

    def expected(self):
        # Independent byte construction, not the production struct.pack expression.
        return b"".join(i.to_bytes(4, "little") for i in range(6)) + b"\xff" * 1000

    def test_exact_contents_pass(self):
        self.assertEqual(self.check(self.expected()), ["0"])

    def test_zero_buffer_is_not_a_control(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=0"):
            self.check(bytes(1024))

    def test_untouched_sentinels_are_checked(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=0"):
            self.check(self.expected()[:-1] + b"\x00")

    def test_draw_count_is_checked(self):
        with self.assertRaisesRegex(RuntimeError, "draws=4"):
            self.check(self.expected(), draws=4)

    def test_ambiguous_buffers_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=2"):
            self.check(self.expected(), copies=2)


if __name__ == "__main__":
    unittest.main()
