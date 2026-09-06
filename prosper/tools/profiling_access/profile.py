#!/usr/bin/env python3
"""Capture scheduler events using the installed restricted helper; never run this client as root.

Usage: python3 profile.py --seconds 10 --output NEW_DIRECTORY [--verify-access]
The capture is system-wide. Keep its raw data private. Exit 0 requires both switch and wakeup
events; this proves usable event data, not loss-free attribution or a performance improvement.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HELPER = "/usr/local/libexec/prosper-perf-capture"
ENV = {**os.environ, "LC_ALL": "C", "PERF_CONFIG": "/dev/null",
       "PERF_CONFIG_NOSYSTEM": "1", "PERF_CONFIG_NOGLOBAL": "1", "DEBUGINFOD_URLS": ""}


def run(command, output, name, timeout):
    with (output / (name + ".out")).open("wb") as stdout, (output / (name + ".err")).open("wb") as stderr:
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=ENV, timeout=timeout)
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            stderr.write(str(exc).encode())
            code = None
    return {"command": command, "returncode": code}


def event_counts(path):
    counts = collections.Counter()
    with path.open() as stream:
        for line in stream:
            event = line.strip().removesuffix(":")
            if re.fullmatch(r"sched:[a-z_]+", event):
                counts[event] += 1
    return dict(counts)


def rejection_ok(result, stderr, expected):
    # Missing helper, denied sudo, timeout or another unrelated error cannot validate a refusal.
    return result["returncode"] == 1 and f"prosper-perf: {expected}" in stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=int, default=10, choices=range(1, 61), metavar="1..60")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-access", action="store_true", help="also verify hostile invocations are refused")
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("run this client as your ordinary host user, not through sudo")
    os.umask(0o077)
    output = args.output.resolve()
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"output must be a new directory: {exc}")
    report = {"status": "FAILED", "scope": "system-wide scheduler events", "kernel": os.uname().release}
    capture = run(["/usr/bin/sudo", "-n", HELPER, "scheduler", str(args.seconds)],
                  output, "capture", args.seconds + 20)
    report["capture"] = capture
    data = output / "scheduler.data"
    (output / "capture.out").rename(data)
    if capture["returncode"] == 0:
        decode = run(["/usr/bin/perf", "script", "-i", str(data), "-F", "event"], output, "events", 30)
        report["decode"] = decode
        counts = event_counts(output / "events.out")
        report["samples"] = counts
        if decode["returncode"] == 0 and all(counts.get(e, 0) > 0 for e in
                                              ("sched:sched_switch", "sched:sched_wakeup")):
            report["status"] = "READY"
    if args.verify_access and report["status"] == "READY":
        refusals = []
        for name, arguments, expected in (
                ("duration", ["scheduler", "61"], "capture duration exceeds 60 seconds"),
                ("command", ["scheduler", "1", "/usr/bin/id"], "expected: scheduler SECONDS"),
                ("output-path", ["scheduler", "1", "--output", "/etc/prosper-perf-MUST-NOT-EXIST"],
                 "expected: scheduler SECONDS")):
            result = run(["/usr/bin/sudo", "-n", HELPER, *arguments], output, "reject-" + name, 5)
            result["verified"] = rejection_ok(result, (output / ("reject-" + name + ".err")).read_text(), expected)
            refusals.append(result)
        report["refusals"] = refusals
        if not all(r["verified"] for r in refusals):
            report["status"] = "FAILED"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Evidence: {output}")
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
