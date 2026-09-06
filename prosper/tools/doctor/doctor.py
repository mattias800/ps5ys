#!/usr/bin/env python3
"""Inventory debugging tools and prove selected capabilities with bounded recordings.

Usage: doctor.py [--probe perf|perf-kernel|scheduler|debugger ... --output NEW_DIRECTORY]
       doctor.py --json

CPU/debugger controls use own children. Scheduler recording requires
--system-wide-scheduler and includes other host processes for two seconds.
User-space and kernel sampling are SEPARATE capabilities, because they fail
separately: kernel.perf_event_paranoid=2 denies kernel profiling while leaving
user-space sampling — the kind the profiling workflows actually use — available.
Ask for what you need; a verdict then describes only that.
No probes by default. Exit 0: inventory completed or every requested probe passed.
Exit 1: at least one requested capability was unavailable/inconclusive. Exit 2: usage.
"""
import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

TOOLS = (
    "perf", "gdb", "hotspot", "heaptrack", "valgrind", "bpftrace", "strace",
    "ltrace", "sysprof-cli", "radeontop", "renderdoccmd", "qrenderdoc",
    "vulkaninfo", "spirv-val", "spirv-as", "spirv-dis", "llvm-mc",
    "stackcollapse-perf.pl", "flamegraph.pl",
)
CPU_CONTROL = (
    "import os,time; end=time.monotonic()+1.5; fd=os.open('/dev/zero',os.O_RDONLY); "
    "\nwhile time.monotonic()<end: os.read(fd,4096)"
)


def read_setting(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc.strerror}"


def inventory():
    modules = {}
    for name in ("renderdoc", "elftools"):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    return {
        "platform": platform.platform(),
        "python": sys.executable,
        "tools": {name: shutil.which(name) for name in TOOLS},
        "python_modules": modules,
        "settings": {
            name: read_setting("/proc/sys/" + name.replace(".", "/"))
            for name in ("kernel.perf_event_paranoid", "kernel.kptr_restrict",
                         "kernel.yama.ptrace_scope")
        },
        "tracefs": {
            path: os.access(path, os.R_OK)
            for path in ("/sys/kernel/tracing/available_events",
                         "/sys/kernel/debug/tracing/available_events")
        },
        "note": "Executable/module presence is inventory, not proof of capability.",
    }


def run(command, directory, name, timeout=15):
    start = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "LC_ALL": "C", "DEBUGINFOD_URLS": ""})
        result = {"command": command, "returncode": proc.returncode,
                  "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    result["elapsed_seconds"] = round(time.monotonic() - start, 4)
    (directory / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def events_in(text):
    # perf script -F event emits one event name per sample, not arbitrary diagnostics.
    return collections.Counter(line.strip().rstrip(":") for line in text.splitlines()
                               if re.fullmatch(r"[\w:-]+:?", line.strip()))


def sample_verdict(record, script, required):
    counts = events_in(script["stdout"])
    ready = (record["returncode"] == 0 and script["returncode"] == 0
             and all(counts[event] > 0 for event in required))
    return {"status": "READY" if ready else "UNAVAILABLE",
            "samples": {event: counts[event] for event in required},
            "reason": "Required samples recorded and decoded." if ready else
                      "Recording/decoding failed or required samples were absent; inspect logs."}


# One entry per capability a caller can ask for by name. Splitting user-space from
# kernel sampling is not cosmetic: perf_event_paranoid=2 denies the kernel arm and
# permits the user-space arm, so a probe requiring both answers UNAVAILABLE on a host
# where the sampling these workflows actually run is fine. A verdict must describe the
# capability that was requested, or it is answering a question nobody asked.
PERF_CASES = {
    "perf": ("cpu", ["cpu-clock:u"]),
    "perf-kernel": ("kernel", ["cpu-clock:k"]),
    "scheduler": ("scheduler", ["sched:sched_switch", "sched:sched_wakeup"]),
}
PERF_SCOPE = {
    "perf": "User-space samples from this probe's own child. Kernel sampling is a separate "
            "capability (--probe perf-kernel); guest symbolication is another.",
    "perf-kernel": "Kernel-space samples from this probe's own child. perf_event_paranoid>=2 "
                   "denies this while leaving user-space sampling available, so read this "
                   "verdict as being about kernel profiling alone.",
    "scheduler": "Two-second system-wide recording, including other processes; "
                 "capability only, not game attribution.",
}


def perf_probe(directory, mode):
    if not shutil.which("perf"):
        return {"status": "UNAVAILABLE", "reason": "perf is not on this environment's PATH."}
    scheduler = mode == "scheduler"
    name, events = PERF_CASES[mode]
    control = ["sleep", "2"] if scheduler else [sys.executable, "-c", CPU_CONTROL]
    data = directory / (name + ".data")
    command = ["perf", "record", "-q", "-o", str(data)]
    if scheduler:
        # Wakeups can be emitted outside the target task's recording context.
        # The child-only handshake recorded switches but zero wakeups on a
        # working host. main() requires explicit consent for this wider scope.
        command += ["-a", "-c", "1"]
    else:
        command += ["-F", "99", "-g"]
    for event in events:
        command += ["-e", event]
    command += ["--", *control]
    rec = run(command, directory, name + "-record")
    decoded = run(["perf", "script", "-i", str(data), "-F", "event"],
                  directory, name + "-decode")
    verdict = sample_verdict(rec, decoded, events)
    return {"status": verdict["status"], "controls": {name: verdict},
            "scope": PERF_SCOPE[mode]}


def debugger_probe(directory):
    if not shutil.which("gdb"):
        return {"status": "UNAVAILABLE", "reason": "gdb is not on PATH."}
    # Only our child is attached. No process-name matching, no calls executed in the inferior.
    # Deliberately not a `with` block: Popen.__exit__ waits with no timeout, so a child left
    # stopped by a debugger that died without detaching would turn a bounded capability probe
    # into an unbounded block. Teardown below escalates instead and always terminates.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = run(["gdb", "-nx", "-nh", "-batch", "-p", str(child.pid),
                      "-ex", 'python print("DOCTOR_ATTACHED_PID=%d" % gdb.selected_inferior().pid)',
                      "-ex", "info threads", "-ex", "detach"], directory, "debugger")
        ready = (result["returncode"] == 0
                 and f"DOCTOR_ATTACHED_PID={child.pid}" in result["stdout"])
        return {"status": "READY" if ready else "UNAVAILABLE",
                "reason": "Attached to this probe's child and detached." if ready else
                "Attach not established; inspect debugger.json.",
                "scope": "Same-environment child only; test host/container crossing separately."}
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGKILL reaps a group-stopped child that ignored SIGTERM.
            child.kill()
            child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="append",
                        choices=("perf", "perf-kernel", "scheduler", "debugger"), default=[])
    parser.add_argument("--output", type=Path, help="new directory on real disk; retained logs")
    parser.add_argument("--system-wide-scheduler", action="store_true",
                        help="allow the scheduler probe to record other host processes for two seconds")
    parser.add_argument("--json", action="store_true", help="print machine-readable inventory/report")
    args = parser.parse_args()
    if args.probe and not args.output:
        parser.error("--probe requires --output NEW_DIRECTORY (use disk, not /tmp)")
    if ("scheduler" in args.probe) != args.system_wide_scheduler:
        parser.error("--probe scheduler requires --system-wide-scheduler, and vice versa; "
                     "this records scheduler events from other host processes")
    report = {"schema_version": 1, "inventory": inventory(), "probes": {}}
    if args.output:
        try:
            args.output = args.output.resolve()
            args.output.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"cannot create a new output directory: {exc}")
    for probe in dict.fromkeys(args.probe):
        report["probes"][probe] = (debugger_probe(args.output) if probe == "debugger"
                                   else perf_probe(args.output, probe))
    if args.output:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, path in report["inventory"]["tools"].items():
            print(f"{'INSTALLED' if path else 'MISSING':11} {name}: {path or '-'}")
        print(report["inventory"]["note"])
        for name, result in report["probes"].items():
            print(f"{result['status']:11} {name}: {json.dumps(result)}")
        if args.output:
            print(f"Evidence: {args.output}")
    return int(any(p["status"] != "READY" for p in report["probes"].values()))


if __name__ == "__main__":
    sys.exit(main())
