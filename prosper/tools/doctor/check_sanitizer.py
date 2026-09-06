#!/usr/bin/env python3
"""Require a clean run AND a specific diagnostic from each deliberate defect."""
import os
import subprocess
import sys


def main():
    cases = {"address-undefined": [("address", "AddressSanitizer: heap-buffer-overflow"),
                                   ("undefined", "runtime error: signed integer overflow")],
             "thread": [("thread", "ThreadSanitizer: data race")]}
    # A checker whose own misuse is a traceback teaches the reader nothing. Fail like a tool.
    if len(sys.argv) != 3 or sys.argv[2] not in cases:
        print(f"usage: {sys.argv[0]} BINARY {{{'|'.join(cases)}}}")
        return 2
    binary, mode = sys.argv[1:]
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
           "UBSAN_OPTIONS": "halt_on_error=1", "TSAN_OPTIONS": "halt_on_error=1"}
    for arm, signature in [("clean", "SANITIZER_CONTROL_CLEAN"), *cases[mode]]:
        result = subprocess.run([binary, arm], capture_output=True, text=True, env=env, timeout=15)
        output = result.stdout + result.stderr
        good_exit = result.returncode == 0 if arm == "clean" else result.returncode != 0
        if not good_exit or signature not in output:
            print(f"UNAVAILABLE: {arm} control did not produce its expected result "
                  f"(exit={result.returncode}, required={signature!r})\n{output}")
            return 1
        print(f"VERIFIED: {arm}: exit={result.returncode}, observed {signature!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
