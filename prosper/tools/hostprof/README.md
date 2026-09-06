# hostprof — native (host-side) sampling profiler for a running prosper process

Answers **"where is the HOST C++ side burning CPU right now?"** — the render/submit thread, a compute
worker, the readback copy, a detile loop — by attaching to a live process with `gdb`, sampling its
threads N times, and aggregating the innermost (leaf) function across samples into a ranked histogram.

It is a *statistical* profiler: each sample is one stack per thread, so the histogram approximates the
fraction of wall-time each function sits on top of the stack. More `-n` samples → tighter estimate.

## Why this exists (and why not `perf`)

- **Fallback when `perf` sampling is unavailable.** `kernel.perf_event_paranoid=2` excludes
  kernel sampling but does not universally forbid user-space events. Test actual recording with
  the [capability controls](../../docs/DEBUGGING_WORKFLOWS.md) before falling back to stopped-stack
  sampling: `doctor.py --probe perf` answers for user-space sampling alone, which is the arm that
  decides whether you need this tool at all. Debugger access has separate credential/namespace/security checks;
  `ptrace_scope=0` alone does not guarantee attach permission.
- **It generalizes the ad-hoc `tools/dbg/*.sh` sampling scripts.** Those dump *raw* all-thread
  backtraces to a fixed path for one specific title and leave the aggregation to hand-written `awk`.
  hostprof takes any pid/name, ranks leaves, strips template + argument noise from C++ symbols, filters
  parked/idle threads, and can emit flamegraph-collapsed stacks.
- **Complements `tools/guest_bt`.** `guest_bt` answers the same question for **guest** threads (managed
  C#/IL2CPP unwinding through the HLE stub boundary). hostprof is purely the **native** (prosper C++ /
  libc) side — no guest unwinding, no symbol re-sectioning.

hostprof is **read-only**: each pass does one `gdb -batch` attach → print → detach, leaving the process
running. Sampling briefly stops the target (a few ms per pass), so don't point it at a latency-critical
run you care about the timing of.

## Requirements

- `gdb` on `PATH`, and permission to attach (`kernel.yama.ptrace_scope = 0`, or run as root/parent).
- The target built with symbols (prosper's default builds are). Python 3.8+.

## Usage

```
hostprof.py <pid|process-name> [-n SAMPLES] [-i INTERVAL] [-d DEPTH]
            [--thread REGEX] [--mode leaf|thread|folded] [--keep-idle] [--raw-names]
hostprof.py --self-test          # verify the symbol simplifier, no process needed
```

| option | meaning |
|---|---|
| `-n, --samples` | number of sampling passes (default 30) |
| `-i, --interval` | seconds between passes (default 0.3) |
| `-d, --depth` | backtrace depth per thread (default 8; raise for `--mode folded`) |
| `--thread REGEX` | only aggregate the thread whose stack matches REGEX anywhere (e.g. a hot function) |
| `--mode leaf` | histogram of the leaf frame across **all** threads (default) |
| `--mode thread` | histogram of the **busiest** thread only (picked by `/proc` CPU time) |
| `--mode folded` | collapsed `root;…;leaf count` stacks — pipe to `flamegraph.pl` |
| `--keep-idle` | do **not** filter parked frames (futex/poll/nanosleep/bare `syscall`/…) — shows the wait breakdown |
| `--raw-names` | keep gdb's full templated symbol spelling instead of the simplified name |

Idle worker threads parked in a bare `syscall`/futex are filtered by default, so the all-thread `leaf`
histogram reflects real work rather than the size of the thread pool. Use `--thread` or `--mode thread`
to isolate a single working thread.

## Examples

```bash
# Boot a title headless, let it reach steady state, then find the hot host function:
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  ./build-linux/boot_trace /path/PPSA27616-app0 &
sleep 45

# What is the render/submit thread doing? (match a function that's always on its stack)
hostprof.py boot_trace -n 40 --thread 'execute_ordered|agc_driver_submit'

# All-thread leaf histogram (idle pool filtered):
hostprof.py boot_trace -n 30

# Flamegraph of the render thread:
hostprof.py boot_trace -n 200 -d 24 --thread 'execute_ordered' --mode folded > out.folded
flamegraph.pl out.folded > out.svg
```

Representative output (Bendy, PPSA27616, native 4K, RADV STRIX_HALO) — the readback + FP16-conversion
cost that `PROSPER_RENDER_TIMING` measures as `backend`-dominant shows up here as its hot leaves:

```
hostprof: 40 non-idle leaf samples over 40 passes, thread ~/execute_ordered|agc_driver_submit/

       %   count  function
    22.0       9  prosper::frontend::execute_item          # compute dispatch execution
    17.0       7  __memmove_avx512_unaligned_erms          # GPU->CPU readback copy
    15.0       6  prosper::gpu::half_to_float               # FP16 texture/RTT decode
    11.0       4  prosper::gpu::float_to_half               # FP16 storage writeback
    ...
```

## Interpreting results / caveats

- **Statistical, not exact.** Use it to *localize* the hot function, then confirm the magnitude with the
  instrumented `PROSPER_RENDER_TIMING` breakdown (per-phase ms) when you need numbers.
- **A bare `syscall` leaf is treated as idle** (the thread is blocked kernel-side, not burning user CPU).
  Real syscall-heavy work still shows its libc wrapper (`__memmove` for a readback, an `ioctl` issuer) as
  the leaf. Pass `--keep-idle` to see what threads are waiting on.
- **Headless `boot_trace` frame dumps add `fputc`/`__close_nocancel`** to the histogram (it writes a BMP
  per frame). Those are the tool harness, not the interactive render path — ignore them, or profile the
  `screenshot`/`prosper-app` frontend instead.

`--self-test` runs the symbol-simplifier unit cases and is a fast smoke check that the parser still
matches the current gdb output format.
