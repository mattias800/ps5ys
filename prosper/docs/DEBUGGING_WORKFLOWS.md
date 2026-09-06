# Debugging and profiling: start from the question

Commands below run from `prosper/` in your own worktree. Build in the environment named by
`LOCAL.md`; on the Fedora development machine that is `distrobox enter ps5ys`. Run artifacts
belong in a new directory on real disk, shared with the container, never the RAM-backed `/tmp`.
Read the title/area's **Ruled out** section before starting an investigation.

## First: prove the instrument works here

```bash
python3 tools/doctor/doctor.py
# These controls touch only their own children; output must be a NEW directory.
python3 tools/doctor/doctor.py --probe perf --probe perf-kernel --probe debugger \
    --output build-linux/doctor-first --json
```

`INSTALLED` is inventory. `READY` requires the control's expected observation. Exit 1 means a
requested capability was unavailable; read `report.json` and the command logs, not an empty
histogram. **Ask only for the capability you need, because they fail separately.**
`--probe perf` is user-space sampling — what the `perf record -g -p` recipe below actually uses —
and `--probe perf-kernel` is kernel sampling, which `kernel.perf_event_paranoid=2` denies on its
own while leaving user-space sampling working. Asking for both on such a host is correct and
returns one READY and one UNAVAILABLE; asking for kernel sampling you do not need turns a usable
host into exit 1. The separate scheduler control
below requires both switch and wakeup events from an explicitly system-wide recording.
The debugger control attaches only
to its own child; it does **not** certify access across a container boundary or to an existing game.

Before attributing a real run, name its exact executable:

```bash
python3 tools/revision/check_build_revision.py build-linux \
    --binary build-linux/prosper-app --against HEAD --strict-dirty
sh tools/gpu_busy.sh vkprobe index_fetch_probe renderdoc_control vkcube
```

The GPU census has different exit conventions: **0 busy, 1 no named consumers, 2 failed census**.
It is not a global GPU utilization meter or a lock. Check again before each measured arm.

## Choose the first instrument

| Question | First evidence | Next step |
| --- | --- | --- |
| Black world, wrong colors, missing geometry | Normal screenshot plus F9 seeded bundle from the failing phase | Replay, inspect the affected submit's descriptors/resources, isolate draws |
| A specific pixel is the wrong colour | An `.rdc` of the failing frame, then `tools/pixel_history/` | It names which of the four causes applies; go to the file that cause implicates |
| Slow gameplay | Windowed app at native resolution, F8 report, calibrated GPU utilization | CPU samples for active work; wait tools for blocked threads |
| Stalled loading or intermittent hang | Repeated guest stacks plus progression evidence | Identify HLE wait/caller; inspect the producer or lock holder |
| Memory growth or corruption | Host allocation profile plus guest allocation/caller logs | Isolate the allocator, then watch the suspect write or run host tests under sanitizers |
| An API returns success but the game cannot progress | HLE return values **and** out-buffer changes | Resolve exact NID and inspect the caller's contract |

### Black or incorrect frames

```bash
build-linux/screenshot "$dump_dir" --seconds 1 --count 6 --timeout 30 --out "$run_dir/shots"
# On a separately routed app run, schedule F9 at a time already located in the failing phase.
PROSPER_CAPTURE_DIR="$run_dir" PROSPER_GRAB_BUNDLE_AFTER_MS=60000 \
    build-linux/prosper-app --dump "$dump_dir"
build-linux/gpu_replay --bundle "$bundle" "$run_dir/replay.bmp"
build-linux/gpu_replay --bundle "$bundle" --bundle-extract-submit 0 "$run_dir/submit.prgcap"
build-linux/gpu_replay --inspect-only "$run_dir/submit.prgcap"
build-linux/gpu_replay --validate "$run_dir/submit.prgcap"
```

Set `dump_dir` and a new `run_dir` first, and use the title's checked-in pad route and launch
environment. Set `bundle` from the completed **bundle written** log line. The 60-second selector
and submit 0 above illustrate syntax; choose the phase and submit from actual evidence.
Keep the capture's screenshot beside its bundle. A logo, movie or pre-rendered loading image
does not establish that the 3D world rendered.

**Before censusing draws to explain a black region, ask the pixel.** A draw census answers "what
was submitted", which is a different question from "what happened here", and inferring the second
from the first is what has produced retracted issues in this project:

```bash
# -c is not optional in practice: without it renderdoccmd writes the .rdc under /tmp, which
# on this box is the RAM-backed tmpfs a capture must never touch. Frame number is appended.
renderdoccmd capture -c "$run_dir/frame" -w build-linux/prosper-app --dump "$dump_dir"
python3 tools/pixel_history/pixel_history.py "$run_dir"/frame_frame*.rdc \
    --output "$run_dir/pixhist"
```

It reports `NOTHING_DREW`, `ALL_REJECTED` (naming the test that rejected), `SHADER_WROTE_BLACK`,
`STORE_LOST_IT`, `PIXEL_WAS_WRITTEN`, `CLEARED_AFTER_DRAW` or `OUTPUT_UNTRUSTED`, with the per-event
detail behind it, and defaults to the brightest pixel rather than the centre. Clears are reported but
are never the subject of a verdict — on a cleared target "the last passing event computed black"
would otherwise be the clear, and blame a shader that never ran (instrument trap 269). On a driver you have not used it on before, run `--expect-control` against
`pixel_history_control` first; see [its AGENTS.md](../tools/pixel_history/AGENTS.md) for why that is
not ceremony.

Replay can expose translation defects, but it does not reproduce every live ownership/race defect.
Repeat before treating an output hash as an oracle; the [determinism tools](../tools/determinism/AGENTS.md)
record this limit. A raw shader dump alone lacks resource tables and launch state.
Detailed commands: [GPU replay](../tools/gpu_replay/README.md),
[GPU timeline](../tools/gpu_timeline/README.md), [screenshots](../tools/screenshot/README.md).

### Slow gameplay

Measure the windowed `prosper-app` GPU presentation path. The screenshot frontend forces readback;
`SDL_VIDEODRIVER=offscreen` can also change the presentation workload. For throughput, use
`--present-mode immediate`, native resolution and full submit cadence. Record the present mode
actually selected. A display-capped rate cannot establish renderer throughput.

```bash
# Broad localization; choose the trigger time from the route, not a different title's recipe.
PROSPER_CAPTURE_DIR="$run_dir" PROSPER_PERF_CAPTURE_AFTER_MS=60000 \
    build-linux/prosper-app --dump "$dump_dir" --present-mode immediate
python3 tools/perf/performance_capture_report.py "$perf_capture" --json

# Attach inside the relevant gameplay interval; target_pid is the exact process you launched.
perf record -F 199 -g -p "$target_pid" -o "$run_dir/cpu.data" -- sleep 15
perf report -i "$run_dir/cpu.data" --stdio
set -o pipefail
perf script -i "$run_dir/cpu.data" | stackcollapse-perf.pl > "$run_dir/cpu.folded"
flamegraph.pl "$run_dir/cpu.folded" > "$run_dir/cpu.svg"
```

Confirm samples and named stacks exist. `-g` needs usable unwind information; unresolved guest
frames require [guest_bt](../tools/guest_bt/README.md), not guesses from anonymous addresses.
If native stacks truncate, try a separate bounded `--call-graph dwarf,16384` recording and measure
its overhead. Do not silently switch stack methods between comparison arms.

`radeontop -t 10 -d - -l 10` is a cheap utilization check, **after** checking its counter against
an unthrottled known workload outside the measurement. A low counter with no validated ceiling
does not prove the GPU is idle. On the local APU, use the control and counter limitations in
`LOCAL.md`; the older vsync-limited cube recipe is not a reliable ceiling.

F8 post-trigger timers, F9 captures, debugger stops, frame dumps, and verbose logs all alter timing.
Use diagnostic runs to find the cause and separate minimally instrumented runs to measure the win.
For stopped-stack fallback only: [hostprof](../tools/hostprof/README.md).

**Before profiling symbols, divide by the core count.** A process consuming about **one**
core-second per wall-second on a many-core box is serial, whatever `perf report` then attributes the
time to — the question stops being "which function" and becomes "why is only one thread doing this".
Read it from `/proc/<pid>/stat` (utime+stime deltas) or `pidstat`, and take it before the flame
graph: it is one number, it cannot be misattributed by a truncated unwind, and it rules out a whole
class of answers. *Sonic Frontiers* at 3.3 fps sat at **1.01 of 32 cores** with `gpu-device` at 5.4%
of the window, which said "serial host work" before any symbol was read; three serial per-texel
conversion loops were the cause (#3405).

**A widely used parallel helper is not evidence that the chain you are profiling uses it — count
the call sites against the loops of that shape.** `parallel_compute_texels` was already live at
**fifteen** sites in `frontends/shared/live/live_compute.cpp`, which is exactly what makes the gap
invisible: the *sampled* conversion chain has seven per-texel arms and exactly **one** of them was
wrapped, so six 4K scalar loops sat among wrapped neighbours and read as parallel to anyone who
found the helper first. Grep for the helper, then grep for the loop shape it wraps, and compare the
counts **within the chain you are measuring**, not across the file. #3408.

### Stalled loading and waits

```bash
python3 tools/perf/wait_profile.py --pid "$target_pid" --seconds 20 --hz 20
python3 tools/perf/stack_profile.py --pid "$target_pid" --samples 5 --interval 5
python3 tools/perf/lock_holder.py --pid "$target_pid" --samples 100 --interval 0.05
python3 tools/guest_bt/guest_bt.py --pid "$target_pid" --initlog "$init_log" --all
```

Record `PROSPER_INITLOG=1` at launch for the module map. `guest_bt` bridges HLE stubs and uses
guest unwind data; IL2CPP symbols are optional. A hardware breakpoint's rbp walk is a useful
fallback when that stack shape actually has frame pointers.

Choose the debugger environment by a successful attach to the actual target. Container-local child
attach, cross-container attach, and host attach are different tests. `No threads` or an empty
stack is not evidence that the game is idle. A single blocked sample is not a hang: frame pacing
also waits. Re-sample and compare progression.

`/proc` state R includes runnable threads waiting for a CPU. It does not measure on-CPU time.
Futex addresses do not distinguish mutexes from condition variables, and a plausible owner word
is only a hint until the lock type is established. See [thread waits](THREAD_WAIT_PROFILING.md).

### Memory growth and API behavior

```bash
heaptrack -o "$run_dir/heap" build-linux/prosper-app --dump "$dump_dir"
python3 tools/hle_calls/hle_calls.py --help
```

Heaptrack observes intercepted **host** allocations; the guest's own allocator and Vulkan memory
need separate accounting. Use `PROSPER_MEMLOG=1 PROSPER_DMEM_CALLER=1` for guest direct-memory
allocation chains, then resolve the caller against the module. These logs can change a race.
For API stalls, [hle_calls](../tools/hle_calls/README.md) supports return values, out-buffer
changes and launch-time tracing. A shared HLE handler can represent several NIDs; preserve the
specific guest call's identity.

## Scheduler tracing: recording and access

The scheduler probe is a capability test, not a game profiler. To measure scheduling latency,
use system-wide switch/wakeup events and filter the report to the game's threads: wakeups may
be emitted by a different task, so a recording filtered only to the game's PID can lose the cause.

**Two routes, and they answer different questions.** The probe below establishes that this host
can record and decode scheduler events at all, using `sudo` interactively. If you need scheduler
recording *without a human at the keyboard*, [`tools/profiling_access/`](../tools/profiling_access/README.md)
is the sanctioned route: one administrator installation grants one account passwordless access to
a scheduler-only, duration- and byte-bounded helper — not unrestricted `perf`, and not a root
shell. Prefer it over widening `sudo` or relaxing global perf settings to get an unattended
capture. Neither route changes what the recording proves: event capture, not loss-free attribution
to a particular game thread.

First verify access with a two-second recording on the **host**, using a new output directory:

```bash
sudo python3 tools/doctor/doctor.py --probe scheduler --system-wide-scheduler \
    --output build-doctor/scheduler-first --json
```

The explicit option acknowledges that the recording includes other host processes. Keep it
locally. The earlier child-only handshake collected switches but zero wakeups on a working host;
it was not a reliable test of wakeup availability. A successful system-wide control establishes
event recording/decoding, not loss-free game attribution.

On a host with a working privileged `perf`, from an administrator shell:

```bash
# run_dir must already exist on real disk; capture is bounded to ten seconds.
sudo perf record -a -c 1 -e sched:sched_switch -e sched:sched_wakeup \
    -e sched:sched_wakeup_new -o "$run_dir/scheduler.data" -- sleep 10
sudo perf sched timehist -i "$run_dir/scheduler.data" -p "$target_pid"
```

Inspect the recording for lost events and confirm the target's threads appear. Use the report's
run time, wait time and scheduling delay fields to separate blocked time from runnable delay.
Retain the complete capture locally; system-wide traces include other processes.

**A mounted tracefs and `perf_event_paranoid=1` are insufficient.** Trace metadata must be readable
and the recorder must have the necessary host-kernel privileges. Root inside a rootless container
is not host root. If host `perf` is absent and tracefs is root-only, administrator setup is needed.
On Fedora Atomic, the package setup command is
`sudo rpm-ostree install --apply-live perf`; if live application is unavailable, follow
rpm-ostree's reported reboot requirement. Then use the bounded privileged recording above.
This avoids changing global perf restrictions or making tracefs broadly readable. These host
setup/recording steps must be verified on the actual host; see `LOCAL.md` for local readiness.

References: [kernel perf access control](https://docs.kernel.org/admin-guide/perf-security.html),
[rpm-ostree live package additions](https://coreos.github.io/rpm-ostree/apply-live/)
and [perf sched manual](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/perf/Documentation/perf-sched.adoc).

## Isolated host sanitizer checks

This builds existing SELF parsing, RDNA2 decoding, BC decoding and submit-gate concurrency tests
against their production sources. It never maps a guest or launches Vulkan.

```bash
mkdir -p build-sanitizers/tmpdir
export TMPDIR="$PWD/build-sanitizers/tmpdir"
cmake -S tools/doctor -B build-sanitizers -G Ninja -DPROSPER_SANITIZER=address-undefined
cmake --build build-sanitizers -j6
ctest --test-dir build-sanitizers --no-tests=error --output-on-failure

cmake -S tools/doctor -B build-tsan -G Ninja -DPROSPER_SANITIZER=thread
cmake --build build-tsan -j6
ctest --test-dir build-tsan --no-tests=error --output-on-failure
```

Both configurations must pass the clean control and detect their deliberately faulty controls
with the expected diagnostic, and both are pinned in CI (`host-sanitizers`, `host-tsan`) so that
requirement cannot quietly stop being true. A runtime initialization failure is not a detected race.
Fedora GCC needs `libasan libubsan libtsan` in the build container. All six registered tests
must execute. `PROSPER_SANITIZER=none` is available for an uninstrumented comparison; it omits
the instrument control and must not be described as sanitizer validation.

This is a focused host suite, not whole-emulator coverage. Guest machine code has no compiler
instrumentation, and guest mappings/TLS require separate feasibility work before a sanitized
whole-game run can be interpreted. [ASan](https://clang.llvm.org/docs/AddressSanitizer.html),
[UBSan](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html),
[TSan](https://clang.llvm.org/docs/ThreadSanitizer.html).

## RenderDoc: prove capture, replay and data inspection

The Fedora `qrenderdoc` executable embeds usable Python bindings even when
`python3 -c 'import renderdoc'` fails. The wrapper below uses those bindings without entering
the interactive UI. It replays Vulkan, so respect the shared-GPU census.

```bash
mkdir -p build-doctor/tmpdir
export TMPDIR="$PWD/build-doctor/tmpdir"
cmake -S tools/doctor -B build-doctor -G Ninja \
    -DPROSPER_SANITIZER=none -DPROSPER_RENDERDOC_CONTROL=ON
cmake --build build-doctor --target renderdoc_control -j6
mkdir -p "$run_dir/rdoc"
renderdoccmd capture -w "$PWD/build-doctor/renderdoc_control" "$run_dir/rdoc/control"
python3 tools/doctor/renderdoc_inspect.py "$run_dir/rdoc/control_capture.rdc" \
    --output "$run_dir/rdoc/replay" --expect-index-control
```

Use the path printed by `RENDERDOC_CONTROL_CAPTURE=` if the capture naming differs.
`renderdoc-devel` supplies the API header; Vulkan development headers are also required.
The control reuses the existing indexed-draw probe: **five draws, one exact 1,024-byte SSBO
result**, including untouched sentinel words. The color attachment is deliberately DONT_CARE
in this probe, so its exported image is not a pixel oracle. For a game RDC, omit
`--expect-index-control`; `replay.json` lists draws/resources and `final-target.png` exports
the last draw's first color target, which need not be the final presented image.

This checks RenderDoc's Vulkan view. Prosper's seeded F9 bundles retain guest-side state that
an RDC cannot replace. `renderdoccmd convert -f capture.rdc -c chrome.json -o events.json`
exports API events; do not mistake their durations for hardware GPU execution time.
RGP hardware tracing remains a separate workflow: [external GPU profiling](GPU_PROFILING_EXTERNAL.md).

## Repeatable performance comparisons

Use at least three valid runs per arm, alternating A/B then B/A. Keep the title revision,
save starting state, input route, checkpoint, driver, resolution, presentation mode and cache
policy the same. Compare the **same phase**, never whole-boot averages or a menu against gameplay.

Retain a small manifest beside each run: source commit and dirty state, executable SHA-256,
route SHA-256, full launch arguments and relevant environment, driver/device, save/cache policy,
checkpoint evidence, measurement start/end, instrumentation, process exit status, and peer census.
Hash the executable actually launched, not only the checkout's HEAD. Use a fresh artifact directory
per arm so a failed run cannot inherit the previous arm's report.

Report per-run median/p95/p99 **frame intervals**, sample count and interval population; compare
the distribution of run results rather than pooling all frames as independent trials. Keep
guest flips, distinct rendered content and host publications separate. F8's 4 Hz rate samples
cannot supply per-frame percentiles. `flip_pacing_report.py` is a guest-flip diagnostic and filters
very short intervals; it does not establish distinct-image or host-present frame-time tails.
Include CPU/RSS and the correctness evidence. A faster run with missing geometry is a regression.

Measure profiler overhead with the same route both with and without that profiler; no universal
percentage applies. Captures and debugger stops belong outside benchmark windows. Mark missing
checkpoints, failed tools and incomplete runs invalid, with the reason retained.
