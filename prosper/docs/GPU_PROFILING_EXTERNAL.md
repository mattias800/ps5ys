# External GPU profiling tools — what works on prosper today

**Read this before building another timer.** prosper is an ordinary Vulkan application, so the free
vendor and open-source GPU tooling works on it directly. Everything below was verified on this
Linux/AMD box on 2026-08-08, on `prosper-app`, with **no change to prosper's own code**.

The project has seventeen `PROSPER_*` timing switches, a per-pass GPU timer, a capture/replay stack
and two report scripts. They are individually good. The tools here answer questions none of them can
— per-draw hardware timing, wavefront occupancy, and barrier/queue events — and they answer them on
**any title, with no per-title work**.

## AMD Radeon GPU Profiler (RGP) — verified working

RADV ships the capture side; nothing needs installing to *produce* a trace.

```bash
# Capture frame 900. Set the output directory FIRST -- see the /tmp warning below.
mkdir -p ~/work/rgp && cd ~/work/rgp
MESA_VK_TRACE=rgp MESA_VK_TRACE_FRAME=900 \
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    <WORKTREE>/build/prosper-app <DUMP_ROOT>/<TITLE>-app0
```

Verified output on Blue Prince (`PPSA25009`), first attempt, 1.8 MB:

```
radv: Thread trace support is enabled (initial buffer size: 32 MiB,
      instruction timing: enabled, cache counters: enabled, queue events: enabled)
RGP capture saved to 'prosper-app_2026.08.08_19.58.22_frame901.rgp'
```

**What it gives that we cannot:** per-draw and per-dispatch *hardware* timing, wavefront occupancy,
instruction timing, cache counters, and **queue events / barriers** — i.e. synchronisation stalls,
which no `PROSPER_*` switch measures at all.

Selectors: `MESA_VK_TRACE_FRAME=<n>`, `MESA_VK_TRACE_TRIGGER=<file>` (create the file to trigger),
`MESA_VK_TRACE_PER_SUBMIT=1`, `RADV_THREAD_TRACE_BUFFER_SIZE=<MiB>`. Other backends: `rra` (ray
tracing), `rmv` (memory), `ctxroll`.

### Use the TRIGGER, not a frame ordinal — and capture the regime you mean

**A frame number is not portable across runs, and a title with performance regimes makes it actively
misleading.** Both halves cost a capture here:

* `MESA_VK_TRACE_FRAME=900` on Blue Prince lands at **t ~ 5 s**, because the load phase runs at
  108-223 flips/s while the collapse runs at 3.2. That capture is a **menu frame** — the wrong regime
  for any performance question, and nothing in the file says so.
* `MESA_VK_TRACE_FRAME=7500`, chosen from a previous run's counters, produced **no capture at all**:
  that run reached only 6,763 presents before it ended. Two runs of the same route pace differently.

Use the trigger file and fire it when the condition you care about is true:

```bash
out=~/work/rgp; mkdir -p $out; cd $out; rm -f $out/trigger
MESA_VK_TRACE=rgp MESA_VK_TRACE_TRIGGER=$out/trigger ... ./build/prosper-app <DUMP> &
sleep 110          # or wait for whatever marks the regime -- a log line, a flip rate, a screenshot
touch $out/trigger
```

**Sanity-check the regime from the capture's own size.** On Blue Prince the menu frame is **1.8 MB**
and the collapsed frame is **35.9 MB** — a 20x difference, matching ~12 draws/submit against ~4,060.
A capture that is far smaller than expected is very likely the wrong phase, and that is the cheapest
check available before anyone spends time reading it.

**Reading it needs the RGP GUI** (free download from AMD, Linux build available). There is no headless
reader, so an agent can *produce* a capture but a human opens it. Say so when handing one over rather
than implying you read it.

## RenderDoc — aimable headlessly since #3321, and `convert` reads it with no Python

```bash
sudo dnf install -y renderdoc renderdoc-devel     # 1.45 on this box, Vulkan supported

# Capture one guest frame, headless, no keypress. Either axis, exactly like F8/F9.
ENABLE_VULKAN_RENDERDOC_CAPTURE=1 \
PROSPER_RENDERDOC_AFTER_MS=175000 \
PROSPER_CAPTURE_DIR=$HOME/work \
    ./prosper-app <DUMP_ROOT>/<TITLE_ID>-app0

renderdoccmd convert -f cap.rdc -c xml -o frame.xml      # full event list, no Python
```

**This section used to say `renderdoccmd capture` triggers on a keypress "which makes it awkward
headless", and recommended RGP instead. That was true of `renderdoccmd` and is no longer true of
prosper.** RenderDoc's in-application API needs no keypress — `librenderdoc.so` exports
`RENDERDOC_GetAPI`, whose struct carries `StartFrameCapture`/`EndFrameCapture`, and those delimit a
capture *explicitly*, so they need neither a keypress nor a present. prosper drives them itself:
`PROSPER_RENDERDOC_AT_FRAME` (ordinal) and `PROSPER_RENDERDOC_AFTER_MS` (wall clock), plus
`PROSPER_RENDERDOC_MAX_ITERS` for the span cap. `frontends/shared/diagnostics/renderdoc_capture.hpp`.

### Aim it at the RENDERER's device. A capture of the wrong one looks perfectly healthy.

**On the default headless route prosper runs two Vulkan instances.** prosper-app owns one for
presentation (`main.cpp`); the live renderer owns another (`tests/fixtures/render_runner.h` — that
header *is* the renderer) and publishes it through `prosper::gpu::set_shared_vulkan_context`. On that
route every guest draw is on the renderer's. **Two qualifications**: `live_compute.cpp:2913` can
create a *third*, private instance when it declines to adopt the renderer's, and present unification
(#1270) can collapse the two into one when the renderer's instance is surface-capable — so "two" is
this route's count, not an invariant.

Passing NULL lets RenderDoc choose the device. RenderDoc documents that choice as arbitrary; observed
here (n=1) it took the one holding the swapchain. **The lesson is therefore "always pass the device
explicitly", not "it picks the swapchain"** — an arbitrary choice that happens to be right once is
worse than a wrong one, because it will not stay right. Measured on *The Messenger*, same route and
identical span logic, differing only in which device the capture is aimed at (two builds, one edit
apart):

| aimed at | chunks | `vkCmdDrawIndexed` | `vkCmdDispatch` | graphics pipelines | compute pipelines |
| --- | ---: | ---: | ---: | ---: | ---: |
| NULL (RenderDoc picks) | 56 | **0** | **0** | **0** | **0** |
| the renderer's instance | **406** | **9** | **2** | **6** | **1** |

The 56-chunk capture is not corrupt and not truncated — it is a **complete, valid capture of the
wrong device**, and it converts and opens perfectly while containing nothing anyone wants. It reads
as a working instrument reporting a negative result, which is the dangerous shape. The trigger now
aims itself and warns loudly when the renderer has not yet published a context.

**A prosper frame is the GUEST's, not the app loop's.** A span of one app-loop iteration also yields
zero draws: guest work is submitted off that thread, and the loop can spin at 260 fps while the guest
renders far slower. The span closes on a guest present, and the log says which of the two closed it.

**A missing system Python module does not rule out scripted replay.** The tested Fedora package
embeds bindings in `qrenderdoc`; `qrenderdoc --python` with the offscreen Qt platform can run
replay scripts without entering the UI event loop. The [verified control and wrapper](DEBUGGING_WORKFLOWS.md#renderdoc-prove-capture-replay-and-data-inspection)
capture five indexed draws, replay them, check exact SSBO contents and export a render target.
**Pixel history is now checked and works here**: RenderDoc 1.45 on RADV/STRIX_HALO reports
`APIProperties.pixelHistory = True`, and `tools/pixel_history/` validates the whole read against a
construction whose answer is known in advance — a pass, a depth failure, a scissor rejection, a
shader discard and a final pass, all at one named pixel. `APIProperties.shaderDebugging` also reports
`True` on the same replay, but nothing has exercised it yet, so treat that as an unverified flag
rather than a capability this project has used. `convert` also exports `xml` or `chrome.json` API events, but those event durations
are not hardware GPU execution timings.

## `radeontop` — the 60-second "is this even GPU-bound?" triage

Verified, and it is the cheapest useful answer in this document: **no capture, no GUI, one run, any
title.** Install with `dnf install radeontop`.

```bash
# 1. Start the title. 2. Sample INSIDE the regime you care about, never across a phase boundary.
radeontop -d - -l 60
```

Measured on *Blue Prince* in its collapsed regime (t > 70 s, 60 samples) against a `vkcube` control:

| | `gpu` mean | max | `spi` | `cb` | `sclk` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vkcube` (control) | **56.31%** | 66.67% | 29.64% | 11.17% | 99.95% |
| Blue Prince, collapsed | **4.17%** | 7.50% | 3.43% | 1.32% | 38.1% |

**Blue Prince at ~3.2 fps leaves the GPU ~96% idle.** Combined with #2215's measured 30-45 ms/submit
`gpu_device`, that means those submits are long because they **wait**, not because they work — a
synchronisation question rather than a shading-cost one, and the two have completely different fixes.

### Run the control. It is one command and it is not optional here.

`radeontop` prints **"Unknown Radeon card"** on this STRIX_HALO APU, so a low reading is ambiguous between
*the GPU is idle* and *radeontop cannot read this chip*. `vkcube` renders continuously; if the counters
move for it, they are live.

**The control also found dead counters.** `ta` (texture addresser) and `ee` read **0.00% under vkcube**,
which certainly samples a texture — so they are not readable on this hardware. **Trust `gpu`, `spi`, `cb`,
`sx`, `sclk`. Do not quote `ta` or `ee`.** A triage step that silently reads zero on an unsupported block
is worse than no triage step, and that distinction exists only because a control ran.

## Also available, free, not yet explored

| tool | install | for |
| --- | --- | --- |
| `umr` | `dnf install umr` | AMD GPU register/ring debugger — hangs and resets |
| `perf` | installed | CPU side; already used across this project |

## The `/tmp` warning, and it is the charter's own

**RGP writes to `/tmp` by default.** On this box `/tmp` is a RAM-backed tmpfs with a per-user quota
shared by every concurrent agent, and filling it does not merely fail your write — it kills the Bash
tool for every agent on the machine (`CLAUDE.md`, *Write run artifacts to the real disk*). A single
frame capture was 1.8 MB, but `MESA_VK_TRACE_PER_SUBMIT=1` over a long run is not bounded by anything
you have chosen. **`cd` to a directory under `$HOME` before capturing**, and move any capture that
lands in `/tmp` immediately.

## Why this document exists

The instinct on hitting a performance question here has been to add a `PROSPER_*` timer. That produced
a large, individually-correct, collectively-unattributable set of instruments — and on 2026-08-08 six
published shares from two lanes were wrong for arithmetic reasons rather than measurement ones
(instrument traps 141, 143). Vendor tooling that reads the hardware's own thread trace is not subject
to that class of error, costs nothing to run, and needs no maintenance from us.

Reach for these first. Build a `PROSPER_*` switch only for something the guest-facing layer knows and
the GPU vendor cannot see.

## Vulkan validation (`PROSPER_VK_VALIDATION=1`)

`vk_validation_scan.py` (#1704) runs the whole ctest suite under the validation layer and diffs
against `tools/vkval/allowlist.txt`, and is registered with ctest as `vkval_scan_logic`. **Four
VUIDs have been fixed off the back of it** (#1713, #1714, #1717, #1726). **That is the project's
validation guard, and it works without any messenger**, because VVL's default `debug_action` writes
to stdout/stderr on its own.

**Do not quote id or message counts from anywhere — including `allowlist.txt`'s own header.** Run
`vk_validation_scan.py` and read what it computes (`[vkval] N distinct message ID(s), M message(s)
total`). The header records a baseline and is amended *sometimes*: #1726, #1717 and #1713 each
amended it, but #1714 deleted its entry without doing so, which leaves the header's last stated
ledger one id high in both columns, and the lavapipe figure at the top has never been amended at
all. This paragraph twice carried a stale figure of its own before saying that — the file's rule is
that an unexplained drop in the id count reads as *"the scan broke"*, so a quoted count is a false
alarm waiting to happen, wherever it is quoted from.

`PROSPER_VK_VALIDATION=1` is for the other case: validating **one interactive or routed run** of a
title, in-process. It enables the layer and registers a `VkDebugUtilsMessenger`, which adds over the
default action:

- **rate limiting** — 8 reports per message id. One violated VUID in a per-draw path otherwise fills
  the disk, and on this machine a large run log takes the shared tmpfs (and every agent's shell)
  with it.
- **a run that says whether it is armed**, so a clean result is falsifiable rather than merely quiet.
- **coverage when the default action is off**, which any `VK_LAYER_*` settings file can arrange.

```bash
PROSPER_VK_VALIDATION=1 PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    ./prosper/build-linux/screenshot --out ~/work --seconds 2 --count 30 <DUMP_ROOT>/<TITLE_ID>-app0
```

It announces which state it is in — the point of the switch:

| line | meaning |
| --- | --- |
| `[vk-validation] active (warnings and errors, 8 per message id)` | armed; a clean run is a real result |
| `... is not installed; NO validation is running` | the layer is absent — do not read the silence |
| `... VK_EXT_debug_utils is not advertised; the layer will run with its DEFAULT ... action` | the layer still validates, but this process does not rate-limit or tag it |
| `... the validation layer was DROPPED and NO validation is running` | instance creation fell back to a bare create info; a clean result is void |
| `... layer loaded but the debug messenger could NOT be registered` | output discarded; a clean result is void |

Off by default: the layer costs real time per draw.

**Coverage caveat.** This covers the shared render-runner instance only. `prosper-app` creates its
own instance, and `live_compute` has a private fallback instance; neither is affected by this
variable.

Worked example (#2998, 2026-08-27): a change that made Tomb Raider's gameplay render as a uniform
single-colour fill produced **zero** VUIDs with the messenger armed, which moved the question from
"is this a Vulkan misuse?" to "it is not, so look at the logic".
