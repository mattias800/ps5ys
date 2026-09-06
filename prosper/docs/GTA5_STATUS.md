# Grand Theft Auto V (`PPSA04263`, RAGE) — status

**Rung 3** on the bring-up ladder: routed gameplay entry with real GPU draws, **and since the
rendering series (`e63f4038`, #2996) the 3D world renders** — the prologue bank heist in full
colour, on a default launch with the game's own **Performance** graphics mode selected. Route:
`scripts/gta5/reach-performance-story.pad`; screenshots in `BLOG.md` (2026-08-26).

> **The line above used to read "the HUD, radar and tutorial text render; the 3D world does not",
> and it stayed there for three days after the world started rendering.** That stale sentence cost
> real time on 2026-08-29: a user reported the world had regressed, and this document was quoted
> back at them as evidence that it had never worked. Treat a rung claim here as stale until the
> tracker or `BLOG.md` agrees with it.

**The world renders only in the game's own Performance graphics mode**, chosen from the landing menu
before the world loads (the default is Fidelity). That is a route property, not a build property —
a run started straight into Story on Fidelity shows the HUD over a dark scene and looks exactly like
a renderer regression.

Tracker: **#1873**. Active frontier: **#2542** and **#2690** — #2542 names ONE hanging compute
program and its title still calls it "the sole remaining cause"; there are at least three (#2690).
(#2481 is CLOSED and superseded by #2542; the
pointer here and in `CLAUDE.md` said #2481 long after it closed). Route: `scripts/gta5/reach-story-mode.pad`
(read its header — the flip timing is measured, not estimated, and the tab navigation needs four R1
presses for a reason).

**Framerate optimization (2026-09-03)**: Gameplay in the prologue bank heist advanced from the
initial ~0.6–0.8 FPS slide-show baseline to **21.0–21.3 FPS (46.99 ms average frame interval)** in native 4K Performance mode,
an overall **~34x speedup** (measured across 106 rendered frames in 4.90 s; Run 6 milestone reached 15.2–15.4 FPS; Run 7 reached 21.0–21.3 FPS; see breakdown below).

Historical design note for the descriptor work: `docs/FLAT_LOAD_DESIGN.md`. Do not start from it; the
descriptor-array lift it describes is complete.

## Linux windowed baseline (2026-09-06, #3065)

At `a5150495e`, a fresh Release build with `-O3 -DNDEBUG -g1 -fno-omit-frame-pointer`
reached the bank gameplay scene on the unchanged `reach-performance-story.pad` route. Both save
roots were fresh. Frontend screenshots confirmed **Graphics Mode: Performance** and subsequently
the bank interior, characters, radar and walking tutorial. The nine-minute run used native rendering,
full cadence, immediate presentation, default cache/worker settings and no diagnostic compute skips;
the app confirmed the renderer's shared present queue. No device-loss signal was recorded.

The native Linux/RADV F8 window measured **5.58 guest flips/s and 5.78 host presents/s** over
5.02 seconds, with 1.51 process CPU cores. Its 720 renderer and 1,917 compute detail records were
not truncated. Measured totals were 2,311 ms graphics and 1,547 ms compute; resource preparation
accounted for 1,179 ms across its frontend/backend layers. These are scoped timing records, not an
exhaustive GPU frame budget. The later F9 bundle completed with 73 submits and a matching frontend
gameplay image, outside the timed window. Artifact hashes and capture details belong to #3065.

A separate 20-second CPU profile retained 1,607 samples with zero reported loss. Compiled-key hashing
was 5.91% of **sampled** CPU; this is not a share of frame time or a promised speedup. Inspection found
that `make_shader_compile_key` computed a provisional hash which all three callers unconditionally
overwrote after attaching diagnostic settings. Removing that first calculation preserves the final key,
equality checks and shader output. End-to-end performance improvement is not yet established.

### Duplicate storage-image preparation (#3065)

On merged `805b49d53`, a later native Performance Story F8 isolated **3.512 ms** in the first
duplicate storage binding's pre-alias-processing interval, against **3.836 ms** total setup for
program hash `1d3f91e7f9a30140` (28 fully warm dispatches). Its 3840×2160 RGBA8 output binds one
canonical storage image and three exact aliases. The interval is not the cost of copying alias
fields: the backend requests an RTT snapshot before discovering the alias, while only the owner
publishes a binding-specific coverage proof. The renderer can materialize a full target on that read.

Exact write-only storage aliases now fold before this redundant snapshot/coverage preparation,
after descriptor, device, shape and DCC validation and the ownership query's pending-write drain.
Reflected storage representation and complete resource-view identity must match. Canonical seeding,
partial-write preservation and one writeback remain; sampled and mixed-access bindings retain the
late path. Synthetic execution checks both aliases' distinct pixels, necessary partial seeds and
missing-seed rejection; disabling only the early fold fails seven assertions. This establishes the
mechanism, not an end-to-end speedup. Native comparison evidence remains in #3065.

The phase-report tool's alias records are **not zero work**: this capture has 101.930 ms in aliases
versus 26.963 ms in real-image records. The report's misleading alias-only wording is tracked in
#3388. Do not treat its real-image-only decomposition as all image setup.

## Gameplay framerate optimization: reaching 21+ FPS (2026-09-03)

**Platform**: Measured on **Windows 11 / Intel Core i9 (24 physical cores) / discrete NVIDIA GeForce RTX 4090 (24 GB VRAM, Vulkan 1.4)**.

> [!NOTE]
> Several transfer-reduction mechanisms below (§2, §3, §4) explicitly address host staging bloat and discrete PCIe bus transfers between system RAM and dedicated VRAM. On unified-memory APU/iGPU architectures (such as Linux / Radeon 8060S), host memory and device memory share the same physical address space, so PCIe-specific bus bottlenecks do not exist there in the same form.

Overnight profiling and optimization of the native-4K Performance story route (`scripts/gta5/reach-performance-story.pad`, prologue bank heist) raised gameplay throughput from **0.6–0.8 FPS (1,600+ ms/frame) to 21.0–21.3 FPS (46.99 ms/frame)** on host hardware.

This represents a **~34x speedup**. The primary bottlenecks identified and resolved are detailed below.

### 1. MRT Flush Breaking Backend Submission Batching (~560 ms/frame reduction)

- **Problem Description**:
  GTA V renders scene geometry through four MRT color targets (G-buffers). In `tests/fixtures/render_runner.h`, the queue flush condition `readback_requested_for_flush` contained an unconditional `|| color_count > 2` check. Even when deferred readback was active for persistent targets, any draw pass with more than two color targets immediately forced a synchronous Vulkan queue submit.
  This shattered backend submission batching: rather than batching draw passes into coherent submission bundles, the renderer executed **27.88 separate `vkQueueSubmit` calls per frame**. Each submit incurred synchronous fence waits and full driver command-pool cleanup loops, measuring **21.4 ms of driver cleanup per call (~598 ms/frame)**.
- **Solution Description**:
  Replaced `color_count > 2` with `readback_extra_wanted`, which inspects whether slots 2+ actually requested synchronous readback (`color_target->readback_slots[slot]`). Passes with deferred readbacks now remain batched across draw passes.
  **Impact**:
  - Vulkan queue submissions dropped from **27.88 down to 2.0 calls per frame** (12x reduction).
  - Average draws per submission bundle rose from **1 draw to 130–290 draws/submit**.
  - Driver cleanup overhead dropped from **598 ms/frame to 1.7 ms/frame**.
  - Framerate immediately surged from 1.5 FPS to 11.8 FPS.

### 2. Native `Unorm2_10_10_10` Storage Format & Multi-Texel Seed Proving

- **Problem Description**:
  GTA V relies heavily on format 50 (`Unorm2_10_10_10`, `VK_FORMAT_A2B10G10R10_UNORM_PACK32`) for composite and lighting surfaces. This format was absent from `native_float_storage_image`, lacking capability bit 24 in `native_storage_format_support_bit`. Consequently, the recompiler fell back to raw float staging conversions, inflating temporary host staging allocations to **132.7 MB per window** and burning CPU cycles in texel packing/unpacking loops (`pack_ms`). Additionally, seed reproving was limited to a single texel per thread (`dispatch_has_enough_threads_for_texels`), failing multi-texel tiles and uploading full 33.4 MB seed payloads over PCIe.
- **Solution Description**:
  - Added `Unorm2_10_10_10` to `native_float_storage_image`, allocated bit 24 in `native_storage_format_support_bit`, expanded the support mask to `(1u << 25) - 1u`, and wired `VK_FORMAT_A2B10G10R10_UNORM_PACK32` in device queries.
  - Parameterized `dispatch_has_enough_threads_for_texels` with `max_texels_per_thread = 16`, proving complete multi-texel coverage and skipping 33.4 MB seed uploads.
  **Impact**: Staging memory bloat was eliminated, and CPU texel packing dropped to 0 ms.

### 3. Bounding PCIe GPU Storage Image Comparisons (38x reduction in compare time)

- **Problem Description**:
  When verifying storage writeback consistency, `max_gpu_compare_image_bytes()` had no ceiling, reading back and comparing 33–66 MB staging images across PCIe on every compute writeback. In F8 performance captures, `gpu_compare_ms` reached **1,138.6 ms per window**, dominating the runtime.
- **Solution Description**:
  Bounded host GPU storage image comparisons over PCIe to targets $\le 2\text{ MiB}$ (`PROSPER_MAX_GPU_COMPARE_IMAGE_MB=2`).
  **Impact**: `gpu_compare_ms` dropped from **1,138.6 ms down to 30.2 ms** (a 38x reduction), speeding up compute execution 3x to 5.4x.

### 4. Eliminating Spurious 4K Scanout and Unbound Slot CPU Readbacks

- **Problem Description**:
  - In `live_renderer.cpp`, `base != front_va` forced synchronous CPU readback of the 4K scanout buffer ($3840 \times 2160 \times 4 = 33.2\text{ MB}$, taking ~9.3 ms/frame). However, under active GPU presentation (`final_gpu_present`), `present_blit_publish` samples directly from Vulkan GPU images (`tgt->image`), so the CPU-copied pixels were immediately discarded.
  - In `render_runner.h`, when slot 0 or extra slots were not bound (`persistent_id == 0`), empty slots evaluated `!persistent_color` or uninitialized readback flags to true, triggering multi-megabyte `vkCmdCopyImageToBuffer` calls into staging buffers and subsequent host CPU memcpy loops for unused memory.
- **Solution Description**:
  - Defer scanout readbacks when `final_gpu_present` is active (`(base != front_va || final_gpu_present)`).
  - Gate `readback_color0_wanted`, `readback_color1_wanted`, and `readback_extra_wanted` on `color_target->persistent_id != 0`, preventing multi-megabyte host readback allocations on unbound slots.
  - In `live_renderer.cpp`, only set `backend_target.readback*` flags when the corresponding slot base address is non-zero.
  **Impact**: Saved ~17.6 ms of CPU readback and PCIe memcpy latency per frame.

### 5. Concurrent Shader Cache Scaling (`std::shared_mutex`)

- **Problem Description**:
  Drawing 442 items per frame across 16 parallel draw realization workers (`PROSPER_DRAW_REALIZE_THREADS=16`) generated 884 shader lookups per frame. In `src/gpu/execute/gpu_executor.cpp`, `ShaderCache` was guarded by a standard `std::mutex`. Even though the cache experienced 100% hits in steady state, all 16 worker threads serialized on this single mutex, accumulating **228.6 ms of contention across threads (14.3 ms wall time)**.
- **Solution Description**:
  Upgraded `ShaderCache::mutex` to `std::shared_mutex`. Cache hits now acquire `std::shared_lock`, allowing all 16 worker threads to query and resolve cached SPIR-V simultaneously. Thread-safe atomic counters (`atomic<uint64_t> hits`, `atomic<uint64_t> last_use`) eliminate data races without locking.
  **Impact**: Eliminated thread serialization; wall time between submissions dropped from 43.9 ms to 12.3 ms.

### 6. Summary of Progression & Verification

| Milestone | Configuration / Fixes | Frame Time (avg) | FPS | Key Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Default launch | ~1,600 ms | ~0.6–0.8 FPS | Initial state |
| **Run 2** | Native `Unorm2_10_10_10` + Bounded GPU compare | ~693 ms | ~1.5–2.0 FPS | Bounded PCIe compare (1138ms -> 30ms) |
| **Run 3** | MRT batching fix (`readback_extra_wanted`) | 84.9 ms | 11.8 FPS | Reduced submits from 28 to 2 (saved 598ms) |
| **Run 5** | 16 realization threads + Mailbox present + 8GB caches | 77.0 ms | 13.0 FPS | Triple-buffering, expanded target budgets |
| **Run 6** | Readback bypass + `shared_mutex` shader cache | 64.8 ms | 15.2–15.4 FPS | Zero thread contention across 16 workers |
| **Run 7 (Final)** | Full unbound slot readback elimination (`live_renderer` + `render_runner`) | **46.99 ms** | **21.0–21.3 FPS** | 0ms spurious readback across all MRT slots (~34x speedup) |

- **Final Performance Capture**: `perf_capture_PPSA04263_20260903-075401-494.prperf`
  - Sample Window: 4.90 s, 106 rendered frames.
  - Guest Flips Rate: **21.00 flips/s**.
  - Effective Renderer Rate: **21.28 FPS** (average dt: 46.99 ms).
  - Measured Graphics Time: **34.65 ms/frame** (28.8 FPS GPU capacity).
  - Inter-frame gap: **12.34 ms**.
  - Test Suite: **267 / 268 passed** (1 skipped, 0 failed).

### 7. Reproduction Recipe

To reproduce the benchmark capture on Windows:
```powershell
$env:PROSPER_PAD_SCRIPT = "@C:/Users/matti/repos/ps5ys/prosper/scripts/gta5/reach-performance-story.pad"
$env:PROSPER_PAD_SCRIPT_LOG = "1"
$env:PROSPER_CAPTURE_DIR = "C:/Users/matti/repos/ps5ys/tmp/captures"
$env:PROSPER_PERF_CAPTURE_AFTER_MS = "270000"
$env:PROSPER_RENDER = "1"
$env:PROSPER_MAX_GPU_COMPARE_IMAGE_MB = "2"
$env:PROSPER_BACKEND_TARGET_CACHE_COUNT = "4096"
$env:PROSPER_BACKEND_TARGET_CACHE_MB = "8192"
$env:PROSPER_BACKEND_TEXTURE_CACHE_MB = "8192"
$env:PROSPER_COMPUTE_IMAGE_CACHE_MB = "4096"
$env:PROSPER_TEXTURE_DECODE_CACHE_MB = "4096"
$env:PROSPER_DETILE_THREADS = "4"
$env:PROSPER_COMPUTE_CONVERSION_THREADS = "4"
$env:PROSPER_DRAW_REALIZE_THREADS = "16"

.\build-mingw-app\prosper-app.exe --dump "C:\Users\matti\repos\ps5ys\testdata\PPSA04263-app0" --volume 0 --present-mode mailbox
```
Inspect the resulting `.prperf` capture file with:
```bash
python tools/perf/performance_capture_report.py <capture>.prperf
```

## What the rendering series changed, and what it teaches (2026-08-26)

A nine-commit series lands the mechanisms below. Recorded here as *mechanisms*, because most of them
are not GTA V facts at all — they are contracts prosper was getting wrong that this title happened to
exercise hardest, and several map onto open issues for other reasons.

### The core rendering defect: two descriptors at one address were two images

**GTA V's 4K output shader writes the four 8x8 quadrants of each 16x16 block through FOUR bindings at
the same guest address.** Captures materialize each descriptor's bytes into an independently-owned
blob, so pointer equality is not a backing identity — and prosper treated the four reconstructed blobs
as four separate images. **Three of the four stores were therefore lost**, leaving the red/green
checker pattern in Performance mode. Two exact image descriptors naming the same non-null guest range
must alias **one** Vulkan image.

This is the single change most directly responsible for the world appearing, and it is worth reading
as a general lesson: *reconstructed* bytes lose the aliasing that the guest expressed through
addresses, so identity has to be re-established explicitly rather than inherited from the capture.

### Typed storage aliases: the guest writes integers and samples them as something else

GTA V writes full-resolution transition surfaces through **integer storage images** and then samples
the *same bits* through normalized, Float32 or packed views. Geometry and allocation are identical;
only the view's numeric interpretation differs. The series creates exactly those storage allocations
**mutable** and leases the consumer's view without a copy. Admitted pairs, deliberately narrow:

| producer | consumer | note |
| --- | --- | --- |
| `R32_UINT` | `R32_SFLOAT` | a depth-like transition surface; exact equal-width copy |
| `Uint8` / `Uint16` storage | UNORM sampled | integer texels sampled as normalized values |
| `R32_UINT` (carrier) | `B10G11R11` | packed R11G11B10; storage stays `R32_UINT` so driver float rounding cannot alter the bits |
| `R10G10B10A2` (GFX10 format 50) | Vulkan `A2B10G10R10` | same low-to-high bit placement; detile + byte copy |

**Not generalized** to signed, sRGB, component-count or size aliases — the series is explicit that this
is an allowlist, not a relaxed key, and that is the right shape.

Note the Fidelity bug in that table: the packed sampled path **fell through the generic
per-component-width test**, because packed formats deliberately report zero width there, and the whole
dispatch was skipped. A zero-width probe silently meaning "unsupported" is a trap worth remembering.

### Two open issues this addresses directly

- **#2723** — every shadow input refused by the DS bridge, cubes gated on `img_dim != 1`. The series
  adds the explicit shape alias: a compute `U#` writes the six `R16_UINT` shadow faces through
  `DIM=2D_ARRAY`, while the later graphics `T#` names the byte-identical allocation as `DIM=CUBE`, and
  graphics lowers cube sampling to a vertical 2D stack.
- **#2402** — the YUV composite draw skipped on NVIDIA because its fragment shader requires subgroup
  size 64 on a 32-wide device. The series runs Wave64 fragment programs at native Wave32 **for this
  title only**, and only for the narrow class whose sole remaining width reason is a control-flow
  `WaveAny`; ballots, lane identity and scalar reductions stay exact, and every other title keeps
  master's fail-visible exact-width contract.

### Wave-vote exactness — and what it means for the hang investigation

The series models the guest Wave64 **exactly** on a host subgroup that is not 64 wide, through
`guest_wave_readlane` with an LDS-scratch path when the guest wave does not fit the host subgroup, and
adds `V_READLANE_B32` / `V_WRITELANE` support. One added comment names a specific defect: a MAC between
a uniform compare and `VCCZ` **created a false Wave64 vote**.

That is the same area this file's own analysis pointed at. The recorded frontier reading was that
`0x413dc6700`'s only loop exit is `v_cmpx` → EXEC → `s_cbranch_execz`, that the emitted module votes
workgroup-wide where the guest votes per-wave, and that the dispatcher ran ~117x past what the data
could justify (#2858). A false wave vote is exactly a defect of that shape. **This is corroboration of
the direction, not proof of the mechanism** — nothing here has yet shown that this specific correction
is what stops that dispatch running away, and #2858's per-ordinal histogram would still be what
settles it.

### Platform contracts worth knowing outside this title

- **Windows cannot page-protect guest mappings for dirty tracking.** A VEH frame is built below the
  interrupted SysV RSP and would corrupt the guest's live red zone. The series keeps one exact
  guest-format mirror for exportable compute results instead, with `memcmp` validation so direct CPU
  writes still fail closed. Linux keeps the ordered journal plus page-protection write watch; the
  mirror must never launder a *known* architectural writer whose bytes happen to compare equal.
- **Driver pipeline caches are not safe to load by default.** Repeated GTA V runs on NVIDIA reproduced
  an `nvoglv64.dll` crash **only** on the cache-load route. UUID/header validation proves device
  compatibility, not blob robustness, so persistence is opt-in. The version-one header prefix is
  parsed field by field rather than cast, so short or unaligned files are rejected before any read,
  and a driver rejection is treated as a cache miss rather than a reason to disable compute.
- **Disabling driver optimization for large modules is a false economy.** It reduced cold compile
  latency on NVIDIA but turned GTA V's repeated 1440p dispatches from ~1.7 ms into **47-50 ms**.
  Optimization is on by default for large portable-CFG modules; the opt-out remains for investigations
  where bounded first-use latency matters more than steady state.
- **Win32 surfaces can transiently publish `currentExtent = 0x0`** during a minimize/fullscreen/DPI
  transition even after SDL reports a non-zero pixel size. That is an unavailable surface, not a fatal
  swapchain failure — keep the old swapchain and retry.

### What is NOT established

- **The validation is Windows/NVIDIA, on a route this repository does not contain.** The compute-skip
  section below cites `scripts/gta5/reach-story-mode-static.pad` and `PROSPER_WAVE64_APPROX`; **neither
  exists in this tree**. So its 25/25-sample result cannot be reproduced here as written, and the
  Linux/RADV behaviour of this series is **unmeasured**.
- **Six tests that pass on master fail with the series applied** — `dynfetch_fold`,
  `indirect_pointer_static_footprint`, `indirect_pointer_descriptor_range`, `gpu_capture_render_replay`,
  `rdna2_to_spirv_exec`, `gpu_execute`. Measured against an unpatched-master control on the same
  machine (302/302 vs 296/302). Four are not referenced by the series, so they are existing assertions
  the new behaviour breaks, and each needs triage before merge.
- **`live_target_format.hpp` calls itself "the one place a LiveTargetPixelFormat is mapped onto
  anything else." It is not** — `tools/gpu_replay/gpu_replay.cpp` maps it a second time, and extending
  the enum from three values to ten left that switch behind. `-Werror=switch` (#2844) caught it; a
  fall-through would have inspected every R8/R32/Rg8/Rgba32/Rg16/R16 target as 8-bit RGBA.

## The Windows Performance route does not require a compute skip (2026-08-26)

The routed native-4K Performance-mode scene now completes with
`PROSPER_COMPUTE_SKIP_PROGRAM` unset. On the current Windows/NVIDIA branch, the fixed
`scripts/gta5/reach-story-mode-static.pad` route captured 25/25 samples over 500 seconds, exited
normally with the guest still running, and recorded zero `VK_ERROR_DEVICE_LOST` or live-compute
disable events. The final bank frame retains the world, character and water-cooler materials, radar,
tutorial text and reflections. `PROSPER_WAVE64_APPROX` was also unset.

The retained selector was not rescuing this route. In the immediately preceding control with
`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700`, the parser announced the selector but emitted zero
`[compute-decline] ... reason=skipped-by-selector` records. A matched non-fast-path dispatch always
reports that decline, and `0x413dc6700` cannot match the sole CPU fast path's one-buffer fill shape.
An exact `PROSPER_COMPUTELOG_CODE=0x413dc6700` trace also recorded zero matching executions throughout
the 500-second no-skip route, including 160 seconds in the bank scene. The old address therefore did
not reach the selector on this route. Removing the environment variable changes no executed work.

This does not retract the historical Linux/RADV evidence below: that older route did execute the
program and could lose the device while consuming a cyclic traversal table. It establishes the
narrower current fact that carrying its address into the patched Windows Performance route was an
inert launch requirement, not a workaround. Keep `PROSPER_COMPUTE_SKIP_PROGRAM` as a generic
diagnostic selector, but do not set it for the accepted GTA V route.

## The F9 bundle works now, and the gameplay frame is dissectable offline (2026-08-19)

`make_capture_manifest()` copied a `GpuCaptureFile` field by field and omitted `format_version`, so
the manifest a frame bundle validates presented itself as version 0 and
`validate_captured_indirect_pointer_relocations()` refused it as *"capture format older than v53"* —
a capture whose provenance was complete. That is what aborted every F9 bundle on this title
(#2554 gate 2, fixed in #2718). Nothing wrong ever reached disk; `serialize_gpu_capture()` always
writes `kVersion`. It was a bad validation of an in-memory field.

**A gameplay frame now captures in 896 MB and replays with `rc=0`**, so this title's frame can be
taken apart offline and deterministically instead of chased live. Route
`scripts/gta5/reach-story-mode.pad`, 4K, Linux/RADV, four runaway compute programs declined, grab
armed at 420 s with `PROSPER_GRAB_BUNDLE_AFTER_MS`.

### The frame

**39 submits, 7 carrying work, 307 draws, 382 operations, and 3 realization failures.**

| submit | draws | computes | realized | what it is |
| ---: | ---: | ---: | ---: | --- |
| 37700 | 185 | 30 | 215/215 | shadow / cubemap / env pre-passes |
| 37705 | 20 | 9 | 29/30 | writes 4K slots |
| 37707 | 15 | 7 | 22/22 | 2992x1496 target, 45 written slot-bindings |
| 37710 | 18 | 4 | 22/24 | writes 4K slots |
| 37725 | 5 | 2 | 7/7 | writes 4K slots |
| 37727 | 62 | 20 | 82/82 | post chain |
| 37731 | 2 | 0 | 2/2 | final 4K composite |

Counting **every** MRT slot with its `cwm`, not slot 0 — a G-buffer names different surfaces per slot,
and `live_renderer.cpp` warns about exactly this:

- **submit 37700 binds a 3840x2160 surface 566 times and writes none of them** (`cwm=0` on every one).
  All its colour lands on 1024x1024, 256x256, 32x32, 4096x2048 and 1024x512 — shadow maps, an
  environment map, and six 256x256 `r11g11b10f` faces with the shape of a cubemap.
- **submit 37727 is a downsample pyramid**: 1920x1080, 960x540, 480x270, 240x135, 80x60, 20x20, 10x10,
  5x5, **1x1**. The 1x1 terminus is an auto-exposure luminance reduction.

### This corroborates the resolved chain above, independently

The lighting output `0x20431c0000` converts to **1 distinct colour — pure black — after both 37705
and 37710**, then carries content after 37725 (214 colours) and composites at 37731 (283 colours).
That is the same "lighting applies almost none" result as the section above, reached offline from a
bundle rather than from live dumps, and it agrees.

**Read that "1 distinct colour" narrowly — it is not "nothing was written".** `0x20431c0000` is
**`rgba16f`** (66,355,200 bytes = 3840x2160x8) and the dump emits 8-bit RGBA (33,177,600 bytes), so
every HDR value below roughly 1/255 converts to 0 and an entire buffer of small-but-nonzero light
collapses to one colour. That is exactly compatible with the section above measuring **47% of pixels
carrying a value**: the two are the same buffer read through different conversions, and the honest
joint statement is *"lit, at a level that does not survive 8-bit conversion"*. A BMP distinct-colour
count is a **floor** on an HDR surface's content, never a measurement of it.

### What actually writes the lighting buffer

`0x20431c0000` is written by ordinary draws, not only by compute:

| submit | draws | shape |
| --- | --- | --- |
| 37705 | `draw[3..9]`+ | `cwm=7`/`f`, **`blend=1`**, depth test on / depth write off, `topo=4`, vcounts 3, 192, 2016 — additive light volumes, `target1` a 1920x1080 surface with `cwm1=0` |
| 37710 | `draw[2..7]`+ | `cwm=f`, `target1=0x2085de0000` (a G-buffer slot) with **`cwm1=3`**, indexed, 459-1650 indices, `depth=1/1/6` (depth **write** on) |

So the accumulation buffer receives additive blended volumes in 37705 and depth-writing indexed
geometry in 37710. Any account of "lighting applies almost none" has to explain these draws, which do
run and are not among the frame's three failures.

**One correction to the composite description.** The *"three point lights with lens-flare streaks"*
recorded for the gameplay frame are not lights. Dumping `0x20431c0000` after submit 37725 shows a
**shaded 3D object** — an office water cooler, with body, banding and three specular highlights — and
the post chain's bloom is what turns those highlights into the streaks seen on screen. The object is
real geometry that is simply too dark to read after tonemapping. This does not change the chain's
conclusion; it sharpens what "essentially all black" looks like.

### The three failures, named

- **37710 draw, `no-effect`** — `cwm=0`, `target-mask=00000000`. Writes nothing by construction.
  Correctly skipped, benign.
- **37710 dispatch, `0x2042f49a00`** — 29 words, `unsupported=0`, an `image_load` x2 / `image_store` x2
  full-screen 4K copy that rejects at pc 16 with `mode=unresolved-operand`. **Already analysed in this
  document** (the reject table above; and binding 6 -> `0x204da00000`, `class=4` StorageImage, below);
  it is also the reject census's *internal positive control*, its rejecting pc being a key. Its
  offline resource table, now re-derivable from a bundle, is `pc16 -> 0x2052ac0000`,
  `pc18 -> 0x2054aa0000`, `pc21 -> 0x204b1a0000`, `pc25 -> 0x204d180000`.

  **A retraction belongs here.** I first wrote this up as a novel "descriptor-*kind* gap" on the
  grounds that the table shows `descriptors=0` while the instruction needs a T#, and filed #2719.
  That is wrong: `descriptors=` prints `stage.descriptor_issue_count` (`gpu_replay.cpp:1010`) — a
  count of descriptor **issues** — so `descriptors=0` means *zero issues recorded*, the opposite of
  the reading. #2719 is closed as a duplicate whose central claim was a field-name inference. Nothing
  was changed on the back of it.
- **37705 dispatch, `0x205b658800`** — 2935 words, 149 resources, LDS 3072, threads 1920x1080,
  **79 unsupported**, first blocker at pc 82: `s_mov_b32 s6, m0` (SOP1 op 0x03 reading `special:124`).
  M0 is modelled elsewhere in the recompiler (LDS base, ADDTID spill slots), so this is narrower than
  "M0 unsupported"; the exact rejecting path is not located. M0 accounts for 20 of the shader's
  instructions (6 reads, 14 writes); whether all 20 are among the 79 is **not** established.

## Ruled out — HTILE byte-preserving suppression (2026-08-29)

| dead hypothesis | evidence that killed it | ref |
| --- | --- | --- |
| Removing the byte-preserving HTILE suppression is safe for GTA V, because peak colour coverage is 99.78% in both arms (#3093's own clearing check) | **False, and the check is the reason it passed.** `3f5460d0` cost GTA its deferred lighting for a day: the world still covered the frame, drawn with no illumination and a grid artifact from depth-dependent sampling. Peak coverage cannot see a wrongly-lit but fully-covered frame. Bisected from a user report over 52 commits, 4 builds, with 30-frame contact sheets reviewed by eye at each step. | #3121 |
| A *uniform* HTILE plane means a fast clear, so uniformity can discriminate a clear from a HiZ refresh where byte equality cannot | **Measured false BEFORE it was implemented.** `PROSPER_HTILE_UNIFORMLOG` over both titles: GTA **6,500/6,500 writes uniform, zero transitions, first word 0x00000000**; Blue Prince **62,000/62,000 uniform, zero transitions, first word 0x00000000**. The two titles are indistinguishable at this site on every available signal, differing only in plane size (73,728 vs 49,152 words = resolution). A discriminator built on uniformity would have been built on a difference that does not exist. | #3121 |
| Restoring the suppression re-breaks Blue Prince, so the two titles are in tension | **False.** One binary, one environment variable, measured the same day: Blue Prince reaches `max_nonblack` **0.2085 in BOTH arms** — the same "~21%" #3093 called its restored healthy value — while GTA is broken in one and correct in the other. There is no trade. | #3121 |

**Still open, and not to be mistaken for solved:** *why* two byte-identical, uniform, all-zero HTILE
writes need opposite handling in the two titles. The restored exception is behaviour measured
correct for both, not an explanation. Decoding HTILE to tell a clear from a refresh — named in the
code comment since long before this — remains the actual fix.

## Ruled out (2026-08-19)

- **The full offline dissection pipeline is verified end to end on a fresh gameplay bundle, and the
  exact invocations are recorded because three of the four are easy to get wrong.** Captured on the
  stable four-decline baseline: **1.7 GB, 75 submits, 1,415 operations**, replays `RC=0`, and the
  replayed composite is pixel-identical in content to the live frame (HUD, radar, tutorial text over
  black) — so the bundle is faithful and the frame is deterministic offline.

  ```bash
  # 1. replay, and emit a target by extent
  gpu_replay --bundle F.prgbundle --bundle-output-target 3840x2160 out.bmp
  # 2. extract ONE submit to a .prgcap  (--bundle is INCOMPATIBLE with --inspect-only)
  gpu_replay --bundle F.prgbundle --bundle-extract-submit <submit-no> sub.prgcap
  # 3. inspect that
  gpu_replay --inspect-only sub.prgcap
  # 4. emit a named target after an operation -- the address needs 0x, and the op must be one
  #    whose draw actually WRITES that address
  gpu_replay --output-target-after 140:0x2058720000 sub.prgcap g.bmp
  ```

  Three traps, all hit: `--bundle` and `--inspect-only` are mutually exclusive (the parenthetical in
  `CLAUDE.md` — "via `--bundle-extract-submit`" — governs all three dissection flags, not just
  `--dump-resource`); the address in `--output-target-after` is parsed with `strtoull(..., 0)`, so a
  leading-zero form is read as **octal** and silently fails the parse; and the operation must be one
  that writes the target, or the tool refuses with `does not write addr=`.

  **What the dissection found, and its limits.** In this frame's 302-operation submit, **297 draws
  bind the three 4K G-buffer surfaces (`0x2085de0000`, `0x2083e00000`, `0x2081e20000`) and write none
  of them — `cwm=0` on every one**, independently reproducing the same observation this document
  records for its own frame's equivalent submit. The submit that does write 4K is a different one,
  where `0x2058720000` takes **137 draws at `cwm=f`**; dumped after operation 140 it is **15%
  non-zero with 88 distinct colours and shows the radar only — no world geometry**.

  **That is one target at one operation and does NOT falsify** the resolved chain's "the world is
  drawn, complete" claim, which rests on 521 dumps across the whole frame. It is recorded as a
  starting point with a working recipe, not as a contradiction. The artifact is retained on the dev
  box so the survey does not need another 14-minute route run. (2026-08-21.)

  **Cost note for whoever runs the full survey — do NOT scan operations blindly.** Each
  `--output-target-after` invocation re-loads the whole `.prgcap` (116 MB for one submit here), so a
  descending scan over a 300-operation submit is 300 full loads for a single target. A sweep written
  that way does not finish. Take the writing draw index `D` straight out of the `--inspect-only`
  listing instead and scan the narrow window `D .. D+80`: operation and draw indices differ only by
  the number of interleaved computes, measured at +5 in one submit here and +17 in another, so a
  window that size is generous. The addresses to cover are the ones whose draws carry a non-zero
  `cwm`; in this frame's post chain that is `0x2056740000`, `0x2058720000`, `0x205a700000`,
  `0x205f1a0000`, `0x205fa20000`, `0x20602a0000`, plus `0x215ed10000` and `0x20471e0000`. And record
  the extent the dump ACTUALLY returned: `0x205f1a0000` declares `extent=3840x2160` on its draws and
  came back 1920x1080 at operation 61, so an address does not pin a resolution within a frame.
  (2026-08-21.)

- **`0x413dc6700`'s SRT slot dw0 carries a LOW-BIT TAG on exactly half its observations — and
  prosper's GTA V packed-pointer path is never reached on this run.** `PROSPER_SRTDUMP=1` on a routed
  run with the other three hangers declined:

  | SRT slot | pointer observations | low bit set |
  | --- | ---: | ---: |
  | **dw0** | 2,140 | **1,074 (50%)** |
  | dw12 | 88 | **0 (0%)** |

  A 50/50 split on one slot and 0/88 on another is a **flag**, not corruption or misalignment — the
  addresses read `…c1`, `…741`, `…8c1`, `…081`, each exactly +1 from the `…c0/…740/…8c0/…080` that
  prosper's own writeback lines use for the same buffers, so prosper is already dropping the bit
  somewhere. **1,793 of the dumps report `nz=2/2`** (non-empty payloads), so this is the resolved
  regime and not the empty-SRT startup window that has voided earlier readings here.

  **CORRECTED WITHIN THE HOUR — the tagged slot is NOT one of the five V# pointers.** This entry
  first said the alternating bit sits on "the slot the five V#s are built from", and built a
  double-buffer-parity argument on it. That is **wrong**. This program loads its five 64-bit pointers
  at `+0x18/+0x20/+0x28/+0x30/+0x50` (§ *The first 88 folds see an EMPTY SRT*), i.e. dwords **6, 8,
  10, 12 and 20**. The slot carrying the 50% tag is **dw0 — byte offset 0, not among them**; and the
  only one of the five that the dump resolved, **dw12 (`+0x30`), is 0 of 88 odd**. Its pointer values
  also sit in the `0x209c…` range, which is where the 120-byte bindings live, not the `0x20f848…`
  traversal tables.

  So the parity reading is **not supported by this measurement**: the slot that alternates is not a
  V# source, and the V# source that was observed does not alternate. What survives is the bare fact
  in the table — one SRT slot carries a low-bit flag on half its observations, and prosper drops that
  bit — with **no established connection to the traversal buffers or to the two orientations**.

  Recorded rather than deleted because it is the fourth mechanism this issue has seen proposed and
  refuted in a day, and because I published the wrong version to the doc, the PR and the issue thread
  before checking which dword offsets the five pointers actually occupy — a check that was two lines
  further up this same document.

  **`packed_pointer` logs zero lines in this run**, so `rdna2_gta5_packed_pointer.cpp` — prosper's
  existing facility for exactly this class of GTA V pointer — is not engaged for this program.

  **What is measured and what is not.** Measured: the counts above, the +1 relationship to the
  writeback addresses, the non-empty regime, the zero packed-pointer hits. **Not** measured, and not
  to be assumed: that the bit *means* parity, that masking it is wrong, or that honouring it would
  change which buffer resolves. The next arm is direct — correlate the bit against the observed
  orientation fold by fold; if the bit tracks the A/B split, it is the selector. (2026-08-21.)

- **The overlapping views carry DIFFERENT STRIDES — 16 and 4 — over the same guest memory, and the
  four "ping-pong" buffers land at offsets 0, 8252, 33024 and 66044 from one span base.** Measured
  2026-08-21 from the binding declarations, all exact:

  | resource | base | size | stride | records |
  | --- | --- | ---: | ---: | ---: |
  | `0x413e1c300`'s span | `0x20f8482140` | 33,024 | **16** | 2064 |
  | the traversal table | `0x20f848417c` | 8,252 | **4** | 2063 |

  Placing the four observed buffer addresses against that span base gives offsets **0**, **8,252**,
  **33,024** and **66,044**. Three of those are exact structure: 8,252 is one table-size in;
  33,024 is *precisely* the span's end address, i.e. the next block's base. **The fourth is not:
  66,044 is 2 x 33,024 minus 4 — one 4-byte record short.**

  So the memory is simultaneously described as 2064 records of 16 bytes and as 2063 records of
  4 bytes, and the "ping-pong pair" is not two instances of one field: `…417c` is *one table into
  block 0* while `…a240` is *offset 0 of block 1*. That is an odd shape for a double-buffer, and it
  sits directly on the addressing suspicion raised by the 2-cycle structure above.

  **What this is and is not.** The strides, sizes, bases and offsets are read from the run's own
  binding declarations and the arithmetic is exact. **Not** established: that a regular array is
  intended, that the 4-byte shortfall at 66,044 is wrong rather than deliberate, or which of the two
  strides describes the guest's real record. All of that needs the guest's own structure, not more
  address arithmetic — and address arithmetic is precisely where a plausible-looking wrong answer is
  easiest to produce.

  **Instrument note, because it cost a run.** `[parentscan]` is gated on `stride == 4`
  (`buffer.resource->stride == 4u`), so it **cannot** scan the stride-16 span. An attempt to read the
  neighbouring view of the same bytes with `PROSPER_COMPUTE_PARENTSCAN=413e1c300` produced **zero**
  scan lines for ten minutes — inert by construction, not a negative result. The "compare the two
  views" arm named above therefore needs an instrument that does not assume a 4-byte record.
  (2026-08-21.)

- **The cyclic table holds STRUCTURED data, not uninitialised garbage — and every cycle sampled is a
  2-CYCLE between a pair of records whose words differ only in bit 30.** `[parentscan-ring]` samples
  from the run that took the device loss (2026-08-21):

  ```
  idx=412 word=0x00000cf2 -> next=414      idx=414 word=0x40000ce2 -> next=412
  idx=420 word=0x00000d32 -> next=422      idx=422 word=0x40000d22 -> next=420
  idx=428 word=0x00000d72 -> next=430      idx=430 word=0x40000d62 -> next=428
  ```

  Read against the guest's own extract (`v_bfe_u32 v1, v1, 3, 27` at pc95 — bits [3:29], so bit 30 is
  masked off): `0x00000cf2 >> 3 = 414` and `0x40000ce2 >> 3 & 0x7ffffff = 412`. The pair points at
  itself.

  **What the shape rules out.** Every sampled word carries low-three-bits `= 2` and each pair
  differs by exactly `0x10` in the index field and by bit 30 (`0x40000000`); the pairs are
  `(n, n+2)` and the pairs themselves are spaced 8 apart. Uninitialised memory does not look like
  this, and neither does a random functional graph — whose cycle-length distribution would not be
  uniformly 2. So **"prosper never populated this allocation" is dead**, and so is "the traversal
  walks noise". The records are real and regular.

  **What it does not establish**, and this is where the next session should start rather than
  assume: whether bit 30 is a flag the *consumer* is meant to honour (a parent/last-sibling marker,
  say), whether the traversal is reading the intended field at all, or whether prosper's V# base or
  stride for this binding is off by a record so that each entry returns its neighbour's link. All
  three produce exactly this signature. The guest's own extract ignores bit 30, so if the pairing is
  real data correctly read, the same 2-cycle would hang a PS5 — which makes "correctly read" the
  least likely of the three and puts the *addressing* of this binding first in line.

  Cheap next arm: dump the same records through a different binding's view of that memory (§ *four
  span granularities*, above) and compare — if the neighbouring 33,024-byte view yields different
  words at the same guest addresses, the addressing is wrong; if identical, the data is. (2026-08-21.)

- **A STABLE 840 s baseline exists: decline all FOUR hanging programs and the route runs clean, with
  a composite that never stales.** Measured 2026-08-21, `tools/screenshot`,
  `PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700,0x413e14900,0x413e16400,0x413d88400`:

  ```
  done: 42/42 screenshot(s)  stop=request-satisfied  source-distinct=42  pixel-distinct=42
        max-source-stale=0.0s  max-pixel-stale=0.0s  guest=running  status=ok
  ```

  **Zero device losses across the whole 840 s**, and **42 of 42 frames pixel-distinct** — the
  composite is live end to end rather than the frozen-after-loss pattern every earlier configuration
  shows. Compare: the previous best in this document is a 400 s run that was *capped* at 400 s and
  did not finish the route, and the runs recorded beside it lose the device and then hold an
  identical CRC for the remainder. This one finishes on its own terms.

  Reaching this needed the fourth program (§ *A FOURTH program hangs*); the three-program decline
  still lost the device.

  **This is a platform, NOT a result about the hang — and the distinction is the point.** Declining a
  program removes its hang *and* its output. `0x413dc6700` is the one program measurably feeding the
  lighting, so the world being absent here is exactly what a needed-but-skipped producer looks like.
  **Nothing in this run bears on whether fixing the hang would light the world.** § *Skipping is not
  fixing* already says this; it is repeated here because a clean 42/42 `status=ok` line is unusually
  easy to mistake for a passing result.

  What it is good for: every further experiment on this title now has a configuration that reaches
  gameplay, stays there, and never loses the device — so a lever's effect can be read without the
  confound of a run that dies at a different moment each time. Given that submit ordering varies run
  to run (#2516), that confound has been present in most measurements taken here. (2026-08-21.)

- **A NAMED MECHANISM for the producer/consumer gap: eight programs reach the traversal table through
  FOUR different span granularities, and prosper gives overlapping guest ranges independent host
  buffers.** `PROSPER_COMPUTE_ADDRESS_WATCH=0x20f848417c` on a routed run, **22,327 hits**:

  | program(s) | base | size |
  | --- | --- | ---: |
  | `0x413cea300` | `0x20f8480000` | 347,040 |
  | `0x413d88400` | `0x20f8480100` | 132,032 |
  | `0x413ce3400`, `0x413ce6000`, `0x413cee500`, `0x413e1c300` | `0x20f8482140` | 33,024 |
  | `0x413dc3400`, `0x413dc6700` | `0x20f848417c` | **8,252** |

  Six programs bind a span that *contains* the table; two bind the 8,252-byte table itself. The
  33,024-byte span is four consecutive tables, so the whole ping-pong set lives inside one resource
  from four programs' point of view and as separate resources from the traversal's.

  **Why that is a hazard here, read from the source rather than assumed.** Each resource is
  materialized into its own `VkBuffer`. The in-dispatch alias check refuses to share one unless
  `gpu_addr` **and** `size` match exactly (plus several other fields), so an 8,252-byte table and a
  33,024-byte span over the same bytes are two independent host copies. Persistent buffers are cached
  on `ComputeBufferCacheKey{gpu_addr, host_data, bytes, materialization}`, and
  `invalidate_cached_buffer_source` takes **that exact key** — there is no overlap-keyed
  invalidation. So a write through the span's copy does not invalidate the table's cached copy, and a
  read through the table's copy can see bytes that predate it.

  That is exactly the shape the measurements demand: prosper's own writes are clean 43/43 (above),
  the reading program did not write the cyclic table (`changed=0`, above), and yet the table it reads
  is cyclic. A stale copy is a producer/consumer gap that needs no corrupting writer — which is also
  why every attempt to name a corrupting *program* has failed.

  **The WITHIN-program case is NOT the bug — checked, because it looks exactly like one.** The same
  census shows `0x413dc6700` reaching this memory through **nine** bindings at one identical
  `base=0x20f848417c size=8252`: loads at bindings 4/5/8 (fetch-pc 53/65/**91**, the traversal read)
  and stores at 28/30/31/32/33/34 (fetch-pc **618/629/641/653/665/677** — the six phase-2 stores).
  Nine resources over one span is a natural place to suspect lost updates, and the writeback log
  *looks* damning: binding 4 reports `changed=641` while binding 8 reports `changed=0` on the same
  address. It is benign. The alias check's conditions are all satisfied here, and on a match
  `buffers[i]` takes the owner's `buffer` and `memory` and the owner inherits `writable` — one
  VkBuffer, one writeback by the owner, `changed=0` on every alias by construction. So the six stores
  and the traversal load share storage exactly as they should, and this program does **not** lose its
  own writes. The hazard above is strictly the **cross-program, cross-dispatch** one, where the spans
  differ and the persistent cache keys differ with them.

  **REFUTED, same day, by the size gate — the persistent cache is NOT the vector.**
  `persistent_compute_buffer_enabled()` ends `return enabled && bytes >= (1u << 20)`: the persistent
  buffer cache applies only at **1 MiB and above**. Every span in the table above is far below it —
  347,040 / 132,032 / 33,024 / **8,252** bytes — so **none of these resources is ever persistently
  cached**, and the cross-dispatch staleness argument cannot operate on them. Each dispatch creates
  its buffer, uploads from guest memory, and writes back.
  The measured half of this entry stands (eight programs, four granularities, 22,327 hits, the
  exact-match alias condition, the exact-key invalidation); the *inference* built on top of it does
  not. Recorded rather than deleted because the reasoning was checkable in one line and I did not
  check it before writing the entry.
  **The consequence is a real narrowing, and it points the opposite way:** if every dispatch reads
  these tables fresh from guest memory, then the cyclic graph the traversal consumes is genuinely
  *in guest memory* — not an artifact of prosper's buffer management. Combined with "prosper's own
  writes are clean 43/43" and "the reading dispatch did not write it (`changed=0`)", what remains is
  a producer that is not the measured compute chain: the guest's own CPU-side construction over
  inputs prosper supplied, or a GPU path outside these dispatches.

  **What is measured and what is not.** Measured: everything listed above. **Not** measured: that a
  lost update actually occurs on this route — no arm yet shows a specific write through one range
  failing to appear through another. That is the next experiment, and it is cheap: watch one address
  through both granularities across a dispatch pair and compare the bytes. Until then this is a
  mechanism with strong circumstantial support, not a demonstrated defect. (2026-08-21.)

- **PROSPER'S OWN WRITES ARE CLEAN: every table it actually writes comes out acyclic and fully
  terminating, 43 of 43.** This is the causal test done *within a single dispatch* — scanning the
  table after the dispatch that wrote it — so it is immune to the cross-dispatch attribution error
  that produced the retracted root cause. Routed run, Linux/RADV, `reach-story-mode`,
  `PROSPER_COMPUTE_PARENTSCAN=413dc6700`, **with no dispatch skipping** (the parent-walk lever is
  deliberately absent, because arming it declines dispatches and would remove the very writes under
  test).

  **140 post-dispatch scans**, 35 each on four bindings/addresses:

  | outcome | count |
  | --- | ---: |
  | `cyclic=0 terminating=2063` | **138** |
  | `cyclic=156 terminating=1907` | 2 |
  | scans where the dispatch **changed** the table | 43 |
  | …of those, acyclic | **43 / 43** |

  A representative clean write: `binding=4 addr=0x20f848a240 records=2063 terminating=2063 cyclic=0
  longest=13 changed=2062`. So when this program runs and writes, it produces a well-formed tree.

  **The two cyclic scans are the whole story, and they both carry `changed=0`:**

  ```
  [parentscan-after] submit=5160 dispatch=38 binding=8 addr=0x20f848417c cyclic=156 cycle-nodes=104 longest=57 changed=0
  [parentscan-after] submit=5160 dispatch=39 binding=4 addr=0x20f848417c cyclic=156 cycle-nodes=104 longest=57 changed=0
  [compute] fatal Vulkan device loss ... program=0x413dc6700 submit=5160 dispatch=39
  ```

  The table was **not written by those dispatches** — it arrived cyclic — and the device loss lands on
  exactly the dispatch that consumes it.

  **So the frontier is now the PRODUCER of `0x20f848417c`'s cyclic content, and it is not this
  program's execution.** Combined with the two results above — translation cannot silently corrupt
  these tables, and the adjacent-program attribution is dead over 1,507 measurements — what remains is
  a producer/consumer question: the traversal consumes a buffer that nobody in the measured compute
  chain wrote correctly, while the buffers that *are* written come out clean. That is the same shape
  as the open producer/consumer gap on the shadow atlas.

  **Bounds on this.** 35 scans per binding is a modest sample from one run, and the run took a device
  loss (after which live compute is disabled process-wide, so nothing later is measured). The scan
  fires only for `stride == 4` resources, so it says nothing about the 64-byte record buffer. And it
  reports the state *after* the dispatch, which is not the same as proving no other agent wrote
  between the scan and the next read. (2026-08-21.)

- **The adjacent-program attribution for the cyclic table is DEAD, now with the negative case the
  earlier retraction demanded.** Routed run, Linux/RADV, `reach-story-mode`, 800 s, two other hanging
  programs declined, `PROSPER_COMPUTE_PARENT_WALK` armed on `0x413dc6700` — **1,507 walk
  measurements**, the largest sample this question has had.

  Reads at `fetch-pc=91` land on exactly two addresses, the ping-pong pair. Splitting each read by
  **which program ran immediately before it**:

  | read address | previous program | cyclic | clean |
  | --- | --- | ---: | ---: |
  | `0x20f848417c` | `0x413dc6700` | 670 | 15 |
  | `0x20f848417c` | `0x413dc3400` | 134 | 3 |
  | `0x20f848a240` | `0x413dc6700` | **0** | 685 |

  `0x20f848417c` is **97.8% cyclic when `0x413dc6700` ran before it and 97.8% cyclic when
  `0x413dc3400` did.** The predecessor does not move the outcome at all, so no program in that
  position explains it — which is precisely the control the 2026-08-14 retraction said was missing,
  now run and now negative. `0x413dc3400`'s section header above still calls it "THE CORRUPTING
  PROGRAM"; on this evidence it is not, and neither is `0x413dc6700`.

  **What DOES separate the two populations is the address itself** — 804/822 cyclic on `…417c`
  against 0/685 on `…a240`, which also differ in shape (`max-depth` 42-68 versus a shallow 11-13, at
  identical `records=2063 roots=2063`). So the question is no longer *"which program corrupts the
  table"* but *"why is one of the two ping-pong buffers persistently cyclic while the other never
  is"*.

  Two soundness checks on the instrument, both passed, because a 97% rate is worthless if the walk is
  not a function of what it reads: across **614** distinct content hashes on `…417c` and **42** on
  `…a240`, **zero hashes produced both verdicts**; and there were **zero** clean->cyclic transitions
  with an unchanged content hash. The walk is deterministic in its input and the table never flips
  without being written.

  **The caveat, which bounds the `…a240` half specifically.** `PROSPER_COMPUTE_PARENT_WALK` is not
  passive: it **declined 882 dispatches** in this run ("DIAGNOSTIC-ONLY skip suspicious dispatch"),
  which is also why the run took zero device losses. 670 of `…a240`'s 685 reads follow a predecessor
  with `previous-executed=0`, i.e. one the instrument itself skipped — so its perfect cleanliness is
  partly "not recently written" and must not be read as "this buffer is healthy". The **417c
  comparison is unaffected**: both of its predecessor populations are equally subject to the skip, so
  the negative result above stands on its own. A control that separates them needs a run where the
  skip is off, which costs a device loss. (2026-08-21.)

- **Every one of the EIGHT programs that write the traversal table has a fully-covered VECTOR data
  path — the link values cannot be wrong through a dropped VALU or a dropped memory op.** First
  census of the whole writer set rather than of `0x413dc6700` alone, made possible by the
  all-sites enumeration in #2798; before it, only the first blocked site of each program was visible.
  Raw programs extracted from the capture with `gpu_replay --dump-compute-raw` (the *programs* are
  real even though that capture's descriptors are empty — see the EMPTY-SRT row above) and censused
  with `shader_inspect`:

  | program | dwords | blocked sites | of which VALU |
  | --- | ---: | ---: | ---: |
  | `0x413ce3400` | 507 | 22 | **0** |
  | `0x413ce6000` | 276 | 10 | **0** |
  | `0x413cea300` | 1 | 0 | **0** |
  | `0x413cee500` | 279 | 2 | **0** |
  | `0x413d88400` | 372 | 12 | **0** |
  | `0x413dc3400` | 882 | 23 | **0** |
  | `0x413dc6700` | 903 | 27 | **0** |
  | `0x413e1c300` | 179 | 6 | **0** |

  `0x413cea300` decoding to a single dword independently corroborates this document's own
  description of it as "the terminator-only program, whose whole body is one `s_endpgm`", which is
  what makes the extraction trustworthy rather than merely self-consistent.

  Across all 102 blocked sites the classes are: **85 scalar branches** (SOPP `0x8`/`0x2`/`0x4`/`0x6`/
  `0x5`), **8 SMEM `0x01`** descriptor loads (misfiled — #2797), and **9 scalar ALU** (SOP1: seven
  `s_bitset1_b32`, one `s_ff1_i32_b64`, one `s_bcnt1_i32_b64`). **Zero vector ALU and zero
  MUBUF/MTBUF/MIMG data operations.** Since each record's link value is computed in VGPRs and stored
  from a VGPR, the vector data path of every writer is entirely inside the recompiler's
  per-instruction coverage.

  **Two limits, because this census is easy to over-read.** (a) *Scalar* arithmetic is NOT zero — the
  nine SOP1 sites are data-path ops, and a VALU may consume an SGPR one of them produced, so "no
  VALU" is not "no arithmetic". (b) The shell's classification is **context-free**: it runs
  table-less, so a site it calls blocked may be handled by the real emitter given a resource table.
  It bounds where a *translation gap* could be; by itself it does not certify the emitted values.
  (2026-08-21, #2798.)

- **CLOSING both of those limits: no translation gap can SILENTLY produce a wrong link value, in any
  of the eight.** The residual left open above is closed by the recompiler's own contract plus one
  measurement, and neither step depends on reading the census.

  The contract: an instruction the emitter cannot handle sets `ok = false`, and the caller in
  `rdna2_emit_cfg.cpp` then **returns false for the whole program** — it does not emit a wrong value
  and continue. So a translation gap costs you the entire shader, which surfaces as a skipped
  dispatch and a `[recompile-reject]` line. It is *fail-visible by construction*.

  The measurement: **all eight writers emitted SPIR-V** — 435, 3481, 5375, 9969, 15230, 15361, 55221
  and 58649 bytes respectively, read from the `shader=` field of `gpu_replay --inspect-only`. None
  was rejected. Therefore no instruction in any of them reached the reject path, and every one of the
  102 sites the table-less shell called blocked is in fact handled by the real emitter with context —
  including all nine SOP1 sites (`s_bitset1_b32`, `s_ff1_i32_b64` and `s_bcnt1_i32_b64` each have
  emitter paths in `rdna2_emit_alu.cpp` / `rdna2_emit_cfg.cpp`).

  **So the two failure modes are disjoint and only one is live.** A translation defect in these
  programs would present as *content that never draws*; it cannot present as *a table that cycles*.
  The cyclic table is therefore produced by something other than a missing instruction — the guest's
  own computation over inputs prosper supplied, or a defect outside the recompiler. That is where the
  remaining work is, and it needs a live route. (2026-08-21, #2798.)

- **The offline dissection of `0x413dc6700` describes the EMPTY-SRT startup variant, not the hanging
  one.** `dc6700.prgcap` was taken with `PROSPER_GPU_CAPTURE_COMPUTE_ADDR` and **no**
  `PROSPER_GPU_CAPTURE_AT=`, so it fired inside the first-88-fold window that
  § *The first 88 folds see an EMPTY SRT* describes. Measured on the derived disassembly: **45**
  `StorageBuffer` variables declared, **exactly one** access chain into any of them, 980 Workgroup,
  0 image ops — against **41** guest MUBUF sites (`shader_inspect` on the same raw dump). That is the
  empty-SRT signature exactly, down to the single surviving non-MUBUF chain. Two consequences:
  (a) the *"the pc88..97 cycle appears to TERMINATE"* reading is explained rather than in tension with
  the device witness — the pointer chase folded to a constant, so the loop exits for that reason and
  says nothing about the resolved variant; (b) the proposed ctest that embeds the program's 903 raw
  dwords is **void as specified**, since raw dwords carry no descriptors and recompiling them
  reproduces the terminating variant. `shader_inspect` states this on the dump itself:
  `status=undetermined-no-resource-table`, `table_dependent=41`, and the rejection is
  **unattributable**. Any arm must supply a *resolved* table. (2026-08-20, #2542.)

- **`dispatch-range=N..M` in a `[cfg-trip-bound] HIT` line does NOT bound the runaway cycle.** It is an
  atomic min/max over switch-case ordinals observed across *all* invocations of that dispatch, so a wave
  that walks 0->8 and then spins, while other invocations run to the end, prints `0..14` — the whole
  table — for a two-block cycle. Phase 0's dispatcher graph, derived from every write of the next-block
  variable (the constant writes plus the taken/untaken pair the vote resolves), is:
  `0->{1,6} 1->{2,3} 2->3 3->{4,5} 4->5 5->6 6->{7,11} 7->8 8->{9,10} 9->8 10->11 11->{12,13} 12->13
  13->14 14->exit`. **The only cycle is 8<->9**; every other edge moves strictly forward and there is no
  back edge to block 0. This is the third direction in which this witness's fields have been misread —
  see trap 172. (2026-08-20, #2542.)

- **"The vote never resolves, so the dispatcher re-enters the same block" is not the mechanism.** The
  vote's final write of the next-block variable is guarded by a conjunction of the has-branch flag and
  the negations of two others; those two are stored `false` in the phase preamble and never rewritten
  anywhere in phase 0, while every branching block sets the has-branch flag, so the guard reduces to
  that flag alone and the branch target *is* applied. (2026-08-20, #2542.)

- **`robustBufferAccess` is enabled on the compute device, so "OOB buffer loads are undefined here" is
  not available as an explanation for the traversal failing to terminate.** `live_compute.cpp` checks
  `supported.robustBufferAccess`, refuses the device without it, sets `enabled.robustBufferAccess =
  VK_TRUE`, and passes it through `pEnabledFeatures`. This matters because the guest loop's only exit is
  the `v_cmpx_ne_u32` on the loaded link, so a load returning 0 past the record count is *what
  terminates it on hardware*, and that contract is switched on. **Not checked, and still open:** whether
  the bound descriptor **range** equals the V#'s `num_records x stride` on the compute path. Vulkan
  bounds a robust access to the bound range, not to the guest's `NUM_RECORDS`, so a looser range would
  return neighbouring bytes where hardware returns 0 — and the FLAT path already documents a deliberate
  "loose-bounds divergence from HW" of exactly this kind. That end-to-end contract has no test (#2795).
  (2026-08-20.)

- **FOLLOW-UP, and it closes the above as an explanation: the MUBUF out-of-bounds contract IS honoured
  on the compute path, so a walk off the end of the traversal buffer DOES terminate.** Traced through
  the source rather than measured, each link checked:
  `pc=91` decodes as `buffer_load_dword v1, v1, s[0:3], 0` with **`idxen=1, offen=0`** (word0
  `0xe0302000`, bit 13 set, bit 12 clear), so the address is a record *index* and hardware bounds it at
  `index >= NUM_RECORDS`. `plan_storage_buffer_materialization` sets `binding_bytes = resource.size` on
  the default path; `live_compute.cpp` then uses that same value for both `VkBufferCreateInfo::size`
  and `VkDescriptorBufferInfo::range`, and enables `robustBufferAccess`. For this descriptor
  (`stride=4, num_records=2063, size=8252`) the bound range is exactly 2063 dwords, the lowering is a
  dword-indexed read, and index >= 2063 falls outside the range and reads 0 — which is the guest loop's
  termination condition. The remaining `zero_pad_*` paths pad with **zeros**, so they preserve the same
  answer rather than breaking it. Note the recompiler emits **no** runtime record-count bound at all
  (`num_records` appears zero times in `rdna2_emit_alu.cpp`); the contract is delegated entirely to the
  descriptor range, which is why it is worth having written down.
  **Consequence for the hang:** the traversal has exactly two exits — a zero link and an out-of-range
  index — and *both* work. So `0x413dc6700` can only spin on a table whose links form a cycle **inside**
  `[0, 2063)`, which is what the parent-walk census independently reports (`cyclic-roots=2062,
  oob-roots=0`). That makes the frontier a **data-production** question — what writes those values —
  and not a translation or bounds question. (2026-08-21.)

  **UPGRADED from a source trace to an EXECUTED measurement the same day (#2800).** The claim above
  was reasoned through the code; it is now run. `tests/gpu/recompiler/test_traversal_chase.cpp` is a
  hand-built kernel carrying this exact shape — the EXEC-narrowing loop plus a real MUBUF load through
  a V#, with encodings derived from the RDNA2 field layouts rather than lifted from the capture — and
  it executes on real Vulkan over link arrays the test supplies, so the data is known-acyclic by
  construction and only the lowering is on trial. Four arms pass: a descending chain gives each lane
  its own chain length (distinct per lane, so a mask-ignoring lowering fails); all-zero links give
  depth 1, proving depths follow the DATA and not the loop shape; **all-out-of-range links give depth
  exactly 2** — step one hands back the out-of-range successor, step two reads the zero
  robustBufferAccess supplies (measured `0, 2, 2, …, 2`); and the emitted module is checked to contain
  a real access chain, without which every arm would pass for the wrong reason on a folded load.
  So this shape is lowered correctly end to end and the translation is **not** the hang's cause.
  Scope, stated so the pass is not over-read: the buffer is bound by the test harness, so this pins
  the recompiler's half. It does **not** exercise `live_compute.cpp`'s `binding_bytes` computation,
  which is what makes the bound range equal `num_records x stride` on a real dispatch — that remains
  the open coverage gap on #2795. (2026-08-21, #2800.)

- **Any predicate over `num_records` or `size_bytes` is UNFALSIFIABLE on the `reach-story-mode`
  route — the run cannot contain a counterexample.** Four successive attempts to classify a
  runtime-selected descriptor-table record as "not a descriptor" were refuted (#2481): page
  residency, region identity, saturated size, and a `num_records` threshold drawn from a census.
  The fourth is the instructive one, because the census *looked* like the two-population control the
  first three lacked.

  It was not, and the arithmetic shows why. `size_bytes = num_records × stride` with a **14-bit**
  stride field, and a record is declined at `size_bytes > 256 MiB`. So a size-declined record must
  carry `num_records ≥ 16,386`, and an accepted one `num_records ≤ 268,435,456` — the only window
  where **either** label is attainable is `[16386, 268435456]`. Measured over 382 accepted and 54
  declined records, accepts top out at **668** and declines start at **973 million**: every single
  record fell *outside* that window. Consequently every threshold in `(668, 973M)` scores an
  identical `54/54` with `0/382` false positives — 10,000, 65,535, 1 M, 200 M all tie. The scorecard
  was measuring the decline filter, not the descriptors.

  A `size > 256 MiB` row in the same table is worse still: ACCEPT is only reachable *past* that
  exact test, so `0/382` there is forced by the source and could never have printed anything else.

  And the threshold dies on a hand-built instance without needing a second title: with `stride == 0`
  **`NUM_RECORDS` is a BYTE count** (RDNA3 §8.4.1, #2528), so any raw buffer ≥ 64 KiB carries
  `num_records ≥ 65536` by definition of the field.

  **So: the next proposal must use a field the decline filter does not, or a route that produces
  records inside `[16386, 268435456]`.** What survives from the census as genuinely empirical is only
  the accept-side observation that no accepted record exceeded 65535 — accepts were free to, since
  `stride=16, num_records=1M` is 16 MiB and would be accepted — and that **two** accepted records are
  legitimately formatless (`stride=112 records=267`), so `fmt == 0` is not a descriptor test either:
  untyped loads move raw dwords regardless of declared format.

- **`0x205b658800` is NOT the lighting pass.** It is a rejected full-screen 1080p compute in the
  submit whose 4K output is black, which made it an attractive candidate. Its 149 resources reference
  **none** of the surfaces the resolved-chain table names — not `0x20431c0000`, not `0x207de60000`,
  `0x2085de0000`, `0x2081e20000` or `0x2083e00000`, and not `0x2052ac0000` (0 hits each). It
  references a 4K surface `0x215ed10000` ten times, so post or composite is the likelier role.
  Hypothesis withdrawn before it was acted on.
- **The `PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT` lever is not a workaround for the bundle gate.** A
  bundle captured under it writes but cannot be reconstructed: `cannot reconstruct bundle submit 1:
  indirect-pointer relocation has stale shader, launch, source, or proof state`. Its own
  *NOT FAITHFUL REPLAY* warning understates it — the bundle is not replayable at all.
- **The RT-group dumps that reported "no world in this target" sampled only menu frames.**
  `PROSPER_DUMP_RTGROUPS` clears `live_gpu_targets` for the whole run (`live_renderer.cpp:1459`, a
  `static const`), forcing CPU readback at every pass; under that load this route never leaves the
  main menu, and the last screenshot of every such run is the menu. Any "target X does not contain the
  world" conclusion from those runs is void — they never reached gameplay.

## Three hanging compute programs, not one — and the structurizer bypass behind them (2026-08-18)

`0x413dc6700` is **not** the only program that hangs the GPU on this route. Declining it and re-running
finds a second; declining both finds a third. The count is a **lower bound** — each decline only reveals
the next one, and no run has yet finished the route with zero device losses.

| # | program | found by declining | device loss at |
| --- | --- | --- | --- |
| 1 | `0x413dc6700` | *(nothing)* | `submit=9016 dispatch=40` |
| 2 | `0x413e14900` | #1 | `submit=34930 dispatch=135` |
| 3 | `0x413e16400` | #1, #2 | `submit=3628 dispatch=2946` |
| 4 | `0x413d88400` | #2, #3 declined + #1 trip-bounded | `submit=4476 dispatch=54` (2026-08-21) |

The submit indices are **not** monotonic down the table, and that is not a typo: each row is a
different run whose trajectory changed when a program was declined, and submit ordering already varies
run to run on this title (#2516). Read the column as "where this run died", never as a progression.

Filed as **#2690**. `PROSPER_COMPUTE_SKIP_PROGRAM` has always accepted a comma list
(`frontends/shared/live/live_compute.cpp:4247`), but every run in this document's history declined only one
program, which cannot reveal a second — it only moves the loss past where most captures stop. With all
three declined a run reached **400 s** with zero losses, but that run was **capped at 400 s and did not
finish the route**, so it is not a zero-loss completion and does not close the search. Read the two
sentences together: three is what declining twice revealed, not a total.

**A FOURTH program hangs, found 2026-08-21, and it extends the lower bound rather than closing it.**
Routed run with `0x413e14900` and `0x413e16400` declined and `0x413dc6700` trip-bounded
(`PROSPER_CFG_TRIP_BOUND=4096 _PROGRAM=0x413dc6700 _PHASE=0`, **647 HIT lines**, so the bound was
genuinely armed and firing). The device still went:

```
[compute] fatal Vulkan device loss stage=queue-submit result=VK_ERROR_DEVICE_LOST(-4)
          program=0x413d88400 submit=4476 dispatch=54 order=4205; disabling live compute
```

`0x413d88400` was already in this document as a resource-census entry — it binds `0x20f8480100`
size 132,032, one of the four span granularities over the traversal table — but had never been seen
taking the device. It is the same pattern the table above records: each decline reveals the next one,
so four is still a lower bound.

**Consequence for the trip-bound lever's reported result.** That loss lands at ~100 s, well before
gameplay: the composite CRC freezes at shot 04 (100 s) and is byte-identical for the remaining
740 s of the run. So this run cannot speak to the "~7x more distinct colours" measurement recorded
below — it never reached the state that measurement describes, and the two are not in conflict.
Submit ordering varies run to run on this title (#2516), so which program takes the device first
varies with it.

**`0x413e16400` is the first with a named fallback cause**, from the reporter added in #2684:

```
[divloop-reject] program=0x413e16400 role=emit condition-region-not-branch-free at pc=230 ...
```

suggesting a chain from a named guard to a hung GPU: *guard refuses → every loop in the shader
discarded → CFG dispatcher → emulated loop → `VK_ERROR_DEVICE_LOST`*.

**Two of those arrows are weaker than the chain reads, and both are load-bearing.**

- *→ CFG dispatcher* is **not** implied by the reject, and this document will not say what replaces
  it. An empty loop list is one input among several to a decision taken in `emit_body`, over values
  the reject site cannot see. That is the whole of what is established here.
  **Three successive attempts to say something sharper were each false**, which is why the sharper
  sentence is gone rather than refined: "falls back to the CFG dispatcher" (the reporter's own line
  until #2700); "a shader with two branches and a refused loop is rejected outright and emits
  nothing" (false — `portable_compute_dpp_ror8` reaches a dispatcher before `complex_compute_cfg` is
  consulted, and `exact_compute_wave_cfg` has no branch-count term); and "every dispatcher entry
  requires `allow_cfg_dispatcher`" (false — `emit_phase` and the LDS fminmax path do not consult it,
  and the flag is not inherited across re-entry). The third was written to fix the second and was
  narrower than it. **A narrower claim about another function is still a claim about another
  function.** The trailing parenthetical is elided in the quote above so this block does not go stale
  when #2700 lands.
- *→ `VK_ERROR_DEVICE_LOST`* is **correlation plus mechanism, not proof**. The
  `RADV_DEBUG=hang` / `vm_fault.log` check that established "genuine hang, not OOB write" has been run
  for `0x413dc6700` only, never for either new program.

### Barrier-phased programs bypass BOTH structured paths

`0x413dc6700` emitted no `[divloop-reject]` line in the runs to date. **Do not read that as "it can
never reach the structured paths" — an earlier draft of this section said exactly that, and it is
wrong.** The bypass is conditional:

- the phased branch is entered only when
  `phased.guarded || initial_dispatch_active || force_barrier_phases`
  (`rdna2_to_spirv.cpp:22055`), and `guarded && initial_dispatch_active` is refused at `:22079`;
- `emit_phase` routes a phase to `emit_cfg_state_machine` only when `!guarded || initial_dispatch_active`
  (`:22101`); otherwise the phase re-enters `emit_body` (`:22106`) and **can** report.

**All three disjuncts matter, and the third is the one to look at first.** `force_barrier_phases` is
how the *phased retry* is entered (`:22866`) — with `!guarded` and `initial_dispatch_active == 0` —
so an enumeration over only the first two makes that retry look unreachable and quietly contradicts
the paragraph below. It also happens to be the most plausible route for this program: it is
barrier-phased, a guest `s_barrier` clears `cfg_dispatch_safe` (`:22717`), so it reaches `:22856`,
takes the `!cfg_dispatch_safe` arm and hits the force retry. A reader working from a two-valued gate
would rule out exactly the path that would settle the question.

So "reaches neither structured path" holds only for `!phased.guarded && initial_dispatch_active != 0`
**on the first entry**. In the guarded case every phase re-enters `emit_body` and can emit
`[structured-wave-reject]`; on the force-retry route the retry is reached only *through* that
reporter's own condition, so it necessarily emits one first. **Which case `0x413dc6700` is in has not
been established**, and per #2690 the absence of a line is not by itself proof of anything — but the
force-retry route above is the cheapest thing to check next, and needs no device loss.

An experiment that routed loop-bearing phases into `emit_body` was tried and **did not move the
outcome**: all three phases still produced dispatcher maps, because `emit_body` has its own gate —
`allow_cfg_dispatcher && exact_compute_wave_cfg && !structured_compute_wave_cfg` (`:22856`) — that
outranks the loop path.

### Conjunct census for the non-phased population (#2695)

Over 1,366 compute recompiles on this route: 906 `exact_wave=0`, **23** dispatcher cases, 437 structured.
The 23 resolve to five distinct `(program, conjunct)` pairs — **4 `nested-wave-forward-if`**, 1
`cross-lane-mbcnt`. A prediction that `cross-lane-mbcnt` would dominate was wrong — it is the rarer of the two. The
reasoning behind the prediction was wrong too, and in a way worth recording: it rested on
`cross-lane-mbcnt` being *the* conjunct with no `native_subgroup_size` escape, when in fact three of
the four reported conjuncts lack one. That claim also reached #2695's commit message, where it is now
uncorrectable, so it is corrected here.

### Ruled out by the 2026-08-18 runs

These belong to the canonical `## Ruled out` section far below; they are kept beside their evidence
here and deliberately use a distinct heading so the two do not collide as anchors.

- **"The guest flip stall discriminates the compute hang."** Falsified by its own control. `present_count`
  (guest `present_flip` only) reads 0.19–0.32 of `frame_seq` with 131–300 s stalls in *both* the default
  and `SKIP` arms. The stall follows the device loss that precedes it, and in the pair measured here
  both arms lost the device — at different times, from different programs. **Two other runs did not lose it:** § *Gameplay
  frame characterised end to end* records two that declined `0x413dc6700` alone with **zero** losses.
  That is an observation, not a mechanism — per the standard in § *Three hanging compute programs, not
  one*, neither finished the route (one stopped with the guest still running, one on its own 520 s
  timeout), so their zero losses are equally consistent with a window that ended before the next
  program's loss. **Nothing in the record separates
  that from nondeterminism**, and the flip measurement here records no duration, no loss timestamp and
  no list of which programs its `SKIP` arm declined. An early partial read showed `SKIP` at
  1.00 and looked like a clean discriminator; that was simply the control not having reached the
  stalling stage yet. (2026-08-18, #2542.)
- **"`PROSPER_CFG_TRIP_BOUND=N` + `_PROGRAM=` arms a bound."** It does not. `_PHASE` is **required**, and
  without it the emitter prints `PROSPER_CFG_TRIP_BOUND_PHASE is REQUIRED and is unset: no bound emitted`
  once, via `call_once`, and emits nothing. Three experiments were run on top of this assumption before
  the line was read. The rule is already stated in this document's own env table — the failure was not
  reading it. The **22,117-colour measurement recorded below** (§ *RESOLVED CHAIN*) **did** set
  `_PHASE=0` and stands; it was publicly doubted before that was checked, and the retraction is on
  #2542. (2026-08-18.)
- **"Implementing the missing end-of-pipe data patch lights the world."** It does not.
  `sceAgcQueueEndOfPipeActionPatchData` (#2708) is genuinely missing and now lands — 8,116 successful
  writes, **zero** refused — but `0x413dc6700` still loses the device at `dispatch=40`, and after that
  loss the composite freezes for the rest of the run (pixel CRC identical t=120s..600s). **This result
  is not scoped to any particular theory of the black world**: it stands whatever the cause turns out
  to be, so do not read it as bearing only on #2705's SRT question. An earlier version of this row
  headlined it as "what leaves the resource table zero" — that premise is **withdrawn**: a live
  memprobe shows 209 of 286 folds carrying non-zero data (#2704).
  The pre-patch payload establishes something worse than "the value was stale": `data_sel=2` routes to
  an 8-byte fence write, so before this change prosper was **writing `0xfffffffffffffffe` as the fence
  value on every one of those packets that executed**. 8,116 is the count of successful patches and is
  an upper bound on the writes, not a count of them: `honor_eop_write` returns early on `!rel_addr`,
  misalignment and `!guest_readable`, only submitted-and-folded packets execute, and nothing shows the
  packets target distinct addresses. (2026-08-18, #2708, #2704.)

## RESOLVED CHAIN: the world IS rendered, and then lit by nothing (2026-08-18)

**GTA V's world is not missing. It is drawn correctly, in full, and then receives almost no light.**

| stage | surface | measured state |
| --- | --- | --- |
| geometry -> G-buffer | `0x207de60000` (slot 0), `0x2085de0000` s4, `0x2081e20000` s3/s1, `0x2083e00000` s2 | **the world, complete** — 474 of 521 dumps with content, richest **99.66% non-zero**, the prologue bank heist fully textured at 4K with legible signage |
| lighting reads the G-buffer | same | **works** — the room's geometry resolves in the lighting output |
| lighting applies light | `0x20431c0000` 3840x2160 `Float16` | **almost none** — 47% of pixels carry a value, essentially all black |
| composite -> scanout | three 4K RGBA8 | **faithful** |

So geometry, vertex fetch, texture decode, materials, rasterization and the recompiler's fragment
output are all **correct**. The defect is one edge: the lighting stage's light input.

### `0x413dc6700` measurably feeds the lighting

With `PROSPER_CFG_TRIP_BOUND=4096 _PROGRAM=0x413dc6700 _PHASE=0` the program **completes** instead of
hanging (17 `HIT` lines confirm the cap fired on that program), gameplay is reached, and the frame
gains **~7x more distinct colours** (≈3,000 -> 22,117) with a visibly larger, brighter light source.

**That is the only intervention that has ever changed this title's light content.** Depth (three
levers), draw rejection, `HIT-CPU`, DCC decode and the composite all changed nothing.

Raising the bound to 100,000 does **not** extend the trend — the run loses the device before
gameplay and collapses to 2 colours. That is loss timing, not dose, and no dose conclusion may be
drawn from the pair: while any dispatch can still hang, a higher bound merely trades truncated output
for an earlier loss.

### The frontier, as one question

**Does a correctly-executing `0x413dc6700` light the world?** No configuration can answer it yet:
unbounded it hangs at dispatch 40, bounded it truncates its own output. Fixing the hang is therefore
the direct route to a lit world, and this is the first evidence tying the hang to the darkness by
mechanism rather than by coincidence of timing.

### A correction that understated its own evidence

The bridge-override arm was first reported as **"73 overrides"**. That was the number of *log lines*
across **7** addresses — and the instrument emits on a **power-of-two schedule**, so lines are not
events. The main depth's own last line reads `count=32768`.

So the arm covered **at least 32,768 samples on `0x2052ac0000`**, not 73: the depth falsification is
roughly **450x better supported** than it was written up as. Recorded because the error ran in the
unusual direction — an under-claim, which nobody is motivated to check, and which would have left the
next reader thinking the strongest negative result in this document rested on a few dozen samples.

Two separate mistakes produced it: counting emitted lines as if they were events, and quoting a
sampled counter without reading its own schedule. Both are cheap to avoid — the schedule is three
lines above the print.

### Choose instruments by census, and beware what they perturb

Two rules this pass paid for repeatedly:

- **`PROSPER_DRAW_CENSUS` does not disable `live_gpu_targets`; `PROSPER_DUMP_RTGROUPS`,
  `PROSPER_DUMP_RTGROUPS_RGBA`, `PROSPER_DUMP_DRAWSTEPS` and `PROSPER_RTTLOG` all do.** Those force
  the CPU readback path, so every per-target pixel measurement carries an asterisk an on-screen
  measurement does not.
- **Identify a surface by draw census, never by shape or sample count.** `0x208e5a0000` was called
  "the world's geometry" and is a 1024x1536 `R8_UNORM` shadow atlas; `0x205f1a0000` looks like a
  scene buffer and takes 66 draws against `0x20431c0000`'s 647. Both misidentifications cost hours.
- **A byte-wise fill is a value fill only on a byte-shaped surface.** `0xff` into a Float32 depth is
  **NaN**, not 1.0 (#2680) — a two-pole experiment run that way tests `{0.0f, NaN}` and silently
  never tests far.
- **Scan a dump fully before believing a null.** BMP rows are stored **bottom-up**, so sampling the
  first megabyte of a 4K dump samples the bottom strip: the lighting output read 0% that way and
  47% on a full scan.

## THE G-BUFFER IS THE FRAME'S BUSIEST TARGET, and the break is inside two stages (2026-08-18)

`PROSPER_DRAW_CENSUS=1`, routed gameplay. This tool is **not** in the `live_gpu_targets` exclusion
list, so unlike `PROSPER_DUMP_RTGROUPS` / `PROSPER_RTTLOG` / `PROSPER_DUMP_DRAWSTEPS` it measures the
renderer in its real configuration rather than forcing CPU readback. Prefer it for any target
question.

```
[draw-census] draws=262144 indirect=0 submit=26578
[draw-census]   distinct colour targets (sampled): 95
[draw-census]   target=0x2085de0000 slot=4 draws=7978
[draw-census]   target=0x2081e20000 slot=3 draws=7275
[draw-census]   target=0x2083e00000 slot=2 draws=6672
[draw-census]   target=0x208e5a0000 slot=0 draws=4921
[draw-census]   target=0x2081e20000 slot=1 draws=3262
```

**~24,000 draws across slots 1-4 of a multi-slot G-buffer**, whose surfaces are then sampled ~1,250
times each by the lighting pass. The deferred pipeline exists and is wired end to end.

### The pipeline, stage by stage, with what is known about each

| stage | surface | state |
| --- | --- | --- |
| geometry -> G-buffer | `0x2085de0000` s4, `0x2081e20000` s3/s1, `0x2083e00000` s2 | ~24,000 draws; **contents unmeasured** |
| lighting samples G-buffer | same | ~1,250 samples each, resolve `HIT-CPU` |
| scene colour | `0x205f1a0000` 1920x1080 `Float10_11_11` | **one light blob, 0.88% non-zero** |
| composite -> scanout | three 4K RGBA8 | runs, ~195 passes each |

So the break is in one of the first two stages, and everything after them is exonerated.

### Eliminated, each with a controlled experiment

- **depth** — three levers, every one confirmed to have fired: `0.0f`, a confirmed `1.0f`
  (`path=f32 filled=1f`), and serving the real retained depth despite `dvalid=0` (**>= 32,768 overrides on the main depth alone**).
  No world at any. Note the earlier "both poles" falsification was **void** — the byte fill made the
  far pole NaN on a Float32 depth (#2680).
- **draw rejection** — zero `[render] skip draw` lines across six routed runs (unconditional
  diagnostic). #2429's wave64 skip does not fire on AMD Radeon 8060S / RADV STRIX_HALO.
- **`HIT-CPU`** — dominates in *Blue Prince*, which renders correctly, so it is the ordinary path.
  What differs is the **miss shape**: GTA V misses the *same renderer-owned* surfaces repeatedly
  (depth 161x, five shadow maps 110x each) where Blue Prince has 19 single first-sight misses on
  guest textures.
- **DCC decode (#2677)** — wrong signature; it produces noise, not black.
- **the composite** — it carries faithfully what the scene buffer holds.

### The measurement that is next, and the trap that blocked it

Whether the G-buffer **contains** anything. `PROSPER_DUMP_RTGROUPS_ADDR` filters on the pass's
**base**, which is the **slot-0** attachment — filtering by `0x2083e00000` (slot 2) matched exactly
one pass in a full run for a target taking 6,672 draws. That is void by wrong key, not an empty
G-buffer. Either find the G-buffer pass's slot-0 base from the same census, or extend the filter to
match any slot.

**Also correcting a claim this document carried:** `0x208e5a0000` is neither the dominant target
(4,921 draws, not ~71,000 of them) nor a scene buffer — it is 1024x1536 `VK_FORMAT_R8_UNORM`, one
byte per pixel, and renders empty across 1,755 passes. For a shadow pass that writes depth and masks
colour, empty colour is **correct**. Its emptiness was never the anomaly it appeared to be.

## Gameplay frame characterised end to end, with the hang declined (2026-08-18)

Two routed runs, Linux/RADV, `scripts/gta5/reach-story-mode.pad`, 4K, `tools/screenshot`, with
`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700`. **Zero device losses in both**, guest still running at the
end of the first; the second ended on its own 520 s timeout. 40 and 69 samples respectively.

**What renders, and what does not.** At gameplay the composite contains:

- the **radar/minimap** — complete: player arrow, three blips, compass rose with `N`, and the
  green/blue/gold legend bar;
- the **tutorial caption** — `The Radar shows your position within the world.`, every glyph present;
- **three point lights with lens-flare streaks**, plus very faint blue structure toward the right;
- **no geometry at all.**

So the lighting and post path reaches the scanout while the geometry it illuminates does not. The 2D
and UI composite is in good shape — the 4K loading screen (Michael, skyline, logo, and
`Entering Story Mode: (90%)`) renders essentially correctly.

**Route note.** This route enters the **prologue** (`Ludendorff, North Yankton, nine years ago.`
renders correctly over black), not the free-roam resume the route README describes. The prologue
legitimately opens on black before fading in, so a capture that stops around 200 s will read as "black
world" for a reason that has nothing to do with the renderer. The first run did exactly that; the
second is why the state above is stated at all. **Capture past 400 s on this route** before concluding
anything about the world.

### Skipping is not fixing — a correction worth keeping

It is tempting to read "world still absent with `0x413dc6700` declined" as evidence against #2542's
framing. **It is not.** A declined dispatch produces nothing, so an absent world is exactly what that
route predicts if the program is a world producer. The SKIP route can show that the *rest* of the
frame survives without a device loss; it cannot demonstrate a correct world, and no run under it
should be quoted as if it could.

### Compute rejects on this frame: one shared mode, and it is not a missing emitter

Nine distinct compute programs are declined, **every one `mode=unresolved-operand`**, across five
instruction families (MUBUF, DS, MIMG, SOP1, VOP3). Per `rdna2_to_spirv.cpp` that mode means the
lowering ran and could not resolve an operand or a resource-table descriptor — so these are descriptor
failures, not absent lowerings.

The ungated `[compute-table]` dump answers the next question directly: **does the rejecting
instruction's own fetch pc appear as a key in the table that program was handed?**

| program | reject pc | keys | pc is a key | min key |
| --- | ---: | ---: | :---: | ---: |
| `0x2042f49a00` | 16 | 4 | **YES** | 16 |
| `0x205b5e8600` | 314 | 75 | NO | 6 |
| `0x205b654a00` | 1180 | 82 | NO | 16 |
| `0x205b657200` | 313 | 34 | NO | 16 |
| `0x205b658800` | 82 | 150 | NO | 6 |
| `0x413cf9200` | 5 | 9 | NO | 64 |
| `0x413cf9a00` | 39 | 5 | NO | 11 |
| `0x413cf9d00` | 70 | 16 | NO | 26 |
| `0x413d14100` | 6 | 10 | NO | 41 |

**8 of 9 reject at a pc for which no resource was offered.** Two — `0x413cf9200` at pc 5 and
`0x413d14100` at pc 6 — reject *before the first key in their own table*, at the top of the program
where a root pointer is loaded. `0x413cf9200` pc 5 decodes as
`buffer_load_dword v17, v0, s[4:7], 0`: an SGPR-based descriptor, and every resource in that table
carries `sgpr=0xffffffff`, so nothing is keyed by SGPR base either.

**This rules out the framing in `gpu_executor.cpp`** that a resource with
`srt=0xffffffff sgpr=0xffffffff` is "invisible to every lookup": the same diagnostic reports
`0 with no lookup key` for all nine, because those entries are still keyed by fetch pc. The failure is
not "the resource has no key" but "no resource was offered for the site that needs one".

Two caveats, both load-bearing:

- **No comparison population.** `[compute-table]` prints only for *rejected* programs (it sits inside
  `report_compute_recompile_skip_once`), so every row above is a reject by construction. This cannot
  show the pattern discriminates rejected from accepted programs; that needs the table dumped for
  accepted ones, which no switch does today.
- **`0x2042f49a00` is the internal positive control** — its rejecting pc *is* a key, so the analysis
  can print YES, and the eight NOs are not a matcher that always misses.

### The sampled depth is declined, and the reason printed is the wrong one

```
[render] DCC-compressed sampled image addr=0x2052ac0000 meta=0x2055310000
         3840x2160x1 fmt=1 tile=24 is unsupported; metadata=0/0 first=0x00
```

`0x2052ac0000` is this title's main depth and `0x2055310000` its HTILE — neither is DCC. The renderer's
sampled-image path never consults `sampled_source_decision()`, and — corrected in review — the
metadata-kind correlator had **no production caller anywhere in the tree**. A query is registered for
it in `live_renderer.cpp`, which made it natural to assume the compute path used it; registering a
query is not calling one, and on master every reference to `classify_compression_metadata_kind`
outside its own definition is a test. Filed as **#2674**.

**A correction, kept because the reasoning error is the instructive part.** The first analysis
concluded that gate 2 (the metadata footprint) is what blocks this surface, and attributed it to
`gfx10_htile_msaa_metadata_bytes()` fail-closing on `sample_count != 4`. That conclusion is not
established by the evidence, because the size expression is

```cpp
!has_ds_live && r.compression_enabled ? gpu_capture_dcc_metadata_footprint(r) : 0u
```

so `metadata=0/0` is *equally* consistent with `has_ds_live` being true — in which case no footprint
helper is consulted at all and the gate discussion is irrelevant. For a title's main depth buffer,
which the renderer demonstrably renders into, that branch is not exotic. The branch that fitted the
hypothesis already held was the one chosen; `ds_live=` was added to the diagnostic so the next run
answers it as a measurement. `first=0x00` was likewise the *unread default*, not a reading of the
plane.

**Measured, 2026-08-18: `ds_live=0`** — and this needs reading precisely, because the obvious reading
is wrong. It does **not** mean "no retained DS image exists for this surface". The lookup that sets
it is itself gated on `!has_live_rtt && !is_storage_image && img_dim == 1 && cls == Texture`
(`live_renderer.cpp:2899`), so a binding the gate rejects never reaches the lookup and reads 0 for a
reason unrelated to residency — on an `img_dim == 6` surface it is *structurally* 0.

What survives is the narrower claim, and it is the one that matters: `has_ds_live` is false, therefore
`!has_ds_live && r.compression_enabled` holds, therefore the size expression **did call a sizing
helper** and that helper returned zero.

**Which helper, and therefore which gate, is still undetermined.** `dcc_metadata_footprint`
(`gpu_capture.cpp`) reaches the HTILE sizer only when **all three** of `img_dim == 6`,
`format == Float32` and `num_components == 1` hold, and otherwise uses the DCC sizer — which
fail-closes on tile mode and **never looks at `sample_count`**. So "the 4xAA gate blocked it" and "it
was never routed to the HTILE sizer at all" are both consistent with everything measured so far.

Two of those three conjuncts were unprinted, not one. #2679 adds `dim=` **and `ncomp=`**, because
`fmt=` cannot stand in for the component count: it prints the `DataFormat` enum, and `Float32` is
reached from raw IMG_FMT 22/64/74/77 with one, two, three or four components
(`agc_shader_layout.cpp`). Note the asymmetry that remains even so — `dim != 6` is **decisive** (the
HTILE sizer cannot have run), while `dim == 6` alone is not. Until all three are read, no gate
conclusion should be quoted from this section.

That is the second withdrawal on this surface, and the pattern is worth naming: each time, a
measurement narrowed the question without settling it, and each time the tempting move was to treat
the surviving hypothesis as confirmed because it was the last one standing. It was not confirmed;
there was simply no instrument pointed at its rival.

What is measured, in full: `kind=HTILE` (the correlator classifies it correctly on this path),
`tile=24` matching `TileMode::Sw64KbZX`, `pipe_aligned=1`, `ds_live=0`, and `samples=1`.

Two readings of that set remain **mutually exclusive and not yet separated**. If the routing predicate
held, the HTILE sizer ran and the sample count is the failing gate; if it did not, the DCC sizer ran,
fail-closed on tile mode, and the sample count was never consulted at all. `dim=` and `ncomp=` are
printed as of #2679 so the next routed run separates them. The function's own comment records that the sample-count gate is conservatism rather than
arithmetic:

> HTILE sizing is independent of the depth sample count, but this API deliberately retains the
> observed 4xaa gate rather than claiming broader support.

`meta_pipe_aligned` (T# word6 bit 19, `agc_shader_layout.cpp:386`) is the third condition. It, the
sample count, `ds_live` and `img_dim` were all absent from the decline line, so the log could not say
which gate failed; #2679 prints them. Making the gate legible is the prerequisite for deciding
whether the 4xAA gate can widen with evidence rather than by inference from a comment.

### Also observed

- **#2445 (dropped `r`/`s`/`m` glyphs) does not reproduce on Linux/RADV.** Both strings the issue
  names render complete here. That is not a falsification — the issue is Windows/NVIDIA — but it
  suggests a platform asymmetry, which is a much smaller search than a general text defect.

## THE CORRUPTING PROGRAM IS `0x413dc3400` (2026-08-15)

Measured with `PROSPER_COMPUTE_TREE_WATCH=0x20f848417c:2063` on a 200 s routed run with the hanging
consumer skipped (`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700`, so zero device losses, 9,291 frames).
The watch reads the table before and after **every** realized dispatch, so a change is attributed to
the dispatch that made it rather than to an interval.

**321 observations. Every clean -> cyclic transition:**

| program | clean -> cyclic | cyclic -> clean | clean -> clean |
| --- | --- | --- | --- |
| `0x413dc3400` | **37** | 0 | 3 |
| `0x413d88400` | 1 | 37 | 2 |
| `0x413e1c300` | 1 | 1 | 158 |
| `0x413cee500` | 0 | 1 | 79 |

`0x413dc3400` made the table cyclic on **37 of its 40 dispatches**. Nothing else does it
systematically.

**Every observed writer reported `toucher=1`.** No change came from a program whose resource table
does not contain the address, so there is no unknown writer and no out-of-bounds path to chase —
the watch was built to be able to report that case and did not.

### The per-submit chain, identical every submit

```
d15,d16  0x413cee500   Morton keys        -> pairs=0     unpaired=0    oob-roots=0   depth=3
d37      0x413dc3400   topology build     -> cycles=19   pairs=919     unpaired=148  depth=68
d39..47  0x413dc6700   the consumer       (skipped in this run; this is what hangs)
d54      0x413d88400   repurposes the RAM -> pairs=0     unpaired=571
```

`0x413dc3400` writes 2,061 of 2,063 records in one dispatch through six store pcs (bindings/pcs
`23:597, 25:608, 26:620, 27:632, 28:644, 29:656`), turning a table with **no** cycles into one with
19 cycles, 57 cyclic roots and depth 68. The consumer runs two dispatches later and walks exactly
that array. **This is the defect: the builder produces a cyclic parent array, and the hang is the
downstream symptom.**

The same run also shows `0x413d88400` at d54 leaving `pairs=0` — the allocation is **reused scratch**
across phases, not one long-lived structure, which is why a pair count compared across phases moves
by hundreds. Compare pair counts only within the same phase.

### What a correct table looks like — measured, not assumed

Against 91 captured 2,063-dword tables:

| | pairs | unpaired | cycles | max depth |
| --- | --- | --- | --- | --- |
| clean parent tables (`live-a240`, `post-36-3`) | **1030** | **0** | 0 | **11** |
| cyclic captures (85, one resource hash) | 1029 | 1 | 1 | 15 |
| `0x413dc3400` output, live | 919 | 148 | 19 | 68 |

This **confirms Codex's LBVH identification quantitatively**: a full binary tree over 1,032 leaves
has 1,031 internal nodes, and a well-formed table measures 1,030 sibling pairs with zero unpaired
records at depth 11, against log2(1032) = 10.01. Pairs are `(odd, even)`: the odd index is the head
(side bit clear), the even index its mate (side bit set).

In the 85-capture set the anomaly is *fully deterministic* — the same 4-member cycle
`(256, 384, 447, 831)` and the same single unpaired record `300` in every one. All 85 share one
resource hash, so this is one table observed 85 times, i.e. the corruption is **stable** across
submits 7629 -> 9694 rather than re-derived; nothing repairs it. The four cycle members are each in
a *well-formed* sibling pair, so the cycle is not caused by a broken pair — the parent links
themselves are wrong. Slot 300 differs from its mate only in bits[2:0] (`0x942` against `0x940`,
both decoding to parent 296), so it is a flag-bit difference and a separate anomaly from the cycle.

**Ruled out by this measurement:** the "lost update" / two-writer race account of the cycle. Within
one dispatch's pre/post pair there is exactly one writer, and its output is cyclic.

## Falsified for `0x413dc3400` (2026-08-15) — checked, not assumed

- **Barrier uniformity.** All eight `OpControlBarrier`s in the emitted module are in uniform blocks.
  Three sit in each dispatcher's **continue target** (`OpLoopMerge %163 %162` — %162 is the continue
  block, reached by every invocation on every iteration) and five at structured merge targets. The
  design comment in `rdna2_to_spirv.cpp` states this invariant explicitly ("the switch merge is
  reached by every invocation on every iteration"); it holds in the artefact. A barrier inside a
  `switch(pc)` *case* would have been a real Vulkan uniformity violation — it is not what is emitted.
- **Native subgroup / multiwave lowering.** `[subgroup] cs=0x413dc3400 … native=0 … multiwave=0`:
  the program is already lowered through the portable wave model, so there is no native lowering to
  blame. `PROSPER_NO_NATIVE_COMPUTE_MULTIWAVE=1` leaves all nine of its module hashes byte-identical.
- **LDS undersizing.** The module declares 384 dwords = 1,536 bytes = 3 × 512-byte
  `COMPUTE_PGM_RSRC2.LDS_SIZE` granules; Codex's ISA read puts the largest accessed LDS address at
  byte 1,028. Sized correctly and over-provisioned either way.
- **A race.** Two different frames produce broken-pair patterns sharing a 60-character suffix
  exactly, with the same first damaged index. Deterministic given the input.
- **A lane/wave/workgroup boundary effect.** Damage index mod 2/3/4/8/16 is flat.
- **An unknown or out-of-bounds writer.** Every observed change reported `toucher=1`.

## The selected-sbuffer contract declines because the SELECTED V# IS FLOAT DATA (2026-08-15)

`PROSPER_GTA5_SBUFFER_REJECT=1` (added on this branch; the six reasons in
`rdna2_gta5_compute_contracts.cpp` were all behind `PROSPER_DBG`, which desyncs the route) reports
exactly two declines for `0x413ce6000` on a whole run:

```
[gta-selected-sbuffer] reject=consumer-resource
[gta-selected-sbuffer] reject=selected-vsharp
    words=c540fa56:c51e1625:4373fd8a:45de36cc
    base=0x1625c540fa56 stride=1310 records=1131675018 size=4294967295
```

**Those four words are floats**: `0xc540fa56` ≈ **-3087.6**, `0xc51e1625` ≈ **-2529.5**,
`0x4373fd8a` ≈ **244.0**, `0x45de36cc` ≈ **7110.9**. That is a world-space AABB, not a descriptor —
and the derived `base=0x1625c540fa56`, `records=1131675018`, `size=0xffffffff` are what you get from
reinterpreting it. **prosper is correctly refusing to manufacture a descriptor out of it**; the
contract's `selected-vsharp` guard is doing its job.

So the selector resolves to a record that does not contain a V# at the expected offset. The chain to
audit, with what is measured about each link:

| link | measured |
| --- | --- |
| the selector's source records, pc70 | `base=0x20f848e2bc stride=8 records=2064 size=16512` — resolves |
| the descriptor array, pc153 | `base=0x203f249b38 stride=120 records=5 size=600` — resolves |
| `s_buffer_load_dwordx4 s[8:11], s[4:7], s106` | SMEM immediate offset **8**, so the V# is expected at `selector*120 + 8` |
| the selected element | **float AABB data** |

Three candidates, none yet excluded: the selector value is wrong; the V# lives at a different offset
within the 120-byte record than `+8`; or the **source records at pc70 are themselves stale because
their producer also does not run** — which would make this a chain of missing producers rather than
one. That last one is the possibility to test first, because it is the same failure this whole
investigation has already found once.

Note the resource map lifts pc156/pc158 as `base=0x203f2e9b38 size=13360 stride=20 entries=5` —
**stride 20, against the contract's expected 120**. Reconciling those two views is likely the fastest
route in.

## The selector chain at `0x413ce6000` pc149..156 — the exact fix site

Decoded with prosper's own opcode constants (`rdna2_decode.hpp`), not guessed:

```
pc149  s_mulk_i32            s106, 120          ; SOPK 0x10. selector * 120, in VCC_LO as scratch
pc150  s_load_dwordx4        s[4:7], s[0:1], m0 ; the descriptor-ARRAY V#
pc153  s_buffer_load_dwordx4 s[8:11], s[4:7], s106  ; SELECT one V# at selector*120
pc156  buffer_load_dwordx3   v[0:2], v6, s[8:11], 0 ; use it   <-- mode=unresolved-operand
```

**120 is exactly the array's stride.** `PROSPER_DYNTRACE_FAIL` confirms the split:

```
BUF(v4) key=0xa8       use_pc=153  base=0x203f249b38 stride=120 records=5 size=600   RESOLVED
BUF(v4) key=0xffffffff use_pc=156  v4=00000000:00000000:00000000:00000000            UNRESOLVED
```

So the descriptor-array lift **finds the array** at pc153 and **cannot resolve the selected element**
at pc156: `key=0xffffffff` is the sentinel the executor's own comment names as matchable by none of
the three routes (fetch pc, SRT offset, SGPR base).

**Why the selector does not resolve. `CONFIDENCE: MED`.** The selector arrives through **VCC_LO used
as an ordinary scalar register** — GTA V's compiler recycles it, which this file already documents
elsewhere. prosper's VCC-as-scalar recognition is
`is_wave64_vcc_lo_scalar_b32_candidate`, and it covers exactly two shapes: `s_cselect_b32` with
scalar-data operands (inline constants, SGPRs, or VCC_LO itself since #2741 — it was inline-only when
this was written), and SOP2 B32 logicals. **`s_mulk_i32` is SOPK and is in neither set**, so the write
at pc149 is not recognised as a scalar-scratch definition. That matches the failure exactly, but the
alternative — that the const-fold breaks somewhere else along `s106`'s chain — has not been
separately excluded, so this is a lead and not a conclusion. Verify before building on it.

This is the same underlying difficulty as the execz VCC-half liveness guard cleared earlier today:
**every remaining obstacle in this program comes from the guest recycling VCC as a general scalar
register, and prosper modelling VCC specially in each place independently.**

## THE REMAINING BLOCKER, EXACTLY (2026-08-15)

Every compute reject on a routed run now names its cause without `PROSPER_DBG`. `0x413ce6000` — the
producer whose absence removes the world — is blocked by **one instruction**:

```
0x413ce6000  mode=unresolved-operand pc=156 words=e0382000,80020006 fmt=12 op=0xe
```

`fmt=12` is MUBUF, `op=0xe` is `buffer_load_dwordx3`. `mode=unresolved-operand` means the **lowering
exists and the descriptor does not resolve** — the emitter is fine, the descriptor is the defect.

pc156 is the **runtime-selected buffer array**: `PROSPER_COMPUTE_RESOURCE_MAP` shows it resolving in
the live table as `binding=10 fetch-pc=156 base=0x203f2e9b38 size=13360 stride=20 entries=5`, while
the pre-specialization const-fold trace (`PROSPER_DYNTRACE_FAIL`) shows it as
`use_pc=156 v4=00000000:00000000:00000000:00000000 base=0x0 stride=0 records=0`. Its sibling at
pc158 is the same shape. Those two are the only unresolved uses of the nineteen.

**So the whole "GTA V has no 3D world" chain reduces to one buffer-array descriptor that does not
const-fold at pc156 of `0x413ce6000`.** That is the next thing to implement.

> **Read the table below as first sightings, not as outcomes — 2026-08-16.** Every row is a program
> that rejected *at least once*. `[compute] skip unsupported program` prints **once per program
> address** by design (`report_compute_recompile_skip_once`), and the skip itself is a `continue`
> inside the per-dispatch loop, so each dispatch is decided again. A row therefore says nothing about
> how often the program runs, and this document previously read the twelve rows as twelve disabled
> programs. `PROSPER_COMPUTE_PROGRAM_CENSUS=1` reports the ratio, and it splits them in two:
>
> | mostly runs (executed / skipped) | never runs (executed / skipped) |
> | --- | --- |
> | `0x413cf9200` 369 / 12 · `0x413dc6700` 352 / 11 | `0x2042f49a00` 0 / 129 |
> | `0x413cf9a00` 351 / 30 · `0x413cee500` 349 / 9 | `0x205b545c00` 0 / 127 · `0x205b54ee00` 0 / 129 |
> | `0x413d63700` 344 / 14 · `0x413d14100` 343 / 38 | `0x205b5e8600` 0 / 128 · `0x205b654a00` 0 / 128 |
> | `0x413cf9d00` 342 / 42 · `0x413ce6000` 103 / 16 | `0x205b657200` 0 / 128 · `0x205b658800` 0 / 128 |
> | `0x413d85e00` 137 / 2 (+6 more at 32–35 / 1) | `0x413ce5200` 0 / 93 · `0x413e1df00` 0 / 85 |
>
> **Update 2026-08-16: the right column is now SEVEN.** `0x205b545c00` and `0x205b54ee00` execute
> every dispatch since the DPP lane-XOR family was admitted (`ROW_XMASK:n`, `0x160..0x16f`, is the
> same permutation family as the already-admitted `ROW_ROR:8` — `ROW_XMASK:n` is XOR n, and XOR 8 is
> exactly `(row_lane - 8) mod 16`, so the lowering was general and only the decoder's admitted control
> values were not). Both went `executed=0 skipped=52` → zero skips on the same route. The remaining
> seven need, in order of tractability: a VCC_LO operand that resolves to no mask value
> (`0x205b5e8600` — **not** cross-block dataflow; see `## Ruled out`), an untracked M0
> read (`0x205b658800`), `image_bvh_intersect_ray` (`0x205b654a00`, `0x205b657200`), `IMAGE_LOAD_MIP`
> on a compressed surface (`0x2042f49a00`), and a descriptor contract (`0x413ce5200`, `0x413e1df00`).
>
> **The frontier is the seven on the right** (the count above this line, not the nine it was before
> the preceding update cleared two), and note that `0x413cf9200` — the program carrying the
> entire hardcoded 15-site contract in `rdna2_gta5_cf9200_contract.cpp` — is on the *left*, running
> 369 of 381 dispatches.
>
> The two columns need different work. The left column is a **descriptor-timing** problem, now with a
> measured mechanism: `PROSPER_COMPUTE_MEMPROBE=413cf9200:0:c0:4` shows the first fold reading
> `bfe767f8:7ee0001a:40e938ea:bfaa07e0` at `(user_sgprs[0..1])+0xc0` — SOPP instruction words, i.e.
> the SRT is not written yet — while every later fold reads `9cc76000:00200020:0000000a:00005204`, a
> valid V# (`base=0x209cc76000`, `stride=32`, `records=10`). One probe reports `CHANGED` between them
> directly. The right column rejects on six *different* opcodes and is a genuine recompiler gap.

The full reject census, now legible. Every one is `mode=unresolved-operand` — an **operand** the
recompiler could not resolve, rather than an instruction it does not implement.

**That is as far as the mode goes, and this line used to claim more.** It read *"i.e. every one is a
descriptor that does not resolve"*, which is true of the MUBUF / MIMG / SMEM / FLAT rows and **false
of `0x205b5e8600`**, whose unresolved operand is a *register* (VCC_LO used as scratch scalar data —
see `## Ruled out`). The retracted gloss is what made "descriptor" the assumed shape of every row
here, and it sent this kernel's frontier item off as cross-block lane-mask dataflow, which its
disassembly does not contain. Read the mode as "some operand did not resolve" and open the row's
instruction before assuming which kind:

| program | pc | fmt/op |
| --- | --- | --- |
| `0x413ce6000` | 156 | MUBUF `buffer_load_dwordx3` |
| `0x413cf9200` | 5 | MUBUF `buffer_load_dword` |
| `0x413cf9a00` | 11 | MUBUF `buffer_load_dword` |
| `0x413cf9d00` | 70 | FLAT/GLOBAL `op=0xc` |
| `0x413d14100` | 6 | MUBUF `buffer_load_dwordx3` |
| `0x2042f49a00` | 16 | MIMG `op=0x1` |
| `0x2042f4a600` | 7 | SMEM `op=0x4` |
| `0x205b545c00` | 98 | VOP2 `op=0xf` |
| `0x205b54ee00` | 90 | VOP2 `op=0xf` |
| `0x205b5e8600` | 314 | SOP2 `op=0xe` |
| `0x205b654a00` | 1180 | MIMG `op=0xe6` |
| `0x205b657200` | 313 | MIMG `op=0xe6` |
| `0x205b658800` | 82 | SOP1 `op=0x3` |

## REFRAME: this is a RAY-TRACING BVH, and prosper already knows the format (2026-08-15)

The "tag" this investigation has been tracking is the **AMD RDNA2 ray-tracing BVH `NODE_TYPE`**, and
prosper's own recompiler says so. `rdna2_to_spirv.cpp` (search `is_box16`) software-emulates
`IMAGE_BVH_INTERSECT_RAY`, loading 28 dwords of a node and branching on:

```cpp
const uint32_t is_tri0  = b.ucmp(Op_IEqual, node_type, b.uconst(0u));
const uint32_t is_tri1  = b.ucmp(Op_IEqual, node_type, b.uconst(1u));
const uint32_t is_box16 = b.ucmp(Op_IEqual, node_type, b.uconst(4u));   // 64-byte node
const uint32_t is_box32 = b.ucmp(Op_IEqual, node_type, b.uconst(5u));   // 128-byte node
```

So bits[2:0] of a node reference are the node type: 0–3 triangle, **4 box16**, **5 box32**, 6
instance, 7 procedural. Everything this investigation measured now has a name:

| observation | reading |
| --- | --- |
| `0x209cc76000` at 64-byte stride | an array of **64-byte nodes** (box16 or triangle) |
| tags 0 / 2 / 5 / 7 dominating | triangle / triangle2 / box32 / procedural |
| **tag 4 appearing only in broken submits** | **box16 nodes entering the scene** |
| the sibling-paired parent table | BVH topology |
| Morton keys and `0x09249249` | an LBVH build, as Codex identified |
| `0x413dc3400`'s `tag == 2 \|\| tag == 5` predicate | it writes links only for two specific node types |

**`PROSPER_DECODED_BVH` machinery already exists** — `DecodedBvhDescriptor` in
`agc_shader_layout.hpp` carries `box_grow`, `triangle_return_mode`, `box_node_64b`, `sort_enabled`,
and the dynfail dump prints `BVH(bvh4)` descriptors. This is a supported surface, not an unknown one.

### The consumers of the BVH do not compile

`IMAGE_BVH_INTERSECT_RAY` is **MIMG opcode 0xe6** (`rdna2_to_spirv.cpp:14145`). Two programs in the
reject census fail on exactly that instruction:

```
0x205b654a00  mode=unresolved-operand pc=1180 fmt=14 op=0xe6   image_bvh_intersect_ray
0x205b657200  mode=unresolved-operand pc=313  fmt=14 op=0xe6   image_bvh_intersect_ray
```

`mode=unresolved-operand` means **the lowering exists and the BVH descriptor does not resolve**.
prosper has the full software traversal emulation; the shaders that would use it are declined for a
descriptor.

**So there are two independent defects on the ray-tracing path, and the second was invisible until
the reject reasons became readable:**

1. `0x413dc3400` builds a cyclic topology once the scene passes a point — the input-dependent defect
   this document tracks above.
2. **The traversal shaders that consume the BVH never compile at all**, because their BVH descriptor
   does not resolve.

Fixing (1) alone cannot render the world if (2) also holds. **(2) is the better first target**: it is
a descriptor-resolution problem on an instruction prosper already implements, it is named exactly, and
unlike (1) it does not depend on scene state.

## The BVH traversal shader's descriptor is never classified as a BVH (2026-08-15)

`PROSPER_DYNTRACE_FAIL_ADDR=205b654a00`. The shader is a **full-screen ray-tracing pass**:

```
launch groups=240x135x1 threads=1920x1080x1 local=8x8x1 user_sgprs=8
pre-specialization raw const-fold recovered 72 descriptor use(s)
```

**Not one of the 72 is a `BVH(bvh4)`.** The dynfail dump distinguishes three kinds — `TEX/IMG(t8)`,
`BUF(v4)` and `BVH(bvh4)` — and this shader, which executes `image_bvh_intersect_ray` at pc1180,
recovers zero BVH descriptors. Two of its uses are unresolved with a **valid base and zero extent**:

```
BUF(v4) key=0xffffffff use_pc=1032 v4=a1f76200:00000020:00000000:00000000
        base=0x20a1f76200 stride=0 records=0 size=0 required=28
BUF(v4) key=0xffffffff use_pc=1266 v4=a1f76400:00000020:00000000:00000000
        base=0x20a1f76400 stride=0 records=0 size=0 required=124
```

`required=28` is notable: prosper's own BVH emulation loads exactly **28 dwords** per node
(`for (uint32_t k = 0; k < 28; ++k) w[k] = load_node(k);`).

**Reading, `CONFIDENCE: MED`.** These are BVH descriptors being classified and decoded as buffer V#s.
A GFX10 BVH descriptor has its own 4-dword layout — `decode_bvh_descriptor` in
`agc_shader_layout.hpp` reads `type`, `box_grow`, `triangle_return_mode`, `box_node_64b`,
`sort_enabled` from it — and interpreting one as a buffer V# yields `num_records = 0`, hence
`size = 0`, hence an unbounded use, hence `unresolved-operand`. That fits every observation, but the
alternative (the guest genuinely supplies a zero-extent descriptor at this point in the route) is not
excluded, and a `key=0xffffffff` means neither fetch pc, SRT offset nor SGPR base matched — which has
its own possible causes.

**FALSIFIED, by running that check.** Decoding the two words with `decode_bvh_descriptor` gives
`base = 0x20a1f7620000` — the BVH layout shifts its base left by 8, and the result lands far outside
the guest address space, which sits around `0x20xxxxxxxx`. The **buffer** decode gives
`base = 0x20a1f76200`, a perfectly plausible guest address. So these are not misclassified BVH
descriptors, and the reading above is dead. Cost: one four-dword computation, no run.

**What the same words do show.** Dwords 2 and 3 are **entirely zero**, while dwords 0 and 1 carry a
sane base. A real buffer V# has a nonzero dword3 (it carries format and type bits), so an all-zero
upper half is the signature of a **partially recovered descriptor** — the const-fold obtained the
low two dwords and not the high two — rather than of a descriptor the guest genuinely wrote as
zero-extent. `num_records = 0` then follows from dword2 being absent, and `size = 0` from that, and
`unresolved-operand` from that. **That is the next thing to test**, and it is a different defect from
anything this document has chased: not a wrong descriptor, a half-read one.

## THE UNIFYING CAUSE: GTA V recycles VCC_LO as a general scalar register (2026-08-15)

The BVH descriptor at the rejecting instruction is **built in the shader**, and it is built through
VCC_LO. `0x205b654a00` pc1180 is `image_bvh_intersect_ray` with its descriptor in `s[16:19]`:

```
pc1171  s19  = <computed>
pc1174  s106 = s19 & 0x000003ff            ; VCC_LO as scalar scratch
pc1176  s17  = <computed>
pc1177  s19  = s106 | 0x81000000           ; and back out of VCC_LO
pc1179  s_waitcnt
pc1180  image_bvh_intersect_ray  v[0..], v6, s[16:19], s[0..]
```

`(x & 0x3ff) | 0x81000000` is the BVH descriptor's dword3 — its size-high bits and type field. **The
descriptor cannot resolve unless the const-fold tracks a value through VCC_LO.**

That is the same obstacle as everywhere else in this title:

| site | what VCC_LO carries | consequence |
| --- | --- | --- |
| `0x205b654a00` pc1174/1177 | the BVH descriptor's dword3 | `image_bvh_intersect_ray` rejects, the ray-tracing pass never compiles |
| `0x413ce6000` pc149 | `s_mulk_i32 s106, 120`, the descriptor-array selector | `buffer_load_dwordx3` at pc156 rejects |
| `0x413ce6000` pc84/90 | integer scratch inside an execz arm | the structurizer's VCC-half liveness guard rejects (cleared on this branch) |
| GTA V generally | `is_wave64_vcc_lo_scalar_cselect` exists precisely for this | already a known pattern in the code |

**prosper models VCC specially in each place independently — the liveness proof, the scalar-scratch
recogniser, the descriptor const-fold — and each place has its own, narrower notion of which VCC
writes count as data.** `is_wave64_vcc_lo_scalar_b32_candidate` admits exactly `s_cselect_b32` with
scalar-data operands and SOP2 B32 logicals. `s_mulk_i32` (SOPK) is not in it. Neither is the
`s_and_b32`/`s_or_b32` pair above being tracked *through* to a descriptor.

**This is the frontier.** Not the tree builder, whose lowering is proven correct by eleven perfect
submits; and not a missing producer, which is falsified. A single coherent treatment of "VCC_LO used
as an ordinary scalar register" — one recogniser consulted by the liveness proof, the scalar model
and the const-fold alike — is what the remaining rejects have in common.

`CONFIDENCE: MED` on that being sufficient. It is established that the descriptor passes through
VCC_LO and that prosper's VCC recognisers do not cover these shapes; it is *not* established that
covering them is enough to make either program compile, because neither has been tried.

## `s_mulk_i32` folding: landed, and it did NOT move the reject (2026-08-15)

The const-fold's SOPK case handles **only** `s_movk_i32`; every other SOPK forgets its destination
*and* invalidates SCC. Its own comment notes that `s_movk/s_version/s_cmovk/s_mulk` do not write SCC
— so `s_mulk_i32` was being charged both costs it does not owe, and the comment demands per-opcode
evidence before widening. That evidence exists: `0x413ce6000` pc149 is `s_mulk_i32 s106, 120` where
120 is the descriptor array's exact record stride, feeding the select at pc153 and the rejecting load
at pc156.

Folded it: multiply a known destination, forget an unknown one as before, and stop clobbering SCC.
245/245 ctest green.

**It did not change the reject.** `0x413ce6000` still fails with `mode=unresolved-operand pc=156`.

**Then the probe was run, and it explains why — the fold was aimed at the wrong thing.**

`PROSPER_DYNTRACE_SGPR=106` gives s106's complete fold history in this program:

```
pc=3    KNOWN 0x00000000
pc=56   FORGOTTEN  words=beea376a
pc=84   KNOWN 0x00000000 / 0xfffff7f0
pc=90   KNOWN 0x7f7fffff
pc=116  FORGOTTEN  words=beea3704      <- the break
pc=131  FORGOTTEN  words=beea3704
pc=149  FORGOTTEN  words=b86a0078      <- s_mulk_i32 s106, 120, with s106 ALREADY unknown
```

pc116 and pc131 are `SOP1 op=0x37 s[106:107], s[4:5]`, which prosper classifies as a **B64
data/mask write** and which the RDNA2 encoding makes **`s_andn1_saveexec_b64`**: `exec = ~s4 & exec`,
and **`s106` receives the OLD EXEC MASK**. Both are immediately followed by `s_cbranch_execz`.

**So the descriptor-array selector is derived from a saved EXEC mask.** `s_mulk_i32 s106, 120` is
multiplying a lane mask by the array's record stride. That is a wave-dependent runtime value, not a
constant, and **no const-fold can ever resolve it** — which is precisely why this program has a
dedicated `selected_sbuffer` contract that certifies the selector's complete *domain* from live
source records instead of folding it.

**Conclusion: the `s_mulk_i32` fold does not address this reject and cannot.** The reject is the
contract's `selected-vsharp` decline — record 4 of the outer array holding float AABB data — and the
const-fold was never on that path. The fold change stays for its own reasons; this reject needs the
contract.

The change is kept on its own merits — it is a documented over-conservatism corrected with ISA
backing and it costs nothing — not because it was shown to help.

## FIRST DRAW-LEVEL VIEW OF A GTA V FRAME — and the captured submits are nearly EMPTY (2026-08-15)

The frame grab now works on this title. Three changes were needed, each opt-in so no existing
contract moves:

| switch | what it addresses |
| --- | --- |
| `PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1` | five packed/indirect-pointer provenance validators that abort the whole bundle; reported per acceptance as inspection-only |
| `PROSPER_CAPTURE_WAIT_FOR_SUBMITS=1` | keeps the window open past empty presents (bounded at 240), because a present COUNT is not a unit of time on a burst-flipping title |
| `PROSPER_CAPTURE_MAX_SUBMITS=N` | bounds the bundle by CONTENT — one GTA V burst appended 3.3 GB and blew even the 3,072 MB maximum, so no byte budget could capture it |

Result: **`frame-bundle written (25 submits)`, 512 MB**, loading as
`bundle v2 submits=25 logical=2002872578 unique=511842881 ratio=0.256 chunks=3773 resources=454`.

### What the frame contains

```
[gpureplay] bundle-submit=19370 operations=0/0 output_bytes=0
[gpureplay] bundle-submit=19371 operations=0/0 output_bytes=0
[gpureplay] bundle-submit=19372 operations=0/0 output_bytes=0
[gpureplay] bundle-submit=19373 operations=0/0 output_bytes=0
[gpureplay] bundle-submit=19374 operations=1/1 output_bytes=1048576
```

**Four consecutive captured submits contain ZERO operations, the fifth one, the sixth three.**
Reproduced on an independent capture (submits 23910–23915 against 19370–19374), and
`operations=%zu/%zu` is `limit / total`, so the second figure is the submit's real operation count.

**But this window is NOT representative, and reading it as "the guest submits nothing" would be
wrong.** The tree watch elsewhere in this document reports the BVH builder at `dispatch=37` and the
scratch reuse at `dispatch=54` **within a single submit** — so submits carrying 50+ operations
demonstrably exist on this route. The capture simply landed on a quiet stretch: `MAX_SUBMITS` bounds
it to the *first* 25 of a burst, and the window opens wherever the timed arm falls.

**What it does establish** is that the capture path now works end to end and that submit-level
operation counts are readable. **What it does not** is anything about where the world's draws are.

**The next step is to aim the capture at a submit known to contain work** — the tree watch names
them exactly (any submit at its `d37`) — rather than at a wall-clock moment. `PROSPER_CAPTURE_FRAMES`
arms on time or frame ordinal; landing on a chosen submit needs the arm to be submit-indexed.

**Read with care, `CONFIDENCE: MED`.** Three things are not established: whether `operations=0/0`
means the guest submitted nothing or the capture recorded nothing; whether the 25-submit window
landed on a representative part of the frame (it is bounded by `MAX_SUBMITS`, so it is the *first* 25
of a burst); and whether the world's work lives in submits outside it. Replay stops at bundle submit
5 with `indirect-pointer relocation has stale shader, launch, source, or proof state` — expected,
since the capture was forced past provenance, and it means the later submits were not reconstructed.

**The next step is to widen the cap and see where operations begin**, which is now a matter of one
setting rather than an unusable tool.

## The F9 frame grab cannot be used on this title — diagnosed and filed (#2549)

> **SUPERSEDED (2026-08-19).** The bundle grab works on this title since #2718; a gameplay
> frame captures in 896 MB and replays. See the 2026-08-19 section at the top. The gates this
> section records were real when written.

The charter names the F9 grab the fastest loop for a graphical bug, and GTA V — a GPU-driven title
whose world is absent — is exactly the case it exists for. It could not be run at all, and the
message said only "the capture window contained no GPU submits".

Instrumenting the submit hook with three counters and the window's wall-clock duration named the
cause immediately:

```
[grab] submit hook reached=26840, 19206 while inactive, 7634 while not capturing;
       window was open 26 ms for 48 presents
```

**48 presents in 26 ms**, and 12 presents in 7 ms — about **1,700 presents/second** against a 23/s
average. **GTA V flips in bursts**, so a window defined as a present COUNT has an effectively random
wall-clock duration, usually far too short to contain a submit.

Two further refusals were found and opened behind `PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1`, which
reports every acceptance as inspection-only: five packed/indirect-pointer provenance validators that
abort the whole bundle when one compute dispatch cannot be exactly proven. Right for a replay bundle,
wrong for a bundle meant to be read — and on this title they fire, so the grab aborted at submit
19358 before the window problem was even reachable.

With both addressed the capture reaches **181 frames / 22,599 submits**, and then hits the byte
budget: 28 frames is 3.4 GB against a 3,072 MB maximum. **There is no setting that both lands on
submits and fits the budget, because one knob controls both.** Filed as **#2549** with a
time-based-window suggestion.

**Consequence for everything above:** there is still no draw-level view of a GTA V frame. Every
conclusion in this document about why the world is absent rests on log statistics, not on the frame's
contents.

## Open, unquantified: 38 of 40 logged `WaitRegMem` waits are on UNMAPPED labels (2026-08-15)

```
[agc] WaitRegMem #3 q=A NOT satisfied at fold time: [0xf58]&0x190 = 0x0, func=5 ref=0xffffffff
      — dependency violated | built@0ms(age=-1ms) pre@build=0x0 LABEL-UNMAPPED
```

Of the 40 logged, **38 carry `LABEL-UNMAPPED`** and 38 are on queue A. The awaited address in that
sample is **`0xf58`** — four-byte aligned, so it passes the call site's gate, and far too small to be
a guest address, which in this title are `0x20xxxxxxxx`.

**That is the same shape as the indirect-dispatch argument truncation** on the previous section
(`0xf8480120` for `0x20f8480120`), which has an in-tree aperture recovery. Whether the same recovery
applies to `WAIT_REG_MEM` labels is untested.

Two things must be checked before treating this as a defect, and neither has been:

- **It may be normal.** `command_processor.cpp` states plainly that an unsatisfied wait is "NORMAL,
  handled state" and that under content-load bursts it "fires thousands of times a minute". The
  barrier model behind `PROSPER_WAIT_DEFER=1` exists precisely because the default folds past them.
- **The count is a LOG CAP, not a measurement.** The diagnostic prints the first 40 and then every
  1024th, so "40" says nothing about the true rate. Quoting it as a frequency would be the rate-limit
  trap the orchestration doc warns about.
- **`wm_addr` may be register space.** PM4 `WAIT_REG_MEM` selects register or memory addressing, and
  this code path reads `wm_addr` as memory unconditionally (`guest_readable(c.wm_addr, 8)`). A
  register-space wait would then always read as unmapped. This area carries substantial prior work
  (#312, #380, #448), so the absence of a `mem_space` check may be deliberate rather than missing —
  it has not been established either way here.

Recorded as an observation with its caveats rather than a lead, so the next reader neither chases it
blind nor loses it.

## `0x413ce6000` IS a bottom-up box32 BVH refit — and pc156 supplies BOUNDS, not references

Codex's ISA read (#2542), with one decode correction to this document: **pc156 is
`buffer_load_dwordx4`, not x3** (`e0382000 80020006`, GFX10.3 MUBUF op `0x0e`), and pc158
(`e0342010 80020406`) is `buffer_load_dwordx2`. Together they load **six dwords into `v0..v5`** —
AABB min/max XYZ.

```
pc156/158   load six BOUND values through the selected s[8:11]
pc212       store dwordx4 v[0:3] to node offset +16
pc214       store dwordx2 v[8:9] to node offset +32
pc218       atomic allocation/counter through s[24:27]
pc224       load dwordx2 v[4:5] through s[24:27]
pc228..251  transform/encode those two indices, including node type 5
pc253       store dwordx4 v[4:7] at node offset +0     <- the CHILD REFERENCES
pc255/257   reload the node bounds
pc260..266  min/max into the running bounds
pc269       load the next parent/index and loop upward
```

So it is a **bottom-up construction/refit of box32 internal nodes**: merge six float bounds, obtain
two child indices, write encoded child references plus bounds, propagate upward.

**Three consequences, and the first retires a link this document asserted:**

1. **The dynamic descriptor at pc156/158 feeds the BOUNDS at +16/+32. It does NOT supply the child
   references written at pc253.** An epoch-stale pc156 can corrupt AABBs; it **does not directly
   explain cyclic child references**. The "unresolved descriptor → wrong output → cycles" chain in
   the sections above is therefore **not established**, and its `CONFIDENCE: MED` was generous.
2. **The cyclic-reference frontier is the pc218/224 path** — the `s[24:27]` resource/counter/index
   data, the index/base arithmetic involving `s16`, and the destination selection. That is where
   ordered execution-epoch capture belongs.
3. **A decline cannot itself perform the measured 0 → 324 cycle addition** — a declined dispatch
   writes nothing. A prior decline could leave stale state a later executing invocation consumes, but
   then the executing path is still part of the defect. The 29 cyclic submits on which every
   `ce6000` dispatch executes already ruled out decline-alone as sufficient.

The generic lift's CPU-epoch problem remains a real correctness issue; it is just not the direct
value source for the reference cycles. **Do not close that link without measuring pc224's
execution-epoch resource and indices.**

## Submit-indexed capture already exists — #2549's workaround was not needed for this

```
PROSPER_GPU_TIMELINE_CAPTURE=<capture.prgcap>
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=1
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_COMPUTE_PROGRAM=0x413ce6000
```

With a semantic selector, `CAPTURE_SUBMIT` is the **minimum** submit and the first later submit
containing that program becomes the endpoint — which is exactly the "aim the capture at a submit that
contains work" this document asked for, and it was already there.
`PROSPER_GPU_TIMELINE_CAPTURE_WHEN_DISPATCH_DIM=XxYxZ` qualifies further; a rolling predecessor
bundle uses `..._CAPTURE_BUNDLE` + `..._CAPTURE_DEPTH`. `...AFTER_COMPUTE_PROGRAM` is a **cross-submit
phase gate** and is the wrong selector when the endpoint is the submit containing `ce6000` itself.

*(The F9 fixes on this branch remain valid for the interactive grab, and #2549's present-count
finding still stands. But the capture this investigation needed did not require them.)*

## The indirect-argument high dword is NOT folded into the modifier

Also falsified by Codex. The HLE builder writes `cmd[1] = uint32_t(a1)` (offset), `cmd[2..3]` = the
64-bit modifier, and PM4 decode reads them faithfully. The failing async-queue packet's live values:

```
base=0   offset=0xf8480120   modifier=0x21
low | (modifier_low << 32) = 0x21f8480120   (unreadable)
0x20f8480120                                (readable)
```

So the modifier is `0x21`, not `0x20` — the "the high dword is in the modifier" reading is dead, and
**there is no evidence of a PM4 decoder bug**.

**The decisive probe belongs before packet construction**, in `agc_cb_dispatch_indirect`: log the
full `a1`, the full `a2`, and which exported NID entered the shared body — both DCB and ACB NIDs
alias one HLE implementation, and the two forms may carry the address differently.

- `a1 == 0x20f8480120` → the HLE builder truncates a full address and must preserve that ABI form.
- `a1 == 0xf8480120` → the high bits never reached the builder, and the missing piece is per-ACB/queue
  base state or another ABI-defined provenance source.

**Until that is measured, aperture recovery stays diagnostic-only.** "Mapped under the common
aperture" is not proof that it is the intended argument buffer — and this document's earlier framing
of the truncation as a defect with an in-tree fix was ahead of the evidence.

## RETRACTED — "THE BREAK" was two different presentation phases compared as one (2026-08-15)

**The section below is wrong in its grouping and its interpretation, and is kept only so the
measurements stay available. Do not cite its conclusion.** Codex falsified it with a
`PROSPER_MEMLOG=1` + `PROSPER_RTTLOG=1` run (#2542):

- **The physical mappings are disjoint, so VA aliasing is dead.** `VA 0x1557c00000 -> phys
  0x105000000`; `VA 0x1543c00000 -> phys 0x102800000`; the dominant `0x208e5a0000` target lies inside
  `VA 0x203de00000 -> phys 0x111000000` and resolves to physical `0x1617a0000`. None overlap. **A
  physical-page RTT identity would miss exactly as the VA-keyed one does** — so the fix I proposed
  would not have worked either.
- **The `0x14…`/`0x15…` misses are not the gameplay composite at all.** They belong to
  `fs=0x2042f83c00` — one full-resolution plus two half-resolution guest-backed planes, an
  early/startup presentation path (very likely video/YUV-style input). They are ordinary guest
  images, so an RTT miss there is **expected**.
- **The real gameplay chain is entirely in `0x20…` and every link HITS:**

```
fs=0x205b34be00                       -> target 0x2056740000
fs=0x205b384a00  samples 0x2056740000 HIT -> target 0x2058720000
fs=0x205b363c00  samples 0x2058720000 HIT -> SCANOUT
```

  The last two stages preserve the producer's counts (`rgb_nonblack=1121820`), and the scanout holds
  the tutorial/UI/light content but no world.

**So the world is already absent at the OUTPUT of `fs=0x205b34be00`. It is not lost in the final
composite, and not lost to an address mismatch.**

What went wrong in my reasoning: I compared misses drawn from one presentation phase against render
targets drawn from another, and read the address-range difference as a mismatch between two sides of
*one* frame. The hit/miss counts and the address ranges were real; the grouping was not. A prefix
histogram over a whole run cannot tell two phases apart, and I did not check that the sampled
surfaces and the targets came from the same pass.

## THE ACTUAL FRONTIER: the inputs of `fs=0x205b34be00` (2026-08-15)

That fragment shader produces `0x2056740000`, the first stage of the gameplay chain, and its output
already lacks the world. Its directly sampled inputs:

| input | state |
| --- | --- |
| `0x2052ac0000` 3840x2160 fmt=1 | **RTT miss + DCC-unsupported warning** |
| `0x2054aa0000` 3840x2160 fmt=11 | **RTT miss** |
| `0x2063380000` 3840x2160 fmt=20 | HIT |
| `0x2085de0000` 3840x2160 fmt=4 | HIT |

**So the DCC problem is NOT superseded — it is the live lead, on the real chain.** `0x2052ac0000` is
a guest-backed 4K surface that misses the RTT cache and then falls through to reading DCC-compressed
bytes as uncompressed.

Codex's qualification, which matters: **DCC warnings must be classified per surface.** Some
DCC-described addresses do have a renderer-owned RTT image and hit; genuinely guest-backed ones such
as `0x2052ac0000` and `0x20e0380000` miss and fall through. The renderer now carries all eight MRT
bindings and the backend publishes them, so there is no evidence `0x2052ac0000` is merely an
untracked secondary MRT — its persistent absence from the RTT cache makes **guest-backed DCC, or an
unidentified producer, the better hypothesis**.

**Next discriminator:** identify `0x2052ac0000`'s allocation and write history and inspect that exact
input — not change RTT identity.

## OLD (retracted): the composite samples `0x14…`/`0x15…` (2026-08-15)

`PROSPER_RTTLOG=1` reports, per sampled texture, whether it resolved to a renderer-owned render
target or fell through to guest memory. Over a routed run:

```
14389  -> HIT
 1638  -> miss
   49  -> RTT PATH SKIPPED
```

A 90% hit rate — so the RTT machinery works. **But the misses are systematically the large surfaces,
and they are in a different address range from every render target prosper draws into:**

```
[rtt] sample tex addr=0x1557c00000 3840x2160 fmt=9 -> miss (cache_size=8)
[rtt] sample tex addr=0x1558c00000 3840x2160 fmt=9 -> miss (cache_size=85)
[rtt] sample tex addr=0x1543c00000 3840x2160 -> miss
[rtt] sample tex addr=0x1547c00000 4096x2048 -> miss
[rtt] sample tex addr=0x14df560000 2048x2048 -> miss   (and four more 2048x2048)
[rtt] sample tex addr=0x1544400000 1920x1080 -> miss
```

**Sampled addresses that HIT are `0x20xx…`** (the prefix census is dominated by `0x2066`, `0x206c`,
`0x20d6`, `0x2067`, `0x205f`, `0x20e0`). **The misses are `0x14…`/`0x15…`.** And the draw census
shows every colour target prosper renders into is `0x20…` — `0x208e5a0000` (~71,000 draws),
`0x20431c0000`, `0x20ec7c0000`, and the scanouts `0x215ed10000` / `0x2160cf0000` / `0x2162cd0000`.

**So the two sides of the frame use different virtual addresses for the same surfaces.** prosper's
RTT cache is keyed on the address its draws rendered to; the composite asks for a different one and
gets nothing, which for a surface prosper rendered into means it samples empty guest memory.

**That is a mechanism that produces exactly the observed symptom**: ~71,000 draws fill a scene
buffer, a composite runs and reaches the scanout, and the screen shows no world.

**`CONFIDENCE: MED-HIGH` on the observation** — the hit/miss split, the size distribution and the
address ranges are all measured. **`CONFIDENCE: LOW` on the cause.** Two readings are open and have
not been separated:

- **VA aliasing.** PS5 titles can map the same physical memory at more than one virtual address
  (`sceKernelMapDirectMemory`). If the guest renders through one and samples through another, a
  VA-keyed cache misses by construction, and the fix is to resolve RTT identity through physical
  memory or an alias map rather than raw VA.
- **A genuinely different surface.** The `0x14…`/`0x15…` surfaces may be resources the guest
  produced by some path prosper does not render at all, in which case the miss is a symptom and the
  absent producer is the defect.

**The measurement that separates them:** whether any `0x14…`/`0x15…` address and any `0x20…` render
target resolve to the same physical pages. That is answerable from the guest's own memory mapping and
needs no further routed run.

**This supersedes the DCC suspicion as the leading candidate.** The three unsupported 4K DCC surfaces
(`0x2052ac0000`, `0x20e0380000`, `0x20df360000`) are all `0x20…` and are not among these misses.

## WHERE the draws go: a scene buffer with ~71,000 draws, and a composite that reaches the scanout (2026-08-15)

`PROSPER_DRAW_CENSUS=1` now records each draw's `CB_COLOR0_BASE` (+ `_EXT`, + `CB_COLOR0_VIEW`),
sampled 1 in 32 and ranked by busiest:

```
[draw-census] draws=131072 indirect=0 submit=14180
[draw-census]   distinct colour targets (sampled): 94
[draw-census]   color0=0x208e5a0000 view=0x0      draws=2228     -> ~71,000 draws
[draw-census]   color0=0x20431c0000 view=0x0      draws=457      -> ~14,600
[draw-census]   color0=0x20ec7c0000 view=0x4002   draws=156  \
[draw-census]   color0=0x20ec7c0000 view=0x0      draws=98    |  a LAYERED target: five distinct
[draw-census]   color0=0x20ec7c0000 view=0x2001   draws=77    |  CB_COLOR0_VIEW slices
[draw-census]   color0=0x20ec7c0000 view=0x8004   draws=55    |
[draw-census]   color0=0x20ec7c0000 view=0x6003   draws=42   /
[draw-census]   color0=0x215ed10000 view=0x0      draws=89   \
[draw-census]   color0=0x2160cf0000 view=0x0      draws=86    |  the SCANOUT buffers
[draw-census]   color0=0x2162cd0000 view=0x0      draws=83   /
```

**Two facts settle where the investigation goes next:**

1. **A single dominant target takes ~71,000 draws** (`0x208e5a0000`).

   > **CORRECTED 2026-08-18 — this is almost certainly NOT the scene colour buffer, and calling it
   > "the world's geometry" sent at least one investigation the wrong way.** Measured with
   > `PROSPER_DUMP_RTGROUPS_RGBA`, which reports every pass regardless of content: prosper renders
   > **1,755 passes into it at 1024x1536 with `format=9` (`VK_FORMAT_R8_UNORM`)**, and the byte count
   > confirms one channel — `actual=1572864` = 1024*1536*1 against an RGBA8 `expected=6291456`.
   > A single-channel 1024x1536 surface taking the most draws in the frame is the shape of a **shadow
   > or AO atlas**, not a 4K scene colour target. An empty one gives wrong lighting, not an absent
   > world.
   >
   > Those passes also render **empty**: at `PROSPER_DUMP_RTGROUPS=1` (dump if >= 1 non-zero byte)
   > not one of them produced a file, while a small non-empty 128x1 minority did. **Beware that
   > asymmetry** — a threshold-gated dump only writes for the class that passes the threshold, so
   > reading "every dump is 128x1" as "every pass is 128x1" inverts the conclusion. The RGBA variant
   > exists precisely because it is threshold-independent.
   >
   > What the original text got right is that draws are *issued* in quantity. What it did not
   > establish is what they are issued *into*.
2. **The scanout buffers each receive ~2,800 draws.** Those addresses are the ones the frame grab
   reported as `present_front_address()` (`0x215ed10000`, `0x2160cf0000`, `0x2162cd0000`), so the
   **composite runs and targets the presented surface**.

**So the scene is drawn and the composite reaches the screen — and the screen has no world.** The
break is between them: whatever the composite samples does not carry `0x208e5a0000`'s contents.

That is a much narrower question than any asked before, and it has a named suspect already in this
document: when a sampled image is DCC-compressed and the uniform fast-clear decode fails,
`live_renderer.cpp` falls through to reading compressed bytes as uncompressed. The three unsupported
4K DCC surfaces are `0x2052ac0000`, `0x20e0380000` and `0x20df360000` — **none of which is
`0x208e5a0000`**, so that specific mechanism is not yet connected to the scene buffer.

**The measurement that closes it: for the composite draws (those whose colour target is a scanout
address), which textures do they sample, and does each resolve to prosper's own rendered image (an
`rtt_hit`) or fall back to reading guest memory?** Guest memory for a surface prosper rendered into
is black — the scanout dump confirms guest-side scanout is uniformly black, as expected when prosper
renders into its own Vulkan images and presents those.

### A note on instrument cost, because it cost a run

The first version of this census took a mutex and inserted into a map on **every** one of 131,072
draws and printed twelve lines at each power of two. The routed run stalled at 1,024 draws and never
reached gameplay. Sampling 1 in 32 and reporting from 4,096 upward fixed it. An instrument that
changes the subject's behaviour measures the instrument.

## THE DRAWS ARE HAPPENING: 131,072 executed, 0 indirect, 0 undecoded packets (2026-08-15)

The most basic number about a missing world had never been measured. `PROSPER_DRAW_CENSUS=1` (added
here, counted before the `render` early-out so it reports what the command stream contained):

```
[draw-census] draws=1024   indirect=0 submit=190
[draw-census] draws=16384  indirect=0 submit=4248
[draw-census] draws=131072 indirect=0 submit=13933
```

**Over 131,000 draws execute, and not one of them is indirect.**

Three things follow, each independently verified:

1. **The indirect dependency latch drops nothing.** A counter at the drop site (also added here —
   the drop was silent unless a capture trace happened to be active) reports **zero** across a full
   routed run that reached gameplay, with the instrument confirmed present in the binary and the
   route confirmed by 44 builder transitions and 9,580 frames. **This retires the premise this
   document opens with** for the current state: "every later indirect draw short-circuits untried"
   is not what is happening.
2. **prosper decodes every packet the title sends.** `pm4_registers.hpp` defines
   `IT_DRAW_INDIRECT_MULTI = 0x2C` and `IT_DRAW_INDEX_INDIRECT_MULTI = 0x38`, and **neither is
   referenced anywhere** — the decoder handles only `R_DRAW_INDEX_INDIRECT = 0x22`. That looked like
   the answer for a GPU-driven title. It is not: the decoder logs each distinct undecoded type-3
   opcode once, and **no `[pm4] unknown raw type-3 opcode` line appears in any run**. GTA V emits
   neither MULTI variant. *(The two constants being defined and unused is still worth knowing — a
   title that does use them would be silently mis-decoded — but it is not this title's defect.)*
3. **So the world's geometry is somewhere in those 131,072 direct draws, and they execute.**

**That relocates the question entirely.** It is no longer "why are the draws not issued" — they are
issued and executed. It is **"why does the output of 131,072 executed draws not reach the screen"**.

The strongest remaining candidate is the one already recorded and never quantified: **three 4K
DCC-compressed sampled images are unsupported**, and when the fast-clear decode fails the renderer
falls through to reading compressed bytes as uncompressed. A composite that samples the scene through
those cannot produce a world image no matter how many draws filled them.

## What is NOT hiding the world — four eliminations, each with a verified lever (2026-08-15)

Every one of these was measured on a routed run with the lever confirmed to have moved, so each is a
**genuine negative rather than a void arm**. None of them restores the world.

| candidate | lever, verified | outcome |
| --- | --- | --- |
| the compute hang / device loss | consumer skipped → **0 device losses**, 9,363 frames | world still absent |
| truncated indirect-dispatch arguments | aperture recovery fires 24×, unreadable **24 → 0** | frame unchanged |
| storage-image contract dropping a whole batch | per-draw drop reports **"kept 0 of 1 draws"** | the batch *is* one draw; identical either way |
| `CB_COLOR_CONTROL.MODE=0` on 131,072 draws | — | known latching artefact (#1706), not per-draw truth |

### The storage-image rejection, precisely

The failing resource is named exactly, and **the DCC hypothesis for it was wrong**:

```
[render] storage-image contract: set=1 binding=46 portable-uvec4 REJECTED
    guest-texel=4 shape=0
    writable=1  compressed=0  arrayed=0  multisampled=0
    reflected-dim=1 guest-dim=1 depth=1 fmt=4 comps=2
```

The only failing term is **`writable=1`** — `portable_storage_shape` requires
`!writable_storage_image`. Compression, arraying, multisampling and the dimensions are all fine. So
**writable portable-uvec4 storage images are unsupported in the graphics path**, and that is a real
gap worth closing on the charter's own terms.

But its blast radius is one draw, not a frame: `PROSPER_RENDER_DROP_UNPROVEN_DRAW=1` reports
`kept 0 of 1 draws in this batch` every time. The conservative whole-batch abort in `render_runner.h`
looked alarming and costs nothing extra here.

### Still open, and unquantified

Three 4K DCC-compressed **sampled** images remain unsupported (fmt 1/4/9). That is a separate
resource from the storage image above — the storage image is not compressed. Whether the composite
depends on those three has not been established.

### The instrument that would answer this — THREE sequential gates, each masking the next (2026-08-16)

`PROSPER_GRAB_BUNDLE_AFTER_MS` on `prosper-app` is the documented fastest loop for "why does this
frame look wrong", and on this title it fails **three times, for unrelated reasons**. Each failure
reads as "frame capture does not work on GTA V", and because they are **sequential** every fix reveals
the next one rather than the bundle: clear the empty window and you meet the provenance abort, clear
provenance and you meet the byte budget. That masking is why this took several runs to walk, and why
the third gate — which no setting can clear — was the last to be seen.

**Gate 1 — the capture window is a PRESENT COUNT, and a present count is not a unit of time.**
Earlier attempts at 170 s with `PROSPER_CAPTURE_FRAMES=1` and 16 reported *"the capture window
contained no GPU submits"*, which read as a window/submit mismatch of unknown kind. The newer
diagnostic names it exactly:

```
[grab] frame-bundle: during this window the submit hook was reached=0, 0 while inactive,
       0 while not capturing; window was open 1 ms for 1 presents      # FRAMES=1
       ... window was open  6 ms for   8 presents                      # FRAMES=8
       ... window was open 38 ms for  64 presents                      # FRAMES=64
```

`reached=0` settles that nothing was mis-classified — the hook never fired. The rate is the finding:
this title flips in **bursts**, ~0.6 ms per present against a ~23–25/s average, so a window's
wall-clock duration is effectively random and usually far too short to contain a submit. Widening the
count does not reliably help, because a burst consumes it: 64 presents elapsed in 38 ms and saw
nothing.

**The fix is `PROSPER_CAPTURE_WAIT_FOR_SUBMITS=1`** (`gpu_timeline.cpp`, opt-in), which extends the
window until at least one submit is captured, bounded by the same 240-present ceiling, and never
shortens a window that already worked. This was already diagnosed on this title under #2549 with the
same burst measurements — recorded here because three separate frame counts were tried before that
flag was found, and the failure message points at `PROSPER_CAPTURE_FRAMES` instead.

**Gate 2 — the bundle then aborts on indirect-pointer provenance.**

```
[grab] frame-bundle: submit 34146 failed (indirect-pointer relocation lacks exact compute
       provenance); grab aborted
```

`validate_captured_indirect_pointer_relocations` requires capture format ≥ v53, a captured recompile
config, an in-range raw shader index, **and exactly one** indirect-pointer carrier. GTA V fails one of
these, and until 2026-08-16 the message was the same sentence for all four, so the log could not say
which arm to pursue; it now names the failing precondition and prints the carrier/marker counts.

**`PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1` accepts it and writes the bundle**, with the tool
stating the limit itself: *"THIS BUNDLE IS FOR INSPECTION, NOT FAITHFUL REPLAY."* That is the right
trade for the question in this document — the bundle is wanted for its **resource tables and
descriptors**, not to reproduce the frame — but a replayed frame from such a bundle is not evidence
about rendering, and must not be used as any.

**Gate 3 — GTA V's working set exceeds the F9 grab's 2 GiB default budget.**

```
[grab] frame-bundle: submit 42862 failed (frame bundle unique bytes 2155499889
       exceeded limit 2147483648); grab aborted        # FRAMES=240, died at frame 153
[grab] frame-bundle: submit 35716 failed (frame bundle unique bytes 2236660079
       exceeded limit 2147483648); grab aborted        # FRAMES=4 + WAIT_FOR_SUBMITS
```

**Read those two together, because the pair is the finding:** 153 frames cost 2.155 GB and *four*
frames cost 2.237 GB. The bytes are therefore **not** per-frame deltas — they are dominated by the
resident working set the first captured frame pulls in, so shrinking the window does not shrink the
bundle and there is no frame count that fits under 2 GiB. (An earlier revision of this section
divided 42,861 submits by 153 frames and published "≈ 14 MB per frame" as a sizing rule. That
arithmetic is right and the inference from it is wrong: the 4-frame run falsifies it outright.)

The lever is the budget, not the window — **`PROSPER_CAPTURE_BUNDLE_MAX_MB`**, 64..3072 MiB against a
2048 MiB default — **and on this title it is not enough at its maximum:**

```
FRAMES=240, limit 2048 MiB -> 2,155,499,889 bytes   (died at frame 153)
FRAMES=4,   limit 2048 MiB -> 2,236,660,079 bytes
FRAMES=4,   limit 3072 MiB -> 3,351,980,610 bytes
FRAMES=1,   limit 3072 MiB -> 3,269,369,026 bytes   <- ONE frame, over the ceiling
```

A single frame costs 3.27 GB against a 3072 MiB (3.22 GB) hard clamp, so **there is currently no
setting under which an F9 bundle of GTA V completes.** The last row is the one that matters: it is not
a window-size problem and cannot be tuned away. (These are overshoot values at the point of abort,
not totals, and they vary run to run, so the true working-set size is unknown and above 3.35 GB.)

The fix worth building is not a bigger number. The question a bundle is wanted for here — *which
compute program binds the empty composite tap* — needs **resource tables and descriptors, not pixel
payloads**; a metadata-only capture would fit inside any budget and answer it directly. Tracked as
#2554.

The three gates are **sequential and each masks the next**, which is why this took several runs to
walk: widen the window and you meet provenance; clear provenance and you meet the size limit; and
widening the window is the wrong lever for gate 1 anyway. The combination that gets through is a
**small** window that waits for submits, plus the override:

```bash
PROSPER_CAPTURE_FRAMES=1 PROSPER_CAPTURE_WAIT_FOR_SUBMITS=1 \
PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1 PROSPER_CAPTURE_BUNDLE_MAX_MB=3072 \
PROSPER_GRAB_BUNDLE_AFTER_MS=380000 PROSPER_CAPTURE_DIR=~/<dir> \
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700 \
PROSPER_PAD_SCRIPT=@scripts/gta5/reach-story-mode.pad \
SDL_VIDEODRIVER=offscreen ./prosper-app <DUMP_ROOT>/PPSA04263-app0
```

Until a bundle exists, every conclusion in this section is from log statistics rather than from the
frame's actual contents.

## The hang is NOT the only blocker — two more, measured (2026-08-15)

Answering "is the compute hang the reason there is no world" directly, by looking at a frame with
the hanging consumer skipped: **zero device losses, 9,363 frames, and the world is still absent.**
Tutorial text and a few light blooms render; no geometry. So the compute chain cannot be the whole
story, and two further blockers are visible in the same run.

### 1. Indirect dispatch arguments arrive with a TRUNCATED address — and the fix is already in-tree, off

```
[agc] indirect dispatch skipped: unreadable arguments at 0xf8480120
```

The real address is `0x20f8480120`; the high byte is gone. `command_processor.cpp` already has an
aperture-recovery path — `(aperture << 32) | (addr & 0xffffffff)` — behind
**`PROSPER_INDIRECT_APERTURE_RECOVERY`, which is opt-in and off by default**. Its own comment records
that with it off, 50 of 64 indirect compute dispatches are skipped as unreadable; with it on, 0 are,
and the probe found the raw low address unmapped and `aperture | low` mapped on 49 of 49.

**Measured here with the lever verified:** recovery fires 24 times
(`0xf8480120 -> 0x20f8480120`, queue 2), unreadable-argument skips go **24 → 0**, device losses stay
at 0. **And the frame is unchanged — still no world.** A genuine negative, not a void arm: the
lever demonstrably moved.

So the truncation is real and worth resolving on its own merits, but it is not what is hiding the
world either.

### 2. Three 4K DCC-compressed sampled images are unsupported

```
[render] DCC-compressed sampled image 3840x2160x1 fmt=1 tile=24 is unsupported; metadata=0/0
[render] DCC-compressed sampled image 3840x2160x1 fmt=4 tile=27 is unsupported; metadata=81920/81920
[render] DCC-compressed sampled image 3840x2160x1 fmt=9 tile=27 is unsupported; metadata=49152/49152
```

`live_renderer.cpp` handles DCC only for the **uniform fast-clear** case
(`gfx10_dcc_fast_clear_rgba8`). When that fails it warns and **falls through to the ordinary
format/detile path, which reads the COMPRESSED base bytes as if uncompressed** — garbage or black.

These are full-screen 4K surfaces, they are **not** renderer-owned RTTs (`!rtt_hit`), and there are
three of them in the formats a scene colour/normal/etc. set would use. **A composite that samples
them cannot produce a world image regardless of what the geometry passes do.**

`CONFIDENCE: MED-HIGH` that this is an independent blocker; `CONFIDENCE: LOW` on it being *the*
remaining one, since nothing yet shows the geometry passes fill those surfaces in the first place.

### Not a lead: `CB_COLOR_CONTROL.MODE=0`

The same run reports 131,072 draws with an unmodeled `CB_COLOR_CONTROL.MODE=0` "still executed as an
ordinary color draw", which looks alarming (MODE 0 is CB_DISABLE). **`render_state.hpp` already
records that prosper's decoded MODE is not per-draw-trustworthy (#1706): a utility sequence's
operation bits stay latched onto later ordinary draws.** So the count measures the latching, not
131,072 draws that should have been suppressed. Checked before chasing.

## Six-reference simulation: the builder is PROVABLY faithful (2026-08-15)

Codex's correction (#2542): `0x413dc3400` reads **six** candidate references per node, not two —
`pc86` loads the first pair, `pc161` follows `(A >> 3) - 4` for a second pair, `pc237` follows
`(B >> 3) - 4` for a third. **A histogram over record dwords 0/1 covers two of six and cannot predict
how many parent slots a dispatch writes**, so `pairs == 1030` is an empirical control for one route
state, not a universal oracle. What survives as a hard oracle is **acyclicity**.

`tools/re/bvh_ref_simulator.py` reproduces all six selections over each captured input and diffs the
expected destination set against the slots the dispatch actually changed:

| dispatch | expected | written | expected-not-written | **written-not-expected** | ref cycles |
| --- | --- | --- | --- | --- | --- |
| s5943 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s7188 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s8842 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s16041 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s17181 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s19002 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s5528 d37 | 1754 | 1457 | 297 | **0** | 0 |
| s9645 d37 | 1788 | 1283 | 505 | **0** | 0 |

**`written but not expected` is ZERO in every frame measured.** The builder never writes a slot its
input does not imply — now established over all six references rather than two.

And the decisive rows are s16041 onward: **`expected = written = 2061`, `missing = 0`, and
`cycles = 117`.** Everything the input implies is written, nothing else is, and the result is cyclic
**because the input is**. There is no discrepancy left for the builder to be responsible for.

(`expected-not-written` is nonzero in other frames because the simulator walks every node while the
dispatch processes only its compacted, depth-parity subset. The informative direction is the other
one, and it is zero everywhere.)

**This closes `0x413dc3400`'s role definitively.** Together with the eleven perfect submits, two
independent methods now say the same thing: the builder is correct and the defect is entirely
upstream, in the node records `0x413ce6000` writes.

## Corrections from Codex (#2542) to earlier sections

- **`0x413cf9000` / `0x413cf9200` are arena aliases, not producers of the 64-byte tag records.**
  `cf9000` is an initializer over an **80-byte** stride; `cf9200` operates **32-byte** records at a
  320-byte V# near `0x209cc7ab00`. Their 117 changes each are a paired initialise/fill of a small
  structure that merely overlaps the watched allocation. **So ranking writers by whole-watch change
  count is misleading, and the +54 cycles this document attributed to `cf9000` should be discounted**
  — its writes are not 64-byte records and decoding them as such is a category error. The genuine
  64-byte-view producers are `cf5400`, `cf6100`, `ce6000` and `d1bf00`. `0x413ce6000`'s **+590** is
  unaffected: it is a real 64-byte-view producer.
- **The `record 4 + 8` arithmetic is exact**, not over-fitted: `pc74` forms `selector = word >> 3`,
  `pc144` `v_readfirstlane_b32 vcc_lo, v7`, `pc149` `s_mulk_i32 vcc_lo, 120`, `pc153`
  `s_buffer_load_dwordx4 … offset:8`. So `4 * 120 + 8 = 488` exactly. **The contract's weakness is
  TEMPORAL, not arithmetic**: `selected_sbuffer_domain()` reads the source and the outer table
  through `complete_resource_bytes()` — the currently CPU-visible mapping — with no command-order
  snapshot coupling either to the bytes the GPU later consumes. Codex has retained evidence of the
  same outer base decoding as **five coherent V#s** at exactly `+8` in one observation and floats in
  another, so the buffer is a reused arena observed at different epochs. **The
  "it holds frustum planes / the contract is over-fitted" framing in this document is therefore
  wrong about the cause** — the layout is right and the epoch is not.
- **The selector is `readfirstlane(source_word >> 3)`**, from the pc70 source records — not from a
  saved EXEC mask, as this document earlier concluded from watching s106 alone. It is still not
  const-foldable (a readfirstlane of live lane data is runtime state), so the conclusion that no fold
  can resolve it stands; the stated *route* to it was wrong.
- **`0x413e1ff00 toucher=0`** — Codex reached the same diagnosis independently: a watch-attribution
  bug, not memory corruption. Already fixed here with `compute_address_window_hits`.

**A systematic caveat this raises over much of the descriptor work above:** every capture in this
document reads guest memory at fold or realization time on the CPU. That is not the execution epoch
the GPU consumes. Codex's recommended discriminator is an ordered GPU-side readback of the pc70
source, the root V# at `s[0:1] + 0xa8`, and all 600 outer bytes, captured **immediately before the
dispatch executes** — a CPU reread is not sufficient.

## ROOT CAUSE, ATTRIBUTED: `0x413ce6000` writes the cyclic child graph (2026-08-15)

Per-dispatch pre/post attribution on the node records at `0x209cc76000`, with the **child-graph cycle
count** as the metric:

```
submit  order   writer         cycles pre -> post
6765    4550    0x413ce6000       0 -> 324    ADDS 324
7180    24812   0x413ce6000     324 -> 398    ADDS 74
7595    26144   0x413ce6000     398 -> 388    removes 10
8835    26556   0x413ce6000     388 -> 402    ADDS 14
15271   16831   0x413ce6000     350 -> 436    ADDS 86
16088   14048   0x413ce6000     436 -> 468    ADDS 32
16088   14546   0x413cf9000     468 -> 469    ADDS 1
```

| program | dispatches that add cycles | total added |
| --- | --- | --- |
| **`0x413ce6000`** | 7 | **590** |
| `0x413cf9000` | 12 | 54 |

**`0x413ce6000` is the program that makes the child graph cyclic**, including the transition from a
completely acyclic graph to 324 cycles in one dispatch. It also *removes* cycles on other dispatches
(−10, −14, −20, −18), which is the signature of an incremental refit that is partly wrong rather than
a rebuild that is wholly wrong.

### The complete chain, every link measured

1. **`0x413ce6000`** writes the 2,063 × 64-byte node records at `0x209cc76000`, and its output
   contains a **cyclic child graph**.
2. **`0x413dc3400`** builds parent links from those records **faithfully** — its lowering is proven
   correct by eleven consecutive perfect submits, and its output tracks its input.
3. The parent table is therefore cyclic.
4. **`0x413dc6700`** walks it with a loop that has no iteration bound and no cycle detector.
5. GPU hang → RADV device loss → live compute disabled process-wide → every later indirect draw
   dropped → **no 3D world**.

**And `0x413ce6000` is exactly the program whose descriptor at pc156 does not resolve**
(`mode=unresolved-operand`, the `selected_sbuffer` contract declining because the buffer it reads
holds frustum planes) and which is declined outright on some dispatches. A program that cannot
resolve one of its descriptors, and is sometimes not run at all, is precisely a program that writes a
partially-correct node array.

**`CONFIDENCE: HIGH`** on the attribution — per-dispatch pre/post, one writer, one metric, and the
0 → 324 transition is unambiguous. **`CONFIDENCE: MED`** that the unresolved descriptor is *why* its
output is wrong; that link is plausible and untested, and the honest alternative is that some other
part of its lowering is at fault.

**This supersedes the earlier "missing producer" framing without contradicting its falsification.**
Absence was correctly ruled out — 29 submits with `0x413ce6000` fully executing still went cyclic.
The cause is its **output**, which the earlier experiments could not distinguish because none of them
measured what it wrote.

## FOUND: the CYCLES ARE IN THE INPUT — the node records' own child graph (2026-08-15)

Two results, and together they close the question.

### 1. The clear hypothesis is FALSIFIED, with the lever verified

`PROSPER_COMPUTE_ZERO_BEFORE=0x413dc3400:0x20f848417c:2063` zeroes the parent array immediately
before every builder dispatch. The clear demonstrably fires. **The later submits are still cyclic** —
`cycles=63..101`, and now with `oob-roots=0`. So the cycles are **not** stale leftovers in the output
array; the builder writes them into a freshly zeroed one.

### 2. The cycles are in the builder's INPUT

Building the child graph directly from the node records — for each node, its two child references
decoded as `index = (ref >> 3) - 4` when the tag is in `{2,5}` — and testing that graph for cycles:

```
s11238 d15,d16,d19..d27   child-graph cycles = 0
s11238 d37                child-graph cycles = 70     <- appears here
s11637 d15,d16 ...        child-graph cycles = 70     (carried into the next submit)
s11637 d37                child-graph cycles = 110
s12036 ...                child-graph cycles = 110
s13220 d37                child-graph cycles = 117
... every later submit    child-graph cycles = 117
```

**The node records at `0x209cc76000` describe a cyclic child graph, and `0x413dc3400` faithfully
reproduces it as a cyclic parent table.** Everything below about the builder now has its explanation:
its lowering is correct, its output tracks its input, and its input is a cyclic graph.

**The cycles appear between dispatch 27 and dispatch 37 of a submit, and they ACCUMULATE**
(70 → 110 → 117), which is the signature of a structure being partially updated rather than rebuilt.
The programs running in that window are `0x413ce3400` at d30 and **`0x413ce6000` at d36 — the bulk
writer of the record array, and the one whose descriptor at pc156 does not resolve.**

**So `0x413ce6000` is back at the centre of this, for a different reason than before.** The earlier
falsification stands and was correct: its *absence* is not the cause, since 29 submits with it fully
executing still went cyclic. The cause is its **output**. A program that is sometimes declined and,
when it does run, has an unresolved descriptor is exactly a program that can write a partially-correct
node array.

### The next measurement, and it is one run

Attribute the cycle appearance to a dispatch the same way the parent table's was: watch
`0x209cc76000` with the tree watch's **pre/post** attribution and the child-graph cycle count as the
metric, rather than the parent-walk metric which is meaningless on a node array. The instrument
exists; only the analysis differs. That names the writer conclusively instead of by elimination
between d30 and d36.

## THE HYPOTHESIS THIS ALL POINTS AT: the parent array is never cleared

If the builder legitimately links only `{2, 5}` children, then **the slots it does not write must
already hold something that terminates the consumer's walk** — the walk is
`while (i != 0) i = bfe(rec[i], 3, 27)`, so a zero terminates and anything else does not.

**It is not cleared.** The tree watch's pre-image at the builder's dispatch is not zero: it is the
previous phase's stride-4 id array (`{0, 0, 0xa9, 1}` repeating). And the slots the builder leaves
alone were measured holding `0x09249249`, `0x12492492`, `0x2db6db6d` — the Morton dilation constant
and multiples — or, sometimes, zero.

**A stale Morton key read as a parent index is exactly a divergent walk.** `0x09249249 >> 3` is
0x1249249, far past 2,063, which terminates by out-of-range; but `0x12492492 >> 3 & 0x7ffffff` is
0x2492492 — also out of range. The values that *do* trap are the ones left over from an earlier
generation of the parent array itself, which are in range by construction.

**This unifies every measurement on this branch:**

- the builder's lowering is correct (eleven perfect submits) ✓
- its output faithfully follows its input (`both` tracks `pairs`) ✓
- the failure mode is "a store that does not execute, leaving a stale slot" — measured directly ✓
- the perfect submits are the ones where nearly every node has two linkable children, so nearly
  every slot gets written and stale values have nowhere to hide ✓
- the transition is at a route position, because that is when the scene starts containing enough
  triangle leaves for unwritten slots to appear ✓
- no producer decline is necessary ✓

**The prediction that would confirm it:** zero the 2,063-record parent array immediately before
`0x413dc3400`'s dispatch and the cyclicity should vanish, with the tree becoming legitimately sparse
(`pairs < 1030`, `cycles = 0`). That is a diagnostic-only experiment — it does not fix anything, since
on hardware something must be doing the clear and the real question is what — but it is decisive, it
needs no ISA knowledge, and it can be built as a `PROSPER_*` switch in one sitting.

**`CONFIDENCE: MED-HIGH`.** Every measurement fits and nothing contradicts it, but it has not been
tested, and the alternative — that the guest's own build does write every slot through a path prosper
declines — is not excluded.

## The sparse tree IS a faithful consequence of its input (2026-08-15)

Testing the right correspondence — for each parent with only one child, the **two child references in
that parent's own 64-byte node record**:

```
submit=10052  lone-child parents=164
   both refs in {2,5} = 0      exactly one = 155      neither = 9
   commonest (tag0,tag1): (5,0) x82, (0,5) x73, (0,0) x6
```

**Never both. 155 of 164 have exactly one linkable reference** — typically one box32 child (type 5)
and one **triangle** child (type 0), which the `tag == 2 || tag == 5` predicate does not link.

Across submits, over all 2,063 node records:

| submit | pairs / unpaired | both refs linkable | exactly one | neither |
| --- | --- | --- | --- | --- |
| 5943 | **1029 / 2** | **1034** | 100 | 929 |
| 7188 | **1029 / 2** | **1034** | 98 | 931 |
| 8842 | **1029 / 2** | **1033** | 98 | 932 |
| 9247 | **1029 / 2** | **1033** | 98 | 932 |
| 10052 | 676 / 74 | 785 | 266 | 1012 |
| 10449 | 616 / 73 | 775 | 271 | 1017 |

**In the perfect submits `both ≈ 1033` and `pairs ≈ 1029` — they track each other.** In the broken
ones `both` falls to ~780 and `exactly one` rises to ~270. The builder's output follows its input.

**So `0x413dc3400` is faithfully building what it is told to build**, and the defect is that the node
records it reads contain roughly 170 fewer linkable child pairs than they do in the frames that come
out right. Combined with the eleven perfect submits, this is now two independent measurements saying
the same thing: **the builder is not the defect; its input is.**

What remains open is whether that input is *wrong* or merely *different* — a scene with more triangle
leaves genuinely has fewer box-to-box links. Distinguishing them needs to know what the guest expects,
which is the question outstanding with Codex, and it cannot be settled from the output alone. The
`0x209cc76000` record array has 23 writers, so "which producer" is not yet a well-posed question
either.

## RETRACTED: that falsification tested the wrong correspondence (2026-08-15)

**The section below asked the wrong question and its conclusion does not follow.** It tested whether
a record's OWN node type predicts whether that record is paired. The predicate does not act on a
record's own type — it acts on the **two child references stored in the PARENT's node record**. The
right test is whether a lone-child parent's two refs differ in linkability, and it says something
quite different (next section). The section is kept for the record; do not cite its conclusion.

## OLD (wrong test): the sparse tree is not "correct but sparse by design"

This reading was flagged as capable of inverting the whole investigation, so it was tested rather
than left standing. If `0x413dc3400`'s `tag == 2 || tag == 5` predicate legitimately skips other node
types, the tree would be *supposed* to be sparse in frames containing them, the 1,030-pair oracle
would be wrong for exactly the frames called broken, and the defect would be the consumer's unbounded
walk over a legitimately-sparse table instead.

Cross-referencing each parent-table record's pairing state against its **node type** in the record
array, from the same dispatch (the aux dump makes this a same-moment comparison):

| submit | unpaired | node types among UNPAIRED | node types among PAIRED |
| --- | --- | --- | --- |
| 10052 | 74 | `{0: 60, 5: 14}` | `{0: 318, 4: 1, 5: 357}` |
| 11238 | 158 | `{0: 37, 5: 121}` | `{0: 414, 5: 518}` |
| 11637 | 200 | `{0: 29, 5: 171}` | `{0: 313, 5: 587}` |
| 12434 | 183 | `{0: 57, 5: 126}` | `{0: 223, 4: 1, 5: 445}` |

**Node type does not determine pairing.** Types 0 and 5 appear on both sides in every sample — a
type-5 record is sometimes paired and sometimes not, and so is a type-0 record. If the predicate
explained the sparseness, unpaired records would be exactly the types outside `{2, 5}`, and they are
not.

**So the tree is genuinely malformed, the 1,030-pair oracle stands, and the consumer is not at
fault.** The eleven perfect submits already showed the lowering is correct; this shows the sparse
output is not a legitimate alternative shape either. Both point at the input.

(The correspondence used here — parent-table index *i* ↔ record-array index *i* — follows from
`0x413dc3400` writing `parent[(ref >> 3) - 4]` where the same reference indexes the node array.)

## CONCLUSION: both rejecting programs build descriptors from LANE MASKS (2026-08-15)

`PROSPER_DYNTRACE_SGPR=106` on `0x205b654a00`, filtered by program identity:

```
pc=1091 s106 <- KNOWN 0x00000005
pc=1092 s106 <- KNOWN 0x00000001
pc=1097 s106 <- FORGOTTEN words=85ea807e     <- SOP2 0x0b, a B64 op reading EXEC_LO
pc=1098 s106 <- FORGOTTEN words=87ea6a00
pc=1101 s106 <- KNOWN 0x00000004
...
pc=1154 s106 <- KNOWN 0x0000ffc8
```

s106 is known on some paths and lost on others, and where it is lost the source is **EXEC** —
pc1097's `0x85ea807e` is a 64-bit scalar op whose `ssrc0` is `EXEC_LO`.

**So both declined programs compute descriptor fields from lane masks:**

| program | descriptor field | source |
| --- | --- | --- |
| `0x205b654a00` | BVH descriptor `s18` = `-1 + VCC_LO` | VCC_LO from **EXEC** at pc1097 |
| `0x413ce6000` | array selector, `s_mulk_i32 s106, 120` | VCC_LO from **`s_andn1_saveexec_b64`** at pc116/131 |

**A constant-folder cannot resolve either, and no widening of it ever will.** EXEC is a runtime
wave state. This is not a gap in the fold's opcode coverage — the `s_mulk_i32`, `s_addk_i32`,
`s_setreg_b32` and `s_waitcnt_vscnt` corrections made on this branch are all real fixes and none of
them could have helped, which is exactly what their verified-lever negatives showed.

**What this means for the fix.** These descriptors need a mechanism that does not fold: either
resolving the descriptor from guest memory at dispatch time (the `selected_sbuffer` contract's
approach — certify the domain, materialise the record), or a lowering that keeps the descriptor
dynamic. Which one is a design question and needs the ISA read of what the guest intends by deriving
a descriptor field from EXEC — plausibly a lane count or an active-mask popcount used as a size.

**`CONFIDENCE: HIGH`** that both fields trace to lane masks — measured per register, per program, with
a program identity that is actually unique. **`CONFIDENCE: LOW`** on what the guest means by it, which
is the open question worth asking.

## The BVH descriptor's unresolved word is `s18`, and it depends on VCC_LO (2026-08-15)

Watching each register of the descriptor `s[16:19]` in `0x205b654a00`, one run each:

| register | state before pc1180 | site |
| --- | --- | --- |
| `s16` | **KNOWN** at pc1163 | |
| `s17` | **KNOWN** at pc1176 | |
| `s18` | **FORGOTTEN** at pc1169 | `words=80126ac1` = `s_add_u32 s18, -1, vcc_lo` |
| `s19` | KNOWN `0x81000000` on some dispatches | `s_or_b32 s19, s106, 0x81000000` |

**`s18` is the word that fails, and it is `-1 + VCC_LO`.**

So the ray-tracing pass's BVH descriptor cannot resolve because **two of its four words are computed
from VCC_LO** — `s18` at pc1169 and `s19` at pc1177 — and prosper does not track a value into VCC_LO
on this path. That is the same obstacle as `0x413ce6000`'s selector and the same one the execz
liveness guard was about, now demonstrated by direct measurement on the exact register rather than
inferred from the ISA.

**This is the sharpest statement of the frontier available:** GTA V's compiler uses VCC_LO as a
general-purpose scalar register, and prosper's descriptor const-fold loses values through it. Two
programs — the ray-tracing pass and the BVH producer — are declined for exactly this, and between
them they are the ray-tracing path.

What is *not* established: **where** s106's value is lost in `0x205b654a00`. The register watch will
say — `PROSPER_DYNTRACE_SGPR=106` filtered to `program=0x205b654a00`, exactly as was done for
`0x413ce6000` — and that is the next measurement. In `0x413ce6000` the answer was
`s_andn1_saveexec_b64`, i.e. a genuine lane mask that no fold can resolve; if the same holds here,
the fix is a contract rather than a fold, and if it does not, it is a fold gap with a named opcode.

## SCC over-invalidation fixed — and it is NOT what gates the BVH descriptor (2026-08-15)

`PROSPER_DYNTRACE_SCC=1` (added here) reports every transition of the fold's tracked SCC with the pc
and words that caused it. On `0x205b654a00` it found **325 conservative SCC losses**, from three SOPK
encodings — and **two of them do not write SCC at all**:

| count | encoding | writes SCC? |
| --- | --- | --- |
| 188 | `s_setreg_b32` (0x13) | **no** — writes a hardware register |
| 43 | `s_waitcnt_vscnt` (0x17, sdst=NULL) | **no** — a wait, register-transparent |
| 94 | `s_addk_i32` (0x0f) | yes, on signed overflow |

Fixed: `s_setreg_b32` and `s_waitcnt_vscnt` no longer touch SCC, and `s_addk_i32` keeps invalidating
it while now folding its **value** when the destination is known.

**Lever verified: conservative SCC invalidations in that program went 325 → 0.**

**And the reject is unchanged.** `0x205b654a00` still fails with `mode=unresolved-operand pc=1180`.
Because the lever demonstrably moved, this is a **genuine negative, not a void arm**: SCC
over-invalidation is not what gates the BVH descriptor.

So of the four registers in the descriptor `s[16:19]`, dword3 (`s19`) was already observed resolving
to `0x81000000` on some dispatches. **The next measurement is `s16`/`s17`/`s18`** — the base and size
words — one watch each.

The change stays on its own merits: two encodings were being charged an SCC write they do not
perform, which is a correctness defect in the fold independent of this title.

## The BVH descriptor resolves only when SCC does (2026-08-15)

`PROSPER_DYNTRACE_SGPR=19` on `0x205b654a00`, now that the watch prints a real program identity:

```
pc=1171 s19 <- FORGOTTEN words=821380c1        pc=1171 s19 <- KNOWN 0x00000000
pc=1177 s19 <- FORGOTTEN words=8813ff6a        pc=1177 s19 <- KNOWN 0x81000000
```

**The descriptor's dword3 resolves on some dispatches and not others**, ending at `0x81000000` when
it does. pc1171 is `s_addc_u32 s19, -1, 0` — **both operands are inline constants**, so the only
input that can make it unknown is **SCC**.

`s_addc_u32` **is** modelled by the fold (`case 0x04`, guarded by `if (scc < 0) { ok = false; }`).
*(I first reported it as unmodelled, from a grep for `kSop2OpcodeAddcU32` that missed the literal
`0x04` at the case label. Corrected here.)* So the BVH descriptor's resolution reduces to: **is SCC
tracked across the instructions before pc1171?**

That makes SCC invalidation the lever, and it is exactly what the `s_mulk_i32` change touched — that
op does not write SCC and the fold was clobbering SCC for it anyway.

**Suggestive but confounded, recorded as such.** Across runs on this branch `0x205b654a00`'s reject
changed from a `compute-struct-reject` to the pc1180 descriptor reject, and the total skip count fell
18 → 15 → 14 → 13. Neither is evidence: the runs differ in more than one variable and the route
reaches different phases. **A clean A/B needs one flag toggled with the artefact hashed first**, which
is not what these runs were.

**The concrete next step** is a census of what invalidates SCC on the path to pc1171 in this program —
`scc = -1` has a handful of sites in the fold, and each is either a real SCC write or a conservative
one like the `s_mulk_i32` case that was corrected.

## The `selected_sbuffer` contract NEVER ACCEPTS — it is not on the path at all (2026-08-15)

`PROSPER_GTA5_SBUFFER_ACCEPT=1` (added here) reports the descriptor the contract publishes on the
dispatches it accepts. Every existing diagnostic on this contract described a **decline**, so the
accept side had never been looked at.

**It reports nothing. Zero accepts on a full routed run**, against two distinct decline reasons.

Yet `0x413ce6000` **executes 129 times of 139**. So those executions do not go through this contract
at all — they go through the **generic runtime-array lift**, which is what puts `entries=5` in the
resource map at `fetch-pc=156/158`.

**Consequence: every section above that reasons about this contract is reasoning about a path that
does not fire.** The record-4 hardcoding, the selector histogram, the frustum-plane content of the
outer buffer — all real observations, and all about machinery that declines and is then bypassed.
They explain the ~10 dispatches that are declined outright; they do not explain the 129 that run.

**The descriptor those 129 executions actually use comes from the generic lift**, and Codex's caveat
applies to it exactly as it did to the contract: *"it too materializes entries eagerly from
CPU-visible memory"*, with no command-order snapshot coupling it to the bytes the GPU consumes. A
descriptor materialised from the wrong epoch is a coherent explanation for node records whose child
graph is cyclic, and it is now the **only** remaining candidate on this path that has not been
tested.

**The next measurement is the generic lift's published entries at `0x413ce6000`'s dispatches**, in
the same style: what addresses it materialises, and whether they change between a dispatch that adds
cycles and one that does not.

## The contract reads a BVH NODE-REFERENCE array as a selector array (2026-08-15)

The selector histogram, taken from the same source records the contract's own domain proof walks:

```
[gta-selected-sbuffer]  selectors distinct=2064 (selector:count) 4:1 5:1 6:1 7:1 8:1 9:1 10:1 ...
[gta-selected-sbuffer]  record-4 selector would be 4; outer has 5 records of 120 bytes
```

**All 2,064 source records have a DIFFERENT selector, and they run 4, 5, 6, 7, 8, … — i.e.
`selector = index + 4`.**

That is not a selector. **It is the BVH node-reference encoding**, the same one `0x413dc3400` decodes
at pc618–619 with `v_lshrrev_b32 v36, 3, v69` followed by `v_add_nc_u32 v36, -4, v36` —
`index = (ref >> 3) - 4`. The source array at pc70 (stride 8, 2,064 records) is an array of **node
references**, and `first >> 3` recovers `index + 4`, exactly as observed.

**So `selected_sbuffer_domain`'s proof passes for an accidental reason.** It admits a record when
`selector * 120` is either exactly 480 (record 4) or wholly out of bounds. With `selector = index+4`:
index 0 gives selector 4 → offset 480 → "record 4"; indices 1..2063 give selectors 5..2067 → offsets
600..248,040, every one past the 600-byte buffer → "wholly OOB". **Both arms are satisfied by an
array that is not a selector array at all.** The proof is not wrong about the arithmetic; it is
answering a question about the wrong data.

The contract then reads 16 bytes at `480 + 8` of a 600-byte buffer that holds float data, gets a
non-descriptor, and declines — correctly, at the last possible moment, having been misled four steps
earlier.

**This is the complete account of `0x413ce6000`'s reject**, and it is a contract defect rather than a
missing lowering, a stale buffer, or a wrong selector value:

1. the source array at pc70 holds BVH node references, not table selectors;
2. `selected_sbuffer_domain` reads `first >> 3` as a selector and its two admissible arms are both
   satisfied by that data for arithmetic reasons;
3. the contract therefore reads record 4 of the outer buffer;
4. the outer buffer holds float data in this scene (unit-normal plane equations, measured);
5. `selected_sbuffer_target_descriptor` rejects it;
6. no `selected_sbuffer_soffset` authority is published, so the descriptor for
   `buffer_load_dwordx3` at pc156 never resolves;
7. `mode=unresolved-operand pc=156`, and the program is declined.

**`CONFIDENCE: HIGH`** on steps 1–5, all measured. **`CONFIDENCE: MED`** on 6–7 being the only
remaining link, since the recompiler's descriptor matching has three routes and only the contract one
has been traced.

## The `selected_sbuffer` contract is over-fitted: the "descriptor array" holds FRUSTUM PLANES (2026-08-15)

Dumping **all five** records of the outer array on decline, rather than only the one the contract
reads, settles what one record could not:

```
record=0@+8   words=3e177e0d:3d39304e:3f7ceb16:beec6371
record=1@+128 words=3d044294:bc9a100f:3ce5d234:be5601a6
record=2@+248 words=bdd106eb:3f7e94ad:bccf32ed:beee4863
record=3@+368 words=be6fa5cf:bdb9257a:3f581219:bdf01a64
record=4@+488 words=3f1a1f1a:bdc0f619:3f4afad1:bfa0b479
```

**None of the five is a descriptor. All five are floats, and they are unit normals plus a scalar:**

| record | xyz | ‖xyz‖ | w |
| --- | --- | --- | --- |
| 0 | (0.1479, 0.0452, 0.9879) | **0.999** | −0.461 |
| 2 | (−0.1020, 0.9944, −0.0253) | **1.000** | −0.465 |
| 4 | (0.6021, −0.0942, 0.7929) | **1.001** | −1.256 |

A unit 3-vector with a scalar is a **plane equation**. Five of them at stride 120 is a camera
frustum, not a descriptor table.

**So in the gameplay scene, the buffer the contract reads as a five-entry V# array holds frustum
planes.** The contract was derived from a phase where record 4 did hold a V#, and it does not
generalise. That also explains its companion `reject=consumer-resource`.

This is why the earlier framing — "record 4 is stale, find its producer" — was the wrong question.
The buffer is not stale; it is a **different buffer's worth of data**, correctly written by whoever
owns it. Either `0x413ce6000`'s pc153 does not load a descriptor on this path in this scene, or the
resource the contract binds at `fetch_pc=153` is not the one the guest means here.

**`CONFIDENCE: HIGH` that these are planes and not descriptors** — three of five have unit-length
normals, which floating-point garbage does not do. `CONFIDENCE: MED` on the consequence, because
"the contract binds the wrong resource" and "the shader takes a different path here" both fit and
have not been separated.

## `0x209cc76000` is a SHARED POOL with 23 writers, not a dedicated record array (2026-08-15)

Watching the **whole** 132,032-byte range (`PROSPER_COMPUTE_TREE_WATCH=0x209cc76000:33008`) rather
than its first 8 KB finds **23 distinct writing programs**, not the seven the narrower window showed:

```
0x413e15400 652   0x413e14200 651   0x413e14500 651   0x413cf9200 603   0x413cf9000 569
0x413cdc200 218   0x413cf5400  43   0x413cf6100  43   0x413ce6000  32   0x413e1ff00  31
0x413ced900  10   0x413d1bf00   9   0x413d21600   8   0x413dc3400   6   0x413d87800   5
0x413e16400   5   0x413e13000   3   0x413e14900   2   0x413d21800   1   0x413d21c00   1
0x413d22000   1   0x413d22b00   1   0x413e13200   1
```

**So this is a shared scratch pool that many programs reuse across phases**, exactly like the
traversal table itself. `132032 = 2063 x 64` is how `0x413dc3400` and `0x413ce6000` *view* it, not
proof that the allocation belongs to them.

Two consequences for anything built on the earlier analysis:

- **"The only program binding the full array is `0x413ce6000`" was a statement about the narrow
  window.** Over the full range there are 23 writers, and the three busiest — `0x413e14200`,
  `0x413e14500`, `0x413e15400`, ~650 changes each — were entirely invisible to every census before
  this one. A watch window narrower than the buffer under study reports a writer set that is
  guaranteed incomplete.
- **The tag histograms remain valid** because they were captured by the aux dump *at the builder's
  own dispatch*, which is the only moment that matters. But attributing a tag change to a producer is
  not possible from writer counts alone with 23 of them interleaved; it needs the last-writer-per-
  record, which nothing currently records.

The early phase is visible in the same data and is a different workload: at submit 3864
`0x413ced900` leaves the pool all-zero (`{0: 2063}`) and `0x413d1bf00` fills it progressively over
eight dispatches. Tag distributions there span all eight values, unlike the two-regime pattern at
builder time.

## The builder's INPUT differs between the two regimes (2026-08-15)

`PROSPER_COMPUTE_TREE_WATCH_AUX=0x209cc76000:33008` captures the 2,063 x 64-byte record array
alongside every builder transition, clean and cyclic, from one run. The tags the six store predicates
test are `record.dword0 & 7` and `record.dword1 & 7`.

**Every submit where the builder emits a perfect tree shares one input signature, and no broken
submit has it:**

| submit | pairs / unpaired | tag(dword0) | tag(dword1) |
| --- | --- | --- | --- |
| 5943 | **1029 / 2** | `{0:970, 2:2, 5:1088, 7:3}` | `{0:977, 1:5, 5:1078, 7:3}` |
| 7188 | **1029 / 2** | `{0:972, 2:2, 5:1086, 7:3}` | `{0:977, 1:5, 5:1078, 7:3}` |
| 8016 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 8842 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 9247 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 5528 | 663 / 64 | `{0:1120, 2:3, **4:2**, 5:935, 7:3}` | `{0:1122, **1:15**, 5:923, 7:3}` |
| 6358 | 608 / 51 | `{0:1098, 2:3, **4:2**, 5:957, 7:3}` | `{0:1107, **1:15**, 5:938, 7:3}` |
| 10052 | 676 / 74 | `{0:1133, 2:3, **4:2**, 5:922, 7:3}` | `{0:1134, **1:15**, 5:911, 7:3}` |

Two differences are systematic in the early broken submits: **tag 4 appears in dword0** (exactly 2
records, never present in a perfect submit) and **tag 1 in dword1 jumps from 5 to 15**.

**But neither is the whole story, and the record says so.** The later broken submits (11238 onward)
carry `1:5` and no tag 4 — the perfect signature on those two axes — and are still broken
(`pairs≈861, unpaired≈193`), with tag 5 *higher* than in the perfect submits (1301 against 1085). So
"tag 4 present" and "tag1 == 15" are correlates of the early regime, not the mechanism. Do not
promote them to a rule.

**What is established:** the array the builder reads changes materially with scene state, and the
builder's output tracks it. Combined with the eleven perfect submits, that puts the defect **upstream
of `0x413dc3400`** — in whatever produces `0x209cc76000`.

**Its producers, from `PROSPER_COMPUTE_TREE_WATCH=0x209cc76000`:** `0x413cf9000` (117 changes),
`0x413cf9200` (117), `0x413cf5400` (29), `0x413cf6100` (29), `0x413ce6000` (26), `0x413d1bf00` (2),
and `0x413e1ff00` (3, **`toucher=0`**). `0x413cf9200` has 20 recompile-empty dispatches of 678.

**RETRACTED — `0x413e1ff00` does NOT write bytes it does not bind.** That was my own instrument
producing a phantom. Its binding 7 is `base=0x209cc76080 size=160 stride=32`, i.e. **128 bytes inside
the watched window**. The tree watch detects changes anywhere in the WINDOW but its `toucher` field
asked whether a program binds the window's **first byte** — two different questions, and the
disagreement reads as an out-of-bounds write. Fixed with `compute_address_window_hits`, and the
retraction is recorded rather than quietly dropped because the phantom was reported as a lead before
the resource map contradicted it.

The same run corrects the producer picture in a way that matters: `0x413cf9000` binds only
`0x209cc76000 size=320`, and `0x413cf9200` binds `size=320` plus a 64-byte view at `+0x140`. **They
write the first 320 bytes — five records — not the bulk.** The only program binding the full
132,032-byte array is `0x413ce6000` (bindings 13/14/17). So it *is* the bulk producer of the records,
which the falsification above does not contradict: it executes on 29 of the cyclic submits, so its
running is not the variable.

## THE BUILDER'S LOWERING IS CORRECT — it emits a perfect tree for eleven submits (2026-08-15)

`PROSPER_COMPUTE_DISPATCH_LOG` now carries the launch geometry, so the outcome and the launch can be
read on one line. Across the whole route:

```
submit  threads       local     tree
4802    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0     <- exactly correct
5217    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
5632    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6047    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6463    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6877    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
7292    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
7706    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
8121    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
8951    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
9360    2063x1x1      256x1x1   pairs=852  unpaired=247 cycles=57  <- and never correct again
...     2063x1x1      256x1x1   cyclic for the remaining 30+ submits
```

**Eleven consecutive submits at pairs=1030, unpaired=0, cycles=0 — the exact clean ground truth — at
an identical launch geometry, from the same compiled module.** A miscompiled shader does not produce
the exactly correct answer eleven times in a row.

**So `0x413dc3400`'s lowering is CORRECT, and the defect is in what it is fed.** Every
lowering-side hypothesis for this program is retired by this one measurement: barrier placement,
LDS sizing, wave model, the compaction, the exec-mask predication, the store addressing. They all
produce a correct tree for eleven submits.

The launch is identical on both sides of the boundary — `threads=2063x1x1 local=256x1x1` throughout —
so an active-record count change is not the trigger either. The transition point is **not a fixed
submit number**: one run breaks at ~6795, another at ~9360, so it tracks route position and scene
content rather than a dispatch ordinal.

**The remaining question is therefore narrow and concrete: what changes in the builder's input at
that boundary?** Its tags come from `0x209cc76000` (2,063 × 64 bytes, binding 4 / fetch-pc 86).
`PROSPER_COMPUTE_TREE_WATCH_AUX=0xADDR:DWORDS` (added here) dumps a second guest range alongside the
watched table, and dumps the builder's clean transitions too so there is a control from the same run
rather than only cyclic samples.

## FALSIFIED: the missing-producer hypothesis is dead (2026-08-15)

`PROSPER_COMPUTE_DISPATCH_LOG` (added here) records **one line per dispatch** with its outcome, which
no existing signal did — every other one is deduped per program, and reading a once-per-program line
as a per-dispatch property is how the root-cause claim two sections down got published and retracted.

One 200 s route, 46 submits carrying a `0x413dc3400` dispatch:

| | tree CYCLIC | tree clean |
| --- | --- | --- |
| `0x413ce6000` had a declined dispatch that submit | 10 | 0 |
| `0x413ce6000` executed on **every** dispatch | **29** | 7 |

**29 submits in which the producer ran on every single dispatch and the builder still produced a
cyclic tree.** A producer decline is therefore **not necessary** for the corruption, and the
"`0x413ce6000` fails to recompile → stale tags → cyclic tree" story is finished. Do not restart it.

Per-dispatch outcomes over the run: `0x413ce6000` 129 executed / 10 recompile-empty; `0x413cf9200`
658 executed / 20 recompile-empty; `0x413dc3400` 53 executed / 0 failures. Those recompile-empty
dispatches are real gaps worth closing on their own merits — an unsupported program is a fatal gap —
but they are **not** the cause of this defect.

### What the same data shows instead: a route-position boundary

The builder's output is clean for the first seven submits and cyclic from submit ~6795 onward,
continuously:

```
3894  clean     6795  CYCLIC     9673  CYCLIC
4307  clean     7211  CYCLIC    10076  CYCLIC
4720  clean     7625  CYCLIC    10482  CYCLIC
5135  clean     8039  CYCLIC    10875  CYCLIC
5550  clean     8457  CYCLIC    11271  CYCLIC
5965  clean     8867  CYCLIC    11674  CYCLIC
6380  clean     9274  CYCLIC    12077  CYCLIC
```

This is a **transition at a point in the route**, not a per-submit coin flip. Whatever changes around
submit 6795 — scene content reaching some size or shape — is the thing to characterise next. The
first clean submit also shows `0x413ce6000` dispatching **94** times against 1 thereafter, so the
early phase is a different workload entirely and the clean result there may not be comparable.

## Causal A/B: forcing the producer off collapses the builder entirely (2026-08-15)

`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700,0x413ce6000`, lever verified in the log
(`-> 2 program(s) will be declined`), 200 s route, against the baseline arm.

| | baseline | producer forced off |
| --- | --- | --- |
| `0x413dc3400` table-changing dispatches | **40** | **1** |
| `0x413dc3400` clean -> cyclic | 37 | **0** |
| clean -> cyclic, all programs | 40 (37 from the builder) | 46 (all from `0x413d88400`) |

**`0x413dc3400` essentially stops writing a tree.** So its output does depend on `0x413ce6000`
having run — the dependency the resource alias predicted is real and now demonstrated, not inferred.

**The confound, stated because it is not excluded.** A declined compute dispatch clears
`producer_epoch_ok`, and the next `ParserStall` latches `indirect_dependencies_ok` false for the rest
of the submit — the indirect latch this document opens with. Forcing `0x413ce6000` to decline
therefore suppresses later indirect dispatches too, so this arm cannot separate

  (a) `0x413dc3400` runs and reads a stale record array, from
  (b) `0x413dc3400` is never dispatched at all.

Both predict what was measured. Separating them needs a per-dispatch execute/decline record for both
programs in the same run, correlated frame by frame — which is the measurement the retracted
root-cause claim needed and never had.

Also note the dumped post-images differ in *provenance* between the arms: in the forced-off arm the
clean -> cyclic transitions come from `0x413d88400`, not the builder, so their `pairs=0` is not
comparable to the baseline builder's `pairs≈819`. The comparable figure is the dispatch count.

## CORRECTION (2026-08-15): the "never executes" claim above is WRONG

Watching the record array itself — `PROSPER_COMPUTE_TREE_WATCH=0x209cc76000:2063` — refutes the
strongest form of the claim in the section below. **`0x413ce6000` does write it**, 26 times on a
150 s route, `toucher=1`. It is not declined on every dispatch; the `[compute] skip unsupported
program` line that suggested otherwise fires **once per program ever**, so it reports "failed at
least once", never "never ran". That distinction is stated elsewhere in this document and I still
built a causal claim on the wrong side of it.

The array's actual writers on one route:

| program | changes | toucher |
| --- | --- | --- |
| `0x413cf9000` | 117 | 1 |
| `0x413cf9200` | 117 | 1 |
| `0x413cf5400` | 29 | 1 |
| `0x413cf6100` | 29 | 1 |
| `0x413ce6000` | **26** | 1 |
| `0x413d1bf00` | 2 | 1 |
| `0x413e1ff00` | 3 | **0** |

**What survives** from the section below, because it was measured rather than inferred:

- `0x413dc3400` corrupts the parent table on 37 of 40 dispatches. Unchanged.
- `0x413dc3400`'s tag source is `0x209cc76000`, binding 4 / fetch-pc 86 — the resource-map alias is
  exact and stands.
- `0x413ce6000` writes that same array, is *sometimes* declined, and its reject is
  `unresolved-operand pc=156`. Also stands.

**What does not survive:** "`0x413ce6000` never executes, therefore the tags are stale, therefore the
tree is cyclic." The producer runs most of the time, so a missing-producer story cannot be asserted
on this evidence. It remains a *candidate* — a partially-written array would still leave some records
stale — but the step from "sometimes declined" to "these particular tags are stale" is not made, and
the frame-level correlation between a decline and a corrupted tree has not been measured. **Do that
measurement before building on it.**

`0x413e1ff00` changing the array three times with **`toucher=0`** is the case the tree watch was
built to be able to report: a program that changes bytes without binding a range containing them.
That is either an out-of-bounds write or a binding path the resource traversal does not model, and
it is unexamined.

## `0x413ce6000` is a producer of the tags, and is sometimes declined (2026-08-15)

**Read the CORRECTION above first — this section's causal claim was overstated and the
"never executes" premise is refuted. The resource alias it establishes is sound.**

The chain is complete and every link is measured.

**The alias, from `PROSPER_COMPUTE_RESOURCE_MAP`:**

```
0x413dc3400  binding=4  fetch-pc=86   base=0x209cc76000 size=132032 stride=64   (READ)
0x413ce6000  binding=13 fetch-pc=212  base=0x209cc76000 size=132032 stride=64   (STORE)
0x413ce6000  binding=14 fetch-pc=214  base=0x209cc76000 size=132032 stride=64   (STORE)
0x413ce6000  binding=17 fetch-pc=253  base=0x209cc76000 size=132032 stride=64   (STORE)
```

`132032 = 2063 x 64` — one 64-byte record per node, exactly the node count. `fetch-pc=86` is the
load Codex identified as the source of the classified references: `pc86 loads v56/v57 from
s24[compacted_node].dword[0:1]`, and `v36 = v56 & 7`, `v37 = v57 & 7` are the tags the six store
predicates test. Eight more of `0x413dc3400`'s bindings (6, 7, 8, 9, 10, 11, 12, 15) are the same
buffer.

**So `0x413ce6000` writes the record array whose tags `0x413dc3400` reads — and `0x413ce6000` is
declined on every single dispatch.**

### The full causal chain

1. `0x413ce6000` fails to recompile: `compute-struct-reject execz pc=76 scalar write pc=84 leaves
   vcc-half s106 live at merge pc=139`. It is realized 40 times per route, at d36, and executes zero
   times.
2. Its output — the 2,063 x 64-byte node record array at `0x209cc76000` — is therefore never written.
3. `0x413dc3400` at d37 reads its six classified references out of that array.
4. The stale references fail the `tag == 2 || tag == 5` predicate, so the matching child stores never
   fire — which is exactly the measured failure mode: **a store that does not execute, leaving the
   slot holding a stale value**.
5. 220–461 parents end up with one child instead of two → the parent array is cyclic.
6. `0x413dc6700` at d39+ walks it with a loop that has no iteration bound and no cycle detector →
   GPU hang → RADV device loss → live compute disabled process-wide → every later indirect draw
   dropped → **no world**.

**This is the charter's rule in its purest form: an unsupported program is a fatal gap, not an
acceptable skip.** One declined compute program, six dispatches upstream, removes the world.

It also explains why `0x413ce3400` gets its tree right. Codex's ISA read: it computes both child
indices directly from the Morton range/split and stores both links unconditionally, with no tag
predicate anywhere in its 391 instructions. It does not depend on `0x209cc76000`, so a missing
producer cannot reach it.

**Do not "fix" this by removing `0x413dc3400`'s predicate.** The guest declines to dereference
non-materialised references on purpose; making the stores unconditional would turn unmaterialised
references into indices and change guest semantics. The fix is upstream: make `0x413ce6000` compile.

### Status of that fix

`PROSPER_VCC_SCALAR_DATA_MERGE=1` (this branch) widens the execz VCC-half proof from "not read" to
"not read as a lane mask" and clears the first reject; `0x413ce6000` then fails at a *second* path
that logs nothing under any variable. Making that second reject legible is the immediate next step —
it is one of the five programs in the `reason=unrecorded` set.

Note on a red herring: `[compute] program 0x413ce6000 is a proven no-op (only proven no-backing
resources)` does appear, but only for an instance with **zero** backed resources (dispatch 968). At
d36, where it matters, it has **17** backed resources including the three stores above.

## THE SECOND TABLE IS BUILT PERFECTLY — by a different program (2026-08-15)

`PROSPER_COMPUTE_TREE_WATCH=0x20f848a240:2063`, same route. The two addresses are **not** a
ping-ponged copy of one structure: they have disjoint writer sets.

| address | writers | result |
| --- | --- | --- |
| `0x20f848417c` | `0x413dc3400`, `0x413d88400`, `0x413e1c300`, `0x413cee500` | **37 of 40 dispatches leave it cyclic** |
| `0x20f848a240` | `0x413ce3400`, `0x413cf5400`, `0x413d88400`, `0x413cdc200` | **clean; 3 of 133 transitions cyclic, each repaired immediately** |

And the decisive line: **`0x413ce3400` produces a PERFECT tree at `0x20f848a240`, every single time.**

```
program=0x413ce3400 changed=2061 pre{... pairs=0 unpaired=0}
                              post{cycles=0 cyclic-roots=0 oob-roots=0 max-depth=11 pairs=1030 unpaired=0}
```

`pairs=1030, unpaired=0, cycles=0, max-depth=11` — exactly the clean ground truth, on all 41 of its
dispatches. So **prosper can already build this structure correctly.** The defect is specific to
`0x413dc3400`, and there is now a working reference program to differentially compare against.

The two are different programs, not two instances of one:

| | `0x413ce3400` (correct) | `0x413dc3400` (broken) |
| --- | --- | --- |
| instructions | 391 | 765 |
| DS (LDS) ops | **0** | 5 (`ds_write_b32` x2, `ds_add_rtn_u32`, `ds_read_b32` x2) |
| `s_barrier` | **0** | 2 |
| CFG dispatchers | — | 3 (one per barrier-free phase) |
| MUBUF | 29 | 37 |

**The distinguishing features of the failing one are exactly the workgroup-cooperative machinery**:
the LDS stream compaction Codex decoded at pc47..74 (`ds_add_rtn_u32` hands each qualifying lane a
ticket; the two barriers bracket counter initialisation and list publication), and the barrier-phased
CFG dispatcher that machinery forces. The correct builder has none of it.

That also resolves an inconsistency in the record: the consumer's own captures show ~1029-1030 pairs,
which matches the `a240` tree rather than `417c`'s 919. The consumer's bound table address alternates
between the two across frames, so it walks a good tree on some frames and the broken one on others —
which is why the hang arrives at a particular dispatch rather than the first.

## The failure mode, exactly: one of the two per-parent stores does not execute

Established from both images of 18 clean -> cyclic transitions.

**Ground truth.** Two clean captures give the target unambiguously: parent-multiplicity
`{2: 1030, 1: 2}` — every parent has exactly two children — with 1,032 distinct parent values over
0..2062, 515 of them above 1031, and 1,031 indices never a parent. Leaves and internals are
interleaved, not segregated into low and high halves.

**`0x413dc3400`'s output** measures `{2: ~872, 1: 220..461}`. For 220–461 parents **exactly one of
the two children carries that parent**, and the missing child's slot still holds a *stale* value —
`0x09249249`, `0x12492492` or `0x2db6db6d`, which are the Morton dilation constant times 1, 2 and 5,
or plain zero. Those slots were never written by this dispatch. **So this is a store that does not
execute, not a store that computes a wrong address or a wrong value.**

### The store site

All six stores sit in six separately exec-predicated blocks of identical shape (mnemonics from
prosper's own decoder, `rdna2_to_spirv.cpp` VOP2 0x16 `v_lshrrev_b32`, 0x1B `v_and_b32`,
0x25 `v_add_nc_u32`):

```
pc=0611  v_and_b32          v36, 7, v69      ; tag = child_ref & 7
pc=0612  v_cmp_eq_u32       s8,  2, v36
pc=0614  v_cmp_eq_u32       vcc, 5, v36
pc=0615  s_or_b64           vcc, vcc, s8     ; tag == 2 || tag == 5
pc=0616  s_and_saveexec_b64 vcc, vcc
pc=0617  s_cbranch_execz    -> 622
pc=0618  v_lshrrev_b32      v36, 3, v69      ; index = child_ref >> 3
pc=0619  v_add_nc_u32       v36, -4, v36     ; index -= 4
pc=0620  buffer_store_dword v40, v36, s0, 0  ; idxen=1, offen=0
pc=0622  s_mov_b64          exec, vcc
```

A child is written only when its reference's low three bits are 2 or 5, into slot `(ref >> 3) - 4`.
**Whether tags outside `{2, 5}` are written by a LATER PASS is the open question** — if they are,
a stale slot here is expected and the defect is a missing program rather than this one (#2542).

### A gap in the toucher census that may matter more

The baseline routed run **skips 13 compute programs as unsupported**, and a skipped program never
realizes a resource table — so `PROSPER_COMPUTE_ADDRESS_WATCH` and the tree watch are
**structurally incapable of seeing any of them**. The clean "no unknown writer" result is therefore
a statement about programs that ran, and only those. Baseline skip list:

```
0x2042f49a00 0x205b545c00 0x205b54ee00 0x205b5e8600 0x205b654a00 0x205b657200
0x205b658800 0x205b67ce00 0x413ce6000 0x413cf9200 0x413cf9a00 0x413cf9d00 0x413d14100
```

If one of those is the pass that fills the slots `0x413dc3400` deliberately leaves alone, the cyclic
table is a **missing program**, not a miscompiled one, and the fix is to make it recompile. Per the
charter's rule that an unsupported program is a fatal gap rather than an acceptable skip, these are
the next thing to implement regardless of how this particular question resolves.

## RETRACTED: "cube shadows are not the missing world" (2026-08-16)

That conclusion is **void, not a demonstrated negative**, and the reason is worth more than the claim.

Pass grouping's `same_targets` compared colour targets, formats and resolve mode — and **not the
depth/stencil attachment**. A render pass has one depth attachment and the backend selects one cached
DS image for the whole grouped call from its first meaningful draw, so a single call could span draws
naming different DS surfaces, or different `DB_DEPTH_VIEW` **slices of one layered allocation**. Every
such draw rendered into whichever face the call happened to select. Codex measured one grouped call
crossing slice 0 → slice 1 at draw 65.

So when I reported "six valid faces" under `PROSPER_DS_GUEST_WRITE_INVALIDATE=0`, the six *handles*
existed and their *contents* did not correspond to six guest faces. The lever I thought I had moved
was never moved.

**Fixed — but NOT on master, and not in the PR that carries this document.** The grouping split and
the cube-face bridge live in the stacked follow-up **#2553**, which is unmerged at the time of
writing; review moved them out of #2552 because they change renderer behaviour at production seams
and need regressions of their own. Everything measured in this subsection was measured with that
branch applied. Do not read the table below as describing current master, and check #2553's state
before building on it.

Grouping there splits on DS identity — depth/stencil bases, HTILE base, and the
`DB_DEPTH_VIEW` slice. Measured on the same route with defaults (invalidation ON):

| cube | faces valid before | after |
| --- | --- | --- |
| `0x2094ec0000` | 1 (`0x01`) | **5** (`0x1f`) |
| `0x20948c0000` | 1 (`0x01`) | **3** (`0x07`) |
| `0x208f340000` | 4 (`0x1e`) | 4 (`0x1e`) |
| `0x2097ec0000` | 5 (`0x3e`) | 5 (`0x3e`) |

Multiple guest faces really were being rendered into one host face, and the earlier per-slice census
could not see it because the census counted the passes the backend *made*, not the faces the guest
*asked for*.

**The cube A/B is still open.** Reaching 6/6 requires the invalidation switch, which perturbs the
frame on its own (the HUD disappears in that arm), so "world still black with the cube bridged" is
not a clean measurement either. The honest state is: the bridge works when faces are resident, the
grouping defect that corrupted them is fixed, and whether the cube is load-bearing for the world is
**not yet established in either direction**.

### The pattern, again

This is the third time this session a negative result turned out to be void because the lever was not
actually moved — after the compute-program "never executes" retraction and the RTTLOG readback that
materialised the pixels it measured. Each time the surface evidence looked sufficient. The thing that
caught all three was someone asking *what would have to be true for this measurement to mean what I
think it means* — twice that someone was Codex.

## Fifth review round on #2550, all four fixed (2026-08-15)

**1 (blocker) — MRT1 was still discarded under the live renderer's calling convention.** The live
call passes `out_rgba1 == nullptr` and receives slot 1 through `BackendMrtOutputs`. On a non-final
split the slot-1 readback therefore landed in `intermediate_mrt.colors[1]`, while the wrapper carried
a seed forward only inside `if (out_rgba1)` — so MRT1 was lost on every split. The MRT2 regression
could not see it because it jumps from MRT0 straight to MRT2. **The claim in the previous round that
"slots 0 and 1 already carry through `seed_rgba`/`seed_rgba1`" was true only of the legacy explicit
API, and is corrected here.** Added `seed_rgba1_slot`, the carry now takes whichever buffer received
slot 1, and there is a second Vulkan split regression written with the live signature.

**2 (high) — the MRT2–7 carry was selected by identity, not residency.** A non-zero
`persistent_id_slots[slot]` does not mean the image survives: persistence also requires
`persistent_color_targets_enabled` and a successful cache/budget allocation, and the slot-creation
path falls back to a transient image while leaving the identity non-zero. The wrapper cannot see
those decisions, so it no longer predicts them — a non-final split segment captures every slot
unconditionally. Regressed with a non-zero MRT2 identity and
`PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS=1`.

**3 (high) — the materialization changes had no regression at their call sites.** Reverting either
production site left everything green. Both decisions are now seams that own their gate —
`mrt_direct_serves()` and `mrt_uniform_live_serves()` — and the mutation reverting the gate inside the
seam turns a named assertion red.

**4 (medium) — the binding test asserted a contract production does not have.** `backend_color_format`
maps every unrecognised value, zero included, onto `R8G8B8A8_UNORM`, so the format term in the
active-binding rule is **total**: it never rejects a slot. The test modelled `raw != 0` and asserted a
zero format made a slot inactive. `PROSPER_MRT_CENSUS` had already reported "format known" for every
slot in all 16,384 pass groups of a routed boot — the column is constant because the predicate is —
and that measurement is in this document. I read it, wrote it down, and then wrote a test asserting
the opposite. The test and the header now encode the real contract, and a backend-linked test pins
the mapping the model depends on.

### A metric with two regimes, and a control that was not one

While verifying, `c4`'s peak read 507,887 where the previous round recorded 3,471,942. A second run
on the same head gave 516,241, and I took that agreement as evidence of a regression in my own diff.
It was not: **rebuilding the PREVIOUS head's sources and running them today gives 507,863.** The
metric has two regimes depending on where the route lands, and two runs inside one regime agree with
each other while saying nothing about the other.

The lesson is the one this document keeps recording in different clothes: *the control has to be the
old code run now*, not a number written down earlier. A remembered measurement is a measurement of a
different machine state. `c2`/`c3` are stable across all of these (4.51–4.53M) and are the numbers
worth quoting.

## Fourth review round on #2550, all four fixed (2026-08-15)

**1 (blocker) — a depth-feedback split still discarded MRT2–7 when persistence was unavailable.**
Carrying the SHAPE through every segment was necessary and not sufficient. When the caller names no
persistent identities — every path running with live GPU targets disabled: captures, per-target
diagnostics, replay export, the recovery switch — the later segment gets a fresh transient
attachment, and LOADing an image that was just created loads nothing. The earlier segment's pixels
reach it only by being read back and **seeded** in, which is exactly what slots 0 and 1 already do
through `seed_rgba`/`seed_rgba1`.

Added `seed_slots[]`, the per-slot twin, with the upload path mirroring slot 1's. The splitter reads
back any non-persistent higher slot it must carry and hands the pixels to the next segment.
`split_segment_contract()` synthesises a carrier target when the caller gave none, since the seed
flags have nowhere else to live — behaviourally identical to no target (every identity is zero) apart
from the carry.

**The regression is a real Vulkan one, and it reproduced the defect before the fix.** Segment 1 writes
depth and RED to MRT2; the consumer samples that depth, which forces the physical split; the final
segment never touches MRT2; `color_target` is null. Before the carry, the final MRT2 read
`(0,0,0)` — after it, `(255,0,0)`. The test also asserts `depth_feedback_split_index() == 1`, so it
cannot pass by failing to split.

**2 (high) — same-pass feedback materialization kept MRT0-only gates.** The direct-path precheck and
the uniform fast path still compared against `draw.color0_base`. For an MRT2+ collision the precheck
said "direct serves", which suppressed the lazy CPU materialisation, and the corrected gate then
refused the direct image — leaving the resource to fall through to guest bytes with no snapshot to
use. All three sites (including the diagnostic classification) now use the all-slot rule.

**3 (medium) — the feedback helper was a second, looser copy of "active".** It fell back to the named
slot-0/1 mask whenever the array mask read zero and never required a defined format, where pass
grouping falls back only when the array representation is ABSENT and requires base + format + mask.
Extracted as `frontends/shared/rtt/mrt_binding.hpp`, used by both, and tested from a real `DrawItem`.

**4 (low) — `test_mrt_extent` printed `OK` before the new checks ran.** Third instance in this branch
of a success signal placed ahead of the work it describes.

### Four rounds, one recurring mistake

Rounds two, three and four each contained at least one finding of the form *a check that cannot
fail*, and three of those were mine in the same shape: **new assertions appended at the end of a
file, landing after the success print or the fail gate.** The cause is mechanical — inserting before
the final `return` — and the fix is mechanical too: put new checks with the checks, and read what is
between them and the exit.

## Third review round on #2550, all four fixed (2026-08-15)

Every one a legacy "only MRT0/1 exist" seam that the widening newly reached.

**1 (blocker) — a depth-feedback split dropped MRT2–7 from every non-final segment.** The splitter
passed `final ? mrt_outputs : nullptr`, and `render_draw_pass_rgba` derives `color_count` from that
pointer — falling back to `out_rgba1`, which the live renderer passes as null. So every pre-final
segment of a five-MRT pass rendered with **one** attachment and discarded its MRT1–4 exports; later
segments were never told to LOAD the higher slots either. The same accumulation loss this PR fixes at
the frontend's pass groups, reappearing at the backend's own physical boundary.

Fixed by extracting `split_segment_contract()`: the attachment SHAPE is identical across segments and
only the pixel destination and the load/readback flags differ. Non-final segments now decline
readback **per slot** — `color_count > 2` alone forced a full-extent copy of every higher slot on
every segment, so the fix needed `readback_slots[]` to exist at all.

**2 (high) — same-pass feedback detection knew only MRT0/MRT1.** Now that a higher slot's image is
persistent and sampled-capable, sampling an active MRT2+ target borrowed the very `VkImage` the pass
was writing. Both halves — the backend's descriptor borrow and the frontend's direct-live decision —
now go through one `mrt_target_feedback()` policy.

**3 (high) — the sparse-MRT gate had no regression at its production seam.** Reverting
`any_slot_bound` to `base` left all 245 tests and the validation scan green. The union is now
`mrt_any_slot_bound()` in the frontend policy header, called by both decisions that key on it and
tested there.

**4 (medium) — the no-readback publication lifecycle skipped MRT2–7.** `publish_persistent_color()`
covered slots 0 and 1; a persistent higher slot could reach `SHADER_READ_ONLY_OPTIMAL` with no
barrier making its writes visible to a later command buffer. Mirrored for every retained slot, keyed
on whether that slot's readback path actually ran.

All four are mutation-verified at their own sites: dropping the segment shape, dropping the segment
load, narrowing the feedback policy to slots 0/1, and narrowing the union to colour-0 each turn a
named assertion red.

Live title after this round: **c2 4,533,513 · c3 4,533,513 · c4 3,472,691** non-black pixels,
0 validation errors.

### The seam this round is really about

Every finding was the same shape: a policy written when only two colour slots could exist, left
behind by a change that made eight possible. They were not in the diff — they were in code the diff
made reachable. Grepping for `persistent_id1` or `color1` finds them; grepping for what changed does
not.

## Second review round on #2550, all five fixed (2026-08-15)

**1 (blocker) — MRT2–7 reused an image in the wrong Vulkan layout.** The retained image was recorded
as `COLOR_ATTACHMENT_OPTIMAL` while the pass left it in `TRANSFER_SRC_OPTIMAL` and no restore
transition existed, so the next group declared an initial layout the image was not in. Codex
reproduced it under `VK_LAYER_KHRONOS_validation` as `VUID-vkCmdDraw-None-09600`; the pixel assertion
still passed on RADV, so the ordinary suite could not see it. Slots 2–7 now take the complete MRT0/1
lifecycle — pre-pass load barrier from the truthful current layout, persistent final layout, readback
transition, restore, and a recorded layout that matches. Mutation-verified: restoring the old
recorded layout brings back exactly `VUID-vkCmdDraw-None-09600` (x2) plus
`VUID-VkImageMemoryBarrier-oldLayout-01197` (x8); the fix gives **zero**, which also proves the layer
was active rather than silently absent.

**The gate already existed.** `tools/vkval/vk_validation_scan.py` runs the whole suite under the
layer, allow-lists known IDs, fails on unknown ones, and refuses a clean verdict unless the layer
provably loaded. Neither VUID is allow-listed, so that scan *is* the regression: `VKVAL_RC=0` now,
and it fails with the defect present.

**2 (high) — sparse MRT passes bypassed the persistence contract.** The colour target was passed to
the backend only when MRT0's `base` was non-zero, and the comment at that very call site already
documented that MRT0-unbound / higher-slot-bound passes are reachable. Such a pass populated
`persistent_id_slots[2..7]` and then handed the backend `nullptr`. Now gated on the union of every
bound slot — the same union the readback flag beside it already used. Effect on the live title: **c4
513,338 → 3,472,691** non-black pixels.

**3 (high) — the cap-order and baseline tests did not exercise the production sites.** They called
two pure predicates, so moving the cap check back below the capture, or deleting the baseline latch,
left every assertion green. Replaced with orchestration seams that *own* the properties:
`frame_bundle_append_submit()` takes the capture step as an injected callable, so a test observes
whether it ran; `frame_bundle_open_window()` owns the latch. Both mutations Codex named now fail —
moving the cap below the capture gives 2 failures, deleting the latch gives 3.

**4 (medium) — the MRT2 negative arm could skip itself.** It entered its only assertion under
`if (cleared.size() == px)` without ever asserting that size, so a regression returning an empty
buffer would have passed the negative control silently. The extent is asserted now.

**5 (medium) — the capture size bound ran before the v55 tail.** A capture near the 4 GiB ceiling
could serialize successfully into a file `read_gpu_capture` then rejects as oversized. Re-checked
after the final tail, with a note that any future tail must move the check again.

### The pattern across both review rounds

Four of the nine findings across the two rounds are the same shape: **a check that cannot fail.** The
negative arm gated on a size it never asserted; the cap and baseline tests called predicates instead
of the sites; the timeline assertions sat after their own fail gate; and the layout defect passed
every pixel assertion while being undefined behaviour. Only the last needed a tool to see — the other
three were readable in the diff, and I wrote all of them.

## Review findings on #2550, all four fixed (2026-08-15)

Codex reviewed the MRT work and found four verified issues. All are real; all four were confirmed
against the source before acting.

**1 (blocker) — MRT2–7 did not survive a render group.** Slots 2+ were transient images created per
backend call with `LOAD_OP_CLEAR` and `initialLayout = UNDEFINED`, and only slots 0 and 1 had a seed
or persistent path. GTA V's G-buffer accumulates across roughly sixteen pass groups per frame, so
each group erased its predecessor's slots 2–4 and only the last survived. The MRT widening made the
recompiler able to emit five outputs while the runtime kept the attachment incomplete.

Fixed by giving slots 2–7 the same persistent-target contract slots 0 and 1 have, with the same
over-budget fallback to a transient image. Measured on the same route, best non-black pixel count per
slot: **c2 240,121 → 4,513,318** and **c3 407,957 → 4,513,318** (54% of a 4K frame), c4 313,908 →
513,338, with 0 validation errors.

**2 (blocker) — layered DS capture identity omitted the slice.** Once two cube faces were valid,
`snapshot_persistent_ds_images` emitted two seeds that were identical apart from the slice; the writer
rejected the second as `duplicate DS seed identity`, so F9 capture failed outright, and restore keyed
every seed at slice 0 so even a written capture could not replay the faces distinctly. Fixed with a
**v55** capture tail carrying each seed's slice (pre-v55 reads back as slice 0).

**3 (high) — `PROSPER_CAPTURE_MAX_SUBMITS` tested the cap after capturing.** Every post-cap submit
paid the full capture cost, and a post-cap capture *failure* set `b.failed` and discarded the
already-valid capped bundle — the exact outcome the cap exists to prevent. The cap is now tested
before any capture work.

**4 (medium/high) — the empty-window hook counters were process totals.** They accumulate from
startup and were printed verbatim as evidence about one capture window, so ordinary activity before
F9 made a genuinely empty window look as though the hook had been reached during it. Now baselined
when the window opens, with a saturating delta so a missed baseline prints 0 rather than a wrapped
count near 2^64.

### Two lessons from the review worth more than the fixes

- **A tail must go at the END of the stream, not beside the data it describes.** The v55 slice was
  first written next to the DS seed records, which round-trips perfectly and desynchronises every
  later tail the moment the version byte is downgraded — the exact thing eleven legacy-reopen
  fixtures do. The capture format's append-only convention is not a style preference; it is what
  makes `serialize at current version → lower the version byte → reparse` work at all.
- **A test that builds a struct by hand does not cover the code that builds it.** Codex's sharpest
  point: the first slice regression constructed `PersistentDsKey` directly, so reverting the
  production decode left it green. The seam is now extracted as `persistent_ds_key_for()` and the
  test calls it — verified by mutation: replacing the decode with `0u` turns both assertions red,
  restoring it turns them green.
- **And the one I got wrong on my own: `ctest` after `cmake --build --target screenshot` tests STALE
  BINARIES.** The v55 change broke 23 assertions in `test_gpu_capture` while I reported 245/245,
  because that target had not been rebuilt. Build everything before quoting a suite result.

## Cube depth: the DS cache overwrote its own faces (2026-08-15, fixed) — and the falsification above is RETRACTED

**Retraction first.** The section below this one records cube shadow maps as falsified. That call was
premature and is withdrawn. It read the *final screenshot*, and Codex's correction is right: the
screenshot is the wrong readout for an input this deep in the frame. Re-run cube-only at both poles,
reading the lighting output `0x20431c0000` instead:

| cube fill | `0x20431c0000` |
| --- | --- |
| `0x00` | max=46, 44.5% non-black (matches baseline) |
| `0xff` | **max=0, 0.0%** |

The shadow term does not merely move the lighting output, it can zero it completely. The lever moves
hard, so the cube path is load-bearing and the earlier "not the missing world" verdict was void, not
negative. Recorded here because it is the second time this session that a black *screenshot* hid a
large change one stage upstream.

**The defect Codex identified underneath the bridge gate.** A cube shadow map is not six neighbouring
allocations — it is ONE six-layer allocation whose faces share a depth base and are selected by
`DB_DEPTH_VIEW.SLICE_START/MAX`. GTA V programs `0x02000000, 0x02002001, 0x02004002 … 0x0200a005`
(start=max=0..5) against a single base. `PersistentDsKey` omitted the slice, so **every face render
for a base collided on one key and overwrote the previous face's host image**; only the last face
survived. No relaxation of the sampling gate could have recovered a valid cube from that cache.

Fixed by adding the slice to the DS identity. Measured on the same route: retained DS surfaces
**29 → 55**, and a cube base that held one entry now holds one per face:

```
0x20972c0000 slice=0 … slice=1 … slice=2 … slice=3 … slice=4
0x20978c0000 slice=0 … slice=1 … slice=2 … slice=3 … slice=4
```

Non-layered surfaces program `DB_DEPTH_VIEW=0` and key at slice 0 exactly as before, so identity
widens only where the guest actually used layers.

**Still open (Codex's stage 2):** the cube T# is still gated out of the bridge by `img_dim == 1`, and
binding it needs the six slices gathered into the backend's stacked sampled representation plus a
**numeric depth conversion** — the guest surface is `Z16` (`DB_Z_INFO.FORMAT=1`) while prosper
canonicalises host attachments to `D32_SFLOAT`, so the bridge must do
`D32 float → clamp [0,1] → quantise to R16_UNORM`, not a bit reinterpretation and not a Z inversion
(reversed-Z is already carried by the stored values and the comparison direction).

## ROOT CAUSE: the fragment recompiler supported only MRT0 and MRT1 (2026-08-15, fixed)

GTA V's G-buffer pass exports **five** render targets. prosper's fragment shell carried **two**, so
three of the five attachments were silently dropped from every G-buffer pass; the deferred lighting
pass then sampled buffers nothing had written, and the world rendered into a G-buffer that was lit by
nothing.

The pass, from `PROSPER_MRT_SHAPE_FOR`:

```
x1754  tmask=0x000fffff smask=0x0003ffff
       c0=0x207de60000 c1=0x207fe40000 c2=0x2083e00000 c3=0x2081e20000 c4=0x2085de0000
```

`tmask & smask = 0x0003ffff` — slots 0–3 at `0xf`, slot 4 at `0x3`. All five enabled by the guest.

**Everything below the shader already carried eight.** The render state decodes all eight slots,
`active_color_count` scans all eight, and the backend's render pass, framebuffer and blend state are
generic over `color_count`. Four hard-coded `2`s in the fragment path were the whole limit:

| site | was |
| --- | --- |
| `rdna2_to_spirv.cpp` fragment shell `v_color` | `std::array<uint32_t, 2>` |
| `fragment_color_export_mask` `realized` | `std::array<bool, 2>` |
| fragment colour-mask scan | `if (in.exp_target < 2)` |
| emit-side `exported` | `std::array<bool, 2>` |
| the "did anything export" guard | `!exported[0] && !exported[1]` |

The consequence was not confined to the shader. `gpu_execute.hpp` gates every slot's Vulkan write
mask by the shader's EXP.EN — `write_mask &= (exp_mask >> slot*4) & 0xf` — so a decoder that can
never set bits above nibble 1 forces **`write_mask = 0` for slots 2–7 regardless of what
CB_TARGET_MASK and CB_SHADER_MASK say**. Slots 0 and 1 survived only because `active_color` has
named-field fallbacks for exactly those two.

Measured, same route, before → after:

```
c2 ACTIVE=0 -> 2151      per-slot write_mask disagreements with the draw's own
c3 ACTIVE=0 -> 2083      mask registers: 5,318 -> 0
c4 ACTIVE=0 -> 1040
```

`ctest --no-tests=error -j4`: **245/245 pass**, exit 0.

**The world is still not lit after this fix.** It is a verified, necessary defect fix — the G-buffer
now reaches the lighting pass complete — and it is not by itself sufficient. Recorded that way rather
than as a solution.

### The instrument that found it, and the two that could not

The disagreement census is the one that mattered: for every draw, the per-slot `write_mask` the
DrawItem *carries* against the one its own `cb_target_mask & cb_shader_mask` *imply*.

```
c2 implied=0xf carried=0x0  x1830
c3 implied=0xf carried=0x0  x2109
c4 implied=0x3 carried=0x0  x1676
```

Neither of the two censuses before it could have found this. A per-slot **activation** census says
slots 2–4 never activate but not why, and its `mask=0` column is equally consistent with "the guest
disabled them" — which is what the mask histogram appeared to confirm, because that histogram was
keyed on groups where slot 4 had a *base* and was dominated by unrelated passes whose masks really
are slot-0-only. The defect only became visible when the same quantity was computed **two ways from
the same draw** and compared.

## Correction: a colour-target census must intersect the WRITE MASK (2026-08-15)

`PROSPER_TARGET_WATCH` and `PROSPER_DRAW_CENSUS` read `CB_COLORn_BASE` for all eight slots and count
a draw as writing that address. **That over-counts, and the earlier figures published from it are
wrong.** A base register is sticky: the guest leaves `CB_COLOR4_BASE` programmed long after it stops
rendering to slot 4, and hardware writes a slot only where `CB_TARGET_MASK & CB_SHADER_MASK` has a
non-zero nibble for it.

Measured (`PROSPER_MRT_CENSUS`, 16,384 pass groups):

```
c0 base=16384 format=16384 mask=15590 ACTIVE=15590
c1 base=11796 format=16384 mask= 2729 ACTIVE= 2729
c2 base=11796 format=16384 mask=    0 ACTIVE=    0
c3 base=11796 format=16384 mask=    0 ACTIVE=    0
c4 base=11796 format=16384 mask=    0 ACTIVE=    0
```

and over the groups where slot 4 *has* a base, the dominant state is
`cb_target_mask=0x0000000f cb_shader_mask=0x0000000f` (x11,647) — **slot 0 only**. Both registers are
present and genuinely narrow; neither is being truncated by prosper, and both default to `0xffffffff`
when absent, so a zero nibble is real guest state.

**So "0x2085de0000 is written by 130,290 of 131,072 draws, 96% of them in slot 4" is retracted.**
Those were stale bindings. The renderer is right to drop them, MRT slots 2–7 are not a defect here,
and the eight-slot census fix — which was itself a correct fix to a slot-0-only census — bought a
number that still needed the mask to mean anything.

The narrower lesson, which cost two rounds: **an eight-slot census with no mask intersection is not
"more complete" than a one-slot census, it is differently wrong** — the first under-reports, the
second over-reports, and only the second looks like progress.

## Where the picture actually goes (2026-08-15) — and three metrics that lied about it

`tools/gpu_timeline/rtt_pass_graph.py` (new) reassembles a frame's render-pass graph from a
`PROSPER_RTTLOG=1` log: every `[rtt] sample tex` line belongs to a draw of the *next* `[rtt] pass`
line, so a pass's inputs are the samples since the previous pass, and `SCANOUT` delimits frames.
Neither a draw census nor a target census can find a lost picture, because both count *work* and the
defect is in the *dataflow*.

What it shows for GTA V's gameplay frame — the deferred lighting pass:

```
0x20431c0000  3840x2160  draws=19   rgb_nonblack=0
   <- 0x207de60000  3840x2160 un8   x19          G-buffer, full
   <- 0x207fe40000  3840x2160 un8   x19          G-buffer, full
   <- 0x2083e00000  3840x2160 un8   x23          G-buffer, full
   <- 0x208f340000   256x256  un16  x1    MISS -- guest bytes
   <- 0x20954c0000   512x512  un16  x1    MISS -- guest bytes      six shadow maps,
   <- 0x2093cc0000   512x512  un16  x1    MISS -- guest bytes      none resolved from
   <- 0x2094ec0000   512x512  un16  x1    MISS -- guest bytes      a renderer-owned image
   <- 0x20948c0000   512x512  un16  x1    MISS -- guest bytes
   <- 0x20942c0000   512x512  un16  x1    MISS -- guest bytes
```

Those six addresses also top the DS-invalidation histogram (`20948c0000` x1010, `2094ec0000` x860,
`2093cc0000` x410 in one run), so they are surfaces prosper retains and then invalidates on a guest
DMA. `PROSPER_DS_GUEST_WRITE_INVALIDATE=0` does move that lever cleanly — `depth-invalid=0
stencil-invalid=0 extent=0`, against 1089/858/1628 with it on — but the route diverged in that run and
never reached the tutorial, so **it is not yet an A/B and is not claimed as one.**

### Three metrics that each read as "the world is black" when it is not

Recorded because all three were acted on in this session before being caught:

1. **`rgb_nonblack` is computed from an RGBA8 conversion, so an HDR f16 target whose values are all
   below 1/255 reports exactly ZERO while carrying a complete scene.** `0x20431c0000` reports
   `rgb_nonblack=0` while the very next pass extracts **47.5% non-black** from it. Reading the metric
   as "black" is a false negative across the entire lighting stage.
2. **`[rtt] sample tex … -> miss` spoke only for the COLOUR cache.** A 4K depth buffer the depth
   bridge had resolved correctly and a texture decoded from guest zeros printed the same word, so
   counting a pass's missing inputs off this log over-counted them. It now prints `DS-DEPTH` /
   `DS-STENCIL` for a bridged binding.
3. **A pass whose readback was deferred has EMPTY CPU pixels, and both `rgb_nonblack` and the new
   dump then report on nothing.** Measured: `0x207de60000` produced `src=0B` on **1,938 of 1,938**
   passes in a default run. The same target reports `rgb_nonblack=8267460` (99.7%) in a run with
   `PROSPER_RTTLOG=1` — because **the instrument's own readback is what materialises the pixels it
   then measures.** So `PROSPER_RTTLOG` changes the subject, and any per-pass content number is a
   statement about a run that was made differently from a default one.

`PROSPER_DUMP_PASS=0xADDR[,…]` (new, with `PROSPER_DUMP_PASS_EVERY`) writes what a pass actually
produced as an image, and reports `src=…B` plus `NO CPU PIXELS (readback deferred)` rather than
writing nothing — which is how defect 3 was found at all.

## The stencil plane of a rendered DS surface was never bridged (2026-08-15, fixed)

`fs=0x205b34be00` samples two planes of **one** depth/stencil surface, and prosper retained only one
of them:

```
[dsbridge] DS cache: dr=0x2052ac0000 dw=0x2052ac0000  sr=0x2054aa0000 sw=0x2054aa0000
                     htile=0x2055310000  3840x2160 fmt=130 (D32_SFLOAT_S8_UINT) dvalid=1 svalid=1
```

`0x2052ac0000` is the depth plane and bridges correctly (#1275). `0x2054aa0000` is the **stencil
plane of the same image**: `find_persistent_ds_sampled` matched only the depth bases (`dr`/`dw`), so
every stencil sample fell through to a guest-byte decode of memory the renderer never writes — zeros.
That is #1275's exact failure mode, left unfixed on the other plane. The guest declares the two planes
`Float32` and `Uint8`, which is what a deferred renderer's material and light-volume classification
reads.

Fixed by matching `sr`/`sw` and binding the image's stencil aspect. Measured over a routed boot:
`calls=262144 hit(depth=42897 stencil=2982)` — roughly 3,000 binds per run now sample the live image
instead of zeros, with 0 validation errors.

**This did NOT change the rendered frame, and the claim is kept separate from the fix.** A/B on the
same route, HUD regions excluded from the metric: before 2.004% of pixels non-black, after 1.993% —
inside the run-to-run spread of the route itself (0.283%–2.584% across five runs). The two frames are
visually identical: the same three bloomed light sources and the same faint structure. So the stencil
bridge is a verified gap fix, not the missing world.

**The instrument that nearly sold it as one.** The bridge's logging printed the first eight hits and
the first eight misses. On a routed boot all sixteen are consumed during startup by one plane, so it
could report neither a rate nor a reason, and the "after" image *looks* like it has new content until
it is put next to the "before" image. Replaced with per-plane counters classifying every miss as
no-entry / depth-invalid / stencil-invalid / extent / uninitialized — the distinction between "we have
no such surface" and "we have it and declined it" calls for opposite fixes and was previously
invisible. The old cache dump also printed with the first miss, which on this route is always before
the cache has any contents, so it read as "prosper retains no DS images" when it meant "not yet".

### Ruled out by the same measurement

- **The DCC "unsupported sampled image" warning on `0x2052ac0000` is a false alarm.** It fires from a
  block that checks `!rtt_hit` but not `has_ds_live`, so it prints even when the depth bridge won the
  binding; `meta=0x2055310000` is the surface's **HTILE**, being read as DCC metadata, and
  `metadata=0/0` means the copy was correctly skipped because the bridge had already won. It is not
  evidence of an unsupported path.
- **`0x2052ac0000` is not an untracked secondary MRT.** It is a depth buffer, which is why it never
  appears in a colour-target census, and it bridges: `[dsbridge] HIT … plane=depth`.

## The two inputs nothing writes (2026-08-15, measured)

The gameplay chain is all `0x20…` and every link hits (Codex, #2542), so the world is already absent
at the output of `fs=0x205b34be00`. That shader takes four inputs. Two of them are surfaces **no draw
in the run ever renders into**:

| input | RTT cache | draws that write it (exact, 131,072-draw run) |
| --- | --- | --- |
| `0x2085de0000` | HIT | **130,290** — slot1=8,262, slot4=122,028 |
| `0x2063380000` | HIT | **1** |
| `0x2052ac0000` (3840x2160 fmt=1) | MISS | **0 — never a colour target** |
| `0x2054aa0000` (fmt=11) | MISS | **0 — never a colour target** |

Measured with `PROSPER_TARGET_WATCH=<addr>,…`, which is exact and unsampled over all eight MRT slots.

**Why the zeros are believable here, when a clean zero usually is not.** Two independent properties of
the same run establish that the instrument could have expressed the result it did not find:

- *The lever moves*: `0x2085de0000` reports 130,290 of 131,072 draws across two slots.
- *The domain is expressible*: `0x2063380000` reports **exactly one** draw. That is the case a sampled
  census structurally cannot represent, and it is the reason this measurement exists — see below.

`0x2052ac0000` is also guest-readable and its first 8,252 bytes stay **all-zero for the whole run**
(`[compute-tree-watch] … pairs=0 unpaired=0`, observed once and never changing). That is expected of any
render target and is *not* independent evidence, since guest memory does not see GPU writes; it is
recorded only so the next reader does not spend a run re-deriving it.

### Two instrument defects this found, both of which had already produced a wrong answer

1. **The draw census read only MRT slot 0.** A deferred renderer writes a G-buffer across slots 0–7 in
   one draw, so a slot-0-only census cannot see most of what a frame renders into — and "is address X
   ever a render target" was exactly the question being put to it. Its own control exposed it: the two
   surfaces that demonstrably HIT the RTT cache never appeared in the census at all. Fixed to census all
   eight slots. `0x2085de0000`'s traffic is 96% in **slot 4**, so the slot-0 census was blind to it.
2. **The census samples 1 in 32 draws, and a sampled zero cannot answer an "ever" question.** A surface
   written by ten draws is missed with probability (31/32)^10 ≈ 73%; by one draw, ≈ 97%. The sampling was
   itself a correct earlier fix — the unsampled first version took a mutex and a map insert on every one
   of 131,072 draws and stalled the routed run at 1,024 — but it silently changed what the instrument
   could conclude. `PROSPER_TARGET_WATCH` is the exact form: bounded watch list, per-address atomics, no
   lock, so it costs eight register lookups per draw and keeps the "ever" question answerable.
   **`0x2063380000` is written by exactly one draw**, so this distinction is not hypothetical — the
   sampled census would have reported it as never written, alongside the two that genuinely are.

An unreadable tree-watch window also used to produce **no output at all**, which is indistinguishable
from "nothing writes this address" — the conclusion it would have supported. It now says so.

## Ruled out / retracted here

- **"Sibling pairs are `(odd, even)`."** Retracted. The clean table has 204 parents whose children
  are `(2j+1, 2j+2)` and **828** whose children are `(2j, 2j+1)` — pairs are adjacent at arbitrary
  parity. The "first broken pair k=205" figure was computed under the fixed-alignment assumption and
  its alignment part goes with it. The adjacency-based unpaired and cycle counts never assumed
  alignment and stand.
- **"Slot 300's bits[2:0] are an anomaly."** Retracted. `post-36-3.bin` has bits[2:0] == 0 in all
  2,063 records while `live-a240.bin` carries a mix of 0 and 2, so that field varies legitimately
  between clean tables.
- **The LDS-undersizing hypothesis.** Dead: the module declares a Workgroup array of exactly 384
  dwords = 1,536 bytes = 3 × 512-byte `COMPUTE_PGM_RSRC2.LDS_SIZE` granules, so the real allocation
  is plumbed through and matched, not defaulted.
- **Synchronization / race hypotheses for this program.** The broken-pair patterns from two
  different frames share a 60-character suffix exactly, and `min` damage index is identical across
  frames: the failure is deterministic given the input.
- **A lane, wave or workgroup boundary effect.** `k mod 2/3/4/8/16` are all flat.
- **`PROSPER_NO_NATIVE_COMPUTE_SUBGROUP=1` as an A/B arm.** Void, not negative: it skips 15 *more*
  programs including `0x413d88400`, so `0x413dc3400` never dispatches and the phase never runs.

## What the structure IS: an LBVH / binary-radix-tree parent table from Morton keys

Identified by Codex from the retained ISA (#2542), and it reframes everything below.

- **2,063 = 2 × 1,032 − 1** — exactly the node count of a full binary tree with 1,032 leaves and
  1,031 internal nodes.
- **Adjacent pairs are SIBLINGS** sharing one parent; bit 30 identifies the child side. The root is
  the single unpaired node. That explains the count, the pairing, the parent chase and the depth
  parity together, where "union-find with path compression" only explained the chase.
- `0x413cee500` contains the standard 3D Morton bit-dilation constants `0x030000ff`, `0x0300f00f`,
  `0x030c30c3` and **`0x09249249`**, then stores `buffer_store_dwordx3` at pc274 — Morton key
  generation.
- `0x413e1c300` loads two three-dword records, runs a 64-lane LDS compare/exchange loop, and stores
  two three-dword records — a Morton sort/merge pass upstream of topology construction.

**CORRECTION — `0x09249249` is NOT an empty-slot sentinel.** This document said it was. It is the
Morton dilation mask, used as a literal by the key generator. In the captured parent view it happens
to behave as an out-of-range terminator, but that may be an intentional empty value, a Morton key
left in repurposed scratch, or another phase's encoding. Call it an **observed OOB/unused value**
until its producing store is identified.

**Naming:** the "malformed head/tail pair" score is measuring a real property, but the neutral name
is **unpaired sibling records**. Keep it as a quantitative correlation and do NOT promote "must be
zero" to a correctness rule: clean samples tolerate up to 13, and the slot-level causal test failed.
A sharper check is whether the active-node count predicts exactly 1,031 sibling pairs.

## Access direction: eight touchers, FIVE may-writers

Codex joined the census's `fetch-pc` values to the retained guest instructions. MUBUF `0x0c..0x0e`
are loads, `0x1c..0x1e` are stores:

| program | watched instructions | direction |
| --- | --- | --- |
| `0x413ce3400` | pc35, 66, 68 … 421, all `buffer_load_dword{,x2}` | **read only** |
| `0x413ce6000` | pc36 `buffer_load_dword` | **read only** |
| `0x413cea300` | terminator-only, `fetch-pc=0xffffffff` | **no data access** |
| `0x413cee500` | pc274 `buffer_store_dwordx3` | **write** |
| `0x413d88400` | 18 watched pcs, all stores | **write** |
| `0x413dc3400` | pc597/608/620/632/644/656 stores | **write** |
| `0x413dc6700` | pc91 load; pc53/65 and 618..677 stores | **read/write** |
| `0x413e1c300` | pc86/95 loads x3; pc166/176 stores x3 | **read/write** |

So the may-write set is **`0x413cee500`, `0x413d88400`, `0x413dc3400`, `0x413dc6700`,
`0x413e1c300`**. The two read-only programs and the terminator leave the writer investigation.

**A statically writable descriptor whose range contains the address is a CANDIDATE writer**, not
proof that a given invocation wrote that 4-byte slot; changed-byte pre/post evidence closes that.

**Address-boundary caveat that must not be lost:** `0x413e1c300`'s observed view is
`base=0x20f8482140 size=33024`, and `base + size == 0x20f848a240` **exactly**. It therefore may write
the first table and proves nothing about the second. The census must be re-run for `0x20f848a240`
before this eight-program set is carried across both ping-pong tables.

## No guest-side escape makes a reachable cycle safe

Checked by Codex against the retained ISA: the pre-loop bound only excludes padded threads (on the
problematic shape `s18 = 2063` and the launch has 2,063 guest threads, so every active root is
represented); the loop has no iteration count and no cycle detector; `v_cmpx_ne_u32` only retires
lanes reaching index zero, so a lane inside a cycle stays active forever; and the `s24` comparison
happens **after** the walk, so it cannot guard entry.

A builder may hold transiently inconsistent parent links while constructing the hierarchy. Hardware
still cannot launch this consumer on reachable cyclic links — they must be repaired before it,
excluded from its active root set, or absent from the bytes it consumes.

## The current account — read this before anything below it

This document is layered: it grew as an investigation log, and several sections below are historical
transcripts kept for the evidence beside them. **Where a lower section disagrees with this one, this
one is current.** Each layer that was superseded now says so where it sits.

As of 2026-08-14, established and each measured rather than inferred:

1. **The missing world is one compute program.** `0x413dc6700` hangs the GPU into a RADV hard
   recovery. That disables live compute for the whole process, so every later indirect draw is
   dropped — and GTA V's world is GPU-driven, so those indirect operations *are* the world. Skipping
   it (`PROSPER_COMPUTE_SKIP_PROGRAM`) gives 0 device losses and the first real scene content.
2. **The hang is a non-terminating loop in that program.** Its 903-dword body contains exactly one
   backward branch (guest pc97 → pc88). The trip-bound witness fires there and reports that no
   invocation ever reached a dispatch ordinal past the loop body; the fence-wait duration
   for that dispatch is ~2,045 ms against sub-millisecond for every other dispatch in the route; the
   device loss follows on the next dispatch.
3. **The loop's data is cyclic at dispatch time.** Pre-dispatch, 806 of 1,782 reads receive a table
   in which 1,805–2,062 of 2,063 roots lead into a cycle. The guest loop cannot terminate on that.
4. **Our lowering of that loop shape is correct.** A hand-built kernel with the same
   `v_cmpx` / `s_cbranch_execz` / back-edge shape runs correctly on real Vulkan
   (`tests/shared/diagnostics/test_cfg_trip_bound.cpp`), so the recompiler is not what fails to exit.

**So the open question is why the table is cyclic** — not whether the loop spins, and not whether we
lower it correctly.

5. **Eight programs TOUCH the table; the writer set is UNKNOWN.** A containment census over a full
   route names `0x413ce3400`, `0x413ce6000`, `0x413cea300`, `0x413cee500`, `0x413d88400`,
   `0x413dc3400`, `0x413dc6700`, `0x413e1c300`. **"Touch" is a resource binding, not an access
   direction** — the census cannot separate a reader from a writer, so it identifies no writer at
   all. An earlier revision of this list said "at least two programs write the table"; that was a
   matcher artifact (base equality, blind to a view whose base differs) and is withdrawn.
6. **One program's write quality tracks the damage — a correlation, not an identification.**
   `0x413dc3400`'s malformed-pair count separates 54 clean reads (0..13) from 126 cyclic ones
   (19..106) with no overlap. Its slot-level causal test **failed**: 14% of cycle nodes sit on
   malformed slots against a 4.4% base rate. It is one candidate among eight.
7. **`0x413dc3400`'s store path is lane-predicated.** All six of its table stores sit inside an
   `s_and_saveexec_b64` / `s_cbranch_execz` region, so *which slots are written* is decided by a
   per-lane mask — and the defect's character is membership, not arithmetic: every record is
   individually well-formed, in the wrong combination.

**The gating question is which of the eight programs WRITE the table.** The census measures a
resource binding, not an access direction, and every downstream narrowing depends on that
distinction — including whether `0x413dc3400` is a writer at all. Only once the writer set is known
does "why do its writes go bad" become answerable, and even then what exists today is a
correlation on an identical module (same SPIR-V hash, launch and 38 buffers either side of the
transition), not established cause.

**Not established, and explicitly tested:** that each malformed pair becomes a 2-cycle. On a same-run
join only 14% of cycle nodes sit on malformed slots against a 4.4% base rate — real enrichment, not a
mechanism. The dispatch-level correlation stands; the slot-level one does not.

**Superseded:** a paragraph here once named `0x413dc6700` as *the* writer and called it a
self-corrupting kernel. Flips also occur in submits where it writes nothing, so that identification
is dead — and the replacement is not "a second writer" either: **eight programs bind ranges covering
this address and the census cannot say which of them write.**

`0x413ce3400` is **back in scope.** It was marked superseded here on the grounds that it wrote only
"related state"; the containment census lists it among the eight. What remains true of it is
narrower: it is never declined on a routed run, so the "producer was refused" hypothesis is dead.

**What is NOT established**, stated precisely because the wording above is easy to over-read:

- **That `0x413dc3400` causes the cycles.** What is measured is a *dispatch-level correlation*: the
  malformed-pair count of its writes separates 54 clean reads (0..13) from 126 cyclic ones (19..106)
  with no overlap. The slot-level test **failed** — only 14% of cycle nodes sit on malformed slots
  against a 4.4% base rate — so a shared upstream cause is not excluded, and no A/B or
  clean-before/bad-after has been run.
- **That `0x413dc3400` writes this table at all.** The census re-run is complete and reports eight
  programs whose resources *contain* the address; a resource binding is not an access direction.
  Nothing here is a direction-qualified observation, so "its writes go bad" is shorthand for a
  correlation between its dispatches and the damage, not an established write.
- **Which program or store introduces a cycle**, and whether the guest algorithm is behaving
  correctly on inputs produced wrongly upstream.

**The next experiment is direction-qualified attribution**: establish, per program, whether it reads
or writes this range. Every further narrowing depends on it.

## RETRACTED 2026-08-14: "the corrupting write is `0x413dc6700`'s own" was one sample

The self-corrupting account was previously inferred from a writeback trace. It is now a direct
before/after on adjacent dispatches of the same program, in the same submit, on current master:

```text
disp 38  read  0x20f848417c  CLEAN   cycles=0  cyclic-roots=0     oob-roots=1385  max-depth=21
disp 38  execute ok, buffers=43, spirv=61143/177420afa4fd9c50
disp 38  writeback binding=4 addr=0x20f848a240 changed=2357
disp 39  read  0x20f848a240  CYCLIC  cycles=6  cyclic-roots=2062  oob-roots=0     max-depth=32
```

**RETRACTION.** The transition above is real and reproducible, and it does NOT establish cause. Two
observations from a wider census of the same instrumentation kill the inference:

- **A complete rewrite left the table CLEAN.** Submit 4312 dispatch 764 wrote 2,061 of 2,063 slots,
  and every read in that submit reports `cycles=0`. If this program's write corrupted the table, the
  most complete write in the route was the best chance to show it.
- **Tables flip to cyclic with NO write from this program.** Submit 4725 reads `0x20f848417c` with
  986 cyclic roots, and `0x413dc6700` writes nothing to it in that submit at all.

So a flip adjacent to its write is not evidence that the write caused it, because flips also happen
without one. What the original evidence established is that the program **writes the table it later
reads** — which was already known — plus one coincidence.

The generalisable error: a transition was observed next to a candidate cause, and adjacency was
treated as causation without checking whether the transition also occurs WITHOUT the cause. The
negative case is the whole test. Reproduce with `PROSPER_COMPUTELOG_CODE=0x413dc6700` plus
`PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64` and read the interleaved
`parent-walk` / `execute` / `writeback` lines.

### EIGHT programs touch the traversal table — the "two writers" census was wrong

The first census matched `resource.gpu_addr == wanted` over top-level resources. Re-run with a
containment matcher (`wanted >= base && wanted - base < size`, over scalars *and* `table_entries`),
the same route reports **eight** programs, 3,931 hits:

```text
0x413ce3400   0x413ce6000   0x413cea300   0x413cee500
0x413d88400   0x413dc3400   0x413dc6700   0x413e1c300
```

`0x413dc6700` holds nine bindings including the loop's read at fetch pc 91; `0x413e1c300` is the most
frequent of all (four bindings, 308 hits each); `0x413cea300` appears with `fetch-pc=0xffffffff` —
that is the terminator-only program, whose whole body is one `s_endpgm`.

**Zero matches came from `table_entries`**, so on this allocation the array form is not exercised and
the entire gain is base-equality → containment: six programs bind a view whose *base differs* while
its range covers the address.

Consequences, and they are large:

- **"The table has two writers" is withdrawn.** It was an artifact of a matcher that could only see
  an exact base.
- **`0x413ce3400` is back in the picture.** This document marked it superseded on the grounds that it
  was "a writer of related state"; it binds a range containing this table.
- **The `0x413dc3400` correlation is unaffected but much less pointed.** Its malformed-pair count
  still separates 54 clean reads from 126 cyclic ones with no overlap — that measurement stands —
  but with seven other programs touching the allocation, "its writes are what go bad" is one
  candidate among several rather than a narrowing to one.
- **"Touches" is not "writes".** The census lists resources, not access direction. Distinguishing
  readers from writers on this allocation is the next thing to measure, and it is now the gating
  question rather than a detail.

### `0x413dc3400`'s write quality separates clean tables from cyclic ones — correlation, not cause

Tracing `0x413dc3400` — one of the eight touchers, and not established as a writer — with
`PROSPER_COMPUTELOG_CODE=0x413dc3400` plus `PROSPER_COMPUTELOG_CHANGED`
(its table is **binding 23** — binding indices are per-program, and reusing the consumer's numbers
here produced a confident wrong answer first), and scoring each write by how many heads it lands
without a matching tail in the next slot:

| reads | n | prior write's malformed pairs |
| --- | --- | --- |
| `cycles=0` | 54 | **0..13** |
| `cycles>0` | 126 | **19..106** |

**No overlap across 180 reads.** The writes are clean (`MISMATCHED=0`) through submit 7480 and then
jump to 77, 66, 67, … 104, 106, and the tables go cyclic exactly when they do.

That makes `0x413dc3400`'s write quality a **quantitative oracle** — not an identification of the
writer, since the census that produced this framing measured bindings rather than access direction
and eight programs touch the allocation. The malformed-pair count: a fix must drive it to zero, and `cyclic-roots` should follow.

**Stated as correlation, because that is what it is.** 180 reads with clean separation and a mechanism
that explains the shape (a head whose tail is absent leaves an orphan tail from an older generation,
and an orphan tail pointing back at its predecessor is exactly the observed 2-cycle) is strong, but no
A/B has yet shown the cycles following the mismatches. The counter-example that would break it is a
write with a high malformed count followed by a clean read; none occurred in 180 reads.

One earlier inference is already dead by this data: **13 malformed pairs at submit 4162 produced no
cycles at all**, so "any malformed pair corrupts the table" is false. There is a threshold between 13
and 19, which is itself a clue — the structure tolerates some inconsistency.

### The shader does not change across the boundary — only its input does

`0x413dc3400` compiles to the **same module** on both sides of the transition:

```text
submit 7480  groups=9x1x1  buffers=38  spirv=57537/8dbb56b7a4feea9c   MISMATCHED=0
submit 7898  groups=9x1x1  buffers=38  spirv=57537/8dbb56b7a4feea9c   MISMATCHED=77
```

Same SPIR-V hash, same launch, same 38 buffers. So the divergence is **data-dependent**, not a
recompilation difference — which rules out cache/key effects and points at how our execution handles
one particular input.

`0x09249249` is the guest's **empty-slot sentinel**, not a prosper fill (it appears nowhere in our
source). Decoded as a record it yields `next = 19,158,153`, far past the 2,063 records, so a walk
reaching it leaves the array and terminates by the RDNA2 out-of-range rule. Many of the slots the
first dirty write touches held it beforehand, i.e. they were previously empty.

**Falsified while forming it:** "clean tables terminate via those OOB sentinels, and cycles appear
once the table fills up." All 204 clean reads in this run have `oob-roots=0`; the cyclic ones range
0..1,690. The relationship runs the other way from the guess, so OOB termination is not what keeps a
clean table acyclic.

### Two patterns in the malformed set that do NOT generalise

Recorded so nobody re-derives them. Both looked convincing on one dispatch:

- **Stride 16.** The first twenty malformed indices at submit 7898 read `884,885 · 900,901 ·
  916,917 · 932,933` — adjacent pairs exactly 16 apart, which would point straight at a 16-lane
  grouping (DPP row, tile). Across the full set the `index mod 16` histogram is spread over ten
  residues, and it is a *different* spread on every dispatch. The regularity was in the first twenty
  entries, not in the population.
- **A distinguishing prior value.** At submit 7898 the malformed heads' previous contents include
  `0x24924924` twenty-six times while well-formed heads never show it — a clean discriminator, and
  `0x24924924` is `0x09249249 << 2`, the same repeating sentinel at another phase. It does not hold:
  submit 8309's malformed priors are `0` and `2`, submit 14727's are `0x3FFFFFFF` and `0`. No value
  is common to the malformed set across dispatches.

**The honest position after that:** the malformed set has no structural signature I have found, so the
correlation with cyclicity is currently the only handle on it, and pattern-hunting on derived metrics
has stopped paying.

### `0x413dc3400`'s table stores are LANE-PREDICATED

`0x413dc3400` disassembles to 882 dwords / 765 instructions, and its six table stores share one
shape — a store to the table and a store to a second buffer at the *same index*, both inside an
EXEC-predicated region:

```text
pc592  s_or_b64            s[106:107], s[16:17], s[14:15]
pc593  s_and_saveexec_b64  (EXEC &= that; old EXEC saved to s[106:107])
pc594  s_cbranch_execz     -> 599            ; skip the block if no lane qualifies
pc595  buffer_store_dword  v42, v58, s[8:11] ; a second buffer
pc597  buffer_store_dword  v48, v58, s[0:3]  ; THE TABLE, same index v58
pc599  s_mov_b64           exec, s[106:107]  ; restore
```

Six such blocks: (595,597), (606,608), (618,620), (630,632), (642,644), (654,656).

Two consequences worth having:

1. **Which slots get written is decided by a per-lane mask**, so a divergence in EXEC handling shows
   up as *missing or extra* stores rather than wrong values — which matches the observed defect
   (records that are individually well-formed, in the wrong combination) far better than an
   arithmetic error would.
2. The program is lowered through `emit_cfg_state_machine` (it is on the `role=terminal` list), where
   EXEC is emulated per invocation and `s_cbranch_execz` needs a cross-lane vote. That is the same
   machinery the consumer's loop exercised — and a hand-built kernel proved it correct for *that*
   shape (`tests/shared/diagnostics/test_cfg_trip_bound.cpp`), which does not extend to `s_and_saveexec_b64` feeding a
   predicated store.

This is a hypothesis, not a result: no measurement yet shows a lane storing when it should not, or
failing to. The next instrument would compare the set of slots written against the set the mask
selects.

### The test was run, and it does NOT support cause

Same-run capture (120 dumped cyclic tables and 98,837 changed-slot records, so the join is valid),
asking whether the 2-cycles sit at slots the malformed writes touched:

```text
write-submit=10308  malformed=91  cycle-nodes=178  on-malformed-slot=26  (14%)
write-submit=10708  malformed=93  cycle-nodes=178  on-malformed-slot=26  (14%)
```

Stable at **14%** across ten tables. With 91 malformed slots in 2,063 the base rate is 4.4%, so this
is roughly 3x enrichment — a real association, and far from the "cycles land on malformed slots"
that would close the chain. **86% of the cycle nodes are somewhere else.**

So the malformed-pair count remains a strong *dispatch-level* correlate (54 clean reads at 0..13
versus 126 cyclic at 19..106, no overlap) while failing as a *slot-level* mechanism. Two readings
survive, and this data does not choose between them:

- the metric is a proxy for some other property of a bad write, and that property produces the
  cycles; or
- malformed pairs and cycles share an upstream cause and neither produces the other.

**What it rules out:** "each malformed pair becomes a 2-cycle." That was the mechanism I expected
and it is wrong.

### Open: why does it start at submit 7898?





`0x413dc3400` writes 2,061 slots with zero malformed pairs for nine consecutive dispatches, then
never again. Whatever changes at that point is the proximate cause, and it is a much smaller question
than the one this investigation started with.

### FALSIFIED: our store INDICES are wrong

Grouped strictly per `(submit, dispatch)` — the ungrouped form of this analysis is meaningless and
produced a confident wrong answer first — every head a dispatch writes has its matching tail at the
very next slot:

```text
submit 4312 dispatch 764: 2061 slots written, 1031 heads, MISMATCHED=0
submit 5555 dispatch  38: 1721 slots written,  864 heads, MISMATCHED=0
```

So the pair-store index arithmetic is correct in our execution. Whatever produces an orphaned tail,
it is not this program emitting a head and a tail at non-adjacent slots.

### The corruption has a SHAPE: overlapping pair writes, not garbage

`PROSPER_COMPUTE_PARENT_WALK_DUMP` captured 138 cyclic tables. Every cycle in every one of them is a
**2-cycle**, and the records involved are structurally valid — correct tag, plausible index:

```text
rec[452] = 0x00000e32   tag=2  bit30=0  next=454     <== cycle
rec[453] = 0x40000e32   tag=2  bit30=1  next=454
rec[454] = 0x40000e22   tag=2  bit30=1  next=452     <== cycle
rec[455] = 0x00000e7a   tag=2  bit30=0  next=463
```

**The table is a sequence of PAIRS**: `rec[k+1] == rec[k] | 0x40000000`, same payload, bit 30 marking
the second element. 920 such pairs across 2,063 records, and the pairing is not parity-locked (523
begin at an even index, 397 at an odd one), so a pair is simply two consecutive slots.

A `bit30=1` record must therefore be immediately preceded by its `bit30=0` twin. In every cycle it is
not: above, slot 453 holds the *tail* of pair (452,453) while slot 454 holds a tail whose head is
gone. **Two writers claimed slot 453** — one writing the tail of (452,453), one the head of
(453,454) — and the survivor's orphaned partner at 454 points back at 452, closing the cycle.

So this is an **overlapping-allocation / lost-update** signature, and the earlier "61 two-cycles" note
was reading it correctly. What changes is that the falsification recorded against it was measured on
`0x413ce3400`'s instruction footprint, which is only one of eight programs binding this allocation.
Re-opened: it falsifies a lost-atomic hypothesis for one program, and no program is established as
the writer.

Two candidates for how two concurrent STORES land on overlapping slots (a different sense of
"two writers" from the program census above — this is about lanes racing within a dispatch),
neither yet tested:

- **The pair store itself.** The program contains 3 `buffer_store_dwordx2` and 5 `buffer_store_dwordx3`
  among its 23 stores, and a `dwordx2` writes exactly two consecutive dwords — a pair. An addressing
  error in the multi-dword store path (element versus byte, or an off-by-one base) shifts a pair by
  one slot and produces precisely this.
- **The slot allocator.** `ds_add_rtn_u32` at guest pc121 is a bump allocator: each thread atomically
  adds its size to an LDS counter and takes the old value as its base. Overlapping bases would do it.
  **Checked and currently NOT suspect:** the emitter lowers it to `OpAtomicIAdd` with
  `Scope_Workgroup` / `MemSem_WGAcqRel`, which is correct for `local=256` (four 64-wide waves), and
  the program has no global-memory atomics at all — 41 MUBUF ops, every one a plain load or store.

### What the same run also settles

- **Most dispatches write nothing.** 52 of 64 writebacks are `changed=0`; the substantial ones are
  `changed=2357`, `2061`, and a few single-digit updates. So the corrupting event is rare and
  identifiable, not a steady drift.
- **Both tables can be cyclic**, contrary to the earlier per-address reading: `0x20f848417c` is
  162 clean / 30 cyclic and `0x20f848a240` is 40 clean / 120 cyclic across 352 resolved reads. The
  asymmetry is real but it is not a property of the address.
- **The resource table varies per dispatch.** The same program address compiles to different modules
  (`spirv=58649/…` several distinct hashes, `61143/177420afa4fd9c50` on the big-write dispatch), and
  `buffers` ranges from 1 to 43. A single dispatch's resource picture is not the program's.

### The open question, now narrow

Why does that write produce cycles? Candidates, none yet tested:

1. the stored VALUES are wrong (our lowering of the store path computes the wrong record);
2. the stored INDICES are wrong (right values, wrong slots);
3. a concurrency effect at `threads=2063 local=256 groups=9` that the guest's algorithm tolerates on
   hardware and our lowering does not preserve.

The oracle is in place either way: `cyclic-roots` on the read immediately after the write is the
number to move, and dispatch 38 of a `0x413dc6700` pair is where to look.

## Where the world went

As of 2026-08-14 the black world is **one compute program**, and that is established by A/B rather
than inferred:

```
PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700
```

| arm | device losses | frame |
| --- | --- | --- |
| default | 1, at ~59 s — reproduced in every run | black + HUD |
| `0x413dc6700` skipped | **0** across a full 300 s route | sun, anamorphic lens flare, radar with street geometry and blips, first-person tutorial text |

`0x413dc6700` hangs the GPU into a RADV **hard recovery** at `queue-submit`. prosper then disables
live compute for the whole process, and every later dispatch and indirect draw in the frame is
refused — so the frame cannot recover even partially.

**Read that skipped-program frame carefully: it is not "the frame minus that dispatch".** A skip goes
through the same decline path as a real refusal, so it clears `producer_epoch_ok` and the next
`ParserStall` latches `indirect_dependencies_ok` for the rest of the submit. The frame is that frame
minus the dispatch *minus every indirect draw and dispatch after the next parser stall*. The zero
device losses and the appearance of real scene content are unaffected by this; what it costs is the
right to attribute the *remaining* blackness in that image to `0x413dc6700` alone.

What the program probably is, from what survives without it: 43 resources,
`threads=2063x1x1 local=256x1x1 groups=9x1x1`, 61,143 SPIR-V dwords, and sky/flare/radar render while
world geometry does not — the shape of a tiled deferred lighting or light-culling pass.

**The frontier is therefore exactly one sentence: `0x413dc6700` must execute without hanging.**

### Why this took so long to see

Until #2538 the live compute backend had **fifteen** `return false` paths in `execute_item()` that were
silent, `trace`-gated, or logged without naming the dispatch. The executor records such a refusal as
`RealizationFailureReason::Unknown`, and **only when a capture trace is active** — so on a default run
a refused dispatch left no record at all. (`ComputeExecutionDeclined`, which an earlier revision of
this file named, is a classification added later by #2536; it did not exist when the frame went
black.) A gameplay submit's 196 realization failures were 59 anonymous declines and 128 cascade
failures, and the cascade hid the cause:

1. a declined dispatch clears `producer_epoch_ok`
2. the next `ParserStall` latches `indirect_dependencies_ok` false for the **rest of the submit**
   (`gpu_executor.cpp:9609`)
3. every later indirect draw and indirect dispatch short-circuits to `IndirectDependencies` without
   being attempted (`:9354`, `:9422`)

GTA V's world is GPU-driven, so the indirect operations *are* the world. One refusal anywhere early in
a submit removes all of it. Several sessions went to the 72 **direct** draws that execute correctly at
3840x2160 with colour writes enabled — they were never the ones failing.

The first domino turned out to be an **empty guest kernel**: `0x413cea300`, whose entire body is one
`s_endpgm`, declared one raw buffer and was refused as `no-bindable-descriptor` 192+ times per run.
Fixed by proving emptiness from the raw stream (`rdna2_program_is_terminator_only`) and reporting such
a dispatch as a successful no-op. A/B: 192 declines to 0.

### The trip bound measures ONE emitter — a null from it is "not measured"

`PROSPER_CFG_TRIP_BOUND` caps the **CFG dispatcher's** back edge. It does not touch either structured
loop emitter, so a program the structurizer accepts is *structurally unmeasurable* by it and reports
nothing whether or not it runs away. `0x413dc6700` happens to be covered — it reaches `role=terminal`
in the structurizer and is lowered by `emit_cfg_state_machine` — which is why its witness fired at all.

This is recorded because the emitter's own comment asserted the opposite for a while: a correction
saying the structured emitters are NOT covered was added directly above a surviving sentence claiming
all three were. Both are now pinned by assertions — a structured loop must be byte-identical when
armed, a dispatcher loop must not be.

### Diagnostics this investigation added, and what each is safe to conclude from

| switch | default | what it does | the trap it avoids |
| --- | --- | --- | --- |
| `PROSPER_CFG_TRIP_BOUND` + `_PROGRAM` + **required** `_PHASE` | off | caps the CFG dispatcher's back edge for one program and phase, and records a device-side hit witness | covers **only** the CFG dispatcher, so a null on a structurizer-accepted program means *not measured*; without `_PHASE` nothing is emitted, because one record cannot describe two phases |
| `PROSPER_COMPUTELOG_RAW` | off | writes a traced program's guest RDNA2 bytes for `tools/shader_inspect` | writes ONE named program to an exact path. Since #3196 `PROSPER_SHADER_DUMP_SUCCESS` also carries the code address in its filenames and `PROSPER_SHADER_DUMP_PROGRAM=0xADDR` narrows it, so hash-matching by hand is no longer the alternative |
| `PROSPER_INDIRECT_APERTURE_RECOVERY` | **off** | rebuilds a base-less queue-2 indirect argument address from the last-seen SetBase aperture | changes execution: the aperture is learned from any SetBase on any queue, and *mapped* is not *this is the argument buffer* |
| `PROSPER_INDIRECTLOG` | off | per-packet base/offset/queue, the three argument dwords, and an end-of-run outcome census | readability was probed and values were not, so a misread surfaced only as a `workgroup-count-limit` decline thousands of operations later |

The aperture recovery is opt-in on purpose and the trade is recorded rather than implied: **off, 50 of
64 indirect compute dispatches skip as "unreadable arguments"; on, 0 do**, and the probe found the raw
low address unmapped and `aperture | low` mapped on 49 of 49. That is real evidence about where those
arguments live and it is still not provenance — one process VA space can hold mapped allocations under
several high-32 prefixes, so accepting an address because 12 bytes are readable admits dispatching
group counts read out of an unrelated live allocation.

## The frame's dataflow, end to end (2026-08-16)

Reassembled with `tools/gpu_timeline/rtt_pass_graph.py` from a `PROSPER_RTTLOG=1` routed boot. This
is the map to reason against; **it locates the break precisely, and the break is not where the draw
and target censuses pointed.**

```
G-buffer (5 MRTs, 11-28 draws)            albedo 0x207fe40000  100%   <- complete and correct
  0x207fe40000 0x2083e00000 0x2081e20000  normals 0x207de60000  99.7%
  0x2085de0000 0x207de60000
        |
        v
deferred lighting  -> 0x20431c0000  19 draws   rgb_nonblack = 0        <-- THE BREAK
        |              reads the whole G-buffer + 6 shadow surfaces
        v
        0x20471e0000  1 draw   45.1%   (reads 0x20431c0000 x14)
        ...later passes raise 0x20431c0000 to 45.3% (3,757,533 px)
                                                   |
                                                   |  NOTHING CARRIES IT ACROSS
                                                   x
final composite -> 0x2056740000  1 draw  13.5%
        <- 0x2063380000  3840x2160 f11f11f10  x24   <-- scene colour, WRITTEN BY NOTHING
        <- 0x2085de0000  x6, bloom pyramid 1920x1080 / 960x540 / 240x135
        v
   0x2058720000  13.5%   ->   SCANOUT 0x2162cd0000  13.5%   (this 13.5% is the HUD)
```

Two facts fix the frontier:

1. **The deferred-lighting pass consumes a complete G-buffer and emits nothing.** Its 19 draws read
   albedo at 100% and normals at 99.7% and produce `rgb_nonblack=0`. Later passes populate
   `0x20431c0000` to 45%, so the buffer is not broken — the lighting resolve specifically is.
2. **The composite samples `0x2063380000` 24 times per frame, and nothing on any decoded path writes
   it.** Across 6,561 frames it is bound as a render target **exactly once**, for one draw, with no
   inputs and no output (`px_nonzero=0`) — that is a clear, not a composite. No draw (0/131,072 over
   all 8 slots), no compute, no resolve, no guest-side GPU write, no DS plane, no dropped draw.
   Meanwhile the lit scene sits in `0x20431c0000` / `0x20471e0000`, which the composite never reads.

So the world is not lost in the composite and not lost at scanout: **the producer of the scene-colour
buffer is missing entirely.**

### Run the route for 400 s, not 200 s — a 200 s run is mostly LOADING (2026-08-16)

**This qualifies every per-frame statistic below, including several I wrote earlier the same day.**
The routed boot reaches established gameplay only near the end of a 200 s run, so a 200 s sample is
dominated by the loading phase and its ratios describe that phase rather than gameplay. Measured with
`screenshot --count 10` (samples every 200 s), by distinct colours in the presented frame:

| t | non-black px | distinct RGB colours |
| --- | --- | --- |
| 200 s | 1,131,517 | **241** |
| 400 s | 1,757,394 | **3,756** |

and an earlier 200 s run reported `distinct_rgb_colors: 2` — a frame that is black to the eye, whose
1,051,186 "non-black" pixels sit at luminance 1-2 (instrument trap 160).

The same counts at 400 s against 200 s show how much the phase moves them:

| | 200 s | 400 s |
| --- | --- | --- |
| lighting `0x20431c0000` passes | 132 | **746** |
| composite `0x2056740000` passes | 9 | **56** |
| scene colour `0x2063380000` as a target | 1 | **1** |
| scene colour sampled | 200 | **1,375** |

So "the 3D chain runs in only ~2% of frames" was an artifact of stopping at 200 s, and should not be
read as a throttling defect. **The one number that does not move is the one that matters**: the guest
binds `0x2063380000` as a colour target exactly once — a start-up clear — while sampling it 1,375
times in 400 s.

**What the frame actually looks like at 400 s**, which is not what the 200 s captures suggested: the
radar renders completely (map, blips, player arrow, N compass, legend bar), the tutorial text is
crisp, and three world lights render with correct bloom halos. It is a real gameplay frame missing
its base scene, not a black screen. With `PROSPER_RTT_ALIAS=2063380000:20431c0000` the same frame
shows the bank interior with correct perspective, a lit doorway and architectural detail —
**19,623 distinct colours against 3,756**.

**Composite input inventory at 400 s** (max `rgb_nonblack` over the run, and pass count):

| input | taps | content | passes |
| --- | --- | --- | --- |
| `0x2063380000` 4K f11f11f10 | x24 | **0** | 1 |
| `0x2085de0000` 4K f16 | x6 | 4,330,576 | 636 |
| `0x206e6a0000` 1080p f16 | x4 | **0** | 56 |
| `0x206e640000` 240x135 f16 | x2 | **0** | 56 |
| `0x20602a0000` 1080p f11f11f10 | x1 | 30,700 | 111 |
| `0x2066be0000` 1080p f11f11f10 | x1 | 112,361 | 110 |
| `0x206b4e0000` 960x540 un8 | x1 | 518,400 | 1 |

Three inputs are empty: the scene colour and the two luminance/exposure-chain buffers. The sparse
bloom buffers are what the three visible lights come through. Note the **format split**: the working
HDR chain (`0x20df360000`, `0x2078ea0000`, `0x2074ee0000`, all heavily sampled) is `f16`, while the
empty tap is `f11f11f10` — so the missing producer is a **format conversion** of a chain that works.

### The deferred lighting resolve is rejected by the DEPTH TEST (2026-08-16)

**This is the sharpest result in this document and it is a prosper defect, not a missing guest
producer.** It is separate from the absent scene-colour producer below; both are live.

The lighting resolve — the 15+ draw pass that consumes the whole G-buffer — emits colour in only
**10 of 122** read-back passes (readback confirmed per pass via `src=`, so these zeros are
measurements and not deferred readbacks). `PROSPER_NO_DEPTH=1` disables the depth test and nothing
else:

| | control | `PROSPER_NO_DEPTH=1` |
| --- | --- | --- |
| read-back passes with 15+ draws | 122 | 120 |
| ...of which carry colour | **10 (8%)** | **65 (54%)** |
| max `rgb_nonblack` | 3,798,218 (46%) | **8,294,400 (100%)** |

A full 4K frame of coverage appears the moment the depth test stops rejecting. So the light-volume
draws are executing and being discarded per-fragment.

**`PROSPER_DEPTH_CLEAR` moves it too, at both poles** — and reading that as a statement about this
pass's initial value is the mistake this paragraph exists to prevent:

| arm | passes with content | max `rgb_nonblack` |
| --- | --- | --- |
| control (derived clear) | 10 of 122 | 3,798,218 |
| `PROSPER_DEPTH_CLEAR=0.0` | **68 of 126** | **8,294,400 (100%)** |
| `PROSPER_DEPTH_CLEAR=1.0` | **0 of 124** | 0 |

`PROSPER_DEPTH_CLEAR_WHY=1` (new) then showed that mixed-compare passes really do exist and really do
latch the wrong pole — the fresh-image value is **first-draw-wins**, so a pass measuring
`greater=15 less=5` derives **1.000** and always-fails its fifteen GREATER draws. That is the #457
class one level deeper, and it looked like the fix.

**It is not the fix, and this is recorded so it is not retried.** Deriving the value by *majority*
instead of arrival order — identical wherever a pass's draws agree, so no existing shape changes, and
246/246 stayed green — made the measured target **worse**: 0 of 111 read-back passes with content
against 10 of 122, with SCANOUT unmoved. So the global `PROSPER_DEPTH_CLEAR=0.0` arm was never
evidence about *this pass's* initial value: it forces **every** depth surface in the frame including
the G-buffer's own, and whatever produces its improvement has another mechanism. The behavioural
change was reverted; `PROSPER_DEPTH_CLEAR_WHY` and its counts were kept.

**The depth-test finding above still stands** — `PROSPER_NO_DEPTH=1` is a per-test lever, not a
per-surface value, and it takes the pass from 10 to 65.

### How much presented content each depth/stencil arm recovers (400 s, same route)

The single most useful table here, because it is measured on the **presented frame** rather than on
an intermediate, and because two independent controls bound the noise at **0.15%** — so every row
below is real.

| arm | SCANOUT `rgb_nonblack` | vs control |
| --- | --- | --- |
| control (`src`) | 1,958,474 | — |
| control (`long`) | 1,955,614 | −0.15% |
| **`PROSPER_NO_DEPTH=1`** | **2,810,666** | **+43.5%** |
| `PROSPER_DEPTH_CLEAR=0.0` | 2,248,371 | +14.8% |
| **`PROSPER_NO_STENCIL=1`** | **2,104,048** | **+7.4%** |
| majority depth-clear derivation | 1,942,739 | −0.8% (refuted, reverted) |

> ## RETRACTED — the metric measures BLOOM, not world content
>
> **Every number in the table above is real and its interpretation was wrong.** `rgb_nonblack` rises
> under those arms because a blown-out white blob and a blue haze flood the frame, not because the
> world appears. **I only found this by opening the images**, which is the whole lesson:
>
> - `PROSPER_NO_DEPTH=1` (2,810,666, +43.5%): a large overexposed blob, blue haze, **no world
>   geometry, and the radar is GONE**. Strictly worse than the control to look at.
> - `PROSPER_NO_STENCIL=1` (2,104,048, +7.4%): the same blob and haze; radar and tutorial text
>   survive; still **no world geometry**.
> - `PROSPER_RTT_ALIAS` — the one arm that genuinely shows the bank interior — scores **19,623
>   distinct colours against the control's 30,262.** It shows *more* world with *fewer* colours.
>
> So on this title neither `nonblack_rgb_pixels` nor `distinct_rgb_colors` separates "the world
> rendered" from "bloom flooded the screen", and they can rank a worse frame higher. This is the
> `## Ruled out` list's colour-metric trap in a new costume: **a bloom smear fools the metric exactly
> as a gradient does.** Quote these counters only alongside the image.
>
> **What survives:** the depth and stencil tests do change what reaches the lighting buffer, and
> `PROSPER_DEPTH_CLEAR=1.0` still drives it to zero while `0.0` does not, so the fresh-image value is
> a real lever on that buffer. What does NOT survive is "disabling the test recovers presented
> content" — it recovers bloom. The mixed-compare and fresh-attachment findings below are unaffected;
> they are structural facts about the pass, not pixel counts.

**These arms are diagnostics, not fixes** — each disables a test the guest asked for.

That is consistent with the DS bridge's own declines on this route: `0x2054aa0000` (the stencil
plane) declines `stencil-invalid` up to **846** times, and `0x2052ac0000` (depth) `depth-invalid`
up to 826. **But note the trap already recorded above**: suppressing those declines wholesale with
`PROSPER_DS_HTILE_INVALIDATE=0` *loses* 29% of presented content, so the invalidation is protecting
something real and "make the bridge always valid" is not the fix either. The open question is what
the correct contents are, not how to stop declining them.

### Why no initial value can work: the resolve mixes GREATER and LESS draws 7:7

`PROSPER_DEPTH_CLEAR_WHY=1` prints the derived value per **colour target**, which is what makes this
answerable. For `0x20431c0000`:

| pass shape | derived | correct? |
| --- | --- | --- |
| `greater=66 less=0`, `greater=8 less=0`, `greater=4 less=0`, … | 0.000 | yes |
| **`greater=7 less=7` draws=15** | **1.000** | **unsatisfiable** |
| **`greater=6 less=7` draws=14** | **1.000** | **unsatisfiable** |

The mixed passes **are** the 14/15-draw resolve, `first_op=1` (LESS) latches the far value, and the
six or seven GREATER draws then always-fail against it.

**This is also exactly why the majority rule failed, and the failure is informative rather than
embarrassing:** at 7 versus 7 the tie-break falls back to first-draw (LESS → 1.0), and at 6 versus 7
the majority *is* LESS → 1.0. Majority reproduces the old answer on precisely the passes that matter.
**No single fresh-image value can serve a pass that mixes both compare directions against one
attachment** — the approximation is not merely mis-tuned here, it is unsatisfiable by construction.

And the attachment really is fresh, which the poles already proved: if it were LOADed from a retained
image the initial value could not matter, yet `PROSPER_DEPTH_CLEAR=1.0` drives the resolve to **0 of
124** passes with content.

**So the direction is: this pass must LOAD the G-buffer's real depth/stencil rather than receive a
freshly-created attachment.** That is a DS-retention problem across the G-buffer → lighting
transition, not a clear-value problem, and it explains why every value-shaped fix has failed.

### The complete chain, end to end

`PROSPER_DSLOG`'s per-pass line reports `persistent` and `valid=<layout>/<depth>/<stencil>` for the
resolve's 14/15-draw passes, and every one of them is keyed correctly to the scene surfaces
(`dr=dw=2052ac0000`, `sr=sw=2054aa0000`, `htile=2055310000`):

| resolve passes (14/15 draws, 4K) | count |
| --- | --- |
| depth **used**, retained depth **INVALID** → falls back to the fresh-image approximation | **171** |
| depth used, retained depth valid | 179 |
| depth not used at all (`use=0/1`, stencil only) | 175 |

So retention is not missing — **validity is**, on about half the passes that actually depth-test.

> **Correction, and it is the kind worth reading.** An earlier revision of this section cited "207
> passes with `valid=1/0/1`" as the dominant case. Those passes carry `use=`**`0`**`/1` — depth is not
> used at all, their per-draw lines read `depth=0/0/opN` with the test disabled — so their depth
> invalidity is irrelevant and they do not belong in this chain. The `valid=` field says whether an
> aspect *could* be loaded; only `use=` says whether the pass asked. **Reading a validity column
> without its use column overstates the population**, which is what happened here. The pass holds the right retained image and cannot
load it, so it falls through to the fresh-image approximation, which on a 7:7 mixed pass is
unsatisfiable and lands on 1.0, which always-fails the GREATER light volumes.

```
HTILE write  ->  depth_valid = false  ->  cannot LOAD retained depth
             ->  fresh-image approximation runs
             ->  mixed 7:7 compares derive 1.0
             ->  every GREATER light-volume draw always-fails
             ->  lighting resolve emits nothing  ->  black world
```

The first link is already measured elsewhere in this document: **730 of 900 invalidations of
`0x2052ac0000` are HTILE-only overlap** (`depth=0 stencil=0 htile_hit=1`, a 655,360-byte write at
`0x2055310000`), and HTILE conservatively discards both aspects.

**The obvious fix — "invalidate only on a fast CLEAR, not on a metadata refresh" — was implemented,
measured, and is INERT here, and the measurement reframes the problem.** A fast clear writes one
value to every tile while a per-tile zmin/zmax refresh does not, so post-write uniformity separates
them without decoding HTILE's layout. Measured on a routed 400 s boot: **all 4,959 HTILE-overlap
writes leave the plane uniform** (`htile_clear=1`), so the discriminator classifies every one as a
clear, behaviour is unchanged, and the code was reverted rather than left dead.

**What that tells us is more useful than the fix would have been: these really do look like fast
clears, so invalidating depth is CORRECT.** The guest is clearing its depth buffer, prosper is right
to drop the retained contents, and the defect is one step later —

> when an HTILE fast clear invalidates depth, prosper has **no clear value** and falls back to
> guessing one from the pass's compare ops. On hardware the clear value is whatever the guest
> programmed (for reverse-Z, the far value 0.0, which its GREATER light volumes pass against). The
> fresh-image approximation guesses **1.0** on this 7:7 mixed pass, and they all fail.

So the fix is to **supply the fast-clear value** rather than to suppress the invalidation. And the
obvious source for it is closed by measurement: `PROSPER_DEPTH_CLEAR_WHY` also prints what
`DB_DEPTH_CLEAR` holds on those passes, and it is **`7.19391e-34`** — a denormal from bits that are
packed integer coordinates rather than a depth value. That is precisely the #371 pathology the latch
comment warns about ("Astro Bot even leaves the packed 1920x1080 max coordinates there ... initialized
every LEQUAL surface to 2.15e-36 and rejected the entire scene"), so **reading the register at pass
time cannot work, and the existing refusal to consume it without an explicit enable is right.**

(The value was visible from the very first `PROSPER_DSLOG` run of this investigation —
`clear=0/7.19391e-34` on the `0x2052ac0000` line — and went unrecognised for a long time.)

What is missing is therefore narrower than "a clear value": a clear performed through **HTILE
metadata** sets no draw's `depth_clear_enable`, so the value it cleared to is never observed at all.
Capturing it has to happen **at the moment of the clear**, not recovered afterwards from register
state that has since been reused.

Also note the blunt version stays falsified: `PROSPER_DS_HTILE_INVALIDATE=0` suppresses invalidation
wholesale and *loses* 29% of presented content, which is exactly what should happen if the clears are
real.

**Do not read `PROSPER_NO_DEPTH=1` as a fix** — it is a discriminator and it breaks other things (the
same family of override made the radar vanish when it was applied to the main 4K depth, recorded at
`live_renderer.cpp` in the `PROSPER_DS_UNBRIDGED_FAR` comment).

### The missing operation, named exactly (2026-08-16)

Tracing the post chain backwards from the composite finds where scene content *does* enter it, and
that pins what is absent to a single conversion.

**The bloom pyramid works, and its source is the lit scene.** The pass writing `0x205f1a0000`
(1920x1080) reads `0x20471e0000` — the 4K **f16** buffer holding the lit scene — and carries
**44% non-black**. The chain below it (960x540 → 480x270 → 240x135 and back up) is populated
throughout. So the guest demonstrably *can* read the lit scene, and does, for bloom.

Meanwhile the composite's base tap `0x2063380000` (4K **f11f11f10**) is empty. That is exactly the
picture on screen: **bloom-lit sources appear, the base scene does not.**

So the absent operation is now a single named conversion:

> **4K f16 `0x20471e0000` → 4K f11f11f10 `0x2063380000`**, once per frame.

**Aliasing the composite's tap to either lit buffer renders the world with the HUD intact** —
`PROSPER_RTT_ALIAS=2063380000:20471e0000` gives **5,413,464 non-black (65% of frame)** and a bank wall
in correct perspective with its architectural lines, the lit source and its bloom, and the complete
radar. (`…:20431c0000` shows a different, darker part of the same room.) Both are diagnostics; neither
is a fix, because prosper must not invent a copy the guest did not issue.

**And they are not the same memory.** `PROSPER_MEMLOG` gives three distinct physical addresses —
scene colour `0x136580000`, HDR/bloom source `0x11a3e0000`, lighting `0x1163c0000` — so prosper is
right to keep them separate and there is no aliasing fix hiding here.

**Which makes guest logic the leading remaining explanation — leading, not sole.** The guest allocates
the buffer, clears it once at start-up, samples it 24x per frame forever, reads its *source* for bloom
in the same frame, and never runs the pass that fills it, so something prosper answers upstream
plausibly selects that path. That is a different hunt from everything above — a guest-decision
question, not a GPU one. It shares the frontier with one unfinished measurement: the image ops inside
the failing compute kernels that no instrument has resolved (see the section below for exactly which,
and why their null does not count yet).

### The composite's tap is renderer-owned, bound by ONE observed draw, with no observed notified refresh (2026-08-17)

Three measurements on the same routed boot, each with its denominator, narrow the tap
`0x2063380000` (`Float10_11_11`, 4K) to a single remaining question.

**One observed draw BINDS it as a colour target — binding is not writing.**
`PROSPER_TARGET_WATCH=0x2063380000` — exact, unsampled, all eight MRT slots, no dimension filter —
reports `draws=1` at every power-of-two checkpoint from 4,096 through **262,144**. One draw, slot 0, and
the count never grows. Whether that draw wrote useful pixels is **not** established here — the open
question below is exactly that, and "bound by one draw" and "written by one draw" are different claims.

**No OBSERVED queued write touches it.** The new `PROSPER_RTT_INVALIDATE_WATCH` reports **131,072
queued guest writes examined, 0 touching either the tap or the f16 source `0x20471e0000`.** That counts
writes reaching `notify_guest_gpu_write`, so it is a statement about notified writes, not about the guest. The snapshot is
therefore never invalidated by a guest write — and this is a measurement rather than a silence, because
the instrument prints its own denominator. It also reports that `0x2063380000` is at some drains **not
in the RTT cache at all**, which is a different state from "cached and never refreshed" and is stated
separately so the two are not conflated.

**It is served from the renderer's own snapshot.** Every one of its 11 `PROSPER_RTT_GUESTPEEK` samples
resolves `HIT-CPU`.

So the composite reads a renderer-owned snapshot associated with one observed binding draw, which no
observed notified write ever refreshes.
**The remaining question is what that single draw does** — whether it is the intended producer rendering
nothing (culled, zero-area, colour writes masked, shader rejected), or merely an initial clear whose
per-frame counterpart is absent. That is one draw out of 262,144, which is a small enough target to
identify directly, and it is the next step.

**On the boundary this does and does not move.** The write watch observes writes that reach
`notify_guest_gpu_write`; a producer writing guest memory without notifying would be invisible here too.
So this remains *no observed path*, not *the guest never issues one* — the same boundary #2542 already
records, now with a denominator of 131,072 rather than an argument. What is genuinely new is that the
surface is renderer-owned and served from a snapshot, which means a guest-memory producer is not even
the mechanism that would fill it.

### THE FAILING KERNEL'S 4K WRITE IS NOT THE COMPOSITE'S TAP — measured, same run (2026-08-17)

**Read this before the section below it.** That section reports that two failing compute programs bind
4K write-capable storage images, and that the only write-class binding of `0x204da00000` came from a
dispatch that never ran. Both facts hold. The *inference* everyone then drew from them — that unblocking
that program feeds the composite — is **false**, and a single run with `PROSPER_RTT_GUESTPEEK`
distinguishes them because it prints each sampled surface's FORMAT:

| surface | format | role | resolved |
| --- | --- | --- | --- |
| `0x2063380000` | **`Float10_11_11`** (fmt 20) | the composite's base tap, sampled 11x | **`HIT-CPU`**, all 11 |
| `0x20471e0000` | **`Float16`** (fmt 4) | the presumed HDR source | `HIT-CPU` |
| `0x204da00000` | **`Float32`** (fmt 1) | what `0x2042f49a00` writes, sampled **once** at pc=75 | `miss` |

`0x204da00000` is a different address AND a different format from the tap. `0x2042f49a00` reads main
depth and stencil and writes two **Float32** images — a depth-derived pass, not the f16 → f11f11f10
conversion the composite is missing. **So this failing kernel is not the DIRECT producer of the tap, nor
of the f16 → f11f11f10 conversion.** That is exactly what the format mismatch establishes and no more:
it does not rule out compute as an *upstream* input to the one graphics draw that binds the tap, so
"the compute path is not the route" would overstate it.

The source-authority work is worth having on its own terms — it **defines** the decision that must be
made before compressed bytes are read as plain texels. But note what is not yet true: **neither runtime
consumer is wired.** The compute adapter is parked and the graphics adapter is unwritten, so the policy
exists and no path is closed by it yet.

**And the tap resolves `HIT-CPU` on every sample**, which changes the question entirely. The composite
does not read empty *guest* memory — the renderer owns that surface and serves its own CPU snapshot. So
a compute dispatch writing guest bytes is not even the mechanism that would fill it. What is empty is
the **renderer's snapshot**, and what fills that is a renderer/draw question.

**`PROSPER_RTT_GUESTPEEK`'s 0% distinguishes nothing here, and this is an instrument trap worth naming.**
All **twenty** distinct `Float10_11_11` surfaces in the run report 0% non-zero guest bytes, and most
resolve `HIT-CPU` — including `0x205f1a0000`, the bloom source this document elsewhere credits with 44%
content. Empty guest memory is the *normal* state of a renderer-owned surface, so "guest bytes are zero"
is not evidence that a surface is unproduced. The claim that the tap is effectively empty stands on the
**aliasing A/B** (SCANOUT 13.5% → 55.6% with real geometry), which is a rendering measurement; it does
not stand on guestpeek.

**One assumption corrected in the other direction:** these addresses are **stable across runs**.
`0x2063380000`, `0x20471e0000`, `0x2052ac0000`, `0x204da00000` and `0x205f1a0000` all appear at the
addresses this document recorded hours earlier, in a fresh boot. Earlier notes here cautioned that
addresses are run-local and re-derivation is required. That is true of *some* allocations — the storage
image `0x205b557e00` binds did move between runs — but not of these, so a cross-run comparison of the
named surfaces is legitimate.

**Where the work actually goes next.** The tap is renderer-owned and its snapshot is empty, and the
earlier colour-target census found **1 draw in 65,536** ever binding `0x2063380000` as a colour target.
So either that single draw is the producer and it renders nothing, or the producer writes it by some
path the resolved-table census cannot see. That is the next thing to establish, and it is a graphics
question rather than a compute one.

### The failing kernels bind 4K write-capable storage images (2026-08-17)

The section below draws the census boundary at "the unresolved operations inside the failing compute
kernels remain outside the census". This looks *inside* that boundary for the first time, using a
diagnostic that was already ungated and already in every routed log: `[compute-table]`, which dumps a
program's whole resource table once per program that fails to recompile.

**Two failing programs bind a storage image of exactly 33,177,600 bytes = 3840 × 2160 × 4** — the
size of a 4K f11f11f10 surface, the very thing the composite's base tap is and does not contain:

| program | binding | guest address | reject |
| --- | --- | --- | --- |
| `0x2042f49a00` | 6 | `0x204da00000`, **identical on all four runs** | `pc=16`, MIMG `op=0x1`, `mode=unresolved-operand` |
| `0x205b557e00` | 13 | run-local (`0x2070f20000`, `0x2072f00000`) | `pc=314`, MIMG `op=0x0`, `srt_tag=0xa0`, `key_res=null pc_res=null` |

`class=4` is `StorageImage` — read/written by `image_load`/`image_store` **without a sampler**, i.e.
the class a compute producer writes through. `0x205b557e00` also samples a 3840×2160 renderer-owned
RTT that reports *"has no readable snapshot -> dispatch skipped (#590)"*, so it reads a 4K surface and
writes a 4K surface, and never runs.

**This is the shape of the missing producer, and it is not proof.** What is established: failing
kernels do bind 4K write-capable images, so the population the census excluded is not empty and is not
irrelevant. What is *not* established: that either surface **is** the composite's base tap. Addresses
are run-local, the tap was identified in an older run, and nothing here correlates the two — the
honest statement is a size and class match on a population that was previously unexamined.

**The failing population is also much larger than this document has been recording.** The reject
census table above lists 13 programs; a routed boot has **30, 32, 36 and 35 distinct failing programs**
across four runs on 2026-08-17. So "the seven failing kernels" understates it by roughly a factor of
five, and any statement of the form "all the failing kernels were cleared" should be read against a
count that was never that small.

#### Watching that surface: the only WRITE-class binding of it came from a dispatch that did not run

`PROSPER_COMPUTE_BINDS=204da00000` on a routed boot, 219 rows, 4 distinct programs. Two of them are
the whole result:

```
[compute-binds] 0x204da00000 bound by program=0x2042f49a00 binding=6 class=4 fetch_pc=21
                addr=0x204da00000 size=33177600 3840x2160 outcome=partial-recompile-empty
[compute-binds] 0x204da00000 bound by program=0x205b557e00 binding=7 class=2 fetch_pc=19
                addr=0x204da00000 size=33177600 3840x2160 outcome=executed
```

One 4K surface. `class=4` is `StorageImage`, the write-capable class; `class=2` is `Texture`,
sampled-only. **The write-class binding is on a dispatch whose recompile failed — it did not run —
while the sampled binding is on a dispatch that executed.** Across the whole run no executed dispatch
was observed binding this surface write-class.

Two qualifications, because both matter for how far this can be pushed:

- **Recompile success is per-dispatch, not per-program.** Both programs also appear in the skip list
  (`0x2042f49a00` at `pc=16` MIMG `op=0x1`; `0x205b557e00` at `pc=314` MIMG `op=0x0`), and a program's
  resource table differs between dispatches, so the same code address can resolve in one invocation
  and not in another. The claim is therefore about *observed bindings*, not about programs.
- **This surface has not been shown to be the composite's base tap.** It is 4K, 33,177,600 bytes, and
  written by nothing that ran — the right shape, on the right scale, in the population the census
  excluded. That is a lead, not an identification.

**The remaining 217 rows are instrument noise worth knowing about:** they come from two programs
binding 256 MB constant buffers whose spans happen to *contain* the watched address. That is the
documented "match the whole span, not the base" behaviour doing its job, but it means this instrument
needs its output filtered by `class=` and by an exact `addr=` match before the signal is visible.

**This row existed only because of the fix in `668a65af`.** Before it, `report_compute_binding_watch`
returned from the `item.spirv.empty()` branch without reporting, so a recompile-failed dispatch
produced no row at all — and this census would have shown the executed *consumer* and nothing else,
i.e. exactly the false "no compute producer writes this surface" conclusion. The reviewer's finding
that the instrument's null did not cover failing shaders is what made this visible.

**Next step, now concrete:** make `0x2042f49a00` recompile. Its reject is a single named instruction —
`pc=16`, MIMG `op=0x1` (`IMAGE_LOAD_MIP`), `mode=unresolved-operand`, `dmask=0x1 dim=1 glc=1` — and
the `[mimg-mip-why]` lines for that program report `proven_at_use=1 mip_vgpr=v2 in_zero_set=1
exec_pristine=1 cfg_known=1` at pcs 16/18/21/25, so the mip level is proven and the failure is the
descriptor, not the mip analysis. Then re-run this census and see whether the write-class binding
becomes `outcome=executed`.

### What invalidates the retained depth — 100% of it is prosper's own writeback (2026-08-17)

**Read the retraction first.** An earlier revision of this section reported "~81% self-inflicted" and
"1,678 **genuine guest** HTILE writes" for the main depth, and named the frontier as the semantics of
those guest writes. **Both numbers were wrong and the frontier was misdirected.** The instrument had a
second hole: `notify_guest_gpu_write_preserving_bytes` knew its own classification, printed it to the
immediate watch, and then called the observer with only `(addr, size)` — so all four byte-preserving
compute writeback paths arrived at the queue as the **default**, and the default was named `gpu`. A
bucket that means "nobody said" was therefore reported as "genuinely the guest".

With the origin carried as data across the notification/observer boundary, and the default renamed to
`unknown` so it can never again be mistaken for a producer:

| origin | count |
| --- | --- |
| `compute-writeback(image-guest-bytes)` | 15,017 |
| `compute-writeback(cpu-fill)` | 10,234 |
| `gpu-preserving` | 5,590 |
| `unknown` (unattributed) | **0** |

**30,841 of 30,841 DS invalidations are prosper invalidating its own caches from its own writebacks.
Not 81% — all of them. There are no genuinely-guest DS invalidations on this route at all.**

For the main depth `dr=0x2052ac0000`, 2,276 invalidations, every one ours:

| aspect hit | origin | count |
| --- | --- | --- |
| `htile_hit=1` | `gpu-preserving` | 1,542 |
| `htile_hit=1` | `compute-writeback(cpu-fill)` | 367 |
| `htile_hit=0` | `compute-writeback(cpu-fill)` | 367 |

Its depth range is still never directly written — every loss of the plane arrives through the
conservative "an HTILE overlap may describe both aspects" rule. But the writer is **us**, and 1,542 of
those come from a path whose whole contract is that *the guest bytes were not modified*.

**So the frontier is not the guest's HTILE semantics.** It is: why does prosper's own byte-preserving
writeback overlap this depth surface's HTILE range at all, and should a write that provably preserves
guest bytes invalidate a detached Vulkan depth image? The second half already has an answer on record
and it is **yes, it must** — byte equality with guest memory says nothing about equality with a
renderer-owned image the renderer has since drawn into, and *Dead Cells* #611 is the counterexample
where sparing it makes gameplay geometry disappear. That is why no preservation policy is proposed
here. The open question is the first half.

**A default is not a measurement.** This is the third time on this title that a default or
unattributed value was read as a positive finding — the others being a black frame compared against a
mismatched trigger, and an `htile_hit=0` aggregate dominated by cube shadow maps. In all three the
number was real and the population behind it was not what the label said. Name unattributed buckets
`unknown`, and check what a "no origin" case actually means before counting it as evidence.

### No OBSERVED DECODED path produces `0x2063380000` — which is not the same as "the guest never issues one"

Read the boundary in this heading before using the table. Every path that can write a surface **and
that prosper resolves** has an instrument, and all of them are empty for `0x2063380000`. That is a
real and useful result, and it is weaker than the claim this section used to make.

**What stays outside the census.** Every instrument below enumerates *resolved* state — a decoded
packet, a built resource table, a bound target. The seven compute programs that fail to recompile do
not have fully resolved tables: `PROSPER_DYNTRACE_FAIL` recovered **15 of 21 image ops** for
`0x205b5e8600` and **4 of 6** for `0x205b657200`, and the candidate-word scan that was meant to
cover the remainder is non-exhaustive in coverage, validation and direction by its own documented
contract (`live_renderer.cpp`, above `scan_for_descriptors_once`). So the unrecovered image ops in
those kernels are **unproven, not cleared** — no instrument here has looked at them.

**The supported conclusion**, therefore: *no observed decoded path produces this surface; guest-side
path selection is the leading hypothesis; the unresolved operations inside the seven failing compute
kernels remain outside the census.* An earlier revision of this section said the elimination was
exhaustive and that "the guest never issues it" — that overstated the evidence in the one direction
the evidence cannot support, and the correction is recorded rather than silently applied because the
overstatement had already reached the PR body and #2542.

**The exact next step, and it closes the gap rather than working around it:** make those two kernels
recompile — the unresolved VCC_LO operand at `0x205b5e8600` pc=314 (a scratch-SGPR bitfield, not a
lane mask, and not cross-block: see `## Ruled out`), `image_bvh_intersect_ray` for
`0x205b657200` — so their resource tables resolve in full, then re-run `PROSPER_COMPUTE_BINDS` over
the completed tables. Until then a null across that population is not evidence. This is the same work
already queued as recompiler frontier, so it is not extra: it is the census and the candidate fix at
once.

The instruments and their results:

| path | instrument | result |
| --- | --- | --- |
| draws, every drop class | `PROSPER_DROPPED_DRAW_CENSUS`, extended to `RetainedDrawNotSelected` + `IndirectArguments` | 256 drops, all on two other targets; **7,871 retained/indirect attempts, 0 dropped** |
| DMA / WRITE_DATA / RELEASE_MEM / EVENT_WRITE | `PROSPER_GUEST_WRITE_WATCH` | nothing |
| DS planes | `PROSPER_DSBRIDGE_LOG` | not a plane of any retained surface |
| guest CPU | `PROSPER_RTT_GUESTPEEK` | 0/1048576 bytes non-zero, stable to sample #1024 |
| undecoded PM4 | `[pm4] unknown raw type-3 opcode` (ungated) | **zero unknown opcodes on the whole route** |
| async-compute queue | `q=` in the WaitRegMem line | ACB *is* processed (38 `q=A` vs 2 `q=D`) |
| predicated jumps | `PROSPER_PREDLOG`, now sampled across the whole run | **8,192+ jumps, `pred=0` on every one, `skip=0`** |
| whole-submit rejection | `[agc] ordered DMA submit rejected` | zero |
| **the guest's own register writes** | `PROSPER_TARGET_WATCH=0x2063380000` — exact, unsampled, all eight MRT slots, no dimension filter | **1 draw out of 65,536**, slot 0, against 7,989 for the lighting buffer `0x20431c0000` in the same run |
| **prosper-built packets prosper never decodes** | `[pm4] undecoded prosper sub-op` (new, ungated) | one family, `r=0x00 op=0x10` — a plain `IT_NOP` pad |
| AGC copy builders | firmware NID list vs registrations, with the unimplemented logger verified live | `sceAgcDcbCopyData` / `AcbCopyData` are **unregistered**, and the title **never calls them** |

That last row is the strongest one: prosper is not losing a producer *through the copy builders*,
because the title never calls them. Note what it does and does not settle — it removes one candidate
producer completely, and says nothing about the unresolved image ops named above.

**Every null above now carries a positive control**, which was not true when this section was first
written. `PROSPER_GUEST_WRITE_WATCH` produced no output at all on the early runs — including for the
heavily-rendered lighting buffer — so its silence proved nothing; watching the HTILE base alongside
makes it fire 22 times in the same run while the scene colour stays at zero. The
undecoded-sub-op reporter closes the last structural gap: the raw-opcode reporter beside it only
catches opcodes prosper does not recognise, while a packet prosper's own builders emit with a sub-op
the decoder never learned would have been dropped in complete silence. There are none but NOPs.

It is stated from `PROSPER_TARGET_WATCH` rather than from `PROSPER_COLORSTATETRACE` deliberately, and
the difference is not cosmetic. The colour-state trace agreed (1 record in 1,437,781) but could not
have disagreed convincingly: invoked as `=3840x2160` it **filters by dimension**, so a bind at any
other extent is absent without saying so, and the natural way to read its output — grepping
`color0=` — misses slots 1..7, which carry 157,855 records each on this route. Two independent ways
to conclude "the guest never renders here" from a measurement that could not have seen it.
`PROSPER_TARGET_WATCH` is exact, unsampled and slot-complete, and its own comment explains why a
sampled census cannot answer an "ever" question at all. **Pass its addresses with a `0x` prefix** —
it parses base-0, so bare digits are read as decimal and it will confidently report nothing about a
different address (instrument trap 180).
It allocates and clears its own 4K HDR scene buffer once, samples it 24× per frame forever, and never
renders to it.

And the emptiness is not confined to that one surface: **the entire post-process chain is clears.**
The first bloom level `0x20602a0000` is written by one draw with *no sampled inputs* and
`px_nonzero=0`, exactly like the scene colour's single pass. A whole phase of the frame is missing
its work while the lighting that feeds it is 45% populated.

### What that leaves, and the caveat on the compute census

The remaining candidate is the compute chain, and specifically the nine programs that never execute.
**Six of them — the `0x205b5*` cluster — dispatch at exactly 1920×1080**, which is the post-chain's
own resolution (`0x20602a0000`, `0x2066be0000` and the rest are all 1920×1080). The only 4K one,
`0x2042f49a00`, is a depth decompress: `PROSPER_COMPUTE_MEMPROBE=2042f49a00:0:40:16` decodes its
source T#s to `0x2052ac0000` (depth) and `0x2054aa0000` (stencil) with `0x2055310000` (HTILE) as
metadata, and it stores to plain copies at `0x204b1a0000` / `0x204d180000` / `0x204da00000`.

**`PROSPER_COMPUTE_BINDS` reports zero programs binding `0x2063380000`, and that null is VOID for
exactly the population in question.** The instrument enumerates a dispatch's resolved resource table,
and the never-executing programs are precisely the ones whose resource tables fail to resolve — so it
cannot see their bindings however correct it is. The positive control that "validated" it
(`0x20431c0000`, many rows) was bound by an *executing* program, which tests the discriminator and
not the domain. Do not cite that zero as evidence about the nine.

## Ruled out

One line per falsified hypothesis, the evidence that killed it, and where. **Read this before forming
a new one** — and note which entries are *solid* versus *void*, because a void result is not a
falsification.

- **The HTILE `gpu-preserving` suppression is load-bearing for GTA V's picture.** FALSE, measured
  2026-08-28 (#3089). **And it should never have shipped: the answer was already recorded above in
  this same document** -- "should a write that provably preserves guest bytes invalidate a detached
  Vulkan depth image? ... **yes, it must** ... and *Dead Cells* #611 is the counterexample where
  sparing it makes gameplay geometry disappear. That is why no preservation policy is proposed
  here." `e63f4038` shipped that preservation policy anyway, and it cost *Blue Prince* its entire
  picture. `e63f4038` ("gpu: GTA V world rendering series", the squash of #2996; branch commit
  `ee0d57c4`, not on master) stopped invalidating retained depth when a guest HTILE write
  compared byte-identical, to keep the depth this title's deferred lighting samples. That rule is
  **unsound in principle**: prosper never writes rendered HiZ back into the guest HTILE plane, so
  the guest copy is a constant the guest's own writes keep reproducing, and the comparator reports
  `changed=0` for a fast CLEAR exactly as readily as for the decompress it was written for -- on
  *Blue Prince* every DS invalidation was suppressed (agree=1, suppressed=59,999) and the title went
  to 0.00% non-black over 16,500+ readbacks. **Removing it does not cost GTA V anything measurable**: one binary, one lever, the
  540 s `reach-story-mode` route — the suppression fires **3,177 times** with it on and **0** with
  it off, and peak colour coverage is **99.78% in both arms**, on the same targets, at comparable
  readback ordinals (#237 vs #225). So the suppression was removed. Recovering that pass's retained
  depth needs the fix the surrounding comment already names — **decoding HTILE to tell a clear from
  a refresh** — not an equality test on a plane prosper does not maintain. Caveat, stated because it
  bounds the claim: the GTA V arms were compared on peak colour coverage and target identity, not by
  a reviewed frame-by-frame diff.
- **The one JumpPatchTarget call per run that hands a non-Jump header (`0x3e718000`) is patching
  somewhere inside a known DCB.** *VOID, not falsified — and the first attempt to kill it was wrong in
  the opposite direction.* An instrumented run showed `patch_target_writable` answering through the
  `guest_writable` OS probe rather than the ring registry, and that was published as "not a command
  buffer, so all three candidates die". **It does not support that.** `remember_dcb_extent` has a
  **single call site** (`hle_agc.cpp:581`, inside `set_regs_indirect`), the array is `thread_local`,
  and it holds **four FIFO slots** evicted oldest-first — so a miss means only "not among ≤4 recently
  registered extents on this thread that had a RegsIndirect packet built into them". A later run
  settles how weak that is: the registry reads **0/4 slots used** at the moment of the refusal, so
  `in_known_dcb` could not have returned true for *any* address. The discriminator was structurally
  incapable of firing, which makes its negative void rather than merely weak.
  **The dump's own window points the other way, and the framing is the evidence — not any single
  recognisable header.** The eight dwords behind `cmd` are two consecutive, perfectly framed packets
  that prosper itself emitted, whose lengths chain and land exactly on `cmd`:
  `0xc0041068` = `PM4(kDwDispatch=6, IT_NOP, R_DISPATCH_DIRECT)` whose five payload dwords decode as a
  coherent `748x1x1` dispatch with modifier `0x21`, followed by `0xc0001000` = `PM4(2, IT_NOP, 0)` with
  payload `0x6875000d`, the marker emitted in `sceAgcCbSetShRegisterRangeDirect` (`hle_agc.cpp:1603`).
  So `cmd` is a packet **boundary** in an AGC stream **prosper itself built**. The address argument was
  misread too: across three runs the addresses share their low **16** bits (`7d34`) — the same offset
  into a relocated allocation, not a heap object. (The first write of this entry claimed low 20 bits;
  that holds for two of the three runs only.) **Candidate 1 (a consumed packet / recycled ring) is now
  the leading reading, not a dead one.** #2715, #2856, #2857.

- **`0x413dc6700` contains no guest barriers, so all eight emitted `OpControlBarrier` are emulation
  scaffolding.** *Falsified — and it was a grep artifact, not a measurement.* The program contains
  **two** `s_barrier` (SOPP `0x0a`, encoded `0xbf8a0000`), at **pc 116 and pc 129** — which are exactly
  the phase boundaries this file already records (phase 0 = `pc 0..<116`, phase 1 = `117..129`). The
  program is barrier-phased, as `:83` and `:4767` say. The "zero" came from a `grep` whose pattern had
  three spaces after `fmt=SOPP` where `shader_inspect` prints four; the tool had reported the two
  barriers all along. Counted from the raw dwords by encoding (`(w >> 23) == 0x17F`, op `0x0a`) rather
  than from formatted text: 2. #2717.
- **What the scope comparison actually shows.** *Solid, and narrower than the retracted version.* The
  guest's loop exit is **per-wave** (`s_cbranch_execz`), while the emitted module votes
  **workgroup-wide** — a 256-way OR of a per-lane pending bit into `LDS[260]`, four waves of 64 in a
  261-entry array, wave index `%92 >> 6`. The module carries **eight** `OpControlBarrier` against the
  guest's **two**. Of the eight, **three** immediately follow a store to the vote word — one per
  dispatcher phase, the write→barrier→read the vote needs. The remaining five are **not attributed**:
  saying "six are the vote's and two are the guest's" would be arithmetic, not a measurement, and the
  mapping from the guest's two `s_barrier` onto specific emitted ones has not been established.
  Whether the scope difference causes anything is untested — see the next row. #2717.
- **`0x413dc6700` never corrupts a clean input.** *Falsified.* Eight consecutive submits do show 88
  dispatches running `0 -> 0` on all four link arrays, which is what suggested it — but with coverage
  widened to nine submits, **one dispatch with a clean input turns a clean 2063-entry array into 537
  cyclic entries while terminating normally**, and is not a runaway. So the consumer does corrupt clean
  input, on one dispatch in nine submits. That also restores this file's own
  `### The writer is the CONSUMER ITSELF` finding, which an earlier revision of this row contradicted.
  #2711, #2717.
- **Our own `PROSPER_COMPUTE_SKIP_PROGRAM` declines leave stale arrays that read as cyclic.** *Solid.*
  A run declining **nothing** shows the same corruption (`0 -> 84` at the first runaway submit) before
  dying at `0x413e14900`. The workaround is not manufacturing the defect. #2711, #2717.

- **`0x413dc6700` computes on zeros because prosper cannot express its pointer-chase load, so GTA V's
  world needs `PhysicalStorageBuffer` / `VK_KHR_buffer_device_address`.** *Solid.* Filed as #2709 and
  argued further in its comments; also carried into #2711's Q1/Q2 answers as a premise. The emitted
  SPIR-V really does contain **one** storage-buffer access chain for **41** MUBUF sites — verified
  exhaustively, since across every op in the module only `OpVariable` and one `OpAccessChain` yield a
  `StorageBuffer` pointer. The inference from that was wrong, because **a legal `NUM_RECORDS=0` fold
  and a never-decoded instruction leave identical evidence**: both leave no access chain, no load, no
  trace at all. This was the fold. The `[buf-op]` disposition line (#2712) says which — all 41 sites,
  `zero-record` — and a live `[dyntrace]` census over **352 program folds** (14,432 buffer-op *site*
  observations: 352 folds x 41 MUBUF sites) then showed **10,824 of those site observations resolve
  concrete descriptors with real extents against 3,608 zero-record**. Per fold the split is clean:
  `pc91` resolves a real V# in **264 of 352** folds and is zero-record in the other 88. Quoting 14,432
  as a fold count conflates the two units and does not survive comparison with any dispatch count. Nothing about this program requires
  address-based memory access; the translated-guest-address work in #2711 Q3 stands on its own merits
  and must not be justified by this program. #2709, #2712.

- **Sampled-depth invalidation is starving the deferred-lighting pass.** *Solid.* The lighting pass
  does read the scene depth `0x2052ac0000` through the bridge, and that bridge did decline — 826 times
  on `depth-invalid` — so the hypothesis was well-founded. It is still wrong. 730 of the 900
  invalidations of that surface are **HTILE-only** overlap (`depth=0 stencil=0 htile_hit=1`, a
  655,360-byte write at the HTILE base `0x2055310000`), and HTILE conservatively discards both
  aspects. Suppressing that with `PROSPER_DS_HTILE_INVALIDATE=0` moved the lever as hard as it can
  move: **depth-invalid declines 1063 → 0, stencil-invalid 852 → 0, depth hits 42,183 → 44,406,
  stencil hits 2,934 → 3,917** — and SCANOUT came back **byte-identical** (1,121,820 / 1,051,186
  `rgb_nonblack`). A bridge that now never declines an aspect still yields the same black world, so
  depth availability is not the constraint on the lighting output. Do not re-run this arm expecting
  pixels; it is a bridge-health improvement whose visual effect is nil.
  (The arm remains default-**on**, i.e. historical behaviour, because separating a fast clear from a
  compute HiZ refresh needs HTILE decoded, and only the first justifies discarding depth.)
- **`0x205b658800`'s M0 read is consumed as data, so a fabricated value is unsafe.** *RETRACTED — the
  read is a pure save/restore after all, and the retraction is more instructive than the entry.*
  I withdrew a proposed M0 fix on the grounds that the saved value is used as data: `pc=82` saves M0
  into `s6`, `pc=232` restores it, and `pc=263`/`pc=266` then appear to consume `s6` as VOP2 sources
  with nothing writing it in between. **That reading is wrong.** `pc=261` and `pc=264` are VOP3
  opcode `0x360` — `v_readlane_b32 s6, v31, N` — which is an architectural **SGPR** write, so `s6` is
  dead before `pc=263` and the launch value really is only round-tripped.
  **How the error was made is the point.** I checked the claim with `shader_inspect`, whose generic
  operand printer spells that destination `dst=vgpr:6`. That is the decoder's own representation, not
  an ISA oracle — and this repo already knows better in two places: `rdna2_decode.cpp:127`
  (*"v_readlane_b32 writes an SGPR"*) and the emitter at `rdna2_to_spirv.cpp:11620`, which treats the
  destination as `sDST`. **Validating a decoder's output with that decoder's own listing is not a
  check**, which the charter says in as many words; an independent disassembly (`llvm-mc -mcpu=gfx1030`)
  settles it in one command. Caught in review (#2552).
  **Consequence:** the M0 poison-tag approach is viable again and should be reconsidered — resolve an
  untracked M0 to 0 while marking the destination unproven, so a restore is harmless while
  `V_WRITELANE` and descriptor uses still reject.
- **The composite's scene-colour T# is stale, mis-derived, or points somewhere prosper invented.**
  *Solid, and this is the version to cite* — it supersedes the provenance argument in the entry
  further down, which reached the right answer by a route that does not establish it (see the note
  there). Aim the shared dynamic fold at the composite pixel shader:
  `PROSPER_DYNTRACE=1 PROSPER_DYNTRACE_ADDR=205b34be00 PROSPER_DYNTRACE_ONCE=1`. It prints the eight
  raw T# dwords read from the guest's own table, and the composite's are
  `20633800 c2400000 021bc3bf 91b003ac 00000000 00700000 00000000 00000000` → `base=0x2063380000`,
  with `have_t8=1` (came through a successful scalar load) and an immediate SRT key (`key=0x0` /
  `key=0x60`). **prosper decodes the guest's live descriptor correctly. The guest itself points its
  composite at the buffer it never fills.**
- **GTA V submits command buffers through an entry point prosper does not implement.** *Solid.* This
  was the one gap every other instrument was blind to by construction — they all observe *decoded*
  streams, and a stream never handed to the decoder is invisible to all of them. prosper registers
  **2 of the 17** `Submit*` exports in `libSceAgcDriver` (`SubmitDcb`, `SubmitAcb`); the other
  fifteen include `SubmitMultiDcbs` (`6UzEidRZwkg`), `SubmitMultiAcbs` (`HF3YllT3mXU`),
  `SubmitCommandBuffer` (`b4fpgH5ZXxQ`), `SubmitMultiCommandBuffers` (`Fj7r9EHzF38`),
  `SubmitMultiCommandBuffersDirect` (`xmWi73o1BR0`), `AgrSubmitDcb` (`AhGvpITrf4M`) and
  `AgrSubmitMultiDcbs` (`+T8Xo6LtFJI`). **GTA V calls none of them** — zero mentions across a routed
  run, and the unimplemented-NID logger demonstrably fires for that library in the same run (it names
  three other `libSceAgc` NIDs). The registration gap is real and worth closing for other titles; it
  is not this title's defect.
- **`sceAgcDcbRewind` lets commands past a ring wrap go unseen.** *Solid, by review rather than
  measurement* (Codex, #2542): `IT_REWIND` (type-3 `0x59`, two dwords, `(initial_state & 1) << 31`) is
  a **validity wait**, not ring control — the decoder's whole behaviour is `while (!Valid()) yield`,
  then advance normally. It never moves the command pointer or delimits a submitted range. The stub
  also leaves no hole: only the builder advances the cursor, so later builders append compactly.
  What *is* missing is the validity synchronisation, which can let post-Rewind packets run before
  their producer publishes them — worth implementing, but log the call count and `initial_state`
  first, because an unconditional no-op is only safe if the title always passes 1.
- **`0x2063380000` and `0x20431c0000` are two virtual mappings of the same physical memory.** *Solid.*
  This was the best remaining structural explanation — it would have accounted for the guest rendering
  to one name and sampling the other, and for the alias experiment working *exactly*. `PROSPER_MEMLOG=1`
  refutes it: both live in the same mapping (`va=0x203de00000 len=0x120f00000 phys=0x111000000`) at
  **different physical offsets**, `0x136580000` for the scene colour against `0x1163c0000` for the
  lighting buffer. They are distinct memory. (The run does contain one genuine physical alias —
  `va 0x2168da0000` and `va 0x2169580000` share `phys 0x23b0b0000` — but that pair is in the swapchain
  range and is unrelated.)
- **The single bind of `0x2063380000` is the pass that should have drawn the scene.** *Solid.* It is at
  **0.0% of the run** — line 83 of 1,441,036 colour-state records — with `raw-format=6`,
  `resolved-cwm=f`. A start-up initialisation clear, thousands of frames before the 3D chain begins at
  ~87%. There is no gameplay-time bind to recover.
- **Predicated jumps are dropping the composite (the #319 shape, one title over).** *Solid.* The
  polarity in `command_processor.cpp` (`skip = cond != 0`) is `CONFIDENCE: MED` and pinned on another
  title, so this was a fair suspicion. Measured across a whole routed run: **8,192+ folded jumps,
  `pred=0` on every one, `skip=0`.** GTA V does not predicate its jumps at all. The old 96-line cap on
  that log is why this looked open — it expired thousands of frames before the 3D chain begins at
  ~87% of the run, so it only ever described the loading screen.
- **`sceAgcAcbJump` is the missing producer (the #319 defect on the async-compute queue).** *Solid.*
  It was genuinely unregistered, and `hle_agc.cpp:675` records that the DCB sibling's absence made
  "the composite never execute" — GTA V's exact picture, and GTA V does drive post-processing through
  the ACB. But a builder mirroring the DCB argument roles refuses on the first call:
  `target=0x0 ndw=0x21 a5=0x5 a6=<the acb again> a7=0x1 a8=0x2ec`, so `a3` is not the target and the
  roles do not transfer. It is also called **exactly once** in a 200 s route, so it cannot be a
  per-frame producer under any ABI. **The builder and its NID registration were removed entirely**
  (review of #2552): a handler whose argument roles are known to be wrong is not a neutral observer —
  registering it moves the call off the unregistered-NID path that would otherwise report it, so the
  guess would have been inherited as a decoded contract. The call is left unregistered.
- **The composite's scene-colour T# is stale or mis-derived — argued from `rtt-guestpeek`
  provenance.** *The conclusion is right; THIS derivation is not, and it is the derivation that would
  have been inherited.* The argument was: the working `0x20431c0000` and the empty `0x2063380000`
  resolve identically (no SRT key, no SGPR key, matched by fetch pc), so provenance gives the failing
  one no special status. Codex refuted the premise (#2542): **`sgpr_base != UINT32_MAX` means direct
  user-SGPR origin, not "fresh by construction"** — a draw can inherit SH state and the value can be an
  earlier bind, and only `PROSPER_UDPROV`'s last-write order establishes freshness relative to *this*
  draw. Likewise a valid `srt_offset` means an immediate-key table load, not that prosper cached the
  T#: `resolve_dynamic_fetch` rereads the table per stage build, and a first realization can still
  observe guest memory before its writer. So identical provenance fields never ruled staleness out.
  **Use the raw-T# entry above instead** — it reads the guest's actual descriptor dwords and settles
  the same question by measurement. Recorded rather than deleted because a `## Ruled out` row wearing
  a plausible argument is exactly what the next reader will not re-derive.
- **A missing device capability blocks the rejected compute kernels.** *Solid.* The native subgroup
  contract is **ENABLED** on this machine (`size_control=1 full_subgroups=1 vote=1 arithmetic=1`,
  sizes 32..64, AMD Radeon 8060S / RADV STRIX_HALO), so `0x205b5e8600`'s VCC_LO read is not blocked by
  a capability. **The second sentence of this row used to read "its mask is untracked because the
  write is in another basic block — a cross-block dataflow limit"; that is wrong, see the next row.**
- **`0x205b5e8600` needs cross-block VCC lane-mask dataflow.** *Falsified, and it would have cost a
  large piece of the wrong work.* Disassembled independently with `llvm-mc -mcpu=gfx1030` rather than
  from prosper's own listing, the reject site is:
  ```
  pc=313  s_lshl_b32 vcc_lo, s80, 14
  pc=314  s_and_b32  vcc_lo, vcc_lo, 0x1c000     <- the rejected instruction
  pc=318  v_add3_u32 v2, vcc_lo, v3, v2
  ```
  **There is no lane mask here at all.** The compiler allocated VCC_LO as an ordinary scratch SGPR;
  the value is `(s80 << 14) & 0x1c000`, a bitfield, consumed by an integer add. Nor is it cross-block:
  the two writes and the read are all in the block starting at pc=310. So the queued work item
  "implement cross-block VCC lane-mask dataflow" does not describe this kernel, and implementing it
  would not clear this reject.
  Two synthetic reproductions of the shape — a straight-line scalar recycle of VCC_LO after a
  `v_cmp`, and the same overwrite placed after a CFG merge so it dominates every read — **both
  recompile** on this head, so the minimal shape is already supported and the cause is something in
  the kernel's surrounding context (a VCC bool tag that survives with no bool value, making
  `wave32_live_mask_operand` select the lane-mask path at `rdna2_to_spirv.cpp:9280` while `mask()`
  then resolves VCC_LO to 0 and sets `ok = false`). Closing it needs the real kernel recompiled with
  its actual resource context, not another synthetic case. `CONFIDENCE: HIGH` on the disassembly and
  on both negative reproductions; `CONFIDENCE: LOW` on the parenthesised mechanism.
  Note also that the reject-census table above says every row is `mode=unresolved-operand`, *"i.e.
  every one is a descriptor that does not resolve"* — that gloss holds for the MUBUF/MIMG/SMEM rows
  and **not** for this one, where the unresolved operand is a register, not a descriptor.
- **`no-entry` at 81% of DS-bridge calls is a defect.** *Instrument.* `find_persistent_ds_sampled` is
  consulted for **every sampled binding**, so `no-entry` counts every ordinary colour texture in the
  frame; the code says so at `tests/fixtures/render_runner.h:2796` ("the no-entry bucket is unbounded and is
  genuinely just 'not ours'"). Of the ~45,700 decisions that concern a real DS plane, ~92% hit. Read
  the *ranked per-address* declines, never the bucket total.
- **The 72 direct draws are failing / are culled / have colour writes masked.** They execute at full
  3840x2160 with `effective=3f` and zero realization failures in the window that presents black. The
  world is drawn by *indirect* operations, which the latch above dropped untried. #2481.
- **Collapsed AABBs culling the world.** The reduction at `0x413ced900` computes a correct bounding
  box, `(-13.71, -23.65, 0)` to `(14.98, 16.95, 4.101)`. #2481.
- **`CB_TARGET_MASK=0` masking colour writes (the #1946 shape).** The main-view pass reports
  `effective=3f`. The pass that showed `target-mask=0 effective=00` was a **512x512 depth-only shadow
  atlas**, where that is correct — selected by a `MIN_DRAWS` filter, not by phase. Instrument trap 159.
- **The sky renders correctly offline and is lost live.** The `--draw-steps` composite that showed a
  blue gradient with a horizon band is prosper's **seed-miss placeholder**: already 100% non-black at
  step 4, before most draws run, and unchanged through step 78. Instrument trap 161.
- **The whole-frame abort on an unsupported storage image.** `render_draw_pass_rgba`
  (`tests/fixtures/render_runner.h:3309`) does abandon an entire 4K frame on the first invalid storage-image
  contract, which is a real disproportionality — but gating it so only the offending draw is dropped
  left the black window unchanged. Reverted rather than landed. #2481.
- **A dark frame containing a dim scene.** Frames measuring 0.00% non-black at threshold 8/255 while
  12.7-13.5% of pixels are non-zero at luminance 1-2, with a CRC that changes every second, are
  **animated dither** — amplified 64x there is no structure, edge or geometry. Instrument trap 160.
- **The recompiler is the frontier.** `[compute] skip unsupported program` named 15 programs; retried
  offline against their own captured resource tables, **16 of 20 recompile cleanly**. Only 9 of 196
  failures in the classified capsule are `shader-recompile`. Instrument trap 157.
- **Unresolved descriptors on the hanging program.** All **43** of `0x413dc6700`'s resources resolve
  with real addresses and sizes; none is null. (The zero-address resources in an earlier dump belong to
  `0x413dc3400`, the neighbouring dispatch.) #2481.
- **The hang is an out-of-bounds write from a mis-sized buffer bound.** `RADV_DEBUG=hang` produced a
  report whose **`vm_fault.log` is 0 bytes** — no page fault, no VM fault. This was the leading
  suspect given #2529/#2535's history with `scalar_buffer_dword_count`. #2481.
- **The hang is a non-uniform `OpControlBarrier` deadlock.** The CFG dispatcher's continue block runs
  two workgroup-scope barriers per iteration and its LDS reduction covers only `[base..base+63]`,
  which looks exactly like a per-wave exit under a workgroup-wide barrier at `local=256`/`wave=64`.
  It is not: the loop-exit predicate reads **slot 260**, the whole-workgroup liveness result, so every
  invocation iterates together. #2481.
- **The hang is a non-uniform early exit.** The 14,370-line disassembled module contains exactly
  **one** `OpReturn`, at the end. No `OpKill`, none inside any of the three dispatcher loops. #2481.
- **The hang is a cyclic traversal table (183 cyclic starts).** **RETRACTED — see the retraction
  section above.** The 183 cycles are in `0x20f848417c`, the table read by the dispatch that
  *completed*; the hanging dispatch reads `0x20f848a240`, which is acyclic with a longest chain of 11
  steps. The measurement was taken from the wrong dispatch's binding. #2542.
- **A lost atomic corrupts the traversal table.** **VOID against the current writer — the evidence
  below is about `0x413ce3400`, and the table's writer is `0x413dc6700` itself.** The corruption
  signature (61 two-cycles) still stands as an observation, and so does the instrument note at the end
  of this entry, which is why the entry is kept rather than deleted. What does not stand is the
  falsification: showing that `0x413ce3400` performs no atomic cannot rule out a lost-atomic
  write-path defect in a *different* program. To settle it, re-run the same footprint analysis on
  `0x413dc6700`'s own stores. Everything from here to the end of this bullet is that superseded
  argument. The table that hangs `0x413dc6700` is a linked list
  whose corruption is 61 **two-cycles** (`i` and `i+2` pointing at each other), the classic signature
  of a non-atomic concurrent insertion — two threads each linking to the other because both read the
  head before either wrote. Its producer `0x413ce3400` performs **no atomic operation of any kind**,
  and the strong form of that is structural rather than grep-shaped: its entire memory footprint is
  **29 buffer loads and stores** (`buffer_load_dword` x15, `buffer_load_dwordx2` x9,
  `buffer_store_dword` x4, `buffer_store_dwordx3` x1) and **zero `ds_*` instructions**, so there is no
  LDS family in which an atomic could hide. Zero `OpAtomic*` in its 3,881-line recompiled SPIR-V
  agrees. **Positive control, a different program in a different capsule**: `0x413ced900` contains
  `buffer_atomic_fmax` x3, `buffer_atomic_fmin` x3, `ds_max_f32` x3 and `ds_min_f32` x3 — so the
  instrument fires on the buffer family *and* the LDS family, and the zero is a real negative.
  **Note what the control also demonstrates:** a `(buffer|global|flat|ds)_atomic_*` pattern silently
  misses LDS atomics entirely, because RDNA2 spells them `ds_max_f32`, not `ds_atomic_max` — the
  control's own `ds_min/max_f32` would not have been found by it. That gap is why the claim above
  rests on the absence of the whole DS family rather than on an atomic-shaped pattern. #2542.
- **The producer/consumer ordering violation is `WAIT_REG_MEM` being barreled through.** The guest
  issues waits prosper cannot satisfy (`[agc] WaitRegMem … dependency violated … LABEL-UNMAPPED`, 40
  per route), and by default an unsatisfied wait does not pause the queue — which would let a
  consumer run before its producer's results land, exactly the symptom. **Falsified** with the
  opt-in barrier model `PROSPER_WAIT_DEFER=1` (#312): the device is still lost, at the same program
  and the same dispatch index. **Lever verified before reading the result** (instrument trap 164):
  the run logs 40 `pausing queue (deferred effects)` and the baseline logs 0, against 28 and 40
  `dependency violated` respectively, so the model was genuinely active. Note the recorded #312
  verdict against defaulting this ON was measured entirely on *Dragon Quest VII*'s heap corruption
  and says nothing about GTA V — this is an independent falsification, not a re-derivation of it.
  #2542.
- **The hang is an unconditionally infinite loop.** The same program executes successfully at dispatch
  38 and hangs at dispatch 39 of the same submit. Whatever spins is data-dependent. #2481.
- **Our `v_cmpx` / `s_cbranch_execz` lowering cannot exit the guest's pointer-chasing loop.**
  `0x413dc6700`'s whole 903-dword body contains exactly **one** backward branch, at guest pc97 back to
  pc88, and that loop's only exit is `s_cbranch_execz` after a `v_cmpx_ne_u32`. Nothing else in the
  loop writes EXEC, so if `v_cmpx` narrowed VCC instead of EXEC — plausible, because the e32
  encoding's destination field still reads as VCC and `shader_inspect` prints it as `special:106` —
  the loop could never end. **Falsified by a hand-built kernel of the identical shape**
  (`tests/shared/diagnostics/test_cfg_trip_bound.cpp`): built by hand rather than derived from the capture or the
  recompiler, its body decrements the index instead of chasing a buffer, so only the control flow is
  on trial. On real Vulkan, lane *i* walks exactly *i* steps for all 128 lanes — per-lane EXEC
  narrowing and the cross-lane `execz` vote are both correct. #2542.
- **Bounding the CFG dispatcher's trip count stops the hang.** Tried at 4096 and at 2^20; the device
  was lost both times, so a bound at those values does not rescue the frame. This is **not** in
  tension with the hit witness firing at 4,096 further down: the witness says the loop *reaches* the
  cap, and the cap then truncates that dispatch's control flow, which produces wrong results for
  every later consumer rather than a working frame. "The bound does not fix the title" and "the loop
  runs away" are both true. #2481.

### Void, not falsified — do not cite these as settled

- **Portable wave64 emulation cost is the hang's mechanism.** A run with
  `PROSPER_NATIVE_COMPUTE_MULTIWAVE=1` still hung, which was reported as a falsification and **is
  not one**: the emitted module is **byte-identical** with and without the switch
  (`d04fd09b13408f9b4da7287fae34f692` in both arms and in a capsule from hours earlier), so the two
  runs are the same run. `[subgroup] … native=64 … multiwave=1` reports the resolved *config*, not the
  emitted lowering. The hypothesis is neither confirmed nor refuted — though it is now less likely on
  other grounds, since `native_subgroup_size` was apparently already 64, which suggests those barriers
  are the guest's own `s_barrier`s rather than emulation scaffolding. Reopening it needs a lever
  verified by module hash **before** its result is read. #2481.

## CORRECTED AND CONVERGED: `0x413dc6700` dispatch 39 takes ~2 SECONDS, and the loss follows it

Measuring the DURATION of every compute fence wait, and reporting any wait over 100 ms even when it
succeeds, produced exactly one line in a whole route:

```
[compute] SLOW fence wait 2045.2 ms result=0 program=0x413dc6700 submit=8116 dispatch=39 order=14036 groups=9x1x1
[compute] fatal Vulkan device loss stage=queue-submit … program=0x413dc6700 submit=8116 dispatch=40 order=14041
```

**One abnormal wait in the entire run — 2,045 ms against sub-millisecond for everything else — on the
dispatch every other instrument has named, immediately followed by the device loss on the next
dispatch.**

Three independent instruments now converge on `0x413dc6700` dispatch 39: the trip-bound hit witness
(`trips=4096`, with no invocation ever reaching an ordinal past the loop body), the fence-wait
duration, and the loss ordering. The witness's fields are dispatcher
quantities; resolve an ordinal against the `dispatch map:` line the same phase prints.

### This corrects the section that used to be here

That section argued "no compute dispatch ever hangs", from **0 fence-wait timeouts across 705 waits**.
The count was right and the inference was wrong: the timeout is **30 seconds** and the event is **2
seconds**, so a dispatch can be three orders of magnitude slower than every other one and still never
trip it. A zero timeout count measures only "nothing exceeded 30 s" — it says nothing about whether
anything is pathologically slow, which is the actual question.

The alternative that motivated the check — that a context reset SIGNALS pending fences, so a killed
job returns `VK_SUCCESS` and looks instant — remains untested and is no longer needed to explain
anything.

What survives from that section: the loss is still reported at `queue-submit` on the *next* dispatch,
so the loss line alone never named the culprit. Instrument trap 170 stands as written about
attribution; only the "not compute's" conclusion drawn from it was wrong.

## Superseded: the earlier reframing

###

Counted across every routed run in this investigation:

| observation | count |
| --- | --- |
| compute dispatches that entered their fence wait | **705** |
| compute dispatches that completed it | **705** |
| `[compute-decline] reason=queue-wait` (a 30-second fence timeout) | **0** |
| `[compute-decline] reason=queue-submit` (device ALREADY lost) | **28** |

`execute_item` submits and then waits on a fence with a **30-second** timeout, and reports a timeout as
`queue-wait`. That has never once fired. Every compute dispatch this title issues completes.

The 28 losses are all `stage=queue-submit`, which means `vkQueueSubmit` returned
`VK_ERROR_DEVICE_LOST` — the device was **already** dead when compute next submitted. The dispatch
named in that message is the first one *after* the reset, not the one that caused it.

**So the GPU hang is not caused by a compute dispatch.** It is almost certainly in the graphics
submission path, and the compute backend is a victim that discovers the dead device and then disables
itself process-wide — which is what drops every later indirect draw.

### What this overturns

- **"`0x413dc6700` hangs the GPU."** Its dispatcher loop genuinely runs past 4,096 iterations — the hit
  witness recorded that directly, at `trips=4096`, on dispatch 39 — but it still *finishes* inside the
  30-second fence. A long loop, not a hang.
- **"`0x413e14900` is a second hanging program."** It is a 753-dword module with **zero loops** running
  `threads=42x1x1 groups=1x1x1` with `result=ok`. It was never a candidate.
- **The trip-bound A/B matrix.** Each cell was a single run of a failure that varies run to run — the
  same build died at `0x413dc6700` dispatch 39 in one run and dispatch 40 in another, and reached
  `0x413e14900` in a third. Single-run cells cannot support the causal reading I gave them.

### What survives

- The cyclic traversal tables are real and measured at the correct timing. They make that loop very
  long. They do not, on this evidence, hang the GPU.
- The queue-2 indirect-argument recovery is real and deterministic: 50 of 64 skipped dispatches now
  execute.
- The consequence chain from the loss onward — compute disabled process-wide, `producer_epoch_ok`
  cleared, indirect latch, no world — is unchanged and still explains the black frame.

**The next investigation is the graphics submit path**, not the recompiler and not compute resources.

## FIXED: indirect compute dispatches on queue 2 had no argument base — 50 of 64 were skipped

**What the fix reliably changes:**

| | before | after |
| --- | --- | --- |
| `indirect dispatch skipped: unreadable arguments` | **50 of 64** | **0** |

That is deterministic and verifiable in every run: the skips are gone and those dispatches execute.

**What it does NOT reliably change, corrected after a second run.** The first run with the fix got past
`0x413dc6700` and died later at `0x413e14900` dispatch 52, and its frame showed sun, lens flare and
radar. I wrote that up as a before/after improvement. **A second run with the same build died at
`0x413dc6700` dispatch 40 again.** The hang is data-dependent, so a single run either side proves
nothing about it, and the frame content had already appeared in earlier *skip* runs — so neither the
survival nor the frame is attributable to this fix.

The honest statement is: the fix removes a real and deterministic class of dropped work; whether that
changes the hang is unmeasured, and would need repeated runs on both sides.

### The defect

`SET_BASE_INDIRECT_ARGS` sets one `indirect_compute_base`, and that is **per-fold** state. A PS5
process has one GPU virtual address space, but GTA V's async-compute queue carries `DispatchIndirect`
packets whose 32-bit payload is a full address *within the already-selected aperture* and no SetBase
of its own — so its base is zero and its arguments resolve to an unmapped low address.

Probed on **49 of 49** such dispatches before changing anything:

```
readable? low=0  aperture20=1  hi-dword=0
          (low=0xf8480120  ap20=0x20f8480120  hi64=0x21f8480120)
```

The raw low address is unmapped; `aperture | low` is mapped; folding the modifier as an ADDR_HI is
not.

**That is evidence about where the arguments live, and it is not provenance — which is why the
recovery is opt-in.** This paragraph used to end "so a wrong aperture leaves behaviour exactly as it
was", and that claim does not hold: the aperture is learned from *any* SetBase on *any* queue, and one
process VA space can contain mapped allocations under several high-32 prefixes. A wrong aperture that
happens to land on a mapped allocation is accepted, and the dispatch then reads group counts out of
whatever lives there. Gated behind `PROSPER_INDIRECT_APERTURE_RECOVERY`, default off.

### What it does NOT fix

The world is still black, and the traversal tables are still cyclic in a large minority of reads
(944 cyclic vs 1,179 clean, against 806/1,782 before — proportionally better, not resolved). So the
skipped dispatches were **a** cause of lost work but not the whole cause of the cyclic structure. The
next device loss is `0x413e14900`, which a dispatcher bound does not save either.

## Superseded: the root-cause candidate as first written

`SET_BASE_INDIRECT_ARGS` sets one shared `indirect_compute_base` in the command processor. Logging
base, offset and queue separately at the `DispatchIndirect` site:

| queue | dispatches | base | outcome |
| --- | --- | --- | --- |
| **1** | 14 | `0x205b690f80` | resolve and run |
| **2** | **50** | **`0x0`** | `args = raw offset` -> unreadable -> **skipped** |

The queue-2 offsets are `0xf8480120`, `0xf8480160`, `0xf84801a0`, `0xf84801e0`, `0xf8480220` … stepping
by 0x40 — and the guest arena those dispatches belong to is at **`0x20f8480000`** (347,040 bytes; it is
the buffer the `s_endpgm` kernel declares). So the intended address is almost certainly
`0x20f8480120`, and what is missing is a `0x20_00000000` base that queue 2 never receives.

**50 of 64 indirect compute dispatches in one route are therefore skipped entirely**, with only
`[agc] indirect dispatch skipped: unreadable arguments at 0x…` to show for it — a message that prints
the SUM, so it cannot distinguish a bad offset from a base that was never set.

### Why this is very likely THE defect

It closes the chain that every other measurement in this document constrains:

1. `SET_BASE_INDIRECT_ARGS` for compute is seen only on queue 1; queue 2's folds start with base 0
2. queue-2 indirect dispatches resolve to an unmapped address and are skipped
3. the skipped dispatches are the maintenance passes for the traversal structure
4. the parent array degrades — **once, irreversibly** (each buffer transitions clean->cyclic exactly
   once and never recovers, which is what a half-applied union-find update looks like)
5. `0x413dc6700` walks the cyclic chain and never terminates — recorded directly by the trip-bound
   hit witness (its third field is a dispatch ordinal — see the correction below)
6. GPU watchdog -> RADV hard recovery -> live compute disabled process-wide
7. `producer_epoch_ok` cleared -> `indirect_dependencies_ok` latched -> every remaining indirect draw
   dropped untried -> **no world**

**Not yet proven**: that supplying the missing base makes the tables stay acyclic. That is the next
experiment, and the cycle census is already the oracle for it — the number to move is
`cyclic-roots`, per dispatch, at pre-dispatch timing.

## THE TABLE IS CYCLIC AT DISPATCH TIME — measured with the right instrument at the right moment

`PROSPER_COMPUTE_PARENT_WALK` reads the selected resource **immediately before `compute({item})`** —
the timing the capsule's post-submit snapshots never had — and models this exact loop
(`while (index != 0) index = (records[index] >> shift) & mask`), reporting cycles directly.

```
PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64
```

**1,782 resolved reads across one route:**

| reads | cycles | cyclic roots (of 2,063) | oob roots | max depth |
| --- | --- | --- | --- | --- |
| 16 | 0 | 0 | 0 | 12..15 |
| 960 | 0 | 0 | **1,894** | 19 |
| **795** | **15** | **1,805** | 0 | 22 |
| 5 | 17 | **2,062** | 0 | 20 |
| 6 | 1 | 1,039 | 269 | — |

**806 of 1,782 dispatches receive a table in which 1,805-2,062 of 2,063 roots lead into a cycle.** The
guest loop cannot terminate on that data, so the dispatcher spinning past 4,096 iterations is the
*correct* behaviour for what it was handed. Combined with the hit witness, the chain is closed: bad
data in, non-terminating walk, GPU hang.

The 960 reads with `oob-roots=1894` terminate by the RDNA2 rule that an out-of-range `idxen` load
returns zero — those dispatches complete, which is why only some dispatches hang.

**Read the whole distribution, not the first rows.** The first three log lines of this run show
`cycles=0 cyclic-roots=0`, and stopping there gives exactly the opposite conclusion — "the data is
fine, so the emulation is broken". The clean reads are a real minority of the population, and they
come first.

### And it is ONE buffer, not a random failure

Correlating every resolved read by the table address it walked:

| table address | outcome | reads |
| --- | --- | --- |
| `0x20f848417c` | clean | **966** |
| `0x20f848417c` | cyclic | 6 |
| `0x20f848a240` | **cyclic** | **800** |
| `0x20f848a240` | clean | 10 |

`0x20f848a240` is cyclic in **800 of 810** reads (98.8%); `0x20f848417c` is clean in **966 of 972**
(99.4%). The program alternates between two traversal tables, and **one of them is systematically
corrupt while the other is systematically correct**. That is not a race or a timing artifact — it is
a property of one buffer.

The predecessor correlation carries no signal: both outcomes follow the same preceding dispatch
(`previous-code=0x413dc6700`, realized, mostly not executed) in the same proportion.

Note this is the same address whose *post-submit* snapshot I measured as acyclic, and used to retract
the cyclic-table hypothesis. At dispatch time it is cyclic. The original hypothesis was right in
substance; the capsule's snapshot timing is what made it look wrong.

### The writer is the CONSUMER ITSELF — this kernel corrupts its own structure

`0x413dc6700` both walks the table and writes it. Its own write-back lines say so:

```
[compute] execute submit=4310 dispatch=776 code=0x413dc6700 threads=2063x1x1 local=256x1x1 buffers=43
[compute]   writeback binding=4 addr=0x20f848a240 size=8252 changed=2062 hash=b24dd1c1…->514428f8…
```

It reads through one binding (`fetch=0x5b`, the loop load) and writes through another
(`binding 4`/`binding 5`, `fetch=0x35`/`0x41`), and the two tables swap roles between dispatches — a
double-buffered structure. Combined with the loop shape (chase a parent index to a root, then take the
**parity of the depth**, `v_and_b32 v1, 1, v2` compared against `s24`), this is the shape of a
**union-find with path compression**: walk to the root, then write compressed parents back.

That reframes the defect entirely. There is no upstream producer to blame — **the corruption is
produced by the same program that later chokes on it.** Dispatch N writes a malformed parent array;
a later dispatch reads it, finds a cycle, and never terminates.

It is consistent with the address split: `0x20f848a240` is the buffer this kernel *writes*, and it is
the one that is cyclic on later reads (800 of 810); `0x20f848417c` is clean 966 of 972.

**Correction to an attribution I nearly published.** I first read the trace as naming `0x413cf9a00`
the writer — a program that is genuinely skipped (`[compute] skip unsupported program 0x413cf9a00`) and
which sits on the recompiler-reject list, so it made a very attractive culprit. It was a parsing
error: my attribution regex latched onto the nearest `program 0x…` text, which included the skip line
itself. The real anchor is the `[compute] execute … code=…` line, and it says `0x413dc6700`.

So the next question is not "who failed to write this" but **"why does this kernel's own output
contain cycles"** — a store-path or algorithm-emulation question about `0x413dc6700`, not a missing
producer.

### What this settles, and what it reopens

Settled: the defect is **upstream data**, not the dispatcher's lowering of this loop. The recompiler
faithfully executes a walk that genuinely does not terminate.

Reopened: **why is the table cyclic?** The earlier producer investigation was measured on
post-submit snapshots and on the wrong dispatch's buffer, so it has to be redone against
`PROSPER_COMPUTE_PARENT_WALK` timings.

**The sentence that used to close this paragraph — "`0x413ce3400` remains the only writer identified
so far" — contradicted the section it sits in and is withdrawn.** The writeback evidence directly
above names `0x413dc6700` as the writer of the table it later chokes on. Both readings were live in
this document at once, which is worse than either being wrong: they imply different next experiments
(find the producer vs. audit this kernel's own stores), and the reconciliation is at the top of the
file.

## CONFIRMED BY HIT RECORD: phase 0's dispatcher loop runs past 4,096 iterations on the hanging dispatch

`PROSPER_CFG_TRIP_BOUND=N` (diagnostic) forces a dispatcher loop out after N iterations. With
`N=100000` the run gets **past** `0x413dc6700` and dies much later at a **different** program:

> **The switch has since changed and these runs predate it.** It now bounds only the program and
> phase named by `PROSPER_CFG_TRIP_BOUND_PROGRAM` / `PROSPER_CFG_TRIP_BOUND_PHASE`, and **the phase
> selector is required** — armed without one it emits nothing and says so. The table below was taken
> when the bound applied to every dispatcher phase of the selected program, so reproduce it by naming
> a phase explicitly. The maps every phase prints when the bound is armed list what is available.

| run | device lost at |
| --- | --- |
| default | `0x413dc6700` submit 4535 **dispatch 39** order 5642 |
| + empty-kernel fix | `0x413dc6700` submit 5968 **dispatch 39** order 17056 |
| + no buffer cache | `0x413dc6700` submit 5963 **dispatch 39** order 9755 |
| + `PROSPER_WAIT_DEFER=1` | `0x413dc6700` submit 4643 **dispatch 39** order 1206 |
| **+ `PROSPER_CFG_TRIP_BOUND=100000`** | **`0x413e14900`** submit 5954 **dispatch 52** order **24374** |

**Isolated control — one program instrumented, only the constant differs.** `PROSPER_CFG_TRIP_BOUND`
is targetable with `PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR`, which leaves every other recompiled module
byte-identical. Both arms below instrument `0x413dc6700` and nothing else — the run log carries a
single arm line naming that one program:

| bound | target | outcome |
| --- | --- | --- |
| **4,096** | only `0x413dc6700` | **gets past it**, dies later at `0x413e14900` |
| **4,000,000,000** | only `0x413dc6700` | **dies at `0x413dc6700`** |

Same instrumentation, same module perturbation, different constant. That removes the two alternatives
an untargeted bound leaves open: it cannot be an *earlier* truncated shader feeding different data
downstream (no other shader is touched), and it cannot be SPIR-V perturbation changing RADV code
generation (both arms carry the identical counter). This is targeted A/B evidence that phase 0's loop is implicated. It is **not** proof that the
cap fired — see the witness section below.

**Which phase spins — bisected.** `PROSPER_CFG_TRIP_BOUND_PHASE=K` bounds only the K-th dispatcher of
the selected program. Three runs, everything else byte-identical:

| phase bounded | guest pc range | outcome |
| --- | --- | --- |
| **0** | **0..116** | **gets past `0x413dc6700`**, dies later at `0x413e14900` |
| 1 | 117..129 | dies at `0x413dc6700` dispatch 39 |
| 2 | 130..902 | dies at `0x413dc6700` dispatch 39 |

**Only bounding phase 0 saves the device** (targeted A/B evidence, not proof of a cap hit). Phase 0 covers guest pc 0..116, and the program's only
loop is at **pc 88..97** — inside it. That is the pointer chase:

```
 88  v_mov_b32_e32     v2, s22
 89  v_cmpx_ne_u32_e32 0, v1        ; EXEC &= (v1 != 0), never restored in the loop
 90  s_cbranch_execz   7            ; exit when EXEC == 0
 91  buffer_load_dword v1, v1, s[0:3], 0 idxen
 95  v_bfe_u32         v1, v1, 3, 27
 97  s_branch          -10
```

So the runaway dispatcher is the one wrapping the EXEC-narrowing walk. The defect is named to a
program, a phase, and a guest **pc** range — pc is a dword offset and RDNA2 instructions are variable
length, so an instruction count cannot be read off it; this line previously called the range
"117-instruction", which it never was.

(The "maximum chain of 11 steps" quoted here came from the capsule-timing measurement that the
retraction above supersedes. The pre-dispatch census is the current figure: 806 of 1,782 reads receive
a table in which 1,805-2,062 of 2,063 roots lead into a cycle.)

**The spinning dispatcher is the PORTABLE one, and that also explains the earlier void result.** The
emitted module for `0x413dc6700` contains **zero `OpGroupNonUniform*` instructions** — so
`b.native_subgroup_size` was 0 and the branch votes take the portable path: publish into
`vote_pending_var` / `vote_value_var` in the switch case, then reduce through LDS scratch behind two
workgroup barriers in the continue block.

That is forced, not incidental. `rdna2_to_spirv.cpp` sets `b.native_subgroup_size = 0` when
`partial_barrier_phases || exact_partial_dispatcher`, and this dispatch is both barrier-phased and
partial (`threads=2063`, `local=256` — the final workgroup carries 15 real threads of 256).

**This retires the `PROSPER_NATIVE_COMPUTE_MULTIWAVE` mystery recorded above as "void, not
falsified".** That switch only moves `config.native_subgroup_size`, which this line then overrides to
0 for exactly this shape — so the module *could not* change, and the byte-identical hashes were the
correct outcome rather than a broken lever. The result stays void as evidence about emulation cost,
but the reason is now known.

It also narrows where the defect can live: the vote machinery under suspicion is the portable
LDS-reduction path for a barrier-phased dispatcher with a partial final workgroup, not the native
subgroup path. Two things checked there and found correct: the vote mailboxes (`vote_pending_var`,
`vote_value_var`, …) are reset at the top of every dispatcher iteration, so a stale vote cannot carry
over; and each lane's LDS contribution is gated on its own `pending` bit.

**The device-side hit witness WORKS, and the cap fires — on the dispatch that hangs.** The shader
writes into the top of the internal GDS buffer when a cap runs out; the host prepares those dwords
before the selected dispatch, reports them after it, and then RESTORES the guest's original values,
so the shared GDS buffer is byte-identical afterwards for whatever dispatch uses it next. It touches
them only when a witness was actually **emitted** for that program — not merely when the selectors
accept it, since a structured loop or a phase ordinal the program lacks satisfies every selector and
emits nothing. **The current record is five dwords — hit flag, phase, highest trip count, and
the lowest and highest dispatcher switch-case ordinal visited — with the last three reduced across
invocations by device-scope atomics.** The per-invocation "last block index" this line used to name
no longer exists: it was one sample, it could not answer whether the dispatcher was cycling, and its
label was published wrongly twice (instrument trap 172).

**Positive control first**: at bound **2** — which a 15-block dispatcher phase cannot satisfy — the
witness produces **1,606 hit records**. It fires.

At bound **4,096**, on the same program and phase:

```
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=38
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=14 trips=4096 submit=5547 dispatch=39
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=40
    ^ HISTORICAL TRANSCRIPT. `last-block` is the label this run printed; the field is a dispatcher
      switch-case ordinal and the per-invocation form was removed rather than renamed again. The
      current line reports `trips` and `dispatch-range` only. See the correction directly below.
```

**Correction (2026-08-14): the third field is a dispatcher switch-case ORDINAL — `next-dispatch`.**
It was first printed as `last-block`, then briefly "corrected" to `next-guest-pc`; both were wrong,
in opposite directions, and the second was published on this branch before being caught. Every write
to the emitter's `pc_var` stores `dispatch_for_block[...]`, so despite the variable's name it is
neither a basic-block index nor a program counter. Instrument trap 172.

**What this does and does not settle.** The numbers 9, 14, 9 are ordinals into phase 0's dispatch
table. Whether ordinal 9 covers the guest's loop body at pc88..97 is a question about that table, not
something derivable by hand from the value — and the hand-mapping is what produced both errors. Each
phase now prints its map when a bound arms:

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map: 0:pc0..<N 1:pc... ...
```

so the ordinal resolves against emitted evidence rather than by hand.

**Re-measured on a routed run, 2026-08-14, with the ATOMIC record — and the number changed.**

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map:
    0:pc0..<41  1:pc41..<50  2:pc50..<55  3:pc55..<61  4:pc61..<67  5:pc67..<73
    6:pc73..<76  7:pc76..<88  8:pc88..<91  9:pc91..<98  10:pc98..<103 ...
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 trips=4096 dispatch-range=0..9   (5 hits, 1 loss)
```

**The earlier `dispatch-range=6..9` was an artifact of last-writer publication and is withdrawn.**
That record was written with plain stores from every invocation reaching the cap, so it reported one
invocation's local extremes; the reduction over all invocations is `0..9`. The claim built on it —
"the state machine visited only ordinals 6..9, so it is cycling in the loop" — does not survive,
because ordinal 0 is the program entry and every invocation passes through it. A ten-ordinal span is
not evidence of a cycle.

**What the corrected record does establish is the CEILING, and it is stronger.** The maximum ordinal
reached, over every invocation and workgroup, is **9** — the loop body at `pc91..98`. Ordinals 10..14
(`pc98..116`, everything after the loop) were **never reached by any lane**. So no invocation ever
left the loop, which is exactly the claim at issue, and it now rests on a true reduction rather than
on whichever invocation happened to write last.

The lesson is worth keeping separate from the conclusion: the conclusion survived, and the derivation
did not. A last-writer record produced a *narrower*, more striking span than the truth, and a narrower
span is the direction that reads as stronger evidence — which is why nobody questioned it.


### Where to look

The dispatcher exits when the whole-workgroup liveness slot (scratch slot `padded_lanes +
wave_count`) reads zero. Non-termination means at least one lane stays *active* forever — its
`active_var` never clears, or its next-`pc` never reaches an exit block. That is a recompiler
correctness question about `emit_cfg_state_machine`, independent of any resource or descriptor.

## RETRACTED: the cyclic-table root cause. The hanging dispatch's table is ACYCLIC.

Earlier revisions of this file, and several comments on #2542 and #2481, stated that `0x413dc6700`
hangs because its traversal table contains 183 cyclic start indices. **That is wrong**, and it was
wrong in the most embarrassing way: I measured the wrong buffer.

`0x413dc6700` runs many times per submit with **different tables**. Same program, same fetch PC,
different resolved base:

| dispatch | binding 8 @ `pc=0x5b` | outcome | cycles | longest chain |
| --- | --- | --- | --- | --- |
| `compute[37]`, source 38, order 16836 | `0x20f848417c` | **completed** | **183** | 58 |
| `failure[1]`, source 39, order 16841 | `0x20f848a240` | **HUNG** | **0** | **11** |

I dumped `--dump-compute-resource 37:8` believing it was the hanging dispatch's table. It is the
*succeeding* one's. The whole cyclic-table chain was built on the buffer belonging to the dispatch
that worked.

Two conclusions follow, and the second is the useful one:

- **The hanging dispatch's table is well-formed.** Zero cycles from all 2063 starts, longest chain
  **11 steps**, 19,038 total steps across every start. That loop cannot hang on this data — it is
  three orders of magnitude short of a watchdog timeout.
- **A table WITH 183 cycles was walked by a dispatch that completed.** So a cycle in this structure
  does not hang the shader either, and the loop model that predicted it must be incomplete — most
  likely the per-dispatch `s18` bound means the cyclic region (indices 412..1238) is not reachable
  from that dispatch's start set.

**So the cause of the hang is unknown again.** What survives is everything about the *consequence*:
`0x413dc6700` hangs deterministically at the same dispatch index, RADV hard-recovers, live compute is
disabled process-wide, the indirect latch drops every remaining draw, and skipping that one program
yields zero device losses and the first real scene content this title has produced. The mechanism
inside the shader is not established, and the "183 must become 0" oracle is void.

The pointer-chase loop is now *less* likely to be the hang: it is bounded at 11 iterations on the
hanging dispatch's own data. **Superseded in one respect — see "`0x413dc6700` has exactly ONE loop"
below.** The "three dispatcher loops" are prosper's own, one per barrier-delimited phase, not three
guest loops: the guest program has a single back-edge and no indirect branches, and it is this one. So
"the other two are unexamined" is a statement about our CFG lowering, and the phases those two wrap
contain no guest loop at all.

## The first 88 folds see an EMPTY SRT — and a capture taken during them is unrepresentative

`0x413dc6700` declares **2 user SGPRs**: one SRT pointer, nothing else. It loads five 64-bit pointers
out of that table at `+0x18/+0x20/+0x28/+0x30/+0x50` and builds all five of its V#s from them.

For the program's first **88** folds those five slots read zero, so every V# decodes
`{base=0, stride=<immediate>, NUM_RECORDS=0}` and all 41 of its buffer ops fold: 18 loads (49 dwords)
to a constant zero, 23 stores (41 dwords) dropped — **none of the 41 MUBUF sites emits a memory
access**. (The module is not access-free: it retains exactly one `StorageBuffer` access chain, from a
non-MUBUF path.) After fold 88
every fold resolves real descriptors. Measured in two runs whose submit numbering otherwise differs,
and the boundary is **deterministic** in both: **88 zero-record folds** (3,608 buffer-op site
observations = 88 folds x 41 sites), contiguous from fold 0, then none. No fold mixes the two states —
0 of 352 are mixed — which is what makes 88 x 41 and 264 x 41 identities rather than coincidences.

`PROSPER_COMPUTE_MEMPROBE=413dc6700:0:0:40` dumps that table and confirms both states are genuine: the
bytes are **unchanged between its two sample points** for the empty ones too, so those dispatches are
really handed an empty table rather than read too early — a genuine empty-SRT input state, not a
fold-time staleness artifact. That is what refutes the emitter-drop reading; it is **not** by itself
proof that no producer is missing, so the supported claim is "a startup phase, and not evidence of a
recompiler defect", not "not a defect".

**The capture trap this creates, which cost this investigation a night.**
`PROSPER_GPU_CAPTURE_COMPUTE_ADDR` fires on the *first* submit containing the program — inside the
startup window. A capture taken that way has all-empty descriptors, emits almost no memory traffic,
replays with **zero** device losses, and reads as a program that does nothing on purpose. Which is
exactly how the "prosper drops this program's entire memory interface" claim in the Ruled-out section
above was reached. To get a representative submit, enumerate first:

```
PROSPER_GPU_CAPTURE=<path> PROSPER_GPU_CAPTURE_COMPUTE_ADDR=0x413dc6700 \
PROSPER_GPU_CAPTURE_AT=99999 PROSPER_GPU_CAPTURE_LOG=1
```

`AT` beyond every candidate prints `[gpucap] candidate at=N submit=M ...` for each match and captures
none; pick a later `at=` on the next run. **The at->submit mapping is not stable across runs** —
capture overhead shifts it, so the index that coincided with the losing submit in one run will not in
the next (measured: the loss landed on `at=2`'s submit in one run and ~400 submits past the captured
one in the next). The capture is also pre-submit, so it holds the *inputs* to the submit, not its
result.

### What the descriptor shape establishes — and what it does not

Every one of the five buffers carries **2063 records**, and the dispatch is `groups=9x1x1
local=256x1x1` = 2304 threads. **The register→base assignment is NOT stable across folds** — the 264
resolved folds split into two orientations, and only `s4` is fixed:

| SGPR | stride | size | 144 folds | 120 folds |
| --- | ---: | ---: | --- | --- |
| `s4` | 64 | 132032 | `0x209cc76000` | `0x209cc76000` (stable) |
| `s0` | 4 | 8252 | `0x20f848417c` | `0x20f848a240` |
| `s8` | 4 | 8252 | `0x20f848a240` | `0x20f848417c` |
| `s16` | 4 | 8252 | `0x20f8482140` | `0x20f849233c` |
| `s12` | 4 | 8252 | `0x20f849233c` | `0x20f8482140` |

The two input/output pairs **swap** between orientations, and they do so in a fixed repeating pattern —
the resolved folds are one 11-fold block of 6 A and 5 B repeated 24 times, i.e. exactly 24 x (6,5) =
144 / 120. It is not a strict A-B alternation: each period carries one A-A seam.
That is the double-buffering `### The writer is the CONSUMER ITSELF` (above) documents, seen from the
register side. Reading either column as "the" mapping is exactly the cross-dispatch attribution error
that produced the retracted cyclic-table root cause.

What the shape establishes: a **2063-record, per-item read/modify/write or traversal-shaped pass** — one
64-byte record per item read and written, plus two u32 inputs and two u32 outputs per item. It is
*consistent with* a scene traversal / visibility pass, and that is a **hypothesis, not a finding**: the
descriptor shape does not identify the records as entities, does not identify the shader as visibility,
and does not establish that it gates world content.

**Read `## RETRACTED: the cyclic-table root cause` (above) before using any base address in that
table.** Those are the
bases resolved across a *whole run*, not one dispatch's. This program runs many times per submit with
**different tables**, and attributing one dispatch's buffer to another is precisely the error that
produced the retracted cyclic-table root cause. Re-deriving the link graph from `0x20f848417c` and
finding it acyclic reproduces the *succeeding* dispatch's measurement, not the hanging one's.

## `0x413dc6700` has exactly ONE loop, and the runaway exceeds what its data can justify by ~100x

Measured 2026-08-21 from `shader_inspect`'s disassembly of the raw dump. This is **ISA structure**, so
unlike the SPIR-V dissection voided by #2794 it is unaffected by the empty-SRT startup window — the
fold changes which memory ops survive, never which branches exist.

| phase | guest pc | instrs | MUBUF | DS | **back-edges** | fwd branches | blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0..115 | 90 | 5 | 1 | **1** | 6 | 14 |
| 1 | 116..129 | 12 | 0 | 2 | **0** | 1 | 2 |
| 2 | 130..902 | 679 | 36 | 2 | **0** | 38 | 72 |

**The whole program has one back-edge and zero indirect branches.** Three further forms are absent
and were checked by encoding rather than assumed: no SOPP `0x17`–`0x1a` (the CDBG branch family), **no
SOPK at all** (so no `s_call_b64`, no `s_subvector_loop_*`), and no `s_getpc_b64` anywhere — there is
no mechanism in this program to materialise a PC.

The block counts are an independent census (leaders = branch targets + fall-throughs). **They line up
with the emitter's announced ordinal→pc map for phase 0 only** — it prints 15 ordinals, the last being
the empty `pc116..<116` terminator, against 14 leaders. That agreement does **not** extend to the other
phases, and the reason is structural rather than a discrepancy: the emitter's partition always contains
its phase's own first pc (`rdna2_emit_cfg.cpp:1659` asserts it), so it announces **4** ordinals for
phase 1 and **73** for phase 2 against the 2 and 72 leaders below. Phase 0 agrees only because pc 0 is
both the program entry and the phase front.

### Guest loops and dispatcher loops are DIFFERENT OBJECTS — this file has invited the conflation

An earlier line here said the pointer chase "is one of three dispatcher loops in the module" and that
"the other two are unexamined". Those three are **prosper's own**: `rdna2_emit_cfg.cpp` emits one
dispatcher per barrier-delimited phase (`b.cfg_phase_ordinal++`), which is what
`PROSPER_CFG_TRIP_BOUND_PHASE` selects. The guest has **one** loop, in phase 0. So "examine the other
two" is a question about **our lowering**, not about the guest program — and phases 1 and 2 wrap code
with no guest loop at all, which makes any repeated visit there unambiguously ours.

### The loop, and why it cannot self-limit

```
88: v_mov_b32     v2, s22            ; save step count
89: v_cmpx_ne_u32 exec, 0, v1        ; lanes whose link is 0 drop out
90: s_cbranch_execz -> 98            ; the ONLY exit
91: buffer_load_dword v1, v1, s[0:3] idxen
93: s_add_i32     s22, s22, 1        ; incremented, NEVER compared
95: v_bfe_u32     v1, v1, 3, 27      ; next = (word >> 3) & 0x7FFFFFF
97: s_branch -> 88                   ; UNCONDITIONAL back-edge
```

`s22` is a step *count*, not a bound; nothing in the body tests it. EXEC is saved to VCC at pc 85 and
restored at pc 98, so the loop is a standard "chase until every lane hits a zero link" with no trip
limit of its own. **Termination is entirely a property of loaded data.**

### The contradiction

`scan_parent_array`'s link function is `(words[i] >> 3) & 0x7FFFFFFu` — **exactly** the shader's
`v_bfe_u32 v1, v1, 3, 27` (BFE with offset 3, width 27, is that mask). So the offline walk follows the
**same edges** the shader does.

**It does not follow the same DOMAIN, and that is an open fork rather than a detail.** The walk calls a
chain terminating at `i == 0 || i >= records`, with `records = byte_count / 4` — the **host buffer's**
size, not the SRD's `NUM_RECORDS`. So it reproduces the edges and *assumes* the extent. If the
"lanes walk past the classified extent" hypothesis below is right, then the acyclic verdict is itself
an extent artifact and the ceiling below is **void rather than exceeded**. Both readings are live;
what is established is the link function, not the domain.

Taking the acyclic verdict at face value *for the moment*: with longest chain 11 (13–16 in other
samples) and 2 of phase 0's 14 blocks inside the loop, the ceiling is `12 + 2x11 + 1 = 35` dispatcher
trips — the `+1` because the loop's entry ordinal is dispatched once more than the body is traversed.

**Measured: 4,096 trips, on 11 separate dispatches** (`trips=4096 dispatch-range=6..14`, submit 5620,
dispatches 38..42 among them). And 4,096 is the **cap**, not the loop's natural length — the run stopped
there because the bound fired — so it is a **floor**. The excess over the data-implied ceiling is
therefore **at least ~117x**, with no upper figure available from this instrument.

### What the witness can and cannot localise

The extremes are **sound as far as they go**: updated on every back-edge traversal, reading `pc_var`
*before* the `hit` predicate, so the cap's own truncation cannot contaminate them
(`rdna2_emit_cfg.cpp:4918-4925`, with `hit` not computed until `:4932`).

**But `dispatch-range=6..14` establishes less than it appears to, and the reason is the reduction, not
the shape.** The two fields are published with `AtomicUMin`/`AtomicUMax` **across invocations**, so the
range is a *union* over every capped invocation rather than one invocation's itinerary. `min=6` in
particular needs no cycling at all to explain: pc 40's `s_cbranch_execz` jumps straight to ordinal 6,
so an invocation can simply *start* its back-edge history there. Read it as "ordinals 0..5
(guest pc 0..72) were never the target of a back-edge in any capped invocation", which is weaker than
"the runaway is confined to the second half".

**What min/max cannot do at all is localise.** A run concentrating its trips in `{8,9}` and a state
machine genuinely cycling across `6..14` produce an identical record, and those two have different
fixes — the first says our lowering is not honouring the `v_cmpx` → EXEC → `s_cbranch_execz` exit, the
second says our dispatcher revisits blocks the guest does not. Separating them needs a per-ordinal
visit histogram (**#2858**).

### `0x413dc6700` is NOT a shipped fxdb shader (2026-08-21)

Checked against the title's **two** shipped shader archives — `fxdb/sga_prospero_final.awc` (magic
`SGD2`, 43.4 MB; "prospero" is the PS5 codename) and `fxdb/sga_prospero_final_init.awc`, which holds
the 195 `final_init` members and is a *separate* file whose contents are absent from the big one — and
against a 4,864-member extraction of them:

| probe | result |
| --- | --- |
| whole 3,612-byte program, byte-identical | absent |
| first 128 / 64 / 32 bytes | absent |
| the 7-dword loop body (pc 91..97) | absent |
| `v_bfe_u32 v1,v1,3,27` + its MUBUF, as a pair | absent |
| same probes against **both packed archives** | absent |

**The null is controlled, which is the only reason it is worth recording** — and the control that
matters is not the machinery but a **positive instance of the class under test, drawn from a different
source**. That exists here: of 20 independently live-dumped programs, **13 are byte-identical to
archive members** (11 of the 13 that exceed 200 bytes). Same probe, same corpus, a shipped program
found — so the probe can express the case, and `0x413dc6700`'s absence is a property of the subject
rather than of the search.

Supporting controls on the same corpus: 20 real bytes lifted from one member match 1,051 files;
**1,318 of 4,864** members contain a MUBUF at all; the exact dword `0xe0302000` appears in **75**;
`v_bfe_u32` occurs 1,888 times across 1,359 members and its exact dword `0xd5480001` in 83 — yet the
`(3,27)` operand pair is absent everywhere. The size gap (13.2 MB extracted against 43.4 MB packed) is
metadata being dropped, not members being missed: every sampled `final` and `ptfx_sprite` member is
present verbatim in the packed file (250/250 and 95/95).

Two consequences. **The archive is a working naming oracle**, on 13 exact matches rather than one —
a live-dumped lighting program resolves to `s5_182_raytraced_lighting_CS_RaytraceReflectionLightPass`.
And `0x413dc6700` is **not** among them, nor are any of the eight `lane-*` traversal-table writers.

**The obvious objection — that a shipped shader could be relocated, patched or specialised at load
time, so absence proves nothing — is answered empirically rather than dismissed.** For 11 of the 13
matches the archived and runtime bytes are *identical*, so this title does not transform programs as a
rule. The two residuals separate the failure modes and place `0x413dc6700` in the harmless one: one
live program matches a member for 128 bytes and then diverges (transformation does happen, and it
leaves a matching **prefix**), while `0x413dc6700` has no head match at **any** length. A
shipped-then-diverged program would still match its prologue; this one never does.

Byte comparison is valid because members begin directly with RDNA2 ISA — the `_Wrapped` in their names
is part of the shader's name, not a container. Corpus-wide, 4,809 of 4,864 members open with a SOPP
dword, there are only 16 distinct first dwords, and **4,627 open with `0xbfa00003`, which is this
program's own first dword** — so a differing prologue cannot explain the miss.

### Also settled here

- **Both `s_barrier`s (pc 116, 129) are reached unconditionally by every wave.** No forward branch
  skips either, and no earlier branch exits to `endpgm` before them, so they are uniform — consistent
  with the barrier lever having been inert (trap 164).
- **#2542's positive control passes but does not cover this.** It shows per-lane EXEC narrowing and
  the cross-lane `execz` vote are correct — with a body that "decrements the index instead of chasing
  a buffer". So it exercises the exit *mechanism* and never the loaded value the exit *depends on*.
  A candidate mechanism that fits every observation without requiring corruption — lanes walking past
  the extent `scan_parent_array` classified, so no zero link is ever met — is recorded as a
  **hypothesis** on #2858, along with the `PROSPER_DYNTRACE_ADDR` check that would settle it before
  anyone touches the recompiler.

## Other open defects

- **Prologue cutscene timeline/animation desynchronization at low framerates**: In the opening story mode
  prologue cutscene (Bank Heist), the scripted sequence consists of six sequential camera angles:
  1) Robber throws a woman to the floor, camera pans down with her on the floor;
  2) Camera switches angle, looking up from the floor at the robbers;
  3) Close-up shot on one robber's face speaking;
  4) Zoomed-out shot, robber hits the teller window;
  5) Close-up shot on the robber ordering the clerk to open the door;
  6) Camera zooms out, robber kicks the security door open.
  At sub-10 FPS throughput, RAGE cutscene progression (advancing against real process time via
  `sceKernelGetProcessTimeCounter`) desynchronizes from rendered animation states: the presentation
  immediately skips the first five camera angles and jumps directly to the robber kicking the door open,
  with the woman already lying static on the floor. Once throughput reaches real-time targets, animation
  event triggers align with the script timeline.
- **#2445** — specific lowercase glyphs (`r`, `s`, `m`) dropped from UI text: "Ente ing Sto y Mode".
  Surrounding text is intact, so it is per-glyph, not a font failure.
- **#2429** — the world cannot render on 32-wide devices: the EXEC-population-count fix requires
  `native_subgroup_size == wave_size == 64`.
- **#2428** (Windows) — frame-rate cliffs of ~60x (62 fps to 0.8 fps) within 60 frames at the gameplay
  transition.
- **#2424** (Windows) — `sceKernelBatchMap` `MAP_DIRECT` fails `ENOMEM` on a
  map-small/unmap/map-larger cycle. No Linux equivalent.

## Instruments worth knowing about here

- **`[buf-op]` (`PROSPER_DBG`)** — the disposition of every buffer op the MUBUF/MTBUF handler sees:
  `[buf-op] program=0x... pc=91 MUBUF op=0xc n=1 store=0 atomic=0 rt=1 zero-record`. It exists because
  a **legal `NUM_RECORDS=0` fold and an instruction that never reached the emitter leave identical
  evidence** in emitted SPIR-V — no access chain, no load, no trace — and telling them apart decides
  whether a program with no memory traffic is a resource-state question or a translation bug. Reported
  from a scope guard, so **every** exit path emits exactly one line and an unlabelled path prints
  `unclassified` rather than nothing; that is what lets a census detect its own incompleteness. `rt=0`
  marks `recompile_coverage()`'s table-less shell (it runs on the failed-draw path and at F9 capture),
  so a genuine hole is `rt=1 ... unclassified`. Four rejects sit above the guard by design — `lds`,
  `tfe`, MTBUF D16, and the opcode switch's `default:` — so "instructions in the stream == lines
  emitted" is not an identity; it happened to hold for `0x413dc6700` (41 == 41). #2712.
- **`[dyntrace]` + `PROSPER_DYNTRACE_ADDR` / `_ONCE`** — the live V# each MUBUF resolves, at fold time.
  This is the only way to see what the front half actually built: the fold is **not** re-run during
  offline replay (the resource table is replayed from the capture, so markers are baked in), so a
  question about descriptor resolution cannot be answered from a `.prgcap` alone. Use `_ONCE` for one
  fold, and **drop it when the question is about variation across dispatches** — a single traced fold
  told this investigation all five pointers were null, which was true of that fold and false of 75% of
  the run.
- **`PROSPER_COMPUTE_SKIP_PROGRAM=0xADDR[,...]`** — decline named compute programs. A bisection tool,
  not a workaround. It announces itself at parse time and reports each skip through the decline census
  as `reason=skipped-by-selector`, so a diagnostic run can never be mistaken for a default one later.
  Ordered **after** the trace and SPIR-V dump, so "dump the module, skip the dispatch" works — which
  is how a recompiler change can be checked against a program that hangs the GPU, at zero device cost.
- **`[compute-decline]`** — every refusal in `execute_item` now names its reason, the dispatch, and a
  **running count**. The count is in the line deliberately: rate-limited diagnostics in this codebase
  have twice been read as censuses.
- **`PROSPER_SUBGROUP_LOG`** — the resolved native-subgroup contract. Read it as an *input*: it says
  what the config resolved to, not what the emitter produced.
- **`RADV_DEBUG=hang`** — the vendor tool, no prosper change needed. Writes a report to
  `~/radv_dumps_<pid>_<ts>` with `vm_fault.log`, `trace.log` (the last CP trace point) and
  `pipeline.log` (shader stats for the hung pipeline). `umr` is not installed here, so per-wave PCs
  are unavailable, and `DISASM` needs an additional RADV flag.

## Shared-GPU policy on this title

Every device-loss experiment here costs a hard recovery on a machine other agents are using. Allow at
most **one expected device loss per hypothesis**, stop that process immediately after the loss, and
investigate before the next routed run. `PROSPER_COMPUTE_SKIP_PROGRAM` plus the SPIR-V dump removes
the need for most of them.
