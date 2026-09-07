# Graphics pipeline and RTT investigation (2026-09-07)

Issues: [#3378](https://github.com/mattias800/prosper/issues/3378),
[#3419](https://github.com/mattias800/prosper/issues/3419),
[#3407](https://github.com/mattias800/prosper/issues/3407).
Baseline: `88aba88c1` (#3418), Linux RADV STRIX_HALO, Vulkan 1.4.

## Graphics pipelines

The renderer now supplies a device-lifetime `VkPipelineCache` to graphics pipeline creation.
This is separate from prosper's bounded map of existing `VkPipeline` handles. The existing
backend resource lock serializes access; failure to create the driver cache retains uncached
pipeline creation. No driver blob is loaded from disk. Cross-run persistence remains deferred:
compute's opt-in persistence has a documented NVIDIA driver crash, and the interactive frontend
uses `_Exit` rather than normal destruction. This change does not promise cross-run reuse.

Depth test/write/compare, stencil enable and operations, culling/winding, topology and primitive
restart are recorded dynamically for every draw, including the draw-isolation diagnostic. The
pipeline key retains attachment formats and topology **class**. A depth-only attachment and a
depth/stencil attachment remain different contracts. Line and triangle pipelines remain distinct.

The commands are required by [core Vulkan 1.3](https://docs.vulkan.org/refpages/latest/refpages/source/VK_VERSION_1_3.html),
below the runtime's 1.4 floor; no experimental driver setting or optional dynamic-state extension
is required. Unrestricted changes between topology classes are deliberately not assumed.

### Joe & Mac: the stutter hypothesis remains unproved

Two native-1080p, windowed immediate-present runs used `PROSPER_FLIP_PACE_FPS=60`, separate saves,
the recorded level-1 route, a frame-3600 screenshot, and an F8 trigger after 85 seconds. Both
screenshots show the level-1 opening correctly. Neither run had concurrent compiler/GPU load.

The five-second post-trigger renderer records were:

| Metric | Baseline | Driver cache + dynamic state |
| --- | ---: | ---: |
| Renderer callbacks | 308 | 314 |
| Pipeline references | 26,488 | 27,632 |
| Pipeline misses | 6 | 4 |
| Evictions | 0 | 0 |
| Pipeline entries during capture | 192–198 | 192–196 |
| Largest pipeline setup step | 2.242 ms | 2.107 ms |
| Largest renderer callback | 5.702 ms | 5.386 ms |

Pipeline setup includes key lookup as well as creation. These single runs do not establish a
stutter cure or a material FPS improvement. The very low warm miss rate also means most rendering
time cannot be removed by either pipeline-cache change. An earlier attempt omitted the `@` in the
pad-script setting and remained at the title screen; its 53-entry cache is **not gameplay evidence**.

## Sonic: preserve integer values without a CPU RTT round trip

The remaining hot sampled-image preparation in the baseline was the generic copy program's
4K RGBA8 renderer targets sampled as `Uint8x4`. The CPU path copies their bytes unchanged. An
80-second diagnostic run performed 2,517 4K sampled preparations, totaling 4.980 seconds in
`prepare_ms`; this count also includes depth imports whose preparation time is zero.

The #3407 WIP admitted the UNORM/UINT pair but still selected the imported image's UNORM format
for its view. Its performance counters did not prove correct sampling. The new path instead
copies a pinned renderer RGBA8 UNORM image into a distinct RGBA8 UINT image before dispatch.
[Vulkan image copies permit size-compatible color formats](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html).
The UINT view consequently supplies integer channel values, matching the working CPU route.

Eligibility requires the same device, exact dimensions, a single 2D image, reflected unsigned
sampling, and explicit transfer-source usage. Other formats and incompatible imports retain the
existing path. The source layout is restored, including when another descriptor borrows that same
image in GENERAL. Each acquired pin is released, including folded aliases. No host staging buffer
or lazy CPU snapshot is needed for an eligible copy.

The live Sonic run confirmed `color-bits-copy=1` for both hot RGBA8/UINT inputs. In the
five-second F8 windows, mean compute setup fell from 1.040 ms over 791 records to 0.432 ms over
798 records. This is one pair of startup captures, not a matched whole-game benchmark or evidence
that audio underruns are fixed. Sonic's title screen has longstanding rendering defects and is
not used as a correctness oracle for this change.

This addresses the measured renderer-owned RGBA8/UINT materialization. It is not a general GPU
detiler: guest-backed tiling, other numeric conversions, and compute guest writeback still exist.
The WIP's CPU conversion cache was not reused; its recorded zero hit rate followed from changing
renderer publications rather than an unwired lookup.

## Reproducing the measurements and guards

The renderer/compute integration test checks every byte value in every channel, repeated updates
at the same image identity, duplicate UINT descriptors, both orderings of mixed UNORM/UINT
descriptors, source preservation, balanced pins, and CPU fallback for missing transfer-source
usage or mismatched extents. Restoring a UNORM destination makes its value assertions fail and
triggers `VUID-vkCmdDispatch-format-07753`. Separately, removing the driver cache makes the
graphics-cache guard fail; putting depth compare back into the pipeline key makes the dynamic
state reuse guard fail. All three mutations were restored before final validation.

The complete build and all 370 configured tests passed. The full Vulkan validation scanner then
passed all 370 tests again with synchronization validation armed by its deliberate-hazard probe.
It found only the four existing, test-scoped allowlisted IDs associated with #1710, #1715 and
#1716 (71 messages); no new validation IDs or synchronization hazards appeared. A prior scan of
a run using only `PROSPER_VK_VALIDATION=1` lacked the legacy fixtures' required validation
findings, so it was not accepted as the full validation guard. The scanner invocation below
enables the layer globally and supplies the verified result.

Build `prosper-app`, `test_pipeline_render`, `test_multidraw_render`, `test_renderer_uint_copy`,
`test_render_diagnostic_paths`, and `test_game_compute` with the normal Vulkan-enabled build.

```sh
PROSPER_VK_VALIDATION=1 ctest --test-dir <BUILD> --no-tests=error -V \
  -R '^(pipeline_render|multidraw_render|renderer_uint_copy|render_diagnostic_paths|game_compute_exec)$'

python3 prosper/tools/vkval/vk_validation_scan.py --build-dir <BUILD> --sync

PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_FLIP_PACE_FPS=60 PROSPER_RENDER_TIMING=1 \
PROSPER_PAD_SCRIPT=@prosper/scripts/joe-mac/reach-gameplay.pad \
PROSPER_SAVE0=<FRESH_SAVE> PROSPER_CAPTURE_DIR=<EVIDENCE> \
PROSPER_CAPTURE_SCREENSHOT_AT_FRAME=3600 PROSPER_CAPTURE_SCREENSHOT=<EVIDENCE>/frame.bmp \
PROSPER_PERF_CAPTURE_AFTER_MS=85000 \
timeout -k 8s -s INT 130s <BUILD>/prosper-app \
  --dump <DUMP_ROOT>/PPSA02801-app0 --fps --present-mode immediate
```

For the Sonic preparation census, use `PPSA03831-app0`, omit the flip cap, enable
`PROSPER_COMPUTE_IMAGE_TIMING=1`, and trigger F8 at 55 seconds. The `.prperf` artifact is JSONL;
sum `backend_pipeline_*` on `renderer` records, and preserve the header/footer to verify revision,
capture duration and dropped records. The per-image log reports `color-bits-copy=1` for the new
device copy. A configured screenshot path is not evidence that a screenshot was written: open it.
