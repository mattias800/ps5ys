# `tools/vkval` — the Vulkan validation guard

Runs prosper's ctest suite under `VK_LAYER_KHRONOS_validation` and fails when a validation message
appears that is not written down in `allowlist.txt`.

## Why

#1690 was a `VK_IMAGE_VIEW_TYPE_3D` view bound to a descriptor whose SPIR-V module declared
`OpTypeImage ... 2D`. That is undefined behaviour, and three conformant drivers resolved it three
ways — two returned the seeded texel, one returned zero. It presented as a Mesa *version*
regression, was attributed to an upstream driver defect, cost `game_compute_exec` its place on CI
(484 executed assertions), and was only settled by disassembling the SPIR-V by hand.

The validation layer names that defect at the dispatch, on the first run:

```
VUID-vkCmdDispatch-viewType-07752
vkCmdDispatch(): the storage image descriptor [VkDescriptorSet ..., Set 0, Binding 5, Index 0]
VkImageViewType is VK_IMAGE_VIEW_TYPE_3D but the OpTypeImage has (Dim = 2D) and (Arrayed = 0).
Either fix in shader or update the VkImageViewType to VK_IMAGE_VIEW_TYPE_2D.
```

That is the difference the guard buys.

## Synchronization validation — `--sync`

**It is a separate check set, and the default run does not include it.** Core validation checks how
each call is *used*; synchronization validation models the hazards *between* calls — the barriers,
the subpass dependencies, the layout transitions — and it is what sees prosper's synchronization.
Add `--sync` to switch it on:

```bash
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux --sync
```

Until #3248 it ran nowhere, which is why five real hazards had accumulated unseen. All five are
fixed; a `--sync` run of the whole suite is clean, and any new one fails the gate like any other id.

**It has its own positive control, because it has its own way of silently not running.** Syncval is
switched on by a layer SETTING, not by loading anything extra, so a spelling this layer version does
not recognise is ignored — and the layer still loads, so the scanner's existing loader proof says
nothing about it. A clean sync verdict would then be indistinguishable from a check set that never
armed. `tools/vkval/sync_probe.cpp` (target `vkval_sync_probe`) therefore commits **one deliberate
write-after-write**, and `--sync` refuses to report anything until that hazard comes back. Two
spellings are tried, one at a time — `VK_LAYER_VALIDATE_SYNC` (current) and `VK_LAYER_ENABLES`
(deprecated) — and the first that arms is the one the run uses. They must not be set together:
layers 1.4.341 answers that with `Validation Warning: [ VALIDATION-SETTINGS ]`, which this scanner's
own parser then reads as an unaccounted-for finding.

`vkval_sync_probe` is deliberately **not** a ctest: it exists to be invalid.

### What it costs

Measured 2026-09-02, Fedora 44 / layers 1.4.341 / Mesa RADV STRIX_HALO, whole ctest suite:
**112.09 s core-only against 113.58 s with syncval**, 343 tests either way. That is 1.4%, and it is
inside this box's noise — the same suite re-run under other lanes' load moved individual tests by
±7 s, and the one test that looked expensive (`rdna2_to_spirv_exec`, +11 s in the first pair)
measured 18.5/31.8/19.1 s core against 46.4/16.0/12.5 s sync over three alternating runs, i.e. the
two ranges overlap completely. **So the honest statement is that no syncval cost is separable from
machine load here, not that it is free.** The suite is dominated by CPU-side unit tests; a workload
that records far more GPU commands per second could pay more.

### What it still cannot see

Syncval is not a general oracle for synchronization. It cannot observe a CPU read through a mapped
pointer, so it does not see a missing host-availability barrier on a readback (#2944): measured with
that defect deliberately live, it produced zero messages while demonstrably armed (the same setting
over the whole suite produced 5 hazards at the time). `docs/GRAPHICS.md` § Ruled out carries both
halves of that measurement. Nor does it see a path no test executes — which is a second blind spot
`--sync` does not fix on its own, and is why `render_diagnostic_paths` exists (below).

### Why it is not on CI yet — measured, not assumed

**The driver is not a variable.** RADV (STRIX_HALO) and lavapipe report the identical hazard set on
layers 1.4.341: same ids, same counts, same tests. That is expected — syncval models the recorded
commands rather than consulting the driver, the same reason "try another GPU" cannot discriminate the
`08600` class described further down.

**The LAYER VERSION is.** Built and scanned inside `podman run ubuntu:24.04` with the runner's own
packages (`vulkan-validationlayers 1.3.275.0`, `mesa-vulkan-drivers 25.2.8`, i.e. lavapipe), a
`--sync` run of this tree reports a **sixth** hazard that 1.4.341 does not report at all:

```
SYNC-HAZARD-WRITE-AFTER-READ  x4  [shadow_compare_render]
vkCmdEndRenderPass():  Hazard WRITE_AFTER_READ in subpass 0 for attachment 1 depth aspect during
store with storeOp VK_ATTACHMENT_STORE_OP_STORE. (usage: SYNC_LATE_FRAGMENT_TESTS_DEPTH_STENCIL_
ATTACHMENT_WRITE, prior_usage: SYNC_FRAGMENT_SHADER_SHADER_SAMPLED_READ, ...)
```

The driver is held constant across that comparison — lavapipe both times — so the variable is
isolated to the layer version. The path it names is the same-pass depth feedback of #1186, which
`render_runner.h` enables **only** when no draw in the pass can write depth (`depth_may_be_written`)
and which puts the attachment in a depth-read-only layout, so the depth aspect the store op names is
not writable. Whether 1.3.275 or 1.4.341 is right about that is not established here, and it is not
this guard's place to guess: switching the shared gate on today would either redden every concurrent
lane's PR or require an allow-list entry for a finding nobody has settled. `--sync` is therefore
opt-in, and the enablement is tracked with that question in **#3255**.

## What it does NOT cover

**A path no test executes.** The guard observes ctest, so code that only runs under an environment
variable is invisible to it however it is configured. `PROSPER_GEOM_PROBE` and `PROSPER_DRAW_ISO`
were both misusing Vulkan for exactly this reason (#3248); the `render_diagnostic_paths` ctest case
now runs them once so the layer sees them. **A new env-gated render path needs a line there in the
change that adds it**, or this guard cannot cover it.

## Running it

```bash
# gate (what CI runs): fail on any message id not in allowlist.txt, or any test that fails
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux

# measure only: print the grouped findings, never fail
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux --report-only

# re-parse a log you already have
python3 tools/vkval/vk_validation_scan.py --log prosper/build-linux/Testing/Temporary/LastTest.log

# one test only, while iterating on a fix (~40 s instead of the full suite)
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
    --report-only --ctest-arg=-R --ctest-arg=descriptor_array_render
```

`--ctest-arg=-R`, not `--ctest-arg -R` — argparse consumes the bare flag as its own and exits 2.

**A focused run ends in a `FAIL` that is an artifact of the filter, not a regression.** Restricting the
test set means the tests that produce the other allow-listed messages never execute, so the scan
reports them missing:

```
[vkval] FAIL: 4 allow-listed id(s) marked `required` produced no messages, while the layer
        demonstrably loaded.
```

That is expected under `-R` and appears on a completely healthy tree. The full scan has no such line.
Read it as "you filtered the suite", and take only the `NEW — not on the allow-list` section as a
finding from a focused run.

The layer package is `vulkan-validationlayers` on Debian/Ubuntu and `vulkan-validation-layers` on
Fedora. **Build in an environment that actually has Vulkan** — a configure that quietly loses it
drops every Vulkan execution test and the scan then observes nothing (#1675). On this project's
Bazzite box that means the `ps5ys` distrobox, not the host.

## When to run this deliberately

CI runs the gate, so most changes need nothing. **Run it yourself, before opening the PR, for any
change touching descriptor counts, descriptor-set layouts, or pipeline keys.**

That trigger list is the shape of the one defect this guard has caught **during review, with every
other check on the PR already green** — which is a narrower claim than "the one defect it has caught",
and the narrowness is the point. The defects catalogued in `allowlist.txt` are also findings this
guard surfaced; fixed entries are removed as their errors disappear. The remaining entries cover
#1710, #1715, and #1716. #2471 is different in kind: it arrived *inside*
a change under active review. That is what justifies running it yourself rather than trusting CI.

**So treat the list as a floor, not a ceiling.** It is derived from a single case, and the
allow-listed IDs are evidence to examine for what else belongs on it. Start here; do not read it
as complete.

The case, and why nothing else could see it:

#2471 bound a 3-entry descriptor array whose `VkPipeline` had been created under an **arity-1**
`VkPipelineLayout` and was replayed with the **arity-3** layout bound
(`VUID-vkCmdDrawIndexed-None-08600`, *"Set 0 binding 0 descriptorCount 3 doesn't match 1"*). Two
independent blind spots hid it, and between them they cover every check a normal PR gets:

- **A pixel assertion cannot see it.** The stale pipeline still draws a correct green quad, so every
  pixel arm passed. Confirmed by mutation: reverting the fix returns the VUID byte-identically **while
  the test still passes**. Author verification, author re-reading, and independent review all went
  green on a spec-invalid draw.
- **On Windows the job does not exist.** `test_descriptor_array_render` is triple-gated in
  `CMakeLists.txt` — `if(TARGET prosper_core)` → `if(UNIX)` → `if(Vulkan_FOUND)` — so the whole
  Vulkan-execution render suite (`multidraw_render`, `indexed_render`, `texture_sample_render` too) is
  never configured there. `UNIX` is the operative gate: Vulkan **is** found in Windows build caches, so
  installing the SDK changes nothing.

And the reflex that does not help: a second **driver** cannot discriminate this class. `08600` is a
**usage** error — the layer compares the pipeline's own layout against the layout the descriptor sets
were bound with and reports a mismatch, without consulting the driver at all. So there is no driver
behaviour to differ, and "try it on another GPU" is wasted effort here.

**So `ctest` green is not evidence about spec validity, on any platform.** This scan is the only check
in the repo that sees this class. Instrument-trap 151 in `docs/GAME_COMPAT_ORCHESTRATION.md` records
the whole case.

## The instrument trap this tool is built around

A validation layer that fails to load produces *exactly* the same output as a clean run: nothing.
So does a correctly loaded layer whose `report_flags` are misconfigured — the layer's default is
`error` only, and a probe that asks for best-practices warnings with the wrong setting key prints
nothing and reads as success. Both were hit while writing this.

`vk_validation_scan.py` therefore refuses to report a clean verdict until it has watched the loader
say `Insert instance layer "VK_LAYER_KHRONOS_validation"` (via `VK_LOADER_DEBUG=layer`) on a probe
binary. If the layer is missing, the scan fails with an install hint rather than passing.

That check alone was not enough, and the way it failed is worth keeping. **The two layer versions
in play frame the same message differently:**

```
layers 1.4.341 (Fedora 44)    Validation Error: [ VUID-x ] | MessageID = 0x...
                              <message text on following lines>

layers 1.3.275 (Ubuntu 24.04, VUID-x(ERROR / SPEC): msgNum: N - Validation Error: [ VUID-x ] ...
                the CI runner)  <everything on one line>
```

The first revision of this parser anchored its pattern to the start of a line. It read all 51
messages on 1.4.341 and **0 of 187** on 1.3.275 — the layer loaded, the probe passed, the scan
reported a clean suite, and the CI gate would have been permanently green while observing nothing.
It was caught only by running the positive control (below) on the CI environment rather than the
development one.

So the scan has a second instrument check, and it is deliberately per-entry rather than
all-or-nothing: **every allow-list entry marked `required` must still produce messages.** If one
stops appearing, exactly two things can have happened — the defect was fixed (delete the line, in
the same change) or the scan stopped seeing what it used to. Both are failures until someone says
which.

Checking each id separately is what catches a *partial* break. An all-absent check alone passes a
VUID rename, or a framing change that affects some message shapes and not others, while silently
halving what the guard can see. The id that genuinely cannot appear everywhere is marked
`environment-dependent` and says why on its reason line, so the exemption is written down rather
than assumed:

* `VUID-vkCmdDispatch-maintenance4-08602` is a newer check than validation layers 1.3.275, which is
  what Ubuntu 24.04 (the CI runner) carries.

Deleting the last allow-list entry — the intended end state — switches the check off.

`test_vk_validation_scan.py` (ctest `vkval_scan_logic`) pins both layer framings verbatim, pins the
`VUID-vkCmdDispatch-viewType-07752` line that motivated all of this, and asserts that the test's own
output contains nothing the scan would read as a validation message. That last one is not paranoia:
ctest captures this test's stdout into the same log the scan parses, so a test that prints sample
messages makes the guard read its own tail — measured, it did.

Smaller closures in the same spirit: a **missing** allow-list file is a hard error rather than an
empty one, and ctest runs with `--no-tests=error`, so "build directory with nothing registered"
cannot report success either.

## `allowlist.txt`

```
<message id> | <required|environment-dependent> | <tests, or *> | <reason, with its issue>
```

Every message id observed when the guard was switched on is listed there. A line the scanner cannot
fully parse is a hard error — the point of the file is that pre-existing findings are *visibly
deferred*, never silently tolerated. Fixing a defect means deleting its line.

**The test list is not decoration.** Several VUIDs are catch-alls —
`VUID-VkShaderModuleCreateInfo-pCode-08737` is VVL's identity for *any* `spirv-val` error at
`vkCreateShaderModule` — so an id-only ledger would defer every future SPIR-V validity defect
anywhere in the suite behind one known emitter bug. An allow-listed id arriving from a test the
ledger does not record fails the scan: a deferral is scoped to where it was measured.

## Cost

Measured on the real GitHub runner (`actions/runs/30719139452`, Linux job): ctest reports
**30.46 s** without the layer and **31.08 s** with it, 168/168 passing either way, so the scan step
costs **31 s** of wall clock on a job that spends minutes compiling. Cheap enough to run on every PR
rather than on a schedule.

The same suite in `podman run --rm ubuntu:24.04` on a faster machine takes 17.0-24.0 s plain and
33.2-34.0 s under the layer, i.e. the layer's *relative* cost is larger the faster the box. Neither
figure is close to needing a dedicated job.

## Reproducing the CI environment locally

```bash
podman run --rm -v "$PWD:/work" ubuntu:24.04     # mesa-vulkan-drivers 25.2.8, vulkan-validationlayers 1.3.275.0
```

Both packages come from the Ubuntu archive at job time rather than from a pin, so check
`apt-cache policy` if the observed set ever shifts under you.

## What this guard does NOT cover

**Its self-validation is borrowed from the defect backlog, and that backlog is meant to shrink.**
The `required` check proves the parser still reads this layer version's output *because* known
defects are still firing. As #1710-#1717 are fixed and their lines deleted, that proof weakens; once
only `environment-dependent` entries remain, a run on the CI driver observes nothing legitimately,
and a parser that quietly stopped matching would again be invisible. Whoever deletes the last
`required` entry is switching that off, and should replace it first — the durable form is a positive
control built into the probe (provoke one known violation and require the scanner to *parse* it),
which proves the parser rather than only the loader. Tracked in **#1725**.

`#1704` also fixed a Vulkan-teardown-from-a-static-destructor defect in
`frontends/shared/live/live_compute.cpp` that the layer exposed. **CI would not catch a regression of
it.** That crash reproduces on validation layers 1.4.341; the runner carries 1.3.275, where the
suite passes with the defect present. Reverting the lifetime change would leave this step green.
Running the guard on a recent layer version locally is what covers it — one more reason not to treat
a green CI run of this step as "no Vulkan misuse anywhere".
