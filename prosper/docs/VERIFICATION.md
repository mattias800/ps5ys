# Verification strategy (automated gates, reviewed baselines)

Every milestone is gated by a self-checking test whose **exit code is the truth**. Routine checks do
not require a person to operate a game window or compare images. Baselines are different: a new or
changed real-game baseline must be visually inspected once so an automated check cannot bless a black
frame, logo, menu, or wrong scene. The graphics path is verified in layers, cheapest first.

## 1. Structural assertions (no GPU) — the bulk of correctness

Each pipeline stage is a pure function tested with exact-value assertions against primary evidence or
an independently generated oracle. This catches most bugs deterministically and instantly:

| Stage | Test | Reference |
|-------|------|-----------|
| PM4 decode | `test_pm4_decode` | packets built by the real AGC Dcb functions |
| CommandProcessor apply | `test_command_processor` | captured packet streams and exact expected folds |
| RenderState extract | `test_render_state` | AMD register definitions plus captured state |
| RDNA2→Vulkan translate | `test_render_state` | Vulkan specification and reviewed translation tables |
| RDNA2 decode + operands | `test_rdna2_decode` | **`llvm-mc -mcpu=gfx1030` encodings** |

Using `llvm-mc` to assemble authoritative RDNA2 encodings (and `glslangValidator` for GLSL→SPIR-V)
means the test inputs are ground truth, not our own assumptions.

### Host sanitizer checks

For memory and concurrency diagnostics, the standalone [host sanitizer suite](DEBUGGING_WORKFLOWS.md#isolated-host-sanitizer-checks)
builds SELF parsing, RDNA2 decoding, BC decoding and submit-gate tests directly against production
sources. ASan/UBSan and TSan configurations include clean and deliberately faulty instrument controls;
a missing runtime or initialization failure cannot count as bug detection. The `host-sanitizers`
and `host-tsan` CI jobs run those two configurations. This is focused host coverage, not sanitized
guest execution.

## 2. Render → readback → pixel/region assertions

`test_vulkan_triangle` renders offscreen on **llvmpipe** (deterministic software rasterizer),
copies the framebuffer to host memory, and asserts specific pixels/regions
(`center == red triangle`, `corner == blue clear`). No window, no eyeballing.

## 3. Golden-hash regression gate

The same test hashes the whole framebuffer (CRC32) and compares to a stored golden value. Any change
that flips a single pixel fails the test. llvmpipe is deterministic, so the hash is stable in CI.
`PROSPER_DUMP_PPM=1` writes `triangle.ppm` for *optional* human/debug inspection — never required.
(On a hardware GPU the raster may differ; the golden gate assumes the CI software rasterizer, while
the region checks in layer 2 stay portable.)

## 4. Execution-differential for the shader recompiler (LIVE)

This is how a recompiled shader is verified **semantically** without a PS5, fully automated:

```
llvm-mc assembles a known RDNA2 snippet  ->  our recompiler emits SPIR-V  ->
run it on Vulkan with known register inputs  ->  read back the result  ->
assert it equals the instruction's math   (e.g. v_add_f32(2.0, 3.0) == 5.0)
```

It proves the recompiler is *correct*, not merely structurally plausible, and every new opcode adds
one more cheap numeric assertion. Implemented as `test_rdna2_to_spirv` — 19 kernels covering
float/int/scalar ALU, convert/compare/select, bitfield, `pkrtz` (bit-exact f16), and **divergent
control flow** (EXEC per-lane predication, `saveexec`/restore, `s_cbranch_execz` if-then). The whole
`GpuState → recompiled shaders → VkPipeline → frame` spine is likewise pixel-verified
(`test_gpustate_render`, `test_pipeline_render`: topology/blend/depth/write-mask each driven from real
registers and asserted by readback).

## 4b. Recompiler coverage metric (data-driven roadmap)

`recompile_coverage()` reports per-instruction recompiler support without needing a complete
vertex/fragment; `shader_histo` runs it over the game's 41 **real** embedded shaders and prints
coverage % + the top first-unsupported opcodes. This turns "what to build next" into data (currently
~67% of instructions; the dominant remaining blocker is `SMEM`), and is regression-tested pure
(`test_recompile_coverage`).

## 4c. Strict SPIR-V validation of every emitter (`spv_validate`)

Drivers are not required to validate SPIR-V, and the two prosper is developed against do not: llvmpipe
and RADV both accepted a module whose `OpAccessChain` result type disagreed with the type it walked.
So acceptance by a driver proves nothing about validity, and `tools/spv_validate` is the gate that
does. It emits one representative module from **every** SPIR-V-producing entry point declared in
any header under `src/gpu/` or `frontends/shared/`, recursively, under either return-type spelling
the tree uses (`std::vector<uint32_t>` and the live path's `SharedShaderWords`), and runs
`spirv-val --target-env vulkan1.1` on each.

Scope, stated precisely because it was previously overstated: this is *per emitter path*, not per
shader a title submits. A game's shader is covered to the extent that it exercises paths the corpus
exercises; the recompiler's execution-differential tests (§4) and the Vulkan validation layer are what
cover the rest.

Three failure modes of the gate itself are closed, all of them found by #1711:

* **A missing validator is a failure, not a pass.** It used to print
  `== PASS (recompiled; spirv-val not found) ==` and exit 0. `spirv-val` is on no GitHub runner image
  and the workflow never installed it, so the gate had validated nothing in CI for its entire life
  while three documents described prosper's shaders as strictly `spirv-val`-gated. CI now installs
  `spirv-tools` on all three test platforms, and removing that breaks the job.
* **An uncovered emitter is a failure.** The corpus only ever walked `recompile_*`, so the
  hand-assembled compute modules in `spirv_builder.cpp` — created at runtime by the live frontend
  exactly like a recompiled shader — were never validated. The tool now scans those headers for
  declarations returning SPIR-V words and fails on any that did not produce a validated module. Coverage is measured at **runtime**, where each module is emitted, not by
  grepping the tool's own source for a call: a textual check is satisfied by a mention in a comment
  or by a call whose result is discarded, and a check a comment can pass is the defect this gate
  exists to stop. A name that is not an emitter (an RDNA2→RDNA2 transform, a caching wrapper) must
  be listed in `kNotEmitters` with a reason, so an unclassified new name fails rather than being
  skipped. A genuine gap is permitted only as a written `kKnownGaps` entry naming its issue, and a
  stale entry — the emitter is covered again, or no longer declared — also fails.
* **`spirv-val`'s diagnostic is printed.** A failure used to say only "REJECTED it".

## 5. Routed multi-frame guards for real games

`tools/snapshot/snapshot.py` boots local game dumps through the normal presented-screenshot frontend,
can replay a title-specific input route from a fresh temporary save, and evaluates several composited
frames from a gameplay-only time window. Threaded games rarely select one deterministic frame, and subtle pixel changes may be
valid improvements, so gameplay guards use coarse contracts: multiple frames must exceed a reviewed
scene-richness threshold, several frames must meet an SSIM threshold against visually reviewed 16x9
luminance reference states, non-black coverage and dimensions must remain correct, and moving routes can require visible pixel
changes. This catches black output, missing major layers, stalled routes, and presentation collapse
without making every pixel part of the API. Exact CRCs identify samples for debugging but do not reject
subtle changes in timing-sensitive gameplay.

Exact frame hashes remain available for checkpoints proven deterministic by two independent captures.
For either mode, `snapshot.py verify NAME` retains images from two runs under
`tools/snapshot/review/NAME/`. A person must inspect every retained image and confirm the intended state,
layers, and progression before recording a new hash or threshold. Routine `snapshot.py check` runs are
then automated. Dumps and evidence images are local and gitignored, so real-game guards do not run in CI.
See `tools/snapshot/README.md` for the complete approval workflow.

## CI

A GitHub Actions workflow builds + runs `ctest` on **Linux and Windows/MinGW** for every push/PR.

**Linux CI runs the Vulkan-execution layers** on Mesa's lavapipe software rasteriser (#1675). It did not
until then: no Vulkan package was ever installed on the runner, so `-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan`
was describing that fact rather than deciding anything — and it silently removed **27 tests / ~1,800
assertions** from every green run, including the entire shader-recompiler differential and the sole
coverage for GPU execution, render state and present. An unregistered test is invisible by construction:
there is no line to read and no count to compare against, so 138/138 looked exactly like full coverage.

Two guards keep that from recurring, and both matter more than the flag itself:

* the Linux job **fails if too few tests register**, so a configure that quietly loses Vulkan is an error
  rather than a smaller green suite;
* a test that skips its substantive block returns ctest's `SKIP_RETURN_CODE`, so the summary reads
  `(Skipped)` instead of counting it as a pass. CI runs `ctest --output-on-failure`, which suppresses
  test stdout on green runs — so a printed `[skip]` line alone is invisible in the summary *and* in the
  log body.

**Windows/MinGW and macOS still disable Vulkan**, so layers 2–4 are Linux-and-local only there.

Four tests are excluded by name on lavapipe because they pass on real hardware and fail on the software
rasteriser; they are listed in `ci.yml` with a reproduction command and tracked in #1681. Dump-backed
tests cannot run in CI at all — dumps are copyrighted and gitignored — and now report `(Skipped)` rather
than passing.

**Three dump-backed tests are pinned to *The Messenger* specifically, not to "some dump" — and they
now report `(Skipped)` instead of failing (#1573).** `module_loads_eboot` pins `PPSA24651-app0`'s
import, segment and relocation counts (`tests/test_module.cpp`); `boot_reaches_first_syscall` links
that title's own `Media/Modules/Il2cppUserAssemblies.prx` and `PS5Util.prx`; `real_shader_render`
recompiles and renders the RDNA2 blobs embedded in its eboot. Configure with `-DGAME_DUMP=` pointing
at another title and all three used to go red for a reason nothing in the ctest output named —
measured on `PPSA01826-app0` (The Pathless), **243/246 with these three failed**:

```text
module_loads_eboot          imports=1665 expected 612 / distinct import libs=50 expected 35
boot_reaches_first_syscall  link: load .../Media/Modules/Il2cppUserAssemblies.prx: cannot open file
real_shader_render          [FAIL] found the game's embedded RDNA2 shader blobs
```

Every plausible reading of that — master is broken, this dump is corrupt, bisect — is wrong, and it is
exactly the run a bring-up agent makes first on a new title.

CMake now derives the configured dump's title id (`sce_sys/param.json`'s `titleId`, falling back to the
`PPSAxxxxx-app0` directory name) and registers those three as **visible ctest skips** naming both the
required and the configured id when it is not `PPSA24651`. The same run is now **243 passed, 3 skipped
out of 246, exit 0**; the reason is printed at configure time and is in the test's own output under
`ctest -V`. Point `-DGAME_DUMP=` back at `PPSA24651-app0`, re-run cmake, and they execute again.

If **neither** source yields a title id the three are registered and run exactly as before. A detection
failure has to stay fail-visible: silently converting three real guards into permanent skips would be
the same defect this section is about, pointed the other way.

### Dump-gated tests, and the loader assertions that no longer are (#2567)

A visible `(Skipped)` fixes the *legibility* of a dump-gated assertion, not its *coverage*. Six ctest
cases carry assertions that only run with a dump — `module_loads_eboot`, `nid_hash_matches_imports`,
`trap_identifies_imports`, `boot_reaches_first_syscall`, `real_shader_render` and `plugin_autolink` —
so a regression in the paths they cover is caught only by an agent who happens to run the suite
locally with that title installed, which is not a gate. #2567 carries the measured per-case census.

**Four of the six cannot be synthesized and should not be.** They are pinned to one real title's
bytes — `PPSA24651`'s import counts, its IL2CPP module layout, the RDNA2 blobs embedded in its
eboot. A synthetic fixture would be a *different* test wearing the same name, which is worse than an
honest skip.

**Ruled out: `trap_identifies_imports` was never one of them (#2997).** It was counted here because
it asserted `prog.slots.size() > 500` — PPSA24651's slot count — but that assertion was incidental
to what the case tests, which is the title-agnostic link → map → stub → dispatch chain. Measured
across all 55 local dumps: slot counts run from 179 (`PPSA16901`) to over 500, and **all 55 dispatch
all four probed libraries correctly**. The count made it *fail* — not skip — on four titles,
including `PPSA15552` (*Dead Cells*, rung 6, guarded) and `PPSA03839` (*Tactics Ogre*, rung 3). With
the assertion made title-agnostic it passes on 55 of 55, so it is dump-*requiring* but not
title-pinned, and it needs neither a gate nor a synthetic fixture.

The distinction is the one worth carrying forward: a case that *needs a dump* is not the same as a
case *pinned to one title's bytes*, and reaching for `prosper_add_title_gated_test` on the first kind
trades a false red for lost coverage on every other dump.

**`plugin_autolink`'s dump-gated block was the exception, and is now covered without a dump.** What it
guards — `link_program`'s export table, the duplicate-export collision guard (#1609) and the alias
record (#1635) — is structural loader behaviour that depends on the shape of a module's dynamic
tables, not on any title's content. Two ctest cases now exercise it everywhere:

| case | what it establishes |
| --- | --- |
| `loader_synth_link` | the export table, first-wins and its alias record, the collision guard and its skip record, cross-module import binding and stub-slot fallback |
| `loader_synth_reject` | the corrupted variants the loader must refuse, and the relocations it must decline to apply |

The fixtures are minimal `ET_SCE_DYNAMIC` PRX modules emitted at test time by `tests/fixtures/synth_prx.h` and
`tests/fixtures/handmade_prx.h`. **Every byte is synthesized from the published ELF64 layout; none is carved
out of a dump**, and none may ever be — this repository is public, and a dump-derived fixture would
put Sony-built code in it however small it was. Nothing is committed as a binary, so there is no
opaque asset to regenerate: the generator *is* the provenance record (#2587's convention, applied to
a generated asset).

The dump-backed `plugin_autolink` block stays exactly as it is. It links real Sony-built modules and
is not redundant with a synthetic pair.

**Three rules bind on any future fixture-based test here, and the first two are what make it evidence
rather than decoration:**

1. **Build the negative arms.** A fixture the loader accepts proves only that the generator and the
   loader agree with each other. Each corrupted variant must differ from a *linking control* in
   exactly one field, so the arm can show its lever moved.
2. **Build at least one instance BY HAND, outside the generator.** A generator emits one geometry and
   every fixture inherits it, so a defect about some *other* legal shape is structurally inexpressible
   in its output and reads as a clean pass — a positive control drawn from the same source as the null
   it validates tests the discriminator, never the domain. `handmade_prx.h` is that instance: two
   `PT_LOAD`s whose file offsets and vaddrs diverge, so every dynamic table is reached through
   `Module::va2foff`'s translation rather than an identity map.
3. **Include a mutation arm for every guard that can fire.** `loader_synth_link` links the same
   flagged module twice, once where its exports collide and once where a single export NID differs; a
   guard that skipped every flagged module would pass the first arm exactly as well as a correct one.

## PROGRESS_TRACKER.md regenerates itself on main — it is no longer a gate

`PROGRESS_TRACKER.md` is **generated** from the `tracker:game` issues. It used to be gated on every
PR (`gen_progress_tracker.py --check`), and **that gate was removed on 2026-08-27** because one of
its two triggers was unfixable by the person it failed.

**You no longer have to do anything.** A `progress-tracker` job regenerates the file on every `main`
push and commits it if it changed. If your PR closes a cited blocker, `main` repairs itself.

The reasoning is kept because it is the useful part:

1. **A tracker was edited mid-flight.** Someone ticked a rung or changed an FPS record while your PR
   was open. This one a person could fix — regenerate, commit, done.
2. **A PR merged that CLOSED an issue this file cites as a blocker.** This one is automatic, and
   **no commit could satisfy it**: the generator drops a blocker from a title's row once that issue
   closes, so a PR whose body says `Fixes #NN` makes master stale *the instant it merges*. The
   correct content does not exist until after the merge. Nobody touches a tracker; the author has
   already merged and moved on; the next unrelated PR ate the red.

Trigger 2 measured on 2026-08-23 rather than reasoned: #2956 merged at 07:41:59Z, its `Fixes #2951`
closed that issue at **07:42:00Z**, and the `Docs` job on an unrelated documentation PR failed at
07:48 against a file neither PR had touched. One second between a green merge and a file that was
wrong. It recurred on 2026-08-27 with #3081/#3085, which is what finally removed the gate: measured
over the preceding 60 master runs, **5 of the 7 Docs failures were this step**.

**What still fails, and should.** The `docs` job keeps `gen_progress_tracker.py --selftest` as a hard
gate — a generator whose regexes quietly widened would emit a plausible wrong file, and that IS
author-fixable. The regeneration job also fails hard when the generator itself errors: an unparseable
tracker body, a zero-issue fetch, or an unresolvable citation writes no file, and swallowing that
would freeze the tracker behind a permanently green job. Only *push contention* is soft, and it
retries three times before warning.

**What was given up**, stated plainly because nothing else now catches it: there is no longer any
signal that the committed file has drifted from the trackers by some route the generator does not
error on. That was judged the right trade against an unfixable gate firing on most Docs failures, and
it is trivially reversible.

`--check` remains in the tool for local use:

```bash
python3 prosper/tools/docs/gen_progress_tracker.py --check    # does the committed file match?
python3 prosper/tools/docs/gen_progress_tracker.py            # regenerate it
```

## Sentinels: assert on the bits, not the decoded value

A test that fills a buffer with a poison byte and then checks the *decoded* field cannot always tell
**"the code wrote 0"** from **"the code wrote nothing"**. prosper's guest allocator poisons with
`0xaf`, and that pattern decodes to a number small enough to pass a tolerance check:

| pattern | as float32 | passes `fabs(x) < 1e-6`? |
|---|---|---|
| **`0xafafafaf`** | **-3.196e-10** | **yes — silently** |
| `0xcdcdcdcd` | -4.3160e+08 | no |
| `0xcccccccc` | -1.0737e+08 | no |
| `0xdddddddd` | -1.9984e+18 | no |
| `0xfefefefe` | -1.6947e+38 | no |
| `0xbaadf00d` | -1.3270e-03 | no |

(`0xafaf…af` as float64 is -5.3e-79, which passes too.) Every other common fill fails loudly, which
is exactly why this one is worth writing down: the habit of "poison the buffer and check the value"
is safe with most patterns and quietly unsafe with ours.

The failure it produces is the one hardest to see — an assertion that is **true** while the
mechanism it is supposed to exercise never ran. It was found in #2951 only because a mutation arm
removed the writer and the test still passed. So:

- **When "the field was written at all" is the thing you are ruling out, compare the raw bits**
  (`(uint32_t)bits != 0xafafafafu`), not the decoded float. Keep the value check too if the value
  matters — the two are not redundant, and a comment should say so, because the next reader will
  assume they are.
- **Never give a byte-fill sentinel and a value sentinel the same name.** `memset(buf, kPoison, n)`
  and `float x = kPoison;` differ by four orders of magnitude and read identically at the call site.
- Prefer a mutation arm over inspection for this class. No amount of reading finds it; deleting the
  writer does, in one run.

Worked example and the surviving arms: `prosper/tests/hle/test_font_render.cpp`. Recorded as
instrument trap 227 in `GAME_COMPAT_ORCHESTRATION.md`.

## Ruled out — an indeterminate value cannot be asserted on, so a test cannot cover it

Sibling of the sentinel section above, and the same class one step further out: there, a *written*
poison value could be mistaken for a real one; here there is no value at all.

- **A regression test cannot demonstrate that a handle was UNINITIALIZED rather than null.** #3210's
  two worst sites (`VkRenderPass rp;` / `VkFramebuffer fb;` in `tests/fixtures/render_runner.h`) read
  an uninitialized automatic on a failed create. Reading one is undefined behaviour with no
  observable value an assertion can name, and in practice this box's RADV *does* leave the output
  handle null — so an "unfixed code crashed" arm would be unreproducible and would prove nothing
  about a driver that behaves differently. Measured: with the fix's control flow intact, deleting
  *every* pre-call `*out = VK_NULL_HANDLE;` **and** both caller-side initializers leaves
  `test_render_object_create_failure` fully green (build rc 0, test rc 0). What the test does cover
  is the half that has observable behaviour — that the failure is reported and the pass is dropped;
  a mutation that reports and then continues past the failure segfaults the process at the first
  injected render. The initialization half is established by reading the code, and that is the
  honest ceiling for this class rather than a gap to be closed with a cleverer test.

## Rule of thumb

Prefer the cheapest layer that can catch a given bug: pure structural asserts for translation logic,
pixel/region plus golden hashes for deterministic raster tests, execution-differential checks for shader
semantics, and routed multi-frame guards for integrated games. Only the final integrated scene needs a
game dump and the GPU; everything upstream is verified on the CPU.
