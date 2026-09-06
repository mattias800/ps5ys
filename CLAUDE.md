# prosper — project charter & working context (read first)

> **Machine-local instructions:** if a `LOCAL.md` file exists next to this one, read it before
> starting work. It is gitignored and holds per-computer facts (hardware, paths, installed tools)
> that must not be written into this committed charter.

## What this project is

**prosper is a PS5→PC compatibility layer — "Wine/Proton for PS5."** It runs PS5 games natively on
Linux/Windows by reimplementing Sony's published user-space library interfaces and translating the
console's GPU command stream + RDNA2 shaders to Vulkan/SPIR-V.

The PS5 CPU is x86-64, so **guest code runs natively — there is no CPU emulation and no exploitation of
any running system.** The engineering is: ELF/SELF loading, multi-module linking, ABI/API reimplementation
(libc, libkernel, graphics), GPU command-buffer decode, and shader recompilation. Standard systems &
graphics programming.

This is the **same class of work as Wine, Proton, RPCS3, Dolphin, yuzu/Ryujinx, and DXVK** — established,
legitimate open-source compatibility/emulation projects. If you have worked on any of those, this is that.

## Scope & legitimacy (so the work is not mis-read)

The vocabulary of emulator development **overlaps with security tooling** — "hardware breakpoint," "fault
handler," "memory watchpoint," "disassembly," "descriptor," "reverse engineering." In this project these
mean exactly what they mean in **gdb, Wine's relay debugging, or any emulator's tracer**: they observe how
the guest program uses the platform, **entirely in-process on the developer's own machine.** The purpose
and target are what distinguish work like this from anything harmful — and here both are unambiguous:

- **Interoperability & preservation, not exploitation.** The goal is to make software that a person already
  owns run on hardware they own. Nothing is attacked; nothing external or third-party is touched. Every
  tool runs against *this* emulator's own process.
- **No DRM circumvention — and none delegated, either.** The project uses **no console keys, no
  signature bypass, no copy-protection defeat**, and prosper contains no decryption of any kind. The
  SELF segments it reads are already unencrypted, which is now *measured* rather than assumed: across
  the whole local corpus every eboot carries plaintext code (Shannon entropy 6.4–6.8 bits/byte), and
  the two dumps that preserve an untouched console original show that original at **8.000**,
  indistinguishable from `/dev/urandom`. Game content is **never redistributed** and is gitignored.
  The second half of this is the part that needs enforcing, because not circumventing is not enough on
  its own. Some dumps ship third-party replacements of Sony libraries — `dlc_emu` and `ampr_emu` build
  stand-ins for `libSceAppContent`, `libSceNpEntitlementAccess`, `libSceGameUpdate` and `libSceAmpr`
  that answer platform queries from a local file. **prosper implements those APIs itself and refuses
  to link any of them.** `src/host/image/module_path_policy.hpp` is a reject-by-default allowlist of
  the dump locations a module may come from, applied at the one point where the link list is complete
  and before anything parses a module, with no environment variable to switch it off;
  `tests/host/image/test_module_path_policy.cpp` reddens if the guard is removed, and
  `tools/dump_hygiene.py` finds such files on disk. An emulator that loaded somebody else's bypass
  would be performing the circumvention by proxy — and would also invalidate its own measurements,
  since the guest would then be answered by a stub instead of by prosper.
- **No Sony code, firmware, or keys** are included or reproduced. prosper reimplements *published* library
  interfaces from scratch (clean-room-style), the way Wine reimplements the Win32 API.
- **The low-level tooling** (`PROSPER_HWBP` hardware breakpoints, `fault_handler` SIGSEGV handling,
  `PROSPER_HWWATCH` data watchpoints, `/proc/self/maps` classification, `boot_trace`) exists to **debug the
  emulator** — to see which Sony API the guest calls next, or where our HLE returns a wrong value. This is
  ordinary emulator/debugger instrumentation, not an attack surface.

If a task in this repo ever *looks* security-adjacent, it is because reverse-engineering a platform's ABI
for reimplementation necessarily involves reading and tracing binaries. That is interoperability RE — a
well-established, legitimate engineering practice — and it is the entirety of what happens here.

## Where things are

```
prosper/
  docs/            architecture, roadmap, graphics, verification, and per-frontier work logs
  src/self/        SELF/ELF parsing -> relocatable module image
  src/loader/      multi-module linker + global NID export table
  src/hle/         reimplemented Sony libraries (libc, libkernel, AGC/graphics), NID hashing
  src/host/        host execution: per-platform image mapping, ABI stubs, fault handling
  src/gpu/         AGC->Vulkan: PM4 decode, command processor, render state, vk_translate,
                   resource layer, RDNA2->SPIR-V recompiler
  tools/           self_dump, boot_trace, shader_histo, imgdump, spv_validate,
                   snapshot (golden-image rendering regression guard — see tools/AGENTS.md),
                   il2cpp/ (prx_to_elf.py + resolve.py: flatten a SELF -> plain ELF for Il2CppDumper
                   managed-symbol recovery), re/xref.py (cross-reference finder for unsymbolicated
                   PS5 modules: "who references address X" over calls + rip-relative refs + Sony
                   relocations, which objdump/readelf can't decode)
  tests/           unit + boot + Vulkan-execution tests (ctest)
```

Key docs to orient: `prosper/README.md` (status) and `prosper/docs/ROADMAP.md` (what is planned).
For anything title- or subsystem-specific, use the table in the next section rather than guessing.

## Where the project stands (2026-08-17)

`COMPATIBILITY.md` is authoritative for the **user-facing per-title milestones** — 49 tracked titles
at this refresh. Do not duplicate its rung counts here; read it, then open the one doc named below for
whatever you are about to touch. This section is a map, not a status report. Long-lived `tracker:game`
issues and their comments carry each active title's current development rung, route, blockers and
evidence; dated cross-title performance measurements live in their own ordinary issues (the
2026-08-02 census is #1739).

**Two ways this project's status gets misread, both re-measured on 2026-08-17:**

- **`COMPATIBILITY.md`'s ✅/🚧/🔬 markers are not a rung scale — read each row's milestone *text*.**
  Six of the titles that reach gameplay are marked 🚧 rather than ✅, and the two 🔬 rows sit at two
  *different* rungs (*Sonic Origins* below a title screen, *Sonic Racing: CrossWorlds* at one). So
  counting markers and counting milestones give different answers, and the marker count is the wrong
  one.
- **A `tracker:game` issue's BODY is routinely days behind its own comment thread, and a fresh
  `updated_at` does not mean the body was re-read** — the timestamp moves on every comment. Three
  live examples today: #1871's body says rung 0 where its latest comment says rung 1; #1895's body
  leaves rungs 1 and 2 unticked where its comments established rung 2 nine days earlier; #1886's
  body says rung 1 where its own last comment, `OREGON_TRAIL_STATUS.md` and `COMPATIBILITY.md` all
  say rung 2. **Read the last comments before quoting a tracker's rung**, and prefer the status doc
  when the two disagree.

**Concurrent game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`** — lane ownership,
shared-GPU policy, the instrument-not-the-subject list, and the dated current handoff.

### Which doc to read next

| Title / area | State | Read this next |
| --- | --- | --- |
| *The Messenger* `PPSA24651` | rung 6 — complete first level, checked against PS5 hardware; `messenger-scene` guard | `docs/MESSENGER_BLACK_RENDER.md` |
| *Dead Cells* `PPSA15552` | rung 6 — full-colour Prisoners' Quarters; `dead-cells-gameplay` guard | `docs/DEAD_CELLS_STATUS.md` |
| *Blasphemous 2* `PPSA13579` | rung 6 — first playable room; `blasphemous2-gameplay` guard | `scripts/blasphemous2/README.md` |
| *Alex Kidd in Miracle World DX* `PPSA02664` | rung 6 — resolved by #1578 (generic 4 KiB mip-tail tiling); `alexkidd-gameplay` guard | `docs/PPSA02664_BLACK_WORLD.md` (historical) |
| *Blue Prince* `PPSA25009` | rung 6 — sustained Day One gameplay and the Mount Holly entrance hall; reviewed automatic gameplay snapshot guard (tracker #1808) | `docs/BLUE_PRINCE_STATUS.md` |
| *Syberia: Remastered* `PPSA30140` | rung 3 — gameplay renders; menu 3D layer and gameplay composite degraded (#1619 / #1627) | `docs/SYBERIA_STATUS.md` |
| *Tales of Graces f Remastered* `PPSA19991` | rung 3 — the Lhant Hill prologue renders with real GPU draws on `scripts/talesgraces/reach-gameplay.pad`. What held it at rung 2 was input, not the renderer: two new-game screens bind their "go" action to the **OPTIONS button** and each raises a Yes/No dialog defaulting to **No**, so a Cross-only route loops forever. Its Unity progression oracle is **not** `Media/levelNN` (there is one level file) but an Addressables `togf_mapmodel`+`togf_maptexture` map pair | `docs/TALES_GRACES_STATUS.md` |
| *Dragon Quest VII Reimagined* `PPSA17942` (a.k.a. DOLL) | rung 2 — title, name entry, onboarding; composition defect and gameplay open | `docs/DRAGON_QUEST_STATUS.md` |
| *Nikoderiko* `PPSA23760` | rung 2 — 3D world dropped; the blocker is the user-data-window mismatch, not the recompiler. **#305 is closed and was never fixed** — it was closed on 2026-08-03 alongside #1794, whose own body says it *"does not fix #305"* — so read it as the record of the defect, not as an open blocker. #1607 (open) carries the title-specific impact; the tracker and status doc still cite #305 as open | `docs/NIKODERIKO_STATUS.md` |
| *Asterix & Obelix: Babylon Mission* `PPSA30490` | rung 2 — logo movies, intro cutscene and title menu; the video-splash seek deadlock is fixed (#1949) | `docs/ASTERIX_BABYLON_STATUS.md` |
| *R-Type Delta: HD Boosted* `PPSA26414` | rung 2 — logo, full-colour opening movie, title screen and attract mode. Launch needs `tools/dropcache.py` first: the title races its own 400 ms user-event delay against prosper's asset load (#1746, a product decision, not an open investigation) | `docs/R_TYPE_DELTA_STATUS.md` |
| *The Oregon Trail* `PPSA19244` | rung 2 — title screen rendered on a default launch (60 s / 3,689 frames clean); the ordered-DMA stall is fixed (#1987) and **#1945 no longer reproduces here** (0 of 9 arms — use `PPSA21406` for that repro). The UI layer composites correctly since #1946 — its solid-block glyphs, black logo panel and flat sky were one defect: the RT0 blend state never reached the GPU | `docs/OREGON_TRAIL_STATUS.md` |
| *Little Nightmares III* `PPSA05143` | rung 2 — title screen rendered on a default launch; the render-thread stall is fixed (#1987). Most title frames arrive with red and green forced to maximum, reading as a yellow background (#2014) | `docs/LITTLE_NIGHTMARES_3_STATUS.md` |
| *ArcRunner* `PPSA21406` | rung 0 **on a default launch** — and the renderer is not the frontier. Under `PROSPER_SUBMIT_STALL_US=1500` the whole 4K intro cinematic renders and the title screen is reached and held; under `PROSPER_POST_SUBMIT_VISIBILITY=1` (new, default OFF) the default route survives with no throttle under **`boot_trace`**, 3 of 3, and a 260 s `boot_trace` run renders the title screen. **That survival claim is `boot_trace`-only, and on this title the frontend has been the variable:** `tools/screenshot` — the frontend the project uses for progression evidence — faulted **2 of 2** on the same lever and the same head, while a later 12-run replication at three code points, current master among them, did not reproduce those deaths. Neither result is withdrawn, so the lever's verdict under the evidence frontend is undecided; quote the frontend, not "the default route survives" (#2217, open; instrument trap 127). The race is named: prosper's post-submit completion-visibility contract is armed only for SDK ≥ 13 and this title requests 10, so its command-chunk recycler is released mid-fold (#2219; race #1226, dose-response #2084). Tracker #1817 deliberately holds the rung at 0 — a frame that needs a non-default switch is not default-route evidence. Removing the version gate is one line, but three rung-6 guarded titles are also pre-13 and the contract that rescues this title **regresses *Sonic Frontiers*** (#2223), so it waits on a cross-title snapshot pass (#2220) | `docs/ARCRUNNER_STATUS.md` |
| *Crisis Core –Final Fantasy VII– Reunion* `PPSA07809` | rung 2 **on a throttled route** — title screen, main menu and the new-game settings flow render; a default run dies 5-12 s into the boot. Same SDK-10 submit race as *ArcRunner*, and the same lever answers it: `PROSPER_POST_SUBMIT_VISIBILITY=1` faults 0 of 3 **under `boot_trace`** (the rung-2 route above is `tools/screenshot`; name the frontend — instrument trap 127). The title screen's key art never draws | `docs/CRISIS_CORE_STATUS.md` |
| *The House of the Dead 2: Remake* `PPSA24203` | rung 3 — a fresh-save route reaches Training 1 with real GPU draws through the normal full-cadence renderer; severe world-rendering defects remain (#1907) | `docs/HOUSE_OF_THE_DEAD_2_STATUS.md` |
| *Sonic Frontiers* `PPSA03831` | **rung 2** — a route reaches Cyber Space gameplay in the guest but the world is black behind the HUD (#2790), which is rung 2 under the rendering bar; 4K opening sequence, auto-save notice, title screen and main menu on a default launch. The four-session black-screen wall was one unregistered NID answering `SCE_OK`: `sceSaveDataTransferringMountPs4` (#2023). The menu heading draws the wrong string (#2206) | `docs/SONIC_FRONTIERS_STATUS.md` |
| *Sonic Racing: CrossWorlds* `PPSA08804` | rung 2 — a pulsed pad route reaches the complete 4K title screen and profile menu; the profile panel is black and the sequence later holds on white (#2013 / #2358 / #2360) | `docs/SONIC_CROSSWORLDS_STATUS.md` |
| *Grand Theft Auto V* `PPSA04263` | **rung 2** — the guest reaches routed gameplay entry with real GPU draws, but the world does not render, and rung 3 requires the scene to render; the HUD, radar and tutorial text render over an absent 3D world. **The missing world is now one compute program**: `0x413dc6700` hangs the GPU into a RADV hard recovery, which disables live compute process-wide and drops every later indirect draw; skipping it (`PROSPER_COMPUTE_SKIP_PROGRAM`) yields 0 device losses and the first real scene content. The descriptor-array lift is complete, and the recompiler is **not** the frontier — 16 of 20 "unsupported" programs recompile cleanly | `docs/GTA5_STATUS.md` (read its `## Ruled out`, including the *void, not falsified* subsection), tracker #1873, issue #2481 |
| *GRIS*, *Space Adventure Cobra*, *Sonic Origins* | rung 6 / rung 6 / rung 1 — GRIS and Cobra are guarded, not merely playable: reviewed `gris-gameplay` / `cobra-gameplay` entries in `tools/snapshot/snapshots.json` (trackers #1869 / #1870). Sonic's black startup loop is fixed (#1905: `sceSaveDataCreateTransactionResource` must return a positive resource id); it renders the 4K SEGA logo and then decoded 4K movie frames, with **no title screen observed** — the old "holds on white to the end" reading is falsified since #2571 made videodec2 decode by default. **#1871's issue *body* still says rung 0 — its 2026-08-17 comment corrects that to rung 1; read the comment thread, not the body** | `docs/GRIS_SONIC_COBRA_BRINGUP.md` |
| Concurrent title work | the 2026-07-31 lane allocation is historical; use live issue claims for ownership and the dated checkpoint for the current cross-lane handoff | `docs/GAME_COMPAT_ORCHESTRATION.md` |
| *Tactics Ogre: Reborn* `PPSA03839` | rung 3 — gameplay reached; HEVC movies render, sprite/HUD composition remains open | `docs/TACTICS_OGRE_STATUS.md` |
| *Beneath* `PPSA27640` | rung 3 — the opening dive in `Science_Remake2`, reached by a Cross-only menu ladder with no code change. **Needs `PROSPER_NULL_PAGE=1` to boot at all**: the Unity player's frame-pointer unwinder reads the return-address slot of the terminal NULL frame pointer (`addr=0x8`, `eboot+0xe44346`), which is diagnosed rather than merely worked around | `docs/BENEATH_STATUS.md` |
| *Earthion* `PPSA28061` | rung 2 — the wrapper's title/main menu and HOW TO PLAY page. **The recorded blocker was wrong**: it was not "the menu needs directional navigation" — that menu is the wrapper's and has no start item, and the title's own HOW TO PLAY page binds **Start to the TOUCHPAD**. Delivering it (13 presses read by the guest) still leaves the game picture black; it never leaves a ~60 s narration cycle. #1773 is fixed and was **not** the cause — the alias now recovers `ud_alias=s9` and the composite is 28/28 pixel-identical | `docs/EARTHION_STATUS.md` |
| *Metaphor: ReFantazio* `PPSA20800` | rung 1 — the loading mascot and the **language-selection screen**, held on a default launch, with legible Latin/Cyrillic/CJK text. Two prosper defects got it here from zero frames in two days, and both were prosper answering a question wrongly: the direct-memory pool starting outside the budget it advertised (#2934), and `sceFontRenderCharGlyphImage` being unregistered so the dispatcher's `return 0` answered SCE_OK and wrote neither out-parameter — the title read the untouched stack as a glyph size and divided by the resulting zero (#2951). **Its font path is now a real rasterizer over the title's OWN TrueType file**, which it supplies through `sceFontOpenFontMemory`; check the *open* call before concluding a font API cannot be answered honestly. No title screen yet; the background behind the menu does not draw (#2952) | `docs/METAPHOR_STATUS.md` |
| Every other title | — | `COMPATIBILITY.md` |
| UE4 / IoStore bring-up (shared) | — | `docs/UE4_APR_IOSTORE_BRINGUP.md`, `docs/CROSS_ENGINE_UE4.md` |
| Renderer performance | July pass complete; the stop decision is recorded | `docs/RENDERER_PERFORMANCE_2026_07.md` |
| Windows port / release | native SDL3+Vulkan frontend, screenshots, release path | `docs/WINDOWS_PORT_HANDOFF.md`, `docs/WINDOWS_RELEASE.md` |
| Linux desktop release (AppImage + tarball) | published on `v*` tags; packaging is built and verified on every PR | `docs/LINUX_RELEASE.md` (users), `packaging/linux/README.md` (how, and its `## Ruled out`) |
| GPU capture / replay / timeline tooling | F9 frame grab → offline `.prgbundle` / `.prgcap` workflow, and every `PROSPER_*` graphics diagnostic | `tools/gpu_replay/README.md`, `tools/gpu_timeline/README.md`, `tools/AGENTS.md` |
| Graphics architecture, resource binding, recompiler | — | `docs/GRAPHICS.md`, `docs/RESOURCE_BINDING.md`, `docs/RECOMPILER_REMAINING.md` |
| AGC command-packet sizes (a builder's dword count is an ABI contract with the guest's own reservations) | — | `docs/AGC_PACKET_SIZES.md` |

### Before you form a hypothesis, read the `## Ruled out` section

Every `docs/*_STATUS.md` in that table — and `PPSA02664_BLACK_WORLD.md`, `MESSENGER_BLACK_RENDER.md`
and `GRIS_SONIC_COBRA_BRINGUP.md` — carries a **`## Ruled out`** section: one line per
already-falsified hypothesis, the evidence that killed it, and the issue/PR link. (Blasphemous 2's
row points at a route README, which has none.) Cross-title
falsifications live in the area doc (`docs/RESOURCE_BINDING.md`, `docs/RECOMPILER_REMAINING.md`).
These exist so nobody re-derives a dead answer at full cost — they are the most expensive knowledge
in the repository. Extend them; see the recording rule under *How to work here*.

The standing warnings that are **not** title-specific:

- **Do not restart the superseded Messenger depth, vertex-fetch, geometry, palette, or tiling
  hypotheses without contradictory new evidence.** Each is falsified with evidence in
  `docs/MESSENGER_BLACK_RENDER.md` § Ruled out. The real cause was a missing recompiler instruction
  plus loss of per-target dimensions (#526/#527, fixed by #528), with #534 and #541 behind it.
- **Do not resume Messenger-specific renderer cache work toward 60 FPS.** The July pass deliberately
  stopped at roughly 12 → 24 FPS on the first level; the remaining synchronous graphics/compute
  boundaries must be evaluated against a **3D** workload first (*Summer Sports Games* `PPSA03416` is
  the first clean candidate). `docs/RENDERER_PERFORMANCE_2026_07.md`.
  **That 24 FPS is stale, and the reason matters more than the number.** It was measured correctly,
  with `prosper-app`, on 2026-07-14 — and #1270 changed the present architecture ten days later
  (2026-07-24), so it describes a renderer that no longer exists. Re-run windowed and uncapped
  (`--present-mode immediate`) on 2026-08-27, the same first level reports a **median 156 fps**, with
  two thirds of samples between 100 and 160 (n=468; p25 130.9, p75 165.7, and only 2 samples below
  100). Two caveats on that median. It is a **presented** rate with no `distinct` companion, so read
  it as an upper bound on the rate of new frames rather than as the frame rate — #3083 records the
  A/B that pins it. And the run is **not stationary**: ~185 fps for the first 29 s, ~120 for the
  final 76 s after `Media/level2`/`level6` stream in, so 156 is the median of a declining sample.
  Every regime survives the comparison — the slowest third is still several times the July figure —
  and the advice above survives with it: this title is nowhere near being the thing to optimise.
- **Before quoting an FPS number from this project, check which harness produced it — and when.**
  Three ways to get a wrong one, and a single run can hit all three:
  - **`tools/screenshot` never calls `set_gpu_present_active`** (the sole call site is
    `frontends/prosper-app/main.cpp:668`), so in that harness the renderer copies every scanout frame
    back to the CPU. Its rate describes the forced-readback path, not the shipped one. **This applies
    only to figures measured after 2026-07-24** — before #1270 there was no GPU-present path, so
    every harness took the readback and the old figures are honest measurements of what then shipped.
  - **A `fifo` run measures the display, not the renderer.** A vsync-locked run of this same title
    on 2026-08-26 pinned 123 of 238 samples at exactly `180.0` — the host refresh rate. Pass `--present-mode
    immediate` before quoting a rate as a throughput, and say which mode the run used. Quote
    `distinct` rather than `presented` — and note the interactive app could not report a distinct
    rate under GPU present at all until #3019 (2026-08-26), so an interactive rate from before that
    date is a publication count.
  - **The capture is part of the measurement.** The uncapped run above has a `42.9` sample sitting
    directly after `[grab] screenshot written`, followed by a compensating `195.3`. That is the frame
    grab, not the title — and quoting it as the run's minimum is the same error as quoting a
    screenshot-harness rate as the renderer's. Drop the samples either side of a capture before
    taking a minimum.
- **Start Windows work from `docs/WINDOWS_PORT_HANDOFF.md`**, not the solved pre-render fence
  investigation; Windows *release* users start from `docs/WINDOWS_RELEASE.md`.
- **A historical capture hash is not a current renderer oracle**, and neither is a target extent or a
  raw draw count on its own. Addresses and operation ordinals are run-local. Re-derive the selector
  with `gpu_timeline --signatures` / `--select` before recording a new bundle.
- **`PROSPER_UD_TAIL_ALIGN` must stay off.** It exists only so the A/B that falsified the
  user-data-tail hypothesis stays reproducible (`docs/RESOURCE_BINDING.md` § Ruled out).
- **Read `GAME_COMPAT_ORCHESTRATION.md`'s instrument-not-the-subject list before believing any
  surprising measurement** — every entry on it is a phantom defect that came from the apparatus
  rather than the subject, and several cost hours. (Deliberately no count here: the list is appended
  to by concurrent lanes, so a restated total is stale as soon as it is written.) Two of its positive rules bind on any graphics investigation, not
  just orchestrated ones: **a decoded-draw census is meaningless without the render phase and a
  positive control** (draw counts vary by two orders of magnitude *within one title* — count per
  sample interval, calibrate against a title known to render the thing you claim is missing, and
  open the frames), and **prefer experiments that detect their own invalidity** over experiments that
  can fail silently. Check a diagnostic's rate limit before quoting its volume as a frequency, and
  check whether it clears `live_gpu_targets` before comparing it against a default run.

- **A positive control drawn from the same source as the null it validates tests the DISCRIMINATOR,
  never the DOMAIN.** This qualifies the rule above, and it is the harder half. "A discriminator that
  cannot show its lever moved is void" is true — but a lever that *does* move proves only that the
  machinery runs, not that the space it searches can express the case you are testing for. Worked
  example (2026-08-06, instrument trap 122): a reviewer's fuzz reported zero instances of a widening
  class. Suspecting the zero, they ran a positive control — **and it passed**: 41,206 constructions,
  17,659 narrowings, 0 widenings. It could not have done otherwise, because it drew from the same
  generator and so inherited the same broken geometry, in which the case was *structurally
  inexpressible*. Interior geometry surfaced 1,277 instances immediately.
  **So: before believing a clean zero, construct one positive instance of the class BY HAND, outside
  whatever produced the null.** A same-source control confirms the machinery fires on cases the source
  can build — which is exactly the assumption in question. This is the sharpest form of the family the
  instrument-trap list records, because unlike the silent instruments in traps 64, 116 and 121, this
  one is *loud*, and the noise is what launders the null.

### Superseded documents

`docs/NEXT_STEP_VERTEX_FETCH.md` (bindless-dynamic vertex fetch, superseded 2026-07-11) and
`docs/RENDER_LOOP.md` (the render bring-up log) are historical records. Both frontiers are complete:
both shader stages recompile and dynamic V#/T#/S# resources resolve on current `main`, and the
render-loop frontier is complete. Each carries its own superseded banner — do not start work from
either, and do not read `RENDER_LOOP.md`'s "Status: open" as current.

## How to work here

- **Concurrent game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`.** It defines the orchestrator and
  subagent contract, shared-GPU policy, evidence/PR cadence, and the dated current handoff for active title lanes.
  Update its current-state sections when ownership, evidence, or the exact frontier changes; keep detailed findings in
  GitHub issues so the document remains a discoverable map rather than the only record.

- **A PR that falsifies a hypothesis records it in the relevant `## Ruled out` section before merging** —
  the same way instrument traps are recorded in `GAME_COMPAT_ORCHESTRATION.md`. One line: the dead
  hypothesis, the evidence that killed it, and the issue/PR link. Title-specific → that title's status
  doc; cross-title → the area doc (`RESOURCE_BINDING.md`, `RECOMPILER_REMAINING.md`, …). A falsification
  that lives only in a PR body or an issue comment is one the next agent will re-derive at full cost.
  **A result reported only in a message to an orchestrator is not in the project record at all** — if a
  lane reports an A/B to you, land it in the doc or the issue before the session ends. Read the
  `## Ruled out` section of every doc you are about to work in **before** forming a hypothesis.

- **File and folder structure mirrors the logical architecture, and you are expected to refactor
  toward that as you go.** A directory is a claim about what belongs together. When the claim is
  false the tree actively misleads, because "where does this live?" gets answered by the wrong
  place — and the cost lands on whoever arrives next, not on whoever left it. So put new code in the
  folder its architecture implies rather than beside whatever file you happened to open, split a file
  that has grown a second responsibility, and move a module that sits in the wrong folder.

  **You do not need permission or a dedicated task for this.** An incidental, mechanical restructure
  alongside the work you are already doing is welcome, and is how the tree stays honest between the
  occasional large passes. Two limits keep it from becoming a hazard:
  - **Move with tools, never by retyping.** `prosper/tools/refactor/` exists for exactly this:
    `move_module.py` (relocate a module and rewrite every include and path citation repo-wide),
    `split_file.py` (a split that proves it was a pure move), `promote_internal.py` (lift
    internal-linkage declarations into a header). Hand-editing a restructure is how this project's
    recorded traps happen — a whole-file `git checkout` that silently reverted another lane's edits
    (trap 41), and a clean merge into a broken file.
  - **Keep the move separable from the behaviour change.** Put it in its own commit so a reviewer
    reads `git mv` plus include rewrites instead of a diff that mixes relocation with logic. If the
    restructure is big enough to conflict with other lanes, it is its own PR.

  `CMakeLists.txt:71` is `file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)`, so a new
  subfolder under `src/` needs no source-list edit. Other roots are listed explicitly; check before
  assuming.

- **Every folder holding real content carries an `AGENTS.md` saying what lives there and what
  belongs there.** The point is navigation: an agent should be able to understand a folder's job
  without opening a source file. A paragraph or two — the folder's responsibility, its boundary
  against its siblings, and anything a newcomer would otherwise learn only by getting it wrong.

  It is a **map, not documentation of the code**. The code documents itself; a per-function summary
  here is stale the first time somebody edits it, and a stale map is worse than none because it is
  believed. Write about what the folder is *for*.

  Measured 2026-08-20 on the commit before this rule landed: **2 `AGENTS.md` files against 89 source
  directories** under `src/`, `frontends/` and `tools/`. The commit that added this rule also seeded
  all of `src/gpu`, taking it to 16 — so treat the ratio as the starting point it described, not as a
  current figure, and re-measure rather than quoting it. Write one when you work in a folder that
  lacks it; correct the existing one when you change what the folder means.

  *"Within reason"* is part of the rule: a folder holding a single file, or a leaf whose name already
  says everything, does not need one. If you cannot write a sentence the reader would not have
  guessed, skip it.

- **The branch is `main`. There is no `master`, and older text that names one is historical.**
  The repository's history was rewritten around 2026-07-04; `main` carries the rewritten line and is
  the default branch. The pre-rewrite `master` was left behind, went stale on 2026-09-02, and was
  **deleted on 2026-09-04** — its tip is preserved as the tag `archive/pre-rewrite-master-2026-09-02`
  if anyone ever needs it, and nothing unique lives there (same content as `main`, different SHAs).
  This is worth a line because the failure mode was expensive rather than obvious: a correct branch
  diffed against the stale `master` reported **3,296 commits of divergence**, and the session-trailer
  gate below read **582** instead of 0 — so a *clean* branch looked contaminated, and the natural
  repair (rebuild it onto `master`) moved good work onto the dead line.

  **Which mentions of `master` were repointed and which were left, as a TEST rather than a list** —
  a list is what let two live references through the first sweep, both of them in categories the list
  did not name: **a record of what was true keeps `master`; anything a reader would run or follow gets
  repointed.** So instrument-trap rows, dated A/B arms and past-tense CI commentary keep the old name,
  while recipes, `git` commands, workflow triggers and present-tense descriptions of how the project
  works today do not. Do not "correct" a record, and do not run a record's commands verbatim.

  **When sweeping, match the bare word too.** The first pass here searched `origin/master` and missed
  every reference spelled just `master` — including `git fetch origin master` at the head of the
  recipe every subagent is told to start from. #3333.

- **Work in your OWN git worktree — the main checkout is shared.** Several agents (and the human)
  run this repo concurrently, so the main working directory and its build dir are contended:
  branch-switching, staging, or `cmake --build` there collides with whatever someone else is
  mid-edit on (a `git checkout` in a dirty tree tangles the index; a shared `build-linux` races
  object files). Before starting non-trivial work, create your own worktree and stay in it:
  ```bash
  git worktree add .claude/worktrees/<your-slug> -b <your-branch>   # isolated tree + branch
  # build in a worktree-local build dir; pass -DGAME_DUMP=... (dump lives outside the worktree)
  cmake -S prosper -B prosper/build-linux -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0
  ```
  Never `cd` back to the main repo root to run git or builds. If you MUST touch the main checkout,
  assume another agent is actively working there — check `git status` first and don't reset or
  switch branches under them. Don't stash there either — and that one is not a rule about the main
  checkout, it is a rule everywhere in this repository, for the reason in the next bullet.
  - **`git stash` is NOT worktree-local — it is the one piece of state the rule above does not
    isolate.** A worktree gets its own `HEAD`, its own index and its own working tree. `refs/stash`
    is a **single ref in the shared `.git` directory**, so every worktree pushes onto and pops from
    **one LIFO stack**. Two agents each obeying the rule above therefore still collide: on
    2026-09-01 lane A stashed one file so it could rebuild without it, lane B stashed in its own
    tree moments later, and A's `git stash pop` took the top of the stack — B's entry — applying an
    unrelated `hle_kernel.cpp` change into A's tree and **dropping B's entry** on the way through
    (#3174, instrument trap 247). Both agents happened to notice the foreign diff. The silent
    version of that mixes two issues' changes into one branch, or discards a lane's WIP as an
    unreachable, GC-eligible commit — and the rebuild you were about to measure now contains
    somebody else's edit under a banner saying only your own change was removed.
    **So do not `git stash` in this repository.** Park changes somewhere that really is yours:
    ```bash
    python3 prosper/tools/wt_stash.py push -m 'A/B: rebuild without this' src/foo.cpp
    #   ... rebuild, confirm the regression arm fails without it ...
    python3 prosper/tools/wt_stash.py pop
    ```
    It snapshots with `git stash create` (which writes no ref and touches no working tree) and
    stores the result under `refs/worktree/prosper-stash/<slot>` — `git-worktree(1)` § REFS:
    *"refs inside refs/bisect, refs/worktree and refs/rewritten are not shared"*. Slots are
    addressed by **name**, never by top-of-stack, and `push` **refuses** an occupied slot instead of
    stacking, so nothing can be displaced by a neighbour or by yourself. `wt_stash.py check` reads
    the shared `refs/stash` and tells you whether what is on it is yours **before** you touch it;
    `wt_stash.py list --all` finds an entry parked in another tree. Two tool-free alternatives are
    equally safe: a temporary commit on your own branch (`git commit -m wip-tmp`, later
    `git reset --soft HEAD~1`), which touches nothing outside `refs/heads/<your-branch>`, and a
    second `git worktree add` when you want both states at once.
    **Nothing enforces this for you.** A `git` alias cannot shadow a builtin, so a `git stash`
    wrapper is impossible; the optional `wt_stash.py install-hook` works but is off by default and
    lives in untracked `.git/hooks`, so it is absent in every fresh clone. Treat this as a rule you
    follow, in the same family as trap 41's `git checkout <branch> -- <file>`: a git command that is
    perfectly safe alone and destructive when several people share one repository.
  - **Nothing cleans your worktree up. This line used to claim "your worktree is auto-cleaned when
    its branch merges", and that mechanism never existed** — no git hook, no `core.hooksPath`, no
    `hooks` key in any settings.json, no `.claude/hooks/`, no workflow, no timer, and — until the
    sweeper below — no script in the repo that called `git worktree remove` or `prune` at all. The
    claim entered as prose in #332 with no code beside it, and by the time #1735 measured the damage
    there were 121 stale trees / 111 GB; a fortnight later, **278**. The likely origin is a real
    harness feature — subagents launched with `isolation: "worktree"` are cleaned up **if
    unchanged** — generalized into a promise about hand-made `git worktree add` trees, which the
    harness knows nothing about, and stripped of the one condition that made it true. A charter that
    promises a safety net nobody has is worse than one that admits the manual step: everybody skips
    the cleanup *and* nobody files the bug.
  - **So remove your own worktree when your branch merges**, and run the sweeper periodically:
    ```bash
    python3 prosper/tools/worktree_reclaim.py              # census; touches nothing
    python3 prosper/tools/worktree_reclaim.py --remove --yes
    ```
    It refuses any tree that is unmerged, dirty (untracked counts), locked, nested, recently touched,
    or held by a live process — checked against `/proc/*/cwd`, `exe`, open fds and mappings, matched
    by inode so a container's `/home` spelling of a `/var/home` path still counts, and re-checked
    immediately before each individual removal. That last guard is not paranoia: **removing a
    worktree a live shell is `cd`'d into wedges that shell irrecoverably**, and it has happened here.
    It never deletes a branch.
    **What it does NOT protect: gitignored files.** `git status` does not list them, so a tree whose
    only content is an ignored `build-linux/` or a run log reads as clean and is removed with it —
    which is mostly the point, but this charter also tells you to put run artifacts in "a gitignored
    worktree-local directory". Move anything you want to keep out of the tree, or `git worktree lock`
    it. On macOS and Windows the tool classifies but never removes: the in-use guard needs `/proc`,
    and without it the tool cannot prove nobody is inside, so it fails closed.
  - **Teardown is part of finishing a PR, and neither half happens by itself.** When your PR merges,
    delete the remote branch and remove your worktree:
    ```bash
    gh pr merge <N> --squash --delete-branch   # the LOCAL half fails if master is checked out
    git ls-remote origin refs/heads/<branch>   # ...so verify; empty output means it really went
    git push origin --delete <branch>          # and do it explicitly if it did not
    cd <anywhere outside the tree>             # removing the tree your shell sits in wedges it
    git worktree remove <path>
    git branch -d <branch>
    ```
    **A subagent's worktree is yours to remove too.** The harness auto-cleans an
    `isolation: "worktree"` tree only when it is **unchanged** — so a subagent that did any real work
    leaves its tree behind permanently, and the agent that spawned it is the only one who knows it is
    finished.
    Why this is spelled out rather than left to the sweeper: measured 2026-08-20, **49 registered
    worktrees**, from which `worktree_reclaim.py` could reclaim exactly **one** — blocked on **35
    trees classified DIRTY**, i.e. holding uncommitted or untracked files. Disk, and note the
    scoping: `du` reported **101 GiB under `.claude/worktrees/`** and **13 registered trees living
    outside that subtree** carrying ~31 GiB more. Re-measure rather than quoting those: the first
    figure read 89 GiB a few hours later on the same day, because this churns constantly. The
    conclusion is what is stable — well over 100 GB, and the sweeper can reclaim almost none of it. Automation cannot
    fix that and never will, because refusing a dirty tree is the guard that stops it destroying
    someone's work. **So commit or discard before you leave a tree**, or you are the only one
    who will ever be able to clean it up. This line used to say "commit, stash or discard", and
    stashing is the one option that makes things worse rather than better: `worktree_reclaim.py`
    never looks at `refs/stash` (grep it), so a tree whose only work is stashed reads as clean and
    loses the DIRTY protection that was keeping it alive — while the work itself survives only as
    an entry on the repo-wide stack described above, attributable to nobody and poppable by every
    other lane.
- **On a Windows host, run git through PowerShell (Windows git), not WSL.** The repo lives on the
  Windows filesystem (`C:\...` = `/mnt/c/...`), and worktrees created from Windows store a
  Windows-path gitdir link (`gitdir: C:/Users/.../.git/worktrees/<name>`). WSL's git can't resolve
  that path — it mangles it to `/mnt/c/.../C:/Users/...` and dies with `fatal: not a git repository`.
  So drive **git** (fetch/checkout/commit/push/worktree/status) from **PowerShell**, and use **WSL
  only for cmake/build/run**. Don't mix the two on one repo (it also avoids CRLF/filemode index churn).
- **Build/run in WSL Ubuntu-24.04 as root.** Build dir `prosper/build-linux` (Linux, primary),
  `prosper/build-windows` (Windows/MinGW, secondary). Game dump at
  `<DUMP_ROOT>/PPSA24651-app0` (gitignored — **never commit it**).
  ```bash
  cd <REPO_ROOT>/prosper/build-linux
  cmake --build . -j8
  ctest --no-tests=error              # report the discovered count and exit code
  ```
  **Configuring `-DGAME_DUMP` at a title other than `PPSA24651` reports three cases as `(Skipped)`,
  not as failures** (#1573): `module_loads_eboot`, `boot_reaches_first_syscall` and
  `real_shader_render` assert against *The Messenger's* own bytes. CMake derives the dump's title id
  and names both ids in the skip. So a bring-up agent's first `ctest` on a new title is green with a
  smaller executed set — read the skip count, not just the exit code. `docs/VERIFICATION.md` § CI.
  - **Editor/LSP (clangd):** CMake exports `compile_commands.json` into the build dir
    (`CMAKE_EXPORT_COMPILE_COMMANDS ON`), and a symlink at the repo root points clangd at it —
    create both with one configure (`cmake -S prosper -B prosper/build-linux`) plus
    `ln -sf prosper/build-linux/compile_commands.json compile_commands.json`. The database
    regenerates on every configure/rebuild, so there is no separate upkeep; but EACH WORKTREE is
    its own source tree and needs its own configure or its files resolve no includes. Verify any
    setup with `clangd --check=<file> --compile-commands-dir=.` (exit 0 = parsed; tweak-internal
    `ExtractFunction ==> FAIL` lines are clangd noise, not diagnostics).
- **Write run artifacts and build temporaries to the real disk, never `/tmp`.** On the Linux box `/tmp`
  is a **RAM-backed tmpfs sized at 50% of RAM, with a per-user quota shared by every concurrent agent**.
  A single `.prgcap` is 200 MB–2.7 GB, a 4K `.bmp` is 24 MB, and a `PROSPER_GFXLOG`/`PROSPER_DBG` run log
  reaches 1.5 GB, so a few sessions of ordinary capture work exhaust the quota. When that happens it does
  not just fail your write — it **eats the machine's RAM and kills the Bash tool outright for every
  agent**, because the harness stores each command's output under `$TMPDIR`. Put captures, frame dumps,
  screenshots and logs under **`$HOME`** — a gitignored worktree-local directory, or a scratch directory
  such as `~/dq-work/`. **Do not send a run's artifacts to `/var/tmp`: distrobox shares `$HOME` but NOT
  `/var/tmp`**, so anything the emulator writes there lands in the *container's* private `/var/tmp` and is
  invisible from the host (the host path shows up inside the container as `/run/host/var/tmp`). The shell
  redirect that captures a run log happens on the host, so a `/var/tmp` run directory silently ends up
  holding only the log while every screenshot and capsule goes somewhere you cannot see. `/var/tmp` is
  still fine for host-side scratch (it is real disk and `systemd-tmpfiles` ages it) — just not for output
  produced inside the container. Always build with `TMPDIR` off the tmpfs too; gcc's compile temporaries
  are large enough on their own to break a `-j8` build with
  `error writing to /tmp/ccXXXX.s: Disk quota exceeded`:
  ```bash
  mkdir -p <worktree>/prosper/build-linux/tmpdir
  distrobox enter ps5ys -- bash -lc "cd <worktree> && TMPDIR=\$PWD/prosper/build-linux/tmpdir \
      cmake --build prosper/build-linux -j6"
  ```
  This recipe used to name a build directory called plain `build`, which nothing else in this file
  does — and the missing hyphen was load-bearing without anyone knowing it. `check_diag_gates.py`
  excluded build trees by substring on the **absolute** path, so following the recipe's *intent*
  ("`TMPDIR` next to the build") with the build directory the rest of this file names landed you on
  `build-linux/tmpdir`, and the scanner then discarded its own test fixtures and reported that its
  signatures were broken. Three lanes hit that inside five minutes (#2658). The scanner is fixed
  and a ctest case now pins it, so `TMPDIR` may sit anywhere; the recipe is spelled consistently
  here so nobody has to rediscover which spelling was safe.
  **Diagnosing an exhausted `/tmp`:** every Bash call returns exit 1 with no output — including `true`
  and `echo hello`, foreground or background — which looks identical to a dead working directory. The
  tell is the **Write tool returning `EDQUOT`** for a path under `/tmp` (if Write to `/tmp` succeeds, it
  is the cwd instead). It is recoverable from inside the session: the commands still *execute*, only the
  output capture fails, so redirect stdout **and** stderr to a file on the real disk, end with `exit 0`,
  read that file, and `rm -rf` scratch that is a day or more old. Leave `/tmp/claude-*` and anything
  touched in the last few hours alone — other agents are live.
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth
  **only with `--no-tests=error`** — plain `ctest` on a build directory with nothing registered prints
  `No tests were found!!!` and **exits 0**, so "no tests ran" and "everything passed" are the same
  exit code (measured on ctest 4.4.0: plain → 0, `--no-tests=error` → 8). This is reachable without
  any mishap — a copy-pasted `--test-dir` pointing at the wrong build directory yields a green run
  that executed nothing, and an agent reporting "ctest green" has then reported a fact about an empty
  directory. `prosper/tools/vkval/vk_validation_scan.py:206` already passes the flag and its comment
  says why; nothing else in the repo does. **Pass `--no-tests=error` whenever you quote a ctest
  result**, and quote the test *count* alongside the exit code — a count is falsifiable, an exit code
  alone is not;
  every SPIR-V emitter is `spirv-val`-gated in CI (`tools/spv_validate`, §4c — one representative
  module per emitter path, not per game shader); rendered frames are asserted by pixels, hashes, or routed content
  metrics. Snapshots are a **release-time regression inventory**, not a day-to-day development or merge
  gate. Before every release, run the full local-only matrix with
  `python3 tools/snapshot/snapshot.py check` against the release candidate and review every result.
  During normal development, including long stretches of render work, authors may skip snapshots
  entirely or run only useful focused guards; the PR author decides, unless the task explicitly requires
  a run. Snapshot results do not define whether master or an individual PR is acceptable: either may
  regress a guarded title, and a correct fix may even require an intentional cross-title tradeoff. The
  release process is where detected regressions are
  fixed or explicitly accepted and documented before publishing artifacts. Record any snapshots that
  were run in the PR, but do not delay routine iteration merely to keep the whole matrix green. Gameplay
  guards inspect multiple frames in route-specific windows and use SSIM over compact luminance
  signatures from several reviewed states, looking for major collapse without rejecting subtle pixel improvements.
  Every new or materially changed baseline requires `snapshot.py verify`,
  inspection of all retained images from both runs, and a factual `review` note. See
  `tools/snapshot/AGENTS.md`.
  - **A green CI GPU-execution job cannot clear a failure on this machine's driver, because CI has no
    GPU.** `.github/workflows/ci.yml` says so in its own comment: the runner uses **Mesa's lavapipe
    software rasteriser**, whose fragment subgroup is **fixed at 8 lanes**, and the workflow already
    gates `recompiled_shaders_render` assertions that need an exact **64**-lane subgroup. So for
    GPU-execution tests "CI green" is a statement about a **smaller domain** than a local run on RADV.
    This is the positive-control rule two sections above — a control drawn from the same source tests
    the discriminator, not the domain — applied to an entire CI stack.
    Measured 2026-09-05 (#3355): five GPU-execution tests — `recompiled_shaders_render`,
    `multidraw_render`, `indexed_render`, `descriptor_array_render`, `gpu_execute` — fail on RADV from
    **both** a reused and a fresh build directory on a verified-idle box, set-equal across two lanes'
    different heads, branches and dumps, while the same five pass in CI. Four of the five are
    indexed-draw or GREEN-quad assertions: draws that render nothing.
    **Two agents independently read that CI log as proof their change was not implicated, and both had
    to withdraw it** — CI reads as neutral infrastructure, which is exactly why nobody asks whether it
    can run the case. The workflow's own instruction is the right one: *"If a GPU test fails here,
    diagnose it."* The gap is not GPU-only: a change dispatched on a host CPU feature (F16C, for
    instance) may take a different path on the runner, so a green suite is not evidence the vectorised
    path ran.
    **The prior runs the other way, though, and it is in the same comment block:** four tests were
    once excluded here as "driver-divergent" (#1681) and **none of them were** — every divergence was
    measured and each time it was prosper's own assertion or fixture that overreached, which is why
    they are gated rather than excluded today. So a local-only GPU failure is a reason to *diagnose*,
    not to assume the driver is at fault; the honest open question is whether the assertion or the
    backend is wrong. Note that history was measured entirely in the lavapipe direction, so it does
    not transfer unexamined to a failure only 64-lane hardware can reach.
- **Per-title bring-up ladder — gameplay is always the target.** Every game climbs the same ladder,
  and progress claims name the rung reached:
  1. **Any graphics at all** — real frames from the live renderer (a splash/logo counts; black or
     diagnostic-only output does not).
  2. **Title screen** reached and rendered.
  3. **Gameplay** reached with real GPU draws **and the gameplay scene actually rendering** — this
     is the standing goal of bring-up; don't stop and declare success at a logo or menu.
     **Execution alone is not rung 3.** A title whose guest reaches the game loop while the world
     is absent — a HUD and a running clock over black — stays at rung 2, with the milestone text
     saying so: it has genuinely got further than a title stuck at a menu, and the *text* is what
     carries that, not the number. The bar is "somewhat correct rendering", deliberately loose:
     a scene a person would recognise as the game. Degraded still counts (*Syberia: Remastered*'s
     composite, *The House of the Dead 2*'s world defects); absent does not (*Grand Theft Auto V*,
     *Sonic Frontiers*). Rung 4 remains the stricter "a human confirms it looks right".
  4. **Manual visual verification** — the user plays it and confirms by eye that it looks right.
  5. **Oracle comparison** — reference screenshots from PS5 hardware, compared against routed
     captures for visual correctness.
  6. **Automatic snapshot testing** — a deterministic route + content guard added to
     `tools/snapshot`, so regressions are caught without a human.
  Only after step 6 is a title considered **"done"**, and work moves to the next game. Earlier rungs
  are milestones worth recording (issue/`COMPATIBILITY.md`), not stopping points.
- **Reach for free vendor/open-source tooling BEFORE building a `PROSPER_*` switch.** prosper is an
  ordinary Vulkan application, so AMD's and Khronos' own tools work on it directly and need no
  per-title work. **Measured on Linux/AMD 2026-08-08 with no change to prosper's code**, stated as
  what was actually observed rather than as what the tools promise:
  - **An RGP capture succeeds** from one variable (`MESA_VK_TRACE=rgp`), and RADV reports
    `instruction timing: enabled, cache counters: enabled, queue events: enabled` on it. **Reading a
    `.rgp` needs AMD's GUI — nobody here has opened one**, so treat its per-draw timing and occupancy
    as the tool's documented capability, not as a project measurement.
  - **`radeontop` answers "is this GPU-bound?" in one run, no capture and no GUI.** *Blue Prince* at
    ~3.2 fps leaves the GPU at **4.17%** against a `vkcube` control at **56.31%** — so with #2215's
    30-45 ms/submit, its submits are long because they **wait**. That control is mandatory here: the
    tool prints "Unknown Radeon card" and its `ta`/`ee` counters are dead on this chip.
  **`docs/GPU_PROFILING_EXTERNAL.md` has the recipes and the way each tool lies** — RGP writes to
  `/tmp`, a frame ordinal silently captures the wrong regime, and the Fedora RenderDoc package ships
  no Python bindings. Build a `PROSPER_*` switch only for something the guest-facing layer knows and
  the GPU vendor cannot see.
- **Build tools — don't avoid them.** This project is a long reverse-engineering effort, and getting
  progressively more complicated games running will keep demanding new instrumentation. When you hit a
  question the existing tools can't answer — "who references this address in the unsymbolicated eboot?",
  "what managed method is at this PC?", "which draw wrote this pixel?", "what does this guest struct look
  like at runtime?" — **write the tool** rather than hand-grinding it once and moving on. The payoff is
  compounding: a good tool turns hours of manual disassembly into a lookup and pays off on every future
  title. Prefer a small, documented, committed tool (in `tools/`, or a gated `PROSPER_*` diagnostic in the
  emulator) over a throwaway script, and reuse what exists first (`self_dump`, `boot_trace`, `PROSPER_HWBP`/
  `HWWATCH`/`PEEK`/`FAULTMEM`, `tools/il2cpp/prx_to_elf.py` + Il2CppDumper, `tools/re/xref.py`, the snapshot
  guard) — the RE toolbox is the force multiplier, so grow it deliberately, verify it on a known answer,
  and land it so the next agent (and the next game) inherits it.
  - **For a graphical bug or an FPS drop, the fastest loop is the F9 frame grab.** While a title runs in
    `prosper-app`, pressing **F9** captures the current frame — the game's real GPU commands, shaders, and
    resources, plus the renderer-owned RTTs it samples (seeded) — into a replayable `.prgbundle` + a `.bmp`
    screenshot (`PROSPER_CAPTURE_DIR`, default cwd). Then `tools/gpu_replay --bundle <file>` reproduces
    that exact frame **offline and deterministically**, and `--inspect-only` / `--draw-steps` /
    `--dump-resource` (via `--bundle-extract-submit`) dissect it draw-by-draw. This answers "which draw
    wrote this pixel / why does this frame look wrong / why is it slow" without re-booting and re-routing
    to catch the moment live — press F9 once, then iterate the fix on the frozen frame.
    It captures *rendered-frame* bugs (not CPU/logic/audio). See `tools/AGENTS.md` (interactive frame grab)
    and `tools/gpu_replay/README.md`.
  - **You do NOT need a human at the keyboard: both captures are schedulable, and an agent can run the
    whole thing headless.** This is written here because it was missed for months — the triggers live in
    `prosper-app`, which is **off by default** (`-DPROSPER_APP=ON`), so an env sweep of `src/` never sees
    them, and lanes built weaker instruments instead. Two rich diagnostics, four one-shot triggers:

    | capture | after a wall-clock delay | at a host frame |
    | --- | --- | --- |
    | **F9** replayable `.prgbundle` + `.bmp` | `PROSPER_GRAB_BUNDLE_AFTER_MS` | `PROSPER_GRAB_BUNDLE_AT_FRAME` |
    | **F8** bounded `.prperf` (5 s before + 5 s after) | `PROSPER_PERF_CAPTURE_AFTER_MS` | `PROSPER_PERF_CAPTURE_AT_FRAME` |

    ```bash
    cmake -S prosper -B build -DPROSPER_APP=ON          # the app is NOT built by default
    SDL_VIDEODRIVER=offscreen \                         # LINUX ONLY -- see below
    PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    PROSPER_CAPTURE_DIR=$HOME/work \
    PROSPER_GRAB_BUNDLE_AFTER_MS=18000 PROSPER_PERF_CAPTURE_AFTER_MS=18000 \
        ./build/prosper-app <DUMP_ROOT>/<TITLE_ID>-app0
    ```

    **`SDL_VIDEODRIVER=offscreen` is Linux-only. Drop it on Windows** — the triggers themselves are
    platform-independent and fire correctly there with an ordinary window. SDL's offscreen driver needs
    `VK_EXT_headless_surface`, which NVIDIA's Windows Vulkan driver does not expose, so the app dies
    before writing anything:

    ```
    Installed Vulkan doesn't implement the VK_EXT_headless_surface extension
    [app] SDL_Vulkan_CreateSurface: VK_EXT_headless_surface extension is not enabled ...
    terminate called without an active exception
    ```

    That signature is recorded because it does not look like what it is: no artifacts and a
    non-graceful exit read as "the capture triggers are broken", when the triggers were never reached.

    Which axis to use: **`_AFTER_MS` aims at an event you can only describe in time** ("when the movie
    starts", "when the rate collapses"); **`_AT_FRAME` aims at an ordinal you already located** with a
    cheap `tools/screenshot` sweep or `PROSPER_CAPTURE_SCREENSHOT_AT_FRAME`, and is the reproducible one.
    Point both at the same moment to get the timing breakdown and the replayable frame together.

    Each is **one-shot and opt-in**, and a malformed value **disables its own trigger** rather than firing
    at an unintended moment — so a typo costs you a capture, never a wrong measurement. Read the
    `.prperf` with `tools/perf/performance_capture_report.py` (it reports by **time**; the record counts
    are not load), and the bundle with `tools/gpu_replay --bundle`.

    **Read the F8 report before optimising anything — and check the capture can SEE the leaf you are
    reading.** Worked example (#2215, 2026-08-07): Blue Prince at 1 fps, and the capture said
    `gpu-device` was **2.2%** of the frame while `renderer-resource` was 47%. That half stands.

    The same example used to end "and `buffer copy` was **2.4%** during the collapse". **That number was
    wrong, and this line kept publishing it for a day after it was retracted.** The capture predated
    #2243's backend sub-buckets, so the only buffer figure in it was the *frontend* materializer's,
    while the leaf two lanes were actually optimising — `res_buffer_copy_ms`, the backend memcpy at
    `tests/fixtures/render_runner.h:5199` — was invisible to it. Re-captured with the sub-buckets live and the
    collapse confirmed by flip rate (1.59/s): `setup_resources 801.9ms [buffer=673.4 (copy=539.2) …]`,
    i.e. **~67%**, not 2.4%. Two independent instruments agree it is large in the collapsed regime and
    on both platforms — `PROSPER_RENDER_TIMING` gives copy/backend **15.7%** on Linux and **18.5%** on
    Windows, and `perf` puts the same leaf at 17.6% of the saturated Linux worker.

    **The lesson is not the arithmetic, it is the propagation.** The 2.4% was corrected on #2215 the
    same day it was posted; the charter was not, so the retracted figure went on steering people from
    the document they read *first*, and the other lane reports it is what sent them chasing five dead
    hypotheses. A correction that lives only in an issue thread is not a correction to the charter —
    if a number here came from your measurement and your measurement changes, the charter is part of
    the fix, not follow-up work. **A stale figure carries the charter's authority, which is exactly the
    authority it no longer deserves.**
- **Reaching the running frame loop:** `PROSPER_GUEST_ARGS=-force-gfx-direct`, plus `PROSPER_RENDER=1`
  to run the live renderer and `PROSPER_GFXLOG=1` for graphics diagnostics.
  - **`PROSPER_GUEST_FS=1` is NOT needed on Linux or Windows, and this line used to say it was.** Guest
    initial-exec TLS is **enabled by default** there; the environment variable actually read is the
    **opt-OUT** `PROSPER_NO_GUEST_FS`, kept for compatibility bisection (`src/host/tls/guest_tls.cpp:58`,
    `:240`; `src/hle/dispatch/dispatch.hpp:218`). `PROSPER_GUEST_FS` is never read as an env var on those
    platforms — setting it is harmless but does nothing, and *believing* it is required is not: it
    turns a default-on path into one people think they are enabling, so nobody checks it when a guest
    TLS problem is the actual cause. On **macOS/Rosetta** `PROSPER_GUEST_FS` does remain the opt-in for
    trap emulation, which is where the confusion came from.
  - Recorded because of how far a wrong default travelled: this line propagated into a review's wording,
    then a briefing, then a shipped code comment, then an author-verification comment — with nobody
    opening `guest_tls.cpp` at any step (#2049). **A default stated in this file is the kind of claim
    every reader inherits without checking; verify against the code before restating one.**
- **Do not reuse snapshot acceleration for interactive or performance runs.** `PROSPER_RENDER_SCALE>1`,
  `PROSPER_RENDER_EVERY>1`, and `PROSPER_RENDER_EVERY_FOR_MS` deliberately reduce resolution or skip
  graphics submits. Keep the defaults (`PROSPER_RENDER_SCALE=1`, `PROSPER_RENDER_EVERY=1`, and no timed
  sampling window) unless the run is explicitly a headless capture or sampling experiment.
- **Correctness-first:** implement real behavior from primary evidence: live captures/traces, guest
  disassembly, published platform contracts, firmware symbol data, and focused tests. Do not ship shims
  that fake output. Mark genuinely uncertain code with `CONFIDENCE: HIGH/MED/LOW`.
- **An unsupported shader/GPU operation is a FATAL gap, not an acceptable skip — support ALL of them.**
  A recompiler op the shader stage uses, a storage/texture format a dispatch writes, an AGC/PM4 packet
  the guest submits: if the guest exercises it, the goal is to implement it, not to `return {}` / skip the
  draw / skip the dispatch and move on. Silent skips drop real rendered content (a skipped LUT/exposure
  dispatch collapses the whole title composite to black; a rejected shader drops its draws) and read as
  "handled" when they are not. Reject paths still exist as a *fail-visible* backstop for genuinely
  unknown encodings (mark `CONFIDENCE: LOW`, log loudly, file an issue with the exact opcode/format), but
  treat every one you hit on a live boot as the next thing to implement. Find the exact failing op with
  `PROSPER_DBG=1` (`[recompile-reject] pc=… op=0x…` from the recompiler) or the `[compute] skip …`
  lines from the live backend, then implement it with a round-trip/execution test — do not leave it skipped.
- **Entitlement and add-content APIs answer from LOCAL INVENTORY — never blanket-approve.** When a
  title asks whether an add-on or entitlement is owned, prosper answers from the content actually
  declared and present locally (`src/hle/service/hle_addcontent.cpp` parses real entitlement labels and keys
  and rejects malformed or duplicated ones). That is the right design for a preservation tool, and it
  is deliberate: a compatibility layer must keep working when the licensing service it would
  otherwise consult no longer exists, so the answer has to be derivable offline.
  **What must never happen is making these calls return "owned" unconditionally to get a title past a
  check.** The distinction is not cosmetic. Faithfully reimplementing a *platform* query — what system
  software this is, whether a user is signed in — is exactly the job, the same way Wine answers
  `GetVersionEx`. Manufacturing a positive answer to an *ownership* query is not reimplementation; it
  is the emulator performing the circumvention itself, and it would make prosper a circumvention tool
  rather than a compatibility layer no matter how the code were phrased.
  This is written down because the failure mode is attractive: a title stalls at a content check, one
  `return 1` makes it progress, and the change looks like progress in a screenshot. If a title cannot
  advance because content is genuinely absent, that is the honest result — record it as the blocker
  and, if the content exists locally, wire it through the inventory. `CONFIDENCE: HIGH`.
  - **The positive half of the same rule, and it is an obligation: every feature and every piece of
    content that IS in the dump must be reachable.** Under-reporting what is present is as much a
    defect as over-reporting it would be a circumvention — it just fails in the direction that looks
    safe, so nobody files it. A stub that answers a local-inventory question with 0 silently removes
    content the user has, and the title is then behaving *correctly* on a wrong answer, which is the
    hardest kind of bug to see: the game's own UI explains the problem in the game's own words, so it
    reads as a licensing wall rather than as our defect.
  - **The test that separates the two cases is whether the answer is derivable from bytes in the
    dump.** "What SKU is this installed application?" is a *platform* query about local content —
    `param.json`'s `contentId`, `applicationCategoryType`, `applicationDrmType` all state it offline,
    so reimplement it faithfully, exactly as `GetVersionEx` is reimplemented. "Did this person buy X?"
    is not derivable from anything local, and no amount of engineering makes it so — stop there and
    record the cap. Derive; never hardcode a full/owned answer. A dump that is a free or trial SKU
    must report that SKU, and the regression test proving you derived rather than hardcoded is a
    **mutation arm where a synthesized free-SKU declaration yields the non-full answer**. Without that
    arm the change is indistinguishable from `return <full>`.
  - **Answer the same question the same way through every library that exposes it.** Sony surfaces SKU
    and add-content state through more than one library, and a divergence between them is a defect
    even when each path looks defensible alone. Worked example (#1873): `hle_service.cpp`'s
    `sceAppContentAppParamGetInt` answered `SKU_FLAG` as full while
    `sceNpEntitlementAccessGetSkuFlag` was unregistered and fell to the return-0 stub — so the answer
    depended on which library the guest happened to ask through, and *Grand Theft Auto V* asks through
    the NP one. Put the derivation in **one** place both paths call; if two paths can still disagree
    after a fix, the defect has moved rather than gone.
- **PR verification and merging.** Keep each PR description self-contained: link the issue or goal, explain the
  failure scenario and behavioral contract, summarize the approach and important invariants, identify affected
  and deliberately unaffected behavior, record risks, and list the exact build/test/snapshot commands and
  results. PRs that advance a game's visible progression must attach representative screenshots and identify
  the title, platform, and reached checkpoint; black or diagnostic-only captures are not progression evidence.
  Use direct frontend captures and name the frontend plus the run route in each caption; output
  produced by forced guest-state diagnostics may illustrate an investigation but is not acceptance evidence.
  - **What "unaltered" forbids is misrepresenting the progress, not annotating it.** The rule exists so a
    screenshot cannot be made to show more than the emulator actually achieved. So: no retouching, no
    compositing several frames into one, no substituting artwork or an icon for something that did not
    render, no cherry-picked frame passed off as the steady state, and no forcing guest state to
    manufacture a better-looking frame. **Annotation that prosper itself draws over its own output is
    fine and often better** — an fps counter, a timestamp, a route or frame ordinal. Those add measured
    facts about the run; they do not change what the run rendered. Keep annotation legible and out of
    the way of whatever the capture is evidence *for*, and say in the caption that it is on.
  - **A PR that adds a screenshot adds a `BLOG.md` entry in the same PR.** `BLOG.md` is the progress
    blog — hand-written, newest first — and it is the file the project owner actually reads to see
    what is new. It is **not** generated, so nothing backfills a missing entry and nothing goes red
    when one is skipped: the picture simply never surfaces, which is the whole failure this file
    exists to prevent.
    **It is a screenshot blog: image-heavy, text-light, always.** The rules used to live in
    `BLOG.md`'s own header, which put authoring instructions in front of every reader who opened the
    file to look at progress. They live here now.
    - **Pictures: as many as you have.** Every screenshot the PR produced goes in the entry. An entry
      covering eight titles shows eight titles, not one — that happened, and the other seven captures
      sat in `assets/screenshots/` invisible to the only audience they existed for. A picture costs a
      reader nothing to skip and everything to not have. Never economise here.
    - **Economise on BYTES, never on count.** A committed capture is a **1920-wide WebP**, not a 4K
      PNG — run `prosper/tools/screenshots/shrink.py <file>` before committing it. Measured
      2026-09-02: `assets/screenshots` was **398 MiB of a 621 MiB pack** while all source, docs and
      tests in the entire history came to **18 MiB**, so images were ~96% of the repository, and the
      cause was resolution rather than quantity. Re-encoding the same 140 frames took them to
      **14.2 MiB** with small UI text still legible. At the old size GitHub's 10 GB ceiling arrives
      in about four years; at this one it does not. This does not soften the bullet above — show
      every picture you have, just do not ship each one at 2.5 MiB.
    - **Words: one sentence saying what now works.** *"We now reach correct rendering of the Sonic
      Frontiers title screen."* Then the screenshot. A second sentence only if the picture misleads
      without one — a defect you want noticed, or a caveat about what is being seen. Say what you are
      **looking at**, not what the change was; the commit subject already says what the change was,
      and that is exactly what made the old generated feed useless.
    - **One exception: a fix that moves several titles at once** earns a short plain-language
      paragraph on the problem and the fix, because those are the ones a reader wants explained. Still
      lead with the pictures. A single-title fix links its tracker instead.
    - Everything else — measurements, bisects, censuses, mechanisms, reasoning — belongs in the
      tracker, the PR or the status doc. Link, never transcribe. If an entry has a table, a code
      block, or more than one or two numbers, it is a report wearing a blog entry's clothes. Six
      entries filed in one day ran 330-946 words against the file's own median of about 60, and the
      owner noticed before any check did.
    Write it even when the news is bad or the picture is ugly: *Sonic Frontiers*' black-world capture
    is in there precisely because the honest record is the point. And write it even when the rung did
    **not** move — a capture of a title that got further and still fell short is worth more than
    silence.
    **A finding with no picture is welcome too**, and this is the half that gets skipped: a lane that
    measures something surprising and ships no screenshot currently leaves no trace anywhere the owner
    reads. *The Plucky Squire*'s cutscene turning out to advance 300x too slowly — because the guest's
    tick rate collapses 147x when the 3D world streams in, and in-game time advances per flip rather
    than per second — is a better read than most screenshots, and it went unwritten because the PR
    happened to contain no image. Keep those short, and skip them when there is genuinely nothing a
    reader would enjoy; a blog nobody wants to read is worse than a thin one.
  Run the strongest relevant local checks and wait for every applicable required CI check. Before
  merging, synchronize with the live target branch when needed, inspect the resulting diff, run `diff --check`,
  and address every known correctness concern.
  - **Exception — a documentation-only PR may be merged immediately, without waiting for CI.** If the diff
    touches nothing but `.md` files, there is no build, no test and no snapshot that CI can tell you
    anything about, so waiting on it only slows the queue. Merge it.
    **Run the docs gate locally first** — it is the one check that can actually fail on a docs diff, and
    it takes a second:
    ```bash
    F=prosper/docs/GAME_COMPAT_ORCHESTRATION.md
    B=$(mktemp)                    # mktemp, not a fixed name: several agents run this concurrently
    git fetch origin -q && git show "origin/main:$F" > "$B"
    python3 prosper/tools/docs/check_numbered_table.py --ordered \
        --table-header Instrument --baseline "$B" "$F"
    git ls-files '*.md' -z | xargs -0 python3 prosper/tools/docs/check_numbered_table.py
    ```
    This matters most for `GAME_COMPAT_ORCHESTRATION.md`'s numbered tables. The gate requires rows to be
    unique and ascending, and requires that **no row present in the base is missing** (`--baseline`).
    Note the local baseline is `origin/main`, while **CI passes `HEAD^1`** — the base *as of when the
    event fired*. `origin/main` is the fresher and therefore stricter of the two, which is why it is
    the right one to check by hand; the gap between them is the subject of instrument trap 189.
    **Gaps are legal** — `--sequential` and its gapless rule were removed on 2026-08-17 (#2089) because a
    gap is usually another lane's unmerged number, which no edit of *your* branch can supply, so the check
    served only to serialize independent lanes; the flag now errors and names its replacement. So if you
    lose a collision, renumber to any number above the current highest and merge — you no longer have to
    wait for the number below yours. Before writing a row, allocate with
    `python3 prosper/tools/docs/trap_number.py`, which scans `main` **and every open PR** (#1729);
    reading the base branch alone is what produced the #2574/#2581 collision.
    **On a PR that touches a numbered table, run the gate against the MERGE RESULT immediately before
    merging** — `python3 prosper/tools/docs/check_merge_result.py` fetches, merges into
    `origin/main` in the object database (no checkout, no index), and runs the gate on the result.
    A green `Docs` job is a statement about a merge that may no longer exist, and unlike a red one
    nobody re-derives it (#2211, instrument trap 189); this is the only check that sees a concurrent
    append, and it is stale the moment another lane lands, so run it last and merge promptly.
    Confirm the diff really is `.md`-only — `git diff --name-only origin/main...HEAD | grep -v '\.md$'`
    should be empty. The exception is about the *diff*, not the intent; one stray file makes it an
    ordinary PR again.
    **This waives the CI wait only.** It does not waive independent review where review is warranted — a
    docs PR that rewrites a `## Ruled out` row, a reproduction route, or a numbered table's semantics can
    be just as wrong as code, and no CI job has ever been able to see that. It also does not waive
    `diff --check`, the trap-41 deletion check, or the session-trailer gate. An agent may merge only when the user or task explicitly
  authorizes it. Each agent owns its PR through merge and branch cleanup: before claiming or starting new work,
  merge the current PR after all gates pass, or explicitly close it with a clear rejected/superseded explanation.
  Do not accumulate floating PRs or silently hand merge ownership to another agent. The PR author owns
  verification; prefer `powershell -File prosper/tools/verify-pr.ps1 core` or,
  for renderer changes, `powershell -File prosper/tools/verify-pr.ps1 renderer -Snapshot <name>`, and post the
  exact-head results.
- **Use independent review whenever it can materially improve confidence.** Treat review as a correctness
  tool, not merely a gate for large diffs: behavioral code changes involving non-obvious judgment, unfamiliar
  code, cross-platform behavior, reverse-engineered semantics, or meaningful failure paths should normally be
  reviewed even when the patch is small and the author verification is green. Complex or high-risk changes—such
  as shader/recompiler
  control flow, memory safety or untrusted bounds, synchronization, ABI-sensitive interfaces, persistent data,
  shared renderer/executor format, size, or indexing semantics, or changes whose failure can silently corrupt
  results—require an independent code review before merge. Review is also required when the validation itself
  is easy to get wrong: for example, a regression test with another path that can produce the expected result,
  or a snapshot route or baseline update that needs judgment to confirm it exercises the intended behavior.
  Judge both implementation risk and verification risk, not merely the diff size. Skip independent review only
  for genuinely routine, mechanical, well-bounded low-risk work where another reader is unlikely to uncover a
  correctness problem; when in doubt, review. Passing author verification does not substitute for
  review where review is warranted: verification demonstrates observed behavior, while review challenges the
  contract assumptions, untested paths, and whether the regression actually proves the intended behavior.
  When uncertain, favor review for reverse-engineered API semantics and other changes where an independent
  reader can materially challenge the reasoning, even when the patch is small. The reviewer inspects the
  assumptions, code, risks, and tests, including whether each regression test would fail without the fix, but
  does not duplicate the author's builds, tests, snapshots, or CI jobs. Once the author is satisfied and the PR
  is published, run author verification and any required review; the author posts exact-head evidence and the
  reviewer posts approval on the corrected exact head. Address every blocking finding and re-review authored
  corrections, then merge only after the merge agent separately confirms author verification, reviewer approval
  where required, and green required CI.
  - **A cited claim reads as a verified one — and reviewers' findings propagate hardest.** A review
    finding carries more authority than the same sentence anywhere else: it has been *checked* by
    definition, and attaching a `file:line` makes it look checked twice. So a **wrong** finding with a
    citation is the most contagious kind of error this project produces. Worked example (#2049 → #2052):
    a reviewer stated an inverted default with a file:line, and that citation is what made it credible
    enough to travel unchallenged into a briefing, then a shipped code comment, then an author's own
    verification comment, then roughly fifteen agent briefings — with **nobody opening the cited file at
    any step**. It was `guest_tls.cpp:46`, which is the Apple-only branch; Linux is `:58`, opt-**out**,
    default **enabled**.
    **So: check a claim against the source, never against anything downstream of the thing being
    checked.** Opening the cited file and running one `grep -rn 'getenv("…")' prosper/src` is the whole
    defence, and it is two commands. This binds on reviewers most of all — when you cite a line, you are
    asserting you read it.
    - **The asymmetry that makes this worse: a finding you WANT to be true gets checked least.** A
      correction that *lowers* a claim is scrutinised, because it costs the author something. A
      correction that *raises* one is waved through, because it agrees with what the author already
      believed — and it lands in a `## Ruled out` row wearing a citation, where nobody is motivated to
      re-derive it. Worked example (2026-08-06): an author marked a VOP3P `NEG` semantic
      `CONFIDENCE: MED`, a reviewer volunteered "published support you hadn't cited", and the author
      raised it to `HIGH`. A second reviewer traced the citation: the reference says the modifier is
      *"valid for floating-point operands only"* — a **restriction**, cutting against the inference —
      and the LLVM fold came from a PR merged and reverted the same day, covering a different
      instruction family. The honest label was the original one; the author had been talked *up* from it.
    - **The multiplier is the citation itself.** A bare "I think you're being too conservative" would
      have drawn scrutiny. A published field description plus a compiler fold read as *already checked* —
      the same authority-transfer described above, now pointed at the thing the author hoped was true.
      **So trace a citation in proportion to how much you want it to be true, and treat "this supports
      raising confidence" as a trigger for verification rather than a reason to skip it.**
    - **Which claims get questioned is not a matter of care, and instrument trap 231 measures it.**
      Over one review, two of a reviewer's phrases reached the author's tree underived and were
      caught only because they sat inside a paragraph being rewritten for an *unrelated* reason. No
      question was ever aimed at the reviewer's numbers. The defence that does not depend on
      somebody suspecting something is to **compute any load-bearing quantity twice, by independent
      routes** — in the same review that is what caught a third wrong figure before it propagated at
      all.
    - The general shape all of these share is worth naming, because it is not "someone was wrong": it is
      **a true statement reached by a route that does not establish it.** The claim survives casual
      checking precisely because it is defensible; what fails is the derivation, and nobody re-runs a
      derivation whose conclusion they already accept. Three instances landed in one day — this citation,
      a `v_med3_f16` lowering correct only for finite inputs, and a pair of test arms asserting a property
      their inputs could not exercise (#2130).
  - **How a verdict is expressed and detected — this is mechanical, and getting it wrong has merged a rejected
    PR.** The reviewer posts one or more **registered reviews** with `gh pr review <N> --comment --body '…'`,
    each stating a verdict as a literal **`APPROVED`** or **`REJECTED`** in the body. The author reads the
    comments, fixes what is raised, and merges only once a review says `APPROVED` **on the current head**.
    Two mechanical facts make this the only workable scheme here, and both are invisible until they bite:
    - **`gh pr comment` does NOT register as a review.** It creates an *issue* comment, fully visible on the PR
      page and absent from `gh pr view --json reviews`. A verdict posted that way is invisible to anyone
      checking programmatically. Several reviews were posted this way before it was noticed.
    - **GitHub refuses `--approve` on a PR opened by the same account**, and every agent here operates as the
      repository owner. So the review *state* is **always `COMMENTED`**, for approvals and rejections alike, and
      `reviewDecision` stays empty. **The GitHub review state carries no verdict in this repository. Never gate
      on it, and never treat "a review exists" as "a review approved".** A merge daemon that counted
      `reviews > 0` merged #2026 with two blocking findings outstanding; the revert is #2044. The verdict lives
      in the prose, so **a human or an agent must read it** — this specific check cannot be automated here.
    - Corollary for the merge step: **a rebase or a new push detaches every review from head.** `reviews` still
      reports rows, with a `commit_id` that is no longer on the branch. Compare each review's `commit_id`
      against the current head; if it is stale, establish by **content** (`git diff <reviewed-sha> <head>`)
      whether the approval still applies rather than assuming in either direction.
    - **The same detachment applies to CI CHECKS**, which the bullet above does not cover: a branch
      pushed after its checks started leaves `gh` reporting green for a commit that is no longer
      what would merge, and every lane here pushes after CI starts. So compare `headRefOid` against
      `git ls-remote origin refs/heads/<branch>` before merging, not only the reviews' `commit_id`.
      One command, and the state is routine. (As of 2026-09-02 no merge here is known to have been
      lost this way; #3259 first cited #3243 and that was wrong on the dates. The rule stands on the
      hazard, not on a scar — full account in `prosper/tools/ci/AGENTS.md`.)
    - **Never gate on the TEXT output of `gh pr checks`.** Its columns are tab-separated and this
      repository's check names contain spaces — `Windows App`, `Linux App Package`,
      `macOS (x86_64 / Rosetta 2)` — so splitting on whitespace puts the second *word of the name*
      where the status belongs. A shell gate doing that merged #3234 with a red `Windows MinGW`;
      an `awk '$2=="pass"'` written the same day counted 2 passing checks where 8 passed. Both
      answers look plausible, which is the whole danger. Use `--json name,state,bucket`.
    - Both rules, plus "an empty check list is VOID rather than green", are implemented in
      `prosper/tools/ci/pr_merge_gate.py` — `python3 prosper/tools/ci/pr_merge_gate.py <PR>` exits 0
      only when every check passed, none is pending, at least one passed, and the PR's recorded
      head is still the branch tip. It exits **2** when it could not evaluate, so "the gate could not
      run" is never confused with "the gate said no".
- **Unpublished desktop-app parity is not a merge requirement.** A Linux-, Windows-, or macOS-specific app
  improvement may merge without matching work in every other frontend unless the issue explicitly promises
  parity or a shared public contract requires it. Unaffected platforms must still retain existing behavior and
  pass their applicable checks; track desirable parity separately rather than blocking the focused change.
- **Evidence hierarchy and independent implementation.** Trust sources in this order: (1) prosper's
  live captures/traces of the real guest, (2) published platform contracts, firmware symbol data, and
  the guest's own disassembly, (3) agreement among independently written secondary implementations,
  then (4) a single secondary implementation as a hypothesis only. PS5-specific AGC, Gen5 descriptors,
  and PS5-only kernel calls require direct title evidence; inherited PS4 behavior still must be checked
  against the exercised guest path. External implementations are verification-only: do not copy or port
  their code, types, comments, prose, or tests. Re-derive behavior in prosper's own architecture and add
  project-owned evidence/tests. Never weaken behavior demonstrated by a live boot to match a secondary
  reference, and mark unresolved evidence with `CONFIDENCE: HIGH/MED/LOW`.
  - **Exception — vendoring permissively-licensed standalone libraries.** The "re-derive, don't copy"
    rule targets other *emulators'* reimplementations of Sony HLE interfaces (clean-room integrity). It
    does **not** forbid vendoring a general-purpose, permissively-licensed (MIT/BSD/zlib), self-contained
    library for a well-defined standalone problem — a codec, hash, math kernel — when that library is
    itself clean-room (no Sony code/keys) and re-deriving it would be disproportionate. Such a dependency
    is a deliberate project-owner decision, vendored **verbatim** under `third_party/<name>/` with its
    LICENSE and a README recording origin/version/why; prosper's own glue (parsing, wiring into the guest
    path) stays project code. First instance: `third_party/libatrac9/` (MIT, ATRAC9 decode — a large
    codec, not a Sony interface). Do not use this exception to shortcut Sony ABI/HLE work.
- **AMD RDNA 2 shader ISA reference:** ["RDNA 2" Instruction Set Architecture: Reference Guide
  (document 70648)](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content). Consult it
  for instruction encodings, operand and condition-code semantics, wave behavior, and memory-instruction
  details when working on the shader recompiler or GPU diagnostics. Treat it as the primary published RDNA 2
  architecture reference; PS5-specific extensions, encodings, and AGC behavior still require prosper's live
  title evidence and focused tests.
- **PS5 3.20 firmware library reference — the definitive NID↔name database (`../PS5-3.20_Libs/`).**
  A `genstub.py`-generated dump of **all 275 PS5 3.20 system libraries**, one `libSceXxx.c` per library.
  Each file lists **every exported function AND its exact NID**: the loader lines read
  `sprx_dlsym(__handle, "<NID>", &__ptr_<funcName>)`, so each is a `<NID> ↔ <funcName>` pair. This is
  the **authoritative PS5-specific symbol map** — trust it over PS4-era symbol lists and secondary tables for
  any Gen5/PS5-only surface (AGC, Ampr, Pad/UserService Gen5, etc.), and use it to see a library's
  *complete* real API surface (e.g. what functions exist that a title might call). Recipes:
  - Resolve an unknown NID → name: `grep -rn '<NID>' ../PS5-3.20_Libs/` (the matching `__ptr_<name>` names it).
  - A library's full export list: `grep -oE '\.global sce[A-Za-z0-9]+' ../PS5-3.20_Libs/libSceXxx.c | sort -u`.
  - Which library exports a symbol: `grep -rl '<funcName>' ../PS5-3.20_Libs/`.
  Caveat: it gives **names + NIDs only, no bodies** — argument layouts and behavior still come from
  prosper's live captures, published contracts, and guest disassembly. Gitignored sibling of the repo;
  never commit its contents.
- **Commit style:** small, verified commits; push to `origin` promptly. Co-author trailer as configured.
  **Do NOT add "Generated with Claude Code" attribution lines** (the 🤖 badge, "Generated with
  [Claude Code](...)" footers, session links) to PR bodies, commit messages, issue text, or
  comments — anywhere. They are noise in the project record; the co-author trailer alone is enough.
  - **This one needs an explicit check, because the harness fights it.** Several agents run here with a
    default that appends a `Claude-Session: https://claude.ai/code/session_…` trailer to every commit
    message; this rule overrides that default, but the default is silent and re-applies on every commit,
    so following the rule by intention alone does not work. Measured per-commit on 2026-08-01:
    **29 of the last 100 commits on `master`** carry one.
    ```bash
    # Gate before pushing — answers "any?", expect 0. `grep -c` exits 1 on zero matches, so never
    # put it in an `&&` chain.
    git log origin/main..HEAD --format=%B | grep -c '^Claude-Session:'

    # How many COMMITS carry one — a different question, and the one to quote as a statistic.
    git log origin/main -100 --format=%H | while read h; do
        git log -1 --format=%B "$h" | grep -q '^Claude-Session:' && echo "$h"; done | wc -l
    ```
    **The two commands disagree, systematically and always in the same direction.** Over the same
    100-commit window the first returns **63** and the second **29**. `%B` is the *body*, and a
    **squash merge concatenates the bodies of every commit it absorbs**, so one squashed commit
    routinely carries several trailers — `grep -c` over `%B` counts trailers, not commits, and
    over-reports on any squash-merged history. The distribution proves the mechanism rather than
    merely fitting it: of the 29, sixteen carry one trailer, four carry two, five carry three, and
    one each carry four, five, seven and eight — which sums to exactly the 63 lines. The first form
    is still exactly right as a zero-gate; it is only wrong when read as a commit count. The gap
    widens on shorter windows, where a single large squash dominates: over the last **20** commits
    the line form reads **25** against a true **11**. All figures measured 2026-08-01.
    Keep `Co-Authored-By:`. **If the count is non-zero, amending is not enough** — the trailer is on
    commits you already made and the harness re-adds it to the fixup. `git filter-branch` is not
    available here. Two recipes that do work: `git rebase <base> --exec '<amend with a cleaned
    message>'` (the exec runs outside the harness), or rebuild the branch —
    `git checkout -b <slug>-clean <base>`, then per commit `git cherry-pick -n <sha>` followed by
    `git commit -F <message-with-the-line-stripped>`. Verify with `git diff <old-head> HEAD` over the
    paths you touched (expect empty), then `git push --force-with-lease`.
- **Rebuilding a branch onto a moved master: never take a whole file from the old branch.** The obvious
  move — `git checkout <old-branch> -- <file>` — reverts that file to its old state wherever ANOTHER
  lane has since edited it, with no conflict, no failing check, and a diff that reads as your own edit.
  It cost #1701 ten lines of documentation this way, caught only because `--stat` showed deletions in a
  file believed touched once. Re-apply your own hunks onto `main`'s version instead; when you do take a
  file whole, **diff it with `git diff origin/main...HEAD -- <file>` — three dots — and read the `-`
  lines**. Three dots diffs against the MERGE BASE, which is exactly the set of lines a merge can
  silently revert; anything `main` gained after you branched cannot be silently lost, it conflicts. Two
  dots diffs the two TIPS, so once `main` moves under your branch it reports everything `main` gained as
  *your* deletions — a phantom revert in the very file another lane just edited (instrument trap 271).
  The `Docs` CI gate does not cover this: it validates table structure and numbering, never prose. See
  instrument-trap 41.
- **A pipeline's exit status is its LAST stage's.** `cmd | tail`, `cmd | head` and `cmd | grep` all
  discard `cmd`'s failure, so `build && test | tail -3 && commit` commits through a red test. Capture
  the status (`cmd > log; rc=$?`), use `set -o pipefail`, or read `${PIPESTATUS[0]}`. Separately,
  `grep -c` **exits 1 on zero matches**, so a legitimate "none found" aborts an `&&` chain — separate it
  with `;`. Four people lost time to these in one session. See instrument-trap 40.
- **Never publish the developer machine's local paths.** This repository is public. An absolute path
  leaks the account name and the private directory layout of someone's computer, and it is never the
  information a reader needs — the *shape* of the command is. So keep absolute host paths out of commit
  messages, PR titles/bodies/comments, issue text and review comments, and out of committed files
  (docs, scripts, tests, code comments). Use placeholders instead, or a path relative to the repository:
  ```text
  <REPO_ROOT>    the checkout root          <WORKTREE>   a worktree root
  <DUMP_ROOT>    where the game dumps live  ~/           the developer's home
  ```
  So write `-DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0`, not the real path, and
  `cd <WORKTREE> && cmake --build build -j6`. Paths *inside* the repo (`prosper/src/gpu/…`) and generic
  system paths (`/usr/lib64/libc.so.6`, `/tmp`, `/var/tmp`) are fine — the rule is about the private
  part above the repository root. This is a privacy matter, not a security one: nothing here is secret,
  and it still should not be published. Sanitize before you post; if something already published slipped
  through, edit it (`gh api -X PATCH repos/OWNER/REPO/issues/comments/<id> -f body=…` edits a comment).
- **Track work in GitHub issues** (`gh issue ...`). The rules:
  - **Any bug you find but do not fix in the same session gets an issue, immediately**, while the
    context is fresh: exact `file:line`, the concrete failure scenario (inputs/state → wrong
    behavior), the introducing commit if known, and a suggested fix. Apply the label `bug` plus
    `bug-hunt` if it came from a systematic review. A bug that lives only in a commit message,
    doc, or your head is a bug that gets rediscovered the hard way.
  - **Before starting non-trivial work, check the tracker** (`gh issue list --label bug-hunt`,
    or by area) — the fix may already be specified, or an issue may conflict with what you're
    about to change. Before fixing an issue, **re-verify it still exists on current `main`**
    (another PR may have fixed it independently).
  - **One issue → one focused PR** where feasible; reference the issue in the PR body
    (`Fixes #NN`) so the merge closes it. When a fix lands that partially addresses an issue,
    comment on the issue with what remains instead of closing it.
  - Larger planned work (frontier steps, refactors) also gets an issue when it will span
    sessions — issues are the durable queue; `docs/` files explain *how*, issues track *what
    remains*. The umbrella issue for the history-review backlog is #72
    (full annotated list: `prosper/docs/BUG_HUNT_BACKLOG.md`).
  - **Long-lived game trackers are progress indices, not bug reports.** Create one as
    `[Game tracker] <Title> (<TITLE_ID>)`, label it `tracker:game` plus the title's existing
    `game:<slug>` label, and keep it open through the compatibility ladder. Its compact body records
    the current rung and latest verified master where known, the best checked-in screenshot, what
    works, links to ordinary blocker issues, the reproduction/input/snapshot route, technical status
    docs, and the six-rung checklist. Ordinary bugs remain separate and close independently. PRs
    update a game tracker with `Refs #NN`, never `Fixes #NN`; do not give the tracker `bug` or
    `in-progress` unless work on the tracker issue itself is actually claimed.
  - **Claiming an issue (multi-agent lock).** Several agents work this repo concurrently; claim
    before you code so two agents never fix the same issue:
    1. Check it's free: no `in-progress` label, no unexpired claim comment, and no remote fix
       branch — `gh issue view NN` + `git ls-remote origin 'refs/heads/fix/issue-NN-*'`.
    2. Claim it: add the `in-progress` label AND post a comment
       `CLAIMING: <agent/session identifier> | branch fix/issue-NN-<slug> | <UTC timestamp>`.
    3. Re-check for a race: re-read the comments right after posting; if two claims landed, the
       EARLIER timestamp wins (tie → lexicographically smaller branch name); the loser comments
       "unclaiming" , removes the label only if they added it, and picks other work.
    4. Push your `fix/issue-NN-<slug>` branch early — the remote branch doubles as the lock.
    5. Release: the merged `Fixes #NN` PR closes the issue (label goes with it). If you abandon
       or park the work, comment what state it's in and remove the `in-progress` label.
    6. Claims expire: `in-progress` with no open PR and no activity for 24 h is stale — anyone
       may re-claim after posting a takeover comment.
  - **Labels are organizational, not access gates.** Two prefixes classify every issue:
    - **`game:<title>`** names a *game* whose specific boot path an issue exercises — `game:messenger`
      (PPSA24651 / The Messenger, IL2CPP/Unity), `game:blasphemous2` (PPSA13579), `game:ue4`
      (PPSA17942 / Unreal Engine), etc.
    - **`area:<subsystem>`** names *shared code used across titles*. `area:infra` is the shared
      substrate — loader, libc, libkernel, the recompiler, generic GPU/AGC, host mapping/ABI. Shared
      code is never bucketed under a game's label (it belongs to every title, so a single `game:` tag
      would be a category error).

    Pick `game:<title>` when the fix is specific to one game; pick `area:infra` (or another `area:`
    subsystem label) when it lives in the shared substrate. A single issue may carry both — e.g. an
    `area:infra` loader bug first observed in `game:blasphemous2`. The labels describe *what* an issue
    touches; they do **not** fence off who may work where. Any file or issue is fair game — when
    several agents run concurrently, coordinate through the claim lock and worktrees above rather than
    by carving up the codebase.
