# prosper

A PS5 (Prospero) → **Linux/Windows** user-space compatibility layer — *Proton for PS5*.

Not a CPU emulator: the PS5 is x86-64, so guest code runs **natively**. `prosper` reimplements the
operating system (FreeBSD-derived), the library ABI (Sony NID-linked modules), and the GPU
(AGC → Vulkan) underneath the unmodified game binary.

**Primary title:** `PPSA24651` — *The Messenger* (Unity 2022 / IL2CPP). The compatibility corpus now
tracks 39 titles across Unity/IL2CPP, Unreal Engine, RAGE, Hedgehog Engine and custom engines; see
[`../COMPATIBILITY.md`](../COMPATIBILITY.md) for the current user-visible milestones. Their SELF
segments are unencrypted, which is what makes the project possible without console keys. Dumps are
user-supplied and gitignored.

For development, start with [Debugging and profiling workflows](docs/DEBUGGING_WORKFLOWS.md):
question-to-tool recipes, capability controls, host sanitizer checks and GPU capture analysis.

## Status
- ✅ **M0–M1 — Recon, tooling & loader.** Format cracked; SELF/ELF → relocatable image → multi-module
  link → NID import binding.
- ✅ **M2–M3 — HLE + boot.** libkernel/libc (virtual/direct/flexible memory + guard pages, threads,
  futex, AIO + file I/O, time, locale); IL2CPP GC + thread pool; system services the games gate on
  (user/pad, SaveData, AvPlayer/AJM, common dialogs + IME, NP/online, system language). Boots
  **through** IL2CPP and Unity's GfxDevice into the running frame loop.
- ✅ **M4 — Graphics (AGC → Vulkan).** AGC frontend (CreateShader + submit; PM4 decode → command
  processor → register-state fold; EOP/DMA/fence writes on the guest timeline). **RDNA2→SPIR-V
  recompiler:** broad scalar/vector ALU coverage + `SCC`; divergent control flow (EXEC predication,
  saveexec/restore, `execz` if-then + loop exits); `SMEM`, `MUBUF` vertex-fetch/load/store, `MIMG`
  sample/load, `LDS` + fail-closed barrier lowering,
  `EXP`/`VINTRP`; EUD-resident descriptor resolution; every SPIR-V emitter `spirv-val`-gated in CI
  (`tools/spv_validate`). Texture decode:
  GFX10 `SW_4KB_S`/`SW_64KB_S` de-swizzle for all element sizes + BC1–7/BC6H, T# format + `DST_SEL`
  swizzle + paired S# sampler. Frame spine → `resolve_pipeline_state` → real `VkGraphicsPipeline`s
  with blend (incl. separate-alpha)/depth/stencil/write-mask/fast-clear + a render-to-texture cache.
- ✅ **The Messenger renders through gameplay:** intro, title, menus, save list, and the complete first
  level render at native 1920×1080 with working scripted gamepad progression. The LUT producer,
  foreground culling, and cross-call depth/stencil lifecycle fixes are regression-covered by Vulkan
  tests and the real-game snapshot guard.
- ✅ **Graphics investigation tooling:** versioned live GPU capture/replay (`.prgcap`), per-draw/resource
  inspection and isolation, strict reflected SPIR-V/runtime descriptor validation, render-target producer
  provenance, normal screenshot capture, and a local content-metric snapshot guard. Native-speed `.prgtl`
  submit/present indexes can select an exact submit or semantic route checkpoint before renderer sampling and
  materialize immutable, content-deduplicated graphics/compute state plus mixed operation order for offline
  replay (#594).
  Offline dependency graphs resolve in-submit resource versions and identify deduplicated prior-submit
  leaves, including temporal read-before-write surfaces, without invoking Vulkan (#595). Capture v7 retains
  bounded, content-addressed raw RDNA2 and exact opcode/PC/state diagnostics for unrealized draws/dispatches;
  `gpu_replay` can inspect and extract the failed stage without rerunning the title (#618).
- ✅ **Dead Cells reaches gameplay reproducibly:** deterministic native-Windows routes pass the splash and menus
  into the first playable scene with rendering disabled or with every retained GPU submit fully executed and
  published (#766). The persistent-depth rejection is fixed: timeline-v5 backing hashes and
  compute writer provenance identify Unity's 32 KiB HTILE fill as the per-frame fast clear, and overlapping
  guest GPU writes now invalidate the detached Vulkan depth cache (#611). A routed live A/B restores the
  foreground/platform/HUD layers that remained black without invalidation.
  Four repeated fragment failures were uniform VCCZ-exit light-accumulation loops. The stage-specific
  structurizer now accepts that shape only when each compare operand has a proved wave-uniform reaching
  definition; varying bounds and the compute wave-mask form still reject. Current routed gameplay submits
  realize every semantic draw (#615).
  Version-8 GPU captures seed temporal render targets, retain complete depth-surface programming, preserve
  exact persistent Vulkan depth/stencil checkpoint planes, and keep content-addressed resource versions for
  faithful offline isolation (#568/#594/#569). Dispatch thread counts and derived workgroup dimensions,
  compute program binding, direct type-1 buffers, and mixed graphics/compute PM4 order now execute correctly
  (#580/#576/#584).
  The later giant translucent Dead Cells gameplay surface was a native-format loss: a 642x362 RGBA16F lighting
  target was rendered and sampled as RGBA8, clamping HDR values before composition. Renderer-owned targets now
  preserve RGBA16F through render, readback, seeding, sampling, and persistent Vulkan caches. Capture v13 records
  the RTT format and remains backward-compatible with v1..v12 artifacts (#773). A compute reset of that lighting
  backing also invalidates the overlapping CPU RTT copy; retaining it had re-seeded the previous frame and driven
  the temporal effect toward white/yellow (#780). Residual window-light banding is tracked separately in #781.
- ✅ **Blasphemous 2 reaches first gameplay:** prelinking the title's optional FMOD core/studio
  plugins lets IL2CPP P/Invoke complete `FMODAudioBanksManager` initialization (#638/#640). The later AGC
  fault was prosper corrupting live DCB state: `+kSrjIVxKFE` is `sceAgcDcbPushMarker`, not the context
  constructor modeled by the old workaround (#641). A dedicated two-pass `sceHttpUriParse` implementation
  replaces the success-with-stale-output stub that crashed the title's telemetry worker (#642). The routed
  run continues through its Http2 calls and loads gameplay scenes, UI, enemies, bosses, audio banks, and
  cutscene assets. Normal-return guest pthreads now leave host `%fs` active before glibc performs its own
  thread cleanup (#644), removing the next host crash. Poll-safe scripted presses and observed-state logging
  (#646) reliably traverse the long opening. PS5 primitive type 7 is the title's procedural RectList clear;
  mapping it to a four-corner Vulkan triangle strip clears stale UI alpha instead of rasterizing three points
  over an otherwise valid world (#654). A 420-frame sampled-render native capture confirms the complete first
  playable room, and screenshot manifests distinguish new publications from actual pixel progress (#648).
  The route and capture recipe are in `scripts/blasphemous2/README.md`.
- 🚧 **Active frontiers (2026-08-11):** GTA V reaches its title/menu and routed gameplay entry, where
  the HUD, radar and tutorial text render over an absent 3D world. The six-stage runtime-selected
  descriptor-array lift is complete. Program-tagged live evidence proved that the earlier "about 57 CFG
  sites" count mixed consequent structurizer messages with independent terminal failures; reviewed work
  now advances the exact failing programs one instruction or resource contract at a time. The current
  routed sample has 29 recompile-empty programs plus 6 invalid-descriptor programs, a 35-program union
  with no new failures versus the preceding sample (#2412/#2481). Subgroup-contract limits are tracked
  separately in #2429. Dragon Quest VII now boots and submits
  continuously on Windows (#2447), but routed
  progression can hit a fixed `sceKernelBatchMap` ENOMEM whose high-volume memory logger suppresses the
  repro (#2448/#2450). Sonic Racing: CrossWorlds has advanced to rung 2: pulsed input reaches its full
  4K title screen, while the later profile panel and terminal white state remain open (#2358/#2360).

The completed Messenger black-render investigation and reusable evidence boundary are recorded in
[`docs/MESSENGER_BLACK_RENDER.md`](docs/MESSENGER_BLACK_RENDER.md). Current work is tracked in GitHub
issues; title failures should produce a reproducible route/capture and a narrowly scoped issue rather
than a moving-revision hypothesis log.

See [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/GRAPHICS.md`](docs/GRAPHICS.md),
[`docs/RENDER_LOOP.md`](docs/RENDER_LOOP.md), and [`docs/VERIFICATION.md`](docs/VERIFICATION.md)
(the agentic-first, programmatic, no-manual-eyeballing verification strategy).

## Layout
```
prosper/
  docs/            architecture, roadmap, graphics, verification, per-frontier logs
  src/self/        SELF/ELF parsing → relocatable module image
  src/loader/      multi-module linker + global export table
  src/hle/         HLE of Sony libraries (libc, libkernel, AGC/graphics, services), NID hashing
  src/host/        host execution: per-platform image mapping, ABI stubs, fault handling
  src/gpu/         AGC→Vulkan: PM4 decode, command processor, render state, vk_translate,
                   texture tiling + BC decode, RDNA2→SPIR-V recompiler
  frontends/       shared boot+render core, windowed prosper-app, SDL3 audio/dialog, controllers
  tools/           self_dump, boot_trace, shader_histo, screenshot, snapshot, gpu_timeline, gpu_replay,
                   spv_validate
  tests/           unit + boot + Vulkan-execution tests (ctest)
  CMakeLists.txt
```

## Build
```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --no-tests=error  # report the discovered count; it varies by platform
```
`ctest` needs `spirv-val` on `PATH` — `spirv-tools` on Fedora/Ubuntu,
`mingw-w64-ucrt-x86_64-spirv-tools` in MSYS2, `brew install spirv-tools` on macOS. The
`spv_validate` test is the strict SPIR-V validation gate and **fails** rather than skipping when
the validator is absent, because silently skipping is how #1711 shipped.
Add `-DPROSPER_APP=ON` for the windowed `prosper-app` frontend (fetches SDL3). Or a tool directly:
```
g++ -O2 -std=c++20 tools/self_dump/self_dump.cpp -o self_dump
./self_dump ../PPSA24651-app0/eboot.bin [--symbols]
```

From the parent repository root, native Windows builds and runs the full graphics + WASAPI audio +
controller/keyboard frontend with:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

## Legal / scope
Interoperability & preservation research on legally-owned titles. `prosper` ships **no** Sony code,
firmware, or keys — it reimplements published library interfaces clean-room style. It only operates on
**already-unencrypted** dumps; it does not defeat DRM/encryption.
