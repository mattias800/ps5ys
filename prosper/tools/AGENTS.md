# AGENTS.md — prosper/tools

Developer/agent tooling. These are debugging and verification aids, not part of
the shipped runtime. Build them from `build-linux/` like everything else.

Start with [Debugging and profiling workflows](../docs/DEBUGGING_WORKFLOWS.md) for the
question-to-tool map. `doctor/` validates external instruments with bounded positive controls,
provides a small standalone sanitizer build of existing host tests, and verifies RenderDoc
capture/replay without requiring an importable system Python module.

- **`evidence/prerender_check.py`** - **run this before publishing a progression screenshot.** It
  answers whether the frame is the game's own PRE-RENDERED picture rather than something prosper
  rendered: a full-screen loading blit is the most convincing false evidence this project has
  produced, and one reached a merged PR, `BLOG.md` and `COMPATIBILITY.md` before a human recognised
  the artwork (instrument trap 230). `python3 tools/evidence/prerender_check.py <shot> <DUMP_ROOT>`;
  exit 2 = the frame IS game artwork and is not progression evidence, exit 1 = the check could not
  run and must not be read as a pass.
- **`profiling_access/`** — optional host-only, scheduler-only autonomous recording via a
  root-owned helper and a narrowly scoped sudo rule. One administrator installation; thereafter
  `profile.py` runs unprivileged and uses `sudo -n`. No arbitrary commands, output paths or perf
  options cross the privilege boundary. See its README for limits, verification and removal.
- **`verify-pr.ps1`** - author-owned PR verification orchestrator for the Windows+WSL development
  environment. Run it only from a clean, pushed PR head: `docs` records the diff check, `core` adds
  Linux and Windows build+ctest, and `renderer -Snapshot NAME` also runs the selected real-game guard
  with `boot_trace` and `screenshot` pinned to that run's selected Linux build directory.
  `-Pr N` posts the generated SHA-bound `AUTHOR VERIFICATION` record. Reviewers inspect its coverage
  but do not rerun it; see the mandatory merge policy in the root `CLAUDE.md`. Run
  `powershell -File tools/test-verify-pr.ps1` after changing the orchestrator; it probes exact-base
  pinning, WSL selection, rejected skip attempts, and untracked-source contamination.
- **`snapshot/`** - routed, multi-frame **rendering regression inventory**. Run
  and review the full `python3 tools/snapshot/snapshot.py check` matrix before
  every release. It is not a day-to-day development or merge gate: PR authors may
  skip snapshots entirely during long iterations or run only useful focused guards,
  unless the task explicitly requires a run. Snapshot results do not define whether
  master or a PR is acceptable; either may contain a detected regression, including
  an intentional cross-title tradeoff. Release review
  decides whether to fix or explicitly accept and document it. New or changed
  baselines require two-run image inspection; see `snapshot/AGENTS.md`.
  `snapshot/when_matched.py` answers the one verdict a red guard cannot explain by itself --
  when a guard fails on STRUCTURE alone, it scores every sample of a run against that guard's
  reviewed references and says whether the run reaches them at any point, which separates "the
  route's timing drifted" from "the state is never reached". **It falsified its own author on its
  first use** (#2899): the timeline-shift reading he had built from three crossed counters and his
  own eyes on the frames did not survive the sweep. That is recorded because an instrument which has
  only ever agreed with the person holding it is indistinguishable from a restatement of what he
  already believed.
- **`vkval/`** — runs the ctest suite under `VK_LAYER_KHRONOS_validation` and **fails on any
  validation message id that is not itemised in `vkval/allowlist.txt`** (#1704). This is the guard
  for the whole class #1690 belongs to: a descriptor bound to a module that declares something
  different is undefined behaviour, so it produces per-driver disagreement rather than a failure,
  and reads as a driver-version bug. `python3 tools/vkval/vk_validation_scan.py --build-dir <dir>`
  to gate, `--report-only` to measure. It refuses to report "clean" until it has watched the loader
  insert the layer — an absent layer and a clean suite print exactly the same nothing. Every
  pre-existing finding is deferred *visibly*, with a reason and an open issue; fixing one means
  deleting its line. See `vkval/README.md`.
- **`boot_trace/`** — boots a SELF/ELF game image through the loader + HLE and
  runs it, with the fault handler, GPU executor, and (under `PROSPER_RENDER`) the
  live Vulkan renderer. The main harness for exercising a real title headlessly.
  - **Periodic BMP frame dumps are opt-in.** `PROSPER_FRAME_DIR` only selects the output directory;
    it does not enable writes. Set `PROSPER_FRAME_DUMPS=1` for the existing first-60/every-10 cadence,
    or use `PROSPER_DUMP_CONTENT`, `PROSPER_FRAME_DUMP_FIRST`, or `PROSPER_FRAME_DUMP_EVERY` as a
    deliberate selector. `PROSPER_NO_FRAME_DUMPS=1` has highest priority and disables all of them.
  - **Who allocated this memory?** `PROSPER_MEMLOG=1` reports every direct-memory reserve / allocate
    / map with its size, which answers *how much* but never *who* — and the same guest allocator
    serves every subsystem, so a title whose heap grows without bound looks identical to one that is
    legitimately loading. Add **`PROSPER_DMEM_CALLER=1`** to print, once per distinct call chain, the
    guest return addresses above each `sceKernelAllocateMainDirectMemory` **and each
    `sceKernelAllocateDirectMemory`** (#3054 — before that, only the Main variant published a chain,
    so a title allocating through the other entry point produced no records at all and
    `PROSPER_DMEM_WRITE_TRACE` could not name any byte it owned). The record names the allocator it
    came from, so `alloc_dmem` and `alloc_main_dmem` lines are told apart. The first two return
    addresses are interned as a bounded run-local `caller-chain=N`; the full bounded chain prints
    once, while every allocation in the `PROSPER_MEMLOG=1` census carries the same correlation ID:
    ```text
    [memhle] alloc_main_dmem len=0x1000000 -> phys=0x21500000 caller-chain=1
    [dmem-caller] caller-chain=1 alloc_main_dmem len=0x1000000 from eboot+0x1b2454f eboot+0x7d0a00 eboot+0x22b1e80 …
    [memhle] alloc_dmem range=[0x0,0x400000000) len=0x40000000 align=0x0 type=0xc -> phys=0x40010000
    [dmem-caller] caller-chain=2 alloc_dmem len=0x40000000 from eboot+0xccb3 eboot+0x20c7c8 eboot+0x7677 …
    ```
    Feed the first address to `tools/re/edis.py` / `tools/re/xref.py` to name the caller. The walk is
    a heuristic stack scan (a stale spill slot looks exactly like a return address), so treat the
    result as a callsite **hint** to confirm against the module's disassembly — never as proof on its
    own. IDs are stable only within one process and the full-chain table is bounded to 64 entries so
    an allocation-heavy boot cannot bury the log. A walk with no guest return address is recorded as
    `caller-chain=unknown`; allocations beyond the distinct-chain limit say `caller-chain=overflow`,
    and the ceiling is announced once. Neither state is a negative caller result.
  - **Is a movie's colour wrong because its chroma plane was never recognised?**
    **`PROSPER_AVPCHROMA_LOG=1`** prints one line per distinct narrow Unorm8 sampled texture — both
    AvPlayer NV12 planes qualify — with every field the chroma test reads and the clause that decided
    it, so a rejection names itself instead of being silent:
    ```text
    [avpchroma] addr=205161c600 1024x540 fmt=9 ncomp=2 tile=0 dim=5 depth=1 … swz=4,5,0,1
                row_bytes=2048 pitch_field=0 registered=2048 … -> CHROMA matched-registered-pitch
    ```
    Why it needs its own diagnostic: an unrecognised chroma plane takes the legacy narrow coverage
    path, which broadcasts its first byte to every channel, so the shader's V equals its U. Luma,
    detail and geometry stay exactly right and only colour collapses onto a single green↔magenta axis
    — invisible to draw counts, colour counts and every non-black metric, and it reads as a shading
    bug. **`PROSPER_AVPCHROMA_DUMP=<dir>`** additionally writes each plane's exact guest bytes at its
    resolved pitch (`PROSPER_AVPCHROMA_DUMP_EVERY=N` samples one sighting in N, because a movie opens
    on a fade from black); converting the dumped pair on the CPU separates a decode/staging defect
    from a sampling one. Both off by default. First use: #2005.
  - **Is prosper overrunning a command-buffer reservation the guest made?** prosper's AGC builders
    append into the *guest's* buffer, and the guest reserves that buffer from the packet sizes it was
    compiled against — so a builder that emits more dwords than the real AGC function overruns it.
    The failure does not look like a size bug: on #1748 it looked like a 210 MiB/s heap leak, and in
    a title with memory to spare it looks like nothing but command-buffer churn and low frame rate.
    **`PROSPER_DCBFULL=1`** reports every Dcb "buffer full" callback and
    **`PROSPER_DCBWIN=N`** every packet appended into a window of ≤ N dwords (N=1 selects 64):
    ```text
    [dcbwin]  window=16 dw : AcquireMem n=8 op=0x10 at +0 (avail 16)
    [dcbwin]  window=16 dw : ReleaseMem/EopAction n=9 op=0x10 at +8 (avail 8)
    [dcbfull] #0 ReleaseMem/EopAction op=0x10 need=9 avail=8 reserved=0 window=16 dw at +8   <- the signature
    ```
    The signature is **`avail>0` AND `reserved=0` AND the same triple repeating** — all three. Two
    ways to read it wrong, both measured on the corpus: the *window* is the current segment of a
    segmented Dcb rather than a reservation (58 false positives on The Pathless), and `avail>0` on
    its own is the ordinary end-of-buffer condition whenever the guest holds a `reserved_dw` back
    (every repeated `avail>0` triple in the corpus carries `reserved=18`, with `avail` tracking each
    packet's own size). Under the full rule the corpus is clean, and re-introducing the #1748 defect
    makes it fire 256 times — it was checked in both directions. Full contract, the per-builder table
    and the census: `docs/AGC_PACKET_SIZES.md`.
  - **Does a PM4 write path store *this value* into guest memory?**
    **`PROSPER_WRITE_TRAP=0xV[,0xV…]`** (up to 8 values) checks every guest-memory store the command
    processor performs — `RELEASE_MEM` data_sel 1/2/3, `EVENT_WRITE`, `WRITE_DATA`, and both
    `DMA_DATA` forms (a copy is scanned to its first 4 KiB) — against the listed 32-bit values, and
    reports each match with a per-value ordinal, the destination, the destination's current qword
    and the packet builder:
    ```text
    [agc] WRITE-TRAP #1[val=0x5a5a0001] kind=REL1 dst=0x2020f3d580+0 pre=0x5a5a0000 pkt=0x30030c2f58 t=4857ms
    ```
    This is the complement of `PROSPER_PROVENANCE_ADDR`, which keys on the *destination*: use the
    value trap when a corruption's poison is **constant across runs but its address is not**, so
    there is no stable address to watch. Read a null result with all three of its limits in mind:
    - **Scope.** It sees the PM4 write paths only. Compute-dispatch writeback, the Vulkan backend's
      readback, and every HLE that writes guest memory are invisible to it, so zero hits means "no
      PM4 write path created that value", never "prosper did not".
    - **Stride.** The scan is 4-byte-strided from each payload's base. `REL*`/`EVENT`/`WDATA`/
      `DMA`-immediate payloads are dword-aligned by construction, but a `DMA` *copy* is a raw byte
      move, so a poison at a source offset that is not a multiple of 4 is not seen — which is
      exactly the shape of an off-by-one store.
    - **Positive control.** Prove a hit can appear before believing that none did: arm a value the
      title is known to write (a fence `0x1`) *alongside* the value under investigation. Counters
      and the print budget are **per value**, so a noisy control cannot starve the log of the value
      you care about; each line carries its own value's ordinal, and printing continues sparsely
      (at powers of two) past the first 64, so a capped report can never be mistaken for a count.

    Malformed input is refused **loudly** (`[write-trap] NOT ARMED …`, `REFUSED …`) rather than
    silently disarming, because a silent disarm is indistinguishable from a hard negative.
  - **Does the guest still own this block when we write to it?** **`PROSPER_INIT_TRIP=1`** reports,
    for every 4-byte immediate-zero `DMA_DATA` init to a consumed-marker fence label, whether the
    destination currently holds pointer-shaped content (`overptr`) and whether it is reachable from
    the guest allocator's own idx=1 free chains (`member`). Detail and totals are separate lines:
    ```text
    [agc] INIT-TRIP-SELFTEST #512 pools=25 probes=9 positives=9 deep=8 deep_positives=8 first_head=0x…
    [agc] INIT-TRIP #512 dst=0x… pre=0x… overptr=1 aligned=1 member=0(list=0 pool=0x0 hops=0) t=…ms
    [agc] INIT-TRIP-TOTALS n=1024 overptr=926 member=10 both=1 t=…ms
    ```
    `overptr` alone decides nothing — a label the guest legitimately *popped* from its pool free list
    still carries the stale link, so the content is identical either way. `member` is the
    discriminator.

    **Read `INIT-TRIP-TOTALS`, never a detail line**, and note that the sample above shows why: the
    detail line at #512 reads `member=0` while the run total is `member=10`. The detail schedule is
    sparse and front-loaded, and on this title the `member` population does not appear until after
    ordinal ~512, so every early sample reads zero. Quoting one of them is exactly how #1767's first
    draft reached a conclusion its own totals contradict. Totals print every 256 for that reason.
    `aligned=0` marks a destination membership could never match anyway (`plausible_node` wants a
    0x20-aligned block base), so it is a structural zero rather than a finding.

    **The `SELFTEST` line is not optional decoration**: a learned bin *head* is by
    construction the first node of its own chain, so a walk that answers "no" to a head is blind
    rather than truthful, and `positives < probes` (or `probes=0`) means every `member=0` beside it
    is an unarmed instrument, not evidence. A head matches at hop 0, though, which proves only that
    the walk is *armed* — so the probe also follows each head's successor (`deep`/`deep_positives`),
    an interior node reachable only by traversing a link. The three failure modes are named in the
    line itself: `NO HEADS`, `BLIND`, `SHALLOW`.

    `mb3_freelist_selftest()` is available to any caller that wants to qualify a membership null the
    same way. It controls the **walk**, not the **call site**: a caller that early-returns for the
    `member==true` population before reaching its own probe will still see 100% `member=0` beside a
    green self-test. `PROSPER_INIT_TRIP` prints a `NOT A CONTROL` banner when
    `PROSPER_MB3_FREELIST_GUARD` is armed alongside it for exactly that reason.
    **`PROSPER_INIT_SUPPRESS=ptr|member`** is the matching A/B arm, not a candidate fix: `ptr`
    suppresses on content, which #1245 proved cannot separate a live label from a freed block, so it
    will also drop legitimate inits. It exists so "is this write load-bearing for the corruption?"
    can be answered by removing it.
    **`PROSPER_REL1_FORGE_SUPPRESS_ALL=1`** is the matching fence-side A/B arm for #1226. It
    suppresses every 32-bit ReleaseMem write that would turn a pointer-shaped qword with a zero low
    dword into `pointer|value`, including live paired fences that real hardware must execute. It is
    therefore diagnostic-only and unsuitable for normal play. A valid run must print
    `FORGE-DECISION-TOTALS` with `candidates == suppressed` and `landed == 0`. Every candidate
    through 256 prints, then every 256th: the worker-fault path uses `_exit(90)` and cannot run an
    atexit summary, while #1226's combined arm has a measured sub-128 population and therefore gets
    an exact terminal census from this bounded dense prefix. Combine it with
    `PROSPER_INIT_SUPPRESS=ptr` for the decisive landed-forges-zero plus init-suppressed arm. A
    malformed selector prints `NOT ARMED` rather than silently becoming a negative result.

    **`PROSPER_FORGE_TRIP=1` now reports which address window matched, and its own totals.** The
    forge-shape test is split in two: the *decision* predicate the default guards use (a DOLL-era
    window that stops at `0x2100000000`) and the *report* predicate a census must use (prosper's
    whole guest-VA window). Each `FORGE-STOMP` line carries `window=narrow` — a guard can see and
    possibly decline this write — or `window=wide-only` — no guard is reachable for it at all — plus
    a running `FORGE-TRIP-TOTALS seen=… narrow=… wide_only=…`. Without the split a census silently
    measures the *predicate* instead of the title: ArcRunner's arena is `[0x2000000000,0xa000000000)`
    and its terminal fault dereferences `0x2100000001`, one byte above the narrow bound, so "how many
    forges are there" could not be asked at all. **A `wide_only=0` is now a real negative**; before
    the split it was unobtainable. `PROSPER_PTRLIKE_WIDE=1` additionally arms **both** guards (the forge
    branch and the `REL1-LIVE` branch) over the wide window — default OFF on purpose, because
    `rel1_stomp_guard()` is default-ON and widening it is an unmeasured suppression over 500 GiB of
    addresses on every title at once. It announces itself, since an A/B lever that cannot show it
    moved turns a hard negative into a void result:
    ```text
    [agc] PTRLIKE-WIDE ARMED: guard window [0x1000000000,0xa000000000) (narrow default was …)
    [agc] PTRLIKE-WIDE NOT ARMED: PROSPER_PTRLIKE_WIDE='yes' is not a number — guards keep the narrow window
    ```
    Pair it with a positive control before quoting a zero: `report_suspect_write()` emits nothing at
    all on some titles (ArcRunner, nine runs), and an emitter that never fires cannot distinguish
    "no such write happened" from "this diagnostic is not reachable here".
  - **Is an `addr=(nil)` fault the guest's, or ours?** **`PROSPER_LAZY_COMMIT_STRICT=1`** (#1944).
    prosper backs a guest touch of a reserved-but-uncommitted VA with 64 KiB of anonymous zeros and
    resumes. When the faulting access is a *read of a pointer field* the guest gets a zero and
    dereferences it on the next instruction, so the fault report names the **dereference** site and a
    register **prosper itself set to 0** — which reads exactly like a guest object-lifetime bug.
    Strict mode declines the commit, so the SIGSEGV reports at the *loading* instruction with the
    real faulting address. The default-path line is also fail-visible now:
    ```text
    [lazy-commit] #1 mapped page=0x2100000000 addr=0x2100000041 access=read rip=0x…  FORGED-PTR-SHAPE(low-dword<=0xffff: …)
    [lazy-commit] #1 DECLINED(strict) page=0x2100000000 addr=0x2100000041 access=read rip=0x…  FORGED-PTR-SHAPE(low-dword<=0xffff: …)
    ```
    The ordinal is a whole-run census (the worker-fault path `_exit()`s, so no atexit summary can
    run), `access=` comes from the x86 page-fault error code, and `FORGED-PTR-SHAPE` marks a faulting
    address in the **first 64 KiB page** of a heap pointer's high half — i.e. a pointer that lost its
    low dword and then took a small structure offset, not a page the guest ever populated. Test the
    page, not the address: the founding case faults at `rdi+0x40` = `0x2100000041`, so a
    `low dword <= 1` form of this marker would have been inert on its own evidence. **A repeated page is not evidence of a
    lost mapping**: every `0x21000000xx` pointer lands in page `0x2100000000`, which is why two
    different guest sites appeared to "first-touch the same page".
  - **Which target holds content right now, and which pass wrote it?** **`PROSPER_DUMP_PERSISTENT`**
    reads back and reports every persistent colour target **of at least 64x64** (`rgb_nonblack`,
    `raw_nonzero_bytes`, and the first non-zero texel's own bytes, so "black" and "empty" stay
    distinguishable) — it also `continue`s silently past a readback failure, so treat its count as a
    lower bound and do not read it as "every target" — and
    **`PROSPER_PASS_LOG`** reports each pass's target identity, `vo=` flag and defer decision. Both
    open a **three-callback** window, and both accept two forms:
    ```text
    PROSPER_DUMP_PERSISTENT=26000       renderer-callback ordinal (unchanged)
    PROSPER_DUMP_PERSISTENT=ms:240000   the first callback at or after 240 s, and the next two
    ```
    Prefer `ms:` unless you have already measured the ordinal in the same configuration. **The
    ordinal is NOT the submit number `[gpucap]` prints** — on one 360 s route it reaches 6,560 while
    the capture's submit counter reaches 26,209 — and overshooting it yields *no census at all*, a
    silence that reads exactly like "every target was empty" (#1968). The `ms:` origin is the first
    armed census check — the same *kind* of origin `PROSPER_GPU_CAPTURE_AFTER_MS` uses, on a
    separate clock, so the two can be aimed at one phase but do not share a timebase.
    `PROSPER_DUMP_PERSISTENT` is deliberately **not** in the `live_gpu_targets` disable list, so it
    observes the normal persistent-GPU-target path;
    `PROSPER_GPU_CAPTURE` **is**, so an env-triggered capture run is on the CPU-readback path.
- **`revision/check_build_revision.py`** — **is the binary you are about to quote a measurement from
  actually built from the code you think it is?** The failure it exists for is silent and produces a
  *confident* wrong answer: a lane checks out new work (or rebases onto a master that moved), runs
  an existing build directory, and reports the result as a measurement of current master. The source
  tree IS current, the run succeeds, and nothing says the executable predates the change under test.
  On 2026-08-06 that nearly turned "#2121 does not move Sonic Frontiers" into a false negative on a
  title that gates on the exact NID #2121 fixed.
  ```bash
  python3 tools/revision/check_build_revision.py build-linux          # vs origin/main
  python3 tools/revision/check_build_revision.py build-linux --binary build-linux/screenshot
  python3 tools/revision/check_build_revision.py --manifest ~/work/manifest.json
  ```
  **It certifies an executable, not a build directory, and that distinction is the whole tool.**
  `prosper_build_revision_refresh` is an unconditional custom target consumers merely depend on, so
  `generated/prosper_build_revision/build_revision.cpp` is rewritten at the **start** of a build — a
  build that then fails leaves it recording the new revision while every executable still embeds the
  old one. The first version of this tool read only that file and therefore **certified a stale
  binary as current**, which is the failure it exists to prevent with a green tick on top; review
  caught it with a two-sided scratch reproduction. It now also requires an executable whose mtime is
  at least the generated source's, and **names which one it certified** so the claim is auditable.
  `configure_file` preserves the mtime when the revision is unchanged, so a rebuild at the same
  revision is not a false alarm. **Exit status is the contract: 0 only when a named executable is
  certified**, 1 for a mismatch, a stale binary, and every case where provenance cannot be
  established — no recorded revision, nothing linked to certify, not a build directory, an
  unresolvable ref. An unestablished provenance is not a match. (Usage errors exit 2, argparse's
  convention.) `--allow-stale` downgrades any refusal to 0 while still printing it, for deliberate
  pre-fix A/B arms — which must then say in the write-up which revision the numbers describe.
  Screenshot manifests carry `build_revision` in their run header, so an archived artifact nobody
  can re-run still says which build produced it. Three things it cannot see, by construction: a
  **dirty tree** (the embedded revision is HEAD — `--strict-dirty` fails when tracked files under
  `prosper/` differ, and resolves git from the checkout that owns the target rather than the cwd,
  which is a real distinction under the worktree rule); an **executable it was not asked about**
  (name the one you will run); and a **binary copied in from elsewhere**, whose mtime says nothing
  about this build directory.
- **`screenshot/`** — writes normal composited PNG sequences plus a JSONL evidence manifest. Use
  `--seconds 1` for wall-clock sampling, warmup or `--render-every N --render-every-for-seconds S`
  for slow software rendering,
  and the pixel-distinct/pixel-stale assertions when visible progression matters. Source publication
  counts alone do not prove that the image changed; see `screenshot/README.md`.
  Every run also reports a **framerate, as two numbers**: `distinct_fps` counts guest frames whose
  content changed, `presented_fps` counts publications. Quote the first. The renderer re-publishes
  its retained frame when a submit produces no present source, so a presented rate reads full speed
  for a frozen title -- that is instrument trap 90 as a counter, and it is what let #2783 hide for
  nine days. `--fps-overlay` burns the rate and its conditions into each PNG; it is off by default,
  and no content metric in the tool ever sees the annotation.
  A run whose **primary guest thread dies** now reports `guest=faulted status=GUEST-FAULT` and exits
  non-zero instead of `status=ok` (#2007), and the manifest summary carries `guest_state` and the
  fault address. Pass `--allow-guest-fault` only when the route deliberately samples a crashing
  title; it yields `status=GUEST-FAULT-ALLOWED`, never `ok`.
  Sampling also **stops** once the guest is dead *and* the present layer has published nothing for
  `--guest-fault-settle-seconds` (1 s), so the run no longer spends its remaining `--seconds` /
  `--timeout` re-photographing one frame (#2584: 24 of 25 PNGs were byte-identical). Every sample
  already taken is kept; the summary reads `saved/requested` with `stop=guest-fault` beside it, and
  the manifest summary carries `stop_reason`, so **a short PNG set is not evidence of a crashed
  harness — read `stop_reason` before assuming a run was truncated**.
  `--no-stop-after-guest-fault` restores full-length sampling. The stop cannot make a failing run
  pass: the saved/requested assertion is excused only in `--seconds` mode with a sample taken,
  **`--max-stale-seconds` / `--max-pixel-stale-seconds` disarm the stop outright** (they are maxima
  over the samples taken, so a shortened run would report a smaller one), and every other flag
  assertion is a floor that fewer samples can only push further from being met.
- **`frameclass/`** — classifies a directory of captured PNGs/BMPs as `LIT` / `UI-ON-BLACK` /
  `SPARSE` / `FLAT`, at native resolution on a 1/16 stride. The class worth having is
  **`UI-ON-BLACK`**: legible content covering under 2%, i.e. a HUD or notice over a world that never
  drew. `distinct_rgb_colors` and `nonblack_rgb_pixels` are already in `screenshot`'s manifest — read
  those directly if they are all you need. **`LIT` is not a promise of a game scene**, and
  `frameclass/AGENTS.md` shows why with five corpus frames on which UI and real scenes interleave on
  both coverage and colour count. Two traps are recorded there: **downsampling before counting
  colours erases thin UI** (a 4K frame at 160x90 reported *Stray*'s legible main menu as flat black),
  and a share-of-non-black test scores a flat **white** clear as a perfect frame. Run `--selftest`
  after touching a threshold.
- **`self_dump/`** — parse a SELF/ELF and print its segment/program-header map, import NIDs, and
  export RVAs. Use `--find-symbol NID` for a focused import/export query, and **`--import-slots`**
  to print the GOT/PLT relocation slot each import lands in — the step that starts every
  "who calls this Sony function, and what does the guest do with the result?" investigation:

  ```bash
  ./build-linux/self_dump <DUMP_ROOT>/<TITLE>-app0/eboot.bin \
      --import-slots --names ../PS5-3.20_Libs | grep scePadGetHandle
  # 0x000007f9b18 u1GRHp+oWoY  libScePad   scePadGetHandle   JUMP_SLOT
  ```

  Then hand that slot to `re/xref.py` (see `re/README.md` for the full recipe, including the PLT
  hop the slot's only direct reference is). `--names` points at the PS5 3.20 firmware stub dump
  (`$PROSPER_PS5_LIBS` is the fallback); **the NID is printed either way**, so an import the dump
  does not name is still actionable.

  The output is deliberately self-describing rather than terse. Each run prints which relocation
  tables it read and how many entries each held, how many symbols and imports it parsed, which name
  table it used, and — when there are **no** rows — the reason, exiting **3** rather than 0. Imports
  with no relocation are listed under `[IMPORTS WITHOUT A SLOT]` instead of being dropped, so a
  `grep` that finds nothing means the module does not import that function, never "it has no slot".
  A `JUMP_SLOT` row is the call path; `GLOB_DAT`/`64` rows are the same address stored elsewhere
  (vtables, static initialisers), which is why a busy title reports several times more slots than
  imports — filter with `grep JUMP_SLOT` when you want only the call site.
- **`guest_bt/`** — symbolicated **guest**-thread backtraces for a live or frozen prosper process:
  "what is this thread doing / waiting for?". Answers it for **any** title, not just IL2CPP/Unity —
  the managed-symbol step is optional, and on a stripped native C++ UE4 title it still walks every
  guest thread under its **real engine thread name** (`RenderThread 1`, `TaskGraphThread`,
  `FAPREventQueueL`, …) with the prosper-side wait frame attached, which is usually enough to say
  which subsystem is stuck. See `guest_bt/README.md`.
- **`hle_calls/`** — per-function call histogram over **every** HLE handler in the prosper binary,
  taken from a live process over a bounded window: "which Sony functions is the guest calling *right
  now*?". Complements `guest_bt` (which says where a thread is *parked*, and so says nothing about a
  thread that is running). It exists because `PROSPER_SVCLOG=1` is opt-in per handler — ~1,035 handler
  registrations against 94 `svc_log` call sites — so svc-log silence bounds the *instrumented*
  surface, not the guest's traffic, and `PROSPER_PROGRESS_UNIMPL` counts only the handlers that do
  **not** exist. Needs no rebuild and no gating env var: it enumerates handlers out of the binary by
  their shared six-`unsigned long` signature and counts them with non-stopping gdb breakpoints.
  Always pair a surprising zero with a handler you know fires — see the README's note on the
  `hit_count` trap that made all 151 handlers read zero.
  **`--values`** additionally records each handler's RETURN value, which is what turns a census into
  evidence about *behaviour* rather than only traffic — the recurring bug shape here is a handler
  answering `SCE_OK` for work it never did, and a call count cannot see that.
  **`--out-bytes N`** covers the half a return value cannot reach: it snapshots N bytes at whichever
  of a0/a1 is a readable pointer, before the call and at its return, and reports what CHANGED. The
  defect this codebase produces most is *success returned, out-struct never written* — every instance
  returns `0`, so a value census is blind to all of them. Its counters keep "the handler wrote
  nothing" (`same-zero`) apart from "this tool never read the bytes" (`null`/`small`/`unreadable`)
  and from the one ambiguous case (`same-nonzero`: the bytes already held what would be written).
  **`--launch`** starts the process under gdb instead of attaching, so the window covers init: the
  handlers a title calls once, before anything could attach, are reachable only this way.
  Every `--values` run prints a `positive-control=` verdict on its own value capture — read it first,
  because `VOID` and `unchecked` mean the run cannot support an inference from an ABSENT value, and
  they look nothing like a failure. See `hle_calls/README.md`.
- **`re/hle_handler_map.py`** — **which Sony NIDs collapse onto ONE prosper handler?** This is the
  blind spot `hle_calls --values` structurally cannot see: the histogram keys on the handler symbol,
  so when one handler answers several Sony entry points they all become a single row (usually `0x0`),
  and a mismodelled answer for one is indistinguishable from a correct answer for all of them. By the
  time a call is counted the collapse has already happened, so it has to be read out of prosper's own
  registration tables. Crossed against `nid_gate_scan.py --all-nids` it names the exact rows a
  `--values` measurement cannot support an inference from: on *Sonic Origins* (`PPSA05325`), **36 of
  247** gated rows. The biggest collapses in the tree are `k_attr_noop` (20 Sony names), `s_ok` (14)
  and `font_ok` (12).
  Registration shapes are **discovered, not hardcoded** — the API list is read out of `class Hle` in
  `dispatch.hpp`, and the wrapper list is every macro/lambda/free function forwarding a parameter
  into a registration call — and the parse is **reconciled against the registry the binary actually
  builds** (`re/hle_registry_dump.cpp`, ctest `re_hle_handler_map_reconcile`), so a shape it cannot
  read fails loudly instead of quietly shrinking the census. Exit `3` means the table is a lower
  bound; exit `2` means nothing was parsed. **`--platform` is required, with no default**:
  `hle_kernel_mem.cpp` defines `register_kernel_mem_hle()` twice in two arms of one `#if`, and
  counting both promotes five single-name handlers to "shared" — that plus per-site (rather than
  per-distinct-name) counting is the whole difference between the 41 this measurement was first
  published with and the 36 that is correct (#2070). There is no platform-independent answer to give,
  and `s_ok` is a case where the platform-blind count is wrong on *every* platform, so the tool
  refuses to pick for you. See `re/README.md`.
- **`re/pak_index.py`** — resolve UE4 `.pak` byte offsets to asset names, and decode a
  `PROSPER_FILELOG=1` run's `[apr] read-submit` stream into an ordered asset load trace. Answers
  "which map/blueprint/texture did the guest actually load, and where did loading stop?" offline,
  from a log captured earlier — no boot and no GPU. Check what content is resident *before*
  concluding that geometry is missing: on PPSA19244 it showed the title loads exactly one of 481
  maps (a splash level), which made its empty base pass correct rather than a defect. See
  `re/README.md`.
- **`re/xref.py`** — find relative data pointers, direct references, runtime function-table
  writers, and indirect callers in a flattened guest module. See `re/README.md`.
- **`re/disasm_words.py`** — name an RDNA2 instruction from the raw dwords a diagnostic printed.
  `[recompile-reject]` prints `words=AABBCCDD,EEFFGGHH`; this pipes those bytes through
  `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` and prints the mnemonic, deduplicated. Takes
  word pairs as arguments or reads a log on stdin (`grep recompile-reject run.log | disasm_words.py`).
  Use it instead of reading an opcode table: neighbouring opcodes in this ISA are unrelated
  operations — `0x305` is `v_mul_lo_u16`, not the shift its neighbour `0x307` is, and an early draft
  of #2067 shipped that guess through `spirv-val` before a bit-exact test caught it. Requires
  `llvm-mc` with the AMDGPU target on PATH.
- **`shader_histo/`** — histogram RDNA2 opcodes across a title's shaders.
- **`shader_inspect/`** — decode one raw `PROSPER_SHADER_DUMP` binary offline. It prints bounded
  instruction PCs, operands, raw words, signed branch immediates, and resolved branch targets so a
  failed shader's CFG can be mapped without hand-counting variable-length instructions.
  **`--mimg-sites` is the machine-readable half (#3184)**: one `mimg-site pc=… op=… dim=…` line per
  image instruction and nothing else, so a tool that needs to know where the MIMG instructions are
  never has to carry its own port of the RDNA2 length rules. `shader_conformance/scan.py` did carry
  one, and now consumes this instead. The trailing `mimg-sites-end` line is a completion sentinel —
  a real empty census must never look like a process that died before printing.
  **And it answers "which opcode?", never "which value?" (#2132).** A fold that stops early looks
  identical whether it ran out of modelled opcodes or out of *readable data* — the reject line is the
  same — and a disassembler can only see the first. Sonic Racing: CrossWorlds' failing vertex fetch
  was hypothesised from a static read as a runtime-selected descriptor the fold could not model; it
  models that idiom completely, and the real cause was a **null pointer** read out of a constant
  buffer the fold read correctly. When the chain runs through memory, use `PROSPER_DYNTRACE_FAIL=1`,
  which reports `base_ok`/`soff_ok`/`unreadable` per step, before reaching for this tool.
  **`--stage` cannot prove a shader is unsupported (#1571).** A raw dump has no descriptors, so the tool
  has no `ShaderResourceTable`, and the recompiler correctly refuses `MIMG`/`MUBUF`/`MTBUF` in any stage
  plus `SMEM` in the vertex/fragment stages (`recompile_fragment_impl` / `recompile_vertex_impl` gate on
  `allow_smem = (rt != nullptr)`; `recompile_compute` passes `true`, and `emit_alu` holds both the MIMG
  `!allow_smem || !rt` and SMEM `!allow_smem` rejects). That is a
  limitation of *this tool's inputs*, not a property of the shader. Such a run reports
  `status=undetermined-no-resource-table` (exit 3) — treat it as **no verdict**, never as a missing
  opcode. Before this was fixed the tool called 109 of 114 known-good shaders `rejected` and agents
  chased those false leads. **For a table-accurate verdict use `gpu_replay --inspect-only`**, which has
  the real descriptors from a capture. See `shader_inspect/README.md`.
  **And a table-less run has no LAUNCH STATE either, which is a second, separate way its reject PC can
  be a phantom.** With a default `ComputeShaderConfig` no user SGPR or workgroup-id register is seeded,
  so the Wave64 MUST dataflow starts with an empty `scalar_words` and every proof resting on a launch
  input fails offline that would hold live. Supply it — `PROSPER_SHADER_INSPECT_USER_SGPRS=N`,
  `PROSPER_SHADER_INSPECT_TGID=xy`, `PROSPER_SHADER_INSPECT_LOCAL=16x3x1` (the census line's `local=`
  is where the last comes from) — and check the offline PC against the live one before reasoning from
  it. Sonic Frontiers' `0x2005717e00` declines at pc96 with the default config and at **pc481**, the
  live PC, once the launch shape is given: 385 dwords and a different instruction apart (#2790).
- **`PROSPER_DBG_PROGRAM=0x…[,0x…]`** — the verbose recompiler stream for NAMED guest program
  addresses only. `PROSPER_DBG=1` enables every `[recompile-reject]`, `[compute-cfg]`,
  `[compute-struct-reject]`, `[compute-cfg-reject]`, `[structured-wave-reject]` and `[divloop-reject]`
  line for every shader in the title, which on a routed run is ~1.5 GB and slow enough to desync the
  pad script that reaches the phase being diagnosed — so "which route declined THIS program, on the run
  that reaches it" was a question the instrument could not be pointed at, and the offline substitute has
  the launch-state trap above. This narrows the same stream to a handful of addresses at negligible
  cost. It is also the only way to see a **route-decline** live: `record_terminal_reject_reason` keeps
  the LAST non-consequent reason, so the census reports whatever the final fallback hit and every
  earlier route's decline is invisible without the verbose stream. Take the addresses from the
  `[compute-census]` per-program lines; they are run-local, so re-derive them per run (#2790).
- **`imgdump/`** — decode/dump a guest texture to an image for inspection.
- **`gpu_busy.sh`** — answers "is a peer lane mid-run on the shared GPU?" with pgrep-convention
  exit codes (0 busy / 1 free / 2 tool error), counting the five consumer names including
  `screenshot_snap`. Replaces the inline `pgrep -x a b c` spelling that was a usage error
  masquerading as a clean box (trap 222, #2948).
- **`determinism/`** — "does replaying this frozen capture twice give the same picture?", and the
  rule for reading the answer. `replay_determinism.sh` replays a capture N times in N separate
  processes with `tools/vkprobe` beside it as a control, alternating order and loaded/unloaded
  blocks; `replay_determinism_report.py` turns that CSV into one of **three** verdicts. The third is
  the point: #2945's failure rate drifts machine-wide over minutes, so "every replay agreed" is
  produced equally by a repaired renderer and by a quiet half-hour, and the report answers
  `UNDECIDED` rather than `DETERMINISTIC` when the control never failed either. ctest case
  `replay_determinism_report` (platform-independent; it reads a CSV and touches no GPU).
- **`gpu_replay/`** — replay a local `PROSPER_GPU_CAPTURE` realized-submit capsule through the same
  Vulkan backend without booting the guest. Capsules include game shaders/resources, use `.prgcap`,
  are gitignored, and must never be committed. The tool exits non-zero on output-hash mismatch.
  See `gpu_replay/README.md` for inspection, validation, dependency graphs, oracle semantics, and extraction.
- **`gpu_timeline/`** — inspect a native-speed `.prgtl` submit/present index recorded with
  `PROSPER_GPU_TIMELINE=<path>`. It does not invoke Vulkan; use it to locate progression and producer
  windows before making an expensive realized `.prgcap`. Timeline files are gitignored and local-only.
- **`colorstate/colorstate_report.py`** — reduce a `PROSPER_COLORSTATETRACE` log to a verdict on the
  "why is it black?" question: do draws reach the **scanout** (vs offscreen only), and are their colour
  writes enabled or suppressed by a zero target/shader mask (since #1724, `MODE` does not gate them; set
  `PROSPER_LEGACY_CB_DISABLE_MASK=1` to restore the old `MODE=DISABLE` override for a revert-free A/B)? Prints a
  per-guest-minute suppressed-percentage series. **Always compare a known-good phase against the bad one** —
  on Plucky the `mode=0` fraction is *higher* while the world renders correctly (88-95%) than while the
  screen is black (83%), because `CB_DISABLE` is the depth/shadow prepass. Reading one phase alone yields a
  confident wrong answer. `--selftest` needs no capture. See `tools/colorstate/README.md`.
- **`spv_validate/`** — the strict SPIR-V validation gate (ctest `spv_validate`). It emits one
  representative module from **every** SPIR-V-producing entry point declared in any header under
  `src/gpu/` or `frontends/shared/` (recursively, under either return-type spelling the tree uses)
  and runs `spirv-val --target-env vulkan1.1` on each. It fails when a module is invalid, when
  `spirv-val` is not installed (it used to report PASS — see #1711), and when a declared emitter did
  not produce a validated module. Coverage is recorded at runtime by passing the emitter's name to
  `dump()`, so a call whose result is discarded does not count. Adding an emitter therefore means
  adding a module; a name that emits no SPIR-V goes in `kNotEmitters` with a reason, and a genuine
  gap goes in `kKnownGaps` with its issue — that list is meant to stay empty.
  **`spirv-val` must be on `PATH`** (`spirv-tools`, `mingw-w64-ucrt-x86_64-spirv-tools`, or
  `brew install spirv-tools`); its absence fails the suite rather than skipping the gate.
- **Writing a test arm in this directory? One question first, and it is cheaper than a mutation run:**
  **"what else in this output could satisfy this assertion?"** An arm discriminates only if it asserts
  on a string **only the branch under test can produce**. Five void arms shipped here in one session
  and every one failed that question: `expect="#11"` for a collision line the per-PR table row also
  prints; `want_rc=1` for a fail-closed path whose silent fallback exits 1 anyway; `expect="4"` for a
  `--quiet` output any report containing a 4 satisfies; a `// s_trap 1` fixture whose operand is a
  valid row. It does not replace mutating the code — an arm can name a unique string and still test
  the wrong branch — but nothing that fails this question is worth mutating. Related: the defects
  those arms missed were all in the tool's **advertised** faculty (a citation auditor blind to
  sentence-final citations while certifying "all resolving"; a conflict scanner that would reject the
  documentation of its own defect; a collision reporter that could not report a collision), which is
  the one place nobody points an arm, because reasoning about it hard feels like testing it.
- **`docs/check_numbered_table.py`** — validate Markdown tables that other documents cite by row
  number. Four classes: **structure** (always on) rejects a blank line that splits a table, which
  in Markdown silently renders everything after it as a *separate* table; **arity** (always on)
  requires every row to have its header's cell count; **`--ordered`**
  (opt-in, plus `--table-header` to select one table) requires the numbered column
  to be unique and strictly ascending; **`--baseline <base copy>`** requires that every row number
  the base had is still present. Order is a convention of the instrument-trap table
  ("append, never renumber"), not a property of numbered tables — most here lead with frame or
  draw ordinals where repeats are correct — so do not apply it broadly. Catches the case where two
  branches append the same row number, which merges textually clean and green. Run by the CI
  `Docs` job; `ctest -R doc_table_checker` covers the checker itself.
  **`--sequential` was removed on 2026-08-17 and now errors** (#2089). It also required the column to
  be *gapless*, which made a lane's `Docs` job red purely because a lower number sat in another
  lane's unmerged branch — unrepairable by its author, since the only local fix is the forbidden
  renumber. It was kept for catching a **deleted** row, and measurement showed it only ever caught an
  *interior* deletion: on master's 186-row Instrument table, dropping the highest row passed green,
  and a
  whole-file `git checkout` from an old revision lost 61 rows and reported "contiguous and unbroken".
  `--baseline` does that job completely — interior, tail, whole-file revert, and renumber — and
  cannot be made to fire by another lane's timing. **Gaps are now legal**, so a collision is repaired
  by bumping to any higher number and merging.
  **`--baseline` takes a FILE PATH, not a git ref** — materialise one with
  `B=$(mktemp); git show "origin/main:<file>" > "$B"` (CI uses `git show HEAD^1:<file>`). Handed a
  ref it used to fail with a bare `cannot read`, which reads as a problem with the document rather
  than with the invocation; it now names the contract (#2675).
  **Every failure is announced on stdout as well**, because this gate is quoted as a copy-pasteable
  recipe and recipes get pasted into pipelines: `check … | tail` discards the exit status *and*
  stderr, so a real failure — and a refused flag — used to print nothing at all and look green. The
  detail still goes to stderr; stdout carries `N problem(s) found.` or
  `THE CHECK DID NOT RUN -- usage error: <what>` (quoted as the tool actually spells it: its output
  is ASCII, and a doc that prettifies a literal defeats a grep for the line the tool printed).
  **On arity, and why it is not optional:** GFM splits a row into cells on `|` *before* it parses
  inline content, so a pipe inside an inline `code span` is a cell boundary, and a row with more
  cells than the header has the excess **silently discarded** on the rendered page. Write `\|`
  inside any code span in a table cell. This is not hypothetical — six rows of the instrument-trap
  table rendered truncated on GitHub for months, trap 40 among them, while this very tool reported
  the file `unbroken` (#2108). What it does **not** cover, stated so silence is not read as
  coverage: HTML tables, delimiter-less pipe blocks (no header to measure against), whether an
  escaped pipe was what the author meant, and fenced regions, which are skipped deliberately.
- **`docs/check_trap_citations.py`** — the other half of the numbering contract: every `trap NNN`
  reference in the repository must name a row that exists. `check_numbered_table.py` validates the
  TABLE and has no idea anything cites it, so until this existed a reference to a row that never
  existed read as perfectly correct — a plausible number in a plausible sentence, findable only by
  opening the table and counting, which nobody does for a number in a code comment. **That is the
  more expensive half:** a duplicate row is visible the moment you look at the table, while a stale
  by-number reference quietly sends the next agent to an unrelated entry. Roughly half the references
  are `.cpp`/`.hpp`/`.py` comments rather than prose, so this is a contract compiled files depend on;
  the tool prints the live counts on every run, so no figure is quoted here to go stale. Deliberately
  narrow about what counts as a reference, because it scans every tracked file and one that fires on
  ordinary prose gets disabled — `trap 41`, `traps 55 and 56`, `instrument-trap 43` and `trap #13`
  count, while `s_trap 1` (an RDNA2 mnemonic), `WRITE-TRAP #1` (a pasted log line), `trap 0x40`,
  `traps 1234` and `entry 99` do not. Each exclusion is measured: removing the leading boundary
  admits 4 extra matches, **3 of which cite row 0** and would turn this repo-wide gate red on correct
  code; removing the trailing boundary makes `\|d{1,3}` bite a prefix out of `0x40` and `1234`, the
  second resolving **silently** to an unrelated row. That boundary is `(?!\|w\|\|.\|d)` and **not** the
  obvious `(?![\|w.])`, which rejects a full stop — so `See instrument trap 41.` stopped being a
  citation at all, losing 22 of 118 references (5 of them in `.cpp`/`.hpp`/test comments) **with the
  gate still green and the success line still saying "all resolving"**. Run by the CI `Docs` job;
  `ctest -R trap_citation_checker` covers it.
- **`docs/check_merge_result.py`** — run the numbered-table gate against the **merge result**, against
  a freshly fetched base, as the last step before merging (#2211). A `pull_request` job validates
  `refs/pull/N/merge`, which GitHub computes when the event fires and never recomputes as the base
  moves, so a **green** required check is a statement about a merge that may no longer exist — and
  unlike a red one, nobody re-derives it (instrument trap 189). Uses `git merge-tree --write-tree`,
  so it touches neither your working tree nor the index and cannot collide with another agent in the
  same repository. Reports a textual conflict as a conflict rather than as a table defect, and a
  checker failure as a checker failure. **`--base` defaults to `origin/main`, which is wrong for a
  stacked PR** — pass the branch it will actually merge into, or the green run describes a merge that
  will never happen.
  **Measured boundary, so its value is not overstated:** for two same-numbered rows inserted at a row
  boundary in an otherwise identical file, separation 0 lines (both appending at the tail) makes git
  *conflict*, and separation ≥ 1 leaves the offending branch **already red on its own head** —
  because inserting a duplicate-numbered row above the tail puts that branch's *own* column out of
  ascending order, which is an artifact of how those cases were generated rather than a general
  property of separated insertions. So in that family `--ordered` alone closes the hole. What it does
  **not** close, and this tool does, is the artifact a *human* produces: resolving that separation-0
  conflict by keeping both rows gives a duplicate on master while **both heads were green**, which is
  how `33, 34, 35, 32` reached master in #1696. `ctest -R doc_merge_result_checker` covers it.
- **`docs/trap_number.py`** — allocate the next instrument-trap row number against `origin/main`
  **and every open PR** (#1729). Prints each claimant so you can see whether you are in a race, not
  just a bare number. An **advisor, not a gate**: two lanes running it in the same minute both see the
  same free number, so it shrinks the collision window and cannot close it — merge order does that,
  and `--ordered` is the backstop. Reading master alone is what produced the #2574/#2581 collision.
  It **errors rather than answering** when `--limit` truncates the PR list or when `gh` is
  unauthenticated: a smaller list yields a confidently wrong "next free number", which is the exact
  defect the tool exists to prevent, wearing the look of a successful run. It also scans (rather than
  skips) any PR whose `files` array is at GitHub's silent 100-entry cap — measured: `cli/cli#14082`
  has 1,161 changed files and `gh --json files` returns 100 with no indicator. When more than one open
  PR claims the same number it says so and suggests stepping *clear* of the contested band rather than
  to the next free number, since every loser stepping to "next free" collides again one number up.
  `ctest -R trap_number` covers it.
- **`output/check_ascii_output.py`** — refuse a non-ASCII character inside a C/C++ **string
  literal**. `stdout` carries bytes and the reader decodes by platform: UTF-8 on Linux, the
  cp1252/cp437 console code page on Windows. One em dash therefore reaches a Windows user as
  mojibake, so `grep` for the line the tool itself printed does not match it — fatal for a tool
  whose documented recipe is `--import-slots | grep <name>` — and a test asserting on the rendered
  string fails on Windows only. #2579 shipped five that way with Linux CI green (#2588). Runs as
  ctest `ascii_output_literals` and in the CI `Docs` job. **Comments and Markdown keep their
  punctuation**: comments are stripped before literals are extracted, so the rule is about the
  bytes a program emits, never source style. Replacements are `--`, `->`, `...`, `"`, `x`.
  Three tiers, because a gate with standing false positives is one people learn to skip: a raw
  non-ASCII character and a `\u`/`\U` escape above 127 **fail**; a `\x`/octal byte escape above 127
  is a **note**, since a multi-byte run like `\xe2\x80\x94` is binary data by construction (the two
  in the tree are UTF-8 fixtures feeding conversion tests, not messages). The note tier is honestly
  weaker for a *lone* high byte — `printf("caf\xe9")` is the same defect in escape form and is only
  noted — so the note **count** is printed on every run to keep growth visible. The tier is decided
  before the quarantine, not after, or a binary fixture added to a quarantined file would be
  reported as the defect class it is not.
  It prints how many files and literals it examined on **every** run, pass or fail — a verdict alone
  cannot be told apart from a scan that saw nothing — and it self-tests both its parsing and its
  **line numbers**, the second because its first revision drifted its counter across an unterminated
  quote and named innocent lines with full confidence.
  What it does **not** cover, stated so silence is not read as coverage: **Python, shell and CMake
  literals** (#2609 — 109 characters in 30 files, the same mojibake class, measured); Markdown and
  comments, deliberately; anything assembled at runtime from a data file, `argv`, or the guest's own
  UTF-8 strings printed through `%s`; **char literals** (`L'—'` is invisible — their spans are
  skipped so a quote inside one cannot open a string, and their contents are never read); **C++23
  delimited escapes** (`\u{2014}`, `\N{EM DASH}`, which GCC and Clang already accept as extensions);
  and the files in its `QUARANTINE` ledger, which #2588 left to
  the GPU/compute lane (#2608). That ledger fails on a count that RISES *and* on one that falls,
  so it cannot outlive the defect it records.
- **`niddiag/`, `fetch_niddb.sh`** — NID (Sony symbol hash) resolution helpers.
- **`PROSPER_MB3_POISON`, `PROSPER_PEND_AGE`, `PROSPER_SUBMIT_STALL_US`** — the three in-emulator
  diagnostics for the MallocBinned3 free-list corruption family (#1945/#1226). `MB3_POISON` walks the
  guest's own size-class-1 bundle chains after each submit and reports a poisoned link *while it is
  still in the chain*, with the label-history of the node holding it — the only moment that node's
  address exists anywhere. `PEND_AGE` reports how long each completion write really sat in the pend
  queue before it landed in guest memory. `SUBMIT_STALL_US` sleeps in the guest's own submit call.
  **Read the confound warning before using the first one as an observer**: on Crisis Core
  (`PPSA07809`) arming `MB3_POISON` is the difference between the title dying in seconds and booting
  to its title screen, because the walk's *duration* changes the outcome — a measurement taken with
  it armed is taken on a title that would otherwise have died. `SUBMIT_STALL_US` is the honest lever
  when a throttle is what you want. `prosper/docs/CRISIS_CORE_STATUS.md` has the dose-response.
  **`PROSPER_SUBMIT_STALL_OUTSIDE=1`** pairs with it: the same sleep, same length, same thread, but
  after `g_agc_state_mu` is released instead of while it is held. The stall's call site is inside the
  submit mutex, so a rescued run has two candidate mechanisms — the delay, or the fact that prosper
  serialises the Dcb and Acb submit entry points for its duration — and this is the arm that
  separates them. It reports `NOT ARMED` when no stall duration is set, so a mis-typed arm cannot
  read as an armed null. Measured on ArcRunner: 0/3 faulted either way against a 3/3 unthrottled
  control, i.e. the rescue is the delay (`prosper/docs/ARCRUNNER_STATUS.md`).
- **`PROSPER_FOLD_MARGIN`** — the **per-fold** account of that same family, and the instrument to
  reach for when a whole-run rate has stopped discriminating. Two halves arm together:
  a per-submit ledger (`hle_agc.cpp`) splitting each fold into `gap` (the guest's own time between
  prosper's return and its next submit), `lock`, `work` and `stall` — the stall **measured**, not
  assumed — and a label-protocol census (`command_processor.cpp`) counting the recycle race in
  **folds** rather than milliseconds. Set `=1` for totals, `=2` for one line per fold. Why folds: a
  wall-clock age is not comparable between a route that adds a fixed delay per submit and one that
  does not, and two ArcRunner `## Ruled out` rows were retracted for exactly that. Three
  practicalities. **Level 2 is not timing-neutral** — an ArcRunner control run lives ~9 s at level 1
  and ~13 s at level 2, so compare level-2 arms only with level-2 arms. **`REBUILD-BEFORE-EXEC` is
  the number to score a candidate fix against**: it fires at the *guest's* own builder call when it
  rebuilds a label whose previous generation prosper has not executed, and it agrees exactly with
  `SUSPECT-REL1-OVERLAP` (fence side) and DMA-INIT-GEN's `depth>=2` (init side) in every arm — three
  points in one protocol, so a claim can be cross-checked rather than trusted. And the per-fold
  `built` column counts what the guest's **builder thread** did *while prosper was inside that fold*,
  which is how the ArcRunner mechanism became visible at all.
- **`PROSPER_POST_SUBMIT_VISIBILITY`** — `=1` forces prosper's post-submit completion-visibility
  model on regardless of the SDK version the guest requested, `=0` forces it off; unset leaves the
  `version >= 13` gate in `agc_reg_defaults.cpp` alone. The model holds a submit's completion writes
  private until the submit scope closes, so the guest cannot observe a half-retired frame. It is an
  A/B lever for whether that gate is right for a pre-13 title: ArcRunner requests version 10, and
  forcing the model on takes its default route from 3-of-3 faulting to 3-of-3 surviving
  (`prosper/docs/ARCRUNNER_STATUS.md` § 2026-08-07).
- **`hostprof/hostprof.py`** — poor-man's **native sampling profiler**: attach to a running process
  (pid or name), sample its threads via repeated `gdb` backtraces, and rank the hot leaf functions —
  the HOST-side "which C++ function is burning CPU" first look (render/submit thread, readback copy,
  detile, FP16 decode). Fallback where the required `perf` recording is denied; a paranoid value
  of 2 excludes kernel sampling but does not universally forbid user-space profiling.
  `--thread REGEX` isolates one thread, `--mode folded` emits flamegraph stacks. Complements
  `guest_bt` (which does the GUEST/managed-C# side). `--self-test` checks the symbol parser.
  See `hostprof/README.md`.
- **`dropcache.py`** — evict a dump (or any path) from the **host page cache** before a timed run,
  and prove it happened. A guest `read`/`pread` is served from the page cache on every launch after
  the first, so a title's asset load can be several times faster than the same bytes off storage —
  which is invisible for most work and decisive for a startup *race*: PPSA26414 reaches `DLLInit` in
  289–362 ms warm and 779–1158 ms evicted, and it faults in the first case and boots in the second
  (`docs/R_TYPE_DELTA_STATUS.md`). `posix_fadvise(POSIX_FADV_DONTNEED)` on the named files only, so it
  needs no root and touches nothing outside them — but **the page cache is global**, so evicting a
  shared dump evicts it for every process on the box and will silently invalidate another lane's
  timing run. Say what you are evicting first. It prints `mincore(2)` residency **before and after**,
  because "evicted it", "it was already cold" and "eviction silently did nothing" otherwise look
  identical, and it exits non-zero (never silently) when pages stay resident, when a path is absent,
  and when a flag is misspelled. `--report-only` measures without evicting; `--self-test` needs no
  dump. **Record cache state alongside any startup-timing number**; without it it is unreproducible.
- **F8 interactive performance capture / `perf/performance_capture_report.py`** — while playing in
  `prosper-app`, press **F8** when performance is bad. The app keeps a cheap 4 Hz rolling ring of
  process CPU/RSS plus guest-flip, rendered-frame, and host-present counts; the press freezes the
  preceding 5 seconds and enables the existing structured renderer/compute timers for the following
  5 seconds. It then atomically publishes one bounded
  `perf_capture_<titleId>_<timestamp>.prperf` under `PROSPER_CAPTURE_DIR` (default cwd). No GPU command
  capture, screenshot, or ordinary frame dump is enabled by F8. The rendered-frame counter is the
  CPU frame-handoff population; shared-device direct GPU present skips that handoff and serializes
  the field as unavailable (`null`), so the report must not infer a zero-rate pacing gap from it.
  Detailed renderer and compute records are capped at 4096 each, and the footer reports exact
  retained and dropped counts — never read the cap as the population. Missing Vulkan timestamp
  samples are explicit: the report leaves GPU device
  time unavailable and reports total GPU wait without inventing a device/host-overhead split. F8
  enables structured timing clocks but not the existing high-volume periodic stderr timing logs. A
  graceful exit before the window completes removes the private `.part` rather than publishing a
  short final file. A homogeneous compute batch also retains its run-local program address and a
  stable SPIR-V hash; the report groups at most the ten costliest hashes by record/dispatch count,
  total/mean/max time, and bounded address list. Mixed batches and older v1 captures report their
  compute time as explicitly unknown identity rather than inventing a program attribution.

  For unattended agent runs, set `PROSPER_PERF_CAPTURE_AFTER_MS=N` to make one automatic arm
  attempt after `N` milliseconds from entry into the app loop. This uses the exact F8 artifact path
  and five-second pre/post windows without desktop focus, synthetic input, screenshots, frame dumps,
  or GPU command capture. The value must be a positive decimal integer whose nanosecond conversion
  fits in 64 bits. Startup logs whether the setting was accepted or ignored, and the trigger line
  proves when the one-shot attempt fired. A failed arm is not retried at a later phase.

  Inspect it offline with:

  ```bash
  python3 tools/perf/performance_capture_report.py capture.prperf
  ```

  The report separates evidence for CPU work outside the timed renderer, renderer resource work,
  GPU device/wait/readback cost, compute batches, and frame-pacing gaps. It deliberately reports
  unavailable/inconclusive fields instead of inventing a verdict. This is a broad localizer, not a
  stack sampler: after it points at host CPU, use `perf` (`hostprof` when unavailable); after it points at a graphics phase,
  use the existing focused timing/capture tools. The always-on pre-trigger cost is one atomic due
  check per app loop and one process sample every 250 ms; detailed clocks run only post-trigger.
- **`perf/compute_phase_report.py`** — roll up the **compute** side of a run, the way
  `PROSPER_RENDER_TIMING` already rolls up graphics. `PROSPER_COMPUTE_PHASE_TIMING=1` and
  `PROSPER_COMPUTE_IMAGE_TIMING=1` emit a 17-timer decomposition *per dispatch*, which is unreadable
  at run scale; this aggregates both record types offline into ms / share / mean-per-dispatch, nested
  so each sub-timer sits under the phase containing it, plus the costliest programs and each one's
  dominant leaf. Image records also carry `addr=`, `persistent=` and `upload-skipped=`; the report
  ranks real bindings by stable shader hash + binding + sampled/storage class, retains at most four
  address variants per group, and prints missing fields from older logs as unknown rather than zero.
  Aliases expose their owner's state in the raw record but remain excluded from cost rollups.
  **Run both switches together** — `setup_ms` spans the image-binding loop, which has
  no sub-timer of its own, so phase timing alone reports a large `setup` its named children do not
  explain. Sampled cache lookup is a sibling of upload preparation; storage cache validation is
  measured *inside* `prepare_ms`, so the table prints storage cache as an included child plus the
  exclusive preparation remainder. It prints an `unattributed` row under every parent and warns on
  impossible nesting instead of publishing a negative residual. It excludes failed dispatches (their
  sub-timers are meaningless, see trap 47 in `docs/GAME_COMPAT_ORCHESTRATION.md`), and warns if its
  model of `execute_item` stops matching the emitter. Counts here cover backend-executed dispatches
  only — CPU-fast-path fills emit no record, so add
  `[render-timing] compute_cpu_fast fills=N` before quoting a rate.
  Restrict a rerun by the F8 report's cross-run SPIR-V identity with
  `PROSPER_COMPUTE_TIMING_HASH=0x...`; use `PROSPER_COMPUTE_TIMING_CODE=0x...` only for a known
  run-local address. If both are present they are an AND. The accepted/ignored banner, first-match
  line, and termination `seen`/`matched` summary are part of the validity contract: zero matches is
  apparatus-invalid. `prosper-app` publishes that summary explicitly before its deliberate `_Exit`;
  ordinary teardown retains the same idempotent destructor fallback. Every selected phase/image
  record carries both identities. The selector's pure
  policy is covered by ctest `compute_timing_selector_policy`; run
  `tools/perf/mutate_compute_timing_selector.sh` to prove the exact mismatch, zero-match, and
  duplicate-summary checks kill their corresponding mutations.
  `test_compute_phase_report.py` self-tests it (ctest `compute_phase_report_logic`), and
  `mutate_compute_phase_report.sh` checks that suite at **per-check granularity** — each mutation must
  be killed by the check written for it, because a survivor masked by red siblings is invisible when
  you only watch the suite's colour (trap 48). It mutates a scratch copy, never the tracked file.
  First result: `docs/RENDERER_PERFORMANCE_2026_07.md` § Astro Bot compute decomposition.
- **`perf/ab_compute.sh`** — A/B one `PROSPER_*` switch against a routed live run, refusing to
  measure while another `prosper-app` holds the GPU and stamping commit/route/reps onto the result.
- **`perf/stack_profile.py`** — when a title is slow because threads are **waiting**, this names the
  code location each one waits at, by periodically attaching gdb and aggregating stacks per thread.
  It is the second half of a pair: read `/proc` first for *how much* a thread blocks (cheap, high
  rate), then use this for *where* (expensive, low rate).
  ```bash
  python3 tools/perf/stack_profile.py --pid $(pgrep -x prosper-app) --samples 12 --interval 5
  ```
  Three things about it are deliberate, and each came from a measured failure during bring-up:
  - **`/proc` alone cannot answer "which lock".** A mutex wait and a condition-variable wait are the
    same state, the same `wchan` (`futex_do_wait`) and the same syscall (202). On a purpose-built
    control with three threads blocked in three known functions, `/proc` reported two of them
    identically; the stack separated them. Do not conclude *which* primitive from `wchan`.
  - **Verify debugger access to the actual target.** Earlier measurements found host+host and
    distrobox+distrobox worked, while host process + in-container gdb was denied. Prefer the
    target's environment, but credentials, namespaces and security policy determine access;
    neither "always host" nor "always container" guarantees it. This matters
    because the denial produces **no stacks**, which is byte-identical to a process with nothing
    blocked — so a mis-sited gdb reads as a clean result. The tool detects that case by name; it
    counts any empty sample as FAILED, prints the failed count even when zero, and exits non-zero if
    every sample failed rather than printing an empty report that reads clean.
  - **Every sample stops the process** (~90 ms on a 4-thread toy, more on prosper). The report prints
    total stopped time as a share of wall clock *before* any finding, and warns above 10%, because a
    profiler that quietly steals wall clock will manufacture the frame-rate problem you came to find.

  `test_stack_profile.py` self-tests the classifier against **recorded** gdb output, so it needs no
  gdb and cannot skip silently. Its fixture is the real output that broke the first version: glibc
  reports `pthread_cond_wait@@GLIBC_2.3.2`, whose version suffix defeated the `$`-anchored patterns,
  so libc internals were reported as the application's blocking site — a wrong answer that looks
  entirely plausible, and was caught only because the control had known-correct answers to contradict.

Verification here is agentic-first (see `docs/VERIFICATION.md`): prefer a
programmatic check (ctest exit code, `spirv-val`, a snapshot hash) over eyeballing.

To drive any runner through a longer input route, set
`PROSPER_PAD_SCRIPT=@scripts/<title>/reach-<state>.pad`. Route files use the same
seconds/flip/pad-read syntax as inline scripts (`3:`, `f300:`, or `p1200:`), accept one entry per
line, `#` comments,
and explicit ranges such as `f300-340:cross`. Full-deflection stick actions use names such as
`left-stick-left` and can be combined with buttons using `+`. See `docs/INPUT_REPLAY.md`.
Set `PROSPER_PAD_RECORD=<path>` on any runner, or use `prosper-app --record <path>`, to capture the
final button stream in that format. Recording uses `fA-B:` flip ranges by default; set
`PROSPER_PAD_RECORD_AXIS=pad-read`, or add `prosper-app --record-axis pad-read`, for `pA-B:` ranges
that advance only on successful input-state reads. Completed button intervals are flushed immediately;
scripted stick directions are supported for playback but are not yet emitted by the recorder.
Set `PROSPER_PAD_SCRIPT_LOG=1` to log each scripted state transition observed at a pad poll.
For long exploratory runs, add `PROSPER_PAD_SCRIPT_RELOAD=1` to live-reload an `@file` route while
preserving its original time/flip/read origin; append only future windows and confirm the reload log.
Wall-clock ranges can be skipped entirely when their duration is shorter than the interval between
polls, especially under synchronous software rendering; use poll-safe holds with neutral gaps,
flip-anchored ranges while presentation advances, or `pA-B:` ranges keyed to the pad-read index printed
by `PROSPER_PAD_SCRIPT_LOG=1`. Point entries use `PROSPER_PAD_FRAME_HOLD` or
`PROSPER_PAD_READ_HOLD` (both default 8) on their count axis.
Pad-read indices advance only for successful `scePadRead`/`scePadReadState` calls; controller metadata
queries and rejected reads do not consume pad-read entries. Seconds and flips keep their first-pad-poll
origin for compatibility with existing routes.

Capture one draw-carrying renderer invocation with:

```bash
PROSPER_GPU_CAPTURE=/tmp/messenger-level.prgcap PROSPER_GPU_CAPTURE_AT=0 \
  PROSPER_GPU_CAPTURE_MIN_DRAWS=30 \
  PROSPER_CAPTURE_REVISION=$(git rev-parse HEAD) \
  PROSPER_CAPTURE_TITLE=PPSA24651 <normal boot_trace command>
./build-linux/gpu_replay /tmp/messenger-level.prgcap /tmp/replayed.bmp
```

`PROSPER_GPU_CAPTURE_MIN_DRAWS`/`MAX_DRAWS` filter by realized item count; `PROSPER_GPU_CAPTURE_AT`
counts matching invocations that reach the registered renderer, after the normal `RENDER_EVERY`
sampling. Aim the live run near the target first; the capture itself writes once.
`PROSPER_GPU_CAPTURE_COMPUTE_ADDR=0x...` additionally requires the exact compute program address.
Use it for a late or intermittent compute fault when predicting a renderer invocation or semantic
submit number would be fragile. If an earlier producer failure leaves a selected indirect dispatch's
arguments unavailable, the capsule keeps that operation unrealized with its zero/unknown launch but
still retains the semantic program's bounded raw shader and descriptor metadata. Inspect or dump that
failed stage offline instead of rerunning the title merely to recover its shader bytes.
`PROSPER_GPU_CAPTURE_SHADER_ADDR=0x...` does the same for any realized or semantic draw stage,
including a shader whose draw failed realization and therefore never reached the renderer.
`PROSPER_GPU_CAPTURE_TARGET_DIM=WxH` requires at least one realized or semantic color target with the
exact dimensions. Use it to select a native composition submit without relying on a title-specific
shader address; malformed or zero dimensions do not match.
The live hook snapshots realized draws, compute dispatches, and the original mixed PM4 operation order
before executing the selected submit, then attaches the rendered pixel oracle afterward. An inspected
mixed capsule must report the same draw/compute/operation counts as the live timing line; `computes=0`
for a known mixed submit indicates an obsolete capture build, not proof that the barriers are unnecessary.
`PROSPER_GPU_CAPTURE_AFTER=N` ignores the first `N` renderer invocations before applying the draw-count
filters and `AT` counter. Pair it with `PROSPER_SUBMITLOG`/`PROSPER_RENDER_FIRST` when several early
scenes share the same draw count as a late target.
Resource bytes are preflighted before allocation and default to a 512 MiB total limit. Raise it with
`PROSPER_GPU_CAPTURE_MAX_MB=1..3072` only when a replayable capsule genuinely needs the data. For a
suspect descriptor or very large submit, set `PROSPER_GPU_CAPTURE_METADATA_ONLY=1`: the thin capsule
keeps shaders, operations, pipeline state, and resource descriptors for `--inspect-only`, `--validate`,
and `--graph`, but deliberately cannot render.
Set `PROSPER_CAPTURE_REVISION` explicitly in WSL worktrees: WSL Git cannot resolve their Windows-path
gitdir links, so the build-time fallback revision is `unknown` there.

**Interactive frame grab (prosper-app hotkey).** When you are *playing* a title in `prosper-app` and see a
graphical bug or an FPS drop, press **F9** to capture that moment for offline debugging — no need to
predict a submit index with `PROSPER_GPU_CAPTURE_AT`.

> **Agents: you do not need the hotkey.** Both F9 and F8 are schedulable, so a headless run can take
> them itself — see the trigger table in `CLAUDE.md` (`PROSPER_GRAB_BUNDLE_AFTER_MS` /
> `PROSPER_GRAB_BUNDLE_AT_FRAME`, `PROSPER_PERF_CAPTURE_AFTER_MS` / `PROSPER_PERF_CAPTURE_AT_FRAME`).
> Build with `-DPROSPER_APP=ON` (off by default, which is why these were long assumed human-only). On
> **Linux** add `SDL_VIDEODRIVER=offscreen` for no window at all; on **Windows** leave it off — SDL's
> offscreen driver needs `VK_EXT_headless_surface`, which the NVIDIA Windows driver does not expose, and
> the app terminates before writing any artifact. The triggers work with an ordinary window on either.
> The scheduled path calls exactly the same code as the keypress, reservation accounting included.

F9 arms a one-shot capture of the next frame
(default 1, `PROSPER_CAPTURE_FRAMES=1..240` to grab an animation over several frames) and writes two
files (to `PROSPER_CAPTURE_DIR`, default cwd):

```
frame_grab_<titleId>_<YYYYMMDD>-<HHMMSS>-<mmm>[-<N>].prgbundle   a replayable capture
frame_grab_<titleId>_<YYYYMMDD>-<HHMMSS>-<mmm>[-<N>].bmp         a convenience screenshot
```

**Both files of one grab share one stem**, and that stem is claimed — with an exclusive create, so it
cannot be taken twice — at the moment F9 is pressed. Read that as a guarantee you can rely on when
triaging: two files with the same stem ARE the same capture, and two files with different stems are
never the same capture, whatever their timestamps say. The optional `-2`, `-3`, … appears only when
the name was already in use, and it applies to the whole capture rather than to one file. A grab that
is interrupted leaves its unwritten artifact as a **zero-byte** file: that is a capture that never
finished, and `gpu_replay` says so in those words rather than reporting a corrupt bundle. (When the
app is still running it does better than a placeholder — a superseded or failed capture has its
reserved file removed and accounted for in the log — so a *missing* sibling is not a corrupt capture
either. The log distinguishes them.)
(Before this scheme the name was a per-process counter, `frame_grab_001`, which let a second title
played in the same directory overwrite the first title's captures, and let an aborted grab's `.bmp`
sit beside a same-named `.prgbundle` from an earlier boot — a mismatched pair that reads as one frame
in two states.)

The console tells you the real paths, and only once they exist. The arming line names the title, not a
file — at arm time the eventual name is not yet a fact — and each artifact is reported by its own line
after it is written:

```
[grab] F9 #1: arming a whole-frame capture for PPSA25009 (Blue Prince)
[grab] screenshot written (armed at guest present 41207, written at guest present 41209) -> ./frame_grab_PPSA25009_20260801-142233-471.bmp
[grab] bundle written -> ./frame_grab_PPSA25009_20260801-142233-471.prgbundle
```

**To collect artifacts from a log mechanically, match the two prefixes `[grab] screenshot written` and
`[grab] bundle written`, and take everything after the first ` -> `.** Not every `[grab]` line is an
artifact report: the layer underneath also prints progress, arming and abort lines (`[grab]
frame-bundle: capturing 1 frames; target path …`, `[grab] frame-bundle written (312 submits) -> …`),
and the env-driven capture flows print their own. Those name a *target* or a *stage*, and a target is
not a file that exists. The two prefixes above are emitted only after the file is on disk.

It is purely on-demand — nothing heavy runs until you
press, so the grab never distorts the very slowdown you are observing (you will see a brief hitch on the
press). It captures the frame *right after* the press, so it is faithful for persistent glitches and
slowdowns (a single-frame transient could slip by a frame). Replay/debug with
`gpu_replay --bundle <that path> out.bmp` — it reproduces the real image even for a deferred
renderer: the capture **seeds** the renderer-owned RTTs the frame *samples* (deferred G-buffer /
temporal-AA history) with their live pixels (#1291), so a single submit no longer replays black. Drill
into one submit with `--bundle-extract-submit K sub.prgcap`, then `--draw-steps` / `--inspect-only` /
`--dump-resource`.
For a scripted route with a stable guest phase line, set
`PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG='exact complete line'` together with the existing
`PROSPER_CAPTURE_BUNDLE=/path/frame.prgbundle` and optional `PROSPER_CAPTURE_BUNDLE_MAX_MB=N` settings.
The matcher is one-shot, requires exact line equality (CR, LF, and CRLF are equivalent terminators), and
arms the same seeded whole-frame capture after skipping exactly one completed present. It is safe across
fragmented stdout `printf`/`puts`/`putchar`/`fputs`/`fwrite`/fd-write output and deliberately rejects
overlong lines; non-stdout streams and descriptors are ignored. The one match diagnostic names every
adapter that contributed to the line. Do not use a substring or a program address as a substitute for a
real phase marker. Persistent color/depth targets must remain enabled so the bundle can seed the frame
boundary it is intended to preserve.
**Equality means the WHOLE line, and a marker that is merely a prefix of it never fires.** This cost a
full ~7-minute routed Astro Bot world-map capture (#1684): the guest printed
`LevelDocument Loaded: worldmap [worldmap]`, the marker was `LevelDocument Loaded: worldmap`, and the run
reached the checkpoint, exited 0, and wrote nothing at all. Since #1684 an armed gate that never fired
says so **at process exit**, on stderr, naming the marker, how many guest-stdout lines were compared, and
the observed line sharing the longest prefix with the marker — escaped, so an invisible trailing byte is
visible. Copy that line verbatim as the next run's marker. Two distinct outcomes to read carefully:
`no completed line reached the observer` means the route never produced guest stdout (a wrong route, not
a wrong marker), whereas a quoted closest line means the marker itself is wrong. The same exit report
covers the other two automatic gates: an unreached `PROSPER_CAPTURE_BUNDLE_AT_PRESENT` states the present
count the run actually reached, and an unobserved `PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE` states how many
presents looked for it. **Absence of a bundle is no longer evidence of anything on its own — read the
exit report.**
Since #2565 the same exit report also covers **`PROSPER_CAPTURE_BUNDLE` set on its own**, which is a
complete no-op: the variable only names a DESTINATION, and nothing but one of the three gates above
(or F9, which uses `PROSPER_CAPTURE_DIR` instead) ever arms a capture into it.

**Malformed capture tunables no longer substitute a policy in silence** (#2565). The parse is strict —
the whole value must be a number, with nothing around it, not even a space — and every rejection now
names the variable, quotes the value with invisible bytes escaped, and states what is actually in
force:

| variable | a bad value used to mean | now |
| --- | --- | --- |
| `PROSPER_CAPTURE_MAX_SUBMITS` | **uncapped** — the inverse of the request | **the process refuses to start**, exit status 3. There is no honest default: `0` means uncapped, so a typo removed the only content bound on exactly the run that needed it. Express "uncapped" by leaving the variable unset. |
| `PROSPER_CAPTURE_BUNDLE_MAX_MB` | the default budget, so a raise was discarded | same value in force, plus one `[grab]` line saying the raise was discarded (or naming the 64/3072 MiB bound it was clamped to) |
| `PROSPER_CAPTURE_FRAMES` | 1 — the very width the `window had no submits` message tells you to widen | same value in force, plus one `[grab]` line; a mistyped remedy no longer reproduces the original failure in silence |

If present counts vary and the title emits no honest guest-stdout marker, use the headless F9 control:
set `PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE=/path/capture.ready` together with
`PROSPER_CAPTURE_BUNDLE=/path/frame.prgbundle`, keep the trigger absent at process start, and create it
only after a lightweight screenshot proves the target phase. The trigger remains as a durable lever
witness and fires once. Automatic trigger-file, fixed-present, and exact-guest-log gates are mutually
exclusive; configuring more than one disables every configured gate so one arm's completion cannot be
mistaken for another. Require the trigger-observed log, `[grab] frame-bundle written`, and a nonzero
bundle; a trigger file or screenshot by itself is not a completed capture.
`PROSPER_SUBMITLOG_DIM=WxH` prints the exact renderer invocation for any submit targeting that Gen5
surface extent, even while `PROSPER_RENDER_FIRST` skips Vulkan work. Use it to start a later render
window on a one-time offscreen producer rather than after the producer has already been lost.
`PROSPER_RENDER_TARGET_DIM=WxH` executes matching target submits even before `PROSPER_RENDER_FIRST`.
This preserves a one-time producer in the RTT cache while still skipping an expensive gap before its
late consumer.
`PROSPER_RENDER_RESOURCE_DIM=WxH` likewise executes submits that sample a matching image, allowing a
producer/consumer A/B without rendering every unrelated submit between them.
`PROSPER_RESOURCE_HASH_DIM=WxH` logs each matching sampled resource's raw guest hash, decoded/sample
hash, nonblack RGB/alpha occupancy, RTT-hit state, last compute/DMA writer, draw index, and PM4 order.
Add `PROSPER_DUMP_RESOURCE_VERSION=1` to write each distinct decoded version as a BMP under
`PROSPER_FRAME_DIR`; this is an inspection artifact, not a pixel oracle. `PROSPER_TARGET_STEP_HASH_DIM`
rerenders matching target passes by prefix and logs per-draw hashes plus dark/white/mean metrics;
`PROSPER_TARGET_STEP_HASH_MIN_DRAWS=N` bounds that intentionally expensive bisect.
`PROSPER_RENDER_DELAY_MS=N` skips synchronous Vulkan work for N milliseconds from the first submit while
the guest and command decoder advance. The `screenshot` frontend exposes this as `--warmup-seconds`, plus
`--warmup-submits` for `PROSPER_RENDER_FIRST`; use wall-clock warmup for progression captures and the exact
submit gate for repeatable renderer investigations.
`PROSPER_GPU_TIMELINE=<path>.prgtl` records every folded submit before renderer sampling plus every
VideoOut flip. `gpu_timeline <path> [--records]` inspects the checksummed index offline. Version 6 also records
compact target-extent spans; use `--signatures DRAWS DISPATCHES` to discover scene shapes and
`--select WxH DRAW_INDEX DRAWS DISPATCHES` to validate the live predicate. Recording is
independent of `PROSPER_RENDER_EVERY`. To materialize one exact indexed submit without rendering the
warmup, set `PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=N` and
`PROSPER_GPU_TIMELINE_CAPTURE=<path>.prgcap` on a second run. Version-2 detail records link the capsule;
version-8 capsules deduplicate content-addressed shader/resource versions, retain complete raw depth-surface
programming, and preserve mixed draw/dispatch order plus explicit unrealized operations. Failed operations
also retain bounded raw stages and exact rejection/state summaries; inspect them with `gpu_replay --inspect-only`
and extract one with `--dump-failed-shader FAILURE:STAGE PATH`. Selection is
intentionally bounded to the consumer and an optional
immediate predecessor. Explicit-depth `.prgbundle` capture provides bounded recursive cross-submit closure;
automatic present-to-producer selection remains #595.
After a translator change, use `gpu_replay --retry-failed-stage FAILURE:STAGE` for one retained shader or
`--retry-failed-chain FAILURE` for a captured split vertex prolog/main pair. These offline paths reuse the exact
captured resource table and report whether the current production recompiler now produces SPIR-V, without a
guest boot or Vulkan replay.
Timeline version 9 retains the version-5 sliding graphics-target window plus aggregate lifetime metadata
when a detailed capture is requested. `PROSPER_GPU_TIMELINE_HISTORY=N` raises the window to at most 65536.
Producer records identify the latest overlapping prior submit/draw/PM4 order, earliest observed graphics
writer, write/submit counts, truncation, and raw first-writer clear/target state. Raw clear registers are
provenance, not proof of an implicit hardware clear. Version 9 also records an explicit producer-history lower
bound and classifies each external image as producer-history, exact-RTT-seeded, generic unknown, or
`phase-history-bounded/unknown`; the last state must never be treated as a proven initialization or replaced
with synthetic zeroes. The timeline intentionally retains no delayed pointers to mutable guest bytes.
Every current submit also records the v5 distinct DS plane/HTILE identities, raw view/format/size programming,
target extents, and test/write/clear counts. `gpu_timeline FILE --depth-summary [WxH]` groups their full lifetimes.
For a focused run, `PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=WxH` adds guest depth/stencil/HTILE hashes and latest
overlapping writer provenance without realizing general resources; use it before attempting a full bundle.
Set `PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR=<path>.prgcap` to snapshot exact submit `N-1` at producer
time alongside selected submit `N`. Replay the pair with `gpu_replay --prepend producer consumer output`.
This is a one-level probe; graph the producer and recurse when it also reads a temporal version.
For bounded recursion, add `PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=<path>.prgbundle`,
`PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=2..4096`, and optionally
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=64..4096` (default 1024). `gpu_replay --bundle bundle output`
executes the ordered window and classifies each temporal frontier as included, seeded, or depth-bounded.
Bundles use content-defined chunks so shifted capture metadata does not defeat cross-submit deduplication.
`PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY=N` atomically installs the rolling bundle every N captured
predecessor submits, preserving the latest complete window if the guest crashes before the selected endpoint.
`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DIM=WxH` restricts predecessor captures to draws targeting that extent;
it is a size diagnostic, not a dependency proof. `gpu_replay --bundle-zero-boundary` supplies transparent
pixels to the oldest unseeded temporal leaves for an explicit A/B test and labels the synthetic seeds.
`PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM=WxH` delays bundle capture until the first matching target
writer while timeline/lifetime recording continues, avoiding progression distortion from irrelevant submits.
`gpu_replay --bundle-tail N` skips older manifests without changing dictionary content; use it only when
lifetime evidence proves the suffix contains the target's beginning, then inspect its lower-bound frontier.
`gpu_replay --bundle-intermediate-through-target WxH` omits later passes from non-final submits only when
dependency evidence proves that the named target family is the sole temporal image frontier.
`gpu_replay --bundle-final-capsule PATH` exports the complete live color RTT cache plus exact valid planes from
the persistent Vulkan depth/stencil cache into a capture-v9 capsule, including base-level depth for 3D image
resources. Verify its standalone output hash against
the bundle before using it for rapid final-submit isolation. `--inspect-only` prints each DS seed's full cache
identity, format, independent validity flags, byte counts, and hashes. Captures v1-v8 remain readable without
invented DS state.
`gpu_replay --bundle-extract-submit N PATH` materializes one exact manifest for normal inspect/graph/validate
work without replaying the bundle.
`gpu_replay --warmup-repeats N CAPTURE OUTPUT` executes the same capsule N times into the persistent RTT/DS
caches before the measured replay, providing an explicit temporal-history convergence probe.
`gpu_replay --bundle-find-ds ADDR` scans compact manifests for guest depth/stencil use, writes, clears, compare
ops, and target extents without reconstructing resource payloads or invoking Vulkan.
`gpu_replay --bundle-ds-summary` groups every DS-active draw by complete captured identity/programming and
reports lifetime transitions manifest-only. Use `--legacy-htile-before-stencil` only for the preserved pre-v6
Dead Cells bundle whose allocation relationship is independently proven; current captures store real HTILE.
`gpu_replay --through-operation N` preserves the inclusive mixed graphics/compute prefix and is the preferred
final-composition bisect after a seeded capsule has matched the full bundle hash.
`gpu_replay --bundle-compact PATH` removes dictionary resources/chunks unreachable from retained rolling
manifests and exits without Vulkan when no image output path is supplied.
Set `PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1` for long unattended runs; it exits only after the selected
capsule and requested bundle are installed, never on capture failure or budget exhaustion.
For timing-sensitive routes, `PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=WxH` selects the first matching
submit at or after `CAPTURE_SUBMIT`; `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=N` bound semantic complexity to reject loading or cinematic passes.
`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=MIN:MAX` narrows where the selected target may occur in the
raw semantic draw sequence. `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=N` add dispatch-count bounds. Derive the full conjunction from
repeated positives and nearby negative samples with the offline v6 selector.
`PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM=ADDR` and
`PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM=ADDR` further require one semantic draw to bind the
configured ES/vertex and pixel programs. A target extent/index selector must match that same draw. Zero or
unset stages are disabled. The program scan happens only after the cheaper timeline predicates and does not
realize shaders or read resources before the endpoint is claimed.
Combined graphics and compute selectors require both operations in one submit; treat this only as a phase
conjunction, not evidence of a producer/consumer relationship.
For a strict cross-submit phase gate, use
`PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM=ADDR`. The exact program arms the runtime request even
before the endpoint lower bound, its own submit is never eligible, and normal endpoint predicates are allowed
only on a later submit at or after the bound. Zero/unset disables it. Before the gate, the runtime writes only
the compact timeline and scans retained dispatch register snapshots: producer history, bundle-start matching,
bundle capture, resource reads, and Vulkan work remain disabled. The arming submit resets and begins the
explicitly bounded producer/bundle window, so it is retained as a possible producer while the endpoint remains
strictly later. The arm log includes observation/skip counters; detailed capsules record the history lower
bound in their capture environment. Every constituent is checked before append/checkpoint; a requested bundle
is latched failed when any constituent has a read-before-write image leaf with only
`phase-history-bounded/unknown` provenance. The standalone endpoint capsule remains available for diagnosis.
Exact live RTT seeds are accepted only when address, extent, and sampled format semantics agree. The phase
latch is not dependency evidence.
When the endpoint moves, predecessor manifests roll forward so the final bundle retains the latest requested
depth; dictionary bytes observed by evicted manifests still count against the unique-byte budget.
Per-target RTT is the normal renderer path. `PROSPER_RTT_SINGLE_TARGET=1` restores the obsolete flattened
single-framebuffer compositor only for diagnostic comparison; it cannot represent real post chains or
offscreen target dimensions.

Use `gpu_replay --inspect-only` to print fixed-function state, native color-target dimensions, resource
hashes, explicit clear intent, guest depth/stencil surface identities, and raw stencil-op provenance.
Use `gpu_replay --graph <capture.prgcap>` to resolve captured resource reads to the latest overlapping
earlier draw/dispatch writer in mixed PM4 order. External versions are deduplicated by logical range and
list every consumer; `future-writer=N` identifies a temporal read-before-write whose prior-submit version
is required. `--graph-json <path> <capture.prgcap>` emits the same closure frontier as structured JSON.
The graph is selected-submit scope: external leaves still require cross-submit producer capture (#595).
Version-4+ capsules also print and restore temporal RTT seeds: exact host-rendered surfaces sampled by the
captured submit whose producer ran in an earlier submit. This keeps replay from silently substituting stale
guest-memory bytes for renderer-owned history; older capsules remain readable and report zero seeds.
The draw header also reports raster state as `raster=cull/front-face/polygon-mode`, using Vulkan enum
values, and the resolved depth bias as `bias=enable/constant/slope/clamp` (all zero for pre-v29
captures, which did not retain bias state). `PROSPER_NO_CULL=1` disables culling; `PROSPER_FLIP_FRONT_FACE=1` preserves the cull mode and
toggles only the resolved winding convention. The latter is useful for isolating `PA_SU_SC_MODE_CNTL`
translation without the overdraw introduced by disabling culling entirely.
`PROSPER_EXECLOG=1` includes command order, target extent, depth/stencil identity, compare/write state, and clear
intent on recompile failures. `PROSPER_DS_CLEARLOG=1` logs only nonzero fold-time `DB_RENDER_CONTROL` clear-enable
writes, which is suitable for detecting transient clear pulses without `PROSPER_GFXLOG` packet volume.
Use `gpu_replay --validate` to reflect every draw's statically used VS/PS descriptor interface and validate it
against the captured runtime resource tables without initializing Vulkan. It exits nonzero for malformed SPIR-V,
stage/set mismatches, missing or duplicate bindings, wrong descriptor classes, or statically provable undersized
buffers. The same gate is available live as `PROSPER_DESCRIPTOR_VALIDATE=warn|strict|poison|all`; strict rejects
invalid draws, poison substitutes conspicuous resources, and all also prints valid manifests.
`--draw N:M` replays an
inclusive contiguous draw range, which is useful for rendering one pass without its downstream
composite/scanout draws; a single `--draw N` remains supported.
`--dump-resource DRAW:vs|ps:BINDING PATH` writes one captured resource's exact backing bytes for
external numeric/image inspection without dereferencing the original guest address.
`--dump-shader DRAW:vs|fs PATH` writes the captured SPIR-V module for validation/disassembly.
Capture v19+ also supports `--dump-realized-shader DRAW:vs|fs PATH`, which writes the exact bounded raw
RDNA2 source for a successfully realized graphics stage; use that output with `shader_inspect`.
`--dump-compute N PATH` writes one realized compute SPIR-V module, and
`--dump-compute-resource N:BINDING PATH` writes its exact pre-dispatch storage-buffer bytes.
`--dump-post-compute-resource N:BINDING PATH` is the execution-side counterpart for storage images: it
runs the retained mixed-operation prefix through that exact realized dispatch, waits for synchronous Vulkan
writeback, and writes descriptor-linear texels. Its log distinguishes the captured seed from the selected
dispatch's immediate-before and synchronous-after raw/linear hashes, and reports R11G11B10F numerical
populations. Prefer `--require-post-change` (which requires descriptor-visible linear content to move), and add
`--expect-post-hash HASH` when an independent raw/backing oracle exists; an ambiguous selector, any failed
prefix dispatch, a selected dispatch that does not execute exactly once, or a failed gate exits nonzero. This
mode executes Vulkan, unlike `--inspect-only` and the captured pre-dispatch dump.
`--compute-only N` executes one realized dispatch in isolation; combine it with
`--override-compute-spv N PATH` to minimize or hardware-A/B a captured shader without changing its exact
resource descriptors. The override disables the pixel oracle. Validated tiled 1D/2D compute storage executes
by default; `PROSPER_DISABLE_COMPUTE_TILED_2D_STORAGE=1` restores the old skip for a diagnostic A/B.
`PROSPER_COMPUTELOG=1` separates pipeline creation, submission, dispatch-wait, import, and writeback failures.
Renderer-owned RTT imports include their hash and nonzero-byte count, and buffer writeback logs include the
first eight dwords so bounds/offset constant buffers can be identified without another code change.

For compute producer provenance, `PROSPER_COMPUTELOG=1` records each `DispatchDirect` packet's
threadgroup counts, compute-program address/hash, and AGC-resolved resources from the register state at
that exact packet. Add `PROSPER_COMPUTELOG_DIM=WxH` to emit only dispatches referencing an image with
those dimensions (for example `1024x32` for Messenger's grading LUT). `PROSPER_COMPUTELOG=all` also
prints a per-submit no-match line while a dimension filter is active. Supported compute executes through
Vulkan in retained PM4 order. `PROSPER_COMPUTELOG_CODE=0x...` and `PROSPER_COMPUTELOG_SIZE=N` restrict
writeback before/after hashes to a matching program and/or storage-buffer size during long live runs.

**`[compute-decline]` — why the live backend refused a dispatch.** Always on, no variable needed.
`execute_item` has sixteen refusal paths (fifteen, plus the selector below); each names itself:

```
[compute-decline] program=0x413cea300 reason=no-bindable-descriptor count=128 submit=3958 dispatch=16 order=1041 groups=1x1x1
```

**Read the `count`.** The line is limited to the first 8 per `(program, reason)` and every 64th
after, and the running count is printed precisely so the limiter cannot be mistaken for the rate —
`[compute] skip unsupported program` prints once per program and has twice been quoted as a census.
A refusal also clears `producer_epoch_ok`, which the next `ParserStall` latches into
`indirect_dependencies_ok` for the rest of the submit, so one decline early in a submit drops every
later **indirect** draw and dispatch untried. A single named decline can therefore account for a
frame's worth of missing geometry.

**`PROSPER_COMPUTE_PARENTSCAN=0xADDR` — CPU-side cyclicity census of a compute program's link arrays.**
For a kernel that walks a per-item successor array (GTA V's `0x413dc6700` is the motivating case), this
classifies every entry of each **stride-4** bound buffer as terminating or cyclic, **before** the
dispatch reads it and again **after** it writes, and prints `records / terminating / cyclic /
cycle-nodes / longest` plus a few ring members. `terminating + cyclic == records` always; the link
encoding it assumes is printed on every line so a wrong guess is visible rather than silent.

It exists because the obvious instrument does not work: arming `PROSPER_GPU_CAPTURE` to obtain the array
**changes which dispatches misbehave** (measured on this title: 11 runaway dispatches of both parities
without a capture, reproducibly, versus 5 odd-only with one armed). This walks bytes the front half has
already materialised — no submit, no GPU state, no reordering — so it can be read against
`PROSPER_CFG_TRIP_BOUND`'s witness in the same run. That pairing is what established the causal
direction (22/22: every runaway read an already-cyclic array).

**The post-dispatch half and the dump additionally require the compute trace** — they live inside the
writeback loop, so `PROSPER_COMPUTELOG_CODE=0xADDR` (or `PROSPER_COMPUTELOG`) must be set as well.
Arming only `PROSPER_COMPUTE_PARENTSCAN` gives the pre-dispatch census and an empty dump directory,
which looks like "nothing corrupted" rather than "the second half never ran".

`PROSPER_COMPUTE_PARENTSCAN_DUMP=DIR` additionally writes the input and output arrays of a dispatch that
turns a **clean** array cyclic — the transition only, with the retained input required to belong to that
same dispatch, so the pair in one filename is genuinely one dispatch. Two caveats: the switch takes a
**program address**, not `=1` (it says so if given something that cannot be one), and coverage varies: `PROSPER_COMPUTE_BUFFER_CACHE_MB=0` raised it
from 3 submits to 9 on a routed run, and **why is not established** — the scan runs before
`acquire_cached_buffer`, so the obvious "a cache hit skips the hook" explanation is wrong.

**`PROSPER_DRAW_LINKSCAN=0xADDR[,0xADDR...]` — CPU-side census of the LINKED LISTS a graphics draw's
scalar buffers contain.** The graphics counterpart of `PROSPER_COMPUTE_PARENTSCAN`, and it answers the
question that instrument answers on compute: a shader that walks a guest list until it sees a terminator
can only run away because the list it was given is cyclic, or because the list it was given is not the
list the guest built. Both are properties of BYTES. It censuses the exact bytes prosper is about to
upload — not guest memory — because the question is what the SHADER sees, and prints, per buffer:

```
[linkscan] ps=0x5008f1400 draw#71 scan=1 set=1 binding=45 cls=0 addr=0x551be0000 declared=33177600
           uploaded_dw=8294400 | hist zero=8294400 term=0 other=0 first_other=[4294967295]=0x00000000
           other_range=0x00000000..0x00000000 | self-walk stride=2 next=+1 term=0xffffffff
           records=4147200 starts=4147200 terminating=0 cyclic=4147200 cycle-nodes=1 oob-starts=0
           longest=0
```

**The one modelling fact to know before reading a result: an out-of-range scalar buffer load returns
architectural ZERO, and zero is a LINK, not an exit.** So an all-zero pool is not an empty list, it is
an infinite one — a self-loop at record 0 that no trip bound can end. `terminating`/`cyclic` reflect
that; `oob-starts` counts the walks that left the buffer at least once, which is the column that
separates "the guest's list is cyclic" from "prosper's descriptor is short".

`declared` beside `uploaded_dw*4` is the truncation check (#1427's short-upload defect reads as zeros
past the end and a zero link is not an exit). The `hist` columns separate the two ways a tile can say
"no lights": a terminator-filled table (correct) and a zero-filled one (a producer that never ran).
With `PROSPER_WRITER_PROVENANCE=1` each scan also names the **last recorded writer** of the range and
prints `guest_write_recorder_summary()` beside it, so a "no writer" verdict cannot be read without also
seeing which recorders fired (`writer_provenance.hpp`'s scope list — guest CPU writes are invisible to
all of them).

Knobs, all strict-parsed, and a malformed one disarms the whole scan rather than silently censusing
under a different encoding: `PROSPER_DRAW_LINKSCAN_STRIDE` (dwords per record, default 2),
`_NEXT` (successor dword within the record, default 1), `_TERM` (default `0xffffffff`), `_MAX` (scans
per binding, default 2), and the optional cross-buffer walk `_HEADS=<binding>` + `_RECORDS=<binding>`,
which starts from every dword of the head table and walks the record pool. Naming one half of that pair
without the other is rejected.

The scan cap is keyed on `(program, set, binding)` and deliberately **not** on the buffer's address: a
title that reallocates its scratch every frame would buy a fresh budget every frame and the instrument
would never stop. The address is on every line.

**`PROSPER_COMPUTE_SKIP_PROGRAM=0xADDR[,0xADDR...]` — decline named compute programs.** A bisection
instrument: one dispatch can take down everything after it (a GPU hang costs the process its compute
backend), so "which dispatch is responsible?" is worth asking directly instead of by rebuilding. It
announces itself at parse time and reports each skip through the census above as
`reason=skipped-by-selector`, so a log from a run made with it set cannot later be mistaken for a
default run. Two things to know before reading a result:

- **A skip is a decline**, so it poisons the submit's indirect latch exactly as a real refusal would.
  The output is *not* "the frame minus that dispatch"; it is that frame minus the dispatch minus
  every indirect operation after the next parser stall.
- **A program taking the CPU fast path never reaches the selector**, so naming one has no effect and
  produces no line.

It is ordered **after** `trace_compute_item` and `maybe_dump_traced_compute_spirv`, which makes
*dump the module, skip the dispatch* work — the only way to inspect what the recompiler produced for
a program that hangs the GPU without losing the device. Combined with `PROSPER_COMPUTELOG_CODE` /
`PROSPER_COMPUTELOG_SPIRV`, a recompiler change can be A/B'd by module hash at zero device cost.

**`PROSPER_SKIP_DRAW_PROGRAM=0xADDR[,0xADDR...]` — decline GRAPHICS draws by shader PROGRAM
identity.** The counterpart of the compute selector above, and it exists for the same reason: one
draw can hang the GPU, RADV then cancels the whole context, and every later submit fails with
`VK_ERROR_DEVICE_LOST` naming a *victim*. Astro Bot's world-map reset was misattributed to compute
for exactly that reason — the `RADV_DEBUG=hang` dump puts the last command-processor trace point
immediately before a `DRAW_INDEX_2`.

An address is matched against a draw's vertex/ES program, its NGG main continuation, and its pixel
program, and the matched stage is reported. Parsing is the strict `0x`-only list parser, so a bare
decimal or a stray comma arms **nothing** and says so rather than producing a confident negative
about a program nobody selected. Armed, it announces itself once and prints each decline as
`[draw-decline] program=0x… stage=vs|vs-chain|ps reason=skipped-by-selector count=N …` — first eight
per (program, stage), then powers of two, with the ordinal on every line.

**Do not read a skipped run as "the frame minus that draw".** Four limits, none visible in the
output:

- **Later passes sample the hole.** The renderer caches a pass's pixels under `CB_COLOR0_BASE` and
  injects them when a later draw samples that address. Decline the draw that fills a target and the
  composite that reads it does not fail — it succeeds against stale or empty pixels.
- **A pass whose every draw is declined renders nothing, not even its clear.** The backend returns
  an empty image for an empty draw list, and a group's clear colour comes from its first item
  whether or not that item was declined. The present layer then serves the previously retained
  frame, which on screen looks like the frame rendered.
- **Only draws that reach the live renderer can be declined** — not one the recompiler rejected or
  whose descriptor contract failed. Naming such a program produces no line, and silence is not proof
  the selector matched.
- **An address names a PROGRAM, not a draw.** Every draw using it goes, routinely thousands a frame.

It is ordered **last** in the per-draw build — after resource resolution and after the
`[render]`/`[rtt]`/`[draw-program]` per-draw lines (instrument trap 166). A declined draw is
therefore still fully realized, validated and logged; only the Vulkan draw call is withheld.

**`PROSPER_DRAW_PROGRAM_CENSUS=1` is where its input comes from.** One `[draw-program] NEW …` line
per distinct (vs, vs-chain, ps) triple, then that triple at powers of two — bounded by the program
count rather than the draw count, so it is usable on a 4K title where `PROSPER_RTTLOG` is a
firehose. It is not a teardown report on purpose: the run it exists for ends in a device loss.

The older **`PROSPER_SKIP_DRAW="N[,N...]"`** drops draws by the submit-local semantic `draw_index`.
That ordinal is not stable across frames, so it cannot name "every draw that runs this shader" —
use it to test a specific suspected draw within one captured submit, and the program selector above
for anything that has to hold across frames.

**`PROSPER_SHADER_DUMP_SUCCESS=DIR` — dump every successfully recompiled shader, and
`PROSPER_SHADER_DUMP_PROGRAM=0xADDR[,0xADDR...]` — narrow that to named guest programs (#3196).**
Each dumped shader writes its raw guest RDNA2 (`.bin`, the input `shader_inspect` decodes) and the
SPIR-V prosper emitted for it (`.spv`), and a chained vertex program additionally writes its NGG main
continuation as a `_main_` `.bin`. **Filenames carry the guest code address**:

```text
success_vs_at_00000005008efd00_35956264829da0c6_62ad8faab4566b7a.bin
success_vs_at_00000005008efd00_35956264829da0c6_62ad8faab4566b7a.spv
```

so recovering a program you identified by address — which is how `PROSPER_SKIP_DRAW_PROGRAM`,
`PROSPER_COMPUTE_SKIP_PROGRAM`, `PROSPER_DRAW_PROGRAM_CENSUS` and the `[buf-op]` /
`[mubuf-unresolved]` lines all name programs — is one glob:
`ls DIR/success_*_at_00000005008efd00_*`. The two hashes are the SPIR-V and the raw RDNA2; they
remain because they are what deduplicates, and one program compiled against different resource
tables legitimately yields several variants under the same address. An address of
`0000000000000000` means no address was available at that emit site, not an unusual address.

The filter is default-OFF, parsed by the strict `0x`-only parser in `gpu/diagnostics/watch_list.hpp`
(so a bare decimal arms nothing rather than arming 5,008), and announces itself once:

```text
[shader-dump] PROSPER_SHADER_DUMP_PROGRAM=0x5008efd00 -> armed on 1 program address(es); ONLY shaders at those addresses are dumped
[shader-dump] vs addr=0x5008efd00 chain=0x0 spv=... raw=... main=... words=…+…/… result=written
```

Four things to know before reading a result:

- **A malformed spec fails OPEN, not closed** — it arms nothing, says so loudly, and dumps
  everything. This is deliberately the opposite of the skip selectors: withholding every module
  would leave an empty directory, and an empty directory reads as *"that program never compiled"*.
- **Naming either half of a vertex chain selects the pair**, because a census line gives you the ES
  address and a `[buf-op]` line may give you the main.
- **Only programs that recompile SUCCESSFULLY reach here.** A rejected program is written by
  `PROSPER_SHADER_DUMP` instead, so naming its address produces nothing — check the arming line
  before concluding the address was wrong.
- **`withheld … opportunities=N` counts dump opportunities, not distinct programs.** A cache hit on
  a withheld program counts again; read it as volume avoided, never as a program count.
`PROSPER_PROVENANCE_DIM=WxH` retains every decoded `CB_COLOR0_BASE` write across submits, then reports
the last matching writer whenever a draw samples an image of that size. Descriptor resolution can be
limited to likely target submits with `PROSPER_PROVENANCE_MIN_DRAWS=N`; color-target history is still
recorded for smaller earlier submits. This distinguishes a sampled GPU image produced by a draw from
one populated by compute/copy/CPU work without requiring the live Vulkan renderer.

For a differential replay, `PROSPER_STENCIL_CLEAR=<0..255>` overrides the initial stencil attachment
value and `PROSPER_STENCIL_REPLACE=<0..255>` overrides the replacement reference of an
ALWAYS+REPLACE stencil-prime draw. These are diagnostic controls only; they do not change guest-state
extraction or the default render path.

`PROSPER_TILECENSUS=1` answers "which surface geometry is the detiling/tiling cost?" without a
profiler. It counts every `tile_surface`/`detile_surface` call keyed by `(op, caller tag, width,
height, bytes-per-element, tile mode)` and dumps the heaviest rows by bytes moved, periodically
rather than at exit — every frontend here leaves via `_exit`, so an `atexit` report never prints.
The caller tag comes from a `TileCensusScope` RAII guard, so a row says *which* call site moved the
bytes rather than only what shape they were; it is part of the key, so one geometry reached from two
sites shows as two rows instead of reporting whichever ran last. It found #3149: on Stray's splash a
single 3840x2160 bpe=8 detile dominated the whole run, because DCC-compressed sampled surfaces were
excluded from the compute image cache and re-detiled on every use.

`PROSPER_NO_DCC_IMAGE_CACHE=1` restores that pre-#3149 exclusion — DCC-compressed sampled surfaces
are kept out of the persistent compute image cache. It exists to bisect a rendering report against
that change with one variable, and to run the A/B that justified it; it is not a performance knob,
and setting it makes the affected titles slower.

## Shared-box hygiene

Several agents and the human run this repo at once, so anything that kills a process is a
cross-lane operation whether or not you meant it that way.

- **`lanekill.py`** — kill processes matching a name that belong to **your** worktree, and refuse
  the ones that do not. `pkill`'s two selectors both answer *what* a process is and neither answers
  *whose* it is: `-f` over-matches substrings and its own shell, `-x` matches accurately and still
  kills every lane's copy. On 2026-08-21 a `pkill -x screenshot` took a concurrent lane's
  measurement sweep at 36 of 64 frames (instrument trap 213). Census by default; `--yes` to signal;
  `--any-tree REASON` to override, loudly. Attribution reuses `worktree_reclaim.py`'s scanner, which
  matches by inode rather than path string — necessary because `/home` is a symlink on the host and
  a real bind mount inside the ps5ys distrobox, so the same directory has two spellings.
- **`worktree_reclaim.py`** — census and removal of stale worktrees, with the same in-use guard.

Both fail closed where `/proc` is unreadable: without it, ownership cannot be established, and
"I could not tell whose this is" must never render as "it is yours".

- **`wt_stash.py`** — park uncommitted work in a ref that is really private to one worktree, plus
  `check`, which asks whether the shared stash stack is currently holding somebody else's work.
  A worktree isolates `HEAD`, the index and the working tree; it does **not** isolate `refs/stash`,
  which is one ref in the common `.git` directory. Two lanes stashing in their own worktrees push
  onto and pop from one LIFO stack, and on 2026-09-01 each popped the other's entry (#3174,
  instrument trap 247). The park goes to `refs/worktree/prosper-stash/<slot>` instead — a namespace
  `git-worktree(1)` documents as not shared — addressed by name rather than by top-of-stack, with
  `push` refusing an occupied slot rather than stacking. Same fail-closed shape as the two above:
  `check` reports an entry it cannot attribute to a branch as `UNATTRIBUTABLE`, never as yours.

## Ruled out

- **A `git` alias cannot guard `git stash`.** `git config alias.stash '!<guard>'` is silently
  ignored — git does not let an alias shadow a builtin — so the guard never runs and the stash
  lands as usual. Measured 2026-09-02 on git 2.55.0: with the alias configured, `git stash push`
  saved an entry and the alias body printed nothing at all. A wrapper of that shape is worse than
  no wrapper, because it reads as coverage and cannot fail. #3174.
- **A `reference-transaction` hook does abort a stash cleanly — but it is not a guard anyone may
  assume.** Measured 2026-09-02: exiting non-zero in the `prepared` phase for `refs/stash` refuses
  the update, leaves the working tree untouched and creates no ref, so there is no half-stashed
  state. It ships as `wt_stash.py install-hook` and is deliberately **off by default**: hooks live
  in untracked `.git/hooks`, so every fresh clone fails open, and once installed it blocks
  `git stash` repo-wide — every worktree, the human's interactive use, and `--autostash` with it.
  So it is an opt-in convenience for one machine, never the reason the collision cannot recur.
