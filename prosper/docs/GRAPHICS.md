# Graphics & Audio bring-up (the M4/M5 frontier)

Status as of the boot reaching multithreaded graphics/audio initialization. The game now runs its
**entire non-graphics runtime** — IL2CPP init, C# startup, the runtime main flow, splash/message
dialog, the flip/render loop scaffolding — and enters GPU + audio setup. This doc is the blueprint
for turning that into actual rendered frames.

## How far the boot gets

```
loader → CRT → C++ ctors → il2cpp_init (GC, metadata, type system) → runtime startup
  → sceSystemServiceHideSplashScreen → sceMsgDialog (auto-dismissed)
  → GRAPHICS init: libSceAgc (GPU command build) + libSceAgcDriver (submission)
  → libSceVideoOut (display / flip / vsync event queues)
  → AUDIO init: libSceAmpr
  → [BLOCKED] multithreaded null-object derefs in graphics/audio worker threads
```

The block is the **headless limit**: our placeholder graphics/audio objects are zeroed, so worker
threads eventually read a null sub-object pointer out of them and dereference it (e.g.
`eboot+0x3b5ea6` `[null+0x30]`, `eboot+0x149c99c` `[null+0x18]`). Zeroed placeholders no longer
suffice — the game needs **real object graphs**, i.e. the actual graphics/audio subsystems.

### The terminal fault is unimplemented libSceAgc, NOT a C++ locale bug (verified 2026-07-04)

**This corrects two earlier mis-diagnoses in this file's history** (first "rune facet never set during
static init", then "std::ctype locale facet array left zero"). Both were wrong: the table-lookup
*instruction shape* at `eboot+0x3b5ea0` (`movzwl 0x2e(%rdi,%rcx,2)`) merely *resembles* `std::ctype`
classification. The object it reads actually comes from an **unimplemented libSceAgc initializer**.

The last log line before the crash is Unity's own `todo: void GfxDevicePS5SharedData::CreateWorkload()`
— i.e. we are inside Unity's PS5 GPU-device setup. The crash **site varies across runs** (multithreaded
graphics workers): `eboot+0x3b5ea6` (`addr=0x30`), `eboot+0x149c99c` (`addr=0x18`), … — all null-field
derefs of zeroed graphics objects, not one deterministic path.

Verified chain for the `CreateWorkload` object (gdb, `break run_entry` first; `pltm` + the unimpl log):

- In `eboot+0x14dd*`'s fn: `obj = r14+0x48` (`lea 0x48(%r14),%r12` @`+0x14dda75`), then immediately
  `call 0x4003ae7d0(obj)` — the object's initializer.
- `0x4003ae7d0` is a **PLT stub**: `jmp *[0x401d95858]`. At runtime that GOT slot holds `0x600004180`,
  which is one of **our unimplemented-import stubs** (`mov $idx,%edi; movabs $prosper_on_unimpl; jmp`).
- `pltm eboot.bin known_names.txt 1d95858` → NID **`+kSrjIVxKFE`**, and the boot log shows
  `unimplemented: libSceAgc::+kSrjIVxKFE -> returning 0`. So `obj`'s initializer is an **unimplemented
  libSceAgc function** that does nothing.
- The caller then passes that *same* `obj` (`r12`, stored at `[rbp-0x198]`) to `0x4003a7b60`, which
  walks `obj+0x38` and reads `[obj+0x40]` as a pointer → it's null (never set by the stubbed init) →
  SIGSEGV. `PROSPER_FAULTMEM` confirms the whole `obj` region is zero at the fault.
- `_Getpctype` works and is called, but only for an unrelated inline `isspace`-style loop
  (`eboot+0x82e893`, `test $0x144`) — it is NOT part of this path.

**Conclusion:** the blocker is the **libSceAgc GPU object graph**. `CreateWorkload` (and sibling
graphics init) call ~24 libSceAgc functions — `23LRUSvYu1M`, `BfBDZGbti7A`, `+kSrjIVxKFE`,
`H7uZqCoNuWk`, `vRoArM9zaIk`, … (full list via `boot_trace`) — **all unimplemented, all returning 0**,
so the GPU objects they should build stay null and the graphics workers deref null. This is squarely
the M4/M5 libSceAgc→Vulkan work below; there is no locale/libc gap to fix. Faking these objects with
plausible-looking fields would be "limping to graphics" (violates correctness-first) — they need the
real AGC object model, reverse-engineered from the call args (`PROSPER_GFXLOG`) + AGC semantics.

**Tooling:** `PROSPER_FAULTMEM=1 ./build-linux/boot_trace <dump>` dumps every GP register + 4 qwords of
guest memory at each pointer-looking one, at fault time on the stopped thread (reliable; live gdb
breakpoints race in this multithreaded, signal-scheduled guest). `pltm eboot.bin known_names.txt
<got_off>` maps a GOT slot to its import NID/name.

**AGC call tracing (RE bootstrap for M4).** All 28 libSceAgc/AgcDriver NIDs the game calls are routed
through per-NID logging thunks (`glog_thunk<I>` in `hle_graphics.cpp`; behaviour unchanged — still
returns 0). `PROSPER_GFXLOG=1 ./build-linux/boot_trace <dump>` now emits a **self-describing** line per
call: `libSceAgc::<NID>  from eboot+0x<callsite>  a0..a5`. ~148 AGC calls fire before the fault.
Call-frequency profile of one run (NID ×count — the hot ones are the AGC command/descriptor ops to
understand first; args' high addresses are heap/GPU-VA that vary per run, but NIDs + callsites are
stable):
- `f3dg2CSgRKY` ×36 — hottest; a per-op/per-command call.
- `d-6uF9sZDIU` ×25, `ZvwO9euwYzc` ×25 — next hottest, paired.
- `TRO721eVt4g` ×5 — `CreateWorkload` per-object init (`eboot+0x14e6661`, `a0=a3=obj, a4=obj-0x48`);
  `obj` is the very object whose null field later faults, so this call (or `+kSrjIVxKFE` ×3, the
  initializer at `eboot+0x3ae7d0`) is what should populate it.
- device/context level (once each): `23LRUSvYu1M`, `BfBDZGbti7A`, `XlNp7jzGiPo`, `MM4IZSEYytQ`.
Next: map each hot NID + arg pattern to the AGC API to build the real object model, then implement the
initializers to construct valid GPU objects (correctness-first — no plausible-looking fakes).

## What's already in place (headless bring-up, correctness-first)

- **Unified GPU memory (lazy)** — `exec_image_linux.cpp` fault handler backs any unmapped page in
  the GPU-VA window `[4 GiB, 64 GiB)` with a real zeroed page on demand and retries. This models the
  PS5's unified CPU/GPU memory (GPU VAs are real RAM). Low-address null derefs stay fatal. It got the
  boot past the format-table fault at GPU VA `0x100000000` into audio init. **Contents are zero until
  the driver layer is real** — a documented placeholder, not faked output.
- **libSceVideoOut** (`hle_graphics.cpp`) — `Open`→handle, `SubmitFlip` increments a flip counter,
  `GetFlipStatus` reports it (correct 0x40 struct — do NOT over-write, it smashes the guest canary),
  `IsFlipPending`→false, event/flip machinery no-op. Simulated flip completion so the render loop
  advances.
- **libSceAgc** getters that return dereferenced objects → stable zeroed singletons.
- **event queues** (`hle_kernel_time.cpp`) — valid queue objects; `WaitEqueue` yields + reports no
  events.
- **Diagnostics** — `PROSPER_GFXLOG` logs AgcDriver call args. Sampled-texture isolation accepts
  `PROSPER_TESTTEX_DRAW=N`, `PROSPER_TESTTEX_BINDING=B`, and `PROSPER_TESTTEX=zero|checker`; both RGBA8 and
  renderer-owned RGBA16F RTT inputs are replaced in their native format. `PROSPER_TESTTEX_FILTER=linear|point`
  uses the same draw/binding selectors for sampler-only A/B tests. `PROSPER_DUMP_RTGROUPS_ADDR=0x...` scopes
  `PROSPER_DUMP_RTGROUPS` to one target during a long run. `PROSPER_SHADER_DUMP_SUCCESS=DIR` captures both the
  exact raw RDNA2 bytes and translated SPIR-V for successfully recompiled graphics and compute shaders, naming each
  file by its guest code address as well as its hashes; `PROSPER_SHADER_DUMP_PROGRAM=0xADDR[,0xADDR...]` narrows that
  to named programs (#3196, contract in `tools/AGENTS.md`). These are
  diagnostics, not renderer policy; record an unmodified output before interpreting an override.

## Build environment

- `libvulkan.so.1` (1.3.275) **is present**; Vulkan **headers are not** (`apt install libvulkan-dev`
  or vendor `vulkan/`). No SDL2/X11 → plan **headless/offscreen Vulkan** first (WSL has no display).
- Graphics libs are sparsely documented; most NIDs don't resolve to names — reverse-engineer from
  call args (`PROSPER_GFXLOG`) and `build-linux/tools/pltm` (maps a module GOT offset → NID).

## ✔ USER-DATA PROBE (2026-07-04): zero-varying fault pair still has PS resource bindings

Follow-up to PR #14's fork between "legitimate zero-varying pipeline, bad resident flag" and
"`[+0xc0]` is fed by shader resource bindings." `PROSPER_PIPETRACE` now logs the decoded
`ShaderUserData` resource-binding table: direct resource offsets, `eud_size_dw`/`srt_size_dw`, and
the four sharp-resource arrays (`sharp[0]` texture, `sharp[2]` sampler, `sharp[3]` storage buffer).

One traced WSL2 boot of the faulting pair showed:

```
CreateInterpolantMapping gs=...c0620 ps=...c0dd0 -> mapping 0 interpolants
  gs user_data: direct_count=11 sharp_counts={0,0,0,0}
  ps user_data: direct_count=11 sharp_counts={0,0,0,1}
    sharp[3] storage: slot0={off=0000,size=1}
```

So the semantics conclusion still holds: this is a genuine zero-varying pair. But the pixel shader is
not resource-empty: it has one storage-buffer sharp binding. That makes hypothesis (2) the better next
thread: the pipeline reflection table `[pipeline+0xc0]` is likely populated from shader resource
bindings/user-data as well as, or instead of, interpolants for this path. The next probe should catch
the Unity builder that folds `ShaderUserData` into `[pipeline+0xc0]/[+0xe0]` and compare the expected
storage binding against the records seen by predicate `eboot+0xd58710`. Chasing `[obj+0x1a0]` first is
lower signal until the resource-binding reflection path is ruled out.

## ✔ SEMANTICS PROBE (2026-07-04): metadata is surfaced CORRECTLY — the "mis-relocation" bet is rejected

Agent-2's fastest-to-confirm bet was that our shader I/O semantics are empty/mis-relocated, making the
interpolant mapping short as a symptom. **Rejected, conclusively, via raw blob bytes** (PROSPER_PIPETRACE
now logs semantic counts/arrays + the raw `0x50..0x5f` header region):

- Shaders that HAVE varyings read correctly: `num_input_semantics=1` (`raw[0x50]=01 00 00 00`),
  `num_output_semantics=1` (`raw[0x56]=01 00`), and `output_semantics[0].semantic=0x0f` (=15) — exactly
  the value consumed by the observed VS→PS linkage. So our offsets (in u32@0x50, out u16@0x56, arrays@0x30/0x38)
  and relocation are correct.
- The **faulting pipelines** use `gs` (num_out=0, `raw[0x56]=00 00`) + `ps` (num_in=0) — genuinely
  **zero-varying shaders** (position-only/blit). All 3 `CreateInterpolantMapping` calls therefore map 0
  interpolants — a correct consequence, not a surfacing bug.

**Conclusion:** the empty pipeline-reflection table `[pipeline+0xc0]` is NOT caused by missing/mangled
semantics. Either these are legitimately 0-varying pipelines where the reader at `0xba6e08` should
tolerate a null companion (and our divergence is that the "resident" flag `[obj+0x1a0]` never gets set),
or `[+0xc0]` is populated from **resource bindings** (textures/samplers/cbuffers via the shader's
`user_data` table) rather than interpolant semantics. Next probe target: capture `[pipeline+0xc0]`/
`[+0xe0]` contents directly (the reflection records) and check the shader `user_data`/resource tables,
not the semantics. `PROSPER_PIPETRACE` retained (semantic + raw-header logging).

## ✔ SKIP PROBE (2026-07-04): the null companion is systemic, not a single gap → STOP hunting per-object writers

Env-gated diagnostic `PROSPER_SKIP_NULL_COMPANION` (exec_image_linux.cpp; default off): at the reader
`eboot+0xba6e08`, log the object's state and redirect RIP to the reader's own skip label
`eboot+0xba6e40` (where its type/flag branches already land), continuing as if the companion weren't
needed. A probe to reveal the *shape* past the fault — NOT a fix (companions are still not real).

Result of one run: **4 bounded skips** (distinct objects, all `[+0x140]=0`; resident-flag low byte
`[+0x1a0]&0xff == 0` on every one — none processed), then the boot advances **past `0xba6e08` to a
NEW null-object deref at `eboot+0x95c823`** (`mov (%r14),%rdi; mov (%rdi),%rax` with `[r14]`→null).

**Interpretation (per the agreed decision tree): systemic, not a narrow gap.** It is NOT an infinite
cascade of the same deref (bounded at the collection size = 4), and it does NOT reach a frame — it
lands on the *next* null GfxDevice object. So the pipeline-residency subsystem simply never ran: every
pipeline lacks a companion and the resident flag, and past that lies another null object of the same
"placeholder is zeroed" kind. **Chasing per-object `[+0x140]` writers is unproductive — skipping one
only reveals the next null.** The productive direction is systemic: drive Unity's deferred
processing/residency pass (the "deferred to a submit/flip/frame boundary we never pump" theme — no
`SubmitFlip` fires before the fault), which ties into the swapchain scaffolding (`prosper_vo_*`) and
the back-half pipeline realization. Next: find what triggers the residency pass and whether pumping
submit/flip fires it, rather than reconstructing per-object writers.

## 🔎 THE [+0x140] WRITER CHAIN (2026-07-04) — host-side RE, found the constructor + gate, root is a pipeline-reflection predicate

Agent-2 (correctly) reassigned this to the front-half: the companion is built pre-submit, upstream of
any GpuState. Traced the writer/constructor of the null `[obj+0x140]` and its gating condition:

**The writer** — `eboot+0x15aef7d..0x15aef83`:
```
g = *[0x1ff28c8]                       ; factory/device singleton (global)
[r13+0x140] = g->vtable[0x88](g, 0, payload)   ; create the GPU companion  <-- THE WRITER
```
The sibling just above (`+0x138`) is the same call with arg1=1 (`g->vtable[0x88](g,1,..)`) — so
`+0x138`/`+0x140` are a pair of companions (two shader stages) created from the same factory.

**The factory is fine (not the cause).** `[0x1ff28c8]` is a lazy Meyers singleton constructed locally
at `eboot+0xc33769`: `operator new(...)` (call `0x809830`) + vtable `0x1d6b8f8` stored at `(obj)`,
then `[0x1ff28c8]=obj`. **No AGC/gfx call gates it** — our heap works, so `g` is non-null at runtime.
So `+0x140` is null NOT because the factory is missing.

**The proximate gate** — the writer block is skipped unless an optional local is present:
```
0x15aef69:  mov -0x200(%rbp),%rcx ; test %rcx,%rcx ; je 0x15aef8a   (skip the +0x140 create)
```
`[rbp-0x200]` is the "present" flag of an optional{payload@-0x210, present@-0x200}, zero-initialized
and filled only if a predicate passes:
```
0x15aee82:  call 0xd58710 ; test %eax,%eax ; je 0x15aef30   (skip filling the optional)
```
(The `+0x138` companion has the identical structure gated by `[rbp-0x1e0]` via the same predicate at
`0x15aec7a`.)

**The root predicate `eboot+0xd58710`** decides whether each companion is created. It reads
`[pipeline+0xe0]` (if 0 → early-out) and iterates the record array at `[pipeline+0xc0]` (0x20-byte
records, checking byte fields at `+0x02`/`+0x22`) — i.e. it scans the pipeline's **reflection/binding
table** and returns non-zero when a matching entry exists. For our 3 resources the second call returns
0 → the `+0x140` optional stays empty → the companion is never created → `+0x140` null → the
`0xba6e08` reader faults.

**So the true root is upstream of the writer: the pipeline's reflection collection at `[pipeline+0xc0]`
is empty/short for these resources**, so `0xd58710` reports "no second-stage companion needed" and the
writer legitimately skips it — but the *reader* at `0xba6e08` derefs it anyway. That `[+0xc0]`
reflection table is populated from the shader register/semantic data that flows through
`CreatePrimState`/`CreateInterpolantMapping` (which we implement into stack scratch, then Unity folds
in). **Next probe (joint):** capture `[pipeline+0xc0]`/`[pipeline+0xe0]` for the faulting objects and
compare against what our `CreateInterpolantMapping` produced — a short/empty interpolant set would
explain the empty reflection table. This is the remaining thread to the true root cause.

## ✔ libSceVideoOut real 1080p60 + swapchain scaffolding (2026-07-04) — and the [+0x140] gate is NOT the display surface

Implemented the 5 previously-unimplemented VideoOut NIDs with real, self-consistent values (resolved
through firmware symbols, guest call sites, and PS4-inherited public contracts): `Nv8c-Kb+DUM` sceVideoOutIsOutputSupported, `PjS5uASwcV8`
sceVideoOutSetBufferAttribute2, `rKBUtgRrtbk` sceVideoOutRegisterBuffers2, `utPrVdxio-8`
sceVideoOutGetOutputStatus, `w0hLuNarQxY` sceVideoOutConfigureOutput. Plus fixed GetResolutionStatus
(was all-zero → now 1920×1080@59.94Hz) and added GetVblankStatus (advancing counter) +
GetDeviceCapabilityInfo (SDR). All output-struct writes are size-exact.

> **The 1080p@59.94 in this entry is no longer the whole answer (#3017).** It remains the *default*,
> so this record still describes what a default run advertises — but that pair was hardcoded, and a
> title that reads the advertised mode and paces to it was therefore told 59.94 Hz on any host and
> 1080p on a 4K one. Since #3017 the mode is **derived** from the host's real display under
> `PROSPER_DISPLAY_MODE` (`legacy` | `host` | `host-high-refresh`), and the vblank period that paces
> the guest comes from the *same* resolved mode rather than from a third constant kept in step by
> hand. `src/hle/graphics/display_mode.hpp` holds the policy and the evidence grading behind each
> refresh enumerant; deriving is opt-in because titles act on this value.

**Verified the game requests exactly what we advertise:** `SetBufferAttribute2` arrives with
`width=0x780 (1920)`, `height=0x438 (1080)`, `pixel_format=0x8000000000000000` (PS5 A8R8G8B8 sRGB);
`RegisterBuffers2` registers **3** framebuffers (triple-buffered). Recorded them in a display-buffer
registry (swapchain scaffolding, `prosper_vo_buffer_count/_display_width/_height/_format/_buffer_addr`)
that the back-half present path turns into swapchain images. test_videoout (27 tests total).

**Hypothesis result (agent-2 asked): the display surface is NOT what gates the `[+0x140]` companion.**
With all 5 VideoOut calls returning real 1080p60 values + buffers registered, the boot still faults at
**the same `eboot+0xba6e08`** (unchanged). So VideoOut is eliminated as the suspect — Unity allocates
the pipeline companion independent of display-surface availability. The residency interception belongs
in the back-half (GpuState/register-context stage), as concluded below.

Note: `GetOutputStatus` currently traces + returns success without writing its output struct — its
exact layout/size is unconfirmed and the game tolerates a no-write return (boot reaches the same
downstream fault). Left non-writing rather than risk a wrong-size write (stack-canary smash).

## ✔ BOUNDED TRACE ANSWER (2026-07-04): the pipeline object is Unity-INTERNAL — no HLE hook exists

Back-half asked one bounded question: does the fault object `r15` (or its `[+0x18]`/`[+0x40]`
packed-register sub-object) ever appear as an arg / return / write of any AGC call we implement
(`CreateShader`, `CreatePrimState`, `CreateInterpolantMapping`, or the register-context builders)?

**Answer: NO.** Logged every construction call's pointer args (`PROSPER_PIPETRACE`) and diffed against
the fault object in the same run (`r15=0x7d622562c300`, sub-object `0x7d62256120f0`):
- `CreateShader` (×36+): header/code blobs live in `0x7d6252dc…`; `*dst` writes into the eboot BSS
  shader registry (`0x402048…`) or a stack slot — never `r15`'s heap region.
- `CreatePrimState` / `CreateInterpolantMapping`: write register `{offset,value}` pairs into **stack
  scratch** (`a0/a1 = 0x7d6255dfb…`); shaders are `0x7d6252dc…`.
- The only `0x7d6225…`-region pointers logged are the AGC context (`…5cb378`) and register-context
  builder args — none in `r15`'s (`…62c300`) or the sub-object's (`…6120f0`) region.

So the register *content* originates from our calls (Unity harvests it into stack scratch), but the
**pipeline object and its `[+0x140]` GPU companion are constructed by Unity's own resource manager** —
there is no game→AGC call that takes/returns/writes them, hence no front-half HLE hook. Per the
back-half's decision tree, pipeline residency will be intercepted at the GpuState/register-context
stage in the back-half; the front-half is off the hook for this object. `PROSPER_PIPETRACE` (logs the
raw pointer args of the shader/pipeline construction calls) is retained for future seams.

## ⚠ RE UPDATE (2026-07-04): the 0xba6e08 object is a Unity PIPELINE object, and NO resource-creation call feeds [+0x140]

Investigating the resource-layer integration (gpu_resources.hpp contract) turned up evidence that
**contradicts the "hook one AGC resource-creation NID → store handle at [obj+0x140]" plan** — flagging
for the back-half agent before writing integration code (per "flag it rather than edit silently").

Findings (via the multi-spec/deref `PROSPER_PEEK`, see below):
- **No AGC/Gnm resource-creation call fires before the fault.** The complete graphics-call set in the
  run is register setup (`23LRUSvYu1M`/`BfBDZGbti7A`/`H7uZqCoNuWk`/register-indirect triplets), the
  register-context builders (now implemented), and `Zw7uUVPulbw`. The only unimplemented graphics
  calls at fault are 5 pre-graphics `libSceVideoOut` queries. There is no texture/buffer/`createResource`
  call with a `(gpu_addr, size, width, height, format)` shape anywhere in the trace.
- **`Zw7uUVPulbw` is a red herring** — disassembly of its consumer (`eboot+0x14dfb04`) shows a
  GPU-timing/profiler loop (computes frametimes: `×0xf4240`, reciprocal-divide, `vcvtsi2ss`→float,
  stored to a ring at `rbx+0xc40`). It does not gate resource upload.
- **The fault object is a Unity pipeline/material object, not a texture/buffer.** `PROSPER_PEEK` of the
  faulting `r15` (`0x…e2c300`, game-heap): `[+0]=0x2b`, `[+8]=0xf`, `[+0x20]=7`, `[+0x28]=0x19`,
  `[+0x18]==[+0x40]`→ a shared sub-object holding **packed register-like fields** (`[+0x10]=0x28a7…000e`,
  `[+0x18]=0xbba2…002b`). `[+0x140]` (the GPU-backing companion, deref'd as `[+0x08]`/`[+0x40]` byte)
  is null; `[+0x520]/[+0x530]` (an array ptr/count) are 0. It sits in a **3-element collection**
  (outer `rbx`: array@`+0x78`, count@`+0x88`=3).
- The deref is **residency-gated**: `0xba6e08` is reached only for resources whose remapped type is in
  mask `bt 0xc8220` (specific formats) AND whose flag `[+0x1a0]==0` ("not resident"). So this is a
  GPU-residency check that assumes an upload/creation step already populated `[+0x140]`.

**Interpretation:** Unity's resource-upload/creation path (which on PS5 would allocate GPU memory +
build a descriptor and set `[+0x140]`) either never ran or was no-op'd by our stubs — but it is *not*
a direct game→AGC `createResource` call we can hook. Populating `[+0x140]` with a fabricated handle to
"advance the boot" would be a correctness-first violation (a fake that moves the fault deeper).
**Proposed next step (needs back-half input):** trace Unity's texture/pipeline upload path to find the
real creation site (likely built from the shader/pipeline AGC calls already implemented +
direct-memory allocation), then wire `resource_create()` there. The `ResourceDesc` contract looks
right for the *eventual* texture case; this specific object is a pipeline object whose companion may
warrant a distinct `ResourceKind`. Left unresolved pending that trace rather than guess-fabricated.

Tooling: `PROSPER_PEEK` now takes multiple `;`-separated specs and a `*pre+off` one-level pointer
chase (`[[reg+pre]+off]`), for classifying linked object graphs at fault time.

## ▶ NEXT FRONTIER (2026-07-04): past all AGC HLE → Unity GPU-resource residency (needs real backing)

After CreateShader + the CommandProcessor submit path (below), the boot advances through the entire
AGC command frontend and now faults **with ZERO unimplemented libSceAgc calls remaining** (only 5
pre-graphics libSceVideoOut queries stay stubbed, tolerated). The command buffer executes:
`sceAgcDriverSubmitDcb` → `gpu::run_command_buffer` folds it into a GpuState (verified: "SubmitDcb #1:
71 dwords -> 12 packets applied"). Unity then proceeds to load `unity default resources`.

**The fault: `eboot+0xba6e08`**, a Unity `GfxDevice` routine (`eboot+0xba6720`, reached via the
`0xd3xxxx`/`0xd4xxxx`/`0x15fxxxx` GfxDevice chain) iterating a resource list. For a resource whose
type is in a specific mask (`bt 0xc8220`) and whose flag `[obj+0x1a0]==0`, it reads a GPU-backing
pointer `[obj+0x140]` and dereferences `[that+0x08]` → SIGSEGV at addr 0x8 because `[obj+0x140]` is
null. `PROSPER_PEEK="r15:0x140,0x1a0,0x520,0x530"` at fault shows the object (a **game-heap**
allocation `0x…e2c300`, i.e. Unity's own, not one of our zeroed singletons) is only partially
initialized: `[+0]=0x2b`, `[+0x140]=0`, `[+0x1a0]=0`, `[+0x520]=0` (array ptr), `[+0x530]=0` (count).

**Diagnosis:** this is no longer an AGC-HLE gap — it is the **AGC→Vulkan resource-backing boundary**.
Unity created a resource object but its GPU-side backing (`+0x140`) was never populated, because our
AGC resource path returns handles/zeros without constructing real GPU objects. Pushing past this
means the real GPU resource layer (textures/buffers/render targets backed by Vulkan), which is the
M4/M5 work in the "Recommended implementation order" below — largely the back-half (render_state /
vk_translate / command_processor → live Vulkan resources) now fed by real submitted command buffers.
Fabricating a `+0x140` object would be a correctness-first violation (a fake that moves the fault
deeper), so this is the point to build the real resource backend rather than stub further.

New diagnostic: `PROSPER_PEEK="rN:0xoff,0xoff,…"` (exec_image_linux.cpp) reads arbitrary offsets off a
register at fault time — for classifying large objects past FAULTMEM's 0x20-byte window.

## ✅ THE BOOT BLOCKER — RESOLVED (2026-07-04): the "source" is the Shader; CreateShader was the gap

**Resolution (supersedes the "SDK-gated / parked" conclusion below).** The null register-source
global `[eboot+0x2048c60]` was never libSceAgc-private state: it is **field +0x10 of a 0x28-byte
shader-registry slot at `0x2048c50`** — one of ~30 slots for Unity's built-in shaders, registered
at graphics init by `eboot+0x14bc002..` → `eboot+0x14e74c0(slot, shader_elf, flag)`. That function
parses a **shader ELF embedded in eboot rodata** (`e_machine=0xe0` EM_AMDGPU; sections
`.shader_header` / `.shader_text`) and calls `sceAgcCreateShader(&slot->shader, header, code)`
(NID `f3dg2CSgRKY`, via the arg-validating wrapper `eboot+0x3ae120`). Our stub returned 0 without
writing `*dst`, so every slot's shader stayed null. The earlier static-scan conclusion "no eboot
code writes the global" was the classic computed-addressing blind spot (same failure mode as the
RGCTX hunt): the writer stores through `slot+0x10`, never through the literal address.

The **register source object IS the Shader**: `SetSource` (eboot+0x3af400) reads `[src+0x08]`
(user_data), `[src+0x28]` (specials), `[src+0x5a]` (type) — the layout established by the guest's own
accesses and captured header bytes. The "classify table" ships inside each shader blob; nothing is fabricated.

**Implemented in `hle_agc.cpp` (all real semantics, layout-verified via eboot disassembly and captures):**
- `f3dg2CSgRKY` **sceAgcCreateShader** — relocates the header's self-relative pointers in place,
  binds the code pointer, patches the leading `SPI/COMPUTE_PGM_LO/HI` sh-register pair (all five
  stage pairs, beyond the initially observed ES/PS pair), guards double-relocation, writes `*dst`, and registers
  the shader in a host-side registry (`prosper_agc_shader_count()`) for the AGC→Vulkan pipeline.
- `V++UgBtQhn0` **sceAgcGetDataPacketPayloadAddress** — called from *inside* eboot's static AGC
  code (register-bank prepare, `eboot+0x3af040`): the returned payload becomes the register bank
  `[sub+0x10]`/`[sub+0x18]`. The banks live in the game's own Dcb data packets.
- `n2fD4A+pb+g` **sceAgcCbSetShRegisterRangeDirect** — IT_SET_SH_REG range packet (+ the marker
  NOP the real library emits).
- `D9sr1xGUriE` **sceAgcCreatePrimState**, `HV4j+E0MBHE` **sceAgcCreateInterpolantMapping** —
  pipeline registers derived from the bound shaders' specials/semantics using generalized semantic
  matching rather than a hard-coded identity layout.

**Result:** all 36 built-in shaders register (`PROSPER_GFXLOG` shows `pgm_patched=1` on each), the
whole `CreateWorkload` register-context chain (`0x3b5ea6` → `0x3b1562` → `0x3b1533` → `0x3afcff`)
completes, and **no unimplemented libSceAgc call remains in the boot**. The boot now faults much
later at `eboot+0xba6e08` (addr=0x8, non-AGC backtrace via `0xd3xxxx`/`0x15fxxxx`) — the next,
separate frontier. Note: the game passes AGC interface version **13**; layouts are verified against
this title rather than inferred from earlier-generation material.

Tooling added: `build-linux/imgdump <module> <out.img>` dumps a module's flat image for offline
`objdump -D -b binary -m i386:x86-64` disassembly (how the registry writer was found).

## (Historical, disproved by #641) `+kSrjIVxKFE` context-init theory

> **Correction (2026-07-13):** the authoritative PS5 3.20 symbol map identifies
> `+kSrjIVxKFE` as `sceAgcDcbPushMarker`, not a register-context constructor. Its first argument is
> a live DCB and its second argument is the marker label. The temporary `g_agc_ctx_init` handler
> described below corrupted that DCB by clearing three 0x70-byte regions on every marker. Removing
> it and emitting a correctly framed marker packet lets Blasphemous 2 render its studio logos and
> title and continue through the EULA. See #641 and `docs/AGC_TRACE.md`. The following section is
> retained only as the reasoning trail that produced the obsolete workaround; none of its
> conclusions about the import's identity or ownership are current.

The boot faults at `eboot+0x3b5ea6` inside a GPU register-setting routine. Full chain, traced under gdb:

- During `GfxDevicePS5` graphics init (a `CreateWorkload`-style fn at `eboot+0x14dd900`), the game
  computes its **register context = device+0x48** (embedded object) and calls `+kSrjIVxKFE(context)`
  as the very first operation on it (`eboot+0x14dda7c` → thunk `eboot+0x3ae7d0` → PLT `eboot+0x19b4730`
  → GOT `+0x1d95858`).
- That GOT slot resolves to **our stub** (`0x600003140`) which tail-jumps to `glog_thunk<14>` —
  i.e. `+kSrjIVxKFE` is `kAgcNids[14]`, currently an **observe-only logger that returns 0**.
- So the context is never initialized: it stays fully zeroed. `[context+0x08]` (the register-index→
  hardware-slot *classify table*), `[context+0x10]`/`[context+0x18]` (the two register-bank output
  buffers) are all null.
- The following register-set loop (`eboot+0x3afb90`, reached via thunk `eboot+0x3a7b60`) calls the
  classifier `eboot+0x3b5ea0`: `classify(table=[context+8], sel, key) = (key < table.limit16[sel]) ?
  table.subarray[sel][key] : 0x7fff`, where the table has 16-bit `limit[sel]` at `+0x2e` and
  `subarray*[sel]` at `+0x08`. With `table==NULL` it reads `[0x30]` → SIGSEGV at addr `0x30`.

**Disproved proposed fix:** implement `+kSrjIVxKFE` as an AGC register-context constructor: allocate the two
register banks, install `[context+0x10]`/`[context+0x18]`, and install a valid classify table at
`[context+0x08]` mapping (register-set selector, SDK register index) → hardware slot. The register
offsets for that mapping are exactly the independently verified tables now stored in
`agc_reg_defaults.cpp`. Note: this AGC code is **statically linked into eboot** — only the leaf
SDK entrypoints like `+kSrjIVxKFE` are imports (PLT/GOT), which is why implementing that one import
unblocks the whole internal register path.

(Superseded theories, for the record: this is NOT a `std::ctype`/`std::locale` facet issue — the
classifier is hit exactly once, not thousands of times — and NOT the `GetRegisterDefaults2` result;
wiring real RegisterDefaults did not move the fault, confirming the context table is installed by
`+kSrjIVxKFE`, not read from `GetRegisterDefaults2` here.)

### Obsolete context-object interpretation

The investigation incorrectly interpreted `+kSrjIVxKFE(context)` as the constructor for the register
context embedded at **device+0x48**. The
context holds an array of **0x70-byte register-set sub-objects at context+0x38** (index 0..2 = the
cx/sh/uc sets): the setter thunks (`eboot+0x3a7aa0/0x3a7b20/0x3a7b60`) and getters all compute
`sub[sel] = (context+0x38) + sel*0x70` (`eboot+0x3b0210`: `rax = rdi + sel*0x70`). Each sub-object:
- `[sub+0x00]` → an owner/state object (chain: `[[sub+0]+0x28]+0x10/0x18`) — **must be non-null**
- `[sub+0x08]` → the register classify table (see above)
- `[sub+0x10]`/`[sub+0x18]` → the two register-bank output buffers
- `[sub+0x32]` (u16), `[sub+0x68]` (flags byte)

### Historical Stage 1 workaround (removed by #641)

`+kSrjIVxKFE` was temporarily implemented in `hle_graphics.cpp` to install a zeroed classify table into each
sub-object's `[sub+0x08]`. With all per-selector limits = 0 the classifier returns `0x7fff` for every
register, so the register-set loops skip every write and never touch the null banks. **Result: the
fault moved from `eboot+0x3b5ea6` to `eboot+0x3b1562`** — a getter that derefs `[sub[0]+0x00]`
(still null) → `[null+0x28]`. At the time this appeared to confirm the theory, but #641 proved that
the handler merely changed live DCB contents and moved the symptom; it was never a valid context
initialization boundary.

### Deeper mechanism (RE'd 2026-07-04) — the "source" object supersedes the direct-table stopgap

Going past `0x3b1562` revealed the real wiring, which **supersedes** stage-1's approach of writing
`[sub+0x08]` directly. The eboot function `SetSource(sub, src)` at **eboot+0x3af400** owns these
fields:
```
[sub+0x00] = src                     ; the "source"/state object (NOT the context back-pointer)
if (src == 0) { [sub+0x08] = 0; return }   ; <-- null source => null table => the boot fault
[sub+0x08] = [src+0x08]              ; the classify table is COPIED FROM the source object
[sub+0x30] = [src+0x5a]; [sub+0x34] = [ [src+0x08] + 0x28 ]; [sub+0x38/0x3c] from [src+0x28]+0x14/0x16
```
So `[sub+0x08]` (the table) is not ours to set — it is pulled from a **source object** `src`, and
`src` is currently **null** because the AGC call that creates it is stubbed. A sibling flush
(eboot+0x3af440) uses `peek(table,idx) = [[table+0x00]+idx*2]` (eboot+0x3b5e90) and treats a
non-`0xffff` result as "sub exhausted" → resets the sub-object. Some sub-objects DO get a valid
(host-allocated) source in other calls, so at least one create-source HLE returns non-null; the
faulting `sub[0]` gets null.

**Root of the chain (RE'd 2026-07-04):** the `src` passed to `SetSource` is a single **global**,
`[eboot+0x2048c60]`, read (never written) at eboot+0x149a54a and eboot+0x14ddb22 and handed to the
sub-object setup (`eboot+0x3a72c0`). That global is **null**, and it is:
- NOT set by any relocation — the eboot's highest reloc offset is `0x1f4e160`, below `0x2048c60`
  (verified with a Module-parser probe over all 51,475 relocs); it lives in zero-init `.bss`.
- NOT written by any eboot code (objdump over the whole image: only the two reads reference it).
- NOT an exported symbol, and its address is never taken (`lea`) — so nothing external is handed a
  pointer to it either.

Conclusion: `[0x2048c60]` is **libSceAgc.prx's private global**, which the real libSceAgc populates
with its register-source object during its own init. Because prosper HLE-stubs libSceAgc instead of
loading a real `.prx`, that init never runs and the global stays null. This is the true root of the
`0x3b5ea6` boot blocker.

**STAGE 2 (real fix), two options:**
1. *Preferred, needs data we don't have yet:* the AGC SDK headers / a real libSceAgc.prx, to know the
   exact register-source object layout and the init entrypoint. Then implement that init in our
   libSceAgc HLE to build+install the object.
2. *Reconstruct it ourselves:* build a source object whose `[src+0x08]` is a populated register-map
   table (peek array @ `+0x00`, subarray ptrs @ `+0x08`, u16 limits @ `+0x2e`) from
   `agc_reg_defaults.cpp`, plus the `[src+0x28]`/`[src+0x5a]` register-count fields, and install its
   pointer into the guest global `0x2048c60` from an early AGC-init HLE. Feasible (we know the guest
   base, so the VA is writable) but the object layout must be fully RE'd first, and the hardcoded
   global VA is title-specific — acceptable as a stepping stone but flagged as such.

The current stage-1 direct-`[sub+0x08]` write is a stopgap that `SetSource` later overwrites; it
stands only as a documented WIP checkpoint.

### Diagnostic result (2026-07-04): the graphics init is a chain of libSceAgc objects → SDK-gated

A throwaway probe installed an empty-but-structurally-valid source object into the guest global and
re-ran the boot. The fault moved only slightly — `0x3b1562 → 0x3b1533` — to the **same class** of
null-deref (`[sub[esi]+0]` → `[+0x28]`) on the next sub-object/path. Conclusion: the AGC graphics
init is a *chain* of libSceAgc-internal object installations; each empty scaffold reveals the next of
the same kind. Building this chain correctly requires the real libSceAgc object layouts, i.e. **the
AGC SDK headers** (or a real libSceAgc.prx). Continuing to hand-fabricate the object graph would
violate correctness-first (endless chain of fakes that never reaches real rendering), so stage 2 is
**parked pending the SDK headers**. Independent, provably-correct graphics-pipeline work (AGC command
decode, shader recompiler, Vulkan backend — all unit-testable in isolation) proceeds meanwhile.

## Ruled out

Cross-title falsifications for the **present / publish path** — what reaches the screen once passes
have rendered, together with the pass identity that decides which surface receives those pixels in
the first place. One line each: the dead hypothesis, the evidence, the link. Extend this rather than
re-deriving — and read it before forming a hypothesis about a frozen, black, or missing frame.

- **The three slot-0 colour-target readers cannot be merged into one accessor as a refactor — the
  merge is a semantic decision, and it decides what the renderer renders to.** `mrt_color_binding`
  (array-first, the active-binding rule behind the attachment count and feedback detection),
  `mrt_pass_color_binding` (named-first, pass identity) and `live_renderer`'s `pass_bases[0]` (the
  raw named field) agree on every draw any producer emits, so unifying them *looks* free. On a draw
  whose named and array representations disagree they answer three different ways, and the sharpest
  case is not the obvious one: with the array naming a surface and the named triple naming none,
  grouping compares that address while `pass_bases[0]` is **0**, so the winner decides whether the
  pass renders to the array's surface or to nothing at all. Hand-built instances for all three
  divergence shapes are in `frontends/shared/tests/test_mrt_binding.cpp`; no producer can emit one,
  so there is no evidence to pick a winner with. #3026 landed the consumer-side check instead and
  filed the decision separately. #3026.
- **"A named/array divergence is inexpressible" is true of captures and live draws and NOT of the
  tree as a whole — the render fixtures emit them today.** #3023 established that captures cannot
  carry a divergence (the wire format holds slots 2 and up only, and
  `restore_legacy_color_target_aliases` re-derives 0/1 from the named triple on load) and that
  `realize_draw_item` mirrors at its single success exit. That is correct and is why the #3023 fix
  is a no-op on real content. It does not generalise: a fixture that takes a captured item and
  updates only the named fields leaves the array holding the capture's original base, and one run of
  `gpu_capture_render_replay` reports **eight** such draws (`array=0x200000 0x0` against a live named
  triple). So the enforceable invariant is "the array mirrors the named triple **or is absent**", not
  equality — a strict-equality guard fires on every legacy and synthetic draw. #3026.

- **A frozen frame with a live guest is not necessarily a guest or a draw problem — check the publish
  gate first.** Sonic Frontiers (PPSA03831) looked dead from t≈140 s with `frame_seq` frozen at 2,081,
  while the guest went on to flip 283 more times and prosper accepted 13,028 further submits and folded
  8,223 more draws. The renderer was selecting a present source by target *identity* and the publisher
  accepts only `w*h*4` bytes, so correctly rendered passes at the wrong extent (1920x1080, 1024x1024,
  3840x3072 against a requested 3840x2160) were silently dropped. Nothing logged it. #1986 / #1990.
- **Removing the publish wall does not restore blacked-out content — the two are separate causes.**
  With the extent contract in place Frontiers publishes again and the composited frame is *still*
  uniformly black (`distinct_rgb_colors == 1`, `nonblack_rgb_pixels == 0`) for the remaining 140 s of a
  300 s arm. This kills #1968 §6 (that the content going black shortly *before* the wall shared the
  wall's cause). The open question is unchanged and is #1968 §5: why no post-intro pass targets the
  flipped VideoOut buffer. #1990.
- **A climbing publish counter does not distinguish "rendering fresh frames" from "re-serving one
  retained frame" — instrument the two branches, do not infer.** On the same Frontiers arm, `frame_seq`
  reaching 5,499 was compatible with either, and the two send the next investigation to opposite places.
  The renderer's `fresh=`/`retained=` totals (carried on every `[rtt] PRESENT SOURCE EXTENT MISMATCH`
  line) settle it: between shortfall #2048 and #4096 **fresh grew by 1 (141 → 142) while retained grew
  by 2,048 (1,109 → 3,157)**. So post-wall Frontiers publishes **one retained black frame, re-served
  thousands of times**, and the last *fresh* 4K composite prosper produced was itself already black.
  That is the surface to investigate: not "why is the served frame black" but "why did the 4K composite
  go black, and then stop being produced at all". #1990.
- **`pixel_crc32` for a black frame is not a fingerprint.** `666f7b3f` is just "black 3840x2160" and
  recurs on three unrelated titles, so it identifies a resolution, not a title or a defect. Do not use
  a black-frame hash as an oracle or to claim two titles share a cause. #1990.
- **"No pass targets the flipped VideoOut buffer" is not necessarily a defect in pass selection — some
  titles never draw into their scanout at all, and the frame is in the buffer anyway.** This kills
  #1968 §5's framing (that a post-intro change makes Frontiers stop targeting the scanout). A
  `PROSPER_PASS_LOG` census over a full Sonic Frontiers boot records **`vo=1` on 0 of ~3,700 passes**,
  in every phase — including the intro, which prosper composites *correctly*. `PROSPER_DUMP_PERSISTENT`
  agrees from the other side: `scanout=MISS` on every submit, no `g_rtt` entry at the flipped address
  (`0x200a160000` / `0x200c140000`), and that address is not the base of a single pass. What prosper
  publishes during the intro is an internal RTT (`0x20851c0000`) that happens to be CPU-materialized
  and happens to hold the movie; when that stops, selection has nothing left, which is the whole of the
  "black composite". So the question to ask of a scanout that no pass writes is not "which pass was
  mis-selected" but **"what is in the guest's flipped buffer"** — for this title, the finished frame.
  #1968.
- **A registered scanout's bytes are swizzled, and every raw-scanout read took them literally.**
  `videoout_copy_*` memcpy'd guest display memory while `VideoOutBufferSnapshot::tiling_mode` had
  recorded the layout all along, so a TILE-mode title's `raw_scanout` present was horizontal-band
  noise. Two things not to re-derive: (1) a `PROSPER_DUMP_SCANOUT` dump that looks like bands is a
  **real, tiled** frame, not an empty one — Frontiers' bands de-swizzle under SW_64KB_R_X into an exact
  SEGA logo and an exact intro shot, while `distinct_rgb_colors` in the low thousands reads as
  "content" for something that is not an image; (2) a `width*height*4` read of a tiled surface is
  **short** — the tail past the nominal end holds real texels of the last block row, so truncating it
  drops part of the bottom-right (measured: the bottom-right 1024x48 of a 3840x2160 scanout loses half
  its non-black pixels). #1968.
- **"Not all its bytes are zero" is NOT evidence that the guest wrote a buffer, and a present path must
  never treat it as such.** It holds only for a freshly zeroed allocation. A title that re-registers a
  scanout over **reused** memory passes it with whatever the previous owner left there, while prosper's
  render-target map misses precisely *because* the address is new — so the test that was meant to
  protect a title publishes stale garbage in place of the retained good frame, which is #1990's failure
  class through a new door. This cost #2026 a revert (#2044). The workable test is differential:
  fingerprint the buffer when the guest registers it and require the contents to have *changed* since
  (`videoout_read_front_linear`). Note the corollary — authorship is not brightness: a frame the guest
  deliberately clears to opaque black is authored, and publishing black is then correct. #1968 / #2044.
- **A page-mapped probe is not proof that a buffer owns the bytes past its nominal end.**
  `gpu::guest_readable` answers "are these pages mapped", so an adjacent unrelated mapping passes it,
  and two separately allocated scanouts that merely land far apart satisfy a registration-stride test
  as well. At 3840x2160 that combination licenses a ~240 KiB silent over-read into another object,
  surfacing as garbage texels in the bottom-right block row. Use
  `host::guest_readable_mapping_containing` — one registered mapping spanning the whole range — *in
  addition to* the stride, because each covers the other's blind spot. #1968 / #2044.
- **Sonic Frontiers' post-intro black frame is not a present-path defect — the guest's own display
  buffer is black, and prosper is now showing exactly that.** With the flipped buffer published, a
  720 s default launch reports `[rtt] GUEST SCANOUT … publish` with ordinals reaching **2,048** (one
  fresh read per guest flip) against **9** `PRESENT SOURCE EXTENT MISMATCH` lines, all of them during
  boot with `fresh=0 retained=0`. That distinction is the whole point and it is easy to get wrong:
  `rgb_nonblack=0, distinct_rgb=1` **cannot** tell "publishing black" from "declining and freezing on
  the last logo", because it ignores alpha and both read identically — the *counters* settle it. Every
  post-140 s frame is a fresh read of the guest buffer, so the buffer really is black; the SEGA logo,
  Cyber Space intro, Sonic Team logo and middleware credits from the same run are exact 4K frames. Do
  not look for a lost render target, a mis-selected pass or a dropped publish for this symptom: the
  remaining blocker is upstream guest progression. #1968 / #2023.
- **A title's "non-draw" scanout writer is findable, and on Sonic Frontiers it is one compute
  dispatch — so prosper is the writer, and there is no identity or aliasing failure to look for.**
  The rows above establish that no pass ever targets the flipped buffer and that the frame is in
  guest memory anyway; neither says *what puts it there*, and "a path that is not a draw" was as far
  as it went. `PROSPER_PROVENANCE_ADDR=<scanout>:<span>` answers it exactly. Over a full boot the
  only writer of either registered VideoOut buffer (`0x200a160000` / `0x200c140000`) is
  `compute-buffer … identity=0x20002fe800 size=33423360` — the storage-image writeback of **one**
  compute program, alternating between the two buffers, packing and tiling its result back into
  guest memory across the padded SW_64KB_R_X footprint. **Do not read that event count as a dispatch
  rate**: `live_compute`'s "skipped identical storage writeback" fast path `continue`s *before* the
  record, so the events are an **upper bound on writebacks that changed something** — that fast
  path also requires the binding to be a cache candidate, so a non-candidate records even with
  identical bytes. 237 over 360 s against ~1,055 flips, collapsing to ~0 per sample once the frame
  is uniformly black and every writeback is byte-identical to the last. The dispatch is still
  issued: a capture selected by `PROSPER_GPU_CAPTURE_COMPUTE_ADDR` fires in the black phase.
  **Zero** `dma-data` and **zero** `write-data` events touch either address, and the draw half is
  settled independently by the pass
  census (`vo=1` on 0 of 8,443 records — every *deferred* pass plus every pass with more than 100
  non-black pixels; 8,443 over ~1,055 flips is 8.0 records per frame, exactly the post chain the
  capture enumerated, so coverage is complete **for that run** — which is the claim the arithmetic
  supports and the code paths do not, since `PROSPER_READBACK_WHY` names six ways a pass on the
  live-target path is not deferred). So the guest composites with a compute dispatch,
  prosper executes it, prosper writes it back, and prosper publishes it. Reach for
  `PROSPER_PROVENANCE_ADDR` first on any title whose scanout no pass targets — it names the writer
  kind, the program's code address and the submit, which no pass census can. **Read its silence with
  one limit in mind, because it is not stated anywhere else:** the address watch arms the
  compute-writeback, `DMA_DATA` and `WRITE_DATA` recorders, but the **colour-target** recorder lives
  inside `diagnose_resource_provenance`, which returns early unless the *separate*
  `PROSPER_PROVENANCE_DIM` is also set — and set to something that *parses* as `WxH` with both
  dimensions non-zero, so `=1` is not enough (`gpu_executor.cpp`). An `ADDR`-only run therefore
  reports **no `color` lines at all**, and that zero is void rather than negative — filed as #2111.
  Set both variables, or settle the draw question with `PROSPER_PASS_LOG`. #1968.
- **Post-intro Frontiers runs a complete post-processing chain over an empty scene — the missing
  thing is geometry, not a composite step.** A capture of the submit containing that composite
  dispatch (selected with `PROSPER_GPU_CAPTURE_COMPUTE_ADDR=0x20002fe800`, 240 s into a default
  launch) holds 15 draws and 2 dispatches, and **every draw is a `vcount=3 indices=0` fullscreen
  triangle**: bloom into a 1920x1080 `R11G11B10F`, a 9-tap bloom composite into a 3840x2160
  `RGBA16F`, a 960x540 -> 64x64 -> 16x16 -> 4x4 -> 2x1 luminance reduction, a tonemap into a
  3840x2160 `RGBA8`, then the dispatch into the scanout. All 17 operations are `realized=yes`,
  `failed=0`. There is no scene pass, no indexed draw and no depth geometry anywhere in it. #1968.
- **Every renderer-owned target the census can see really is black in that phase, on the default
  path, measured.** A `PROSPER_DUMP_PERSISTENT` census (not in the `live_gpu_targets` disable list,
  and taken with no capture armed) enumerates **17** persistent colour targets over three
  consecutive black-phase submits, in each of two independent runs. Read 17 as a **lower bound**:
  the census skips anything smaller than 64x64 and silently `continue`s on a readback failure, so
  the 16x16 / 4x4 / 2x1 tail of the luminance chain is outside it by construction. **The headline
  survives that gap transitively rather than by enumeration**, and it is worth seeing why: those
  unseen stages are exposure *scalars*, not image content — a 2x1 luminance value cannot be the
  missing picture however it reads — and **in the captured frame** their consumer is the tonemap,
  whose 3840x2160 `RGBA8` output IS in the census and reads opaque black. Read that consumer as
  *that frame's*: a one-submit capture cannot see a cross-frame read by construction, and a
  luminance reduction is exactly the quantity a renderer adapts temporally, so "the tonemap is the
  only consumer" is a claim this instrument cannot make. **The scalar half is what carries the
  headline**; the tonemap corroborates it. The silent readback-failure path is bounded separately
  and weakly:
  the enumerated count is **identical (17) in both runs**, so it is not dropping a target
  nondeterministically. Exactly one
  enumerated target has content — the CPU-materialised movie composite, still holding the last
  credits frame at `rgb_nonblack=778215/8294400`, which is *stale* and not an input to the post
  chain (`0x2059ea0000` in one run and `0x2059eb0000` in the next: **the address is run-local**,
  re-derive it rather than grepping for either). Every 4K `RGBA16F` target reads
  `00 00 00 00 00 00 00 3c` per texel (RGB bit-zero, alpha exactly 1.0) and every 4K `RGBA8` target
  reads `00 00 00 ff` (opaque black); the bloom target and three others are entirely zero. Nothing
  is lost between any target and the scanout because there is nothing anywhere to lose. Corroborated
  by a full `PROSPER_DBG=1` run: **0** recompile-rejects of any kind and **0** `[compute] skip` —
  not self-contradictory with what follows, because they are different tokens: `[compute] skip` is
  the unsupported-program / descriptor-contract reject, while the only skipped dispatch in the whole
  boot is #657's `64x64x6` layered image, twice, at boot, reported as `layered image deferred to
  #657 -> dispatch skipped (#590)`.
  **That zero has a positive control, and it needs one**: all four reject emitters
  (`[recompile-reject]`, `[cfg-recompile-reject]`, `[vertex-recompile-reject]`,
  `[exec-recompile-reject]`) are `PROSPER_DBG`-gated, so an unset variable is indistinguishable from
  a clean run. The same binary, the same variable and the same log filter print **118** reject lines
  on *Nikoderiko* (`PPSA23760`) in 130 s. #1968 / #2023.
- **`nz=0` on a `gpu_replay --inspect-only` resource marked `temporal-RTT-seed` is NOT evidence that
  the target is empty.** That count is over the resource's *guest-memory* bytes, and a
  renderer-owned target is legitimately all-zero there on the persistent-GPU-target path — prosper
  renders into a Vulkan image, not into guest memory. The content is on the `rtt-seed` line for the
  same address, and the two disagree: Frontiers' 4K `RGBA16F` post-chain buffer reports `nz=0` on
  the `PS TEX` line while its seed hashes to `6c4969fdca4e4383`, which is **not** the all-zero hash
  for 66,355,200 bytes — the persistent census then showed it is opaque black with alpha 1.0, a
  different finding from empty. Read the seed, not the `nz`. A cheap decoder for the hashes: they
  come from `gpu_capture_hash` (`gpu_capture.cpp`), so the hash of *N* zero bytes — or of any
  repeating texel pattern, which is how "opaque black" and "alpha 1.0" were identified above — can
  be computed offline and compared. **Do not reach for a stock FNV-1a 64 to do it.**
  `gpu_capture_hash` uses the FNV prime with a **basis of `0x14650fb0739d0383`** — the FNV-1a 64
  offset basis with a digit dropped; a standard implementation produces a value that can never
  match a prosper hash, and that mismatch reads as "the buffer is not zero" — the exact false
  conclusion this row exists to prevent. Same family as the `px_nonblack` inversion. #1968.
- **The census ordinal `PROSPER_DUMP_PERSISTENT` / `PROSPER_PASS_LOG` take is NOT the submit number
  `[gpucap]` prints.** On one 360 s Frontiers route the renderer-callback ordinal reaches **6,560**
  while the capture's submit counter reaches **26,209** — a 4x gap, in the direction that makes an
  ordinal estimated from a capture overshoot. Overshooting produces **no census at all**, and that
  silence is indistinguishable from "the census ran and every target was empty". Aim these windows
  with the `ms:<millis>` form instead (`diagnostic_window.hpp`), whose origin is the same *kind* as
  `PROSPER_GPU_CAPTURE_AFTER_MS`'s — the first armed check on its own path — so the two can be aimed
  at one moment to within the gap between those two lazily-started clocks, not exactly. #1968.
- **Arming `PROSPER_GPU_CAPTURE` takes the run off the default rendering path.** It is in the
  `live_gpu_targets` disable list (`live_renderer.cpp`), alongside `PROSPER_DUMP_RTGROUPS` and the
  rest, so every env-triggered capture run is the CPU-readback path. The guest's own command stream
  and its resources are unaffected — which is why a capture is still the right instrument for "what
  did the guest submit" — but do not compare a capture run's *renderer* behaviour, target residency
  or publish provenance against a default run. #1968.
- **A `0x0` native pass extent does not occur in practice — do not reach for MRT-prefix truncation to
  explain a missing colour attachment.** The renderer truncated the prefix whenever a bound slot's
  extent differed from MRT0's, and `native_w`/`native_h` are `0` whenever the guest's
  `CB_COLOR0_ATTRIB2` was never seen, so any real MRT1 compared unequal against `0` and was dropped
  with nothing logged (#2114). A per-pass census over **17 titles and ~72,900 passes** found **zero**
  passes at `native=0x0` and **zero** truncations of any kind — while five of those titles bind 464
  real MRT1 attachments, so the path is genuinely exercised rather than merely absent. The defect is
  **latent, not active**: it was closed as a guard, and no title's rendering changed.
  **Those zeros are only readable because the detector was proved able to report non-zero.** Forcing
  `native_w`/`native_h` to 0 at the MRT decision moved the counter `0 -> 2,030` and collapsed
  *rendered* MRT1 attachments `206 -> 0` under the old predicate, against `206` under the fix.
  Without that arm a census reporting zero is void, not negative.
  **Bound on the instrument, and it is the part worth inheriting:** `PROSPER_RTTLOG` is itself in the
  `live_gpu_targets` disable list — same family as the `PROSPER_GPU_CAPTURE` entry above — so this
  census ran the CPU-render path, not the default persistent-GPU-target route. No input to the
  truncation decision depends on that flag, so the `native=0x0` and truncation counts stand; but the
  forced arm never exercised the GPU-side cost of *keeping* an attachment, which is argued from the
  extent-keyed target cache rather than measured. #2114 / #2127.

- **The wrong composite of #2932 is not a present-source SELECTION error, and not the extent
  contract — the pass prosper publishes is the guest's own scanout pass, and that pass renders the
  flat white itself.** Both entries above send a "the frame exists and the screen is wrong"
  investigation at the publish gate; on *BALAN WONDERWORLD* (`PPSA02058`) the publish gate is
  innocent. Measured 2026-08-23, `tools/screenshot`, default route, no pad input,
  `PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct`, Linux/RADV: every wrong sample is
  `source=composited`, 3840x2160, `distinct_rgb_colors == 1`, `nonblack_rgb_pixels == 8294400`, so
  the renderer composited it and the publisher accepted it at exactly `w*h*4`.
  `PROSPER_PASS_LOG` over a 20 s window then names the producer: **1,255 passes target a registered
  VideoOut buffer (`vo=1`), and 403 of them read back `px_nonblack=8235719` — the exact
  `nonblack_rgb_pixels` the manifest records for every content sample of the same run — while 832
  read back 8,294,400.** Same pass, same target, two outcomes. It is not buffer parity either: the
  two flipped buffers carry both outcomes in near-equal proportion (`0x9fc0000000` 417 white / 202
  content, `0x9fc2000000` 415 / 201). So the question is what that one scanout pass samples on the
  frames it comes out white, not which candidate the renderer chose. #2932.
- **The screenshot duty cycle UNDERSTATES how often the renderer gets it right, so do not use it as
  the fix metric.** On the same BALAN run the renderer produced the real menu on **403 of 1,255**
  scanout passes (32%) while a 1 s sample grid caught it on 5 of 40 (12.5%); a separate 110-sample
  grid on the same title caught 15 of 110 (13.6%). A sparse grid over a fast, irregular alternation
  is a sample of the grid as much as of the title — instrument trap 211's family. Count producing
  passes, and only then check that the samples follow. #2932.
- **prosper never makes a readback AVAILABLE to the host, and fixing that does not fix #2932 or
  #2937.** Waiting a fence orders execution; it does not perform the availability operation that
  moves a transfer write into the host domain, and no readback in `tests/fixtures/render_runner.h`
  emits `VK_ACCESS_TRANSFER_WRITE_BIT -> VK_ACCESS_HOST_READ_BIT` at `TRANSFER -> HOST`
  (`VK_ACCESS_HOST_READ_BIT` appears nowhere in the render path). That is a real spec gap and is
  filed on its own terms, but it is **not** either defect's cause: adding the barrier at all four
  readback sites left BALAN unchanged in a matched A/B — identical `--seconds 1 --count 110` arms,
  only the lever differing, **15 of 110 content frames with the barrier against 17 of 110
  without** — and moved no Vulkan-execution test out of the noise band described below. #2932 /
  #2937.
- **#2937's five RADV failures are not a RADV defect: a standalone Vulkan program driving prosper's
  own SPIR-V renders both draw kinds correctly.** A ~150-line program with no prosper code, using
  the vertex+fragment modules dumped from `test_indexed_render` itself and the same pipeline shape
  (no vertex input state, TRIANGLE_LIST, dynamic scissor, one `STORAGE_BUFFER` at binding 3,
  host-visible index and vertex buffers), renders the non-indexed and the identity-indexed draw
  identically on the same device and driver: **1,500 indexed draws over five 300-iteration runs, 0
  failures.** The same SPIR-V through the renderer backend rasterizes nothing. So the defect is in
  what prosper does around the draw, not in the shader, the driver or the hardware. #2937.
  **SUPERSEDED, do not act on the sentence above.** The rebuilt equivalent of that program
  (`tools/vkprobe`) creates its device the same natural way and thereby builds an INVALID pipeline
  from any prosper vertex module — `VUID-RuntimeSpirv-NonWritable-06341`, because prosper's vertex
  shaders bind storage buffers and `vertexPipelineStoresAndAtomics` was never enabled — and with no
  validation layers loaded it reports coverage numbers regardless. Whether the original program had
  the same gap is INFERENCE (it was deleted and cannot be inspected), but the rebuild with the
  feature enabled and the pipeline validating clean **does reproduce the failure**, so "not the
  driver or the hardware" is not supported. See the vkprobe row further down this section, and
  #2945 / #2937.
- **And it is not in prosper's caches, pools or arenas, nor in RADV's shader path.** Every one of
  these left `test_indexed_render` at exactly 4 failures: `PROSPER_NO_INDEX_ARENA`,
  `PROSPER_NO_BACKEND_BUFFER_ARENA`, `PROSPER_NO_BACKEND_BUFFER_POOL`, `PROSPER_NO_MEMORY_POOL`,
  `PROSPER_NO_BACKEND_PIPELINE_CACHE`, `PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE`,
  `PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS`, `PROSPER_NO_BACKEND_PERSISTENT_TEXTURES`,
  `PROSPER_RTT_NOSEED`, `PROSPER_NO_DEPTH`, `PROSPER_NO_STENCIL`, `PROSPER_NO_SHARED_VULKAN_DEVICE`,
  and `RADV_DEBUG=nongg` / `llvm` / `syncshaders` / `nooutoforder` / `zerovram` /
  `nodcc,nohiz,nofastclears`. The index data is correct where it is bound, checked by instrumenting
  `vkCmdBindIndexBuffer` to print the buffer handle, the offset and the first indices read back from
  the mapped slice. #2937.
- **Do not read #2937's failure list as a set of deterministic failures — three of the five are
  flaky, and one of those passes most of the time.** Measured 30 runs each of the same binary at
  master (`deb61138`), Linux/RADV: `descriptor_array_render` **23/30 pass**, `multidraw_render`
  **3/30**, `indexed_render` **0/30**, `gpu_execute` **0/30**. An A/B on the first two needs far more
  than a dozen samples per arm to say anything — twelve-run arms of one unchanged binary produced
  10/12 and 5/12 for `descriptor_array_render` on the same afternoon. Only the 0/30 pair is a stable
  discriminator. #2937.
- **Correct pixels are not evidence that a readback is synchronized, and this is the reason #2944
  survived unnoticed for the life of the backend.** Every readback buffer in the render backend is
  host-visible memory this platform's driver keeps coherent enough that the *unfixed* code returns
  the right bytes. Measured directly: with the host-read barrier deleted from the composited-frame
  readback, `test_host_read_barrier`'s pixel assertion ("the pass rendered and read back a full
  frame") still **passes** while the structural assertion goes red. So a readback test that checks
  content can never discriminate here — assert that the barrier is RECORDED. #2944.
- **Synchronization validation does not detect a missing host-read barrier — do not reach for it as
  the oracle for this class.** Measured 2026-09-02, Fedora 44 / validation layers 1.4.341 / Mesa
  RADV STRIX_HALO, with the composited-frame barrier deleted so the defect was live:
  `VK_LAYER_VALIDATE_SYNC=1` over `test_pipeline_render` and `test_host_read_barrier` produced
  **zero** messages. The lever is proven to move — the same setting over the whole ctest suite
  produced **5 SYNC-HAZARD messages** (one READ-AFTER-WRITE in `vulkan_triangle_render`, four
  WRITE-AFTER-WRITE in `texture_mip_render`), so syncval was armed and reporting; it simply cannot
  observe a CPU read through a mapped pointer, which is the access the missing dependency protects.
  Those 5 are a separate, real finding (#3248) and are invisible to `tools/vkval`, which did not
  enable syncval. `--sync` now does; the sentence above is still true of what it can see. #2944.
- **HOST_COHERENT memory does NOT exempt a readback from the host-domain dependency.** Coherence
  removes the need for `vkInvalidateMappedMemoryRanges`; it does not perform the availability
  operation. The two halves are independent, and the backend had only the second: three of the seven
  readback sites already invalidated (they may be backed by non-coherent HOST_CACHED memory) while
  none had the barrier. #2944.
- **The same blindness holds for the COMPUTE backend, and it is measured there too.** A dispatch
  result is checked by VALUE rather than by pixel, so it is tempting to think a byte-exact assertion
  would catch a missing availability operation. It does not. With `live_compute.cpp`'s dispatch-result
  barrier deleted, `game_compute_exec` (**439 assertions**, including a byte-exact Unorm8 storage-image
  writeback) and `live_compute_descriptor_array` both still **pass**, and so does
  `live_compute_host_read_barrier`'s own "wrote the result back to guest memory" arm — only its
  structural arm goes red. #3249.
- **The live compute allocator cannot hand out non-coherent memory, so NO compute site needed a new
  invalidate.** A real negative result, and the opposite of the render backend's: `host_memory_type`
  tries `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` and falls back to `HOST_VISIBLE|HOST_COHERENT` — the
  cached tier is `wanted | HOST_CACHED`, not `HOST_CACHED` alone — so every allocation the compute
  path maps is coherent. 0 of its 6 host-read sites want `vkInvalidateMappedMemoryRanges`; all of
  them want the dependency. #3249.
- **The compute path's end-of-command-buffer image restore does not incidentally supply the
  dependency.** Every borrowed image is handed back with a `COMPUTE_SHADER|TRANSFER -> ALL_COMMANDS`
  barrier before `vkEndCommandBuffer`, which looks like it would cover everything.
  `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` does **not** include `VK_PIPELINE_STAGE_HOST_BIT`, and that
  barrier's destination access mask does not include `VK_ACCESS_HOST_READ_BIT`. #3249.
- **A device write with no host read in its OWN dispatch is still not exempt, because the compute
  memory POOL recycles host-visible allocations across bindings and submits.** This is why #3249's
  own producer list was short. `release_memory` returns an allocation to `memory_pool.available` and
  `allocate_memory` hands it to any later binding of the same memory type, where the host maps it and
  reads the retained contents to decide whether an upload is needed ("Pooled host-visible allocations
  retain their previous contents"). So the comparator baselines and the depth-bits bridge staging
  buffer — neither of which is mapped by the binding that writes it — are host-read later anyway. The
  invariant is per-allocation, not per-binding. #3249.
- **All five hazards syncval reported were REAL. None was a layer false positive, and none was safe
  by other means.** Classified individually before fixing, because "silence the layer" is the wrong
  answer for a hazard that is genuinely covered:
  * the four `SYNC-HAZARD-WRITE-AFTER-WRITE` in `texture_mip_render` are one defect seen four times
    (two assembled-chain renders x two copied levels). A `vkCmdPipelineBarrier` does sit between the
    clear and each copy, so an EXECUTION dependency exists — but every one of them names the copy
    SOURCE image, and write-after-write additionally needs the first write made available on the
    DESTINATION. One `VkImageMemoryBarrier` over the cleared range closes all four;
  * the one `SYNC-HAZARD-READ-AFTER-WRITE` in `vulkan_triangle_render` is the implicit final subpass
    dependency being weaker than it looks: with no dependency declared, Vulkan supplies
    `srcStage = ALL_COMMANDS -> dstStage = BOTTOM_OF_PIPE` with `dstAccessMask = 0`, which orders
    execution and makes nothing visible, so the following `vkCmdCopyImageToBuffer` reads the
    attachment unsynchronized against the pass's own finalLayout transition. Same defect #2945 fixed
    inside `render_draw_pass_rgba`.
  Both were confirmed by mutation rather than by reading: removing each fix returns its exact
  message, and adding it removes it. #3248.
- **The driver is not a variable for these hazards, so a lavapipe/RADV disagreement is not the
  reason to distrust one.** Measured 2026-09-02, layers 1.4.341, whole ctest suite: RADV
  (STRIX_HALO) and lavapipe report the identical set — same 2 ids, same 5 messages, same 2 tests.
  Syncval models the recorded commands rather than consulting the driver, which is also why "try
  another GPU" cannot discriminate this class (the same reason recorded for `08600` in
  `tools/vkval/README.md`). #3248.
- **A pixel assertion is blind to the mip-assembly hazard for a sharper reason than usual, and a
  content test can never be written for it.** Assembly clears to black deliberately so a level the
  guest never rendered stays unavailable instead of being derived from a neighbour. A clear landing
  after a copy therefore yields a black level — byte-identical to what a *correct* run produces for
  a missing level. Measured: with the barrier removed, `texture_mip_render` passes, including its
  "missing final renderer mip remains black" assertion. #3248.
- **The LAYER VERSION is a variable even though the driver is not — so "syncval is clean" needs the
  version stated.** With all five fixed, a `--sync` run inside `podman run ubuntu:24.04` carrying the
  CI runner's own packages (validation layers **1.3.275**, mesa 25.2.8 = lavapipe) reports a SIXTH
  hazard that 1.4.341 reports on neither driver: `SYNC-HAZARD-WRITE-AFTER-READ` x4 in
  `shadow_compare_render`, on the same-pass depth feedback of #1186 — a depth store op said to
  conflict with the in-subpass sampled read. The driver is held constant across that comparison
  (lavapipe both times), which isolates the variable to the layer version. **Not classified**: the
  path is enabled only when no draw in the pass can write depth and the attachment sits in a
  depth-read-only layout, so either 1.3.275 ignores the read-only layout or a subpass
  SELF-dependency is genuinely missing (the `VK_SUBPASS_EXTERNAL` pair from #2945 cannot order
  anything inside subpass 0). Deliberately not allow-listed and not "fixed" with a barrier: it is
  why `--sync` is opt-in rather than the CI gate. #3255.

## The renderer is not deterministic on frozen input (2026-08-23) — #2945

The cheapest reproduction of #2932 and #2937 does not need a guest, a route, or a screenshot grid.
**One `.prgbundle`, replayed offline, produces a different picture from run to run.**

```bash
# Capture any frame of an affected title (no keyboard needed; ~40 s, ~80 MB).
SDL_VIDEODRIVER=offscreen PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_CAPTURE_DIR=<work> PROSPER_GRAB_BUNDLE_AFTER_MS=21000 \
    timeout 55 ./prosper-app <DUMP_ROOT>/PPSA02058-app0

# Replay it. The per-submit hash on the [gpureplay] lines is the metric.
./gpu_replay --bundle <work>/frame_grab_*.prgbundle <work>/out.bmp
```

Measured on one *BALAN WONDERWORLD* (`PPSA02058`) bundle, master `deb61138`, Linux / AMD Radeon 8060S
(RADV STRIX_HALO), **15 replays of the same file with the same binary**:

| submit | operations | distinct output hashes over 15 replays |
| --- | --- | --- |
| 3537 (the scene) | 118 | **5** — `5c54906d637de17f` x7, `d6feaabf815c3096` x5, and `5a0e2fd6393fbfb5`, `ad8fd75518fd10b9`, `126927e49922ef95` once each |
| 3540 (the tail) | 11 | 1 — `a5e7b61cbf984383`, stable |

Same bytes in, five pictures out. The instability is inside the large scene submit and the eleven-op
tail is stable, so it is not the capture, not the seeds, and not the publish path.

**Why this matters more than the duty cycle.** Every other handle on #2932 costs a boot and a
sampling grid, and every one of them measures the grid as much as the defect. This costs ~40 s, has
no guest in it, and gives a falsifiable scalar. Bisect against the hash, not against a screenshot.

**It also explains the shape of both issues** (tracked as #2945). #2932's alternation between the real frame and a flat
clear, and #2937's Vulkan-execution failures that move when you add an `fprintf` or a query, are the
same observation at two scales: the renderer's output is a function of timing as well as of input.

Known and *not* the discriminator: the same bundle-replay log on a **content** frame of the same
title shows the identical skipped dispatch,
`[compute] program 0x3007980000 image 0x306c0f0000 1920x1080x1 ... DCC metadata extent is
unsupported -> dispatch skipped (#590)`, so #590 costs a real 1920x1080 storage-image dispatch every
frame on this title but does not separate a white frame from a good one.
### Renderer determinism (#2945 / #2932 / #2937)

**START HERE — the reproduction, end to end, three seconds a sample.** Everything else in this
section is what the reproduction was used to rule out. Do not re-derive it from a boot.

```bash
# ONCE: pull the scene submit out of the bundle attached to #2945 (~650 MB, ~40 s).
gpu_replay --bundle frame_grab_PPSA02058_*.prgbundle --bundle-extract-submit 3537 s3537.prgcap

# THE REPRODUCTION: one indexed draw, ~3 s, BINARY outcome.
gpu_replay s3537.prgcap --draw 42
#   hash=9068fcf09de07383   the half-screen composite triangle rendered   (correct)
#   hash=a5e7b61cbf984383   the scanout is untouched: the draw drew nothing

# WHAT FAILS, directly observed rather than inferred:
PROSPER_GEOM_PROBE=42 gpu_replay s3537.prgcap --draw 42 --recompile-raw
#   [geom-probe] verts-written=3 finite=3 on-screen=0 clipped=3 (offscreen=3 w<=0=3 nan/inf=0)
#   [geom-probe]   clip-bbox x[-1,-1] y[1,1] z[0,0] w[0,0] -> DEGENERATE
#   i.e. every storage-buffer load in the vertex shader returned 0, so all three vertices
#   collapse to one clip point with w=0 and the clipper discards the primitive.

# WHAT IS PROVABLY IDENTICAL between a good and a bad run -- do not re-check these:
PROSPER_BUFLOG=1        gpu_replay s3537.prgcap --draw 42   # host-side source words
PROSPER_BUFFER_ECHO=1   gpu_replay s3537.prgcap --draw 42   # DEVICE-side slices + index buffer,
                                                            # and the descriptor set/offset/range
RADV_DEBUG=shaders      gpu_replay s3537.prgcap --draw 42   # the compiled ISA
```

Draw 42 was found by prefix bisect: `--through-operation N` over 8 replays each is 1 distinct hash
through operation 55 and 3 of 8 at 56, and operation 56 is that draw. Draws 0, 1, 7, 9, 11, 31, 32
and 41 replay deterministically 8 of 8 while 42, 43, 60 and 90 do not — the affected ones are all in
the same six-attachment 4K scanout pass, which is the sharpest untested lead in this section.

**And the same class reproduces with no prosper code in the process** —
`tools/vkprobe --vs vs.spv --fs vs.spv.frag --iterations 200`. Read that tool's README before
quoting anything from it, including a clean run.

**RESOLVED 2026-08-23 — it is the DRIVER or the HARDWARE, not prosper's generated SPIR-V, and not
storage-buffer loads.** The fork above was decided by hand-writing SPIR-V that does the same loads
and running it beside prosper's own module *in one process*. The whole decision now costs one
command:

```bash
# Hand-written controls: tools/vkprobe/shaders/*.spvasm (spirv-as --target-env vulkan1.1).
# --vs-b runs a second module pair interleaved render-by-render, so both meet the same window.
vkprobe --vs prosper_vs.spv  --fs prosper_fs.spv \
        --vs-b minimal_ssbo_vs.spv --fs-b minimal_green_fs.spv --iterations 20

# What the shader actually SAW -- "[1,2,3]" correct, "[1,--,--]" every invocation got index 0,
# "[--,--,--]" no vertex invocation ran at all. Module a never writes, so its "[--,--,--]" is the
# readback path's own positive control.
vkprobe --vs no_ssbo_vs.spv --fs minimal_green_fs.spv \
        --vs-b index_readback_vs.spv --fs-b minimal_green_fs.spv --readback-dwords 16:3

# AND YOU MUST PUT LOAD ON THE GPU, or you will measure a null. See the trigger row below.
```


The observable is one frozen `.prgbundle` replayed offline by `gpu_replay` producing a different
picture from run to run. Everything below was measured on master `406ff0fd`, Linux / AMD Radeon
8060S (RADV STRIX_HALO), Mesa 26.1.4.

- **READ THIS ROW FIRST: the failure rate drifts machine-wide over minutes, so a NON-INTERLEAVED A/B
  here is void, not negative.** The identical command (`gpu_replay s3537.prgcap --draw 42
  --recompile-raw`) measured 0/20 failures in one ten-minute window and 12/12 in the next, with no
  change to the binary, the environment or the machine's process list. Every apparent lever in this
  investigation that was measured by running arm A then arm B evaporated when the arms were
  interleaved run-by-run: `PROSPER_SERIAL_DRAW_REALIZE` looked like 19/20-vs-10/20 and is 10/10
  distinct on both arms interleaved; `RADV_DEBUG=zerovram`, `nodcc` and
  `PROSPER_NO_BACKEND_PERSISTENT_TEXTURES` each looked like 4-distinct-of-10 against a 10-of-10
  baseline and are indistinguishable interleaved — two of them produced *byte-identical* hash
  sequences, which is what exposed them. **Interleave the arms, quote n, and prefer a modal fraction
  to a distinct count.** #2945.
- **`gpu_replay --warmup-repeats N` is not an idempotent oracle.** It calls `execute_frame` N times
  without re-restoring the RTT seeds (`gpu_replay.cpp`), so each repeat renders on top of the
  previous one's persistent targets and the hashes drift by construction. Ten distinct hashes over
  ten repeats says nothing. Use separate processes. #2945.
- **`PROSPER_RTT_NOSEED=1` destroys the discriminator on a composite draw and reads as a fix.**
  Without the seed the target starts cleared, so "the draw rendered black over half the frame" and
  "the draw rendered nothing" both hash to the all-zero image — 25 of 25 replays identical, which
  looks like determinism restored. Measured with `PROSPER_GEOM_PROBE` instead, the same lever is
  *worse* than the default (9 of 20 degenerate against 3 of 20). #2945.
- **The renderer's nondeterminism is NOT the missing external subpass dependencies, though those
  were real and are now fixed.** Every render pass was created with `dependencyCount = 0`, so
  Vulkan's default outgoing external dependency (`dstStageMask = BOTTOM_OF_PIPE`, `dstAccessMask =
  0`) made each pass's writes available to nothing, and `vkCmdCopyImageToBuffer` read attachments
  whose `finalLayout` transition had just been performed by `vkCmdEndRenderPass` with no dependency
  at all. Synchronization validation named it exactly — 10 × `SYNC-HAZARD-READ-AFTER-WRITE`,
  0 after the fix, and 10 again with `PROSPER_NO_RENDERPASS_EXTERNAL_DEPS=1` on the same binary.
  **Its effect on the nondeterminism is UNDECIDED, not zero.** Interleaved A/B on the bundle,
  30 replays per arm on the merged (`ALL_COMMANDS`) masks, gave 1 distinct hash in *both* arms --
  the whole measurement landed in one of the quiet windows trap 223 describes, so it discriminates
  nothing. The earlier "18 distinct against 17" figure was taken on the narrow masks and is
  withdrawn -- not because those masks were shown to break anything (#2950 establishes that the
  guards used to say so fail on plain master too), but because that build's dependencies were not
  the ones being merged. Land the dependencies on their own merits; the bundle A/B
  owes a re-run in a window where the baseline is unstable. #2945.
- **The layer's verbatim words for the over-limit vertex interface, recorded once so nothing has to
  re-derive the arithmetic.** Everything else that cites this cites this row:

  ```
  Validation Error: [ VUID-RuntimeSpirv-Location-06272 ] | MessageID = 0xa3614f8b
  vkCreateShaderModule(): SPIR-V (Vertex stage) output interface variable
  (Location = 31 | Component = 3 | Type = OpTypeFloat 32 bits) along with 4 built-in components,
  exceeds component limit maxVertexOutputComponents (128).
  ```

  Note *"along with 4 built-in components"*: the layer is counting components and folding
  gl_Position's four in, which is where 32 locations x 4 + 4 = 132 against 128 comes from. Read the
  VUID's own spec text and you get a **Location-count** rule that 32 locations at 0..31 satisfies
  exactly — so the arithmetic beside the observation is the layer's, not the spec sentence's. The
  observation is the evidence; quote the message, not a re-derivation. #2945.
- **Nor is the over-limit vertex output interface, though that was real too.** `SPI_PS_INPUT_CNTL_0..31`
  are sticky context registers and `extract_render_state` marked a slot valid when the register was
  merely present, so a pixel shader with one interpolant reported `valid_mask=0xffffffff` and the
  vertex emitter fanned one `EXP PARAM0` out to 32 locations — 128 components plus gl_Position's 4,
  against a `maxVertexOutputComponents` of 128 (`VUID-RuntimeSpirv-Location-06272`, the only
  validation error in the whole replay). Bounding the fan-out to the attributes the fragment program
  reads removes the error; the draw still vanishes at the same rate. Interleaved, 30 per arm, twice:
  23 bad fixed / 20 bad mutated before the masks were widened, 15 / 12 after. Both arms fail about
  half the time either way, so on this the answer is a genuine negative rather than an undecided
  one. #2945.
- **Not the host-visible upload path, in any of its forms.** Interleaved arms on the one-draw
  reproduction, 12-30 runs each, all indistinguishable from the default:
  `PROSPER_NO_BACKEND_BUFFER_ARENA`, `PROSPER_NO_BACKEND_BUFFER_POOL`, `PROSPER_NO_MEMORY_POOL`,
  `PROSPER_NO_INDEX_ARENA` (individually and together), `PROSPER_NO_BACKEND_PIPELINE_CACHE`,
  `PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE`, `PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS`,
  `PROSPER_NO_BACKEND_PERSISTENT_TEXTURES`, `PROSPER_NO_BACKEND_BATCH_SUBMITS`,
  `PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS`, `PROSPER_NO_SHARED_VULKAN_DEVICE`,
  `PROSPER_NO_SHADER_CACHE`, `PROSPER_NO_DEPTH`, `PROSPER_SERIAL_DRAW_REALIZE`,
  `PROSPER_DRAW_REALIZE_THREADS=1`. #2945.
- **Not the driver, on every lever that was tried, and not address-space layout.**
  `RADV_DEBUG=zerovram | nongg | llvm | syncshaders | nodcc | nofastclears` all measure the same as
  the default when interleaved, and `setarch -R` (ASLR off) still produced both outcomes — 8 good, 6
  bad, 1 good over 15 runs, flipping at the same point in time as the ASLR-on arm beside it. #2945.
- **Not an explicit host-write visibility barrier, and not descriptor-pool address recycling.** A
  `HOST_WRITE -> MEMORY_READ` barrier at the top of every pass command buffer (redundant per spec,
  added as an A/B) measured 2 bad of 15 against the default's 4 of 15; leaking every per-pass
  `VkDescriptorPool` so no descriptor VA is ever reused measured 3 of 15 against 4 of 15. Both
  levers were removed again. #2945.
- **The one-draw reproduction, and what is PROVABLY IDENTICAL between the two outcomes.** The
  divergence localizes to a single operation: prefixes of *BALAN WONDERWORLD*'s submit 3537 through
  operation 55 replay to one hash over 8 runs, and adding operation 56 — draw 42, an indexed
  three-index composite triangle into the guest scanout — makes it nondeterministic.
  `gpu_replay s3537.prgcap --draw 42` renders that draw alone in ~3 s with a binary outcome, and
  `PROSPER_GEOM_PROBE=42` shows the failing runs emit clip position **(-1, 1, 0, 0) for all three
  vertices** — every storage-buffer load in the vertex shader returning 0, so the primitive is
  discarded with `after_clip=0`. Between a good and a bad run these are identical: the host-side
  source words (`PROSPER_BUFLOG`, md5-identical over 10 runs), the GPU-side contents of every bound
  slice and of the index buffer read back through the GPU in the same command buffer
  (`PROSPER_BUFFER_ECHO`, including a deliberately mutated `w = 2.0` that the memory carried while
  the shader read 0), the descriptor allocation result, set handles and per-binding
  buffer/offset/range, and the compiled RDNA2 ISA (`RADV_DEBUG=shaders`, byte-identical). Re-taken
  after the probe was corrected — the copies moved out of the render pass, `TRANSFER_SRC` added
  under the switch, and `gpu_replay`'s geometry-probe rebuild given the same dead-varying bound the
  live path applies — and unchanged: 6 of 12 replays vanish, and every one of the 12 reads the
  correct `3f800000 3f800000 00000000 3f800000` and indices `0 4 5` off the device. **Do not
  re-derive any of that; start from what the control cannot see.** #2945.
- **Declaring an explicit external subpass dependency REPLACES the implicit one, so any mask narrower
  than the default silently removes visibility — and neither validation nor ctest can see it.** The
  implicit incoming dependency is `srcStageMask=TOP_OF_PIPE, dstStageMask=ALL_COMMANDS,
  srcAccessMask=0, dstAccessMask=<all reads and writes>`; the outgoing one is
  `srcStageMask=ALL_COMMANDS, srcAccessMask=<all writes>, dstStageMask=BOTTOM_OF_PIPE,
  dstAccessMask=0`. The first attempt at the fix above set the incoming `dstAccessMask` to the
  ATTACHMENT accesses only, which dropped visibility for every `SHADER_READ` in the pass — every
  texture a transfer had just uploaded and every buffer a compute dispatch had just written.
  Synchronization validation stays clean either way — removing a visibility operation the application
  never declared is not a hazard the layer can see — and all 302 ctest cases pass on both. **Keep
  both dependencies a superset of the defaults — `ALL_COMMANDS` / `MEMORY_READ|MEMORY_WRITE` — so
  they can only add ordering.** The tempting edit is to narrow them for performance; a narrowing that
  remains a superset of the implicit defaults is legitimate, one that does not is a silent
  visibility hole.
  **And the masks are asymmetric by requirement, not by taste.** `ALL_COMMANDS` is legal only on the
  `VK_SUBPASS_EXTERNAL` side of each dependency; on the subpass side, VUID-00837/-00838 allow only
  stages the bind point supports, so it must be `ALL_GRAPHICS`. Getting that wrong is rejected 923
  times across 15 tests — but **only by `tools/vkval/vk_validation_scan.py`, which is a CI job and
  is not part of a local `ctest` run.** Synchronization validation on the replay path was clean,
  303 local ctest cases passed, and the snapshot guards were already red for #2950's reasons, so
  nothing local saw it. If you touch these dependencies, run that scan.
  **This row rests on the Vulkan contract, NOT on a measurement, and the measurement that looked
  like one is withdrawn.** The narrow masks were first written up as having "collapsed three rung-6
  guards", because `messenger-scene`, `dead-cells-gameplay` and `gris-gameplay` all failed at
  `structural matches 0` on that build. That run had no control. #2950 then established that those
  three fail identically on plain `origin/master` and on a 2026-08-02 build, so the narrow-mask run
  could not distinguish "the masks broke it" from "already broken", and no snapshot guard on that
  machine is a usable renderer oracle until #2950 resolves. The reason to keep the masks wide is
  that a non-superset dependency removes visibility the pass relies on — which is true from the
  spec, and was true before anything was measured. #2945 / #2950.
- **THE CONTROL WAS INVALID, so every reading anyone has quoted from it — in either direction — is
  void, and the corrected control reproduces the defect with no prosper code in the process.**
  `tools/vkprobe` drives prosper's dumped SPIR-V through a bare Vulkan pipeline. It created its
  device **without `vertexPipelineStoresAndAtomics`**, and prosper's recompiled vertex shaders fetch
  through `STORAGE_BUFFER` descriptors, which Vulkan requires to be `NonWritable` in the vertex
  stage unless that feature is enabled (`VUID-RuntimeSpirv-NonWritable-06341`). Every pipeline it
  built was therefore invalid, and because it loads no validation layers it reported coverage
  numbers regardless. That voids its 3,000-clean-iteration result and the two failures later used to
  withdraw it. **It also casts the same doubt over #2937's original 1,500-draw measurement — the one
  that turned "RADV is broken" into "prosper is broken" — but that is INFERENCE, not fact:** that
  program was deleted during cleanup and cannot be inspected, and the inference rests only on the
  rebuild sharing its shape and on a hand-written control having no reason to enable the feature.
  Cheap to check if anyone recovers it; expensive to assume either way.
  **With the feature enabled and the pipeline validating clean, the control still fails.** Measured
  as `vkprobe --vs vs.spv --fs vs.spv.frag --iterations N` on the #2937 module dump (default
  records, default indices `0,1,2`; the fully-covered figure is 496 pixels of a 64x64 target, not
  the whole target, because that dump's record stride differs from the tool's default): 3 of 5
  iterations empty in one run *under the validation layers with zero core-validation findings*, 38
  of 200 in another, and 13 of 150 in a third — **3 failing runs of about 63**. `--iterations` is
  the sample count; the failure is per-PROCESS in shape, so quote runs.
  **The arm attribution survives the obvious confound.** The first version always submitted
  non-indexed then indexed, which made "the indexed draw failed" indistinguishable from "the second
  submit failed". The tool now alternates the order every iteration and reports both breakdowns:
  the non-indexed arm has never come back empty in any run, before or after the change. **The
  DE-CONFOUNDED evidence is narrower than that total and should be quoted as such: one failing run
  of 150 alternating iterations, 13 indexed empties split 9 first / 4 second, 0 non-indexed.** The
  ~9,000 iterations behind the "never" were nearly all run under the old loop, where non-indexed was
  always first. One run is enough to refute a deterministic positional mechanism — it predicts 13/0
  or 0/13 and predicts non-indexed empties at the same rate, and neither happened — but it is one
  run. The alternation is also strict parity, so a period-4 alias survives it.
  **What that does and does not license, stated exactly, because the obvious phrasing is wrong.**
  The control does not reproduce prosper's host-side Vulkan usage — it *avoids* it; the pass,
  descriptors and synchronisation are vkprobe's own. So the correct conclusion is that **prosper's
  host-side usage is not NECESSARY for the failure**, which is weaker than "cleared" and is the
  claim the evidence supports. It may still be sufficient on its own, and it may still be what
  raises the rate. The tempting sentence — "this clears descriptors, sync and lifetimes" — is the
  one to avoid: it contradicts the amplification reading in the same breath, since an amplifier has
  to live somewhere, and the shared SPIR-V and the shared driver are common to both sides.
  What the row above narrowed is therefore not wasted but re-scoped: those twenty levers are all
  *unnecessary* conditions too.
  **The rates differ, by roughly an order of magnitude on the unit this row insists on (RUNS):**
  prosper's one-draw reproduction fails about half its runs in a bad window, the control 3 of ~63.
  Not "orders" plural — that is only reachable per-iteration, which the drift rule above rejects.
  A gap that size still suggests something on prosper's side raises the rate, but it is one
  comparison across different workloads and it is not strong evidence. The remaining suspects are prosper's generated SPIR-V (which the control executes),
  prosper's host-side usage as an amplifier, and the driver/hardware.
  **The index-memory lead: falsified as a NECESSARY condition, undecided as a rate factor.** The
  arms differ in more than indexedness — only the indexed one binds an index buffer, and vkprobe's
  was `HOST_VISIBLE|HOST_COHERENT`. Since the characterised mechanism is loads returning zero, an
  index fetch reading zeros gives `0,0,0`, a degenerate triangle and zero coverage: the tool's
  entire observable. `--device-local-indices` stages the same bytes into DEVICE_LOCAL through a
  transfer and makes it a single-variable A/B. Caught in a failing window, **the DEVICE_LOCAL arm
  failed 2 of 10 iterations**, with the selected memory type's property flags printed as `0x1` —
  DEVICE_LOCAL and demonstrably not HOST_VISIBLE — so the failure does not require host-coherent
  index memory. A 20-run-per-arm interleaved A/B immediately afterwards gave 1 failing run against
  0, which does not separate the rates: at this base rate a quiet 20-run arm is likely either way,
  and roughly 48 runs per arm are needed before that null means anything. One positive outranks that
  null for the necessity question.
  **The probe now reports the memory type it actually got, and that guard is the point.**
  `memory_type` matches a SUPERSET, so asking for DEVICE_LOCAL on an APU, on lavapipe, or on any
  device that orders a ReBAR heap first returns memory that is also HOST_VISIBLE — the arm would
  have printed "DEVICE_LOCAL (staged)" and changed nothing about where the indices live. It now
  excludes HOST_VISIBLE, exits 2 if no such type exists, and prints the selected type's flags.
  An experiment that fails silently behind a loud label is the subject of the row below. #2945.
  **The transferable lesson is about the instrument, not the bug: a control is only a control once
  it is VALID, and a program that loads no validation layers cannot tell you that it is.** Two
  investigations were steered by this one for weeks. `vkprobe` now enables the feature, exits 2 if
  the device lacks it, and refuses a vertex interface over `maxVertexOutputComponents`; its README
  says a clean run proves nothing on its own. #2945 / #2937 / #2950.

- **THE VERDICT (2026-08-23): #2945 is RADV or the hardware. prosper's generated SPIR-V is not the
  cause, and a storage-buffer load is not even required.** A hand-written vertex module
  (`tools/vkprobe/shaders/minimal_ssbo_vs.spvasm`) performing the same loads through the same
  descriptor shape, run **in the same process** as prosper's dumped module and interleaved
  render-by-render with it (`vkprobe --vs-b`), fails at the same rate: **52 failing runs of 400 for
  prosper's module against 54 of 400 for the hand-written one** (220 against 228 empty indexed
  iterations of 8,000 each). The paired form is what carries this, and the sharpest statement of it
  is not the two totals but their agreement **run by run: the two arms disagreed on 2 of 400 runs.**
  A separate paired campaign, `no_ssbo_vs` — which declares the storage buffer and never reads it —
  beside `index_readback_vs`, gives 142 and 138 failing runs of 400, so **a storage-buffer load is
  not required either**. Do *not* read 142-vs-52 as "no_ssbo fails more": those are different
  processes with a different option set (`--readback-dwords` is on for one), run in different
  windows, and nothing in this design makes them comparable. Only the within-campaign pairs are. Every module is `spirv-val --target-env vulkan1.2` clean, prosper's
  four dumps included. Linux, AMD Radeon 8060S (RADV STRIX_HALO, `0x1586`), Mesa 26.1.6 host /
  26.1.4 container, kernel 7.1.5. #2945.
- **It is `vkCmdDrawIndexed` specifically, and that arm is not noisy — it is silent.** Across the
  same campaign the **non-indexed arm came back empty 0 times in 32,000 iterations**, while the
  indexed arm beside it in the same iteration, on the same pipeline, over the same buffers, came
  back empty 1,792 times. The arms alternate order every iteration and the by-position line is
  printed beside the by-arm line, so this is an attribution to the draw KIND rather than to
  submission order. #2945.
- **The positive control that makes those numbers readable: the identical binary and the identical
  module files render correctly on lavapipe.** 20 iterations, both arms, 0 empty, for prosper's
  dumped module and for both hand-written ones — taken *during* the window in which RADV was failing
  20 of 20. Without it "the hand-written module fails too" would be as consistent with a defect in
  `vkprobe` as with one in the driver. #2945.
- **What the GPU actually does — and the FIRST attempt at this row was unfalsifiable, which is worth
  more than the answer.** `index_readback_vs` stores `gl_VertexIndex + 1` into the record buffer, so
  the host can see which vertex indices the shader was handed. The first campaign ran it with the
  **identity** indices `0,1,2` — where "the index was fetched correctly" and "the shader got the
  ordinal and the index buffer was never read" produce the *same* readback. On that design the arm
  looked 91.6% correct. Re-run with `--indices 3,4,5`, where the only correct answer is
  `[--,--,--,4,5,6,--,--]`, **1,400 indexed iterations gave the correct answer 17 times — 1.2%**:
  288 (20.6%) reported the shader saw `0,1,2`, the ordinals rather than the uploaded indices, and
  1,076 (76.9%) wrote nothing in the watched window at all. So the identity design was hiding most
  of the failures, and any rate in this section measured with identity indices is a **lower bound**.
  **Use non-identity indices for anything diagnostic.** #2945.
- **What that readback does NOT establish, stated because the obvious reading is over-strong.** Three
  earlier sentences here — "the index fetch returned zeros", "no vertex invocation ran at all", and
  "it never fetches a wrong non-zero index" — are **withdrawn**. None is separable by this
  instrument. A shader handed an out-of-range index writes outside the descriptor's range and the
  write is dropped, which is indistinguishable from never running; a lost index reading as `0` and
  the same index reading as `7` both land outside the watched window in the same way; and the
  `[1,2,3]` pattern on an indexed arm is equally consistent with the shader having been handed the
  ordinals and with the host's sentinel reset never becoming visible, leaving the *previous*
  render's bytes to be read back. What is established is narrower and enough: **the vertex indices
  reaching the shader are, most of the time under load, not the ones the host uploaded**, and
  lavapipe returns the correct distinct answer for every index set on the same binary and files.
  #2945.
- **The non-indexed arm's "0 of 32,000" is partly a LIMIT OF THE OBSERVABLE, not purely a property
  of the hardware — do not quote it as a clean arm.** For `index_readback_vs` the non-indexed arm's
  correct readback is `[1,2,3]`, which is also exactly what a stale read of the previous render
  returns, so that arm cannot report the failure even in principle. It measured `[1,2,3]` on 1,400
  of 1,400 diagnostic iterations, and that number is undiagnostic by construction. The arm
  difference is still real for `minimal_ssbo_vs`, whose non-indexed coverage stayed at exactly 496
  throughout — but "indexed draws are affected and non-indexed ones are not" is a claim this
  apparatus supports only weakly, and it is not load-bearing for the verdict. #2945.
- **GPU load from another process INDUCES the failing regime on demand — but it is SUFFICIENT, not
  NECESSARY, and the first version of this row got that wrong.** Three heavy `gpu_replay`
  full-submit replays flipped a passing box into failing **within one round** of a 3-second detector
  loop and it recovered ~15 s after they stopped; a second `vkprobe` used purely as load reproduces
  it more weakly, so the load need not be prosper. That is a real lever over trap 223's drift and it
  is repeatable. **What does not hold is the converse:** an independent reviewer measured 13/20 and
  8/20 failing with no `prosper-app`, `screenshot`, `boot_trace`, `gpu_replay` or `vkprobe` running,
  verified through `/proc/*/fd`. So a clean arm on a quiet box is **weak, not a clearance**, and any
  0-of-n here reads as *undecided*. Two further consequences: quote the wall-clock **span** beside
  the n, because 400 runs inside 173 s are one drift period rather than 400 samples of what varies;
  and `pgrep -x 'prosper-app|screenshot|boot_trace'` matches none of `gpu_replay`, `vkprobe` or
  `ctest`, which is exactly the load that matters. #2945.
- **prosper's own one-draw reproduction and the bare-Vulkan control go bad TOGETHER, so the
  control's verdict transfers.** 309 interleaved rounds, each running the paired `vkprobe` and then
  `gpu_replay s3537.prgcap --draw 42`: **P(vkprobe fails | the replay drew nothing) = 26/33 = 0.79**
  against **P(vkprobe fails | the replay rendered) = 37/276 = 0.13**. Six-fold. This is what licenses
  reading a `vkprobe` result as a statement about #2945 rather than about a different defect that
  happens to look similar. #2945.
- **Not the index type, not where the index memory lives, and not any RADV lever.** Measured in a
  saturated window, 10 iterations each on `no_ssbo_vs`: default, `RADV_DEBUG=nongg`, `llvm`,
  `zerovram`, `syncshaders`, `nocache`, and `--device-local-indices` all gave **10 of 10 indexed
  empty and 0 of 10 non-indexed**. `--device-local-indices` prints the selected memory type's
  property flags as `0x1`, so it demonstrably is not HOST_VISIBLE. `--index-bits 16` fails as well,
  with the same readback patterns. `RADV_DEBUG=llvm` failing rules out ACO; `nongg` failing rules
  out NGG. #2945.
- **A clean synchronization-validation run is still not a clearance, and this time the layer was
  proved to be printing.** Core + sync validation reported **zero findings** during a run whose
  indexed arm was empty 10 of 10. The layer was loaded (`VK_LOADER_DEBUG=layer` shows the library)
  *and* demonstrated to emit, by a deliberate stage-mismatch arm that printed
  `VUID-VkPipelineShaderStageCreateInfo-pName-00707`. Without that second half the zero would have
  been indistinguishable from a layer that never ran — which is the shape of every silent-instrument
  trap in this file. #2945.
- **#2937's five failures do NOT collapse into #2945 — measured, and this kills the tempting
  unification.** The signature invites it: every failing assertion in those tests is an indexed draw
  while the non-indexed one in the same pair passes, and they were found under `ctest -j4`, which is
  the concurrent GPU load #2945 needs. But the two behave differently in the way that matters.
  Interleaved A/B, six rounds, `ctest --no-tests=error -j1` over exactly those five cases: **5 of 5
  fail in every quiet round**, and 3-5 of 5 under a sustained `gpu_replay` load beside them — if
  anything slightly *fewer*. So load is not their trigger, and they are **deterministic** where
  #2945 is stochastic (its worst measured arm is 142 failing runs of 400). **That is the whole of the
  argument, and one tempting extra leg is void:** "on the same quiet GPU `vkprobe` measures 0 failing
  runs of 48, so a generic driver indexed-draw defect cannot explain them" does not hold, both
  because a quiet-box null here is undecided rather than negative (see the load row) and because
  `vkprobe`'s identity-index design is structurally unable to see most of the defect — trap 122's
  shape, a control blind to the class it is being used to exclude. So: #2937 is **not this defect's
  load-triggered stochastic dropout**, which is what the determinism measurement shows; whether it
  shares a deeper cause is open, and it is not excluded from being prosper's own. #2945 / #2937.
- **And #2950 is UNTESTED against this, so do not assume it either.** Its guards fail
  deterministically, reproduce on a 2026-08-02 build and on Mesa 26.0.3, and its two arms differ
  only slightly (SSIM 0.0029698 against 0.0029666) — the same *deterministic* character as #2937 and
  the opposite of #2945's. A guard run is sustained GPU load, so the load regime is consistent with
  it, but consistency is all there is: nothing has been run. The cheap test is to run `vkprobe`
  paired beside a guard and see whether the control fails while the guard does. #2945 / #2950.
- **Superseded by the rows above: "the remaining suspects are prosper's generated SPIR-V, prosper's
  host-side usage as an amplifier, and the driver/hardware."** The first is dead. The third is the
  answer. The middle one is *unresolved and now unimportant* — prosper's rate is higher than the
  control's **per draw** (10.7% of 309 replay rounds bad, against ~3.9% of the control's indexed
  iterations) and LOWER per run (33 against 63 of those same 309 rounds), so name the unit before
  quoting it. Either way the two move together (P(vkprobe fails | the replay drew nothing) = 0.79
  against 0.13) and the load rows explain the association at least as well as an amplifier does:
  prosper's reproduction generates its own GPU load. Do not spend anything more on it before the
  driver defect is reported. #2945.
- **The GOOD hash moves; the FAILURE hash cannot move for a RASTERIZATION reason. Key any
  determinism harness on the failure.** BALAN's `s3537 --draw 42` rendered `9068fcf09de07383` on
  2026-08-23 and `ccc433ff6d980383` on 2026-09-02. The failure value `a5e7b61cbf984383` did not
  move, and the reason is structural rather than empirical: "the draw drew nothing" leaves the
  scanout exactly as the capsule seeded it, so the output hash is the SEED's hash — which the
  capsule carries and `gpu_replay <capsule> --inspect-only` prints, four lines above where this
  section quotes it:
  `rtt-seed addr=0000009fc0000000 extent=3840x2160 format=rgba8 bytes=33177600 hash=a5e7b61cbf984383`.
  Both values come from the same `gpu_capture_hash` over a byte vector, so the identity is exact,
  not an observed coincidence. What is NOT claimed here is why the good value moved: the row four
  below shows the *rebuilt 2026-08-23 binary* returning the same new value, which rules out the
  obvious explanation that the renderer changed the picture. An earlier version of this row asserted
  that explanation and cited "43 commits touching `src/render` and `src/gpu`" — wrong three ways:
  the causal claim is falsified by that row, `prosper/src/render` exists at neither endpoint (the
  path matched nothing), and the count over `prosper/src/gpu` is 657 first-parent commits. The
  operative half is untouched by any of it: **a harness keyed on the good hash reports every round
  of a perfectly healthy renderer as a failure.** `tools/vkprobe/correlate_with_replay.sh` shipped
  with exactly that keying and would score 100% `other` on current `main`; it now derives the
  failure key from the capsule and reads `bad`/`rendered`. #2945.
- **A hand-made positive instance of the failure costs one command, and no determinism null on this
  issue should be believed without one.** The characterised mechanism is "every storage-buffer load
  in the vertex shader returns 0", so supply that by hand rather than waiting for it:

  ```bash
  head -c 192 /dev/zero > zeros.bin        # binding 3 is 6 records x 32 bytes; --list-resources 42:vs
  gpu_replay s3537.prgcap --draw 42 --override-resource 42:vs:3 zeros.bin out.bmp
  #   [resource-override] draw=42 stage=vs binding=3 ... new-hash=fea2d2bd51234c83
  #   output=3840x2160 target=0000009fc0000000 draw=42 hash=a5e7b61cbf984383   <- the failure value
  ```

  This is the same-source-control trap in reverse (instrument trap 122): a campaign that has never
  shown its own apparatus reporting the class cannot distinguish a clean renderer from a blind
  harness. #2945.
- **The trigger row's "heavy GPU load" means concurrent SUBMISSION, not a busy GPU — the documented
  recipe barely moves the utilisation.** Three concurrent full-submit `gpu_replay` replays of the
  650 MB `s3537` capsule — the exact recipe recorded above as flipping a passing box within one
  round — take amdgpu's `gpu_busy_percent` from an idle 0–2% to about **5–9%**, with the CPU load
  average above 13. The replays are CPU-bound on capsule parsing and resource upload. So do not
  read "the GPU was loaded" into that row, and do not treat a low utilisation figure as evidence
  that a load arm failed to arm. `tools/determinism/replay_determinism.sh` records
  `gpu_busy_percent` on every row so the load condition measures its own premise instead of
  assuming it. #2945.
- **The SAME binary and the SAME capsule render a different picture today than on 2026-08-23, so no
  hash recorded in this section is a durable oracle.** `s3537.prgcap --draw 42` is written up above
  as a binary outcome between `9068fcf09de07383` (rendered) and `a5e7b61cbf984383` (drew nothing).
  Rebuilt from the exact commit the capsule carries (`rev=08c23efd`) and replayed on 2026-09-02,
  that binary returns **`ccc433ff6d980383`** — 5 of 5 on the host (Mesa 26.2.1) and 3 of 3 inside
  the build container (Mesa **26.1.4**, the version the original figure was taken on). Current
  `main` returns the same value. So the moved value is not prosper's doing and not a Mesa
  userspace-version difference; the other thing that moved on this box is the kernel, 7.1.5 →
  7.2.1. `ccc433ff6d980383` is a real render rather than a new failure mode: it differs from the
  untouched seed on 100% of sampled pixels, because that draw covers the whole 3840x2160 target.
  Caveat stated plainly: the rebuild is the same source at the same commit, not provably the same
  object bytes as the build that produced the 2026-08-23 figure. #2945.
- **RE-MEASURED 2026-09-03: 3,875 replays, one output hash per arm, and the verdict is still
  UNDECIDED rather than "fixed".** Four campaigns run through
  `tools/determinism/replay_determinism.sh`, unloaded and self-loaded blocks alternating, peers
  recorded per row:

  | campaign | binary / driver | subject | replays | distinct hashes | span |
  | --- | --- | --- | --- | --- | --- |
  | `new` | current `main`, host Mesa 26.2.1 | `s3537` whole submit / `--draw 42` | 707 / 706 | **1 / 1** | 1.75 h |
  | `old` | rebuilt `08c23efd`, host Mesa 26.2.1 | same | 701 / 701 | **1 / 1** | 1.75 h |
  | `container` | current `main`, container Mesa **26.1.4** | same | 296 / 295 | **1 / 1** | 1.13 h |
  | `stray` | current `main`, host | a seven-submit *Stray* `.prgbundle` | 469 | **1** | 1.73 h |

  Against the 2026-08-23 figure of **5 distinct hashes over 15 replays** of the same submit, i.e. a
  53% per-replay non-modal rate. With 0 deviations the 95% upper bound is 0.42% on the largest arm.
  GPU utilisation over the campaign was 0-37% (mean ~11%), and six full `ctest --no-tests=error -j4`
  runs taken during it were 345/345 six times over — the `-j4` load regime #2937 was found in.

  **Three qualifications, all of which narrow the claim rather than the verdict.** (1) The four
  campaigns ran CONCURRENTLY: their spans sum to 6.37 h inside a **1.75 h** wall-clock window, so
  the dataset samples 1.75 h four times over, not seven hours — and each campaign was load for the
  others, which means **no row labelled `no-selfload` was taken on an idle GPU**. Read the
  `gpu_pct` column, not the condition label. (2) The control figure was taken by a parse that read
  only the first of `vkprobe`'s per-pattern readback lines, and the correct pattern sorts before the
  common wrong ones, so the honest statement of it is **"no round's lexicographically-smallest
  indexed readback pattern was other than the correct one in 2,172 rounds"** — weaker than "the
  control fired 0 times", and conservative in the direction that matters (a missed fire pushes the
  verdict toward UNDECIDED, never toward DETERMINISTIC). The parse now aggregates every line; the
  corrected re-run is the row below. (3) Every subject row carried a real hash — 0 of 3,875 recorded
  `none` — which had to be checked rather than assumed, because the runner recorded exactly that for
  every bundle replay until #3270 taught it that `--bundle` prints no `output=` line.

  None of that moves the verdict, which is **UNDECIDED** either way: by the drift row at the top of
  this section the campaign has not shown that it met a window in which this class is expressible.
  That is what `replay_determinism_report.py` prints, and the sentence to quote is "no instance
  observed in 3,875 replays over a 1.75 h window", never "the renderer is deterministic". #2945.
- **The corrected control parse, re-run: still 0 fires, now properly counted.** The parse above read
  only the first of `vkprobe`'s per-pattern readback lines; aggregated across all of them and re-run
  on the same box under alternating self-load, **600 rounds, 0 failing, 600 usable replays and 1
  distinct hash** over a 0.48 h window (GPU busy 0-16%, mean ~7%). So the weaker claim and the
  strong one now agree, and "the control did not fire" is a properly counted statement rather than
  a statement about lexicographic order. It does not change the verdict: a control that does not
  fire leaves the campaign UNDECIDED however carefully it is counted, and **that is now the finding
  worth acting on — the failing regime cannot be reproduced on this box at all.** Until someone can
  make `vkprobe` fail here again, every determinism campaign on this issue returns UNDECIDED by
  construction, and the open question is what stopped the box entering that regime rather than
  whether prosper is clean. #2945.


### WITHDRAWN (2026-09-05, later the same day): #3374 does not reproduce on a fresh build — #3374

**Everything in the two subsections below was measured from ONE long-lived worktree's incremental
build directory, and none of it reproduces from a clean configure of the same commits.** Read them as
a record of an instrument failure, not as findings about the renderer. The re-measurement, on
`c4f57b1de` — two commits after the `d38e892cc` they were taken at, both of them documentation or an
off-by-default diagnostic, so the rendering code is the same code:

| path | result |
| --- | --- |
| live `prosper-app`, *New Joe & Mac* and *Alex Kidd*, played by the project owner | **correct**, no shear, both titles |
| live `tools/screenshot`, native 1920x1080, the guard's own route, ~150 gameplay frames opened | **correct**, every frame |
| the same, under concurrent GPU load (`gpu_busy_percent` 18.1% -> 28.2%, three emulator processes) | **correct**, every frame |
| offline `gpu_replay --bundle` on **RADV** — the exact command below | **correct**: level 1 complete, volcano, twin palms, "Ready?", the 1PL marker, both cavemen, full HUD |

And the "cross-implementation split" itself, re-measured on a capsule captured today, RADV against
lavapipe: **maximum channel delta 4 of 255, mean 1.01 over the 22.8% of pixels that differ at all,
and zero pixels differing by more than 32.** That is rounding between two rasterizers. The images are
visually identical. There is no split to explain.

**What went wrong, and the rule it re-teaches.** The failing measurements all came from
`.claude/worktrees/lane-fps-rung6`, a worktree reused across branch switches for days. This project
already records that exact trap — *five phantom Vulkan failures on master from a tree reused across
branch switches, all five passing on a fresh configure*. The clean runs above are from a worktree
configured from scratch. **The lane's build directory was destroyed during routine cleanup before
this was suspected**, so the mechanism is inferred from the pattern rather than demonstrated; that
loss is itself worth recording, because the build tree is evidence when its output is the finding.

Two claims that outran their evidence and are withdrawn with it: that the five titles' shear is a
RADV defect (#3375), and that dropped indexed draws are its mechanism (#3377). What survives
independently is the `tools/vkprobe` measurement that **RADV returns varying output for an unchanged
indexed draw under load while lavapipe does not** — that was taken with prosper-free, hand-written
shaders in a separate binary, so a stale prosper build cannot explain it. It is a real observation
about this driver and grounds for an upstream report. It is **not** an explanation of a picture that
no longer exists.

**Before believing any renderer defect measured from a long-lived worktree, re-measure it from a
clean configure.** A stale object file produces a defect that survives every A/B you can run inside
that tree, because every arm inherits it.

### (WITHDRAWN — see above) A cross-implementation split, with a lavapipe positive control — and the four titles it covered (2026-09-05) — #3374

The row above closes with *"the failing regime cannot be reproduced on this box at all"*. It can, on a
different subject, and this one does not need a regime: **one captured frame, replayed offline, is
wrong on RADV every single time and right on lavapipe every single time.**

```bash
# Capture (headless, no keyboard): the guard's own route, native scale, trigger the grab at t=90 s.
PROSPER_RENDER=1 PROSPER_RENDER_SCALE=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_PAD_SCRIPT=@prosper/scripts/joe-mac/reach-gameplay.pad \
PROSPER_CAPTURE_BUNDLE=<work>/frame.prgbundle \
PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE=<work>/capture.ready \
    screenshot <DUMP_ROOT>/PPSA02801-app0 --seconds 5 --count 25   # touch the trigger at t=90 s

gpu_replay --bundle <work>/frame.prgbundle --bundle-extract-submit <N> <work>/submit.prgcap

# THE A/B. Same binary, same capsule, same 88 operations. Only the ICD differs.
gpu_replay <work>/submit.prgcap out_radv.bmp                          # sheared
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
gpu_replay <work>/submit.prgcap out_lvp.bmp                           # the level, correct
```

**Measured on `d38e892cc`, Linux / AMD Radeon 8060S (RADV STRIX_HALO), Mesa 26.1.4.** *New Joe & Mac*
(`PPSA02801`): **25 of 25** RADV replays byte-identical to each other (`0ab3cfbe25…`), taken across
both a heavily loaded window and a quiet one after a two-minute cool-down, and the lavapipe replay of
the same capsule renders level 1 complete — twin palms, volcano, flower bank, Joe, the HUD with its
score, health bar and lives counter. Reproduced the same way on *Asterix & Obelix: Slap Them All!*
(`PPSA08576`), *Rugrats: Adventure in Gameland* (`PPSA23396`) and — a 3D scene at native 3840x2160,
174 operations — *Summer Sports Games* (`PPSA03416`), whose javelin event comes back with its hoarding,
takeoff zone, throw reticle and packed stands intact.

**The RADV result does not move across driver versions.** The container runs Mesa **26.1.4** and the host
Mesa **26.2.1**; the same binary on the same capsule returns the **byte-identical** broken frame on both
(`0ab3cfbe25…`, 3 of 3 on the host). So a Mesa upgrade does not fix it, and — more usefully — a
byte-identical result across two driver versions argues against a race and for something stable that both
RADV versions resolve the same way.

**What that settles for #3374, and what it deliberately does NOT.** Four of the five rung-6 Unity titles
whose scene art arrives through large sheared triangles are **one defect rather than four separate
ones**: the identical capsule
carrying the identical decoded draws, indices, descriptors, uploads and recompiled SPIR-V produces the
correct scene when a different Vulkan implementation executes it. Every prosper-side stage was
independently checked on the RADV run and is clean — `PROSPER_BUFLOG` source words, `PROSPER_BUFVERIFY`
(346 buffers, 0 mismatched), `PROSPER_BUFFER_ECHO` descriptor set/offset/range, and `PROSPER_INDEX_ECHO`
(below) — and a full replay under `VK_LAYER_KHRONOS_validation`, proved loaded with `VK_LOADER_DEBUG=layer`,
reports zero findings.

**It does not assign the fault, and an earlier revision of this section did — read instrument trap 38.**
A cross-implementation split *localises the disagreement*; it does not say who is wrong, and trap 38's own
worked example is a lavapipe-versus-RADV split that turned out to be prosper's undefined behaviour. Both
implementations execute **prosper's own recompiled SPIR-V**, so an implementation-defined or undefined
construct in it produces exactly this picture, and the byte-identical agreement of two RADV versions is
consistent with that reading as much as with a driver defect. What IS established is narrower and still
worth a lot: the guest data, the PM4/index decode, resource realization, descriptors and uploads are not
the cause, because they are the same bytes in the run that renders correctly. **The open fork is a RADV
defect versus UB in the recompiled vertex stage** — **and the subsection below settles it: the UB side is
refuted, and not by enumeration.** What follows here is the state of the evidence before that was
measured; read it for the falsification list, not for the verdict. Note also
that validation's zero findings cover neither synchronization hazards (not enabled on that run) nor an
out-of-range read that robustness resolves.

**The fifth title, *Evergate* (`PPSA01885`), is NOT covered by this** — which is why the heading says four.
Its dominant symptom, a blown-out white/orange frame, **reproduces on lavapipe too**, so that part is
prosper's; thin diagonal streaks over it are absent on lavapipe and do look like this defect, so it may
carry both. Caveat on even that: its guard accelerates the opening (`PROSPER_RENDER_EVERY=500` for 90 s)
and a native re-aim drops that, moving the timeline, so the captured frame is not confirmed to be the
reviewed tutorial room. Treat Evergate as **unresolved**, not as excluded.

**It is deterministic, which is why it is filed HERE and not as more of #2945.** #2945 is stochastic
and load-triggered; this is 25 of 25 in both regimes, and no lever moves it: `PROSPER_NO_BACKEND_BUFFER_ARENA`,
`PROSPER_NO_BACKEND_BUFFER_POOL` (together and separately), `PROSPER_NO_BACKEND_RESOURCE_SHARE`,
`PROSPER_NO_BACKEND_PIPELINE_CACHE`, `PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE`, `PROSPER_NO_MEMORY_POOL`,
`PROSPER_BACKEND_BUFFER_ARENA_KB=262144`, and `RADV_DEBUG=syncshaders | zerovram | nocache`,
`RADV_PERFTEST=nosam` all return the byte-identical broken frame.

**Three RADV arms are VOID rather than negative, and this qualifies #2945's row too.** `RADV_DEBUG=llvm`,
`nongg` and `nonggc` also return the byte-identical frame, but they change nothing to return it: with
`shaderstats,nocache` forcing a fresh compile in both arms, the SGPR / VGPR / code-size lines are
**md5-identical** to the default, and the driver's own output contains **zero** `ngg` tokens — so there
is no NGG to disable and this Mesa (26.1.4 / 26.2.1) does not select an LLVM backend. #2945's list reads
*"`RADV_DEBUG=llvm` failing rules out ACO; `nongg` failing rules out NGG"*; on this Mesa neither
inference holds, because neither lever moves. Pair any RADV compile lever with `shaderstats,nocache` and
show the stats differ before quoting a null from it. Its character matches
#2937 — deterministic — rather than #2945's stochastic dropout. (#3371's own tests measured **stochastic**
here, 3-8 failures of 24 runs, which matches #3371's own title and is the opposite of what the #2937 row
above records; the title capsule is the deterministic subject, the tests are not.) Whether these share
a cause is open — but #3371's own next step, a missing host-upload barrier in prosper, is materially
undercut by the subsection below: `tools/vkprobe` reproduces a dropout of the same signature with no
prosper code in the process, fence-waiting before it touches mapped memory and reading coverage back
through a real
`vkCmdCopyImageToBuffer`. A barrier prosper is missing cannot explain a failure in a program that does not
contain prosper.

**`PROSPER_INDEX_ECHO=1`** (new, `tests/fixtures/render_runner.h`) is what retired **prosper's half** of
the index family at title scale. It reads the index bytes back from the host-visible memory at exactly the
`(buffer, offset)` pair about to be bound and compares them against the indices prosper decoded, printing
`icount`, `vcount`, `vertexOffset`, `instanceCount`, arena/dedicated, `want_max` and `got_max`. On the
Joe & Mac frame: 85 indexed draws, `mismatched=0` on every one, `got_max < vcount` on every one, `voff=0`,
`inst=1` — and the last two matter, because the in-range verdict is `got_max + vertexOffset < vcount` and
only `voff=0` makes the printed comparison the whole of it. `PROSPER_BUFLOG` and `PROSPER_BUFVERIFY` look
at STORAGE buffers and cannot see any of it, and an out-of-range storage read is `robustBufferAccess`
rather than a validation error.
**Scope, because the obvious reading is stronger than the instrument.** It reads the **host** mapping, not
device memory, so a clean result retires *"prosper computed or placed the wrong indices"* and says
**nothing about what the GPU read back**; `mismatched` compares the arena against the same vector the
memcpy sourced from, so it cannot express a wrong decode either. The device-side question is
`PROSPER_BUFFER_ECHO`'s — which copies index slices back through the GPU and never fires, because its
16-slice budget is spent on storage buffers first (#3376). That gap is why the host-side reading was
needed.

**Do not use `PROSPER_GEOM_PROBE` on these capsules.** Its per-draw verdicts contradict the pixels —
see instrument trap 266. The `--draw-steps` per-operation contribution is the instrument that held.


### (WITHDRAWN as an explanation of #3374 — see above) RADV returns varying output for an unchanged indexed draw under load (2026-09-05)

The subsection above left an open fork — a RADV defect against implementation-defined or undefined
behaviour in prosper's recompiled vertex stage — and named auditing that SPIR-V as the next step. **That
audit is not the thing that answers it.** Enumerating UB classes can never finish, and two measurements
refute the whole side of the fork regardless of what an enumeration would have found.

**1. The variance argument, and what it does and does not reach.** *Implementation-defined* behaviour is a
choice the compiler makes once, so it yields a consistent wrong answer for a given compilation and cannot
make one implementation give several different answers to identical input. **That reasoning does NOT extend
to undefined behaviour**, and this subsection asserted the broader claim until review caught it: reading an
undefined value, or an out-of-range read where robustness does not resolve it, returns whatever the memory
held, which varies with unrelated GPU activity — so load-dependent variance is that class's *signature*
rather than evidence against it, and the paragraph above already names those exact two classes as the ones
validation did not cover. The variance below therefore refutes the implementation-defined half on its own;
what retires the undefined half is an arm, not an argument. `tools/vkprobe`'s `no_ssbo_vs` declares the same
storage buffer at the same set 0 binding 3 and **never reads it**, deriving position from `gl_VertexIndex`
alone — and it fails on this driver anyway, **142 of 400 runs** (`tools/vkprobe/README.md`). Do not read
that against the 54 of 400 the storage-buffer module scored: those are different campaigns in different
windows, and that README disclaims the cross-campaign comparison explicitly. The single figure carries the
argument on its own. And the arm reaches further than a storage-buffer load: `no_ssbo_vs`'s only `OpLoad`
is of the `VertexIndex` builtin and its only `OpAccessChain` is a constant index into its own output block
— no function-local variables, no `OpUndef`, no branches, no dynamic indexing, no division, position built
from `OpSelect` over constants. There is very little undefined behaviour that module is *capable* of, so it
retires module UB as a class rather than the storage-read subset of it. Interleaved run-by-run, same
capsule, same draw (`--draw 12`, a four-vertex six-index quad), same binary, under induced GPU load,
**n = 20 per arm**:

| implementation | distinct outputs in 20 runs |
| --- | --- |
| RADV | **4** — including the untouched RTT-seed hash (the draw contributed *nothing*) **10 times** |
| lavapipe | **1** |

Arms alternate run by run because this box's failure rate drifts machine-wide over minutes (trap 223);
measured as arm-A-then-arm-B the same comparison is void. Quiet-window controls show why: the identical
RADV command measured 4 distinct outputs in one window and 1 in 10 runs in another.

**2. The prosper-free control. The failure does not need prosper's SPIR-V.** `tools/vkprobe` driving the
hand-written `minimal_ssbo_vs` + `minimal_green_fs` — no prosper code and no prosper-generated module in
the process — 12 processes x 25 iterations per arm, interleaved process-by-process, under the same load:

| implementation | indexed-arm iterations that drew nothing |
| --- | --- |
| RADV | **28 / 300** |
| lavapipe | **0 / 300** |

Together these retire the UB side: a module prosper did not write fails the same way, and the failure
varies run to run on one implementation while the other is exact. The UB enumeration on #3374 (no
`OpUndef`, no uninitialised variables, no dynamic vector extracts, no division, and the position slice's
only shift by a constant) stands as a record but is **no longer load-bearing**.

**3. How far the dropouts reach into the picture.** They are present at full-frame scale and they are
large; whether they *are* the shear is the open part, and the paragraph after the table is the one that
settles what this measurement does and does not license. Per-operation contribution from
`--draw-steps --draw-steps-every 1` on the Joe & Mac capsule, both implementations, 86 comparable
operations:

| | draws contributing pixels |
| --- | --- |
| lavapipe | **84 of 86** |
| RADV | **67 of 86** |

**19 draws contribute on lavapipe and nothing whatsoever on RADV** (2,461 to 23,286 pixels each), and the
set of 19 is **identical across two independent runs**. The reverse direction is smaller and was initially
unremarked: 84 - 19 = 65 draws contribute on both, so **2 draws contribute on RADV and nothing on
lavapipe**.

**This establishes that draws are lost, NOT that the loss is the shear** — the stronger claim stood here
until review rejected it, and the distinction is worth keeping because the two failure pictures are
different. Deleting a draw leaves a **hole**; #3374's own row for `PPSA02801` describes the opposite, art
"present and recognisable" with *positive* sheared shards smeared across the frame. If those shards are in
the RADV replay they are painted by the 67 draws that do contribute, not by the 19 that do not. The
arithmetic bounds it the same way: 19 draws of at most 23,286 pixels is at most 442,434 of a 2,073,600-pixel
frame, **under 22%**, against a single full-screen quad in the same capsule that paints 2,013,079 pixels on
its own. One further limit on the instrument: `--draw-steps` reports each operation's *delta* against the
prefix before it, so "contributes nothing" is a statement about that draw in that prefix and not an
independent per-draw measurement of "was dropped".

So: dropped indexed draws are established, and they may be *part of* what produces the shear. Whether
they account for the picture is open, and the cheap way to close it is already to hand — compare the RADV
replay frame against the live shear, and check whether the shards coincide with the survivors or with the
missing set.

**Why the full frame looks deterministic while one draw does not.** The 25-of-25 byte-identical full-frame
result in the subsection above is real but is a property of the **saturated regime**, not of the mechanism:
a full frame is its own GPU load, and the same nineteen draws lose on every run measured. Variance falls
as the prefix grows — `--draw 12` gives 4 distinct outputs in 10 runs, `--draw 0:12`,
`--draw 0:30` and `--draw 0:60` give 2 (9 of 10 modal), the full frame gives 1 of 10. **So this is #2945's
family after all, seen at saturation** — the earlier reading of "deterministic, therefore not #2945" was
right about the measurement and wrong about what it implied. Read the two subsections together.

**What is still NOT claimed.** That RADV is non-conformant in its *rendering* of these draws is not
established by any of this; what is established is that it is non-deterministic on deterministic input and
that prosper's SPIR-V is not required for that. Those are the grounds for an upstream report — a
reproduction that varies run to run, on a module the reporter did not generate, against a second
implementation that does not vary — and they are stronger than the cross-implementation split alone,
which trap 38 correctly refuses to accept as an attribution.


## Recommended implementation order

1. **Real unified memory.** Make GPU allocations CPU/GPU-VA *aliased*: when the guest maps direct
   memory, back it so the GPU VA it later uses equals a valid CPU address (single physical page seen
   at both). This replaces the lazy zeroed-page placeholder with real, coherent memory — the
   prerequisite for everything else. Trace the libSceAgc memory-map calls to learn the GPU-VA scheme.
2. **libSceVideoOut swapchain.** Real window/offscreen images + a Vulkan swapchain (or headless
   render target); `SubmitFlip` presents the current buffer.
3. **libSceAgc → Vulkan.** Decode the AGC command buffers (draw/dispatch/state) and translate to
   Vulkan command buffers. This is the largest piece.
4. **RDNA2 shader recompiler.** Translate the game's GCN/RDNA2 shader ISA (or the AGC shader blobs)
   to SPIR-V. The other large piece.
5. **libSceAmpr audio.** Mix/output via a host audio backend (or a null sink first).

## Sonic Frontiers shader validation (2026-09-07, #3416 / #3417)

The captured position-only guest vertex program contains no PARAM export, while its paired fragment
program consumes attribute zero. Sticky input controls select PARAM0; they do not request
`DEFAULT_VAL`. The vertex recompiler now declares every proven consumed location even when the guest
never writes it. This satisfies Vulkan's interface contract without inserting an invented default
or overwriting a real export. Unknown consumption does not expand the interface from stale controls.

The renderer and standalone compute device also enable `shaderImageGatherExtended` when supported,
matching the recompiler's existing dynamic-offset gather capability (#3417). A 105-second baseline
reported the missing gather feature and missing vertex-output VUIDs. After both fixes and #3415, a
45-second windowed intro run presented 451 frames with the validation messenger active and **zero
warnings or errors**. This covers that intro path, not the entire game's shader set.

`vertex_output_budget` checks that a consumed-but-unexported parameter has a declaration and no
fabricated store, while a real export still writes it. `spv_validate` validates the new module shape;
`interp_render` verifies real interpolants and explicit defaults still render correctly (3/3 tests).

## Testing

Everything above must stay behind programmatic checks (agentic-first). Current suite (7 tests):
module parse, NID hashing, trap, boot (asserts the GC stop-the-world runs), setjmp, HLE registration,
SceKernelStat layout. Add: unified-memory aliasing test, AGC command-decode unit tests, a boot check
that asserts the first `SubmitFlip`.
