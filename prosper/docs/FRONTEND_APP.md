# prosper-app — the OS-integration frontend (design)

**Status:** **the app runs the game in a window, with audio out and controller in** (issue #164).
`prosper-app --dump <app0>` (built with `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON`)
boots the title, composites its GPU submits to an SDL3 window on the desktop, routes `sceAudioOut` to
the host via SDL3, and feeds a host controller into `libScePad`. P0 (present) + P0b (shared boot) +
the shared renderer (#177) + P1 (audio) + P2 (controllers) are landed; P3 is polish.

## What it is

`prosper-app` is a **separate frontend binary** that turns the headless prosper core into a real
desktop app: a window showing the game, audio out, controller in, and close-to-quit. It exists so
we can watch prosper run on a real GPU with sound — a human-perspective smoke test — **without
compromising the headless-first core**, which stays the source of truth for agentic/CI verification.

The frontend runs both as a Linux application (including through WSLg) and as a native MinGW Windows
application. Native Windows uses SDL3's Win32/WASAPI/XInput-HIDAPI backends and a Vulkan SDK; it boots
the same guest and renderer directly, without WSL. See `WINDOWS_PORT_HANDOFF.md` for the current build
and routed screenshot validation commands.

## First principle: the dependency arrow points one way

```
prosper_core   (static lib — ZERO windowing/surface/OS-device deps; headless; CI-tested)
     ▲ links
     │      audio_set_sink() · pad_set_backend() · present frame lease · run/request_stop()
frontends/prosper-app   (new binary — owns ALL OS integration, via SDL3)
```

The core never links SDL, a window system, or a presentation surface. The frontend depends on the
core; the core never depends on the frontend. Delete `frontends/` and CI is unaffected. This makes
"separate" and "headless-first" structural guarantees, not conventions to remember.

## The seam contract (three of four already exist)

The core already exposes exactly the injection points a frontend needs. The frontend *supplies*
implementations; the core calls them.

### 1. Audio — `AudioSink` (exists, `src/hle/audio/audio.hpp`)
```cpp
struct AudioSink {
    virtual bool open(int port, const AudioPortInfo& info);        // configure a port
    virtual void output(int port, const void* pcm, int frames) = 0;// one interleaved-PCM grain
    virtual void set_volume(int port, uint32_t mask, const int* vols);
    virtual void close(int port);
};
void audio_set_sink(AudioSink*);   // nullptr restores the built-in silent/real-time sink
```
Frontend supplies an **SDL3 audio sink** (queue each `output()` grain to an SDL audio stream). The
`feat/audio-sdl3` branch is already most of this — coordinate with it, don't duplicate.

### 2. Controllers — `PadBackend` (exists, `src/input/pad.hpp`)
```cpp
struct PadBackend {
    virtual bool poll(int index, HostPadState& out) = 0;  // called on the guest's input thread; MUST be thread-safe
};
void pad_set_backend(PadBackend*);   // nullptr restores the neutral (no-device) backend
```
Frontend supplies an **SDL3 `GameController` backend**, mapping SDL buttons/axes → `HostPadState`.
The pad header already documents frontends installing this from the harness.

### 3. Video — rendered-frame ownership (exists, `src/gpu/present/videoout_present.hpp`)
```cpp
struct PresentFrameLease {
    uint64_t frame_seq;
    uint32_t width, height;
    std::shared_ptr<const std::vector<uint8_t>> rgba;
};
bool present_acquire_rendered_frame(PresentFrameLease& out);
size_t present_readback(void* dst, size_t dst_cap); // compatibility/capture copy
```
The renderer's selected scanout, the present layer, and `prosper-app` share one immutable CPU-pixel
allocation. Acquiring a lease does not copy the image, and its storage stays valid if the renderer
publishes a newer frame while the app uploads the old one. Screenshot, capture, replay, and legacy
readback APIs deliberately retain copying semantics. The frontend remains a pure *consumer* of finished
frames and polls `present_count()` for guest flip pacing.

### 4. Lifecycle — stop request exists; guest consumption remains
The frontend has the minimal, headless-agnostic control:
```cpp
// in the run harness / a small host lifecycle header
void prosper_request_stop();     // idempotent; sets a flag the guest run-loop checks
bool prosper_stop_requested();
```
`boot_trace` keeps its fixed-budget behavior for CI and the frontend calls `prosper_request_stop()`
on window close. `run_entry` does not consume the flag yet, so the current app cannot safely join a
booted guest. It flushes logs and calls `std::_Exit` instead; returning from `main` after detaching the
guest caused a reproducible Windows access violation when static teardown raced live guest threads.
The remaining lifecycle work is a guest flip-boundary check followed by a real join and normal
teardown (#352).

**A second, non-obvious cost of that missing join, and the partial answer to it (#3225).** `_Exit`
becomes `exit_group()`, and a thread that is inside a syscall at that moment cannot be torn down
until it returns from the kernel. When the syscall is an amdgpu command submission the task parks in
`__drm_exec_lock_obj` on a GEM reservation nobody will now release: the process becomes an unreapable
zombie (`SIGKILL` is a no-op) and the compositor's own DRM work queues behind it — a frozen desktop
that only a root-forced GPU reset clears. Because there is no ordering on this path at all, that
in-flight submission is abandoned on *every* close; that it usually costs nothing is luck about where
the guest thread happened to be. `src/host/platform/gpu_submit_gate.hpp` bounds it: the frontend
closes a gate that every guest-thread submission enters, so new submissions are refused
(`VK_ERROR_DEVICE_LOST`) and a bounded drain can wait for the ones already inside. It is a
mitigation, not the fix — a thread that never returns still expires the drain — and the fix remains
the join above.

## Where the dump path comes from (#1469)

`boot_program()` needs an app0 directory. Argv supplies it for every agent route, snapshot run, and CI
check, and that remains the primary path — it runs before the window exists, so the guest is already
up by the time the swapchain is created. Nothing about it changed when interactive sources were added.

A released build also reaches people who never type a command line, so two more sources feed the same
`start_guest()`:

```
argv  ──────────────────────────────┐
window drop (SDL_EVENT_DROP_FILE) ──┼─> resolve_app0_root ─> decide_open_action ─> start_guest
host folder picker (SDL dialog) ────┘         (game_path.hpp — pure, unit-tested)
```

`game_path.hpp` holds the whole decision surface behind an injected filesystem probe, so path
resolution and the open/reject/relaunch choice are covered by `test_game_path.cpp` in ordinary CI
without a window, a picker, or a real dump — the same pure-seam pattern as `present_mode.hpp` and
`window_controls.hpp`.

Two constraints shape the design:

- **A PS5 title is a directory**, so the host dialog is `SDL_ShowOpenFolderDialog`. SDL ships it; the
  build simply enables `SDL_DIALOG` when `PROSPER_APP` is on. The audio- and pad-only frontends have
  no window to parent a dialog to and keep it out.
- **One boot ATTEMPT per process.** `run_entry` never observes `prosper_request_stop()`, so the app
  exits rather than joining a guest (see the lifecycle seam above, and #352). Opening a title after
  the attempt is spent starts a fresh process — `relaunch_with_dump()` appends `--dump <new title>`
  to this run's own arguments, which selects the new game because `--dump` is last-wins. When the
  cooperative stop lands, that call site is the single place that becomes an in-process teardown
  plus `start_guest()`.

  The attempt latches *before* `boot_program()` runs, not on its success, and `start_guest()` enforces
  that itself rather than trusting callers. A failed boot is exactly as unrepeatable as a successful
  one: `link_program` appends into the `Program` it is given (`src/loader/linker.cpp`) and never
  resets it, and the surrounding setup — `register_builtin_hle`, `set_module_exports`,
  `set_app0_root`, `install_trap_handler`, the live renderer, the host backends — is one-shot. A
  second in-process call would link the new title behind the failed one's modules and leave `imgs[0]`,
  the image the guest thread actually enters, pointing at the title that failed.

The idle window paints a flat colour through the ordinary `present_frame` path rather than leaving
undefined swapchain contents on screen, and stops as soon as `present_frame_seq()` shows anything has
been published. Those frames do not count toward `--frames`, which asserts real content.

## The game library (#1471)

Stage 1 adds the data behind a library view — and the app's first **persistent user state**.

```
--games-dir  ─┐
PROSPER_GAMES_DIR ─┼─> resolve_games_dir ─> scan_game_library ─> [GameEntry]  ─> --list-games
config file  ─┘        (app_config.hpp)      (game_library.hpp)                  and, later, a grid
```

Both headers are pure with injected filesystem IO, so the scan, the param.json metadata reading, and the
settings precedence are unit-tested in the default core build — no window, no ImGui, no real dump. They
live under `frontends/`, header-only, and `prosper_core` is untouched: the dependency arrow above is
unchanged.

Design points that are deliberate rather than incidental:

- **Precedence is `--games-dir` > `PROSPER_GAMES_DIR` > persisted setting.** The persisted value exists
  only so a GUI user is not asked for a folder every launch; it must never override a command line. This
  keeps the project's existing convention (a `PROSPER_*` override in front of a default) authoritative.
- **Persistence is only ever explicit** (`--set-games-dir`). Nothing infers a library location from a
  folder the user happened to open — that inference is wrong as often as right, and being wrong points
  the library somewhere the user never chose.
- **Unknown settings are carried through a rewrite.** `--set-games-dir` rewrites the whole file, so
  tolerating unknown keys on *read* is not enough: without preserving them, a release build would delete
  whatever a newer build had stored.
- **`scan_game_library` reuses `resolve_app0_root()`** from the interactive-open path above, so all three
  entry points agree on what a title is. The listing is "what a drop or the picker would accept", which
  is not identical to "will boot" — the gate accepts `sce_sys/param.json` alone.
- **`--list-games` is the headless contract:** tab-separated records on stdout, commentary on stderr,
  exit 0 / 1 (empty) / 2 (unset or not a directory), and it returns before any window or Vulkan exists.

`param.json` name selection is worth knowing about, because it was silently wrong before this work. The
display name is the entry for the dump's own `defaultLanguage`, and the language object is located by
*shape* — `"lang"` followed by `:` then `{` — anchored inside `localizedParameters` and bounded to that
object. The previous logic required the object to appear textually before the `"defaultLanguage"` line,
which fails on dumps that declare the default first: the first match is then defaultLanguage's own
*value*, and the fallback returned whichever language came first in the file. The Plucky Squire
(PPSA15319) read as "DER KÜHNE KNAPPE". The same function drives the window title, so that was wrong
there too.

### Stage 2: the view

Dear ImGui (`third_party/imgui`, MIT) and stb_image (`third_party/stb`, public domain) are vendored
**frontend-only** — `prosper_core` links neither, so the arrow above is unchanged and deleting
`frontends/` still leaves CI unaffected. `library_ui.{hpp,cpp}` draws the grid; `library_nav.hpp` holds
the selection rules and is pure and unit-tested, so the grid's behaviour is covered without a window.

Decisions worth knowing:

- **The library renders through the app's existing device and swapchain**, adding only a render pass and
  per-image framebuffers. No second device, no second window.
- **It owns its own command buffer, semaphores and fence** and does its own acquire/present, rather than
  threading a second mode through `present_frame`. The two never run at once — the library *is* the idle
  state and is torn down the instant a guest boots (one game per launch, #352) — so keeping the game
  present path byte-identical was worth more than sharing sync objects.
- **The swapchain is now created with `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` as well as
  `TRANSFER_DST`.** The game path only ever *blits* into swapchain images, so the attachment bit was
  absent; without it every framebuffer built from those images is invalid and each draw is silently
  dropped. The render pass still appears to clear, which makes it look like a UI bug rather than a
  swapchain one — Vulkan validation names it immediately
  (`VUID-VkFramebufferCreateInfo-pAttachments-00877`).
- **ImGui's own keyboard/gamepad nav is deliberately off.** Selection is driven by `library_nav.hpp`, and
  running both means the arrow keys move a widget focus as well as the selection, and Enter activates
  whatever widget that focus landed on instead of launching the highlighted game. A consequence worth
  knowing: turning off `NavEnableGamepad` also silences the SDL3 backend's gamepad feed, so
  `ImGuiKey_Gamepad*` is never set. Controller navigation therefore needs the pad read directly *and*
  `SDL_INIT_GAMEPAD` initialized while the library is alive — neither is true today (the pad backend
  initializes it inside `start_guest`), so the library is keyboard/mouse only.
- **`CheckVkResultFn` is always set.** ImGui's Vulkan backend silently ignores every `VkResult`
  otherwise, so a failed pipeline or descriptor allocation would present only as a window that draws
  nothing.
- **Failure is never fatal.** If the UI cannot be created the app logs it and falls back to the flat idle
  colour; a driver that cannot host the library costs the user a library, not the app.

## The Vulkan-context decision: two contexts, one shared CPU frame

**The frontend owns its own Vulkan context (instance + device + swapchain on the window surface).
The core keeps its existing headless render device. Frames cross the boundary through immutable shared
CPU storage acquired with `present_acquire_rendered_frame`.** Rationale:

- The core's render device stays **surface-free and unchanged** — no swapchain dependency leaks into
  the renderer, no `#ifdef`, headless + CI identical.
- The seam already emits CPU pixels, so the frontend is a pure consumer — the cleanest boundary, and
  the same seam becomes a shared-memory frame ring if the two are ever split into separate processes.
- Cost = the renderer's required GPU readback plus one frontend upload per frame. There are no
  renderer-cache-to-return, return-to-present, or present-to-app copies. The Messenger is 2D at 1080p
  (~8 MB/frame), but native
  Windows measurements showed that memory-type selection matters: uncached host-visible readback cost
  roughly 570 ms per frame on an RTX 4090, versus about 1.8 ms with `HOST_CACHED` memory.

**When to revisit:** direct image presentation remains worthwhile after the renderer stops rebuilding
pipelines and image/buffer objects per batch. At that point, unify on a single frontend-provided device
or share the render target via `VK_KHR_external_memory` (zero-copy), while retaining `present_readback`
for headless tests and screenshot tooling.

**Update (#1270, landed 2026-07-24):** the "unify on a single device" half of this landed for real game
boots, ahead of the pipeline/resource-caching precondition this note originally set — `prosper-app`
adopts the *renderer's* device (rather than a frontend-provided one) and GPU-blits its front-buffer image
straight to the swapchain by default, with `PROSPER_APP_GPU_PRESENT=0` as the explicit opt-out. The
two-context design above is still real and still runs: it is what `--test-pattern`, any failed device
adoption, and `tools/screenshot` all use today. `present_readback` was retained exactly as this note
anticipated, for headless tests and screenshot tooling.

## Present loop (sketch)

```
init:  SDL_Init(VIDEO|AUDIO|GAMECONTROLLER); create window;
       create VkInstance(+VK_KHR_surface, win32/wayland), VkSurfaceKHR, pick device, swapchain,
       one staging buffer + one sampled VkImage sized to present_width()×present_height();
       audio_set_sink(&sdl_sink); pad_set_backend(&sdl_pads);
       start the guest run-loop on its own thread.

frame: SDL_PollEvent → on SDL_QUIT / window-close: prosper_request_stop(); break;
       Pause / F10 toggles a guest flip-boundary wait and the SDL audio device;
       F11 / Alt+Enter toggles borderless desktop fullscreen;
       pixel-size change → recreate the Vulkan swapchain before the next present;
       if present_count() advanced since last shown:
           lease = present_acquire_rendered_frame();
           upload lease.rgba → VkImage; blit/scale VkImage → acquired swapchain image; vkQueuePresentKHR;
       else: small sleep / wait on a frame condvar to avoid spinning.

quit target: prosper_request_stop(); join guest thread; destroy swapchain/device/window; SDL_Quit().
quit today:  prosper_request_stop(); flush logs; direct process exit while the guest is still live.
```
Handle swapchain resize (`present_width/height` change or window resize → recreate). Vsync via
`VK_PRESENT_MODE_FIFO`.

## Threading

- **Guest run-loop thread**: drives the guest (as `boot_trace` does), renders into scanout on flip.
- **Main/UI thread**: SDL event pump + swapchain present. It holds an immutable shared frame lease
  while copying to its Vulkan staging allocation; publication only swaps shared ownership under the
  present mutex.
- Audio/pad callbacks run on whatever thread the core calls them from; the SDL sink/backend must be
  thread-safe (the pad header already requires it).

Target shutdown ordering: `request_stop` → guest thread observes the flag at its loop boundary and
returns → join → tear down GPU/window. Until that check exists, direct process exit deliberately
skips frontend/HLE static teardown so it cannot race the detached guest — preceded, since #3225, by
`gpu_submit_gate_begin_shutdown()` → `gpu_submit_gate_drain()` → `vkDeviceWaitIdle`, which bounds the
window in which that exit can kill a thread inside a GPU submission. The drain is also what makes the
`vkDeviceWaitIdle` legal: it is the external synchronization of every `VkQueue` that call requires, so
a drain that TIMES OUT deliberately skips the wait rather than racing a live submit. `tools/screenshot`
runs the same two steps before its own `_exit` (`screenshot.cpp`), for the same reason.

## Target / build layout

- New dir `frontends/prosper-app/` with its own `main.cpp` + the SDL sink/backend.
- Its own CMake target linking `prosper_core`, gated by `-DPROSPER_APP=ON` (**default OFF**), and
  on `find_package(SDL3)` + `find_package(Vulkan)` being present. The core and all existing tests
  build and pass with the frontend absent — never a core/CI dependency.
- SDL3 unifies window + audio + gamepad in one dep, matching the existing SDL3 direction
  (`feat/audio-sdl3`, the gamepad frontend).

## Deployment on Windows

For native Windows, configure MinGW with `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON
-DPROSPER_PAD_SDL3=ON`, build `prosper-app.exe`, and verify the window/swapchain first with
`--test-pattern --frames 120`. `scripts/run-windows.ps1` performs that full configure/build/run path:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

The complete manual PowerShell recipe is in `WINDOWS_PORT_HANDOFF.md`.

### Guest time during synchronous GPU work

Prosper currently realizes and executes an AGC submit synchronously on the guest submitter. The
guest monotonic clock discounts host GPU time beyond the refresh interval represented by that
submit's completed VideoOut flips. This keeps shader compilation, resource conversion, execution,
and readback stalls from becoming multi-second frame deltas that skip time-based guest states. Wall
clock/RTC surfaces remain tied to host time, and monotonic time continues normally outside the
backend scope, including while media or wait loops prepare the next frame. This is intentionally
narrower than the opt-in `PROSPER_DET_CLOCK`, which holds monotonic time between every pair of flips.

Set `PROSPER_NO_GPU_TIME_COMPENSATION=1` only for a diagnostic A/B against the old behavior.

### Live renderer performance diagnostics

Set `PROSPER_RENDER_TIMING=1` before launching to print aggregate timings every 25 operations:

```powershell
$env:PROSPER_RENDER_TIMING = '1'
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0 -NoBuild
```

The output separates guest-state realization, ordered graphics/compute execution, CPU resource
decode/detile, Vulkan target and pipeline setup, upload/recording, GPU fence wait, readback, cleanup,
and frontend publication. Cumulative `[render-timing]` lines show the whole run; `[render-window]`
lines show only the latest 25 submits/calls so a scene transition is immediately visible. The
`backend-submit` lines sum every Vulkan call made by those exact ordered submits, rather than mixing
an independent 25-call backend window across submit boundaries. Compare `measured` with `detail` and
its `other` remainder to verify that the subphase attribution covers the frontend's backend wall time.
The core window also reports graphics-shader cache hits, misses, bypasses, and actual miss compilation time.

`backend-submit draw_setup.resources` is sub-attributed on its own line:

```text
[render-window] backend-submit resources avg_ms: texture=.. (upload=.. bind=.. lookup=..)
                buffer=.. (acquire=.. copy=..) descriptor=.. other=..
```

Read it before treating `resources` as descriptor cost. The interval it measures spans the whole
per-draw resource block, which also builds every **texture upload** (image and memory creation plus
the staging memcpy) and every **storage-buffer upload**, so the term can be dominated by pixel and
vertex bytes while its name suggests descriptor bookkeeping. `texture` and `buffer` cover the whole
per-resource branch. `upload` and `bind` are nested *inside* `texture` and cover only cache-**miss**
work, so `lookup` (= `texture - upload - bind`) is what a cache **hit** still pays per reference.
`descriptor` is the per-draw set layout allocation and update. `other` is the remainder against the
`resources` total on the preceding line, so the split states its own completeness — a large `other`
means the attribution is incomplete and should not be trusted.

`acquire` and `copy` are nested inside `buffer` and separate the two mechanisms that live in it:
`acquire` is pool/arena acquisition, which on a pool miss is
`vkCreateBuffer` + `vkAllocateMemory` + `vkBindBufferMemory` + `vkMapMemory`, and `copy` is the
staging `memcpy`. Read them together with the `backend_buffer_pool hits/misses/cached/evictions`
line: evictions approaching misses means the pool is at its capacity ceiling and destroying one
buffer for every one it creates, which `PROSPER_BACKEND_BUFFER_POOL_MB` raises. `buffer - acquire -
copy` is the transient fallback path, which allocates and copies together and is deliberately
attributed to neither.

The older independent 25-backend-call timing windows are disabled by default because their multi-line logging
can materially perturb the target call charged for printing them on Windows. Set
`PROSPER_BACKEND_TIMING_WINDOWS=1` only when that cross-submit legacy view or its memory-pool line is needed.
Use `PROSPER_RENDER_TIMING=detail` to additionally print individual texture and buffer resource builds
taking at least 0.5 ms (each capped at 250 lines). Set
`PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT=N` to begin those detail lines at renderer submit N, so boot
textures do not consume the cap before the scene under study. Each
texture lines identify local reuse, persistent hit/miss/invalidation, RTT sourcing, or an uncached decode;
buffer lines include the declared/uploaded range and whether direct guest backing was used. The
aggregate output also reports retained RTT bytes, decode-scratch capacity, and validation-scratch capacity;
use those figures before treating a process-memory increase as an unbounded cache.
Set `PROSPER_RTT_TIMING=1` with timing enabled to print one lightweight `[rtt-timing]` line per rendered target
group. It includes the submit number, target address, dimensions, draw count, and that target's exact Vulkan
phase breakdown without scanning pixels or printing every draw. `measured`, `detail`, and `other` show the
frontend-observed call duration, attributed backend phase sum, and remaining wrapper/logging cost for that
specific target. `PROSPER_RTTLOG=1` includes the same line plus
the more expensive visual RTT diagnostics. Use `PROSPER_RTTLOG_MIN_SUBMIT=N` and
`PROSPER_RTTLOG_MAX_SUBMIT=N` to bound either mode; unrestricted output can still perturb wall-clock routes.
For scene selection that is stable across runs, set `PROSPER_RTT_TIMING_MIN_DRAWS=N`. Lightweight records are
buffered for one ordered submit and emitted together only when its total backend draw count reaches N, so all
target calls are retained without logging the boot/menu workloads. For example, 300 selects the current
Dead Cells post-parse workload while excluding its 54-56-draw loading loop.
Selected lightweight records are formatted together and written to stderr in one batch at submit completion;
this preserves one parseable line per target without paying a Windows console write for every target.
Compute phase/image timing can likewise be restricted to one exact program address with
`PROSPER_COMPUTE_TIMING_CODE=0x...`. This filter does not enable the heavier compute trace; only the
selected timing records pay the stable-identity hash used in their output. Use it for a representative
long-running dispatch A/B rather than flooding every compute program's per-image records. Because guest
program addresses can move across runs,
`PROSPER_COMPUTE_TIMING_HASH=0x...` instead selects the stable FNV-1a hash of the translated SPIR-V
module (the same identity printed by an F8 performance-capture report). When both selectors are set,
they are an **AND**: the current address and stable hash must both match. The hash parser accepts only a
complete unsigned decimal or `0x` hexadecimal `uint64_t`; signs, whitespace, overflow, and trailing
characters fail closed with an explicit `ignored` banner. An accepted selector prints one bounded
`first-match` proof and a termination summary with `seen`/`matched`; `matched=0` is labelled
`INVALID-zero-matches`, not evidence that the shader did not run. Because a booted app deliberately
uses `_Exit` while its detached guest thread is still alive, it reports the summary explicitly just
before that exit; the ordinary destructor remains an idempotent fallback. Selected `[compute-phase]`,
`[compute-image]`, and `[compute-image-writeback]` lines carry both `code=` and `hash=` so copied logs
retain their own identity. A selected image record additionally carries its guest `addr=`, whether it
resolved to a `persistent=` image, and whether validation made the upload `upload-skipped=`. Alias
records name `alias_of=` and copy those two state bits from the real owner; they remain no-work records.

#### Reading the compute side at run scale

`PROSPER_COMPUTE_PHASE_TIMING=1` emits one `[compute-phase]` line per dispatch and
`PROSPER_COMPUTE_IMAGE_TIMING=1` one `[compute-image]` line per image binding. A routed title produces
tens of thousands of each, so pipe the run log through **`tools/perf/compute_phase_report.py`**, which
rolls both record types into one table: ms, share of total, and mean per dispatch, nested so each
sub-timer sits under the phase whose interval contains it, plus the programs that cost the most and the
dominant leaf inside each. Its text output also ranks real image bindings by stable shader hash,
binding, and sampled/storage class, with persistence/upload-skip counts and a bounded address list;
older records without those fields stay explicitly unknown. `--program 0xADDR` restricts to one kernel
and `--csv` emits the phase table for further processing. `--since-submit N` skips boot/warmup, but
**it suppresses the image section**:
`[compute-image]` records carry no submit ordinal, so they cannot be filtered to match, and showing
them would divide a whole-log numerator by a filtered denominator. Drop the flag for the image
decomposition.

Counts from this tool cover **backend-executed dispatches only** — CPU-fast-path fills return before
`execute_item` and emit no record, nearly a quarter of Astro Bot's dispatches. Add
`[render-timing] compute_cpu_fast fills=N` to the denominator before quoting any rate.

**Run both switches together whenever `setup` is the dominant phase.** `setup_ms` spans the whole
image-binding loop as well as descriptor validation and buffer binding, and that loop has *no* sub-timer
on the `[compute-phase]` line — so on an image-heavy title `[compute-phase]` alone reports a large
`setup` whose named children explain almost none of it. The report prints that gap as an explicit
`unattributed` row under every parent rather than dropping it, and `[compute-image]` is what decomposes
this particular one. The sampled-image cache timer completes before upload preparation begins, but the
storage-image cache timer is nested inside `prepare_ms`; the report prints storage cache as an included
child and warns in both redirected output and stderr if a record violates that hierarchy. Overhead of
both switches together, measured on Astro Bot at five matched dispatch
ordinals, is 1.5-1.9 %.

Ordered graphics/compute submits retain a bounded journal of exact guest-memory ranges written by the
compute backend. A persistent texture validated in an earlier graphics span can therefore skip its repeated
full-byte scan when no later write overlaps it. Across submits, exact byte comparison is the initial
fail-closed source of truth. Linux may promote a repeatedly unchanged direct-memory source to a
page-protection write watch: texture sources below 1 MiB stay on exact comparison, 1-8 MiB sources arm
immediately, and larger
sources arm after three exact unchanged validations. Promotions are limited to 8 MiB per ordered submit,
with at most one source larger than that limit, and adjacent pages are protected in coalesced runs. Set
`PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB`, `PROSPER_TEXTURE_WRITE_WATCH_MIN_KB`,
`PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS`, or `PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB` for controlled
A/B runs.

**Every one of these takes a plain decimal integer and nothing else, and a malformed value is refused
on stderr while the default stands (#3253).** That is not pedantry about input: on each of them **0 is
a meaningful and maximally aggressive setting** — defer nothing, no size floor, promote on first sight,
unbounded arming — and a bare `strtoull` answers 0 for anything it cannot parse. So `=8mb`, `=8 KB` or
a stray quote used to select the *opposite* arm from the one you asked for, with nothing in the run's
output saying so. No sign, no `0x`, no units, no surrounding whitespace; unset takes the default in
silence.

Compute buffer/image residency uses the same exact-first rule with `PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS`
and `PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_MB` — **but it does not have the "1-8 MiB arm immediately" half**,
and this paragraph claimed it did until #3155. That path has always passed a defer minimum of **one byte**,
which makes the size exemption unreachable, so every compute source must climb the stability ladder however
small it is. `PROSPER_COMPUTE_WATCH_DEFER_MIN_KB` exposes that minimum for A/B (unset = 1 byte, i.e. today's
behaviour; `=8192` gives renderer parity; `=0` defers nothing). Like the rest of the family since #3253, a
malformed value is refused loudly and keeps the default. Whether the asymmetry is right is open —
see `RENDERER_PERFORMANCE_2026_07.md` § *Compute write-watch promotion census*.

`PROSPER_WATCH_PROMOTE_CENSUS=1` reports what any of those settings actually bought, every 256 submits and
at exit: how many acquisitions each of the three proofs decided (submit journal, armed page watch, full byte
compare), the bytes each cost, and the stability counter seen at every promotion decision. Every line is a
running total and prints its own denominator — read `decisions=` and `validated=` first, because comparing
two arms' tallies taken at different points in a run is the mistake that produced #3155's retracted numbers.
`PROSPER_WRITE_WATCH_MAX_KB` is an emergency host-wide range
limit (zero/unset is unbounded); a declined watch always returns to exact comparison.

`PROSPER_COMPUTE_BORROW_CENSUS=1` reports the other side of the same boundary, on the same cadence:
whether a graphics sampled descriptor managed to lease the device image a compute dispatch had just
produced, instead of re-reading the surface out of guest memory. Four `[compute-borrow-census]`
lines, all running totals with their own denominators — the consumer's precondition partitioned by
term, the borrow's outcome partitioned by gate (`no_cache_entry`, `export_unpublished`,
`content_invalid`, `no_image`, `authority_changed`, `hit`), why `authority_changed` fired (a real
overlapping guest write, an unarmed submit journal, or a cross-submit export; and what the page
watch said), and the producer's publish gate partitioned the same way. A zero bucket still prints.

Two of its fields answer questions nothing else can. On a lookup miss the variable additionally
enables an O(cache) scan for an entry at the same guest address, which turns "nothing is cached
under this key" into "an entry here disagrees on `tile_mode`" — the twenty-three-field cache key
otherwise makes those indistinguishable. **That field list belongs to the EXACT key only.** When the
exact key misses, the importer retries under a format-*aliased* key, and that key rewrites `format`
and `vk_format` by construction — so scanning it would report those two as differing on every miss
whether or not they were the reason, and would point a reader at exactly the normalisation the alias
arm forbids. The retry is therefore not scanned, which is also why `exact_key_scans` (scans
performed) and `no_cache_entry` (final outcomes) are different numbers. Read the field list against
**`same_addr=`, and against nothing else** — a field bit and `same_addr` are incremented under one
and the same condition, so that is the list's exact denominator. The three numbers on that line nest
(`exact_key_scans` ⊇ non-`rescued` ⊇ `same_addr`), and dividing by either wider one dilutes the field
list with lookups that found nothing at the address at all — which is the *other* decision-table row,
the one where the producer is absent rather than keyed differently.

`rescued=` counts the exact-key misses the format-alias retry then saved. Those are successful
imports whose mask names `format`/`vk_format` by construction, so they are counted and their fields
deliberately excluded — otherwise a title whose alias path routinely succeeds reports a field list
dominated by the alias working as designed. And the `Unknown` that `guest_gpu_writes_since` returns —
which its own header warns a caller cannot decompose — is split into the three parts that ARE
separable from the borrow site: `journal_unarmed` (not armed on the consuming thread),
`journal_unjournaled` (the producer's publish carried no submit serial, so it was stamped while the
journal was inactive or already overflowed), and `journal_undecided` (armed, serial present — a
different submit, or this submit's journal has since overflowed). The last name is deliberately
vague, because it covers two causes this instrument cannot tell apart and a narrower name would be
a claim rather than a count.

Proven-full write-only storage targets at least 16 MiB do not copy their first successful result into
an immediately redundant CPU source baseline. The submit journal remains the initial source authority;
if another architectural writer invalidates it, the next invocation takes ordinary writeback and
establishes the exact baseline needed for later identical-result skips. A failed attempt to replace a
host result fallback with a GPU baseline invalidates that obsolete fallback before returning. Partial,
readable, and replay-owned targets retain their original exact contract. Set
`PROSPER_COLD_STORAGE_SNAPSHOT_MIN_MB` to retune the crossover — it validated its input before #3253 but
fell back in silence, which fails the other way: a typo left you measuring the default while believing
you had moved it. The broader
`PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1` control restores immediate storage-result snapshots.

A sampled image that successfully imports an authoritative renderer-owned Vulkan image does not validate,
snapshot, convert, or cache its guest backing. This includes a persistent depth/stencil import, whose
authority is intentionally separate from the color-RTT registry: the imported image is the descriptor the
dispatch consumes, while the raw guest bytes may be stale. Set
`PROSPER_NO_IMPORTED_IMAGE_GUEST_BYPASS=1` to restore the redundant guest preparation for an A/B.

Windows protection-fault watches are deliberately unsupported: the Windows
exception dispatcher writes below the interrupted stack pointer before a vectored handler runs, which can
corrupt the guest's valid SysV red-zone locals. Timing therefore reports those watch attempts as `unknown`
and the remaining exact validation bytes; do not re-enable page-fault watches without an exception-delivery
mechanism that preserves all 128 guest red-zone bytes. For the same-submit A/B control, set
`PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1`. Set
`PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE=1` to keep every comparison while checking and logging any
disagreement with the journal's unchanged decision; use that audit before extending writer coverage.
The instrumentation does not take clock samples when the variable is unset. This is the first tool
to use when the window presents correctly but a title is not interactive; do not infer a GPU
bottleneck from low FPS without the stage breakdown.

On native Windows, large aligned direct-memory views are demand-paged. Their committed 16 KiB pages
are retained in a mapping-generation-invalidated bitmap so repeated GPU resource reads do not repeat
the same `VirtualQuery` calls. Set `PROSPER_NO_SPARSE_DMEM_PAGE_CACHE=1` for a control run when
investigating this path; it is expected to be substantially slower in resource-heavy scenes. A
thread-local positive HLE mapping lookup can be disabled separately with
`PROSPER_NO_SPARSE_DMEM_ACCESS_CACHE=1`.

When the `tables=` bucket is unexpectedly large, set `PROSPER_STAGE_FOLD_PROFILE=1`. It ranks the
shader address and user-SGPR base pairs responsible for scalar table folding, including average and
maximum time, decoded instructions, dynamic fetches, SRT uses, guest readability checks, and the
decode/probe/interpreter split. It prints every 4096 folds by default; set
`PROSPER_STAGE_FOLD_PROFILE_CALLS=<N>` to change that window. The profiler is intended for short
diagnostic runs and takes no timing samples when disabled.

Transient Vulkan memory uses a bounded, exact-requirements pool because every backend call waits for its
fence before cleanup. The optional backend timing windows include `memory_pool` hits, misses, cached allocation count/bytes,
and budget-driven discards. The default budget is 512 MiB; override it with
`PROSPER_MEMORY_POOL_MB=<MiB>`, or set `PROSPER_NO_MEMORY_POOL=1` for an A/B run against direct
`vkAllocateMemory`/`vkFreeMemory`. The pool retains only allocation objects: images, buffers, views,
descriptors, and their layout/content rules keep their existing per-call lifetimes.

**Texture staging is a second, separate cache with its own budget.** It retains a staging block
together with its CPU **mapping**, because dropping the mapping returns the pages and the next write
faults all of them back in through amdgpu — ~30% of the frame on a 4K upload (#3405). Its budget is
`PROSPER_MAPPED_STAGING_MB`, default **256 MiB**, accounted independently of the transient pool
above, so peak retained host memory is the sum of the two rather than the pool figure alone.
`PROSPER_NO_MAPPED_STAGING=1` disables reuse while leaving the allocation path identical — the
A/B seam that isolates it — and `PROSPER_NO_MEMORY_POOL=1` disables this cache as well, so that
flag still means "no staging is pooled" as it did before this cache existed.

Within one already-synchronous backend call, identical immutable Vulkan contracts share objects instead of
recreating them for every draw. Storage buffers require the same nonzero guest address, size, and complete
captured bytes; synthetic resources with no identity remain distinct. Texture image views and samplers use
their complete image/view/swizzle/filter/address/LOD contract. Descriptor-set layouts and pipeline layouts
use their complete binding/layout contracts, and all sets allocate from one call-wide descriptor pool. Every
object remains call-local and is destroyed after the existing fence wait, so this adds no cross-submit
freshness assumption. Set `PROSPER_NO_BACKEND_RESOURCE_SHARE=1` to disable the keyed object reuse for an
A/B; the safe call-wide descriptor-pool consolidation remains enabled in both modes.

Host-visible storage buffers use a separate bounded pool across backend calls. By default, the backend
packs call-local logical uploads into non-overlapping slices of a persistently mapped host-coherent arena,
using the device's storage-buffer offset alignment. Each Vulkan descriptor keeps the exact aligned offset
and logical byte range, and every slice is rewritten before use. The backend waits for its fence before
returning arenas to the pool, so later uploads cannot race in-flight work. Arenas start at 1 MiB; set
`PROSPER_BACKEND_BUFFER_ARENA_KB=<KiB>` to tune that target or
`PROSPER_NO_BACKEND_BUFFER_ARENA=1` to restore one pooled buffer per logical upload. Oversized uploads and
the fallback path still use power-of-two capacity classes. The thread-local pool is capped at 4096 buffers
and 256 MiB by default; set `PROSPER_BACKEND_BUFFER_POOL_MB=<MiB>` to change the byte budget or
`PROSPER_NO_BACKEND_BUFFER_POOL=1` for the former create/map/destroy path. With renderer timing enabled,
`backend_buffer_pool` reports cumulative hits, misses, cached count/bytes, and evictions.

The persistent compute device has the same exact-requirements allocation pool with a separate 256 MiB
default budget (`PROSPER_COMPUTE_MEMORY_POOL_MB=<MiB>`). `PROSPER_NO_MEMORY_POOL=1` disables both
graphics and compute pools. Decoded texture scratch vectors are also retained across callbacks; they
are storage only, and every partial decode/read explicitly clears its unwritten tail.

Storage buffers whose complete backing range passes the executor's generation-scoped readability guard
are handed to the synchronous Vulkan backend as immutable views of unified guest memory. The backend hashes
and copies every visible byte into its upload arena before returning; ordered submission batches retain only
that completed Vulkan upload, never the guest pointer. This removes frontend allocation and copy work without
adding cross-submit freshness or lifetime assumptions. Captured host backing follows the same rule only when
it covers the complete requested range, and partial/unreadable ranges retain the zero-filled owned-copy
fallback. Set `PROSPER_NO_FRONTEND_BUFFER_VIEW=1` to restore per-reference materialization for an A/B.
Renderer timing reports buffer `views`, logical bytes, and bytes that still required materialization.

Within one renderer callback, repeated guest-backed texture descriptions reuse the first decoded
pixel buffer. A bounded process-wide cache also covers supported guest-backed linear and tiled 2D sampled
textures (`Unorm8`, one/two-channel `Unorm16`, `Float16`, `Float32`, packed 32-bit, and BC formats),
and supported uncompressed volume inputs. Every cross-submit lookup validates
the exact source range before reusing decoded pixels: direct narrow sources compare their native
bytes, tiled sources retain the padded tiled range, and block-compressed sources use their exact
block dimensions and block size. Volume sources retain the complete tiled 3D span. The decode key includes
the selected mip-tail and layer layout so distinct views cannot alias. Changed bytes,
mapping/readability changes, or address reuse invalidates the entry and creates a new content-version ID.
Live render targets, storage images,
captured host backing, unsupported cube/volume formats, and unresolved DCC states remain excluded. The
default ceiling is one eighth of host physical memory, clamped to 1-2 GiB; use
`PROSPER_TEXTURE_DECODE_CACHE_MB=<MiB>` to change it or
`PROSPER_NO_TEXTURE_DECODE_CACHE=1` for an A/B run. Timing reports cross-submit `texture_cache`
hits/misses/invalidations separately from `textures` (all texture uses) and `reused` (both local and
persistent decodes avoided).
Complete readable source ranges are compared directly against the cache's retained encoded bytes;
partial sparse ranges keep the guarded scratch-copy fallback. This preserves the same exact byte and
readability checks without copying large texture sources before comparing them. Set
`PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY=1` for an A/B against the previous copy-then-compare path.
After an exact validation successfully promotes a Linux source to a write watch, the cache releases
its redundant encoded snapshot. Guest GPU/DMA write notifications dirty the same watch as CPU faults.
An unchanged watch permits reuse; Dirty/Unknown conservatively invalidates and re-decodes instead of
relying on a hash. Set
`PROSPER_KEEP_TEXTURE_SOURCE_SNAPSHOTS=1` for a same-build memory/performance A/B. Validation audit
modes retain the snapshot they need for byte-exact checking.

The Vulkan backend may retain an optimal sampled image only when that exact frontend validation
supplies a nonzero content-version ID. Cache hits skip image allocation, staging allocation, pixel
copy, transfer commands, and upload barriers. Exact image-view and sampler contracts over a retained
image remain resident with that image, under a 32-contract per-image bound; set
`PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS=1` to retain images while restoring callback-local
bindings for an A/B. Its default byte ceiling is one eighth of the largest device-local heap,
clamped to 1-2 GiB, with at most 1024 allocations. Set
`PROSPER_BACKEND_TEXTURE_CACHE_MB=<MiB>` to change the byte budget or
`PROSPER_NO_BACKEND_PERSISTENT_TEXTURES=1` for a forced-upload A/B. Backend timing reports
`persistent=hits/misses` and the current cache bytes. The frontend decoded-pixel budget and backend
device-image budget are separate: a hot immutable atlas can occupy space in both, trading bounded
residency for lower frame time.

Intermediate live color targets remain on the GPU by default. A later graphics pass that samples the
same guest target identity, extent, and format binds the retained image directly. Intermediate scanout
spans also defer readback until the final renderer callback; a final scanout pass reads the accumulated
target normally, while a submit that ends elsewhere materializes its cached scanout once before publishing.
Deferred scanouts are pinned against bounded target-cache eviction until that final callback.
Ordered DMA producers, captures, pixel diagnostics, same-target feedback, and other authoritative-readback
spans keep the CPU path. Guest GPU writes invalidate overlapping retained targets through the ordered write
observer. Set `PROSPER_NO_INTERMEDIATE_SCANOUT_DEFER=1` to restore per-span scanout readback for an A/B. Set
`PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1` for a complete frontend A/B, or
`PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS=1` to disable backend retention independently. The backend
cache defaults to 256 MiB and can be changed with `PROSPER_BACKEND_TARGET_CACHE_MB=<MiB>`.

Graphics RDNA2-to-SPIR-V results use a process-wide bounded cache (4096 entries and 128 MiB by
default). Its key contains the shader bytes and only the resource-table fields consumed by compilation;
current guest addresses and backing data remain on each draw and are never cached. Use
`PROSPER_NO_SHADER_CACHE=1` for a direct-recompiler A/B run or
`PROSPER_SHADER_CACHE_MB=<MiB>` to change the byte budget. `test_shader_recompile_cache` compares
every tested miss byte-for-byte with the direct recompiler and verifies that runtime-only resource
changes hit while descriptor-interface changes miss.

Shader code span, PC-relative dispatch metadata, and fragment interpolation discovery use a separate
64 MiB immutable-analysis cache. Every lookup validates the complete instruction/embedded-table span
byte-for-byte, so same-address shader patching invalidates the entry. Concrete user SGPRs, descriptor
table bytes, guest addresses, and resource backing remain per-draw inputs and are never reused by this
cache. Set `PROSPER_NO_SHADER_ANALYSIS_CACHE=1` for the direct-analysis A/B or
`PROSPER_SHADER_ANALYSIS_CACHE_MB=<MiB>` to change its budget.

The Vulkan backend retains graphics pipelines across render-target calls. Live draws carry a
process-unique, never-recycled identity from the exact shader cache, so a hot lookup does not copy or
hash complete SPIR-V modules. Capture, replay, and test draws without those identities fall back to an
exact full-module key. The rest of the key contains the descriptor-layout contract and every baked
fixed-function value; equality still compares the complete key after hashing.

**A shader-cache identity does not imply descriptor ARITY, and the key must carry it separately (#2471).**
An identity names the exact compile key, including every descriptor's class and binding — but a module is
byte-identical whether its binding was declared with one descriptor or with N, so arity reaches the
pipeline *layout* without reaching this key. When the two disagree, a cached `VkPipeline` created under an
arity-1 layout is replayed with an arity-N layout bound, which is `VUID-vkCmdDraw*-None-08600`
(*"Set 0 binding 0 descriptorCount 3 doesn't match 1"*) raised at the draw, thousands of lines from either
cache. Both key branches were affected and for different reasons — the identity branch appended no count at
all, and the full-module fallback hardcoded `1` — so the arity vector is now appended once, ahead of the
branch, and `descriptor_arity()` is the single source of truth for the pool, the layout, the write and the
key. **This class is invisible to a pixel assertion**: the stale pipeline still draws correct output, so the
test that guards it asserts the cache *decision* (an arity-3 draw must miss against an arity-1 baseline,
and a second arity-3 draw must still hit) rather than the frame. Only a validation layer sees the rest, so
run `tools/vkval/vk_validation_scan.py` on any change that touches descriptor counts or layouts.

`PROSPER_PIPEKEY_LOG=1` prints one line per draw pairing the two independent decisions — the pipeline-cache
key and hit/miss, against the resolved `VkPipelineLayout` handle and the per-binding arity — which is what
makes a divergence between them readable:

```
[pipekey] draw=0 key=be1fbdc545dac514 words=423 MISS exact_ids=0 use_desc=1 n_sets=1 arity=[3:1] layout=0x…5a0 pipe=(nil)
[pipekey] draw=0 key=d8ac419006f6961a words=423 MISS exact_ids=0 use_desc=1 n_sets=1 arity=[3:3] layout=0x…3b0 pipe=(nil)
[pipekey] draw=0 key=d8ac419006f6961a words=423 HIT  exact_ids=0 use_desc=1 n_sets=1 arity=[3:3] layout=0x…3b0 pipe=0x…240
```

Two draws sharing a `key=` while differing in `arity=` or `layout=` is the defect; the same `key=` with the
same `arity=` and `layout=` is a correct reuse. The cache is bounded to
4096 entries by default. Set `PROSPER_PIPELINE_CACHE_ENTRIES=<N>` to change the entry limit or
`PROSPER_NO_BACKEND_PIPELINE_CACHE=1` for a transient-pipeline A/B. With `PROSPER_RENDER_TIMING=1`,
the backend and submit-aligned windows report references, hits, misses, bypasses, current entries, and
evictions. A pipeline hit also skips temporary Vulkan shader-module creation.

Pipeline layouts use a separate exact cross-call cache keyed by every descriptor set's ordered binding,
type, count, and stage contract. Descriptor pools, descriptor sets, and descriptor-set layouts remain
call-local; Vulkan pipeline-layout compatibility follows the complete descriptor contract rather than the
temporary layout handle. The cache retains at most 256 layouts and evicts the least-recently-used entry not
referenced by the current backend call. Set `PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES=<N>` to change the bound
or `PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE=1` for a call-local A/B.

Do not cache `build_stage_table` results using only shader addresses and user-SGPR values. Descriptor
tables are reached through guest pointers, and their memory can change while every pointer/register
value remains identical. That experiment caused Messenger to remain on its initial loading screen and
was removed after an enabled/disabled A/B test. A future table cache needs explicit guest-memory
versioning or equivalent invalidation; the `tables=` timing bucket measures this work without caching it.

Windows readability checks retain positive results across synchronous submits only for explicitly
tracked, currently readable kernel-HLE mappings. This is mapping-topology reuse, not a table or
guest-content cache: HLE map, unmap, and protection operations advance a process generation and
discard all retained ranges before the next submit. Host-managed guest stacks, diagnostic mappings,
and other untracked regions are reused only within one submit. Guest bytes are still interpreted on
every fold. Set `PROSPER_NO_GUEST_READ_CACHE=1` for the uncached A/B path. The contract assumes memory
referenced by a synchronous GPU submit remains mapped until that submit returns, which was already
required before the cross-submit reuse was added.

Windows direct memory uses a delete-on-close sparse temporary file as its shared physical backing.
Mapped views are ordinary `MEM_COMMIT` file pages and can participate in cross-submit readability reuse;
the kernel demand-pages untouched storage without delivering a user-mode first-touch exception. This keeps
large logical arenas sparse and physical aliases coherent without letting Windows exception dispatch
overwrite guest SysV red-zone locals. If sparse backing cannot be created, direct-memory mapping fails
closed instead of silently returning to the unsafe `SEC_RESERVE` exception path.

**Present-path correction (2026-09-01, #3087):** this paragraph used to open with "the current renderer
remains a deterministic readback-based implementation," stated with no date, as though it described the
renderer generally. **#1270 (landed 2026-07-24) falsified that for the shipped app.** `prosper-app`
defaults to GPU present for any real game boot: the renderer publishes its front-buffer image directly
and the app GPU-blits it straight to the swapchain, skipping the CPU round-trip
(`PROSPER_APP_GPU_PRESENT=0` is the explicit opt-out; the app also falls back to the private-device CPU
path at runtime if shared-device adoption fails). The CPU readback described in "The Vulkan-context
decision" above is still real code — it is what `--test-pattern`, any adoption failure, and every caller
that never invokes `set_gpu_present_active` still take, which includes `tools/screenshot` (the sole call
site is `frontends/prosper-app/main.cpp:668`) and every headless/ctest/CI path. Before quoting an FPS
figure or calling "the renderer" readback-based, check which of the two paths produced it — see
CLAUDE.md's "before quoting an FPS number" bullet, which carries this exact distinction; it is not
restated here. Issue #702, which this paragraph used to cite as tracking "direct image presentation,"
closed 2026-07-19 — five days before #1270 delivered exactly that for the interactive path.

On the native Windows Messenger first level, the exact-byte texture cache improved the same-binary
workload from about 20 FPS to 24 FPS (resource construction 10.1-10.5 ms to about 3.9 ms). **That figure
is itself a readback-path measurement from 2026-07-14, ten days before #1270**, and is not a current
renderer rate — see CLAUDE.md and `RENDERER_PERFORMANCE_2026_07.md` for the 2026-08-27 windowed
re-measurement (median ~156 presented fps under GPU present) before treating it as current. The
remaining roughly 36-38 ms submit time was primarily ordered graphics/compute backend work: four
graphics spans, four compute dispatches, synchronous fence waits/readbacks, and transient Vulkan
pipelines/resources. Further work should be validated against a 3D title and converge graphics/compute
resource ownership; title-specific 2D cache additions are not the current priority.
The complete A/B table, heavy-frame budget, corrected capture graph, rejected experiments, and handoff
decision are preserved in `RENDERER_PERFORMANCE_2026_07.md`.

WSLg remains a useful alternate path for running the Linux build:

1. Prereq check: `vulkaninfo` in the WSL shell confirms a usable Vulkan device (vendor WSL ICD, else
   Mesa Dozen over D3D12). This is the one external dependency worth verifying up front.
2. Build the frontend in WSL with `-DBUILD_FRONTEND_APP=ON`.
3. Run it; WSLg surfaces the window on the Windows desktop with audio + a taskbar/Start entry.
4. Optional: a `.desktop` file (WSLg auto-registers Start-menu shortcuts) or a one-line `.bat`/`.lnk`
   launcher on the Windows side.

## Keeping it agentic-first

The human-facing app is **opt-in and additive**. The BMP/CRC headless path stays the default and the
CI source of truth. Nothing the frontend adds gates a core test. A future *frontend* smoke test, if
wanted, should use an offscreen/hidden surface and stay out of the default `ctest` set (needs a real
Vulkan device + display, which CI may lack).

## P0b status — boot via the shared `boot_program()` helper (option **(b)**, landed)

The chosen path was **(b): extract a shared `boot_program()` helper** both `boot_trace` and the
frontend call, rather than duplicate the boot glue.

- `src/host/image/boot_program.hpp/.cpp` (in `prosper_core`, Linux and Windows): links the fixed module set
  (honoring `PROSPER_NO_PSN`, dropping absent modules), registers the built-in HLE, maps images, sets
  up TLS/unwind/procparam, installs the import stubs + trap handler, registers the PSN/SaveData
  module-start ranges, and runs the dependent-module init_arrays. An `after_hle_registered` hook lets
  a caller install host frontends (audio sink / pad backend) at the exact point `boot_trace` always
  did. On success the caller registers its renderer and calls `run_entry`.
- `boot_trace` migrated to call it (behavior-preserving; verified it still boots + renders).
- `prosper-app` calls it too (`--dump <app0>`): the guest boots and runs on its own thread while the
  window owns present.

**Remaining for "show the game" — the composite renderer (render-frontier-owned):** booting is not
enough. Nothing reaches the window until a live renderer is registered via `set_submit_renderer` —
the ~350-line DrawItem→Vulkan lambda in `boot_trace` that composites each submit and calls
`present_write_frame`. Verified empirically: `prosper-app --dump …` boots the game and opens the
window but presents **0 frames** without it. That lambda is the render frontier's actively-evolving
code, so registering/extracting it is a **coordinated next step with that workstream**, not a
duplicate. Until then the app is fully functional via `--test-pattern` (and any external feeder of
`present_write_frame`).

## Phased plan

- **P0a — window + present** ✅ **done**: lifecycle hook, SDL3 window, Vulkan swapchain,
  present-from-readback, `SDL_QUIT`/Esc→stop. Verified with `--test-pattern`.
- **P0b — shared boot** ✅ **done**: `boot_program()` helper; `boot_trace` + `prosper-app` both call it.
- **renderer (#177)** ✅ **done**: shared `register_live_renderer` (frontends/shared); the window
  shows the composited game (verified `--dump … --frames 3`).
- **P1 — audio** ✅ **done**: `prosper-app` installs the SDL3 `AudioSink` (`sceAudioOut` → host).
- **P2 — controllers** ✅ **done**: installs the SDL3 `PadBackend` (host gamepad → `libScePad`).
- **P3 — polish** (in progress): resize/fullscreen, present-mode selection, pause/resume at a guest
  flip boundary with coherent audio, and non-modal SaveData
  percentage progress presentation and SaveData virtual-slot LIST selection are implemented. The LIST
  boundary retains only guest virtual directory names and never exposes a host file picker;
  quit UX remains.
  Native Windows build/run packaging is done via `scripts/run-windows.ps1`. Cooperative guest-stop
  at a flip boundary is a follow-up (today the
  guest thread is detached at window-close and reclaimed by process exit).

## Ruled out

- **"The headless path is safer by design."** The issue observed that `tools/screenshot` had never
  taken the desktop down and read that as a property of the frontend. It is not: `screenshot.cpp`
  detaches its guest thread and calls `_exit` in exactly the same shape `prosper-app` does, so it can
  abandon an in-flight submission identically. What differs is the consequence — no compositor is
  sharing the device with a headless run, so a stuck reservation has nothing to block. Both frontends
  now take the gate (#3225).
- **"A GPU hang / driver fault is involved."** No fault is logged at all: nothing in `journalctl -k`
  for 30 minutes around the freeze, and the amdgpu hang detector never fires — there is no stuck job
  to detect, only a held lock, which is why there is no self-recovery and no timeout (#3225).

## Risks & open questions

- **Vulkan availability** — native Windows needs the Vulkan SDK at build time and a working host
  driver at runtime. WSLg additionally needs a usable WSL Vulkan ICD.
- **The run/stop hook touches the run harness** (`boot_trace`/how the guest loop is driven), which the
  render-frontier and audio workstreams also use — coordinate; keep the fixed-budget CI path intact.
- **`feat/audio-sdl3` overlap** — P1 must build on that branch, not fork it.
- **Present latency/vsync** defaults to FIFO. `--present-mode mailbox` opts into low-latency vsync,
  and `--present-mode immediate` explicitly permits tearing; unsupported optional modes fall back to
  FIFO with a diagnostic.
- **Later zero-copy** (`external_memory`) is deliberately deferred; the readback seam was the v1 answer.
  **Superseded 2026-07-24 for the shipped app:** #1270 made GPU present (renderer device adoption + a
  direct front-buffer blit, not `external_memory`) the default for a real game boot; the readback seam
  remains the fallback (`PROSPER_APP_GPU_PRESENT=0`, adoption failure, `--test-pattern`, `tools/screenshot`).
- **area:** shared/host infrastructure — needs an `area:` decision and coordination before build.
