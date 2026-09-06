# Sonic Frontiers (`PPSA03831`) — status

Tracker: [#1891](https://github.com/mattias800/prosper/issues/1891). Engine: **Hedgehog Engine 2
(Needle renderer)**, SurfRide UI, statically linked CRIWARE middleware — determined from the eboot's
own embedded shader source paths (`Library\hedgehog\…`, `Library\needle\…`) and the dump's
`NeedleShader.pac` / `raw/hedgehog/` asset trees, not assumed from the publisher. *Sonic Origins*
(#1871) and *Sonic Racing: CrossWorlds* (#1895) share parts of the same stack.

## Current rung — gameplay reached, world not rendered

A committed input route (`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`) takes the title
from boot to **`GameModeStage` running a Cyber Space stage (`w6d01`)** with real GPU work: the
guest streams all one hundred `w6d01_trr_s00..s99` terrain sectors plus `w6d01_gedit`, loads
`ui_gamemodestage*.pac`, streams `sound/cyber_sound/bgm_cyber.awb`, and its submitted draw rate
rises from **48 draws/flip on the title screen to a sustained 449 draws/flip** in the stage.

The gameplay HUD composites correctly at 3840x2160 — ring counter, the five Red Star Ring slots,
the boost gauge, and **a stage timer that advances monotonically with the guest's flips**
(00:52.39 -> 00:56.76 across one 55-sample capture; 01:02.36 -> 01:05.83 across a second run).
A running stage clock is the discriminator this title offers and a menu cannot fake it: no
aggregate frame metric was used to make the call. Checked-in capture:
`assets/screenshots/sonic-frontiers-cyberspace-hud.webp` (direct unmodified `tools/screenshot` frame,
3840x2160, route arm, stage clock at 00:55.89).

**What is not there is the world.** The 3840x2160 frame is black behind the HUD, because
**16 of the stage's 32 compute programs never execute** (#2790). See the section below. So the rung is deliberately not
ticked as a rendered-gameplay milestone: the route reaches gameplay, and a rendering defect stands
between that and a gameplay screenshot.

## What stood between the title screen and gameplay: a twelve-page boot notice queue

A no-input arm never leaves the title screen because a **modal notice queue** opens over it and
stays. Its pages are the game's own post-update notices — "Extras", "Update notification",
"Game update", "Update Details", "Update", "Additional options", "Action Marks", "New Game+" — each
a blue header band over a full-width body panel. Measured behaviour, all from live captures:

| Question | Answer, and how it was measured |
| --- | --- |
| How many pages? | **Twelve.** Confirms 40 flips apart: presses 1-12 each advance one page and press 13 activates a main-menu entry. |
| What advances it? | **Face buttons only.** A single-button sweep (triangle, square, circle, options, touchpad, l1, r1, right, left, down, up, cross, 60 flips apart) advanced the panel on every face button and on **none** of the four d-pad directions — `right`, `left`, `down` and `up` left the header on "Update Details" for 240 flips, then `cross` advanced it. |
| Where is the cursor afterwards? | On **"Extras"**, the last of the six main-menu entries, so five `up` reach "New Game" whether or not the list wraps. |
| Is the queue route-stable? | **Yes, and independently of save state.** This row used to say "only against an isolated save area"; that is withdrawn. A `boot_trace` arm of the committed route reaches `GameModeStage` **with an existing save present**, so twelve confirms consumed twelve pages with that save in place. The page count is the whole of the queue-length story and it is speed-independent — one page per face press (row above) — so this **falsifies** that story rather than leaving it unseparated — the page count, not the wider question of whether save state matters, which the 2×2 above does not settle. ([#2790](https://github.com/mattias800/prosper/issues/2790), PR #3049.) |

This is why 405 dense confirms (one every 20 flips) got no further than six did: the pages are
consumed one per press with an animation between them, so spacing, not volume, is what clears the
queue. It is also why the earlier reading of this panel as "renders almost no text" (#2206) is
incomplete — see below.

## Route

`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`, flip-anchored, with a header explaining
every window.

**If a run parks in the "Extras" menu it desynced — that is not a render or input wall.** Measured
2026-08-27: one live-renderer arm sat there for its whole length (24/24 frames with content,
`pixel-distinct=24`, guest healthy) and never reached the stage. It is the same end state the
Ruled-out rows below record for **two** distinct causes, both ending in a confirm activating
"Extras": a `--warmup-seconds` window, and flip/wall-clock desync under host load. My failing arm used
**no** `--warmup-seconds`, so that cause is excluded there and host load is the remaining candidate.
Check `uptime` and which of the two applies before diagnosing anything.

**Save data was never shown to explain this**, although one arm made it look that way. The
queue-length *mechanism* is separately falsified (see the notice-queue table above); that is a
statement about the mechanism, not about whether the save matters. The 2×2, with arm counts, because
the asymmetry is the whole point:

| frontend | save state | reaches `GameModeStage` |
| --- | --- | --- |
| `boot_trace` | existing save present | **yes** — `ui_gamemodestage` markers, and 189 log lines naming `w6d01_trr` terrain sectors (arm count not recorded) |
| `boot_trace` | empty `PROSPER_SAVE0` root | **yes** (arm count not recorded) |
| live renderer | existing save present | no — parked in Extras (**n=1**, no `--warmup-seconds`) |
| live renderer | save moved aside, and `PROSPER_SAVE0` | yes (two arms) |

Two limits on what that table can carry. The cells that do the overturning have **no recorded arm
count**, so a table whose only counted cell is the `n=1` it overturns is weak evidence for a positive
claim. And `boot_trace` reaching `GameModeStage` is not the same as *the authored route* having run —
this document's own step table has the path passing `GameModeOpening` and `raw/event/scene/ev0020*`
at f2600, and neither arm was checked for those markers. A `PROSPER_FILELOG` arm with
`PROSPER_PAD_SCRIPT_LOG=1` would settle it by grep and has not been run.

What the table does support is the **negative**: the single renderer failure sits inside this title's
own 1.5–3.4 flips/s spread, so "save present" and "slow arm" were never separated on that frontend.
That is enough to withdraw the precondition and not enough to assert its opposite — **do not treat an
empty save root as a precondition of this route, and do not treat it as established that the save is
irrelevant.**

**`boot_trace` is the fast loop** — **17–30 flips/s**, against the live arm's 1.5–3.4, so `f2900` is
~100–200 s rather than 14–32 minutes. Today's arm measured ~17 and ~30 is recorded elsewhere in this
document, so treat the range as 17–30 rather than either endpoint. The nearest independent figures
are **two `tools/screenshot` arms with Vulkan rendering suppressed** (the `--warmup-seconds` row
below): 6,994 flips / 420 s = 16.7/s and 1,669 / 90 s = 18.5/s. Those land in the same band, but they
are a *different frontend* and n=2 — corroboration that a non-rendering arm runs at roughly this
rate, not a second measurement of `boot_trace`. The nine-arm row below is about `boot_trace`
desynchronising under host load and publishes no flip rate at all; do not read it as supporting this
number.

```bash
PROSPER_PAD_SCRIPT=@scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad \
PROSPER_PAD_SCRIPT_LOG=1 PROSPER_FILELOG=1 \
  timeout --foreground -s TERM -k 5s 220 ./build-linux/boot_trace <DUMP_ROOT>/PPSA03831-app0
```

Both switches are required, and omitting either fails in a different direction. Measured across five
arms of this route:

| switches | `.pac` lines | route position | mode markers |
| --- | --- | --- | --- |
| neither | 48 | — | — |
| `PROSPER_PAD_SCRIPT_LOG` only | 52 | present | **absent** |
| both | **1140** | present | **present** |

`PROSPER_FILELOG` is what makes the mode markers appear. Note a bare arm still prints ~48 lines
containing `.pac` — but they are `[apr] WARNING:` duplicate-asset notices, not `[file] open` records,
so a grep for `.pac` returns hits while no game mode is nameable. That is exactly enough to look like
file logging is already on. `PROSPER_PAD_SCRIPT_LOG`
is what gives the route's position in its own anchor units. **Both of this document's `boot_trace`
errors came from missing one of them:** the row retracted below was written from an arm with
*neither*, whose absent flip counter became "boot_trace has no flip counter"; and a later arm with
`PAD_SCRIPT_LOG` but no `FILELOG` showed no mode markers and nearly became "the stage was never
reached".

**On a live renderer arm, budget generously:** single-run rates of 1.5, 2.8 and 3.4 flips/s across
three arms of this route, so `f2900` landed between ~14 and ~32 minutes and one arm missed it inside
a 1400 s timeout. `--render-every 6` moves it ~1.5x (2.84 → 4.27) in one matched pair.

The shape:

| Flips | Input | Reaches |
| --- | --- | --- |
| f1100-f1540 | 12 x `cross`, 40 apart | clears the boot notice queue |
| f1700-f1940 | 5 x `up`, 60 apart | main-menu cursor "Extras" -> "New Game" |
| f2100 | `cross` | confirmation dialog |
| f2300 | `cross` | answers it |
| f2600 | `cross` | `GameModeOpening` + `raw/event/scene/ev0020*` |
| f2900 | `cross` | `GameModeStage` + `w6d01` terrain + `gedit` + stage HUD pack |
| f3300+ | held `cross` | skips the in-engine opening into the stage |
| f5200+ | `left-stick-up` blocks | forward motion under player control |

**The sampling window has to span the whole run, and the defaults do not.** `--seconds` is the
*interval between samples*, not a start delay; the loop runs `while (saved < count)`; and `--timeout`
defaults to a hard **900 s** (`tools/screenshot/screenshot.cpp`). A live arm on this title runs at
1.5–3.4 flips/s, so `f2900` is 850–1,930 s and the `f5200` gameplay block is 1,530–3,470 s — a
`--seconds 3 --count 55` plan samples for 165 s and exits before the route's first input is due.
Sample sparsely across a long timeout instead: 55 samples 70 s apart covers 3,850 s.

**No `--warmup-seconds` on this route.** It suppresses Vulkan rendering, which moves the guest's flip
rate off wall clock and fires every window at the wrong guest state (see the Ruled out row). It is
also not a way to skip the wait: there is no capture-delay flag that leaves rendering live, so the
window has to be long rather than late.

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_PAD_SCRIPT=@prosper/scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad \
PROSPER_SAVE0=~/frontiers-work/save/run1 \
  ./build/screenshot <DUMP_ROOT>/PPSA03831-app0 \
  --seconds 70 --count 55 --timeout 4200 --out ~/frontiers-work/shots/run1
```

**Reproduction:** the file-oracle result reproduced on four CPU-only `boot_trace` arms and the live
HUD-with-running-clock result on two `tools/screenshot` arms. The route itself is flip-anchored, so the same
windows drive a CPU-only arm at ~30 flips/s and a live 3840x2160 arm at ~3 flips/s unchanged — that
is why it needs no per-host tuning, and also why `--warmup-seconds` breaks it: moving the flip rate
relative to wall clock moves every window off the guest state it was authored for.

## The world is black in the stage — 16 of 32 compute programs never execute (#2790)

The gameplay HUD is correct and complete; behind it the frame is black. The cause is upstream of the
composite: with `PROSPER_COMPUTE_PROGRAM_CENSUS=1` in the stage the final census block reads

```
[compute-census] 131072 dispatch decisions over 32 program(s)
```

and **sixteen programs are listed with `executed=0`** — the census prints per-program detail only for
programs that skipped at least once (#2745), so sixteen of the thirty-two never ran at all. Three of
them dispatch **2880 threads wide**, the exact width of this title's scene target, so they are
screen-space passes over the frame the player is supposed to see.

The reject classes, and their share of the sixteen:

| Reject | Programs | Encoding |
| --- | --- | --- |
| `unresolved-operand`, SOP2 reading **SCC** (`ssrc0=253`) | 5 | `886a6bfd` `fmt=0 op=0x10` |
| `unresolved-operand`, SOPP `s_cbranch_execz` | 2 | `bf880027` `fmt=4 op=0x8` |
| `unresolved-operand`, SOP2 | 2 | `856a802b` / `856a802f` `fmt=0 op=0xa` |
| `unresolved-operand`, MIMG | 2 | `f0040308,00000101` `op=0x1 dim=1`; `f0380328,00091103` `op=0xe dim=5` |
| `unresolved-operand`, SOP1 writing `dst=126` (EXEC_LO) | 1 | `befe3bff,00000000` `fmt=1 op=0x3b` |
| `unresolved-operand`, VOP1 with SDWA | 1 | `7e2c0ef9,00061216` `fmt=7 op=0x7` |
| `compute-cfg-reject reason=exact-wave-dispatcher-unsafe guest-barrier=1` | 2 | — |
| `unrecorded` | 1 | — |

The SCC group is not an unknown encoding: `rdna2_emit_alu.cpp:764-792` deliberately **poisons** the
tracked SCC when a wave-mask op writes `SCC = (mask != 0)`, because that is a cross-lane reduction
the per-invocation model cannot form, "so a later consumer … rejects instead of misreading". The
poison is the right default; the cost here is five whole programs, and the `s_cbranch_execz` pair
looks like the same family reached through EXECZ. So the largest single lever is plausibly
**wave-level SCC/EXECZ semantics for compute** rather than sixteen unrelated gaps.
`CONFIDENCE: MED` on that grouping — the SOP2 operand decode and the family argument are inference;
the census numbers and encodings are measured.

### What the presented frame looks like, and what it rules out

- Almost every published frame is `guest_scanout` — prosper composited nothing for that flip and
  republished the guest's own display buffer, which holds only the HUD. In the stage window:
  3 composited of 55 samples, 2 of 50, and 3 of 100 across three arms.
- The guest-composited HUD spans `x[64..3776] y[59..2090]` of 3840x2160 — full frame, correctly
  placed, about 1% of pixels non-black.
- The frames prosper *does* composite are confined to the top-left **2880x1620**, exactly 75% of each
  axis and unscaled, and contain a flat blue-grey gradient — sky and fog with no geometry, which is
  what a scene target looks like when its shading passes never ran. 2880x1620 is Hedgehog Engine 2
  dynamic resolution at 75%; `CONFIDENCE: MED` that the missing step is the guest's own
  upscale/resolve.
- **The present path is not broken in general.** In the same arms prosper composites the in-engine
  *cutscene* correctly at full width (`x[0..3839] y[272..1887]`, letterboxed). Only the stage fails.
- **It is not a `--warmup-seconds` artifact.** The control arm used `--warmup-seconds 90`, so the
  renderer was live continuously from flip 1517 — before `GameModeStage` loaded at flip ~2900 — and
  produced the same HUD-over-black frames from flip 3429 to 4325 and the same 2880-wide rect.

This is the frontier for this title, and it is plausibly a **Hedgehog Engine 2** finding rather than
a Frontiers one: *Sonic Origins* (#1871) and *Sonic Racing: CrossWorlds* (#1895) share the Needle
stack, and CrossWorlds' "the composite then goes uniform" (#2013) deserves a census taken the same
way before it is treated as unrelated.

## Rung 2 — title screen and main menu (still current, still checked in)

A default launch reaches the whole 4K opening sequence, the auto-save notice screen, the title
screen and the main menu. Checked-in captures, all direct unmodified `tools/screenshot` frames from
a no-input arm at 3840×2160:

| Screen | Asset |
| --- | --- |
| SEGA logo | `assets/screenshots/sonic-frontiers-sega-logo.webp` |
| Cyber Space opening | `assets/screenshots/sonic-frontiers-opening-sequence.webp` |
| Sonic Team logo | `assets/screenshots/sonic-frontiers-sonic-team-logo.webp` |
| Middleware credits | `assets/screenshots/sonic-frontiers-middleware-credits.webp` |
| Auto-save notice | `assets/screenshots/sonic-frontiers-autosave-notice.webp` |
| **Title screen** | `assets/screenshots/sonic-frontiers-title-screen.webp` |
| **Main menu** | `assets/screenshots/sonic-frontiers-main-menu.webp` |

## What reaching rung 2 took: one unregistered NID

The title stalled after `raw/ui/ui_gamemodeinitialize.pac` for four investigation sessions, with the
symptom migrating from "the guest stopped submitting" to "prosper stopped publishing" to "the guest
composites nothing over an empty scene". The proximate cause was upstream of all three:

**`sceSaveDataTransferringMountPs4` (`RjMlsR8EXrw`, `libSceSaveData`) was not registered**, so it
reached `prosper_on_unimpl`'s `return 0` — which for this contract *is* `SCE_OK`. That is the FALSE
SUCCESS class (#2081). Frontiers zeroes a 32-byte mount-point result, calls this to look for a PS4
save to import (the main menu's own "Carry over from PlayStation®4" entry), is told the mount
succeeded, then formats `"<mountPoint>/gamedata"` out of the still-empty result and opens
**`/gamedata`** at filesystem root. That open fails `ENOENT`; the title retries once per frame,
forever, and `GameModeInitialize` never hands off to `GameModeTitle`.

Fixed by answering from local inventory: prosper has no PS4 save-data area and no local dump carries
one, so the honest answer is `SCE_SAVE_DATA_ERROR_NOT_FOUND` with the result untouched — the same
answer its already-registered sibling `sceSaveDataTransferringMount` (`WAzWTZm1H+I`) gives, whose
comment records the *identical* downstream signature on Dragon Quest VII (a `/GameSaveData245.dat`
open at filesystem root). Two titles, two sibling NIDs, one defect.

### The measured A/B

Two 60 s CPU-only `boot_trace` arms, same host, same session, binaries differing by exactly this
commit (`PROSPER_NO_COMPUTE=1 PROSPER_FILELOG=1 PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1`):

| | before | after |
| --- | --- | --- |
| `/gamedata` open attempts | ~1,450 (every frame to the end of the arm) | **0** |
| dispatcher hits on `RjMlsR8EXrw` | 1,319 | **0** (registered) |
| distinct unimplemented NIDs | 4 | 3 (the `libSceJson2` trio, #1967) |
| last content opened | `/gamedata` (ENOENT, forever) | `ui_gamemodetitle_en.pac`, `bgm.awb` |
| `draws_cum` at t=60 s | 53,459 | **94,842** |

## Reproduction — title screen only (no input)

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
timeout --foreground -s TERM -k 5s 340s \
  ./build/screenshot <DUMP_ROOT>/PPSA03831-app0 \
  --seconds 20 --count 15 --timeout 320 --out ~/frontiers-work/shots \
  --manifest ~/frontiers-work/manifest.json
```

**Sample past 50 s.** The renderer publishes nothing for the first ~946 present callbacks and a
*successful* frame prints no log line, so a short arm reads as rung 0 (instrument trap 87). The
title screen lands around **t = 200 s** and the main menu around **t = 220 s** on this host.

For anything that does not need pixels, a CPU-only `boot_trace` arm with `PROSPER_NO_COMPUTE=1
PROSPER_FILELOG=1` reaches `ui_gamemodetitle_en.pac` in about 25 s and is the fast iteration loop —
the guest's file sequence is the progression oracle for this title.

**Save state changes the route.** The mounted `/savedata0` area is `PROSPER_SAVE0` — **not**
`PROSPER_SAVEDATA_DIR`, which controls only the save-data-*memory* store. Frontiers writes
`option/`, `arcade/` and `challenge/` there on first boot and reads them on every later boot, so a
run inherits whatever an earlier run left. Since #2734 that root is namespaced by title id, so it is
no longer shared with *other titles* — but it is still shared with every earlier run of Frontiers
itself, including a peer lane's. Pass `PROSPER_SAVE0=<private dir>` for a first-boot arm; both routes
reach the title screen (measured), but they are not the same route.

### The errno is `CONFIDENCE: MED`, and two other titles disagree with the guess — a lead, not a finding

Frontiers gates on the **sign** alone (`test eax,eax; js` at all five call sites), so `NOT_FOUND`
(`0x809F0008`) is sufficient for everything this title does and the rung-2 result does not rest on
it. But the corpus says something about the *precise* code that nobody should have to re-derive:

**Two independent titles have a dedicated arm for `0x809F000F` and none for `0x809F0008`.** Of the
five local titles that call `sceSaveDataTransferringMountPs4`, three const-compare the result —
PPSA03839 against `0x809F0003` (a retry loop), and **PPSA07809 and PPSA08804 against `0x809F000F`**.
PPSA08804's compare is inside its error arm *past* the branch, at `+0x4e41a32`. When this was
written that made it invisible to `nid_gate_scan`, which stopped at the first branch and bucketed the
site as a plain non-zero test; the scan now follows both arms and reports it as `const` (PR #2637),
so it no longer has to be taken on the hand-read. `--no-follow-arms` reproduces the older reading.

`0x809F000F` appears nowhere in prosper, and the PS5 3.20 library dump carries names and NIDs only —
no constants — so what it means is unresolved. Start from this rather than from
`0x809F0008` if a title ever turns out to need the exact code. (Review of PR #2208.)

## Known defects on the title screen

- The title-screen heading renders the string **"Try Again"** where the SONIC FRONTIERS logo
  belongs, in large blue type. The surrounding menu strings are correct and legible ("New Game",
  "Language", "Carry over from PlayStation®4", "Copyright", "User manual", "Extras"), and the
  version string `1.41` draws, so this is a wrong string/asset selection rather than a text-render
  failure.
- Shortly after the main menu a full-width panel with a blue header band slides over the title
  screen and stays. Its body renders almost no text — a handful of glyph marks and a diamond
  cursor — over a correct SurfRide background. No input was driven in either arm.

Both are filed as [#2206](https://github.com/mattias800/prosper/issues/2206).

## The Cyber Space stage's compute frontier (2026-08-21)

The route on [PR #2791](https://github.com/mattias800/prosper/pull/2791) reaches `GameModeStage` on
Cyber Space `w6d01`; the HUD composites and the world behind it is black. `#2790` measured the cause
as compute programs that never execute, and the census is the instrument:

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_COMPUTE_TRANSLATE_ONLY=1 \
PROSPER_COMPUTE_PROGRAM_CENSUS=1 \
PROSPER_DBG_PROGRAM=0x2005714000,0x2005717e00,0x200571bd00 \
PROSPER_PAD_SCRIPT=@<route>/reach-gameplay.pad PROSPER_SAVE0=<private dir> \
  ./build/boot_trace <DUMP_ROOT>/PPSA03831-app0
```

**Guest program addresses are run-local — re-derive them from the `[compute-census]` per-program
lines of the run you are in, not from this document.** And **read the program count on the census
line first**: this route desynchronises when the host is busy, and an arm that stayed in the menus
reports `over 5 program(s)` at the same denominator as one that reached the stage reports
`over 30 program(s)`. Nothing else in the block distinguishes them. Check `uptime` first — on a
shared machine the load that breaks the route is not yours and is not visible from the arm. `PROSPER_DBG_PROGRAM` then narrows the
verbose recompiler stream to those addresses; `PROSPER_DBG=1` is ~1.5 GB here and desyncs the pad
script before it reaches the stage.

At `262144 dispatch decisions over 32 program(s)`, **14 programs are listed and every one is
`executed=0`** (the census prints per-program detail only for programs that skipped at least once,
#2745). Three of them dispatch at the display width — `240x135x1` groups of `16x1`, `16x2` and
`16x3`, i.e. 3840x135, 3840x270 and 3840x405 threads — and are the screen-space passes the world
depends on.

**All three reach the CFG dispatcher's body.** #2790's handoff named two levers,
`s_cbranch_execz` and `image_load_mip`; both were reject PCs printed by the *straight-line* emitter,
which these programs only reach after two earlier routes decline. The first decline is ordinary
route selection (`backward else`, `role=route-decline`); the second was a real defect in the Wave64
MUST dataflow, fixed by adding V_LSHL_ADD_U32 to `scalar_alu_source_words`' B32 list. With that in
place the `wave64-ambiguous-mask-read` decline does not occur anywhere in the run, all three programs
reach the CFG dispatcher's body, and all three stop at:

```text
[mimg-mip] program=0x2005714000 image_load_mip declined pc=33 shape=0 proven_zero_mip=0
           img_dim=5/1 samples=1 mips=12 mip_tail=0 compressed=0 array_in_gfx=0
           addr=0x2026900000 2048x2048x1 dataformat=1 ncomp=2 tile=27 dmask=0x3 unorm=0 glc=0
           layer_stride=0
```

`IMAGE_LOAD_MIP` where the *resource* declares `2D_ARRAY` with a **12-level mip chain** while the
*instruction* addresses it `dim:2D`, and the mip operand is not one of the recognised provably-zero
shapes. `rdna2_emit_alu.cpp` only ever specialises this op away after proving the mip is zero and
the resource is single-level; a 12-mip resource has no such proof, so it declines. Tracked as
[#2818](https://github.com/mattias800/prosper/issues/2818).

### `image_get_resinfo` on a 2D_ARRAY T# no longer declines (2026-08-29)

Separate from the three scene-width kernels above, one stage program (`0x200581bb00`, `1x1x6` groups
of `256x1`) declined at **pc=31** on `image_get_resinfo` with `dim=5` (2D_ARRAY), `dmask=0x3`. The
SPIR-V lowering had already been widened for the arrayed form by #325 — `image_get_resinfo` branches
on `tex_is_arrayed(binding)`, asks for the ivec3 query and reports its third component as the layer
count, which is exactly what GET_RESINFO's third result means for a 2D_ARRAY T#. What still declined
were the two *gates* in front of it: `emit_alu`'s dim dispatch and the coverage predicate — the same
lag #2265 records for the atomic coverage predicate.

Measured on the route, `boot_trace` + `PROSPER_COMPUTE_TRANSLATE_ONLY=1`, compared at the same census
denominator (1,048,576 dispatch decisions over 32 programs):

| | reject site | opcode |
| --- | --- | --- |
| before | pc=**31** | `fmt=14 op=0xe` — `image_get_resinfo` dim=5 |
| after | pc=**130** | `fmt=4 op=0x5` — `s_cbranch_scc1` |

`op=0xe dim=5` goes from one occurrence to **none** anywhere in the run, and the program advances 99
instructions to an unrelated gap.

**No program leaves the skip list and no image change is claimed** — 14 listed before and after. This
closes one gap on one program's path; that program is still blocked further along, and the three
scene-width kernels above are untouched by it.

CUBE (`dim=3`) is deliberately still declined — two other stage programs issue it (`0xf0380118`,
`dmask=0x1`), and admitting it would need the stacked-face lowering (#273) to promise what
GET_RESINFO's third result means for a cube, which this did not establish.

### `image_load_mip` is the FIRST blocker of at least two, not the only one (2026-08-21)

**This section used to say "All three now block on exactly one instruction", and #2818's summary
still says `IMAGE_LOAD_MIP` is "the single remaining blocker for all three". That is falsified.**
Behind the MIMG decline, all three programs stop again on **`s_getpc_b64`**.

The A/B is one variable in one binary: a **measurement-only** build (never merged, and it must never
be) that accepts the declining `IMAGE_LOAD_MIP` as an ordinary LOD-0 `OpImageFetch`, discarding the
guest's mip operand. That result is knowingly WRONG whenever the guest asks for a level other than
zero — it exists only to answer "is anything behind this reject", and it is the cheapest instrument
that can answer it, because the recompiler aborts at its first fatal site and therefore reports
exactly one.

| program | reject on master | reject with `IMAGE_LOAD_MIP` accepted |
| --- | --- | --- |
| `0x2005714000` | `pc=33 words=f0040308 fmt=14 op=0x1` (`image_load_mip`) | `pc=517 words=be801f00 fmt=1 op=0x1f` (`s_getpc_b64`) |
| `0x2005717e00` | `pc=81 words=f0040308 fmt=14 op=0x1` | `pc=527 words=be801f00 fmt=1 op=0x1f` |
| `0x200571bd00` | `pc=81 words=f0040308 fmt=14 op=0x1` | `pc=527 words=be801f00 fmt=1 op=0x1f` |

**The census does not move**: 13 programs listed and every one `executed=0`, at
`65536 dispatch decisions over 30 program(s)`, in both arms, with the three scene-width programs'
skip counts unchanged (39 / 40 / 335 against 39 / 40 / 317). So this is the fourth measured instance
of "a decline was cleared and nothing about the frame or the census changed", and the first where
the reason is visibly that another decline was waiting one route further in.

`s_getpc_b64` is rejected by `rdna2_emit_alu.cpp:1062` unless the PC-relative embedded-table
pre-pass (`detect_pcrel_tables`, `rdna2_cfg_support.hpp`) folded a table load from this shader — the
same family as R-Type Delta's #2783. **That half is now fixed** ([#2859](https://github.com/mattias800/prosper/issues/2859)):
the pre-pass recognised an untyped (`MUBUF`) and a scalar (`SMEM`) consumer but not a **typed**
one, and Frontiers' consumer is `tbuffer_load_format_x v10, v10, s[0:3], 0 offen` at BUF_FMT 22
(`32_FLOAT`) — a one-component 32-bit typed format, which performs no conversion at all, so the
existing constant-lookup fold was already exactly right for its *values*.

The format was only half the proof, and the other half is the more interesting one because it is the
one that would have been silently wrong. A FORMAT load also applies the descriptor's **DST_SEL**
channel routing, which a raw `buffer_load_dword*` ignores — and Frontiers' own table descriptor
carries word3 `0x10005004`, i.e. `DST_SEL = (X, 0, 0, 0)`. Its actual load writes only X and X is
identity, so the live case is unaffected; a four-component typed fetch through that same descriptor
would not have been. The typed fold therefore requires identity routing for the channels the opcode
writes, as well as a conversion-free format. (Whether a typed fetch really does honour DST_SEL is
`CONFIDENCE: MED` and prosper says it two different ways — see #2869 — but the obligation is
fail-closed under either reading, so the guard is right regardless.)

The declining `[mimg-mip]` line's sample above prints `dataformat=1 ncomp=2`: that `1` is prosper's
own `DataFormat` ordinal (`Float32`), **not** the guest `IMG_FMT 64` named in the paragraph below it.
The two numberings collide constantly and the field is labelled to keep them apart.

### The blocker list for these three programs is exactly two, and one of them is now closed

`tools/shader_inspect` on the `PROSPER_SHADER_DUMP` bytes answers "how many blockers" directly, and
nobody had pointed it at these programs. Its generic-coverage enumeration lists every instruction
the per-instruction emitter refuses:

```text
$ shader_inspect exec_cs_2005714000.bin
generic-coverage total=437 alu=420 exp=0 table=3 unsupported=14 first=MIMG/0x1
generic-unsupported pc=0033 fmt=MIMG op=0x1        <- image_load_mip (#2818)
generic-unsupported pc=0344 fmt=SOPP op=0x8        <- s_cbranch_execz, lowered by the CFG dispatcher
   ... eleven more SOPP branches ...
generic-unsupported pc=0517 fmt=SOP1 op=0x1f       <- s_getpc_b64 (#2859)
generic-unsupported pc=0536 fmt=SOP1 op=0x1f
```

All three programs give the same two non-branch families and nothing else. **Read that enumeration
before costing out any single reject** — it is a static per-instruction pass, so it over-reports
(the `SOPP` branches are fine, and it does not run `detect_pcrel_tables`, so it still lists a
`s_getpc_b64` whose table the real pipeline folds), but it bounds the problem from above, which one
terminal reject line can never do.

And the pair is now measured to be the *whole* list. With #2859's fold in place **and** the same
measurement-only LOD-0 build, in a routed `boot_trace` arm at
`65536 dispatch decisions over 30 program(s)`, all three programs **disappear from the census skip
list entirely** — 13 listed before, 10 after, and the three that left are exactly
`0x2005714000` / `0x2005717e00` / `0x200571bd00`. The census lists only programs that skipped at
least once, so leaving it means they recompiled and executed.

So `IMAGE_LOAD_MIP` really is the last blocker for these three now — which is what #2818 claimed
before it was true, and for a different reason.

**How to rebuild the measurement arm** (it is deliberately not in the tree — it renders wrong
content whenever the guest asks for a level other than zero, and a screenshot from it would be
believed): in `rdna2_emit_alu.cpp`, immediately before the `is_zero_mip_load && (...)` decline,
accept the instruction when an env switch is set and `res->sample_count == 1 &&
!res->compression_enabled`. Execution then falls through to the ordinary `image_fetch_2d` at LOD 0.
Run it only under `PROSPER_COMPUTE_TRANSLATE_ONLY=1`, where nothing is submitted.

**The generalisable rule, and it cost this lane its first two hours: a terminal reject line is a
lower bound of one.** It names the site the recompiler stopped at, which is the first fatal one on
whatever route it took — never how many more are behind it. Before costing out a fix for a named
blocker, spend one arm proving there is nothing behind it.

### Where the guest's mip levels actually are

The `[mimg-mip]` line now carries the resource's identity, and with `PROSPER_TDUMP=1` the picture is
unambiguous. The declining resource is **2048x2048, `IMG_FMT 64` (32_32_FLOAT -> RG32F),
`type=13` (2D_ARRAY) with `depth=1`, `tile_mode=27` (SW_64KB_R_X), `BASE_LEVEL=0`, `LAST_LEVEL=11`,
`MAX_MIP=11`** — a complete 12-level pyramid of a 2048x2048 two-channel float surface, which is
exactly 12 levels for a 2048-wide chain.

And the guest binds **thirteen** descriptors to one such allocation:

```text
[tdump] t=2025e500 c4000000 01ffc1ff d1b0022c 00000000 007000b0 ... mips=0..0  max_mip=11 depth=1
[tdump] t=2025e500 c4000000 01ffc1ff d1b1122c 00000000 007000b0 ... mips=1..1  max_mip=11 depth=1
        ... one per level ...
[tdump] t=2025e500 c4000000 01ffc1ff d1bbb22c 00000000 007000b0 ... mips=11..11 max_mip=11 depth=1
[tdump] t=2025e500 c4000000 01ffc1ff d1bb022c 00000000 007000b0 ... mips=0..11 max_mip=11 depth=1
```

Twelve **single-level** views, one per level, plus one **whole-chain** view. That is a pyramid the
guest builds itself: each level is written through its own single-level descriptor and the finished
pyramid is read back through the chain descriptor at a runtime LOD.

**So the guest's mip levels are ordinary guest memory**, at the byte offsets
`tiled_mip_level_layout` already computes — and prosper's existing single-level path places each of
them correctly *today*, one descriptor at a time (`image_base_level_view` applies `BASE_LEVEL`'s
`mip_offset`, and SW_64KB_R_X is one of the tile modes it supports, tail included). What prosper
cannot do is view them together. That matters for whoever takes #2818, because it says the faithful
implementation **uploads real guest bytes** rather than synthesizing levels, and the machinery to
locate each one is already written and already exercised by the guest's own twelve descriptors.

**What the renderer can and cannot do with mips today**, because getting this wrong sends the next
investigation to the wrong file:

* The **graphics** path does build chains, gated to plain-2D (`img_dim == 1`), depth-1, non-storage,
  non-RTT, RGBA8 sampled textures declaring `declared_mip_levels > 1`
  (`tests/fixtures/render_runner.h:6044-6055` — that file is the live offscreen Vulkan backend,
  included by `frontends/shared/live/live_renderer.cpp:39`). `tests/gpu/test_texture_mip_render.cpp`
  is a registered ctest asserting a declared 3-level chain samples level 2.
* The **compute** path did not: its single `VkImageCreateInfo` for guest images set
  `ici.mipLevels = 1` unconditionally. **Fixed by #3048** — see the next section; the sentence is
  kept because everything else in this bullet list is still current.
* **And the graphics chain is GENERATED, not uploaded.** Staging carries level 0 only; levels
  1..N-1 come from a linear-filtered `vkCmdBlitImage` cascade at upload time
  (`render_runner.h:6257-6259`, `:7245`).

Frontiers' resource misses that gate on two counts at once — it is `img_dim=5` (2D_ARRAY) and it is
a compute binding. But the third bullet is the one that matters most for whoever takes #2818:
widening the gate would produce levels **synthesized by downsampling level 0**, not the guest's own
mip data. That would render, and it would be wrong, in the exact way the "no plausible constants"
rule exists to prevent.

**The census did not move when the MUST defect was fixed** — 14 programs, all `executed=0`, before
and after at the same denominator — and neither did the composite. Both were measured, not assumed.

### Compute images now carry the declared chain, and the LOD operand is emitted (#3048)

Both halves of #3048 landed together, because neither works alone: emitting a `Lod` against a
one-level image fetches a level that does not exist, and materializing levels nobody addresses
changes nothing.

* **One derivation, two consumers.** `src/gpu/resources/mip_chain_plan.{hpp,cpp}` answers "how many
  levels does this resource's compute image have, and where does each level's guest data live".
  `live_compute`'s image creation and the recompiler's MIMG lowering both call
  `shader_resource_compute_mip_chain_levels`, so they cannot disagree — the drift that #2265
  recorded (three copies of one predicate) is what this shape exists to prevent.
* **Real guest bytes, not a generated cascade.** Each level is detiled from the guest's own
  `tiled_mip_level_layout` offset, tail levels included, straight into the staging buffer. The
  graphics path's blit cascade is deliberately not reused here: this section's own analysis is why.
* **The provenance the placement needs is now carried.** `ShaderResource` gained the allocation's
  level-zero element extent, its bytes-per-block, its effective `MAX_MIP` and the view's
  `BASE_LEVEL` (`mip_chain_*`), because `gpu_addr`/`width`/`height` have already been shifted onto
  the selected level and nothing downstream could recover the rest. Captures serialize them at
  format v57; a pre-v57 file reads as "not modelled" and fails closed to one level.
  **`gpu_replay` gets the chain too, from capsules taken since #3202** — and needed no new format
  version to do it. A tiled chain stores level zero *last*, so the rest of the allocation lies
  below the address the descriptor names; the capture writer now extends that resource's captured
  range down to the allocation base, and the resource's already-serialized blob offset is then the
  count of owned bytes preceding `gpu_addr`. Replay publishes it as
  `ShaderResource::host_data_prefix_bytes`, and the one derivation both consumers read accepts a
  host-backed resource **only** when its span covers the whole allocation. A capsule taken before
  that change does not, so it still replays with a single-level image and still refuses
  `IMAGE_LOAD_MIP` — visibly, which is the required behaviour: a capture that cannot express the
  chain must decline rather than fetch levels it does not own. Re-grab the frame rather than
  re-reading an old bundle.
* **Reject-by-default.** Linear chains, a selected level packed in the tail, layered or volume
  views, DCC, block-compressed and converting formats, a shifted `BASE_LEVEL`, and a `host_data`
  backing that does not cover the whole allocation all keep the historical single-level image.
  (That last term was an outright refusal of every `host_data` resource until #3202; it is now a
  coverage test, and everything that owns only the selected level — a `--override-resource`
  replacement, a compute-internal snapshot, a synthetic fixture, a pre-#3202 capsule — still fails
  it.) A binding whose shape cannot carry the chain
  (imported/renderer-owned surfaces above all) is declined **fail-visibly** rather than silently
  built with fewer levels than its compiled module addresses. **That decline is wider than it
  needs to be** — it does not depend on the module actually issuing `IMAGE_LOAD_MIP`, because the
  backend cannot see which ops a compiled module contains, so a sampling-only use of such a
  resource on an RTT-aliased binding would be dropped for nothing. It reports as
  `[compute-mip-chain] declined …` on a geometric schedule (`skip_image` alone warns once per
  address for the whole process while the dispatch is dropped every frame, which reads as a
  one-off when it is not).
* **The LOD is clamped** to the materialized level count. Half of that is certain and half is
  inference, and the code says so at the site (`CONFIDENCE: MED`): an out-of-range `Lod` operand is
  UNDEFINED in SPIR-V, so *some* clamp is mandatory whatever the hardware does — but *saturating to
  the last level* is read from doc 70648's BASE_LEVEL/LAST_LEVEL fields, whose MIMG section does not
  spell out the out-of-range behaviour, and no capture here exercises one. Returning zeros is the
  other candidate.

**Not yet measured on this title.** The change is covered by a registered execution test — a real
tiled nine-level guest chain, dispatched through the production compute backend, asserting the
result is *level one's own texels* — but no routed arm has been run since, so nothing here claims
the Cyber Space world renders. Two things are worth measuring next, in this order: whether the three
scene-width programs now leave the census skip list on a **default** build (they did on the
measurement-only LOD-0 build, which is a weaker statement), and only then the composite's non-black
percentage and bounding box. The `## Ruled out` row above — "fixing a recompiler decline that
unblocks these programs will change the frame" — is **not established**, and has now failed four
times; do not assume the fifth.

## Ruled out

One line per falsified hypothesis with the evidence that killed it. Read this before forming a new
one. Twelve rows were established on #1968 / #2023 and are copied here so they survive those issues
being closed; the rest are this document's own. **Do not restate the row count in this paragraph** —
a stated total is stale as soon as the next lane appends, and every lane that adds a row would have
to remember to update it. The last one did not (review of #2820).

| Hypothesis | Verdict and evidence |
| --- | --- |
| Making `gpu_replay` materialize the declared mip chain needs a new capture-format version, because the placement the reader must know is not in the file | **Falsified.** The placement *is* in the file, and had been since before v57: `collect_intervals` anchors a resource's captured range at an interval whose `begin` may sit below `gpu_addr`, and `assign_blob_range` already serializes `blob_offset = gpu_addr - interval.begin` — which is exactly the count of owned bytes preceding the descriptor's address, because a blob's byte *i* is the guest byte at `blob.guest_addr + i` by construction. Only the *writer* had to change (own the allocation, not the selected level); the reader publishes an existing serialized number as `host_data_prefix_bytes`. Pre-#3202 capsules fail closed on the same number without a version gate. #3202 |
| All three scene-target-width kernels are blocked by `s_cbranch_execz` — a stronger claim than #2790's census, which records it for two of the three | **Partly falsified — two of the three moved, one never did.** The `V_LSHL_ADD_U32` allowlist entry the census predates has landed, and re-measuring on `33417dca` shows `0x2005717e00` and `0x200571bd00` no longer rejecting on `s_cbranch_execz` at pc=28 but on `image_load_mip` at **pc=81** — "a reject PC names where a fact was consumed, not where it was lost", exactly as that commit predicted. `0x2005714000` did **not** move: the census already recorded it at pc=33 `image_load_mip`, byte-identical. So all three of THESE THREE kernels now stop on the SAME instruction — which narrows this particular group to one unimplemented operation, and is **not** a claim that the title's world needs only that: `image_load_mip` is the first blocker of at least two (see the section above, and #2859 for the `s_getpc_b64` half that has since landed). Their `[mimg-mip]` profiles are identical too — `img_dim=5/1 mips=12 addr=0x2026900000 2048x2048`. Note `image_get_resinfo` at pc=73 belongs to `0x2005a13f00` / `0x2006e24000`, **not** to any of these three; an earlier revision of this row mis-attributed it by pairing separately-collected program and reject lists. #2790, #3048 |
| `boot_trace` cannot position a flip-anchored route because it prints no flip counter | **NOT a falsification — this row was wrong and is retracted.** It was written from a 180 s arm that produced 155 lines with no flip marker, but the arm simply had no counter switch enabled. Two exist, both in `src/hle` and so linked into `boot_trace`: `PROSPER_PAD_SCRIPT_LOG=1` prints `[pad-script] elapsed=… frame=<flips since first poll> read=… buttons=…`, i.e. the route's own flip position per input (`hle_pad.cpp`, `docs/INPUT_REPLAY.md`), and `PROSPER_PROGRESS=<secs>` prints `flips=` from `prosper_vo_flip_count()`. Structurally it could not have been true either: `fN` anchors are resolved by `pad_frame_now()` reading that same counter in every frontend. Kept as a row because the mistake is the reusable part — **a null from an instrument whose switch was never thrown is not evidence**, and this document already says four times over that the CPU-only `boot_trace` arm IS the fast loop and does reach the stage. The operational note I attached to it does **not** stand either: I saw one `boot_trace` survive a plain `timeout`, but the tree says the opposite (`hle_agc.cpp` — "a `timeout`-bounded diagnostic run dies on SIGTERM") and this document's own recipe uses `timeout --foreground -s TERM -k 5s`. I first attributed that to the missing `--foreground` flag; **that explanation is backwards** — GNU `timeout` *without* `--foreground` signals the whole process group, which is the more lethal configuration, so omitting it cannot explain a survivor. The observation stands unexplained; use the established recipe and do not build on it. ([#2790](https://github.com/mattias800/prosper/issues/2790), and PR #3049 which retracted this row.) |
| The unregistered `libSceFont` surface is why some of this title's text or UI fails to draw | **Falsified, by counting.** This dump names `libSceFont` and `libSceFontFt` only as `_stub_weak` entries in its dynamic library table, and its eboot and every `sce_module/*.prx` import **zero** of the 156 NIDs those two libraries export. Positive control on the same sweep in the same run: *Metaphor* 25, *Astro Bot* 54. Confirmed by the authoritative instrument too — `self_dump --import-slots` reports 0 import slots here against 25 / 54 there. A library named in a binary is not a library called from it. #2951 |
| The stall after `ui_gamemodeinitialize.pac` is a rendering, present or publish defect | **Falsified.** It was one unregistered Sony import. With `RjMlsR8EXrw` registered and nothing else changed, the same binary reaches the title screen and the main menu — while the renderer, present path and publish gate are byte-identical. (This PR.) |
| The guest stops submitting GPU work after the opening logo | **Falsified.** A same-process A/B across the freeze recorded 11 `agc_driver_submit_dcb`, 4 draws, 7 dispatches and 1 flip per frame *after* it, against 15/1/3/1 during the intro. (#1968.) |
| The SONIC TEAM logo movie never signals completion and UI init waits on it | **Falsified.** `PROSPER_DENY_SUBSTR=.usm` makes the open fail and the guest reaches the *identical* terminal state — 177 file opens against the control's 178, the difference being exactly the denied movie. (#1968.) |
| The stall is prosper's Videodec2 HLE, or the AvPlayer consumer-driven-EOF defect (#1973) | **Falsified — neither path is reachable here.** The eboot's 39 `DT_NEEDED` entries name no `libSceVideodec2`, `libSceAvPlayer` or `libSceAjm`, and neither do the three `sce_module/` PRXs; decode is CRI Sofdec2, statically linked. (#1968.) |
| Some guest thread is blocked on an unposted wait | **Falsified.** `guest_bt --all` over all 60 guest threads found every one parked in an ordinary idle wait, with the main thread in the engine's own frame limiter. (#1968.) |
| The guest polls a Sony service that never changes / waits on a Sony answer | **Falsified at the service layer.** A 154-handler sweep over a 15-flip window recorded exactly four calls per frame: `sceUserServiceGetEvent`, `sceSystemServiceGetStatus`, `sceSaveDataUmount2`, `agc_dcb_set_flip`. The wait was not a *poll* — it was an ENOENT retry the service layer cannot see. (#2023.) |
| The guest is blocked on an **unimplemented** NID | **Falsified as stated, and it was the right question asked with a blind instrument.** The `PROSPER_PROGRESS_UNIMPL` arm that returned "3 distinct, 5 calls, none polled" ran a route on which the guest had **no save data**, and Frontiers only reaches the PS4-transfer probe after reading `/savedata0/optiondata`. Re-run against a populated save area, the same instrument reports a fourth NID called **1,319 times**. A per-title save area is part of the route (see *Reproduction*); a census taken on one route does not bound the other. (#2023, this PR.) |
| #657's skipped `64x64x6` layered-image dispatch contributes | **Falsified.** It fires exactly twice, ~20,000 submits before the wall, and never again. Still a real gap; not this. |
| #1967 (`libSceJson2`) is the proximate cause | **Falsified.** Three one-shot calls early in boot; no json2 handler is entered during the stalled window. Still a real latent defect. |
| The declined ordered-DMA submit of #1982 | **Falsified.** `grep -cE 'ordered DMA submit (rejected\|not executed)'` = 0 across every arm of this title. |
| Same defect as Little Nightmares III (#1962) | **Falsified — the opposite shape.** Here `present_count` climbed while `frame_seq` froze and submits kept arriving; there both froze together. |
| `pixel_crc32=666f7b3f` links this to #1962 / #1982 | **No — it is just "black 3840×2160"** and recurs on unrelated titles. Never group by a black-frame hash. |
| The frame going black shortly *before* the publish wall shares the wall's cause | **Falsified.** With the publish wall removed (#1990) the black survived unchanged; the last publishable frame was already black. Two defects. |
| A no-input arm sits on the title screen because prosper stalls, or because the menu is unreachable | **Falsified.** A twelve-page modal notice queue is open over the menu. Twelve confirms clear it and the thirteenth activates a menu entry; the same binary then reaches `GameModeStage`. Nothing in the emulator was changed. (This document.) |
| The panel over the menu "renders almost none of its text" (#2206) | **Incomplete rather than wrong.** With any face button pressed, the same panel renders its header *and* body correctly — "Extras: The acquired additional content will be accessible from the Extras menu.", "Update Details: The following content has been added in the update: -Action Chain Challenge -New Koco -Birthday Decorations -Status map -New Game+". The blank panel is the *no-input* state of a queue nobody had advanced. |
| The title heading permanently draws "Try Again" instead of the logo (#2206) | **State-dependent, not permanent.** With the notice queue cleared, the SONIC FRONTIERS logo renders correctly at 3840x2160. In the same frames the six main-menu entries do *not* render their text, while the original rung-2 capture rendered the entries and got the heading wrong — the two are anti-correlated, which points at string/element resolution rather than at the text renderer. |
| The d-pad can drive the boot notice queue | **Falsified.** A twelve-button sweep advanced the panel on every face button and on none of `up`, `down`, `left`, `right`. A route that used directional windows there would silently deliver nothing. |
| Pressing confirm often enough clears the notice queue | **Falsified.** 405 confirms at 20-flip spacing reached exactly the same state as 6 did; 12 confirms at 40-flip spacing cleared it. Presses landing inside a page's transition animation are discarded, so spacing decides the outcome and volume does not. |
| The black world in the stage is an artifact of `--warmup-seconds` skipping the renderer past the stage's setup | **Falsified by a control arm.** With `--warmup-seconds 90` the renderer is live continuously from flip 1517, before `GameModeStage` loads at flip ~2900, and the same HUD-over-black frames appear from flip 3429 to 4325 with the same 2880-wide composited rect. (#2790.) |
| The black world is a compositing/present defect | **Falsified as the primary cause.** Sixteen of the stage's thirty-two compute programs have `executed=0`, three of them 2880-thread-wide screen-space passes; and the same build composites the in-engine cutscene correctly at full width. The composite is downstream of a scene target that was never shaded. (#2790.) |
| The three scene-target-width stage programs are blocked by `s_cbranch_execz` and `image_load_mip`, the encodings their reject lines name | **Half falsified.** Both were reported by the straight-line emitter, two routes downstream of the decline that mattered. Live, with `PROSPER_DBG_PROGRAM` on each address, the CFG dispatcher declined `wave64-ambiguous-mask-read` at pc481 / pc481 / pc471 — one missing entry in `scalar_alu_source_words`, which charged `v_lshl_add_u32 v7, v6, 2, vcc_lo` (a 32-bit read of VCC_LO used as scalar scratch) the whole VCC pair. With it fixed that decline occurs **0** times, all three reach the dispatcher body, and all three converge on `image_load_mip`. **`s_cbranch_execz` is dead as a lever here** — the dispatcher lowers it fine. See `RECOMPILER_REMAINING.md` § Ruled out. (This row said "`image_load_mip` **alone**"; the row below falsifies the *alone*, not the rest of it — `s_getpc_b64` is waiting behind the MIMG site.) |
| Fixing a recompiler decline that unblocks these programs will change the frame | **Not established, and twice now it has not.** #2758 took `executed=0` to `executed=6` with no image change; #2801 cleared five SCC-site declines with no image change; this change cleared three dispatcher declines with **no census change at all** (14 listed, all `executed=0`, both arms at `262144 dispatch decisions over 32 program(s)`) and no composite change. A fourth instance is the row below: accepting `image_load_mip` moves the reject on and changes neither the census nor the frame, because another decline is waiting behind it. Measure the composite separately — non-black percentage and bounding box — before claiming anything about the world. |
| `IMAGE_LOAD_MIP` is the single remaining blocker for the three scene-width stage programs (#2818, and this document's own earlier wording) | **Falsified by a one-variable A/B.** A measurement-only build that accepts the declining `IMAGE_LOAD_MIP` at LOD 0 moves all three programs' terminal reject from the MIMG site to **`s_getpc_b64`** (`be801f00`, SOP1 op 0x1f) at pc 517 / 527 / 527, and leaves the census byte-for-byte where it was — 13 listed, all `executed=0`, at `65536 dispatch decisions over 30 program(s)`. `image_load_mip` is the first of two blockers, and the second — the PC-relative embedded-table fold refusing a TYPED consumer — is fixed by this PR (#2859). With both cleared the three programs leave the census skip list entirely, so the pair is the complete list. **A terminal reject line is a lower bound of one**: the recompiler aborts at its first fatal site, so it can never say how many are behind it; `tools/shader_inspect`'s generic-coverage enumeration bounds it from above in one command. (#2859, this document.) |
| A live `tools/screenshot` arm may use `--warmup-seconds` to reach the stage faster on this route | **Falsified in two arms, and the failure is silent.** `--warmup-seconds` suppresses Vulkan rendering, which raises the guest's flip rate to ~17-18/s (6,994 flips in 420 s on one arm, 1,669 in 90 s on another) against ~3.2-3.5/s once the renderer is live — but the boot's own progression is paced by asset loading and movie playback in WALL CLOCK, not by flips, so a flip-anchored route fires its windows against a guest that is nowhere near the state they were authored for. At `--warmup-seconds 90` the five `up` presses at f1700-f1940 were delivered ~50 s BEFORE the main menu existed, and the arm then activated "Extras" with the f2100 confirm and sat in the notices list for the rest of the run (open the frames: sample 7 is the title screen with the cursor still on "Extras", sample 24 is the "Update Details" notice page). At `--warmup-seconds 420` the same route produced **40 identical all-black 3840x2160 frames, one distinct frame in 3,752 publications, and 5 compute programs seen** — which reads exactly like "this title renders nothing", and is apparatus. Run the live arm with **no warmup**; the CPU-only `boot_trace` arm is the fast loop and it does reach the stage. (This document.) |
| A routed `boot_trace` arm on this title reaches `GameModeStage` reliably, so its census can be read without checking | **Falsified — it desynchronises under HOST LOAD, and the failure is silent.** Measured across nine routed arms on one host, same route: the ones taken while the machine was quiet reached the stage (`65536 dispatch decisions over 30 program(s)`); the ones taken while other agents were building and linking on the same box did not (`over 5 program(s)` at the same denominator, having sat in the menus for the whole run). `uptime` read a load average of **28** during the failing streak. This is the same mechanism as the `--warmup-seconds` row above — the route's windows are anchored on display flips while the boot's own progression is paced by asset loading and movie playback in wall clock, so anything that moves the flip rate relative to wall clock moves the inputs off their targets. On a shared machine that is **not observable from inside the arm**, so: check `uptime` before starting, and **read the program count on the census line before believing anything in the block.** 5 means the arm never left the menus and every number under it is about a different part of the game; 30 means it is in the stage. A census quoted without that number is worthless. (This document.) |
| The intro's 3.3 fps is GPU-bound | **No.** `gpu-device` is 272.1 ms of a **5020 ms** F8 window (5.4%), the main thread is 2.1% on `drm_syncobj_array_wait_timeout`, and process CPU is 1.01 of 32 cores. An independent 4760 ms window taken after #3406 reads 287 ms (6.0%), so the verdict does not rest on one capture. **Do not pair 272 ms with the 4760 ms window** — that is two captures spliced, and it does not divide to 5.4%. The GPU is idle waiting for the host. #3405 |
| `seed_reprove` re-proving costs the frame | **No.** A/B with `PROSPER_SEED_REPROVE=0`, same route and capture window: guest flips **3.39/s in both arms**, and the hot compute program's setup was 982.7 ms vs 980.7 ms. Its `[seed-skip-verify]` lines name 4K textures and the same code address, which is what made it the obvious suspect. #3405 |
| The per-dispatch `linear_seed` allocation is the cost | **No.** Implemented (a `thread_local` vector reused across dispatches, removing the allocation entirely), measured, and **reverted**: flips 3.36/3.19/3.39 against a 3.39 baseline, and every symbol in the perf profile unchanged within noise. `linear_seed` is only reached on the `atomic_image` and `zero_padded_tail` branches and nothing established the hot program takes either, which is what makes the null decisive rather than merely negative. **Scope:** the reuse's effect was not independently counted, so this kills the named allocation and **not** the broader per-dispatch-staging class — #3406 then found that class real in a different form (the mapping, not the allocation). #3405 |
| The compute image cache is thrashing | **No.** `PROSPER_COMPUTE_IMAGE_CACHE_MB=8192` left the hot program's setup at 1011-1110 ms against 1147 ms; the limit is honoured uncapped (`persistent_compute_image_limit`, `live_compute.cpp:1171`), so that was a real arm. Uploads were already being skipped: `upload-skipped=1` on all 19 persistent 33 MB storage images. **No eviction count was read, so the sweep bounds the effect rather than proving the cache healthy** — the structural reason it cannot be thrashing is the `not_native_exact` row in this same section: these planes never reach the cache gate at all, because `live_compute.cpp:7293` skips the whole guest-source block for a renderer-owned surface. #3405 |
| The compute memory pool is too small | **No.** `PROSPER_COMPUTE_MEMORY_POOL_MB=8192` left setup at **1130 ms against a 1147 ms baseline**; the limit is honoured uncapped (`compute_memory_pool_limit`, `live_compute.cpp:1328`). *(An earlier wording quoted "flips 3.99/s" for this arm with no paired control. It has no baseline in this document and the nearest one — 3.39/s, two rows up — is from a different capture, which makes a null read as an 18% win. Use the setup pair, which is controlled.)* Note the field `allocation-reused` in `[compute-image]` is `bi.forced_seed_allocation_reused`, a seed flag — **not** a pool hit rate; reading it as one produced a wrong conclusion first time round, and nothing currently reports the pool's real hit rate. #3405 |
| Staging is write-combined, so the copies are slow | **No — but this is a code-reading result, not a measurement.** `host_memory_type` (`live_compute.cpp:3233`) tries `HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED` first and only falls back if the device exposes no such type, and `persistent_ds_transfer_buffer` (`tests/fixtures/render_runner.h:4330`) deliberately selects it for readbacks, its own comment recording ~236 MB/s measured there. No run is quoted confirming the selected `memoryTypeIndex`'s flags on this device; on RADV/STRIX_HALO the fallback is unlikely to bite, but one line of "the selected type reported `HOST_CACHED`" would make this a measurement. #3405 |
| `upload_mapping.unmap()` costs the frame | **No.** A dedicated sub-timer on `upload_mapping.unmap()` recorded **zero** records above 0.5 ms across a full run. **The sub-timer's own record count was not retained**, so what this establishes is "no unmap teardown exceeded 0.5 ms in that run", not "the timer fired N times and none exceeded" — the two read identically here. Re-run with the count if you need the stronger form. #3405 |
| `not_native_exact` publishes starve the FMV planes' cache | **No.** Instrumenting the refused triples shows every refusal is a tiny surface — 956 x `2x2`, 34 x `1x1`, a few `128x32`/`64x64`. None are the 4K planes. The planes are uncached for a different reason: they are renderer-sourced, so `live_compute.cpp:7293` skips the whole guest-source block that would have built a cache key. #3407 |
| An FMV always presents new content, so caching conversions cannot help | **No.** The decisive number needs no content sampling at all: over a ~130 s run each converting address shows **441 conversions against 148 distinct `LiveTargetSnapshot::pixels` buffers**, so a pointer-identity key collapses **66%** of them on `const`-means-`const` alone. A three-probe content fingerprint puts redundancy far higher (one 33 MB plane: 497 conversions, 1 distinct sample; another: 497 with 98) — but that is **192 sampled bytes of a 33 MB buffer over two of the three converting addresses**, so read it as an upper bound on redundancy, not as a measured 90%. It is the *dispatches* that repeat, not the frames. #3407 |
| A conversion cache keyed on the renderer publication will collapse the intro's duplicate conversions (#3407 fix 2) | **No — measured at a structural 0% hit rate**, and this is the *implemented* version, not an estimate. The cache holds the `LiveTargetSnapshot::pixels` `shared_ptr`, so pointer identity is a sound content proof. Instrumented at the acquire, `cached-pub[n]` is always exactly `now-pub[n-1]` and never the current one, with `content-valid=1` throughout — the wiring is right, the workload cannot be cached this way. 380 conversions over 181 distinct pointers; pointers recur but never consecutively. **This also corrects how #3407's own "441 conversions → 148 distinct buffers" reads**: that is allocator address reuse, not three conversions sharing a buffer, and the consumer cannot exploit it. #3407 |
| The intro's renderer-owned planes are read back from the GPU once per frame | **No — once per READ, and there are ~896 of them.** `PROSPER_RTT_REMAT_LOG` (new): `reads=448 reuse=0 remat-null=448 bytes=33177600` on each of two 4K planes, i.e. **zero reuse and ~30 GB of GPU→CPU readback in 100 s**. `live_renderer.cpp` publishes `pass_pixels` or resets `rgba`, and under GPU-present (the default since #1270) the readback is deferred so `pass_pixels` is empty — every render pass nulls the CPU publication and the next compute read pays a full 33 MB readback. A reset-site census clears all four explicit `rgba.reset()` sites: none touches these addresses. #3407 |
| The direct GPU import is unavailable for the intro's planes, which is why they take the CPU path | **No — it is available on 571 of 571 candidates and then REFUSED on format.** `direct RTT candidate … available=1 imported=3840x2160/0`. The descriptor requests `DataFormat 11` (`Uint8`) x4 = **RGBA8_UINT**; the renderer's image is `LiveTargetPixelFormat 0` = **Rgba8Unorm**. `direct_sampled_rtt_compatible` refuses a UNORM image against a UINT descriptor — the same eight bits per channel, differently interpreted — so prosper reads 33 MB back, reinterprets it on one core, and uploads 33 MB again. **The project already accepts these are the same bytes**: `live_compute.cpp:8698` sets `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` for exactly this UINT/UNORM class on compute-owned storage images and says so in its comment; renderer-owned colour targets never got the same treatment. The fix is MUTABLE_FORMAT on renderer targets plus widening that predicate — **not** porting tiling maths to a shader, because for this cost there is no conversion to port. #3407 |

### Void, not falsified — do not cite these as settled

- **Pooling the readback destination buffer is worth doing.** Implemented, measured and reverted —
  but the A/B does **not** settle it, and it is recorded here so nobody re-runs the same
  three-per-arm shape expecting it to. What is solid is the bisection: a colour-target readback
  splits into setup **121.6 ms (18.4%)** and the `output.assign` copy **537.7 ms (81.6%)**, so
  pooling attacks the smaller share. What is not solid is the comparison. One arm read flips
  4.59/4.79/5.03 (median 4.79) with `readback` 1195 ms; the other read 5.04/4.98/4.98 (median 4.98)
  with `readback` 1261 ms — so **the two instruments disagree, and the verdict does not depend on
  which arm is which**: whichever way round they are read, the flip median favours one arm and the
  component timer the other, with only the signs swapping. (The run logs are not attached and the
  triples are unlabelled in #3407, so do not "correct" the order and conclude you have overturned
  this.) n=3 per arm, and no pooled-buffer reuse counter was read. It was reverted on the component
  timer plus the 18.4% ceiling, which is a judgement call and **not** evidence. What would settle
  it: the pooled-deleter design in #3407, which makes the destination storage stable enough to drop
  the `assign` memset and copy in parallel — i.e. it attacks the 81.6%. Take both or neither.
  #3407
