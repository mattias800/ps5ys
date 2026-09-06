# prosper — progress blog

**Newest first.** Every screenshot checked into this repository, and the story around the ones worth
a story. Read down from the top and stop when you reach something you have already seen.

**This file is written by hand.** It used to be generated from git history, which meant its captions
were commit subject lines — they said what the *change* was, never what the *picture* is, and there
was nowhere to put "we finally reached gameplay in this one, look." That is the whole point of a
blog, so the generator and its CI gate are gone.

[`COMPATIBILITY.md`](COMPATIBILITY.md) remains the per-title overview and
[`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md) the per-title rung table — that one *is* still generated
from the tracker issues, and still gated, because it is a projection of state rather than a story.

<!-- How to add an entry: see CLAUDE.md, the BLOG.md bullet under PR verification.
     Image-heavy, text-light. Pictures: as many as you have. Words: one sentence. -->

> An entry is evidence of what rendered **on the day it was written**. It is not a claim about the
> title's current state — for that, read the tracker. Nothing is ever removed when a title moves on,
> because the point of a blog is that it records *when* things happened.

## 2026-09-06

### We can now ask a black pixel why it is black, and get an answer

Four causes produce an identical black pixel — nothing drew there, something drew and was rejected,
the shader ran and computed black, or the shader computed a colour and the store lost it — and they
send you to four different files. `tools/pixel_history/` names which one, in a single call, per
event. It is validated against a construction whose answer is known in advance, and that control has
already caught two things we had assumed about RenderDoc and were wrong about — including it
reporting a *neighbouring* draw's colour as the shader output of a draw whose fragment never ran.
We would have believed both on a game capture.

## 2026-09-05

### The sheared Unity titles were never sheared — the build directory was

Five games we had filed as rendering their levels through torn triangles turn out to render
perfectly. The owner played two of them and said so; every re-measurement since agrees, including the
one command the original report was built on. What produced the broken pictures was a build directory
that had been reused across branch switches for days, and every experiment we ran lived inside it, so
every experiment agreed with every other one. Two conclusions built on top of it have been withdrawn.

The lesson is cheap to state and was expensive to learn: a defect that reproduces perfectly in one
worktree and has no witness anywhere else is a defect in that worktree.


### Nineteen draws go missing on this machine's graphics driver, and it is the driver's doing

Follow-up to the entry below, and this one has no picture — the result is a pair of counts. On the *New
Joe & Mac* frame we captured, **nineteen of its eighty-six draws paint nothing at all on this machine's
AMD driver and paint correctly on a software one**, the same nineteen every run. We had left it open
whether the fault was the driver's or ours, since both run shaders we generate. It is not ours: the same
driver returns four different pictures from one unchanged four-vertex draw where the software one returns
the same picture twenty times, and a shader nobody here wrote fails the same way. Whether those missing
nineteen are the whole of the shear is a separate question, and still open — they are far too few pixels
to cover the frame.


### Four sheared Unity titles already render correctly — the shear is decided by the graphics driver

*New Joe & Mac*, *Asterix & Obelix: Slap Them All!*, *Rugrats: Adventure in Gameland* and *Summer Sports
Games* all draw their scene art through huge sheared triangles on this machine's AMD driver. We captured
one frame of each and replayed it offline through a different Vulkan implementation, with no change to
prosper and no change to the captured frame — and all four come back recognisable — three of them cleanly, and Summer
Sports with a grey form still sweeping the upper right that we cannot tell apart from its stadium roof —
so the geometry, the indices and the textures were right all along and something about how the frame is *executed* takes them apart.
Which side is at fault is not settled in this entry; the one above it settles it. Filed on #3374; the
reproduction takes three seconds.

These pictures are **offline replays of a captured frame, not live runs** — the software Vulkan device
is far too slow to play on, so this is diagnosis rather than a rung change.

![New Joe & Mac level 1 on this machine's AMD driver: the art is all there, smeared into triangular shards](assets/screenshots/joe-mac-shear-radv.webp)

![The same captured frame replayed on a software Vulkan device: twin palms, volcano, flower bank, Joe, and the full HUD](assets/screenshots/joe-mac-correct-lavapipe.webp)

![Asterix's forest level on the AMD driver: Asterix stretched into a band across the top of the screen](assets/screenshots/asterix-shear-radv.webp)

![The same frame on the software device: Asterix on the path, both portraits with their health and rage gauges, the score and TOP score](assets/screenshots/asterix-correct-lavapipe.webp)

![Rugrats' nursery on the AMD driver: Tommy shredded into vertical streaks](assets/screenshots/rugrats-shear-radv.webp)

![The same frame on the software device: Tommy at the playpen gate, complete](assets/screenshots/rugrats-correct-lavapipe.webp)

![Summer Sports Games' javelin event on the AMD driver: the stands sheared into grey slabs, the hoarding and the takeoff zone gone](assets/screenshots/summer-sports-shear-radv.webp)

![The same frame on the software device: the Summer Sports Games hoarding, the yellow and purple takeoff zone, the throw reticle and the packed stands. The grey form across the upper right is still there and is probably the stadium roof, but we have not confirmed that. Downscaled from the native 3840x2160 capture](assets/screenshots/summer-sports-correct-lavapipe.webp)

Evergate is the exception, and worth showing for that: its blown-out white frame comes back blown out on
the software device too, so that half is ours whatever happens to the streaks over it.

![Evergate on the AMD driver: a blown-out white and orange frame with diagonal streaks across it](assets/screenshots/evergate-shear-radv.webp)

![The same frame on the software device: the streaks are gone and the blow-out is not](assets/screenshots/evergate-still-blown-out-lavapipe.webp)


### Khazan stops dying eight seconds in, and starts drawing the wrong thing instead

The First Berserker: Khazan aborted at 8.6 seconds of every boot with "PS5 Out of Memory". It was not
short of memory: two functions in a library we do not implement were left unregistered, so prosper
answered them with *success* and never filled in the answer they asked for — and the game read
whatever happened to be on its stack as a count, multiplied it by sixteen, and asked its allocator
for 244 terabytes. Telling it "not implemented" instead keeps the guest alive for the full two
minutes we gave it. What it draws is not the game yet: a yellow quadrant, then this gradient, which
it holds.

![Khazan composites a brown gradient where the picture should be](assets/screenshots/khazan-composite-gradient.webp)

![An earlier frame: red and green pinned to maximum over the top-left quarter](assets/screenshots/khazan-composite-yellow-quadrant.webp)

### Space Adventure Cobra reaches its menu and its desert level again

We now reach Cobra's main menu and its opening desert level, from a route that until tonight produced
a single black frame for its whole run.

![Space Adventure Cobra's main menu at 1920x1080: a red planet against a starfield, Cobra standing at the right in his blue suit and teal cape, and the menu list part-way through its animate-in, showing NEW GAME and the beginning of MULTIPLAYER](assets/screenshots/cobra-main-menu.webp)

![The opening desert level: the lit glass walkway across the bottom of the frame, golden dunes and dark rock outcrops behind it, and a character silhouette on the sand. Large translucent blue polygons are smeared across the middle of the frame and there is a dark band along the horizon -- both are real defects, not compression](assets/screenshots/cobra-desert-level.webp)

`sceAvPlayerJumpToTime` — the call a game makes to move a video to another point — was repositioning
the movie correctly and never telling the game the seek had landed. Cobra waits fifteen seconds for
that message, gives up, and never asks for another video frame, so its opening movies never played
and nothing after them was reached. Most of the games here that seek a video wait for the same
message.

The second picture is the honest one: the level draws, and it draws wrong. Its automatic guard is
still red. ([#2899](https://github.com/mattias800/prosper/issues/2899))

## 2026-09-04

### Stray's world is not going missing on the way to the screen — it arrives, and rounds to black

No picture worth showing: the frame is still the menu over black. The finding is what sits behind it.
We had assumed the game built its picture in one place and copied it to the buffer the screen reads,
and that we were losing the copy. Measuring every render target this time, rather than only the
first, says otherwise — the two display buffers are themselves render targets, attached to passes
that carry draws. Nothing is being lost in transit. What the existing pixel traces
already showed, and what this reframes, is that the world's colour reaches the end of the pipeline so
faint that storing it rounds it to zero. So the question stops being "where did the picture go" and
becomes "why is it arriving that dim", which is a much better question to be stuck on.

### RenderDoc works on prosper now, and the first capture was a beautiful picture of the wrong thing

No picture — or rather 56 of them, all useless. RenderDoc has been installed the whole time and
unusable, because it only starts a capture on a *keypress* and every agent here runs headless.
prosper now drives its in-application API itself, so a capture aims at a frame ordinal like F8 and F9
do. The first working one held zero draws: prosper runs two Vulkan devices, and asked to pick,
RenderDoc took the one that only presents. Details in [#3321](https://github.com/mattias800/prosper/issues/3321).
### The compute half of that same round trip was doing it too

No picture, and the headline is a null. #3309 pooled the *graphics* side's 63 MiB scratch buffers;
the compute side was still allocating one per dispatch, for the same surfaces, at the same size.
Measured here: 14.4 ms to allocate and fill 64 MiB against 3.1 ms into a buffer that is already
mapped — and pooling it **did not move Stray's frame rate at all**, because the surfaces that
actually dominate this route average 19.5 MiB, under the threshold where that penalty exists.
What the same runs did settle is where to look next: compute **setup** turns out to be the larger
half (51.1 s against 46.8 s of writeback, over 62,556 dispatches) and nobody had ever measured it,
and 247 GiB of full-surface snapshot copying happens because a class of dispatches writes about a
quarter of a surface and has to bring the other three quarters in first.
[#3307](https://github.com/mattias800/prosper/issues/3307)

### Two thirds of Stray's biggest render cost was the allocator, not the work

No picture. The largest single leaf in a capture of Stray's 9.76 fps title screen is the graphics
frontend's texture materializer, 1110 of 5020 ms, and 84% of it turned out to be one surface: a 4K
HDR intermediate that a compute dispatch rewrites every frame and the renderer decodes back out of
guest memory every time it is sampled.
The part worth reading is what that decode was spending — it built two 63 MiB scratch buffers per
reference, and past glibc's mmap threshold that is not an allocation but sixteen thousand page
faults and a munmap, so **17.8 of the 26.0 ms was the allocator and the kernel rather than the
detile**. The buffers are pooled now and the tiled source is read where it already lives, because
guest memory is mapped 1:1 and copying it only ever produced a second address for the same bytes.
That leaf went 1110 ms to 498, and the title screen from **9.76 to 11.55 frames a second** — the
picture is unchanged, which is the point.
[#3309](https://github.com/mattias800/prosper/pull/3309)

### The line that said the texture was missing was printing the wrong field

No picture. Stray's title screen renders its menu over black, and five image samples report their
descriptor unresolved; the reject line prints four provenance fields, all empty, which everyone
(including this lane, at first) read as "the texture is not there". Four of the five sites also say
the shader never wrote the register the descriptor lives in — which means none of those four routes
could have fired, so the emptiness restated the question instead of answering it. The route that did
run was the one field the line never printed, and in three of the four it had found something the
whole time: a constant buffer sitting at exactly the requested register, thrown away for being the
wrong kind of thing without a word in the log.
[#3126](https://github.com/mattias800/prosper/issues/3126)
[#1634](https://github.com/mattias800/prosper/issues/1634)
### Stray was refusing a compute kernel over a register it had not read yet

No picture — the kernel still does not run, and the reason it was skipped is the interesting part.
One of Stray's title-screen compute programs died four dwords in, on `v_lshl_add_u32 v11, s14, 3, v0`
— its own "which thread am I" arithmetic. The instruction was never the problem: we have lowered it
for months. The problem was `s14`, which this shader uses as the workgroup id *and*, 153 dwords
later, as somewhere to park M0 while it does something else. Our containment for that parking spot
was marking the register unreadable at every block in the shader, including the first one — so the
kernel was refused at instruction four for something it does at instruction 157. It now reasons about
which saves can actually reach a given block — and The Plucky Squire turned out to have a kernel
failing on the byte-identical instruction at the byte-identical address, which also advances. Neither
game looks any different yet; both kernels stop a little further along.
[#3308](https://github.com/mattias800/prosper/issues/3308)

## 2026-09-03

### The frozen frame that gave five pictures now gives one, 3,875 times running

No picture, and the number is the point. #2945 is the smallest reproduction we have of the
renderer's nondeterminism — one captured frame, replayed offline, coming back different every few
runs. Re-measured overnight it did not vary once: 3,875 replays across two captures, two builds and
two Vulkan drivers, over a 1.75-hour window, against a recorded 5-distinct-hashes-in-15 a
fortnight ago.
The verdict is still not "fixed", because the bare-Vulkan control that reproduces the same defect
with no prosper code in the process stayed quiet too — and a quiet window looks exactly like a
repaired renderer, so the new campaign tool reports UNDECIDED and names the two things it cannot
tell apart.
[#2945](https://github.com/mattias800/prosper/issues/2945)

## 2026-09-02

### The same thing was true of every compute dispatch, and the doc said that half was already right

No picture again. The note we filed while fixing the graphics readback said the compute backend was
one of the two places in the tree that already got this right — it isn't: the barrier it has is on
the comparator's one-word flag, not on the dispatch results that become guest memory. Deleting the
new barrier leaves a 439-assertion compute test, including a byte-exact image writeback, entirely
green.
[#3249](https://github.com/mattias800/prosper/issues/3249)

### The five hazards nobody could see, and two instruments that were answering wrongly

No picture. Vulkan's synchronization validation had never run anywhere in this project, and switching
it on found five real hazards — including a mip chain being cleared and copied into with nothing
ordering the two writes, whose corruption would have looked exactly like the black level the clear is
deliberately there to produce. Two env-gated diagnostics turned out to be misusing Vulkan too, and
both were misreporting because of it: the geometry probe was arming on shaders it could not capture
and calling the resulting silence "the draw produced no primitives", and the draw-isolation pass was
naming a culprit draw while re-rendering under different state from the pass it was isolating.
[#3248](https://github.com/mattias800/prosper/issues/3248)

### Every frame we have ever read back was read without asking the GPU to hand it over

No picture, and that is the finding: on this hardware a readback with no host-availability barrier
comes back looking perfectly correct, so nothing has ever complained. Deleting the barrier we just
added leaves the pixel check green and only the structural check red — and Vulkan's synchronization
validation, which is armed and reporting five other hazards in the same run, cannot see it either,
because it has no way to watch the CPU read a mapped pointer.
[#2944](https://github.com/mattias800/prosper/issues/2944)

### The one setting that made compute validation cheaper was the one our own test binary could not survive

No picture. Cached compute sources are proven unchanged by a full byte compare unless a cheap
page-protection watch has been armed first, and the only setting that measurably armed them crashed
`test_game_compute` — because that binary tells the watch layer it has a SIGSEGV handler and never
installed one. So the mechanism was untestable in the harness that would have proved it, and the crash
read as a defect in the emulator. The harness now installs the handler, and a census inside the decision
reports what each setting actually buys.

### A guest asking to sleep for 584 years was served in zero milliseconds

No picture — the finding is that prosper's guest sleeps saturate a hostile interval to the largest
unsigned nanosecond count, and the clock underneath then reads that as a *signed* one and hands back
a deadline in the past. So the clamp written to stop a long sleep becoming a short one produced the
shortest sleep there is. Fixed alongside `select()`-as-sleep, which was the last guest sleep still
resolving on Windows' ~15.6 ms scheduler tick.
[#3038](https://github.com/mattias800/prosper/issues/3038)

### On Windows, every `sinf` the guest ever called returned whatever was left lying in xmm0

No picture — this one is Windows-only and invisible on the box we develop on. The import trampoline
that converts a guest call to the host ABI moved integer registers and nothing else, so a float
argument, everything behind it, and every float *return* were wrong: 78 functions, most of them the
maths library. It now builds the conversion from each handler's own declaration, and a new test
executes the real trampoline bytes across the ABI boundary on any machine.
[#2955](https://github.com/mattias800/prosper/issues/2955)

### The "exotic" image instruction blocking a Stray draw was an ordinary one, spelled differently

A three-dword `image_load_mip` had been rejecting a whole Stray fragment shader, and the extra dword
looked like an unknown packet. It is RDNA2's NSA address encoding: the same `[x, y, mip]` the
two-dword form carries, with each address naming any register instead of the next one along — so the
bug was our rule about how many address dwords an instruction may have, not the instruction. The
same fix also stopped refusing textures small enough that their entire mip pyramid fits in one tile.
[#3237](https://github.com/mattias800/prosper/pull/3237)

### The shader-conformance scanner stopped carrying its own copy of the RDNA2 decoder

No picture in this one — the finding is that `shader_histo` has been reading past the end of an
array on every eboot with an undecodable instruction, and segfaulting outright on 4 of the 55 local
dumps. Its hand-written list of RDNA2 format names went one entry short when `VOP3P` was added to
the enum, so `VOP3P` printed as "UNK" and `Unknown` indexed off the end
([#3229](https://github.com/mattias800/prosper/issues/3229)).

We found it while deleting a different copy of decoder knowledge: `shader_conformance/scan.py`
carried a Python port of the recompiler's instruction-length rules so it could tell an image
instruction from an operand dword that looks like one. It now calls prosper's real decoder instead
([#3184](https://github.com/mattias800/prosper/issues/3184)). The port turned out to be exactly
right — 11,266 real shaders out of the dump library, 199,521 image instructions, zero disagreements
— which is the point: a copy that is correct today is still the one nobody updates tomorrow.

### Every async-compute indirect dispatch a PS5 game makes was being thrown away

No picture — this one is invisible until you look for it, which is the point. `sceAgcAcbDispatchIndirect`
hands prosper the *address* of its dispatch arguments, not an offset, because the async-compute ring
has no base register to offset from; we were sharing the graphics ring's builder, which kept 32 bits
of that address and threw the rest away. Astro Bot's world map lost a full-screen lighting pass to it
on every boot, and so would any other title that dispatches from the async queue.
[#3218](https://github.com/mattias800/prosper/issues/3218)

### Astro Bot's world map is lit, and no longer takes the GPU down with it

The world map now renders and keeps animating for the whole run, where before it froze on a white
screen a minute in — that freeze was a RADV device reset, and it is gone (0 submission failures
against 65, 62 and 57 on three control runs).

![Astro Bot at 3840x2160, the world map 90 s into a default launch: a deep-space field of stars and drifting pale-blue shards, light shafts crossing the frame, coloured confetti and green fragments down the left edge, all of it animating](assets/screenshots/astro-bot-worldmap-lit.webp)

![The same run at 150 s: the camera has moved, a large white-and-red framed panel now fills the lower half and the starfield continues behind it -- the frame is still changing, not frozen](assets/screenshots/astro-bot-worldmap-lit-later.webp)

This is what the same route showed before, at the same moment — the frozen frame after the reset:

![The control run at 150 s: a completely white frame. The GPU was reset at around 75 s and every frame after it is this same picture](assets/screenshots/astro-bot-worldmap-device-reset-frozen.webp)

The cause was one line of binding policy. When a guest compute program uses GDS, the recompiler
hard-codes descriptor binding 127 for it — but prosper's binding assignment renumbered every
resource in the table, that one included, so the emitted shader asked for binding 127 and the
runtime table no longer had it. The descriptor check correctly refused the pair and the dispatch was
declined, every time, for the life of the process. Five of Astro Bot's compute programs were
disabled that way on every boot, one of them the pass that builds the per-tile light lists — so the
world-map pixel shader was handed an arena of nothing but zeros, and because its walk stops on
`0xffffffff` rather than on zero, an empty list is an infinite one. The shader looped until the
driver killed the device. [#3214](https://github.com/mattias800/prosper/issues/3214)

The geometry is blown out white in places and the lighting is not yet right — that is the next thing.


### Astro Bot's world-map GPU hang is one loop in the pixel shader, and we can now point at it

No picture: the world map still renders only the nebula backdrop. What is new is that the hang has
a mechanism instead of a suspect. The draw that kills the GPU was named by its *vertex* program, but
that program has no loop at all — it is 8 straight-line blocks. Its *pixel* partner is a 136,875-word
module with exactly one loop, the state machine prosper builds for shaders whose control flow it
cannot structurize, and that loop never ends: capping it at 4,096 iterations takes a deterministic
53-device-loss run to zero, while capping it at a million leaves the crash exactly where it was.

A new per-loop cap then narrowed it further, to one of the four loops inside that shader — a
per-screen-tile walk down a linked list of lights that is supposed to stop at `0xffffffff` and never
does ([#3193][i3193]).

[i3193]: https://github.com/mattias800/prosper/issues/3193

### An F9 capture now owns the whole mip chain, so the fetch that stops Sonic Frontiers can be studied offline

No picture — this one is a tool. Compute images carry the guest's declared mip chain since
[#3048][i3048], but `gpu_replay` did not: a capture owned the *level* the descriptor named, and a
tiled chain stores level zero last, so every other level sat below the captured range and replay
fell back to a single-level image with `IMAGE_LOAD_MIP` still refused. Captures now own the whole
allocation, and the offset that was already in the file turns out to say exactly how much of it
lies below the descriptor's address — so no new format version was needed
([#3202][i3202]). A bundle grabbed before this still declines, visibly, rather than fetching
levels it does not have; re-grab the frame.

[i3048]: https://github.com/mattias800/prosper/issues/3048
[i3202]: https://github.com/mattias800/prosper/issues/3202

## 2026-09-01

### The shader that hangs Astro Bot's world map can now be pulled out of a run by name

No picture — this one is a tool. Astro Bot's GPU reset was narrowed to one vertex program,
`0x5008efd00` ([#3193][i3193]), and then the obvious next step turned out to be impossible: the
successful-shader dump named its files by content hash, so "give me the shader at that address" had
no answer. It does now — the guest address is in the filename, and `PROSPER_SHADER_DUMP_PROGRAM`
narrows a 4K run to one program ([#3196][i3196]).

Pointed at Astro Bot, it produced that program's raw RDNA2 and four SPIR-V variants over a single
guest stream. All four validate, so whatever hangs the GPU is not malformed SPIR-V. Why it hangs is
still open; the bytes to answer it with are one command away instead of unavailable.

[i3193]: https://github.com/mattias800/prosper/issues/3193
[i3196]: https://github.com/mattias800/prosper/issues/3196
### Astro Bot's world map stops killing the GPU, and the draw that does it has a name

No picture: what the world map renders is still the nebula backdrop we already published. What
changed is that the run no longer dies there.

Loading the world map took the whole GPU down — a driver-level hard recovery, after which every
frame was the same frozen picture for the rest of the run. The message blamed a compute program,
but that program was just the first thing to fail *after* the device was already gone. It was a
single **draw**, and there was no way to ask which one: prosper could decline a compute program by
name but not a graphics one. It can now, and bisecting thirty-two candidates took four runs — with
one draw declined, a five-minute run has zero device losses and keeps producing new frames the whole
way. ([#3193](https://github.com/mattias800/prosper/issues/3193))

### The three getenv calls the profiler pointed at were the three we could not fix

No picture — this one is pure CPU time. `getenv` costs 1.24% of Blue Prince's gameplay frame
because a per-draw guard like `if (getenv("PROSPER_GFXLOG"))` rescans the whole environment block
for every draw, to answer a question whose answer never changes.

The interesting part is what the sweep found. Every one of the four call sites [#3094][i3094]
nominated as the obvious fix turns out to be unsafe to cache: three are armed at runtime by tests
that toggle them between phases, and caching those does not make a test fail — it makes it go
*vacuous*, still printing `[ok]` against a stale value. A fourth class is worse, because no gate
can currently see it: `gpu_replay` re-applies a whole allowlist of renderer switches once per
bundle submit, through a variable rather than a literal. So the sweep became a screening problem
rather than a mechanical one. Forty-two sites across the renderer backend, the draw executor and
the live frontend now read once instead of per draw, measured at 30% fewer `getenv` calls across
the render test suite, and the sites are pinned so they cannot quietly grow back.

[i3094]: https://github.com/mattias800/prosper/issues/3094

### The compute path waits 19-38% of its wall clock on a fence, and pipelining it cannot help

No picture — this is a negative, and an expensive one to have found the slow way.

Three UE titles spend a fifth to a third of their wall clock inside `vkWaitForFences` on the compute
path, with `vkQueueSubmit` at about 1% everywhere. That reads as an obvious win: stop waiting on each
dispatch, overlap them, take the time back. A dispatch ring was built for it and works correctly.

It buys nothing, for a reason no tuning changes. On these routes the guest submits roughly **one
dispatch per batch** — 23,294 dispatches across ~23,400 batches on *Stray* — and the drain at the end
of each batch is mandatory, because the guest may read its own memory as soon as the submit returns.
Every slot is emptied immediately after it is filled, so the ring never overlaps anything. Throughput
moved 67.5 → 68.5 fps, which is noise; the only real effect was less variance.

The falsification is written down rather than the code being merged, because a default-off switch
whose own measurement says it can never fire is worse than no switch. What is *not* ruled out is
deferring across batches — that is where the sized win still lives, and nobody has tried it.

### The Stray vertex shaders that "the recompiler can't compile" compile fine

No picture — this one is a correction, not a rung.

Two of Stray's title-screen vertex shaders are dropped at a vertex attribute fetch, which reads like
a missing shader lowering. It is not: the instruction is implemented, the stage has its resource
table, and the fetch's descriptor is the only thing missing. The shader builds that descriptor by
dereferencing a user-data pointer that is outside the eight-register window the pipeline actually
loads, so the pointer it reads is the tail of the previous descriptor — the same `0x0004dfac…` value
[#305](https://github.com/mattias800/prosper/issues/305) has been chasing on two other UE4 titles.
Stray is the third, and the first where one replay shows both halves. The reject line now says which
of those two stories it is, so the next reader does not have to re-run the game to find out.

### Stray's missing full-screen draws are not a missing opcode — they are a missing wave

No picture: the shader still does not compile, and a black frame is not progress evidence. But the
thing it is waiting for turned out to be much larger than the reject line said.

Two of Stray's dropped title-screen draws are full-screen, and both die in one vertex program that
opens with a cross-lane `v_mbcnt` prosper refuses in a vertex stage. Satisfying just that instruction
with a throwaway probe moves the failure exactly ten instructions, onto an LDS write: the program is
not a vertex shader at all but a merged NGG geometry threadgroup, doing vertex compaction with shared
memory, three barriers, cross-wave prefix scans and lane reads. It needs eight wave/workgroup
primitives, and `v_mbcnt` is only the first one in program order.

A Vulkan vertex stage has no shared memory and no promise that a subgroup is the guest's wave, so
there is nothing correct to lower this onto — the honest destination is a mesh shader. The reject now
says which instruction actually disqualified the program instead of pointing at a lowerable one.
([#3135](https://github.com/mattias800/prosper/issues/3135))

## 2026-08-31

### Stray's splash runs 66% faster, and the title screen now holds 65 fps

No picture: the title screen still renders black, so there is nothing new to look at yet. But the
sequence in front of it stopped crawling.

One 4K surface was being de-swizzled from scratch thousands of times — 241 GiB of tiling work in
under two minutes — because DCC-compressed textures were barred from the compute image cache. The
bar was checking the wrong thing: nothing on that path ever reads the compression metadata it
inspected, so the cache could have held those textures all along. Five detiles now do what 3,860 did.

Splash goes 37 → 62 fps; carried through to the title screen, a run averages 65. ([#3150](https://github.com/mattias800/prosper/pull/3150))

## 2026-08-30

### Stray's black title screen is not a loading failure — the data arrives, then leaves

No picture in this one: the title screen still renders black, so there is nothing new to look at. But
we now know what is *not* wrong, and that took four instruments.

The 4K background reads its texture from the game's pak. That read works. It delivers the bytes
byte-for-byte — verified by re-reading each destination immediately after the write and `memcmp`ing
it against the source, 21,499 of 21,499 writes on the title route — and a plain `dd` of the pak from
outside the emulator agrees with what landed in memory, exactly. Sixty-five seconds later, when the
shader samples that same address, it is entirely zero. Another surface is already zero **two seconds**
after its write, and stays zero across five samples and 152 seconds.

So something reclaims these ranges between load and use, and the pak reader — where this
investigation spent its first day — was never the place to look.

Two things nearly sent us the wrong way, and both are worth repeating. We argued the read finished
first because it appeared 1,428 log lines earlier; that is not evidence at all, since those lines come
from different threads sharing one stderr. And the write-verifier initially compared *counts* of
non-zero values rather than the values themselves, so it would have called a completely different
buffer a match — and it passed its own mutation test while doing so. Both are now recorded as
instrument traps, because the failure that survives a green test is the expensive kind.

[#3142](https://github.com/mattias800/prosper/issues/3142)

## 2026-08-29

### Tactics Ogre: Reborn comes back from 25 days of black

<p align="center"><img src="assets/screenshots/tactics-ogre-title-restored.webp" alt="Tactics Ogre: Reborn — the illustrated prologue map of the Valerian Isles with a subtitle line"></p>

One save-data call did it. `sceSaveDataDirNameSearchPs4` was registered in #2302 and answered
`NOT_FOUND` instead of the old success — a well-argued change whose own commit message noted it was
"not observed being CALLED at any boot depth reached so far". This title calls it, once, and on the
error it drew a single frame and then waited forever.

Nine automated bisect steps over 700 commits found it. The fix answers the question the way the PS5
sibling already does: zero hits, written explicitly, and success.
[#3124](https://github.com/mattias800/prosper/issues/3124)

### Grand Theft Auto V renders its world again

A regression took GTA V's lighting for a day — the bank heist still drew, but unlit and under a grid
artifact. Restored.

<p align="center"><img src="assets/screenshots/gta5-prologue-bank-restored.webp" alt="Grand Theft Auto V — the prologue bank interior in full colour, the masked gunman in a red plaid shirt, water cooler, holiday cards and radar"></p>

The cause was one line, and the reason it got through is worth more than the fix. #3093 made every
HTILE write discard retained depth, to cure Blue Prince's black frame. It checked GTA and cleared
it on **peak colour coverage — 99.78% in both arms** — but a world drawn with no lighting still
covers 99.78% of the frame. The metric could not see the defect it was chosen to rule out.

Restoring the exception fixes GTA and leaves Blue Prince exactly where it was: 0.2085 non-black in
both arms, the same figure #3093 called healthy. There was never a trade-off between the two titles.

Two hypotheses died on the way, both by measurement rather than argument. A "uniform HTILE plane
means a fast clear" discriminator was checked before being written and is false — GTA's writes are
6,500/6,500 uniform and Blue Prince's 62,000/62,000, all-zero, zero transitions, indistinguishable.
So we still do not know why two identical-looking writes need opposite handling.
[#3121](https://github.com/mattias800/prosper/issues/3121)

## 2026-08-29

### Unbound: Worlds Apart was never rendering-broken — it was reading a stale save

Its whole intro cinematic plays in full colour once the run gets a save directory of its own; the
black screens we had been reading as a broken composite came from a leftover save on the shared box
that made the title resume into a state it draws nothing for.

![Unbound: Worlds Apart at 3840x2160 — a sunlit village clearing in the intro cinematic: thatched huts strung with orange bunting, tall trees and drifting fireflies, pink mushrooms in the foreground grass, the small red-cloaked character at the right, and a prompt reading Press Square to skip](assets/screenshots/unbound-worlds-apart-intro-cinematic-village.webp)

![Unbound: Worlds Apart at 3840x2160 — the title screen held for the full 200 s of a default launch with no input: the UNBOUND / Worlds Apart wordmark over a dark forest lit by fire, a cloaked figure at the left, the Cross prompt below and the Unreal Engine logo in the corner](assets/screenshots/unbound-worlds-apart-title-screen.webp)

The title screen also renders continuously now — 40 of 40 samples across 200 s, where the survey a
week ago caught it on about 9%.

## 2026-08-28

### Dragon Quest VII reaches the field

The field HUD is live — minimap, party status, the Pilchard Bay banner — and the player is standing
in the harbour rather than watching it.

![Dragon Quest VII Reimagined at 3840x2160: the player character stands outside a harbour house with an orange quest marker over its door and a rowing boat beached to the right, foliage and a cliff on the left. The circular minimap sits at bottom-left and the party block at bottom-right reads Lv.1, HP 22, MP 7. Colour is badly degraded — the buildings are blown to white and the ground crushed to navy — but the scene is structurally complete](assets/screenshots/dragon-quest-vii-pilchard-bay-gameplay.webp)

![Dragon Quest VII Reimagined: the Pilchard Bay location banner appearing as the player enters the area, with the field HUD live. The world behind it is largely lost to the composite collapse](assets/screenshots/dragon-quest-vii-field-hud.webp)

![Dragon Quest VII Reimagined: the same harbour after a left-stick window — the quest-marker house that stood centre-left is now upper-right, a cliff face has entered from the left, and the minimap has scrolled to match. The player has walked](assets/screenshots/dragon-quest-vii-walked-to-cliff.webp)

The colour is plainly wrong, and depending on the run a quarter to a half of the frames still lose
the world to the composite — geometry and the HUD are fine; it is the lit-material shading that is
broken.
Nothing was blocking control, though: the opening chapter is simply very long, and every route we
had gave it about forty confirms before deciding it was a wall.

### Blue Prince is back

Master had been rendering a pure black frame; the title screen and its desk of curiosities are
whole again, and the fix costs GTA V nothing.

![Blue Prince title screen: the BLUE PRINCE logo over NEW GAME and SETTINGS on the left, and a dark study on the right with a globe, a red paper crown on a stack of books, an hourglass, a violin, a pocket watch and blueprints spread across a desk](assets/screenshots/blue-prince-title-restored.webp)

One line in the GTA V rendering foundation stopped prosper from discarding a depth buffer when the
guest rewrote its HTILE metadata with identical bytes — sound-looking, because identical bytes ought
to mean nothing changed. But prosper never writes rendered HiZ back into that guest plane, so the
plane is a constant the game keeps rewriting, and "unchanged" was equally true of Blue Prince's
per-frame depth *clear*. Its depth was never cleared, stale depth rejected every piece of geometry,
and the screen went black. ([#3089](https://github.com/mattias800/prosper/issues/3089))

## 2026-08-27

### Tomb Raider's world really is textured now

Croft Manor's assault course — brickwork, sandstone, mossy wooden platforms, gravel, ivy, Lara, and
Winston bringing the tea.

![Croft Manor assault course: Lara on wooden platforms with moss, red brick and sandstone walls, gravel ground, ivy and trees under a bright sky, Winston carrying a tea tray at the left](assets/screenshots/tomb-raider-croft-manor-assault-course.webp)

The decode cache was validating 262144 of 90177536 bytes — 0.29% — of the 256-layer world atlas, so
a decode taken while the atlas was nearly empty was reused all run and the walls wore whatever had
been in that memory earlier.
[#2998](https://github.com/mattias800/prosper/issues/2998)


### Windows audio: the underruns are not a pacing bug

Every title crackles on Windows and only on Windows, and the obvious cause — the sink hands the
sound card too little cushion — is wrong. Blasphemous 2's guest produces **84 audio grains a
second against the 187 that continuous playback needs**, so the device queue holds zero bytes at
53% of the moments it is sampled, however carefully the sink paces the half it does get.

Holding a deeper cushion, which is the fix everyone reaches for first, measures *worse*: 69%
empty, matching the pre-fix pacer it was meant to improve on. Removing the pacing makes the
guest deliver less often, not more.

No picture — this one is a number. #3072 has the hunt.


### ~~Tomb Raider's world is textured~~ — RETRACTED, that was a loading screen

**This entry was wrong and its picture is withdrawn.** The image published as Croft Manor's textured
brickwork was the game blitting its own pre-rendered loading picture, `2/PIX/HD/MANSION.DDS` —
pixel-identical to the checked-in capture (mean abs diff 0.02/255, 100% of pixels within 8/255).
Displaying a full-screen 2D image requires no world rendering at all, so it never showed what it was
captioned as showing. The project owner spotted it; no automated check did, and none could have.

It is left here rather than deleted because the blog's own rule is that it records what was claimed
and when. Recorded as instrument trap 230 — it looked *better* than the emulator could plausibly
render, and that is the tell.

![The genuine render of the same level: geometry correct but every surface a flat cream colour](assets/screenshots/tomb-raider-croft-manor-untextured.webp)

![Tomb Raider II title screen: Lara's model, the logo, game-select thumbnails and the Lara's Home menu entry](assets/screenshots/tomb-raider-title-screen-tr2.webp)

[#325](https://github.com/mattias800/prosper/issues/325) · [#2998](https://github.com/mattias800/prosper/issues/2998)


### What "the wrong surfaces" actually looks like

Lara's Home, and every wall and floor is wearing something real from elsewhere in the game — her
passport, the Game Boy collectibles, an inventory document page.

![Croft Manor interior: walls tiled with Game Boy console artwork and a UK passport page, the floor covered in a document reading THIS PAGE IS RESERVED FOR OFFICIAL OBSERVATIONS, Lara silhouetted in the centre](assets/screenshots/tomb-raider-croft-manor-interior-wrong-textures.webp)

The textures decode correctly and the geometry is right; the wrong content is being selected.
[#2998](https://github.com/mattias800/prosper/issues/2998)


### The Messenger's first level runs at about 156 fps, not 24

Windowed and uncapped on current master — the charter's long-standing "roughly 24 FPS" predates a
change to how frames reach the screen and has not described this title since July.

![The Messenger's opening: an 8-bit sunset over the ocean, the great tree, the Messenger on a plank platform, and a dialogue box reading "Demon army this and magic scroll that, nothing's happened in centuries, so why are we still hiding?"](assets/screenshots/messenger-first-level-windowed-2026-08-27.webp)

That is a presented rate rather than a count of new frames, so treat it as an upper bound.
[#3083](https://github.com/mattias800/prosper/issues/3083)


### A frame checker that would have called Stray's working menu "nothing rendered"

No picture in this one, because the finding *is* the picture we nearly got wrong. A classifier we
were about to rely on downsamples frames before counting colours — and at 160x90 a 4K frame loses
white menu text on black entirely. It reported Stray's main menu, with START GAME / SETTINGS /
CREDITS and a legible build stamp on it, as flat black with 61 colours. One edit away from writing
that title up as not rendering.

The replacement checks frames at full size, and the thing it looks for is a HUD or notice drawn over
a world that never appeared — which is what separates "this title renders nothing" from "this title
is at rung 2". Pointed at every screenshot in the repo it finds four, three of them the Sonic
Frontiers and Metaphor frames already on record as exactly that.

It also refuses to guess above that line. Coverage and colour count cannot tell a logo from a scene:
in our own screenshots a flat Gameloft splash covers more of the frame than The Messenger's title
art, and that title art uses 36 colours against the splash's 2,159.
[#3059](https://github.com/mattias800/prosper/pull/3059)
## 2026-08-26

### Grand Theft Auto V renders its world

The prologue bank heist on a default launch, with the game's own Performance graphics mode selected —
until now the HUD and radar drew over nothing.

![Bank lobby: a hostage face-down with her hands raised, wrapped presents stacked behind her, floor markings and overhead light reflections](assets/screenshots/gta5-prologue-bank-lobby.webp)

![Bank interior: the masked gunman in a red plaid shirt, a water cooler, holiday cards pinned to the wall, radar bottom-left](assets/screenshots/gta5-prologue-bank-interior.webp)

Gameplay runs at a few frames a second, and the texture path is most of the frame.
[#1873](https://github.com/mattias800/prosper/issues/1873)


### Tomb Raider's world stopped being a pile of shards

Croft Manor now renders with correct geometry — same route, same scene, same build, before and after.

![Croft Manor before the fix: the world shattered into stretched triangles](assets/screenshots/tomb-raider-world-before-index-fix.webp)

![Croft Manor after the fix: steps, walls, hedges and trees all correctly shaped, with Lara and Winston](assets/screenshots/tomb-raider-gameplay.webp)

Surfaces are still untextured — that is the next thing. [#2990](https://github.com/mattias800/prosper/issues/2990)

### The fps counter works now, and it can tell a frozen picture from a running one

`--fps` read `no frames published yet` for the whole life of every real game boot, on every platform.
It now shows the presented rate and the rate the picture actually *changes* at, and says
`picture not changing` when those disagree -- which is what a hung title looks like from outside,
since a frozen picture still presents at 60.

No picture: it is a number on a HUD, easier to see by running `--fps` than to photograph.

### Tomb Raider I-III Remastered boots for the first time, and reaches its title screen

We now reach the rendered Tomb Raider I title screen. This title had never been launched in prosper
before today, and it needed no code change at all — only a pad route to clear the game's own 40-page
EULA, which Cross refuses to accept until you have scrolled to the last page.

![Tomb Raider I-III Remastered — the Tomb Raider I title screen](assets/screenshots/tomb-raider-title-screen.webp)

## 2026-08-25

### GRIS has sound now

GRIS used to run in complete silence — no title music, no intro-movie audio, no gameplay music.
The game was asking the console's audio hardware to decompress its music, and prosper was quietly
dropping every single one of those requests. The requests are answered now: the intro movie plays
with its soundtrack, and the music keeps playing through gameplay for the whole session.

No picture for this one — it's audio. Start GRIS and listen.

## 2026-08-23

### Metaphor: ReFantazio can read its own font now, and the first thing it wanted to say was hello in twelve languages

<p align="center"><img src="assets/screenshots/metaphor-language-select.webp" alt="Metaphor: ReFantazio — the language-selection screen: twelve languages listed in white serif type over black, English highlighted with a blue brush-stroke, the list reading English, Deutsch, Español (España), Español (Latinoamérica), Français, Italiano, Português, Русский, 日本語, 中文(繁體), 中文(简体), 한국어"></p>

<p align="center"><img src="assets/screenshots/metaphor-loading-mascot.webp" alt="Metaphor: ReFantazio — the loading screen's winged fairy perched on an open book, drawn in blue and red over black in the lower right corner"></p>

Sony's font library has a call that draws one letter and then tells you how big the letter it drew
was; prosper had never implemented it, so it politely reported success and left the answer blank.
The game read the blank — whatever the last function to use that piece of stack had left behind,
which happened to be 285,196,807 — decided its letters were two hundred and eighty-five million
pixels wide, ordered a texture that size, got one that was never really built, and divided by its
zero pixel format. Five seconds into every boot, forty minutes of tracing away from the font code.

The fix is not a better number, because there is no honest number to invent: it is a real
rasterizer. The game hands us its own 180 KB font file when it starts, so prosper now reads that
file and draws the actual outlines out of it — which is why the Cyrillic, Japanese, Chinese and
Korean above are all correct. Nothing here is a guess; it is the game's own typeface
([#2951](https://github.com/mattias800/prosper/issues/2951)).

The same change made two facts measurable that had only been suspicions. *Sonic Frontiers* and
*Sonic Origins* both name this font library in their binaries, so it looked like a plausible cause
for Sonic Origins' missing wordmark — but neither title imports a single function from it, and
counting is what settled that rather than argument. And Astro Bot, which imports fifty-four of
them, was quietly missing one all along.

### Metaphor: ReFantazio was byte-reversing four gigabytes of its own heap

No picture — the frames it now produces are still black. We had been telling the game about memory
it could not actually reach, and it handled the resulting refusal by asking its endian converter to
byte-swap "however many bytes I just read", which was the error code. It now loads its assets,
opens its audio and publishes frames before dying of something else
([#2934](https://github.com/mattias800/prosper/issues/2934),
[#2951](https://github.com/mattias800/prosper/issues/2951)).

### Two Unreal titles were being quietly charged 2 GiB for memory they never got

No picture with this one — neither title renders yet. *Sifu* and *The First Berserker: Khazan* both
die a few seconds into boot with Unreal's own out-of-memory report, and the suspicion was that
prosper's direct-memory pool really had run dry. Half of that turned out to be true in a way nobody
had spotted: every time the guest asked us to place a buffer somewhere we had to refuse, we took the
physical memory for it anyway and then forgot we had it. Khazan does that 4,646 times per boot and
Sifu 31,716 times, so Sifu was losing nearly two gigabytes of the pool to allocations that never
existed. That is now fixed.

The other half is the more useful finding, and it is a negative one: with the leak gone, both titles
still assert — and at that moment prosper's pool still has a 230 MiB block free and has failed no
allocation the guest asked for. So the pool was never what stopped them, and whatever is really
going on is inside Unreal's own allocator. Those thousands of refused mappings are not a loss
either: prosper refuses them precisely because the guest already has memory there, and refusing is
what stops us overwriting it. Details in
[#2908](https://github.com/mattias800/prosper/issues/2908).

### The same captured frame, replayed twice, two different pictures

<p align="center"><img src="assets/screenshots/balan-replay-same-file-menu.webp" alt="BALAN WONDERWORLD - the language-select menu over the red-and-gold theatre backdrop, with the option pills and their text rendered"></p>

<p align="center"><img src="assets/screenshots/balan-replay-same-file-slivers.webp" alt="BALAN WONDERWORLD - the same menu from the same captured frame, but the theatre is gone and most of the option pills and glyphs have collapsed into thin diagonal slivers on white"></p>

Both of those came out of **one captured file**, replayed offline by `gpu_replay` on the same
binary minutes apart, with no game running. That is the defect: some of BALAN's menu draws
intermittently produce no geometry at all, so the panels and letters that should have covered the
screen shrink to slivers or disappear, taking the theatre backdrop with them - and it is the same
fault that leaves several other titles showing a flat white frame. It is now reproducible from a
single draw in about three seconds instead of a two-minute boot, and there is a long list of things
it is not: [#2945](https://github.com/mattias800/prosper/issues/2945).

## 2026-08-22

### New Joe & Mac: Caveman Ninja plays start to finish

<p align="center"><img src="assets/screenshots/joe-mac.webp" alt="New Joe & Mac: Caveman Ninja — gameplay: Joe crouched in a jungle level with palms, pink blossom, a volcano behind and coiled snakes either side, with the name plate, score, health bar and a lives counter reading x3"></p>

<p align="center"><img src="assets/screenshots/joe-mac-menu.webp" alt="New Joe & Mac: Caveman Ninja — the game's menu at 1920x1080"></p>

We play this one through, and have done for a while — it is rung 6 with a reviewed `joe-mac-gameplay`
snapshot guard, and it had somehow never appeared here. Tracker
[#1876](https://github.com/mattias800/prosper/issues/1876).

## 2026-08-22

### BALAN's language menu was never waiting for Cross

<p align="center"><img src="assets/screenshots/balan-wonderworld-prologue.webp" alt="BALAN WONDERWORLD — the opening story cutscene at 3840x2160: Leo and Emma standing in a city park at golden hour, a basketball court with graffiti-covered fencing behind them, children playing, trees and a brick building in the background, and speaker cabinets flanking the frame"></p>

Three titles sat on a screen whose own prompt named a button. Only one really was waiting for it.
BALAN's language menu answers Cross with a modal — *change the language to English?* — that 109
Cross presses never got past; **Down** does, and behind it are the title screen, the main menu, and
this, the opening cutscene — 3070 decoded 4K pictures. Unbound wanted **Square**, the button its
cinematic actually asks for. Trackers [#2882](https://github.com/mattias800/prosper/issues/2882), [#2883](https://github.com/mattias800/prosper/issues/2883),
[#2886](https://github.com/mattias800/prosper/issues/2886).

### The eight titles nobody had ever run

<p align="center"><img src="assets/screenshots/unbound-worlds-apart-title-screen.webp" alt="Unbound: Worlds Apart — the title screen at 3840x2160: the UNBOUND / Worlds Apart wordmark in a pale carved typeface over a dark blue forest, a cloaked figure standing left of a glowing blue portal, with a Cross-button prompt below"></p>

<p align="center"><img src="assets/screenshots/balan-wonderworld-language-select.webp" alt="BALAN WONDERWORLD — the language-select menu at 3840x2160, over the game's theatre artwork, with button glyphs along the bottom"></p>

<p align="center"><img src="assets/screenshots/stray-brightness-calibration.webp" alt="Stray — the 4K brightness-calibration screen, a dim reference image with a slider and the game's own Cross Accept prompt"></p>

<p align="center"><img src="assets/screenshots/little-nightmares-2-tarsier-logo.webp" alt="Little Nightmares II — the Tarsier Studios logo at 3840x2160, part of the boot logo sequence"></p>

Eight tracked titles had never been booted. All eight now have: three reach a menu or title screen,
one a logo sequence, four render nothing. *Unbound* is the best of them — that is its real title
screen, on 9% of frames. The point was the comparison, not the eight runs: what they share is
symptoms, and every rung-0 wall is its own.
[`NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md).

### Sonic Origins reaches its title screen, and the wall was a dialog box

<p align="center"><img src="assets/screenshots/sonic-origins-title-screen.webp" alt="Sonic Origins — the title screen at 3840x2160: the classic winged gold ring emblem with blue stars, Sonic peering over a red-and-white striped banner, in front of the painted South Island seascape with cliffs, clouds and sunlit water"></p>

<p align="center"><img src="assets/screenshots/sonic-origins-autosave-notice.webp" alt="Sonic Origins — the game's own boot notice at 3840x2160: a white panel over the cyan and green striped menu background, a glowing gold ring icon, the text 'This title supports auto save. When this icon is shown, do not turn off the power. Save data may be corrupted.' and a Close button marked with the Cross glyph"></p>

The most-investigated title in this repository, and the last wall was a modal waiting for CROSS.
Nine lanes went at its black startup frame. What finally moved it was tonight's save-data fix — the
guest loops on `cmp eax,0x809f0018` and sleeps, and prosper returned exactly that. Rung 1 to rung 2.
The banner is empty because the wordmark never draws
([#2920](https://github.com/mattias800/prosper/issues/2920)).
### Sonic Origins' SONIC TEAM logo is blue

<p align="center"><img src="assets/screenshots/sonic-origins-sonic-team-logo-blue.webp" alt="Sonic Origins — the SONIC TEAM logo at 3840x2160: the blue Sonic head silhouette and blue SONIC TEAM wordmark on a near-white background"></p>

The 2026-08-20 entry further down this page has the same frame in **purple**, and that was an
honest record of what prosper drew that day: every movie in the title composited with its two chroma
components collapsed onto one, so a gold ring rendered green and anything blue rendered magenta.
[#2731](https://github.com/mattias800/prosper/issues/2731) is fixed, and this is the same logo from
the same route on current master — a direct, unmodified `tools/screenshot` capture, default launch
with no input, sample 14 of a 420 s run at t=150 s. The old entry stays where it is.

Refs [#2904](https://github.com/mattias800/prosper/issues/2904).

### Beast of Reincarnation: the whole game was under a coat of white paint

<p align="center"><img src="assets/screenshots/beast-of-reincarnation-deluxe-bonus-dialog.webp" alt="Beast of Reincarnation — the game's own Digital Deluxe bonus dialog: an item list with Big Dipper, Black Shiba Skin, Special Hat, Amber and crop seedlings, a scrollbar, an orange note and an OK button, rendered at 3840x2160"></p>

<p align="center"><img src="assets/screenshots/beast-of-reincarnation-game-freak-logo.webp" alt="Beast of Reincarnation — the GAME FREAK developer logo in white on black, rendered at 3840x2160"></p>

Game Freak's first PS5 title, and the corpus's second confirmed UE5. A default launch shows a flat
white 4K frame for four minutes, while the guest runs happily and prosper renders hundreds of draw
batches a second. Two causes: a missing SDWA instruction form dropping 404 draws a run, and 8,192
fast-clear-eliminate draws painted as ordinary colour
([#1588](https://github.com/mattias800/prosper/issues/1588)). Under a default-off lever, this is
underneath.
### Gollum's boot was killed by a divide by zero, and the divisor was a channel count nobody wrote

`libSceAudiodec` was entirely unimplemented, so its nine entry points fell to the dispatcher's
`return 0` — which is `SCE_OK`. Unreal's Electra player believed it had an AAC decoder, ran a decode
that wrote nothing, and divided by the channel count nothing had written. Now implemented against the
real ABI, recovered from the title's own call sites. Still rung 0; next blocker
[#2898](https://github.com/mattias800/prosper/issues/2898).
### Hi-Fi RUSH reaches its title screen on the first try

<p align="center"><img src="assets/screenshots/hifi-rush-title.webp" alt="Hi-Fi RUSH title screen — the yellow branding, shattered logo and Press Any Button prompt, rendered at 3840x2160"></p>

<p align="center"><img src="assets/screenshots/hifi-rush-rooftop-black-materials.webp" alt="Hi-Fi RUSH Vandelay rooftop — correct geometry, depth and sky gradient, with every opaque surface shaded flat black"></p>

Added to the library in the evening, at its title screen a few hours later — 281 ms to
`BOOT_COMPLETE`, default launch, no throttle, no pad. The second picture is the Vandelay rooftop,
where geometry, depth and the sky gradient are all correct and every opaque surface is a silhouette,
which narrows the defect nicely.
### Khazan spent four seconds booting and then waited forever for one wrong hexadecimal digit

All 79 threads parked in a syscall, `sceSaveDataGetEventResult` polled 5,020 times in 90 seconds.
The guest's wait loop accepts exactly one value; prosper answered the code meaning *"still in
flight"*, returned when nothing was in flight. Run as a control, *Earthion* turned out to have had
its whole save-data subsystem dead for the same reason.
### PGA TOUR 2K25 told us exactly what was wrong, in English, and it was not what it said

*"PSN is an old version that cannot be used by the current player runtime"* — and nothing was old,
and no module was missing. Unity's PSN native half links into the user-assemblies module, prosper
started it with a null parameter block, and the mismatch branch's own error message read the version
off that null pointer. The error handler was the crash.
### Yakuza Kiwami allocates its entire game heap through a Sony API nobody had implemented

No picture with this one — the title still does not render. The finding is what moved.

*Yakuza Kiwami* (`PPSA31334`) died **0.0 seconds** into every boot, writing to address `0x1d0000`.
That address is the tell: it is far too low to be a real guest pointer, and it is what you get when
an allocator is handed a base address of roughly nothing and starts walking.

The base came from `sceAmprAmmGetVirtualAddressRanges`. AMM is the memory-mapping half of
libSceAmpr — the same command-buffer construct prosper already used for asynchronous **file reads**,
pointed at pages instead of bytes — and this title runs its *whole* game heap through it: it asks
AMM for up to 512 GiB of address space, hands it 10 GiB of physical memory, and then maps 2 MiB
chunks in on demand for the rest of the run. All seven of the AMM entry points it needs fell to
prosper's unimplemented default, which returns 0 and writes nothing.

Returning 0 sounds harmless. It is not, when the guest is reading your *out-parameters*: the
initialiser reaches that call on a path that never zeroes its own struct, so "wrote nothing" meant
the allocator took its virtual-address window from whatever was left on the stack. Everything after
that was the allocator faithfully doing what it was told.

With AMM implemented the boot now reserves a real 68 GiB window, takes its 10 GiB pool, and services
23 map commands before it gets somewhere new — far enough to initialise save data and start loading
assets, where it stops with the game's own message:

```
Failed!! Load Devil2 Shader Archive
Failed!! Load Ptc Shader Archive
```

That is the next wall, and it is a different one: `sceAmprAprCommandBufferReadFileGatherScatter`
([#2872](https://github.com/mattias800/prosper/issues/2872)), the scatter/gather sibling of a file
read prosper already implements. The archives never arrive, so the object is null, so the next
method call dereferences it. Still rung 0 — but the fault moved from the memory allocator to the
asset loader, which is the direction that counts. Tracker
[#2864](https://github.com/mattias800/prosper/issues/2864).

One footnote worth having: this turned out not to be a one-title fix. *Judgment* (`PPSA02739`),
onboarded the same day, imports **all seven** of the same AMM entry points — and both of the follow-up
gaps too, the scatter/gather read and AMM's `Unmap`. Checked by NID against its own import table,
not inferred from the shared publisher.

### Our first CryEngine title deadlocks 81 ms in, on a library it never asked for

No picture this time — the interesting thing about *Sniper Ghost Warrior Contracts 2* (`PPSA03130`)
is that there was nothing to photograph, and *why* there was nothing.

The first boot attempt produced no screenshots, no log past the renderer line, and an empty
manifest. That is a shape worth recognising, because it reads like a defect and is not one: the run
had been killed from outside. `tools/screenshot --timeout` cannot fire during boot — the deadline is
checked inside the sampling loop, and that loop is only reached after `boot_program()` returns.
`boot_program()` ends by running the guest's own module initialisers, so a title can sit inside it
forever with the tool's own limit inoperative.

prosper has recorded seven boot phases for a long time. It turned out **no build the project ships
could print any of them** — the feature was compile-time optional, the default build excluded the
whole folder from `prosper_core`, `enable()` was never called anywhere in the tree, and nothing
subscribed to the event bus. Four independent reasons, any one sufficient. `PROSPER_BOOTPHASE=1` now
prints them, and the answer arrived in one run:

```text
[bootphase] +80.6ms MODULES_MAPPED
[bootphase] +81.0ms STUBS_INSTALLED
[bootphase] +81.1ms GUEST_INITS_RUNNING     <- and BOOT_COMPLETE never comes
```

Not slow, not starved — **stuck**. Over 221 s the process used 0.00 s of CPU and read 0 bytes, with
every thread parked in a futex. `guest_bt` named the frame: the `module_start` of
`sce_module/libSceNpCppWebApi.prx`, a library this title **does not import** (its own NP library is
the unrelated `libSceNpWebApi2`). prosper preloads it because the file exists, under a rule added for
*Sonic Origins*, which really does import it. Remove that one file from the tree and the same binary
reaches `BOOT_COMPLETE` in 70 ms and runs.

So a module preloaded for one title had been silently wedging another, and the fix is to preload it
only when something actually imports it.

Which raised the obvious question a reviewer asked and I had not: *how many titles does that change?*
I had checked two. The answer is a census — across the tracked titles, 42 ship that PRX, 40 keep it, and two
lose it: this title, and **Sonic Frontiers**, which nobody had looked at and which has no snapshot
guard to notice. It appears to be harmless (import resolution is by NID, and not one of the 41,638
NIDs that module exports is imported by anything in Frontiers' link graph) but "appears to be" is the
honest phrasing, and a confirming boot of Frontiers belongs to the lane that owns it. A flag on a
shared list is never a two-title question.

Behind that wall the title is still at rung 0, and honestly so. Unmodified, with the fix in, it
boots in 91 ms, streams its assets, and drives a 4K present loop at ~21 flips a second — while prosper
composites exactly nothing. Every sample is a raw guest scanout: one distinct colour, zero non-black
pixels, `published_frames=0`. Two runs on two different trees agree, so it is not an artifact. The
next wall is that no pass produces a present source at all — an ordinary graphics problem, and a much
better place to be than a deadlock.

One footnote worth keeping, because it nearly became a finding. Mid-run the thing looked *parked*: 1 %
CPU, no disk reads, eighteen threads asleep, and exactly one of them — Wwise's `AK::BankManager` —
blocked on a mutex while everything else waited on conditions. That asymmetry reads like a deadlock
with a culprit's name attached. It wasn't; the run resumed thirty seconds later. The box was 70-90 %
I/O-stalled by an unrelated archive extraction the whole time, and a warm page cache meant the "no
disk reads" number was measuring the wrong thing entirely — the read *syscalls* were climbing fine.
A mutex wait is not proof of a deadlock. The holder may just be slow.

### Sonic Frontiers' black world had two locks on the door, not one

No picture in this one — the world is still black. What changed is that we now know how far away it
is, and the number is smaller than it looked.

The Cyber Space stage reaches gameplay with a running clock, and the world behind the HUD is black
because sixteen of the stage's compute programs never execute. Three of them are the ones that
matter: screen-width passes over the frame the player is supposed to see. Their reject line has
named `image_load_mip` for a while, and the issue tracking it said so: *the* single remaining
blocker.

A reject line cannot say that. The recompiler stops at the **first** instruction it cannot lower and
reports that one; it has no way to tell you what is behind it. So we built a throwaway measurement
build — accept `image_load_mip` at LOD 0, knowingly wrong, never merged, run it where nothing is
submitted — purely to ask "and then what?". And then the same three programs stopped again, all of
them, on `s_getpc_b64`.

That second lock turned out to be a small one. The compiler had put a little constant table straight
into the shader blob and addressed it PC-relatively, which prosper already folds — but only when the
table is read back with an *untyped* load. Frontiers reads it with a typed one,
`tbuffer_load_format_x` at `32_FLOAT`, which is a 32-bit format that converts nothing, so the bytes
it fetches are the bytes that are there. The fold was already correct for it; a guard spelled
"untyped only" was the whole obstacle. That is what this PR fixes.

Two things nearly went wrong on the way, and both are the same shape — a check that looked like a
check. A typed fetch also honours the descriptor's channel routing, which the untyped one ignores,
and this game's table descriptor routes three of its four channels to a constant zero; the first
version of the fix only looked at the format, which would have been right here and silently wrong one
instruction over. And the test written to pin the *second* correction passed under a mutation that
deliberately broke the thing it was pinning — it was asserting "did the program compile", and that
program does not compile for an unrelated reason, so it could never have failed. It asks the detector
directly now.

With both cleared, all three programs vanish from the skipped list — they recompile and they run.
(Only one of the two is fixed here. The other is still open, so this measurement was taken with a
throwaway build that waves the mip instruction through at level zero — deliberately wrong output,
never merged, run where nothing is submitted. It answers "is anything else in the way", and the
answer is no.)

```text
before   [compute-census] 65536 dispatch decisions over 30 program(s)   13 programs listed, all executed=0
after    [compute-census] 65536 dispatch decisions over 30 program(s)   10 programs listed
         gone: 0x2005714000 (3840x135)  0x2005717e00 (3840x405)  0x200571bd00 (3840x270)
```

So the remaining work on those three really is one instruction now, which is what everyone thought
yesterday and was not true yesterday. And we know what it has to read: the guest builds itself a
2048x2048 two-channel float pyramid and binds **thirteen** descriptors to it — one per mip level to
write each one, plus one whole-chain view to read the finished thing back. The levels are sitting in
guest memory the whole time, at offsets prosper already computes correctly for each of those twelve
single-level descriptors. It just has never been able to look at them together.


### The Messenger's title screen runs at 206 fps and 0 fps at the same time

<p align="center"><img src="assets/screenshots/messenger-title-fps-overlay.webp" alt="The Messenger's title screen with a burned-in overlay reading 2.9 FPS (206.3 PRESENTED) 1920X1080"></p>

That counter is burned in by the new `screenshot --fps-overlay`, and both numbers are true. prosper
handed this picture to the display **206 times a second**. Across one 120-second stretch of that run
it published **25,015 frames and exactly one of them differed** from the frame before it. It is a
still image being re-served at full speed.

Nothing is wrong here: it is a title screen, and a title screen is allowed to sit still. The point is
that **a framerate counted from presents cannot tell this apart from a hung game.** That is not
hypothetical — it is how R-Type Delta's regression (#2783) hid for nine days while the guest reached
stage 1 and every presented frame was the same retained one. Any counter we shipped that reported
"206 fps" here would have reported it there too.

So prosper now counts **distinct guest frames** — publications whose pixels actually changed — and
every place a framerate appears shows both, honest number first. `PROGRESS_TRACKER.md` has an FPS
column for it, sourced from a new `FPS record:` line in the game trackers.

### And the first real number: Blue Prince at 4.7 fps

<p align="center"><img src="assets/screenshots/blue-prince-cinematic-fps-overlay.webp" alt="Blue Prince's opening cinematic — a blueprint of Mount Holly — with an overlay reading 4.7 FPS 94% ACTIVE 1920X1080"></p>

The Day One opening cinematic, at native 1080p with no snapshot acceleration, over fifteen minutes.
**4.7 frames per second, and 95% of the run was spent producing them** — that second number is what
says the measurement is honest rather than an average of a fast stretch and a frozen one. The rate
never left the 4.53–4.80 band across all 59 windows. (The overlay in the shot reads 94%, because it
is the running figure at that moment; 95% is where the full run finished.)

This is the "we have work to do" end of the scale, and it is now written down where you can find it
without re-running anything. For comparison the same instrument reads The Messenger's animated
stretches at 15–23 fps.

### Why The Plucky Squire's cutscene never ends

No picture for this one — it is a measurement, and it replaced a guess that had been sitting in the
notes as the frontier.

The record said the route "stops driving input at 525 s", implying the cutscene was waiting for a
button. It is not waiting at all. It advances roughly **300x too slowly to finish**, and two facts
multiply into that:

- the guest's tick rate **collapses 147x** when the 3D world streams in — about 25 polls per second
  in the menus, **0.19** once the level loads;
- in-game time advances **per flip, not per second**. That is deliberate and correct — each flip is
  budgeted one refresh interval — but it means the game clock moves ~16.7 ms per flip however long
  that flip actually took.

At 0.19 flips per second, a 60-second intro needs hours. The 1,200-second run that "never reached
gameplay" had bought a few seconds of it.

It is demonstrated rather than argued: raising only the flip rate walks the guest straight past the
intro to the storybook camera. So the wall is throughput, and the title is CPU-bound in texture
realization — the GPU sits at 5-16% throughout.

### Two more titles reach gameplay — and one reaches it in the dark

**Beneath** (`PPSA27640`) plays. This is the opening dive aboard the science ship: the waypoint
marker counts down as you move, and the characters talk over it. A cutscene would not have a live
distance readout, which is how we know it is the real thing.

<p align="center"><img src="assets/screenshots/beneath-gameplay.webp" alt="Beneath — the opening dive, waypoint HUD reading 21m, dialogue subtitles over a dark seabed"></p>

It is very dark, and that is the game rather than us — but it is worth an eye-check when this one
comes up for manual verification. Getting here needed no renderer work at all; the title was one
input route away. What it *did* need was `PROSPER_NULL_PAGE=1`, and the reason is a nice one: the
game's stack unwinder walks one hop past the end of the frame-pointer chain **on purpose**, because
it stops on a null return address rather than a null frame pointer. We enter the guest with `rbp`
zeroed — which is correct — so that last read lands on address `0x8` and faults. The flag gives the
guest back a low page that reads as zero, which is what the console gave it.

### R-Type Delta, blank for nine days, draws stage 1 again

<p align="center"><img src="assets/screenshots/rtype-delta-stage1-restored.webp" alt="R-Type Delta stage 1: the R-9 and its Force device over a sunset cityscape with enemy formations and the BEAM and score HUD"></p>

The R-9, its Force pod, enemy formations, and a city at sunset. This one is a good story. For nine
days the title rendered its logo and its whole opening movie and then went blank forever — the guest
was fine, reaching stage 1 and writing its save, while prosper published the same retained frame on
every flip.

One shader did that. `sprite_i_vv.ags` is the title's **sprite vertex shader**, so it draws the
menus, the HUD and the gameplay — everything except the logo and the movie, which is exactly the
symptom. The recompiler had been refusing it since a change in August that saved a wave mask in a
register pair and never ended that lifetime; when the shader later reused the same pair for an
ordinary table address, the stale mask made it look like ballot bits and the read was refused.

Reaching it also needed a route, and the title screen taught us something: its prompt is the PS5
**OPTIONS** glyph, not Cross. A Cross-only route sits there forever.

### Sonic Frontiers reaches Cyber Space — and the world is black

<p align="center"><img src="assets/screenshots/sonic-frontiers-cyberspace-hud.webp" alt="Sonic Frontiers Cyber Space — stage clock at 00:55.89, ring counter, star medals and speedometer over an entirely black screen"></p>

Not a pretty picture, and it is here because it is honest. That is a real running stage — the clock
reads 55 seconds, the speedometer needle moves, the music plays — with a hundred streamed terrain
sectors behind a world that never draws. Sixteen of the stage's thirty-two compute programs never
execute.

Three separate recompiler fixes have now unblocked programs on this title and changed **zero
pixels**, so we have stopped assuming the next one will be different. It also prompted a rule
change: reaching gameplay is no longer enough for rung 3, which now asks that the scene actually
render. Frontiers and *Grand Theft Auto V* both sit at rung 2 because of it. Neither regressed —
we just stopped counting a black screen as a win.

## 2026-08-20

### dragon-quest-vii-opening-chapter.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-opening-chapter.webp" alt="dragon quest vii opening chapter"></p>

feat(dq7): the route reaches the opening chapter in Estard, and Unreal titles get an IoStore package oracle (#2779)

`b166e6de` · [`assets/screenshots/dragon-quest-vii-opening-chapter.webp`](assets/screenshots/dragon-quest-vii-opening-chapter.webp)

### bendy-dark-revival-gameplay.png

<p align="center"><img src="assets/screenshots/bendy-dark-revival-gameplay.webp" alt="bendy dark revival gameplay"></p>

feat(bendy-dark-revival): rung 2 -> rung 3 — a route reaches the PPSA27624 Chapter 1 sections (#2769)

`642fe84b` · [`assets/screenshots/bendy-dark-revival-gameplay.webp`](assets/screenshots/bendy-dark-revival-gameplay.webp)

### tales-graces-f-gameplay-dialogue.png

<p align="center"><img src="assets/screenshots/tales-graces-f-gameplay-dialogue.webp" alt="tales graces f gameplay dialogue"></p>

feat(route): Tales of Graces f Remastered reaches gameplay -- the wall was two OPTIONS-button gates, not the renderer (#2763)

`dd010553` · [`assets/screenshots/tales-graces-f-gameplay-dialogue.webp`](assets/screenshots/tales-graces-f-gameplay-dialogue.webp)

### tales-graces-f-gameplay.png

<p align="center"><img src="assets/screenshots/tales-graces-f-gameplay.webp" alt="tales graces f gameplay"></p>

feat(route): Tales of Graces f Remastered reaches gameplay -- the wall was two OPTIONS-button gates, not the renderer (#2763)

`dd010553` · [`assets/screenshots/tales-graces-f-gameplay.webp`](assets/screenshots/tales-graces-f-gameplay.webp)

### sonic-origins-sonic-team-logo.png

<p align="center"><img src="assets/screenshots/sonic-origins-sonic-team-logo.webp" alt="sonic origins sonic team logo"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`8c233eb6` · [`assets/screenshots/sonic-origins-sonic-team-logo.webp`](assets/screenshots/sonic-origins-sonic-team-logo.webp)

### tales-graces-f-title-no-input.png

<p align="center"><img src="assets/screenshots/tales-graces-f-title-no-input.webp" alt="tales graces f title no input"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`8c233eb6` · [`assets/screenshots/tales-graces-f-title-no-input.webp`](assets/screenshots/tales-graces-f-title-no-input.webp)

### issue-2731-tales-graces-f-movie-chroma.png

<p align="center"><img src="prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.webp" alt="issue 2731 tales graces f movie chroma"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`8c233eb6` · [`prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.webp`](prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.webp)

### issue-2734-little-nightmares-3-corrupt-save-modal.png

<p align="center"><img src="prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.webp" alt="issue 2734 little nightmares 3 corrupt save modal"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`8c233eb6` · [`prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.webp`](prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.webp)

### asterix-babylon-gameplay.png

<p align="center"><img src="assets/screenshots/asterix-babylon-gameplay.webp" alt="asterix babylon gameplay"></p>

bringup(asterix-babylon): a Triangle-aware input route reaches GAMEPLAY (rung 2 -> 3) (#2736)

`65ebd16c` · [`assets/screenshots/asterix-babylon-gameplay.webp`](assets/screenshots/asterix-babylon-gameplay.webp)

## 2026-08-19

### plucky-squire-chapter1-intro.png

<p align="center"><img src="assets/screenshots/plucky-squire-chapter1-intro.webp" alt="plucky squire chapter1 intro"></p>

docs(plucky): status doc — the chapter-one world renders, plus five falsified hypotheses (#2742)

`b21bf552` · [`assets/screenshots/plucky-squire-chapter1-intro.webp`](assets/screenshots/plucky-squire-chapter1-intro.webp)

## 2026-08-08

### sonic-crossworlds-title.png

<p align="center"><img src="assets/screenshots/sonic-crossworlds-title.webp" alt="sonic crossworlds title"></p>

docs(crossworlds): rung 2 — the title screen renders completely, reproduced across two runs (#2360)

`5166cdb4` · [`assets/screenshots/sonic-crossworlds-title.webp`](assets/screenshots/sonic-crossworlds-title.webp)

## 2026-08-07

### arcrunner-title-screen-default-route.png

<p align="center"><img src="assets/screenshots/arcrunner-title-screen-default-route.webp" alt="arcrunner title screen default route"></p>

ArcRunner: the per-fold account — the guest's builder is released mid-fold, and the contract that forbids it is version-gated off (#2219)

`c42b89c3` · [`assets/screenshots/arcrunner-title-screen-default-route.webp`](assets/screenshots/arcrunner-title-screen-default-route.webp)

## 2026-08-06

### sonic-frontiers-autosave-notice.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-autosave-notice.webp" alt="sonic frontiers autosave notice"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`0be16d5f` · [`assets/screenshots/sonic-frontiers-autosave-notice.webp`](assets/screenshots/sonic-frontiers-autosave-notice.webp)

### sonic-frontiers-main-menu.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-main-menu.webp" alt="sonic frontiers main menu"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`0be16d5f` · [`assets/screenshots/sonic-frontiers-main-menu.webp`](assets/screenshots/sonic-frontiers-main-menu.webp)

### sonic-frontiers-title-screen.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-title-screen.webp" alt="sonic frontiers title screen"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`0be16d5f` · [`assets/screenshots/sonic-frontiers-title-screen.webp`](assets/screenshots/sonic-frontiers-title-screen.webp)

### arcrunner-title-screen.png

<p align="center"><img src="assets/screenshots/arcrunner-title-screen.webp" alt="arcrunner title screen"></p>

docs(arcrunner): the title screen is behind the movie, and the throttle rescues by DELAY not by lock hold (#2205)

`15e22519` · [`assets/screenshots/arcrunner-title-screen.webp`](assets/screenshots/arcrunner-title-screen.webp)

### sonic-origins-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-origins-sega-logo.webp" alt="sonic origins sega logo"></p>

fix(savedata): sceSaveDataCreateTransactionResource must return a positive resource id (#2121)

`726aa8da` · [`assets/screenshots/sonic-origins-sega-logo.webp`](assets/screenshots/sonic-origins-sega-logo.webp)

### arcrunner-default-route-logos.png

<p align="center"><img src="assets/screenshots/arcrunner-default-route-logos.webp" alt="arcrunner default route logos"></p>

ArcRunner: the init-side generation census — the #1756 question answered, four falsifications, and real graphics off the throttle (#1226) (#2122)

`bc063d75` · [`assets/screenshots/arcrunner-default-route-logos.webp`](assets/screenshots/arcrunner-default-route-logos.webp)

### arcrunner-intro-city.png

<p align="center"><img src="assets/screenshots/arcrunner-intro-city.webp" alt="arcrunner intro city"></p>

ArcRunner renders its intro cinematic — the #1226 fault is a submit-timing race (follow-up to #2077) (#2086)

`3b47dedf` · [`assets/screenshots/arcrunner-intro-city.webp`](assets/screenshots/arcrunner-intro-city.webp)

### arcrunner-intro-space-station.png

<p align="center"><img src="assets/screenshots/arcrunner-intro-space-station.webp" alt="arcrunner intro space station"></p>

ArcRunner renders its intro cinematic — the #1226 fault is a submit-timing race (follow-up to #2077) (#2086)

`3b47dedf` · [`assets/screenshots/arcrunner-intro-space-station.webp`](assets/screenshots/arcrunner-intro-space-station.webp)

### rtype-delta-force-select.png

<p align="center"><img src="assets/screenshots/rtype-delta-force-select.webp" alt="rtype delta force select"></p>

fix(gpu): a saved-EXEC wave mask must survive the NGG fetch-index fold — R-Type Delta reaches its title screen (#2061)

`7cc74ef5` · [`assets/screenshots/rtype-delta-force-select.webp`](assets/screenshots/rtype-delta-force-select.webp)

### rtype-delta-title.png

<p align="center"><img src="assets/screenshots/rtype-delta-title.webp" alt="rtype delta title"></p>

fix(gpu): a saved-EXEC wave mask must survive the NGG fetch-index fold — R-Type Delta reaches its title screen (#2061)

`7cc74ef5` · [`assets/screenshots/rtype-delta-title.webp`](assets/screenshots/rtype-delta-title.webp)

### crisis-core-main-menu.png

<p align="center"><img src="assets/screenshots/crisis-core-main-menu.webp" alt="crisis core main menu"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`6ff91efa` · [`assets/screenshots/crisis-core-main-menu.webp`](assets/screenshots/crisis-core-main-menu.webp)

### crisis-core-title.png

<p align="center"><img src="assets/screenshots/crisis-core-title.webp" alt="crisis core title"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`6ff91efa` · [`assets/screenshots/crisis-core-title.webp`](assets/screenshots/crisis-core-title.webp)

### crisis-core-voice-language.png

<p align="center"><img src="assets/screenshots/crisis-core-voice-language.webp" alt="crisis core voice language"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`6ff91efa` · [`assets/screenshots/crisis-core-voice-language.webp`](assets/screenshots/crisis-core-voice-language.webp)

### sonic-frontiers-middleware-credits.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-middleware-credits.webp" alt="sonic frontiers middleware credits"></p>

fix(present): publish the flipped VideoOut buffer when no pass ever targets it — with a real guest-authorship test and the composited-frame gate intact (#2068)

`fe554cc8` · [`assets/screenshots/sonic-frontiers-middleware-credits.webp`](assets/screenshots/sonic-frontiers-middleware-credits.webp)

### sonic-frontiers-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-sega-logo.webp" alt="sonic frontiers sega logo"></p>

fix(present): publish the flipped VideoOut buffer when no pass ever targets it — with a real guest-authorship test and the composited-frame gate intact (#2068)

`fe554cc8` · [`assets/screenshots/sonic-frontiers-sega-logo.webp`](assets/screenshots/sonic-frontiers-sega-logo.webp)

### rtype-delta-opening-movie-colour.png

<p align="center"><img src="assets/screenshots/rtype-delta-opening-movie-colour.webp" alt="rtype delta opening movie colour"></p>

fix(gpu): recognise an AvPlayer chroma plane declared as a one-layer 2D array (#2037)

`7569a81c` · [`assets/screenshots/rtype-delta-opening-movie-colour.webp`](assets/screenshots/rtype-delta-opening-movie-colour.webp)

### issue-1946-health-warning-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1946-health-warning-before-after.webp" alt="issue 1946 health warning before after"></p>

fix(agc): offer the render-target-0 blend key on every SDK version — The Oregon Trail's whole UI layer was unblended (#1946) (#2031)

`c76504b4` · [`prosper/docs/screenshots/issue-1946-health-warning-before-after.webp`](prosper/docs/screenshots/issue-1946-health-warning-before-after.webp)

### issue-1946-slate-blend-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1946-slate-blend-before-after.webp" alt="issue 1946 slate blend before after"></p>

fix(agc): offer the render-target-0 blend key on every SDK version — The Oregon Trail's whole UI layer was unblended (#1946) (#2031)

`c76504b4` · [`prosper/docs/screenshots/issue-1946-slate-blend-before-after.webp`](prosper/docs/screenshots/issue-1946-slate-blend-before-after.webp)

### sonic-crossworlds-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-crossworlds-sega-logo.webp" alt="sonic crossworlds sega logo"></p>

docs(crossworlds): Sonic Racing: CrossWorlds reaches rung 1 — the SEGA logo renders (#2039)

`99a98232` · [`assets/screenshots/sonic-crossworlds-sega-logo.webp`](assets/screenshots/sonic-crossworlds-sega-logo.webp)

### little-nightmares-3-eula.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-eula.webp" alt="little nightmares 3 eula"></p>

docs(little-nightmares-3): #2014 is a wrong background clear, not a channel map — plus the first input route (#2041)

`8e7061f0` · [`assets/screenshots/little-nightmares-3-eula.webp`](assets/screenshots/little-nightmares-3-eula.webp)

## 2026-08-05

### little-nightmares-3-title-screen.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-title-screen.webp" alt="little nightmares 3 title screen"></p>

docs(little-nightmares-3): rung 2 — the title screen renders; land the ruled-out record (#2017)

`dafc7d0f` · [`assets/screenshots/little-nightmares-3-title-screen.webp`](assets/screenshots/little-nightmares-3-title-screen.webp)

### rtype-delta-rung1-logo-and-opening-movie.png

<p align="center"><img src="assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.webp" alt="rtype delta rung1 logo and opening movie"></p>

docs(rtype): R-Type Delta reaches rung 1 — the #1746 startup race is host CPU speed, not a prosper defect (#2009)

`91bd49ce` · [`assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.webp`](assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.webp)

### oregon-trail-title-screen.png

<p align="center"><img src="assets/screenshots/oregon-trail-title-screen.webp" alt="oregon trail title screen"></p>

diag(fault): probe every guest-pointer register at a worker fault, not just rdx/rax (#1992)

`c292b920` · [`assets/screenshots/oregon-trail-title-screen.webp`](assets/screenshots/oregon-trail-title-screen.webp)

### oregon-trail-gameloft-splash.png

<p align="center"><img src="assets/screenshots/oregon-trail-gameloft-splash.webp" alt="oregon trail gameloft splash"></p>

fix(hle): scePthread* must return libkernel-encoded errors — Oregon Trail advances three checkpoints (#1984)

`9710c4e5` · [`assets/screenshots/oregon-trail-gameloft-splash.webp`](assets/screenshots/oregon-trail-gameloft-splash.webp)

### oregon-trail-health-warning.png

<p align="center"><img src="assets/screenshots/oregon-trail-health-warning.webp" alt="oregon trail health warning"></p>

fix(hle): scePthread* must return libkernel-encoded errors — Oregon Trail advances three checkpoints (#1984)

`9710c4e5` · [`assets/screenshots/oregon-trail-health-warning.webp`](assets/screenshots/oregon-trail-health-warning.webp)

### bendy-dark-revival-title.png

<p align="center"><img src="assets/screenshots/bendy-dark-revival-title.webp" alt="bendy dark revival title"></p>

fix(avplayer): end a source nobody consumes on its own media clock (#1981)

`b7cb661c` · [`assets/screenshots/bendy-dark-revival-title.webp`](assets/screenshots/bendy-dark-revival-title.webp)

### asterix-babylon-intro-cutscene.png

<p align="center"><img src="assets/screenshots/asterix-babylon-intro-cutscene.webp" alt="asterix babylon intro cutscene"></p>

feat(avplayer): implement sceAvPlayerJumpToTime and honour the guest file-replacement reader (#1974)

`ea299e97` · [`assets/screenshots/asterix-babylon-intro-cutscene.webp`](assets/screenshots/asterix-babylon-intro-cutscene.webp)

### asterix-babylon-title.png

<p align="center"><img src="assets/screenshots/asterix-babylon-title.webp" alt="asterix babylon title"></p>

feat(avplayer): implement sceAvPlayerJumpToTime and honour the guest file-replacement reader (#1974)

`ea299e97` · [`assets/screenshots/asterix-babylon-title.webp`](assets/screenshots/asterix-babylon-title.webp)

### little-nightmares-3-boot-splash.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-boot-splash.webp" alt="little nightmares 3 boot splash"></p>

fix(ajm): accept the config revision — Little Nightmares III reaches rung 1 (#1966)

`8a21470f` · [`assets/screenshots/little-nightmares-3-boot-splash.webp`](assets/screenshots/little-nightmares-3-boot-splash.webp)

### sonic-frontiers-opening-sequence.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-opening-sequence.webp" alt="sonic frontiers opening sequence"></p>

docs(sonic-frontiers): rung 1 — it renders; the rung-0 reading was a failure-only diagnostic (#1969)

`56c6a3d7` · [`assets/screenshots/sonic-frontiers-opening-sequence.webp`](assets/screenshots/sonic-frontiers-opening-sequence.webp)

### sonic-frontiers-sonic-team-logo.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-sonic-team-logo.webp" alt="sonic frontiers sonic team logo"></p>

docs(sonic-frontiers): rung 1 — it renders; the rung-0 reading was a failure-only diagnostic (#1969)

`56c6a3d7` · [`assets/screenshots/sonic-frontiers-sonic-team-logo.webp`](assets/screenshots/sonic-frontiers-sonic-team-logo.webp)

## 2026-08-04

### oregon-trail-legal-popup.png

<p align="center"><img src="assets/screenshots/oregon-trail-legal-popup.webp" alt="oregon trail legal popup"></p>

feat(oregon): reach rung 1 — the startup legal popup renders on a default launch (#1950)

`1f510121` · [`assets/screenshots/oregon-trail-legal-popup.webp`](assets/screenshots/oregon-trail-legal-popup.webp)

### beneath-title.png

<p align="center"><img src="assets/screenshots/beneath-title.webp" alt="beneath title"></p>

docs: record Beneath title screen

`a2034338` · [`assets/screenshots/beneath-title.webp`](assets/screenshots/beneath-title.webp)

### forgotten-city-title.png

<p align="center"><img src="assets/screenshots/forgotten-city-title.webp" alt="forgotten city title"></p>

docs: record first-run compatibility baselines

`95ee2215` · [`assets/screenshots/forgotten-city-title.webp`](assets/screenshots/forgotten-city-title.webp)

### tactics-ogre-hevc-movie.png

<p align="center"><img src="assets/screenshots/tactics-ogre-hevc-movie.webp" alt="tactics ogre hevc movie"></p>

docs(tactics-ogre): record restored HEVC presentation

`fa6c95dc` · [`assets/screenshots/tactics-ogre-hevc-movie.webp`](assets/screenshots/tactics-ogre-hevc-movie.webp)

### tactics-ogre-reborn-gameplay.png

<p align="center"><img src="assets/screenshots/tactics-ogre-reborn-gameplay.webp" alt="tactics ogre reborn gameplay"></p>

Document Tactics Ogre tutorial battle

`2d1913ad` · [`assets/screenshots/tactics-ogre-reborn-gameplay.webp`](assets/screenshots/tactics-ogre-reborn-gameplay.webp)

## 2026-08-03

### tactics-ogre-reborn-opening-scene.png

<p align="center"><img src="assets/screenshots/tactics-ogre-reborn-opening-scene.webp" alt="tactics ogre reborn opening scene"></p>

Document Tactics Ogre opening scene

`b3cca048` · [`assets/screenshots/tactics-ogre-reborn-opening-scene.webp`](assets/screenshots/tactics-ogre-reborn-opening-scene.webp)

### house-of-the-dead-2-remake-gameplay.png

<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-gameplay.webp" alt="house of the dead 2 remake gameplay"></p>

Document House of the Dead 2 gameplay

`f977e012` · [`assets/screenshots/house-of-the-dead-2-remake-gameplay.webp`](assets/screenshots/house-of-the-dead-2-remake-gameplay.webp)

### tactics-ogre-title.png

<p align="center"><img src="assets/screenshots/tactics-ogre-title.webp" alt="tactics ogre title"></p>

Add Tactics Ogre title milestone

`da52181f` · [`assets/screenshots/tactics-ogre-title.webp`](assets/screenshots/tactics-ogre-title.webp)

### house-of-the-dead-2-remake-title.png

<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-title.webp" alt="house of the dead 2 remake title"></p>

Document newly tested game compatibility

`a1aff2e9` · [`assets/screenshots/house-of-the-dead-2-remake-title.webp`](assets/screenshots/house-of-the-dead-2-remake-title.webp)

### rtype-delta-movie-coordinate-progress.png

<p align="center"><img src="assets/screenshots/rtype-delta-movie-coordinate-progress.webp" alt="rtype delta movie coordinate progress"></p>

fix(gpu): honor unnormalized sample coordinates

`941da33c` · [`assets/screenshots/rtype-delta-movie-coordinate-progress.webp`](assets/screenshots/rtype-delta-movie-coordinate-progress.webp)

### blue-prince-hall.png

<p align="center"><img src="assets/screenshots/blue-prince-hall.webp" alt="blue prince hall"></p>

Fix Blue Prince hall snapshot guard (#1813)

`bf861656` · [`assets/screenshots/blue-prince-hall.webp`](assets/screenshots/blue-prince-hall.webp)

## 2026-08-02

### earthion-title-menu.png

<p align="center"><img src="assets/screenshots/earthion-title-menu.webp" alt="earthion title menu"></p>

feat(earthion): route past the intro — the "missing picture" was a black text page (#1590) (#1775)

`8b1f6254` · [`assets/screenshots/earthion-title-menu.webp`](assets/screenshots/earthion-title-menu.webp)

### tales-graces-f-options.png

<p align="center"><img src="assets/screenshots/tales-graces-f-options.webp" alt="tales graces f options"></p>

feat(talesgraces): routes to the title screen and the new-game Options screen (PPSA19991 rung 2) (#1759)

`e50594cc` · [`assets/screenshots/tales-graces-f-options.webp`](assets/screenshots/tales-graces-f-options.webp)

### tales-graces-f-title.png

<p align="center"><img src="assets/screenshots/tales-graces-f-title.webp" alt="tales graces f title"></p>

feat(talesgraces): routes to the title screen and the new-game Options screen (PPSA19991 rung 2) (#1759)

`e50594cc` · [`assets/screenshots/tales-graces-f-title.webp`](assets/screenshots/tales-graces-f-title.webp)

### bendy-gameplay.png

<p align="center"><img src="assets/screenshots/bendy-gameplay.webp" alt="bendy gameplay"></p>

docs(compat): boot sweep of fourteen titles on c79f742e — corrected rungs, four new rows, and a frame-rate census (#1740)

`673aacb1` · [`assets/screenshots/bendy-gameplay.webp`](assets/screenshots/bendy-gameplay.webp)

### bendy-title.png

<p align="center"><img src="assets/screenshots/bendy-title.webp" alt="bendy title"></p>

docs(compat): boot sweep of fourteen titles on c79f742e — corrected rungs, four new rows, and a frame-rate census (#1740)

`673aacb1` · [`assets/screenshots/bendy-title.webp`](assets/screenshots/bendy-title.webp)

### pathless-title.png

<p align="center"><img src="assets/screenshots/pathless-title.webp" alt="pathless title"></p>

docs(compat): boot sweep of fourteen titles on c79f742e — corrected rungs, four new rows, and a frame-rate census (#1740)

`673aacb1` · [`assets/screenshots/pathless-title.webp`](assets/screenshots/pathless-title.webp)

### plucky-squire-title.png

<p align="center"><img src="assets/screenshots/plucky-squire-title.webp" alt="plucky squire title"></p>

docs(compat): boot sweep of fourteen titles on c79f742e — corrected rungs, four new rows, and a frame-rate census (#1740)

`673aacb1` · [`assets/screenshots/plucky-squire-title.webp`](assets/screenshots/plucky-squire-title.webp)

### astro-bot-opening-cinematic.png

<p align="center"><img src="assets/screenshots/astro-bot-opening-cinematic.webp" alt="astro bot opening cinematic"></p>

docs(astro): Astro Bot enters COMPATIBILITY.md at rung 2 — the title screen renders (#1736)

`a1f3b05c` · [`assets/screenshots/astro-bot-opening-cinematic.webp`](assets/screenshots/astro-bot-opening-cinematic.webp)

### astro-bot-title.png

<p align="center"><img src="assets/screenshots/astro-bot-title.webp" alt="astro bot title"></p>

docs(astro): Astro Bot enters COMPATIBILITY.md at rung 2 — the title screen renders (#1736)

`a1f3b05c` · [`assets/screenshots/astro-bot-title.webp`](assets/screenshots/astro-bot-title.webp)

### astro-bot-worldmap-background.png

<p align="center"><img src="assets/screenshots/astro-bot-worldmap-background.webp" alt="astro bot worldmap background"></p>

fix(gpu): CB_COLOR_CONTROL.MODE must not override the guest's colour write mask (#1728)

`58abbc7d` · [`assets/screenshots/astro-bot-worldmap-background.webp`](assets/screenshots/astro-bot-worldmap-background.webp)

## 2026-08-01

### tales-graces-f-criware.png

<p align="center"><img src="assets/screenshots/tales-graces-f-criware.webp" alt="tales graces f criware"></p>

fix(hle): deliver the APR completion event for a zero-tag binding (#1666) (#1667)

`f1c716e8` · [`assets/screenshots/tales-graces-f-criware.webp`](assets/screenshots/tales-graces-f-criware.webp)

### tales-graces-f-publisher.png

<p align="center"><img src="assets/screenshots/tales-graces-f-publisher.webp" alt="tales graces f publisher"></p>

fix(hle): deliver the APR completion event for a zero-tag binding (#1666) (#1667)

`f1c716e8` · [`assets/screenshots/tales-graces-f-publisher.webp`](assets/screenshots/tales-graces-f-publisher.webp)

### issue-1630-grid-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-grid-after.webp" alt="issue 1630 grid after"></p>

feat(app): per-title background art and focus music in the library (#1647)

`0553c329` · [`prosper/docs/screenshots/issue-1630-grid-after.webp`](prosper/docs/screenshots/issue-1630-grid-after.webp)

### issue-1630-grid-before.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-grid-before.webp" alt="issue 1630 grid before"></p>

feat(app): per-title background art and focus music in the library (#1647)

`0553c329` · [`prosper/docs/screenshots/issue-1630-grid-before.webp`](prosper/docs/screenshots/issue-1630-grid-before.webp)

### issue-1630-library-background-1.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-1.webp" alt="issue 1630 library background 1"></p>

feat(app): per-title background art and focus music in the library (#1647)

`0553c329` · [`prosper/docs/screenshots/issue-1630-library-background-1.webp`](prosper/docs/screenshots/issue-1630-library-background-1.webp)

### issue-1630-library-background-2.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-2.webp" alt="issue 1630 library background 2"></p>

feat(app): per-title background art and focus music in the library (#1647)

`0553c329` · [`prosper/docs/screenshots/issue-1630-library-background-2.webp`](prosper/docs/screenshots/issue-1630-library-background-2.webp)

### issue-1630-library-background-3.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-3.webp" alt="issue 1630 library background 3"></p>

feat(app): per-title background art and focus music in the library (#1647)

`0553c329` · [`prosper/docs/screenshots/issue-1630-library-background-3.webp`](prosper/docs/screenshots/issue-1630-library-background-3.webp)

### syberia-gameplay.png

<p align="center"><img src="assets/screenshots/syberia-gameplay.webp" alt="syberia gameplay"></p>

docs(syberia): validated route to gameplay, and localize the black scene to one format gap (#1622)

`757d29a3` · [`assets/screenshots/syberia-gameplay.webp`](assets/screenshots/syberia-gameplay.webp)

### syberia-title.png

<p align="center"><img src="assets/screenshots/syberia-title.webp" alt="syberia title"></p>

docs(syberia): validated route to gameplay, and localize the black scene to one format gap (#1622)

`757d29a3` · [`assets/screenshots/syberia-title.webp`](assets/screenshots/syberia-title.webp)

### worms-armageddon-gameplay.png

<p align="center"><img src="assets/screenshots/worms-armageddon-gameplay.webp" alt="worms armageddon gameplay"></p>

fix(pad): scePadGetHandle looks up an open handle instead of fabricating one (#1623)

`45024ab2` · [`assets/screenshots/worms-armageddon-gameplay.webp`](assets/screenshots/worms-armageddon-gameplay.webp)

## 2026-07-31

### syberia-profile.png

<p align="center"><img src="assets/screenshots/syberia-profile.webp" alt="syberia profile"></p>

fix(agc): register sceAgcAcbWriteData — Syberia goes from hard hang to its profile menu (#1610)

`0502aaf1` · [`assets/screenshots/syberia-profile.webp`](assets/screenshots/syberia-profile.webp)

### nikoderiko-title.png

<p align="center"><img src="assets/screenshots/nikoderiko-title.webp" alt="nikoderiko title"></p>

docs(compat): add Nikoderiko at title screen and The Oregon Trail at research tier (#1608)

`408201a4` · [`assets/screenshots/nikoderiko-title.webp`](assets/screenshots/nikoderiko-title.webp)

### greak-title.png

<p align="center"><img src="assets/screenshots/greak-title.webp" alt="greak title"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`68259cee` · [`assets/screenshots/greak-title.webp`](assets/screenshots/greak-title.webp)

### greak.png

<p align="center"><img src="assets/screenshots/greak.webp" alt="greak"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`68259cee` · [`assets/screenshots/greak.webp`](assets/screenshots/greak.webp)

### rugrats-title.png

<p align="center"><img src="assets/screenshots/rugrats-title.webp" alt="rugrats title"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`68259cee` · [`assets/screenshots/rugrats-title.webp`](assets/screenshots/rugrats-title.webp)

### rugrats.png

<p align="center"><img src="assets/screenshots/rugrats.webp" alt="rugrats"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`68259cee` · [`assets/screenshots/rugrats.webp`](assets/screenshots/rugrats.webp)

### asterix-slap-them-all.png

<p align="center"><img src="assets/screenshots/asterix-slap-them-all.webp" alt="asterix slap them all"></p>

docs(compat): add Asterix Slap Them All and Summer Sports Games at gameplay (#1604)

`8f6095a8` · [`assets/screenshots/asterix-slap-them-all.webp`](assets/screenshots/asterix-slap-them-all.webp)

### summer-sports-games.png

<p align="center"><img src="assets/screenshots/summer-sports-games.webp" alt="summer sports games"></p>

docs(compat): add Asterix Slap Them All and Summer Sports Games at gameplay (#1604)

`8f6095a8` · [`assets/screenshots/summer-sports-games.webp`](assets/screenshots/summer-sports-games.webp)

### joe-mac-menu.png

<p align="center"><img src="assets/screenshots/joe-mac-menu.webp" alt="joe mac menu"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`46bf1a27` · [`assets/screenshots/joe-mac-menu.webp`](assets/screenshots/joe-mac-menu.webp)

### joe-mac.png

<p align="center"><img src="assets/screenshots/joe-mac.webp" alt="joe mac"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`46bf1a27` · [`assets/screenshots/joe-mac.webp`](assets/screenshots/joe-mac.webp)

### worms-armageddon-title.png

<p align="center"><img src="assets/screenshots/worms-armageddon-title.webp" alt="worms armageddon title"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`46bf1a27` · [`assets/screenshots/worms-armageddon-title.webp`](assets/screenshots/worms-armageddon-title.webp)

### alex-kidd.png

<p align="center"><img src="assets/screenshots/alex-kidd.webp" alt="alex kidd"></p>

test(snapshot): reviewed alexkidd-gameplay content guard — PPSA02664 reaches ladder rung 6 (#1582)

`44a08fa5` · [`assets/screenshots/alex-kidd.webp`](assets/screenshots/alex-kidd.webp)

### dragon-quest-vii-onboarding.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-onboarding.webp" alt="dragon quest vii onboarding"></p>

docs: record Dragon Quest VII onboarding

`ae95a013` · [`assets/screenshots/dragon-quest-vii-onboarding.webp`](assets/screenshots/dragon-quest-vii-onboarding.webp)

### dragon-quest-vii-name-confirmation.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-confirmation.webp" alt="dragon quest vii name confirmation"></p>

Document DQ7 name confirmation milestone

`e02961e2` · [`assets/screenshots/dragon-quest-vii-name-confirmation.webp`](assets/screenshots/dragon-quest-vii-name-confirmation.webp)

### dragon-quest-vii-name-entry.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-entry.webp" alt="dragon quest vii name entry"></p>

Document Dragon Quest name entry

`1a6ef445` · [`assets/screenshots/dragon-quest-vii-name-entry.webp`](assets/screenshots/dragon-quest-vii-name-entry.webp)

### space-adventure-cobra.png

<p align="center"><img src="assets/screenshots/space-adventure-cobra.webp" alt="space adventure cobra"></p>

fix(runtime): preserve guest TLS across write-watch faults

`ce09bf50` · [`assets/screenshots/space-adventure-cobra.webp`](assets/screenshots/space-adventure-cobra.webp)

### gris.png

<p align="center"><img src="assets/screenshots/gris.webp" alt="gris"></p>

docs: record GRIS opening gameplay

`4424ba00` · [`assets/screenshots/gris.webp`](assets/screenshots/gris.webp)

### issue-1459-astrobot-blue-fmv-gpu-present.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.webp" alt="issue 1459 astrobot blue fmv gpu present"></p>

docs: capture Astro Bot blue intro

`264c70c2` · [`prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.webp`](prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.webp)

## 2026-07-30

### issue-1471-library-empty.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-empty.webp" alt="issue 1471 library empty"></p>

fix(app): address review findings on the library view

`f5ffe44d` · [`prosper/docs/screenshots/issue-1471-library-empty.webp`](prosper/docs/screenshots/issue-1471-library-empty.webp)

### issue-1471-library-scrolled.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-scrolled.webp" alt="issue 1471 library scrolled"></p>

fix(app): address review findings on the library view

`f5ffe44d` · [`prosper/docs/screenshots/issue-1471-library-scrolled.webp`](prosper/docs/screenshots/issue-1471-library-scrolled.webp)

### issue-1471-library-grid.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-grid.webp" alt="issue 1471 library grid"></p>

feat(app): draw the game library with Dear ImGui

`352a2a0b` · [`prosper/docs/screenshots/issue-1471-library-grid.webp`](prosper/docs/screenshots/issue-1471-library-grid.webp)

### issue-1459-astrobot-linux-indirect-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.webp" alt="issue 1459 astrobot linux indirect title"></p>

gpu: execute AGC indirect work after producers

`53598b4c` · [`prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.webp`](prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.webp)

## 2026-07-29

### issue-1469-drop-messenger.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-drop-messenger.webp" alt="issue 1469 drop messenger"></p>

docs(app): interactive-open evidence screenshots (#1469)

`2b329eb7` · [`prosper/docs/screenshots/issue-1469-drop-messenger.webp`](prosper/docs/screenshots/issue-1469-drop-messenger.webp)

### issue-1469-picker-messenger.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-picker-messenger.webp" alt="issue 1469 picker messenger"></p>

docs(app): interactive-open evidence screenshots (#1469)

`2b329eb7` · [`prosper/docs/screenshots/issue-1469-picker-messenger.webp`](prosper/docs/screenshots/issue-1469-picker-messenger.webp)

### issue-1469-reject-not-a-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-reject-not-a-title.webp" alt="issue 1469 reject not a title"></p>

docs(app): interactive-open evidence screenshots (#1469)

`2b329eb7` · [`prosper/docs/screenshots/issue-1469-reject-not-a-title.webp`](prosper/docs/screenshots/issue-1469-reject-not-a-title.webp)

### issue-1469-relaunch-blasphemous2.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.webp" alt="issue 1469 relaunch blasphemous2"></p>

docs(app): interactive-open evidence screenshots (#1469)

`2b329eb7` · [`prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.webp`](prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.webp)

### issue-1459-astrobot-worldmap-current.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.webp" alt="issue 1459 astrobot worldmap current"></p>

docs: capture current Astro Bot world map

`14953eeb` · [`prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.webp`](prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.webp)

### issue-1466-astrobot-direct-tile.png

<p align="center"><img src="prosper/docs/screenshots/issue-1466-astrobot-direct-tile.webp" alt="issue 1466 astrobot direct tile"></p>

perf(gpu): tile mapped storage images directly

`1d60f71e` · [`prosper/docs/screenshots/issue-1466-astrobot-direct-tile.webp`](prosper/docs/screenshots/issue-1466-astrobot-direct-tile.webp)

## 2026-07-26

### issue-1287-hall-live-fixed.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-live-fixed.webp" alt="issue 1287 hall live fixed"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`67760515` · [`prosper/docs/screenshots/issue-1287-hall-live-fixed.webp`](prosper/docs/screenshots/issue-1287-hall-live-fixed.webp)

### issue-1287-hall-live-vs-oracle.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.webp" alt="issue 1287 hall live vs oracle"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`67760515` · [`prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.webp`](prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.webp)

### issue-1287-manor-approach-live.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-manor-approach-live.webp" alt="issue 1287 manor approach live"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`67760515` · [`prosper/docs/screenshots/issue-1287-manor-approach-live.webp`](prosper/docs/screenshots/issue-1287-manor-approach-live.webp)

### issue-1427-hall-geometry-restored.png

<p align="center"><img src="prosper/docs/screenshots/issue-1427-hall-geometry-restored.webp" alt="issue 1427 hall geometry restored"></p>

fix(render): upload a buffer binding's whole declared range, not the first 1 MiB (#1429)

`323c5244` · [`prosper/docs/screenshots/issue-1427-hall-geometry-restored.webp`](prosper/docs/screenshots/issue-1427-hall-geometry-restored.webp)

### issue-1427-oracle-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1427-oracle-before-after.webp" alt="issue 1427 oracle before after"></p>

fix(render): upload a buffer binding's whole declared range, not the first 1 MiB (#1429)

`323c5244` · [`prosper/docs/screenshots/issue-1427-oracle-before-after.webp`](prosper/docs/screenshots/issue-1427-oracle-before-after.webp)

### issue-1287-hall-materials-fixed.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-materials-fixed.webp" alt="issue 1287 hall materials fixed"></p>

docs: Blue Prince hall with correct materials (#1287 milestone frame) (#1418)

`83e98a6f` · [`prosper/docs/screenshots/issue-1287-hall-materials-fixed.webp`](prosper/docs/screenshots/issue-1287-hall-materials-fixed.webp)

## 2026-07-25

### issue-1334-hall-default-tonemapped.png

<p align="center"><img src="prosper/docs/screenshots/issue-1334-hall-default-tonemapped.webp" alt="issue 1334 hall default tonemapped"></p>

fix(gpu): GPU-copy the MSAA resolve into the destination persistent image (#1382)

`af21d480` · [`prosper/docs/screenshots/issue-1334-hall-default-tonemapped.webp`](prosper/docs/screenshots/issue-1334-hall-default-tonemapped.webp)

### issue-1287-hall-bundle-tonemapped.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.webp" alt="issue 1287 hall bundle tonemapped"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`9e22c190` · [`prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.webp`](prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.webp)

### issue-1287-hall-magenta-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-magenta-prosper.webp" alt="issue 1287 hall magenta prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`9e22c190` · [`prosper/docs/screenshots/issue-1287-hall-magenta-prosper.webp`](prosper/docs/screenshots/issue-1287-hall-magenta-prosper.webp)

### issue-1287-hall-night-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-night-prosper.webp" alt="issue 1287 hall night prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`9e22c190` · [`prosper/docs/screenshots/issue-1287-hall-night-prosper.webp`](prosper/docs/screenshots/issue-1287-hall-night-prosper.webp)

### issue-1287-hall-nobatch-live.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-nobatch-live.webp" alt="issue 1287 hall nobatch live"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`9e22c190` · [`prosper/docs/screenshots/issue-1287-hall-nobatch-live.webp`](prosper/docs/screenshots/issue-1287-hall-nobatch-live.webp)

### issue-1287-vestibule-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-vestibule-prosper.webp" alt="issue 1287 vestibule prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`9e22c190` · [`prosper/docs/screenshots/issue-1287-vestibule-prosper.webp`](prosper/docs/screenshots/issue-1287-vestibule-prosper.webp)

### issue-1356-gris-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1356-gris-title.webp" alt="issue 1356 gris title"></p>

feat: bring GRIS and Cobra to title with audio (#1368)

`6fd8be05` · [`prosper/docs/screenshots/issue-1356-gris-title.webp`](prosper/docs/screenshots/issue-1356-gris-title.webp)

### issue-1356-space-adventure-cobra-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.webp" alt="issue 1356 space adventure cobra title"></p>

feat: bring GRIS and Cobra to title with audio (#1368)

`6fd8be05` · [`prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.webp`](prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.webp)

### dragon-quest-vii-title.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-title.webp" alt="dragon quest vii title"></p>

docs: publish Dragon Quest VII title capture

`43ff887f` · [`assets/screenshots/dragon-quest-vii-title.webp`](assets/screenshots/dragon-quest-vii-title.webp)

### issue-1352-wall-shading-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1352-wall-shading-after.webp" alt="issue 1352 wall shading after"></p>

fix(gpu): DEPTH_CLEAR_ENABLE acts only through the enabled depth-write path (#1354)

`7eb15f24` · [`prosper/docs/screenshots/issue-1352-wall-shading-after.webp`](prosper/docs/screenshots/issue-1352-wall-shading-after.webp)

### issue-1352-wall-shading-before.png

<p align="center"><img src="prosper/docs/screenshots/issue-1352-wall-shading-before.webp" alt="issue 1352 wall shading before"></p>

fix(gpu): DEPTH_CLEAR_ENABLE acts only through the enabled depth-write path (#1354)

`7eb15f24` · [`prosper/docs/screenshots/issue-1352-wall-shading-before.webp`](prosper/docs/screenshots/issue-1352-wall-shading-before.webp)

## 2026-07-24

### blue-prince-title.png

<p align="center"><img src="assets/screenshots/blue-prince-title.webp" alt="blue prince title"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`28cc99ba` · [`assets/screenshots/blue-prince-title.webp`](assets/screenshots/blue-prince-title.webp)

### terminator-title.png

<p align="center"><img src="assets/screenshots/terminator-title.webp" alt="terminator title"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`28cc99ba` · [`assets/screenshots/terminator-title.webp`](assets/screenshots/terminator-title.webp)

### terminator.png

<p align="center"><img src="assets/screenshots/terminator.webp" alt="terminator"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`28cc99ba` · [`assets/screenshots/terminator.webp`](assets/screenshots/terminator.webp)

### gta5-main-menu.png

<p align="center"><img src="assets/screenshots/gta5-main-menu.webp" alt="gta5 main menu"></p>

docs: show GTA V current renderer state (#1339)

`70bab8f9` · [`assets/screenshots/gta5-main-menu.webp`](assets/screenshots/gta5-main-menu.webp)

### gta5-title.png

<p align="center"><img src="assets/screenshots/gta5-title.webp" alt="gta5 title"></p>

docs: show GTA V current renderer state (#1339)

`70bab8f9` · [`assets/screenshots/gta5-title.webp`](assets/screenshots/gta5-title.webp)

### blasphemous2-title.png

<p align="center"><img src="assets/screenshots/blasphemous2-title.webp" alt="blasphemous2 title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/blasphemous2-title.webp`](assets/screenshots/blasphemous2-title.webp)

### blasphemous2.png

<p align="center"><img src="assets/screenshots/blasphemous2.webp" alt="blasphemous2"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/blasphemous2.webp`](assets/screenshots/blasphemous2.webp)

### dead-cells-title.png

<p align="center"><img src="assets/screenshots/dead-cells-title.webp" alt="dead cells title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/dead-cells-title.webp`](assets/screenshots/dead-cells-title.webp)

### dead-cells.png

<p align="center"><img src="assets/screenshots/dead-cells.webp" alt="dead cells"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/dead-cells.webp`](assets/screenshots/dead-cells.webp)

### evergate-title.png

<p align="center"><img src="assets/screenshots/evergate-title.webp" alt="evergate title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/evergate-title.webp`](assets/screenshots/evergate-title.webp)

### evergate.png

<p align="center"><img src="assets/screenshots/evergate.webp" alt="evergate"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/evergate.webp`](assets/screenshots/evergate.webp)

### messenger-title.png

<p align="center"><img src="assets/screenshots/messenger-title.webp" alt="messenger title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/messenger-title.webp`](assets/screenshots/messenger-title.webp)

### messenger.png

<p align="center"><img src="assets/screenshots/messenger.webp" alt="messenger"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`795e609d` · [`assets/screenshots/messenger.webp`](assets/screenshots/messenger.webp)

## 2026-07-19

### issue-897-astrobot-linux-natural-opening-midfade.png

<p align="center"><img src="prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.webp" alt="issue 897 astrobot linux natural opening midfade"></p>

docs(astrobot): attach natural Linux graphics captures

`c7ed9204` · [`prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.webp`](prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.webp)

### issue-897-astrobot-linux-natural-opening-visible.png

<p align="center"><img src="prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.webp" alt="issue 897 astrobot linux natural opening visible"></p>

docs(astrobot): attach natural Linux graphics captures

`c7ed9204` · [`prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.webp`](prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.webp)

## 2026-07-18

### issue-825-astrobot-windows-sony-presents.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.webp" alt="issue 825 astrobot windows sony presents"></p>

docs(astrobot): attach Windows progress captures

`a0620f7a` · [`prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.webp`](prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.webp)

### issue-825-astrobot-windows-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-windows-title.webp" alt="issue 825 astrobot windows title"></p>

docs(astrobot): attach Windows progress captures

`a0620f7a` · [`prosper/docs/screenshots/issue-825-astrobot-windows-title.webp`](prosper/docs/screenshots/issue-825-astrobot-windows-title.webp)

## 2026-07-17

### issue-825-astrobot-linux-sony-presents.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.webp" alt="issue 825 astrobot linux sony presents"></p>

docs(astrobot): attach Linux loading screenshot

`6b45f80f` · [`prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.webp`](prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.webp)
