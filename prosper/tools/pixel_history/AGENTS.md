# Pixel history — why is this pixel this colour?

This folder answers one question about a rendered frame, and it is the question most of
this project's black-render investigations actually turn on: **a pixel is the wrong
colour — did anything draw there at all, was it drawn and thrown away, did the shader run
and compute black, or did the shader compute a colour that the store then lost?**

Those answers send you to different files. Two more exist for cases that would otherwise be
misreported: `CLEARED_AFTER_DRAW` (a later clear wiped surviving draws) and
`OUTPUT_UNTRUSTED` (the explaining event has an unbound pixel shader, so its output is
undefined and the tool refuses to guess). Telling them apart by inference —
draw censuses, disabling passes, bisecting state — is what has historically cost this
project hours per defect, and the census route has produced retracted issues. RenderDoc
answers it directly, per event, and `pixel_history.py` turns that into one call with one
verdict.

What lives here:

- `pixel_history.py` — the tool. Takes an `.rdc`, picks a pixel (the **brightest** one by
  default, because on a black frame the centre is exactly where nothing happened), and
  reports `NOTHING_DREW` / `ALL_REJECTED` / `SHADER_WROTE_BLACK` / `STORE_LOST_IT` with
  the per-event detail behind it. It refuses to report at all if the replay driver says it
  has no pixel-history support, because an empty history and an unsupported driver look
  identical and mean opposite things.
- `control.c` + `shaders/` — a construction with a **known answer**, in five regions producing
  **four different verdicts**: the draw sequence (`PIXEL_WAS_WRITTEN`), a passing draw that
  computes black (`SHADER_WROTE_BLACK`), a passing draw through a `colorWriteMask=0` pipeline
  (`STORE_LOST_IT`), and a region where every draw is discarded (`ALL_REJECTED`). A control that
  only ever produced one verdict would test the machinery and not the distinctions. Run
  `pixel_history.py --expect-control` against its capture on any new driver before trusting the
  tool there; it checks every region's verdict *and* every rejection reason, which is the part a
  final picture cannot see.
- `test_pixel_history.py` — the verdict logic, which is the part CI can reach.

## The control is not ceremony

It has earned itself **three times**, every time against an assumption that looked safe — and
each one was written down as a confident comment before it was measured.

1. **RenderDoc populates `shaderOut` for events whose fragment shader never ran** — a scissored
   draw reports the *previous* draw's colour. On a game capture that is invisible: the number
   looks like a shader output and there is nothing to check it against. The control caught it
   because the value belonged to a draw whose colour was known to be different. Instrument
   trap 268.
2. **A clear IS a pixel-history event, and it passes.** The control's own comment had asserted
   the opposite. A region whose every draw was discarded read `SHADER_WROTE_BLACK` instead of
   `ALL_REJECTED`, because the clear sat in its history as a passing black event.
3. **And the two kinds of clear are not the same set.** Having keyed the fix on
   `ActionFlags::Clear`, a renderpass `loadOp = CLEAR` — the *common* form — still read
   `SHADER_WROTE_BLACK`, because it records `ResourceUsage::Clear` but carries no
   `ActionFlags::Clear`. The control could not see it, because it built only the flagged form:
   **it was validating the path the code handled.** It now builds both, and the check fails if
   either detection path is lost. Instrument trap 269.

Generalise: **a tool that reads a value cannot tell you the value is meaningful.** Only a
construction whose answer you already know can — and only if that construction contains the hard
case. The first attempt at (2) fixed the control by moving the clear out of the captured frame,
which made it pass by removing the phenomenon. That is the failure mode to watch for: a control
that has been adjusted until it agrees with the code is no longer evidence about the code.

Event ids order events by submission and are what `PixelModification` sorts by, so the
"a clear landed after the last surviving draw" test is sound within one queue. It is not
meaningful across queues — a limitation inherited from RenderDoc's pixel history, not introduced
here.

## Boundary

This folder does not own capture (`renderdoccmd`, or prosper's own F9 bundles), replay of
prosper's own command stream (`tools/gpu_replay/`), or instrument capability checks
(`tools/doctor/`). It owns the question above and the control that makes its answers
trustworthy. Prosper's seeded F9 bundles retain guest-side state an `.rdc` cannot replace;
prefer them when the question is about prosper's translation rather than about what the GPU
was asked to do.
