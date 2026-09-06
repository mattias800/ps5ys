# Pixel history — why is this pixel this colour?

This folder answers one question about a rendered frame, and it is the question most of
this project's black-render investigations actually turn on: **a pixel is the wrong
colour — did anything draw there at all, was it drawn and thrown away, did the shader run
and compute black, or did the shader compute a colour that the store then lost?**

Those four answers send you to four different files. Telling them apart by inference —
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
- `control.c` + `shaders/` — a construction with a **known answer**: five full-screen draws
  at one named pixel, differing in exactly one property each, engineered to produce a pass,
  a depth failure, a scissor rejection, a shader discard and a final pass. Run
  `pixel_history.py --expect-control` against its capture on any new driver before
  trusting the tool there.
- `test_pixel_history.py` — the verdict logic, which is the part CI can reach.

## The control is not ceremony

It has already earned itself once. RenderDoc populates `shaderOut` for events whose
fragment shader **never ran** — a scissored draw reports the *previous* draw's colour. On
a game capture that is invisible: the number looks like a shader output and there is
nothing to check it against. The control caught it immediately, because the value it
reported belonged to a draw whose colour was known to be different. The tool now suppresses
shader output for every rejection that happens before fragment execution, and says
`suppressed (no fragment ran)` rather than printing a plausible lie.

Generalise from that: **a tool that reads a value cannot tell you the value is meaningful.**
Only a construction whose answer you already know can.

## Boundary

This folder does not own capture (`renderdoccmd`, or prosper's own F9 bundles), replay of
prosper's own command stream (`tools/gpu_replay/`), or instrument capability checks
(`tools/doctor/`). It owns the question above and the control that makes its answers
trustworthy. Prosper's seeded F9 bundles retain guest-side state an `.rdc` cannot replace;
prefer them when the question is about prosper's translation rather than about what the GPU
was asked to do.
