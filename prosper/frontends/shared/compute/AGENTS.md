# `frontends/shared/compute` — what the compute backend decides, separated from how it decides it

Header-only, Vulkan-free, dependency-light decision logic and instrumentation for the live compute
backend (`live/live_compute.cpp`) and for the one boundary it shares with the renderer
(`live/live_renderer.cpp`). Nothing here touches a device, guest memory, or the write-watch
mechanism: every file answers a question from plain metadata, so it is unit-testable
without a GPU. The tests live in `frontends/shared/tests/`.

**Why the decisions live here and not beside their call sites.** `live_compute.cpp` is eleven
thousand lines, and its gates were historically spelled as long `||` chains inside the functions
that acted on them. A chain like that answers *whether* and never *which*, so a surface that
declined for one of eight reasons produced exactly the same evidence as one that declined for
another — and several nights of this project's history were spent guessing which. Lifting the gate
into a `constexpr` classifier here makes the reason a value, which makes it countable, which makes
"the borrow misses" into "the borrow misses because the entry's key disagrees on `tile_mode`".

**The boundary against its siblings.** `shared/texture/` owns the *validation* policies both caches
share — whether guest bytes must be re-read, how many proven-unchanged looks earn a page watch.
This folder owns the compute backend's own eligibility and identity decisions plus the censuses that
report them. Cache materialization, Vulkan resource creation and the guest↔device copies stay in
`live/`, which is where the types they act on are declared.

**A census here is a partition, not a log.** Every one of these files counts a *decision*: each
observation lands in exactly one bucket, the report always prints the denominator, and a bucket that
is zero still prints. The last part is deliberate and has been paid for — a bucket that disappears
when empty is indistinguishable from an instrument that never reached it, which is how a correct
finding got falsified by a zero read off the wrong field: `[render-timing]`'s pre-existing
`protect=` counts mprotect *failures* and is healthily zero, and a truncated grep read it as the
activity counter that sits further along the same 288-character line (#3307, and the
instrument-trap row #3317 adds). Counting is
unconditional wherever the counted branch is expensive enough that a relaxed atomic add is free by
comparison; only the *report* is gated on an environment variable.

**The one thing a newcomer gets wrong.** `compute_transfer_gate_census.hpp`'s counters are gated on
a *selector* — `PROSPER_COMPUTE_TRANSFER_PRODUCER_HASH` / `..._CONSUMER_HASH` — so they observe
nothing at all unless a specific producer/consumer program pair was named on the command line. The
other censuses here count on every run. Reading a zero from the transfer-gate census on a default
launch says nothing about the code it instruments.
