# `frontends/shared/tests` — shared frontend policy and metadata checks

Small CPU-side tests for the decisions shared by the app, screenshot tool and live backends:
texture/write-watch policies, compute alias plans and censuses, render-target/present extents,
capture windows and performance records, and scratch-buffer lifetime rules. Put a test here when
hand-built descriptors, counters, byte arrays or policy inputs can establish the contract without
running a guest or submitting Vulkan work. Include positive and negative cases; an alias-plan test
proves metadata grouping and access union, not that the live caller seeds, binds or executes correctly.

Device-backed integration belongs with the existing harnesses in `tests/shared/live/`,
`tests/shared/present/` or the relevant `tests/gpu/` family. Some neighboring tests also have CPU
arms, so navigate by the contract rather than assuming the directory name guarantees GPU use.
`tests/fixtures/render_runner.h` is the real graphics backend used by those harnesses, not a mock;
the compute backend is separate. Do not pull either into a metadata-only test just to exercise a
decision that already has a device-free interface.

Register CPU-only tests in the platform-independent test section of `prosper/CMakeLists.txt`, with
`frontends` for `shared/...` includes and `src` plus `prosper_core` only where needed. Do not gate
them on the live-renderer target or Vulkan availability. The present-blit policy check is a narrow
exception requiring Vulkan headers/types but no device; its CMake guard is not a template for
ordinary policy tests. Environment-controlled variants need their own CTest environment when the
implementation reads a setting once per process.
