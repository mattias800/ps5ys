# `tests/gpu/recompiler` — translation tests and the legacy compute execution fixture

Most tests here ask a question of `src/gpu/recompiler` alone: does this instruction stream decode,
does it recompile, does the module it produces have the shape the contract promises, and — just as
often — does a stream that must NOT compile still reject, and reject *loudly*. No device is created,
so these run everywhere and are the cheapest place in the project to pin a translator contract.
The historical exception is `test_game_compute.cpp`: it executes synthetic shaders through the live
Vulkan backend, including storage writeback, cache authority, and real mapped-page write watches.
Its existing end-to-end fixtures remain here; new standalone execution tests belong in `execute/`.

**The boundary against its siblings.** `tests/gpu/execute/` runs modules on a real device and
asserts pixels or buffer contents; anything needing a queue belongs there, not here. Two large
recompiler tests also live one level up in `tests/gpu/` for historical reasons —
`test_rdna2_spirv_struct.cpp` (structural module validation) and `test_recompile_coverage.cpp` —
and new work usually belongs in this folder instead.

**Prefer a small binary over another arm in a big one, whenever rejects are the subject.** Most
tests here return at their first `[FAIL]`, which is fine for a suite that should be all-green but
useless the moment you want to know *which* assertions moved: a mutation run reports the first
failure and nothing after it. A reject-contract test is exactly that case, because the way you
establish it is load-bearing at all is to break the mechanism deliberately and read which arms
redden. So a focused reject test gets its own executable and a `CHECK` macro that counts failures
and keeps going (`test_entry_m0_dispatcher.cpp`, `test_vertex_fetch_reject_diagnostic.cpp`).

**A reject arm without controls beside it is not evidence.** An empty result means the stream was
refused; it never says by whom or why, and a recompiler has many reasons to refuse. The arms that
have held up here carry a positive control with the same region shape and one instruction changed,
so a reject that came from a mis-encoded branch or an unrepresentable region cannot pass as the
contract being tested. Where the route matters, say so in the module: assert an opcode only one
emitter produces, or read `last_terminal_reject_reason()` (it needs a non-zero program address —
`record_terminal_reject_reason` early-returns on zero) and assert on the tag.

Register new cases in `prosper/CMakeLists.txt` under `if(TARGET prosper_core)`, which is
platform-independent — the file also has `if(WIN32)`/`if(UNIX)` branches, and a case registered in
the wrong one silently never runs while ctest stays green. Verify by name with `ctest -N`, never by
the total count.
