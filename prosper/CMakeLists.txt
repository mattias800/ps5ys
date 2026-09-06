cmake_minimum_required(VERSION 3.20)
project(prosper LANGUAGES C CXX)   # C for the vendored LibAtrac9 decoder (third_party/libatrac9)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Agentic-first: binaries must run with no PATH/DLL setup so CI (ctest) is hermetic.
# On mingw, statically link the GCC/C++ runtimes.
if(MINGW OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()

# --- Diagnostics (optional observer-only instrumentation) -----------------
# Boot timeline capture, event bus, JSON export. OFF by default: zero cost when
# disabled (stubs compile to no-ops). ON adds ~300 lines of optional code.
option(PROSPER_DIAGNOSTICS "Enable diagnostics instrumentation (boot timeline, event bus)" OFF)

# --- macOS Vulkan (MoltenVK) override ------------------------------------
# prosper on macOS is built x86_64 (to run the guest under Rosetta), but Homebrew's MoltenVK/loader
# are arm64-only, so find_package(Vulkan) would hand back an unlinkable arm64 lib. Point the build at
# a UNIVERSAL (x86_64+arm64) MoltenVK instead — either the LunarG Vulkan SDK, or the prebuilt
# KhronosGroup/MoltenVK release (`MoltenVK-macos.tar`, universal). Pass:
#   -DPROSPER_MACOS_MOLTENVK=/path/to/libMoltenVK.dylib   (universal)
#   -DPROSPER_MACOS_VULKAN_INCLUDE=/path/to/vulkan/headers (dir containing vulkan/vulkan.h)
# When set we pre-create the Vulkan::Vulkan imported target from it, so every later
# find_package(Vulkan) is skipped (guarded on `NOT TARGET Vulkan::Vulkan`). Headers are
# architecture-independent, so Homebrew's vulkan-headers work for the include dir.
if(APPLE AND PROSPER_MACOS_MOLTENVK)
  if(NOT PROSPER_MACOS_VULKAN_INCLUDE)
    set(PROSPER_MACOS_VULKAN_INCLUDE "/opt/homebrew/include")   # Homebrew vulkan-headers (arch-independent)
  endif()
  add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
  set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${PROSPER_MACOS_MOLTENVK}"
    INTERFACE_INCLUDE_DIRECTORIES "${PROSPER_MACOS_VULKAN_INCLUDE}")
  set(Vulkan_FOUND TRUE)
  set(Vulkan_LIBRARY "${PROSPER_MACOS_MOLTENVK}")
  set(Vulkan_INCLUDE_DIR "${PROSPER_MACOS_VULKAN_INCLUDE}")
  # The app dlopen's / links MoltenVK by leaf name; add its dir to the executables' rpath so
  # SDL_Vulkan_LoadLibrary("libMoltenVK.dylib") and the direct link both resolve at runtime.
  get_filename_component(_mvk_dir "${PROSPER_MACOS_MOLTENVK}" DIRECTORY)
  set(CMAKE_BUILD_RPATH "${_mvk_dir}")
  message(STATUS "macOS Vulkan: using universal MoltenVK ${PROSPER_MACOS_MOLTENVK}")
endif()

# --- tools ---------------------------------------------------------------
# self_dump: SELF/ELF inspector and NID import extractor (M0).
add_executable(self_dump tools/self_dump/self_dump.cpp)

# index_fetch_probe: bare-Vulkan oracle for indexed-draw index fetch (#2961). The #2961 hypothesis
# (RADV/STRIX_HALO reading sequential ordinals) was falsified with this tool; it stays so any
# future suspicion costs one command. Standalone on purpose — no prosper code, so it cannot
if(NOT TARGET Vulkan::Vulkan)
  find_package(Vulkan QUIET)
endif()
if(TARGET Vulkan::Vulkan)
  add_executable(index_fetch_probe tools/index_fetch_probe/probe.c)
  target_link_libraries(index_fetch_probe PRIVATE Vulkan::Vulkan)
endif()

# --- third-party: LibAtrac9 (MIT, vendored) - ATRAC9 audio decode -------------------------
# Vendored verbatim (see third_party/libatrac9/README.md). C99; needs libm. Compiled as its own
# static lib so the C sources stay isolated from prosper's C++ and are trivially auditable/updatable.
file(GLOB LIBATRAC9_SRC CONFIGURE_DEPENDS third_party/libatrac9/src/*.c)
add_library(libatrac9 STATIC ${LIBATRAC9_SRC})
target_include_directories(libatrac9 PUBLIC third_party/libatrac9/src)
set_target_properties(libatrac9 PROPERTIES C_STANDARD 99)
if(NOT WIN32)
  target_link_libraries(libatrac9 PUBLIC m)
endif()

# --- core library (M1+) --------------------------------------------------
# Loader / HLE / kernel / gpu sources accrete here per the roadmap.
include(cmake/ProsperBuildRevision.cmake)
prosper_add_build_revision_library(prosper_build_revision WORK_TREE "${CMAKE_SOURCE_DIR}/..")

file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)
# Exclude the diagnostics HISTORY/REPORT layer when the feature is disabled (default). When enabled,
# the PROSPER_DIAGNOSTICS define activates the real implementation; the headers always compile
# (stubs when disabled).
#
# Scoped to core/ and storage/ deliberately: src/diagnostics/boot_phase_log.cpp must be in EVERY
# build. It is the one file here that answers "where did boot_program() stop?", and boot_program()
# runs guest code (run_guest_inits), so a title can sit inside it indefinitely while every frontend
# prints nothing. An instrument for that has to exist in the default build — excluding it with the
# rest of the folder is how the boot phases came to be unprintable in every build the project
# shipped. See src/diagnostics/AGENTS.md.
if(NOT PROSPER_DIAGNOSTICS)
  list(FILTER PROSPER_SRC EXCLUDE REGEX "src/diagnostics/(core|storage)/")
endif()
if(PROSPER_SRC)
  add_library(prosper_core STATIC ${PROSPER_SRC})
  target_include_directories(prosper_core PUBLIC src)
  # stb_truetype.h backs the libSceFont Ft (FreeType-edition) rasterizer in
  # src/hle/util/hle_font.cpp. PRIVATE: it is an implementation detail of one translation unit, not
  # part of prosper_core's interface. See third_party/stb/README.md -- unlike stb_image.h next to
  # it, THIS header is on the guest path.
  target_include_directories(prosper_core PRIVATE third_party/stb)
  target_link_libraries(prosper_core PUBLIC libatrac9 PRIVATE prosper_build_revision)
  target_compile_definitions(prosper_core PRIVATE PROSPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(PROSPER_DIAGNOSTICS)
    target_compile_definitions(prosper_core PRIVATE PROSPER_DIAGNOSTICS)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(prosper_core PUBLIC Threads::Threads)
  # Windows: the guest futex HLE (sceKernelWaitOnAddress/WakeByAddress) is backed by the native Win32
  # WaitOnAddress/WakeByAddressSingle/All, which live in the Synchronization import lib (not kernel32).
  if(WIN32)
    target_link_libraries(prosper_core PUBLIC synchronization bcrypt)
  endif()
endif()

# Bounded interactive performance capture (F8): pure host-side state + serialization shared by the
# SDL app and the live graphics/compute backends. It deliberately has no SDL/Vulkan dependency, so
# its ring, bounds, and atomic file contract run in ordinary headless CI.
add_library(prosper_performance_capture STATIC frontends/shared/perf/performance_capture.cpp)
target_include_directories(prosper_performance_capture PUBLIC frontends)
target_link_libraries(prosper_performance_capture PRIVATE prosper_build_revision)

# --- tests (agentic-first: self-checking, exit-code = truth) -------------
enable_testing()

# Pure selection/accounting policy behind the stable compute-timing hash filter. This must remain
# runnable without prosper_core, a game dump, or Vulkan so a broken diagnostic cannot hide behind a
# skipped execution test on a host without graphics support.
add_executable(test_compute_timing_selector tests/shared/compute/test_compute_timing_selector.cpp)
target_include_directories(test_compute_timing_selector PRIVATE frontends)
add_test(NAME compute_timing_selector_policy COMMAND test_compute_timing_selector)

# Pure fail-closed policy for a behavior-neutral census of hypothetical deferred compute results.
# It has no Vulkan dependency and does not change the live backend's synchronous writeback.
add_executable(test_compute_authority_census tests/shared/compute/test_compute_authority_census.cpp)
target_include_directories(test_compute_authority_census PRIVATE frontends)
add_test(NAME compute_authority_census_policy COMMAND test_compute_authority_census)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  # self_dump --import-slots reads DT_JMPREL/DT_RELA to answer "which GOT slot does this Sony import
  # land in" (#1625). The fixtures are PS5-shaped ELFs written byte by byte in the test, so they are
  # an independent statement of the relocation layout rather than anything self_dump produced, and
  # they run in CI with no game dump. Includes the zero-result arms: a tool that answers "nothing"
  # when it means "I did not parse anything" is the defect class of #2399.
  add_test(NAME self_dump_import_slots
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/self_dump/test_import_slots.py"
                   "$<TARGET_FILE:self_dump>")
  add_test(NAME re_xref_decoder
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_xref.py")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME profiling_access
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/profiling_access/test_access.py")
    set_tests_properties(profiling_access PROPERTIES TIMEOUT 30)
  endif()
  # The human-authored snapshot store. Runs with no dump and no emulator: every arm builds its own
  # BMPs. The arms are chosen around SILENT failure -- a signature that cannot separate two scenes
  # would pass every regression, and an accept() that moved the signature without the reference
  # image would show a misleading "expected" at exactly the moment someone is deciding whether to
  # trust it.
  add_test(NAME snapshot_snaps_tool
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/snapshot/test_snaps.py")
  # iostore_index turns a .ucas read offset into a package name, which is the only instrument that
  # separates "reached gameplay" from "sat on an animating menu" on an Unreal title. Its failure
  # mode is a plausible WRONG name rather than an error, so the self-test pins the two-step
  # physical->logical->chunk mapping on a fixture where the two offset spaces genuinely differ --
  # a naive "physical == logical" reading passes every other check and misnames most of a stream.
  add_test(NAME re_iostore_index
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/iostore_index.py"
                   "--self-test")
  # worktree_reclaim deletes directories, so its tests are mutation arms first: each builds a tree
  # that MUST NOT be removed (dirty, untracked-only, unmerged, held by a live process, recently
  # touched, locked, nested) and asserts both the refusal and which guard produced it. It also
  # pins the instrument bug found while writing it -- examining a worktree used to make it look
  # freshly touched, which reported all 278 real trees as busy (#1735).
  #
  # Linux only, and that is a statement about the tool rather than about the test. Its in-use
  # guard reads /proc, so on macOS and Windows it cannot prove nobody is inside a tree and fails
  # closed: it classifies, and removes nothing, there. Testing removal guards where removal never
  # happens would assert on a configuration nobody runs. The fail-closed behaviour itself is
  # covered by a platform-independent arm, so the property that protects those platforms is still
  # verified everywhere this suite runs.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME worktree_reclaim_tool
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/test_worktree_reclaim.py")
  endif()

  # lanekill SIGNALS processes, so its arms are refusals first: a process in another worktree, a
  # path that attributes to nothing, a lookalike directory sharing a name prefix, and an
  # end-to-end run where exactly one of two live processes dies.
  #
  # That end-to-end arm exists because the first version of this suite tested only ATTRIBUTION.
  # A reviewer mutated the tool six ways -- each turning it into a cross-lane killer -- and every
  # arm stayed green, because none of them reached the code that decides to send a signal. Each
  # of those mutations now reddens, verified by re-running them.
  #
  # Linux-only, and that is a statement about the tool: attribution reads /proc. The fail-closed
  # arm is Linux-only TOO, which is a deliberate narrowing -- it used to run everywhere by
  # installing a stub and asserting the stub's own return value, which said nothing about
  # lanekill and stayed green when the gate was deleted. It now runs main() against a live
  # process in its own worktree and asserts the refusal spared it, which needs a real process.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME lanekill_tool
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/test_lanekill.py")
  endif()
  # wt_stash parks uncommitted work in refs/worktree/, which git documents as per-worktree, because
  # refs/stash is NOT: every worktree of this repo shares one stack, so two lanes each obeying the
  # "work in your own worktree" rule still pop each other's entries (#3174, instrument trap 247).
  #
  # The first arm is a positive control built by hand from git primitives: two linked worktrees,
  # plain `git stash` in each, and an assertion that lane A's pop really does hand it lane B's
  # content. Every later arm claims the tool removes that failure, and none of them means anything
  # if the fixture cannot produce it -- so a failure there reports VOID rather than a red arm.
  #
  # Platform-independent deliberately. Unlike worktree_reclaim and lanekill above, nothing here
  # reads /proc: the whole subject is ref namespacing, which behaves the same everywhere git runs,
  # and the collision this pins is likeliest on the Windows/PowerShell route the charter also
  # prescribes.
  add_test(NAME wt_stash_tool
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/test_wt_stash.py")
  add_test(NAME re_edis_mapping
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_edis.py")
  # nid_gate_scan answers "does this title BRANCH on an imported function's return value?", and a
  # wrong answer there is a plausible-looking table rather than a visible failure. Three of its
  # cases are regressions for classifier bugs found only after a table had been published (#2002).
  add_test(NAME re_nid_gate_classifier
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_nid_gate_scan.py")
  # stub_nid_map inverts that resolution to answer "which import is the stub at address X?" while
  # reading a call chain. Both ways it can be wrong are silent: a dropped stub entry point makes a
  # real import report NOT A STUB, and a dropped fail-visible line makes "not an import" and "wrong
  # address" the same empty output (#2186).
  add_test(NAME re_stub_nid_map
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_stub_nid_map.py")
  # hle_handler_map answers "how many Sony entry points collapse onto ONE prosper handler?" — the
  # blind spot a runtime return-value histogram cannot see, because it keys on the handler. Every
  # way the parser can be wrong yields a plausible table rather than a failure, so the shapes, the
  # `#if` arms and the per-name (not per-site) counting are all pinned (#2070).
  add_test(NAME re_hle_handler_map
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_hle_handler_map.py")
  # ...and reconciled against the registry the compiled binary actually builds, so a registration
  # SHAPE the parser does not recognise fails loudly instead of quietly shrinking the census.
  add_test(NAME re_hle_handler_map_reconcile
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/re/test_hle_registry_reconcile.py"
                   "$<TARGET_FILE:hle_registry_dump>")
  # prx_to_elf.py writes an ELF header by hand, four bytes apart from a field that means something
  # else entirely — #2016 put the program-header count in the section-header-count slot, and the
  # result was a malformed image no consumer complained about until binutils refused the file. The
  # test reads the produced bytes at their gABI offsets; nothing else can see this class of bug.
  add_test(NAME il2cpp_prx_to_elf_header
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/il2cpp/test_prx_to_elf.py")
  # resolve.py's other half: the nearest-preceding lookup that turns an address into a method NAME.
  # Every way it can be wrong is silent -- a window off by one, the bisect tie rule, or a dropped
  # record filter each yield a plausible name rather than an error, and that name is what a reader
  # trusts to place a breakpoint. The window arms sit exactly ON the 0x8000 edge for that reason:
  # an arm a byte outside it passes whether the comparison is `<` or `<=`, which is how the first
  # version of this file reported a green boundary test it could not actually fail (#2526).
  add_test(NAME il2cpp_resolve_lookup
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/il2cpp/test_il2cpp_tools.py")
  # replay_determinism_report decides whether a renderer-determinism campaign may be read as a
  # NEGATIVE. #2945's failure rate drifts machine-wide over minutes, so a campaign that sits in a
  # quiet window and a campaign on a repaired renderer produce identical numbers -- which is why
  # the tool has three verdicts and the third (UNDECIDED: one hash, but the bare-Vulkan control
  # never failed either) is the one being pinned. Every arm here except one still passes with the
  # control check deleted, so that arm is a MUTATION: it defeats the check and requires the quiet
  # campaign to flip to DETERMINISTIC.
  #
  # Platform-independent, and that is a statement about the tool: this half touches no Vulkan, no
  # GPU and no capture. It asserts on the READING of a campaign, which is where both of this
  # issue's published misreadings happened.
  add_test(NAME replay_determinism_report
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/determinism/test_replay_determinism_report.py")
  # hle_calls decides, per run, whether a reader should TRUST an entire measurement -- a nine-state
  # verdict machine whose most consequential branch is VOID vs VIOLATED ("this run cannot say" vs
  # "distrust every value in it"). It had no test at all, because hle_calls_gdb.py does `import gdb`
  # at module scope and nothing in it was reachable outside a gdb process. #2053 lifted the verdict
  # into hle_calls_control.py, which the gdb script now forwards to, so this asserts on the code that
  # actually runs rather than on a copy.
  add_test(NAME hle_calls_positive_control
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/hle_calls/test_hle_calls.py")
  # The sibling above is pure; this one cannot be. #2075 was not a logic bug: `--values` recorded
  # NOTHING on prosper's own default build, because gdb's FinishBreakpoint.return_value comes from
  # the DWARF return type and `Release` carries no `-g`. Only a real gdb over a real no-debug-info
  # binary reproduces that, so this compiles test_values_fixture.cpp twice (-g0 and -g) and runs the
  # real driver over each. A missing gdb/compiler exits 77 -> reported as `(Skipped)` by name, which
  # is the one thing a silently-skipped end-to-end test cannot do.
  add_test(NAME hle_calls_value_capture
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/hle_calls/test_hle_calls_values.py")
  set_tests_properties(hle_calls_value_capture PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  add_test(NAME snapshot_guard_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/snapshot/test_snapshot.py")
  add_test(NAME gpu_replay_regress_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_regress.py")
  # Half its cases are CORRECT shapes an earlier revision of the checker wrongly rejected. A doc
  # check that fires on valid input gets deleted rather than heeded, so the false-positive cases
  # are what keep it alive.
  add_test(NAME doc_table_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_numbered_table.py")
  # Every `trap NNN` reference in the repository must name a row that exists. Half this suite is
  # FALSE-POSITIVE cases (`s_trap 1`, a bare number, "trapdoor"), because this checker scans every
  # tracked file and one that fires on ordinary prose or on an RDNA2 mnemonic gets disabled.
  add_test(NAME trap_citation_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_trap_citations.py")
  # The trap-number allocator (#1729). Its network half is deliberately untested — a test needing
  # authentication gets disabled — but its LAST section pins its table parser against
  # check_numbered_table's, on the repository's real table. Those are two independent parsers, and a
  # divergence would hand out a number the gate then rejects, on somebody else's PR, hours later.
  add_test(NAME trap_number
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/docs/test_trap_number.py")
  # The PROGRESS_TRACKER.md generator's parser (#2730). Only the --selftest arm is registered, not
  # the live regeneration: reading the trackers needs `gh` and authentication, and a test that
  # needs credentials gets disabled -- CI runs the live --check itself, in the Docs job. What is
  # registered is the half that can fail SILENTLY: the parser accepts a compliant tracker and
  # rejects seven hand-written violations, so a regex that quietly widened (and would therefore
  # emit a plausible WRONG file rather than an error) fails here instead of shipping.
  add_test(NAME progress_tracker_generator
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/gen_progress_tracker.py" --selftest)
  # The pre-merge merge-result check (#2211). Its cases build real throwaway git repositories and
  # run real merges: the git plumbing IS the tool, so a mocked merge would test nothing. The case
  # that earns its keep is the clean merge whose RESULT holds a duplicate -- the state no head-time
  # check can observe, because at head time the other branch's row did not exist.
  add_test(NAME doc_merge_result_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_merge_result.py")
  # Parsing/gating logic of the Vulkan validation guard (#1704). Pure — it does not create a Vulkan
  # instance, so it runs everywhere. A parser that stops matching the layer's message framing would
  # report a clean suite forever, which is the one way this guard can fail silently.
  add_test(NAME vkval_scan_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/vkval/test_vk_validation_scan.py")
  # ctest EXITS 0 when it finds no tests, so "nothing ran" and "everything passed" share a status
  # (#2187). Four CI jobs and both author-verification runs were reading that status. They now pass
  # --no-tests=error; this registers the checker that keeps the next one from omitting it. The
  # --selftest arm is what is registered rather than the repository scan, because a scan needs the
  # checkout root and this test must run from any build directory -- CI runs the scan itself, in
  # the Docs job. Its own failure mode is silence: a matcher that stops matching reports a clean
  # tree forever, which is why the selftest carries the prose lines an earlier revision got wrong.
  add_test(NAME ctest_gate_checker
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_ctest_gate.py"
                   --selftest)
  # `print(__doc__)` is a usage message only while the header is a docstring; rewritten as `#`
  # comments it prints the literal string `None` and the tool still exits non-zero, so nothing
  # looks broken and the caller is told nothing (#2399). Three tools have shipped that way, the
  # last of them twice -- resolve.py regressed in #2642 inside the very change that added a line
  # to its usage block, with nine checks green, because the only guard in the tree named xref.py.
  # So this asks the whole tree instead of one file. Unlike ctest_gate_checker above, the scan
  # itself is registered rather than only the selftest: the root it needs is ${CMAKE_SOURCE_DIR},
  # known at configure time, so it runs from any build directory. --selftest runs beside it
  # because a walk that matches nothing and a clean tree otherwise print the same thing.
  add_test(NAME tools_usage_text
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_usage_text.py"
                   --selftest --root "${CMAKE_SOURCE_DIR}")
  # The merge gate (#3259). Its arms are all REFUSALS, and a gate that refuses everything would
  # satisfy every one of them -- so the suite leads with an all-pass positive control, and each
  # refusal asserts on the REASON rather than on the boolean. One arm reconstructs a merge that
  # actually went wrong: a red `Windows MinGW` read as green because a shell split the check NAME
  # on whitespace and landed on `MinGW` where the status belongs (#3234). The stale-head arm is
  # NOT a reconstruction -- #3259 first cited #3243 as one and that was wrong on the dates -- it
  # guards a state this repository produces routinely, since every lane pushes after CI starts.
  add_test(NAME pr_merge_gate
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_pr_merge_gate.py")
  # PROSPER_ENV_ON/_VALUE sample getenv once and return it forever. Applied to a name a TEST arms at
  # runtime, the arm becomes invisible -- and the test does not fail, it goes vacuous and keeps
  # printing [ok], which is why #2214 surfaced only as one unrelated-looking Vulkan binding error on
  # a single CI job. The scanner carries its own self-tests because its failure mode is silence: a
  # pattern that stops matching reports a clean tree forever. Three of those cases cover the wrapper
  # call forms (set_env/unset_env), which the first revision missed -- it passed a tree that still
  # contained the defect.
  add_test(NAME cached_env_arming_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_cached_env.py"
                   "${CMAKE_SOURCE_DIR}")
  # The coverage half of test_env_numeric_sites (#3267). That test MIRRORS each converted site
  # rather than calling it -- the sites cache their reads in function-local statics, so a runtime
  # arm would go vacuous exactly as cached_env_arming_logic above describes -- and a mirror drifts
  # silently: convert a site, forget the arm, suite stays green. This refuses that by comparing the
  # set of knobs parsed through diagnostics/env_numeric.hpp against the set the test arms. It is a
  # text scan, so it needs nothing built.
  add_test(NAME env_numeric_arms
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_env_numeric_arms.py"
                   "${CMAKE_SOURCE_DIR}")
  # stdout carries BYTES, and Windows decodes them as cp1252/cp437. One em dash in a printf literal
  # therefore reaches a Windows user as three mojibake characters, so a grep for the line the tool
  # itself printed does not match it -- and a test asserting on the rendered string fails on
  # Windows only, which is how
  # #2579 shipped five of them past a green Linux CI (#2588). Also runs in the Docs job, which
  # needs nothing built: this is a text scan, and a defect it can see should not wait for a build.
  # The scanner self-tests its own parsing AND its line numbers, because both failure modes are
  # silent -- a pattern that stops matching reports a clean tree forever, and a drifted line counter
  # sends the reader to innocent code while the real site goes unexamined (the first revision did).
  add_test(NAME ascii_output_literals
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/output/check_ascii_output.py"
                   "${CMAKE_SOURCE_DIR}")
  # A diagnostic whose PRODUCER is armed by a different switch than its PRINTER reports unarmed
  # state as a confident, well-formed ZERO -- indistinguishable from a measured negative, and
  # exactly what the interesting hypothesis usually predicts (#2149; #2111 emitted 183 such
  # negatives). Two arms: --selftest pins the three signatures against reduced forms of the four
  # MEASURED instances and runs from any build directory; the repository scan compares the tree
  # against tools/env/diag_gate_baseline.txt and runs in the Docs CI job, because it needs the
  # checkout root. Half the selftest cases assert on what must NOT match -- a checker that fires
  # on sound code gets regenerated past, not heeded -- and its own failure mode is silence: a
  # signature that stops matching reports a clean tree forever.
  add_test(NAME diag_gate_selftest
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_diag_gates.py"
                   --selftest)
  # The same selftest with $TMPDIR pointed INSIDE a build tree, which is where this repository's
  # own guidance puts it (`prosper/build-linux`). The selftest writes its fixtures into $TMPDIR, and
  # collect_files used to exclude by substring on the ABSOLUTE path, so that TMPDIR discarded every
  # fixture: twelve positive arms printed `got=[]` under the banner "the scanner's own signatures
  # are broken" while the twelve must-not-match arms passed VACUOUSLY. Three lanes hit it inside
  # five minutes and one read it as a real regression (#2658, #2660, #2661).
  #
  # --require-hostile-tmpdir is what keeps this case honest. `tempfile` falls back to /tmp without
  # a word when $TMPDIR names a directory that does not exist, so a case that merely SETS the
  # variable would go green on an environment it never had -- an arm that cannot show its lever
  # moved. The flag asserts the temp root really is a build-tree path before any arm runs.
  set(PROSPER_DIAG_GATE_HOSTILE_TMPDIR "${CMAKE_CURRENT_BINARY_DIR}/build-hostile-tmpdir")
  file(MAKE_DIRECTORY "${PROSPER_DIAG_GATE_HOSTILE_TMPDIR}")
  add_test(NAME diag_gate_selftest_hostile_tmpdir
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_diag_gates.py"
                   --selftest --require-hostile-tmpdir)
  set_tests_properties(diag_gate_selftest_hostile_tmpdir PROPERTIES
                       ENVIRONMENT "TMPDIR=${PROSPER_DIAG_GATE_HOSTILE_TMPDIR}")
  # Most cases here assert on what must NOT appear (negative time, failed dispatches summed in, an
  # image share computed against a filtered denominator, a per-binding mean over aliased folds),
  # because each is a table this tool has already produced once. Note what this gate does NOT do: it
  # feeds the tool SYNTHETIC records, so it pins the tool against its own model and can never notice
  # that model drifting from live_compute.cpp. Only the tool's runtime broken_sum/broken_nest warning
  # detects real drift, and only when someone reads stderr.
  add_test(NAME compute_phase_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_compute_phase_report.py")
  # The audio-delivery and flip-pacing reports, recovered from #2984 (#3033). Registered rather
  # than left as scripts with a --self-test nobody runs: both shipped with working self-tests in
  # that PR and neither was wired into ctest, which is a test that cannot go red. Same synthetic-
  # record caveat as compute_phase_report_logic above — these pin each tool against its own model
  # of the log format, so they catch a parser regression and cannot notice the EMITTER drifting.
  add_test(NAME audio_delivery_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_audio_delivery_report.py")
  add_test(NAME flip_pacing_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_flip_pacing_report.py")
  # The queue-depth timeline reader (#3033). Registered for a reason specific to this one: its
  # verdict is a DISCRIMINATOR that a pacer A/B rests on, and its first version could not tell a
  # port that was starved from a port that was merely silent -- both read queued=0. It reported a
  # 600 s Blasphemous 2 run as 53% dry before that gate existed. The arms pin all three conditions
  # of the gate, each of which was found untested by mutating it away.
  add_test(NAME audio_queue_timeline_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_audio_queue_timeline_report.py")
  # The shader-conformance scanner (#3036), registered because its whole value is a CLEAN result
  # and its original failure mode was manufacturing one: a capture that yielded zero examined
  # shaders exited 0, and findings could be attributed to a capture whose dumps had all failed.
  # It drives the scanner end to end against a stub gpu_replay and a `spirv-dis` shim, so it needs
  # neither a GPU nor the Vulkan toolchain, and it pins the exit contract (0 clean / 1 mismatch /
  # 2 could-not-scan) rather than only the two parsers. Ten mutations of the logic it covers,
  # including `DIM_ARRAYED` dropping the 2D_ARRAY case that is the tool's entire reason to exist,
  # each turn it red.
  #
  # It needs the BUILT `shader_inspect` since #3184: the scanner's MIMG walk is now prosper's own
  # RDNA2 decoder rather than a Python port of its length rules, so the decode arms exercise
  # `rdna2_decode.cpp` itself and a change to its length dispatch reddens this suite. Guarded on
  # `TARGET prosper_core` because that is what governs the executable; the generator expression
  # resolves against the target added later in this file. Without the binary the suite exits 2
  # (could not test) rather than reporting a pass.
  if(TARGET prosper_core)
    add_test(NAME shader_conformance_scan_logic
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/shader_conformance/scan.py" "--self-test"
                     "--mimg-decoder" "$<TARGET_FILE:shader_inspect>")
  endif()
  # stack_profile's frame classifier, against RECORDED gdb output. Deliberately needs no gdb and no
  # ptrace: an end-to-end variant would skip silently wherever ptrace is unavailable, and a silently
  # skipped test is indistinguishable from a passing one. What it pins is the case that already
  # produced a wrong answer once — a versioned glibc symbol (`pthread_cond_wait@@GLIBC_2.3.2`) that
  # defeated the `$`-anchored plumbing patterns, so a libc internal was named as the application's
  # blocking site. Reverting only the version-stripping fix fails four of these cases.
  add_test(NAME stack_profile_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_stack_profile.py")
  if(GIT_EXECUTABLE)
    # The provenance GATE: refuses to quote a measurement from a build that is not the revision the
    # author thinks it is. Pure argument/exit-status contract, no build required — but it resolves
    # real refs and creates a scratch repository, so it needs git exactly as its sibling below does.
    add_test(NAME check_build_revision
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/revision/test_check_build_revision.py")
    add_test(NAME build_revision_refresh
             COMMAND "${Python3_EXECUTABLE}"
               "${CMAKE_SOURCE_DIR}/tools/revision/test_build_revision.py"
               --cmake "${CMAKE_COMMAND}"
               --git "${GIT_EXECUTABLE}"
               --module "${CMAKE_SOURCE_DIR}/cmake/ProsperBuildRevision.cmake"
               --scratch-root "${CMAKE_CURRENT_BINARY_DIR}/build-revision-regression"
               --generator "${CMAKE_GENERATOR}")
  endif()
endif()
# Path to the (gitignored) local game dump; override with -DGAME_DUMP=...
#
# The relative fallback resolves to `<source>/../PPSA24651-app0`, which inside a git worktree is a
# directory no dump lives in — that is instrument trap 126, where a build silently runs a SMALLER
# suite and still prints "100% tests passed" with exit 0, because the percentage is of what ran.
# Preferring $PROSPER_GAME_ROOT makes the common case resolve on its own without putting anybody's
# local directory layout in this file. `-DGAME_DUMP=` still wins: a CACHE entry that already exists
# is not overwritten by set(... CACHE ...).
if(DEFINED ENV{PROSPER_GAME_ROOT})
  set(GAME_DUMP "$ENV{PROSPER_GAME_ROOT}/PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
else()
  set(GAME_DUMP "${CMAKE_SOURCE_DIR}/../PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
endif()
set(HAVE_GAME_DUMP OFF)
if(EXISTS "${GAME_DUMP}/eboot.bin")
  set(HAVE_GAME_DUMP ON)
else()
  # Loud, and says how to fix it: this message is the ONLY announcement, it happens once at configure
  # time, and every later build inherits the reduced suite in silence.
  message(WARNING "Game dump not found at ${GAME_DUMP} — dump-backed tests will be SKIPPED, so "
                  "ctest will report a smaller count and still exit 0. Set PROSPER_GAME_ROOT or "
                  "pass -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0 if you did not intend this.")
endif()

# Which title `GAME_DUMP` actually points at (#1573). A handful of ctest cases assert against ONE
# title's bytes — import counts, an IL2CPP module layout, that title's embedded shader blobs — so
# pointing GAME_DUMP at a different game turns them red for a reason nothing in the ctest output
# names. A bring-up agent's first action on a new title is exactly that configure, so the red run is
# the first thing they see, and every plausible reading of it (master is broken / this dump is
# corrupt / bisect) is wrong.
#
# `sce_sys/param.json`'s `titleId` is authoritative; the `PPSAxxxxx-app0` directory name is a second,
# independent source for a dump laid out without `sce_sys/`. If NEITHER yields an id the title-gated
# cases are registered and RUN as before — a detection failure must stay fail-visible rather than
# converting three real guards into silent skips, which is the defect this whole area is about.
# CONFIDENCE: HIGH (both sources read directly from every dump on the dev box).
set(GAME_DUMP_TITLE_ID "")
if(HAVE_GAME_DUMP)
  if(EXISTS "${GAME_DUMP}/sce_sys/param.json")
    file(READ "${GAME_DUMP}/sce_sys/param.json" prosper_param_json)
    string(JSON prosper_param_title_id ERROR_VARIABLE prosper_param_error
           GET "${prosper_param_json}" "titleId")
    if(NOT prosper_param_error)
      set(GAME_DUMP_TITLE_ID "${prosper_param_title_id}")
    endif()
    unset(prosper_param_json)
  endif()
  if(GAME_DUMP_TITLE_ID STREQUAL "")
    get_filename_component(prosper_dump_dir_name "${GAME_DUMP}" NAME)
    if(prosper_dump_dir_name MATCHES "^([A-Z]+[0-9]+)")
      set(GAME_DUMP_TITLE_ID "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(GAME_DUMP_TITLE_ID STREQUAL "")
    message(STATUS "Game dump title id could not be determined from ${GAME_DUMP} — "
                   "title-specific tests will run against it")
  else()
    message(STATUS "Game dump title id: ${GAME_DUMP_TITLE_ID}")
  endif()
endif()

# Register `name` as a ctest case that is SKIPPED with `reason`, rather than not registered at all.
# `cmake -E echo` prints the reason and exits 0; SKIP_REGULAR_EXPRESSION is what turns that into a
# `(Skipped)` result. The point is that the case still occupies a line in the ctest summary: a test
# that silently disappears from the registered set costs coverage with no line to read and no count
# to compare against (the failure mode `docs/VERIFICATION.md` § CI records for the Vulkan flag).
function(prosper_add_skipped_test name reason)
  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: ${reason}")
  set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
endfunction()

# Register a ctest case that asserts against ONE title's content. Runs normally when GAME_DUMP is
# that title (or when no title id could be derived); otherwise registers a visible skip naming both
# the required title and the configured one, so the reason is in the run instead of in a doc.
function(prosper_add_title_gated_test name title)
  if(GAME_DUMP_TITLE_ID STREQUAL "${title}" OR GAME_DUMP_TITLE_ID STREQUAL "")
    add_test(NAME ${name} COMMAND ${ARGN})
  else()
    prosper_add_skipped_test(${name}
      "${name} asserts against ${title}'s own content; GAME_DUMP is configured with \
${GAME_DUMP_TITLE_ID}. Re-run cmake with -DGAME_DUMP=<dump root>/${title}-app0 to exercise it.")
    # Also say it here. ctest prints `(Skipped)` but not the reason unless it is run with `-V`, and
    # configure is the moment the reader chose the dump — so this is where the explanation lands in
    # front of them without their having to know to ask for it.
    message(STATUS
      "ctest case '${name}' will be SKIPPED: it asserts against ${title}'s own content, "
      "GAME_DUMP is ${GAME_DUMP_TITLE_ID}")
  endif()
endfunction()

if(TARGET prosper_core)
  # nid_census: every guest import that falls to the dispatcher's return-0 default (#2081).
  # Links prosper_core because both of its inputs must be the real ones — `Module::load` for the
  # imports the loader will actually try to bind, `module_export_nids` for what a sibling module
  # already defines, and the live `Hle` registry for whether a handler exists. A source grep over
  # register_fn() call sites cannot see any of the three.
  add_executable(nid_census tools/nid_census/nid_census.cpp)
  target_link_libraries(nid_census PRIVATE prosper_core)

  add_executable(test_module tests/test_module.cpp)
  target_link_libraries(test_module PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    # Messenger-specific: the assertions pin PPSA24651's import/segment/relocation counts (#1573).
    prosper_add_title_gated_test(module_loads_eboot PPSA24651
             test_module "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_nid tests/hle/dispatch/test_nid.cpp)
  target_link_libraries(test_nid PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME nid_hash_matches_imports
             COMMAND test_nid "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_hle_registered tests/hle/test_hle_registered.cpp)
  target_link_libraries(test_hle_registered PRIVATE prosper_core)
  add_test(NAME hle_functions_registered COMMAND test_hle_registered)

  # No NID registered twice with a DIFFERENT handler (the #330 shadow class): a later register_*()
  # silently overwriting a real impl with a stub returns garbage to the guest. Pure, no game dump.
  add_executable(test_hle_no_shadow tests/hle/test_hle_no_shadow.cpp)
  target_link_libraries(test_hle_no_shadow PRIVATE prosper_core)
  add_test(NAME hle_no_shadow COMMAND test_hle_no_shadow)

  # Runtime IL2CPP symbolication (#2551): name the C# method containing a live guest PC. The
  # fixture is committed and synthetic — no game dump, no Il2CppDumper — so this never degrades
  # into a silent skip. The binary doubles as the probe the agreement test drives.
  add_executable(test_il2cpp_symbols tests/host/symbols/test_il2cpp_symbols.cpp)
  target_link_libraries(test_il2cpp_symbols PRIVATE prosper_core)
  add_test(NAME il2cpp_symbol_resolution
           COMMAND test_il2cpp_symbols
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/il2cpp_symtab_fixture.symtab")
  if(Python3_Interpreter_FOUND)
    # ...and the runtime resolver must answer exactly what tools/il2cpp/resolve.py answers. They
    # share the record set by construction (resolve.py emits the table the runtime reads), but the
    # nearest-preceding rule, its window and its tie handling are implemented twice, so this drives
    # both CLIs over one synthesized corpus and requires them to agree address-for-address.
    add_test(NAME il2cpp_symtab_agreement
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/il2cpp/test_symtab_agreement.py"
                     "$<TARGET_FILE:test_il2cpp_symbols>")
  else()
    prosper_add_skipped_test(il2cpp_symtab_agreement
      "il2cpp_symtab_agreement drives tools/il2cpp/resolve.py; no Python 3 interpreter was found.")
  endif()

  # Ground truth for tools/re/hle_handler_map.py: the registry register_builtin_hle() really builds
  # (#2070). Pure setup — no game dump, no GPU — so the static parser can be reconciled against the
  # compiled binary on every run instead of being trusted.
  add_executable(hle_registry_dump tools/re/hle_registry_dump.cpp)
  target_link_libraries(hle_registry_dump PRIVATE prosper_core)

  # Cooperative stop signal (issue #164 P0): request_stop/stop_requested the frontend uses to wind
  # the guest run-loop down on window-close. Pure, no deps.
  add_executable(test_lifecycle tests/host/platform/test_lifecycle.cpp)
  target_link_libraries(test_lifecycle PRIVATE prosper_core)
  add_test(NAME lifecycle_stop_signal COMMAND test_lifecycle)

  # GPU submit gate (#3225): the counted region + bounded drain prosper-app runs before std::_Exit,
  # so a detached guest thread cannot be killed inside an amdgpu submission (which parks the task in
  # __drm_exec_lock_obj and takes the host compositor down with it). Pure — no Vulkan, no dump — so
  # the admission/refusal/drain-timeout contract is asserted on every host, including CI without a
  # GPU. Its own timing assertions are one-sided (a lower bound on a timeout, an upper bound far
  # from the deadline), so it does not go red under host load.
  add_executable(test_gpu_submit_gate tests/host/platform/test_gpu_submit_gate.cpp)
  target_link_libraries(test_gpu_submit_gate PRIVATE prosper_core)
  add_test(NAME gpu_submit_gate COMMAND test_gpu_submit_gate)

  # Screenshot evidence classification/JSON is pure and must remain testable without Vulkan or a dump.
  add_executable(test_capture_manifest tests/test_capture_manifest.cpp
                                       tools/screenshot/capture_manifest.cpp
                                       # The framerate the manifest reports. A pure TU with no
                                       # dependency beyond the standard library, so it links here
                                       # without dragging prosper_core (or Vulkan) into the test.
                                       src/gpu/present/present_frame_rate.cpp)
  # `src` and the revision library are for `build_revision.hpp` only: the run header records the
  # revision the binary was compiled from, so the manifest self-describes. Still no Vulkan, no dump.
  target_include_directories(test_capture_manifest PRIVATE tools/screenshot src)
  target_link_libraries(test_capture_manifest PRIVATE prosper_build_revision)
  add_test(NAME capture_manifest COMMAND test_capture_manifest)

  # The burned-in fps annotation (`screenshot --fps-overlay`). Rasterizing text into an RGBA buffer
  # is pure, and an overlay that silently draws nothing looks exactly like one that was never asked
  # for -- so it is asserted on pixels, here, without a device.
  add_executable(test_overlay_text tests/test_overlay_text.cpp
                                   tools/screenshot/overlay_text.cpp)
  target_include_directories(test_overlay_text PRIVATE tools/screenshot)
  add_test(NAME overlay_text COMMAND test_overlay_text)

  # va2foff resolution order (issue #113): DYNAMIC/PROCPARAM/TLS must never shadow
  # the PT_LOAD that truly maps a VA. Pure unit test — no game dump required.
  add_executable(test_va2foff tests/test_va2foff.cpp)
  target_link_libraries(test_va2foff PRIVATE prosper_core)
  add_test(NAME va2foff_load_priority COMMAND test_va2foff)

  # SELF/ELF/loader parsing must handle a malformed/truncated dump without an out-of-bounds read:
  # overflow-safe rd() bounds (self_read_ok), NUL-terminated str_at, and the linker init-array p+8
  # guard. Pure, no game dump.
  add_executable(test_self_bounds tests/test_self_bounds.cpp)
  target_link_libraries(test_self_bounds PRIVATE prosper_core)
  add_test(NAME self_bounds COMMAND test_self_bounds)

  # SELF data-segment key masks flag bits >= 32 (#339): the SCE segment id is bits [31:20]; without
  # the mask a data segment keys the map wrong and every phdr lookup misses. Pure, no game dump.
  add_executable(test_self_segment_key tests/test_self_segment_key.cpp)
  target_link_libraries(test_self_segment_key PRIVATE prosper_core)
  add_test(NAME self_segment_key COMMAND test_self_segment_key)

  # sceKernelNanosleep's GUEST timespec contract (#3013). Registered OUTSIDE the if(UNIX) block
  # below ON PURPOSE: what it pins is a WINDOWS defect -- MinGW-w64 declares a 32-bit `long tv_nsec`
  # while the guest timespec is FreeBSD x86-64's 64-bit field, so writing the remainder through a
  # host struct covered 12 of 16 bytes -- and a Unix-only registration would build it everywhere
  # except the platform it is about. The second property, that a malformed tv_nsec returns promptly
  # rather than sleeping ~2 s or ~292 years, is platform-independent. Asserts promptness, not
  # accuracy: sleep precision belongs to host::sleep_until_steady_ns and a wall-clock assertion on
  # it would flake on a loaded host.
  add_executable(test_kernel_nanosleep tests/hle/kernel/test_kernel_nanosleep.cpp)
  target_link_libraries(test_kernel_nanosleep PRIVATE prosper_core)
  add_test(NAME kernel_nanosleep COMMAND test_kernel_nanosleep)

  # The guest semaphore timed wait (#3013). Platform-neutral like the nanosleep case above and for
  # the same reason: the fix is Windows-only, so gating on UNIX would build the test everywhere
  # except where the defect is.
  #
  # It asserts the MECHANISM -- which primitive served the wait, via sleep_backend_name() -- rather
  # than a wall-clock duration, following test_videoout.cpp and what test_kernel_nanosleep.cpp's
  # header asks for. The first version asserted durations only and review showed NONE of its arms
  # could redden on the pre-fix code: its 40 ms ceiling sat 2.6x above the 15.57 ms defect, and its
  # other two arms measured a POSTED semaphore, which unfixed winpthreads serves in 0.01 ms -- better
  # than the poll loop, so they preferred the defect. A mechanism assertion cannot be satisfied by a
  # lucky duration and does not flake under load. Mutation-checked: reverting to the winpthreads
  # delegation reddens the MECHANISM arm and prints "via none".
  #
  # It used to say "reddens two arms", the second being a wall-clock ceiling, and that ceiling is
  # gone: it would have PASSED the defect, and it also flaked on macOS/Rosetta. The mechanism
  # arm is the discriminator; the bounds that remain are a stub/unit guard.
  #
  # The measurements and the argument live in ONE place, the test file's own header. They were
  # briefly in both, with two different lists -- and a third Rosetta figure existed in neither
  # (23.22 ms, found in a CI log during review). Two copies of a figure list is one copy too
  # many, and this is the copy that gets read second.
  add_executable(test_kernel_sem_timedwait tests/hle/kernel/test_kernel_sem_timedwait.cpp)
  target_link_libraries(test_kernel_sem_timedwait PRIVATE prosper_core)
  add_test(NAME kernel_sem_timedwait COMMAND test_kernel_sem_timedwait)

  # scePthreadCondTimedwait resolves the CONDVAR'S clock for its relative microsecond timeout
  # (#3056), which its POSIX sibling already did and this spelling did not. CROSS-PLATFORM, and
  # deliberately so: the defect is in the shared HLE body, not in either platform's wait primitive,
  # and the discriminator (SCE_KERNEL_CLOCK_VIRTUAL is process CPU time, so an idle wait cannot
  # reach its deadline) reads a RETURN VALUE rather than a duration. See the test's own header for
  # why realtime and monotonic clocks cannot discriminate anything here.
  add_executable(test_kernel_cond_timedwait_clock tests/hle/kernel/test_kernel_cond_timedwait_clock.cpp)
  target_link_libraries(test_kernel_cond_timedwait_clock PRIVATE prosper_core)
  add_test(NAME kernel_cond_timedwait_clock COMMAND test_kernel_cond_timedwait_clock)
  # Bounded: the arm parks a worker on a condvar whose deadline an idle process cannot reach, and
  # relies on the main thread's signal to release it. A regression that loses the signal would
  # otherwise hang the suite rather than fail it.
  set_tests_properties(kernel_cond_timedwait_clock PROPERTIES TIMEOUT 60)
  # Bounded for the same reason as the nanosleep case: the fixed path returns in ~5 ms, so a
  # regression that reverts to a blocking wait on a semaphore nothing will ever post would hang
  # rather than fail, and CTest's default 1500 s hang reads as neither pass nor fail.
  set_tests_properties(kernel_sem_timedwait PROPERTIES TIMEOUT 60)

  # scePthreadSemInit must forward a real sem_init() failure instead of discarding it and
  # manufacturing success (#3068). Platform-neutral: it probes the host sem_init() with an
  # out-of-range value first and only asserts the guest-visible contract when the host call
  # actually failed, so it is meaningful on every CI platform (EINVAL on Linux, ENOSYS on Darwin)
  # without needing per-platform code, and skips itself explicitly rather than passing vacuously
  # if some future libc happens to accept the probe value.
  add_executable(test_kernel_sem_init_error tests/hle/kernel/test_kernel_sem_init_error.cpp)
  target_link_libraries(test_kernel_sem_init_error PRIVATE prosper_core)
  add_test(NAME kernel_sem_init_error COMMAND test_kernel_sem_init_error)
  # Bounded because arm 3 guards a ~292-YEAR sleep: without a timeout a regression there presents as
  # CTest hanging for its default 1500 s rather than as a failure, and a hang is the one outcome nobody
  # reads as a red test. Generous against a loaded host -- what it catches is seconds-to-forever.
  set_tests_properties(kernel_nanosleep PROPERTIES TIMEOUT 60)

  # LABEL: `timing-margin` -- the two tests above whose green verdict rests on a MEASURED wall-clock
  # margin, and which print that margin on stdout so it can be quoted.
  #
  # #3067 is why this label exists. A red `macOS (x86_64 / Rosetta 2)` job was recorded as
  # "unresolved, two candidates" while the failing job's own log already named the arm and the
  # number -- `[FAIL] and returned inside one winpthreads tick of the request, not after it` /
  # `(timeout took 13.58 ms via none)` against a 12 ms ceiling. The number was there because the
  # test prints it; it was only in the log because that run was RED. `--output-on-failure` discards
  # it from every green run, so the runs on either side of a near-miss carry no evidence at all and
  # the headroom cannot be checked without reddening the job first -- the #3208 defect (a threshold
  # whose margin nobody has measured recently) with the measurement one flag away.
  #
  # The CI jobs re-run this label with `-V` after the suite, so every green log carries the figures.
  # `kernel_nanosleep` is deliberately NOT labelled: its wall-clock bounds are 500 ms order-of-
  # magnitude guards against a ~292-year sleep and it prints no measured value, so a verbose re-run
  # would report nothing. Add the label when a test both DEPENDS ON and PRINTS a margin.
  set_tests_properties(kernel_sem_timedwait kernel_cond_timedwait_clock
                       PROPERTIES LABELS "timing-margin")

  # Guest static-TLS Variant II layout uses each module's real PT_TLS p_align (#143). The layout
  # code is Linux-only (guest %fs), so gate the test on it too.
  if(UNIX)
    add_executable(test_guest_tls_align tests/loader/test_guest_tls_align.cpp)
    target_link_libraries(test_guest_tls_align PRIVATE prosper_core)
    add_test(NAME guest_tls_align COMMAND test_guest_tls_align)
  endif()

  # Cross-module TLS resolution (#136/#338): dynamic and initial-exec relocations resolve through
  # the DEFINING module's id, symbol offset, and shared static layout. Pure synthetic unit test.
  add_executable(test_dtpmod_tls tests/test_dtpmod_tls.cpp)
  target_link_libraries(test_dtpmod_tls PRIVATE prosper_core)
  add_test(NAME dtpmod_tls_cross_module COMMAND test_dtpmod_tls)

  # AMD SSE4a INSERTQ/EXTRQ emulation (the SIGILL handler emulates these Zen2-only ops on Intel
  # hosts). Golden bit-math vectors — header-only, no game dump required.
  add_executable(test_sse4a tests/host/x86/test_sse4a.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_sse4a PRIVATE src)
  add_test(NAME sse4a_insertq_extrq COMMAND test_sse4a)

  # macOS/Rosetta %fs trap-emulation decode core (guest-TLS access redirection). Golden
  # instruction vectors incl. the #727 no-REX AH/CH/DH/BH byte-register rules — header-only,
  # runs on Linux where the __APPLE__ handler itself never compiles.
  add_executable(test_fs_emu tests/host/tls/test_fs_emu.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_fs_emu PRIVATE src)
  add_test(NAME fs_emu_decode COMMAND test_fs_emu)

  # Signal-handler-safe raw syscalls (src/host/platform/raw_syscall.hpp): the fault_handler's
  # PROSPER_NULL_PAGE backing must not touch errno/TLS on guest-%fs threads (issue #1071).
  if(UNIX AND NOT APPLE)
    add_executable(test_raw_syscall tests/host/platform/test_raw_syscall.cpp)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    target_include_directories(test_raw_syscall PRIVATE src)
    target_include_directories(test_raw_syscall PRIVATE tests)
    add_test(NAME raw_syscall COMMAND test_raw_syscall)
  endif()

  # sceAudioOut HLE: format decode, port lifecycle, PCM forwarding, volume, batch, error paths
  # (via a capturing AudioSink — headless, no device).
  add_executable(test_audio tests/hle/audio/test_audio.cpp)
  target_link_libraries(test_audio PRIVATE prosper_core)
  add_test(NAME audio_hle COMMAND test_audio)

  # NGS2 waveform-block streaming (#1059/#1060 part 3): a fed block makes the voice Playing and advances
  # num_decoded_samples across renders (the Dead Cells music stream clock). POSIX-gated (needs a fixed
  # guest-address PCM mapping); the lifecycle assertions run everywhere.
  add_executable(test_ngs2_streaming tests/hle/test_ngs2_streaming.cpp)
  target_link_libraries(test_ngs2_streaming PRIVATE prosper_core)
  add_test(NAME ngs2_streaming COMMAND test_ngs2_streaming)

  # libSceAjm (#187): headless audio-decode lifecycle — handle out-params filled with valid values
  # (context/instance/batch), inputs validated, batch reports complete. Pure, no game dump.
  add_executable(test_ajm tests/hle/test_ajm.cpp)
  target_link_libraries(test_ajm PRIVATE prosper_core)
  add_test(NAME ajm_hle COMMAND test_ajm)

  add_executable(test_avplayer tests/gpu/texture/test_avplayer.cpp)
  target_link_libraries(test_avplayer PRIVATE prosper_core)
  add_test(NAME avplayer_hle COMMAND test_avplayer)

  # #2270. Three ctest entries over ONE binary, because each arm's environment variable is read once
  # into a function-local static (deliberately — this is a per-frame path), so the arms cannot share
  # a process.
  add_executable(test_videodec2_decode tests/hle/video/test_videodec2_decode.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_videodec2_decode PRIVATE tests)
  target_link_libraries(test_videodec2_decode PRIVATE prosper_core)
  add_test(NAME videodec2_decode COMMAND test_videodec2_decode)
  add_test(NAME videodec2_decode_optout COMMAND test_videodec2_decode --no-decode)
  add_test(NAME videodec2_decode_format COMMAND test_videodec2_decode --format-probe)

  # #2898. The SPS/VUI parser and the picture-info fill are pure host-side code exercised
  # against hand-built SPS bitstreams written by an independent bit-writer in the test.
  # The fill layout asserts every offset the guest has been observed reading, and carries
  # a mutation arm (absent-meta fill over a poisoned buffer) so removing the fill cannot
  # pass vacuously.
  add_executable(test_h264_sps tests/hle/video/test_h264_sps.cpp)
  target_link_libraries(test_h264_sps PRIVATE prosper_core)
  add_test(NAME h264_sps COMMAND test_h264_sps)

  add_executable(test_json2 tests/hle/test_json2.cpp)
  target_link_libraries(test_json2 PRIVATE prosper_core)
  add_test(NAME json2_hle COMMAND test_json2)

  add_executable(test_equeue_events tests/hle/test_equeue_events.cpp)
  target_link_libraries(test_equeue_events PRIVATE prosper_core)
  add_test(NAME equeue_events COMMAND test_equeue_events)

  # APR completion delivery on a shared equeue (#1673): the constant-tag (CRI) dialect must not be
  # coalesced away, while the counter dialect (#208) and genuine level sources must still coalesce.
  add_executable(test_apr_equeue_completion tests/hle/test_apr_equeue_completion.cpp)
  target_link_libraries(test_apr_equeue_completion PRIVATE prosper_core)
  add_test(NAME apr_equeue_completion COMMAND test_apr_equeue_completion)

  # Variadic libc formatters (issue #122): real va_list capture so %f/XMM + stack + %s-past-4th
  # args format correctly (was a 3-integer-register forward). Cross-platform (host libc printf).
  add_executable(test_printf tests/hle/test_printf.cpp)
  target_link_libraries(test_printf PRIVATE prosper_core)
  add_test(NAME printf_varargs COMMAND test_printf)

  # Windows' aligned allocator family must remain paired end-to-end. Mixing _aligned_malloc with
  # plain free corrupts the heap under aligned game-buffer churn; ordinary and aligned guest
  # allocations intentionally share one compatible family on that host.
  add_executable(test_libc_alloc tests/hle/test_libc_alloc.cpp)
  target_link_libraries(test_libc_alloc PRIVATE prosper_core)
  target_include_directories(test_libc_alloc PRIVATE tests)
  add_test(NAME libc_alloc COMMAND test_libc_alloc)

  # EventFlag/Sema delete-under-waiter safety (issue #104): delete wakes parked waiters with EACCES
  # and defers the free to the last one out, instead of free()ing the object under a live wait (UAF).
  add_executable(test_sync_delete tests/hle/kernel/test_sync_delete.cpp)
  target_link_libraries(test_sync_delete PRIVATE prosper_core)
  add_test(NAME sync_delete COMMAND test_sync_delete)

  # Anchored wall-clock sources (#92): CLOCK_REALTIME ≈ host now, one shared epoch across
  # time()/gettimeofday/sceRtc*, FreeBSD clockid mapping, valid LocalTime calendar dates.
  add_executable(test_wall_clock tests/hle/test_wall_clock.cpp)
  target_link_libraries(test_wall_clock PRIVATE prosper_core)
  add_test(NAME wall_clock COMMAND test_wall_clock)

  # select/pselect pure-sleep shape honours its timeout; a descriptor-set query stays fail-visible
  # (#1660). Asserts elapsed time, not the return value -- the stub this replaces also returned 0.
  add_executable(test_select_sleep tests/hle/kernel/test_select_sleep.cpp)
  target_link_libraries(test_select_sleep PRIVATE prosper_core)
  add_test(NAME select_sleep COMMAND test_select_sleep)
  # This test drives real sleeps through the HLE, and its #3038 arms feed it timeouts a regression
  # would take LITERALLY: carrying an out-of-range tv_nsec into the total makes 0x7fff... a
  # 292-year wait, so the failure mode is a wedge rather than a red. Measured while mutation-testing
  # this PR -- with the range rule removed, test_select_sleep does not return. A bound turns that
  # into a legible timeout instead of ctest's 25-minute default. The whole run is ~0.5 s, so 60 s is
  # a hundredfold headroom and cannot flake under load.
  set_tests_properties(select_sleep PROPERTIES TIMEOUT 60)

  # Opt-in flip-paced monotonic clock (#240): exact frame steps, thread-safe first-flip anchor,
  # and no coupling into CLOCK_REALTIME while virtual time is paused between flips.
  add_executable(test_det_clock tests/hle/test_det_clock.cpp)
  target_link_libraries(test_det_clock PRIVATE prosper_core)
  add_test(NAME det_clock COMMAND test_det_clock)

  # Synchronous host-GPU work must not become a giant guest frame delta. The clock retains one
  # caller-supplied frame budget, stays monotonic under concurrent reads, and leaves realtime alone.
  add_executable(test_gpu_time_compensation tests/hle/kernel/test_gpu_time_compensation.cpp)
  target_link_libraries(test_gpu_time_compensation PRIVATE prosper_core)
  add_test(NAME gpu_time_compensation COMMAND test_gpu_time_compensation)

  # UserService/libkernel startup stubs (HLE audit): getters write deterministic defaults; the
  # libkernel registration/hook calls resolve to benign OK no-ops.
  add_executable(test_service_getters tests/hle/dispatch/test_service_getters.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_service_getters PRIVATE tests)
  target_link_libraries(test_service_getters PRIVATE prosper_core)
  add_test(NAME service_getters COMMAND test_service_getters)

  # #2185: reallocalign was unregistered, so the dispatcher's 0 reached the guest as a NULL
  # allocation — a false FAILURE (reads as OOM), the mirror of the #2081 family below. Asserts the
  # extended alignment survives a resize, which plain realloc never promised. Pure, no game dump.
  add_executable(test_reallocalign tests/hle/test_reallocalign.cpp)
  target_link_libraries(test_reallocalign PRIVATE prosper_core)
  add_test(NAME reallocalign COMMAND test_reallocalign)

  # #2168: a condvar destroy must refuse while a thread is parked inside it.
  add_executable(test_cond_destroy_busy tests/hle/test_cond_destroy_busy.cpp)
  target_link_libraries(test_cond_destroy_busy PRIVATE prosper_core)
  add_test(NAME cond_destroy_busy COMMAND test_cond_destroy_busy)

  # #2178: no Sony pthread spelling may return a bare FreeBSD errno.
  add_executable(test_sce_pthread_encoding tests/hle/test_sce_pthread_encoding.cpp)
  target_link_libraries(test_sce_pthread_encoding PRIVATE prosper_core)
  add_test(NAME sce_pthread_encoding COMMAND test_sce_pthread_encoding)

  # #2168 (barrier half): a barrier destroy must refuse while threads are parked.
  add_executable(test_barrier_destroy_busy tests/hle/test_barrier_destroy_busy.cpp)
  target_link_libraries(test_barrier_destroy_busy PRIVATE prosper_core)
  add_test(NAME barrier_destroy_busy COMMAND test_barrier_destroy_busy)

  add_executable(test_reallocf tests/hle/test_reallocf.cpp)
  target_link_libraries(test_reallocf PRIVATE prosper_core)
  add_test(NAME reallocf COMMAND test_reallocf)

  # #2081 FALSE SUCCESS: an unregistered NID answers the dispatcher's 0, which for an SCE_OK
  # contract is a success report over an out-parameter nothing wrote. Pure, no game dump.
  add_executable(test_false_success_nids tests/hle/test_false_success_nids.cpp)
  target_link_libraries(test_false_success_nids PRIVATE prosper_core)
  add_test(NAME false_success_nids COMMAND test_false_success_nids)

  # #2004: sceAppContentAddcontUnmount releases the claim the mount takes, so a mount/unmount/
  # re-mount cycle works instead of being permanently BUSY. Pure, no game dump.
  add_executable(test_addcontent_unmount tests/hle/service/test_addcontent_unmount.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_addcontent_unmount PRIVATE tests)
  target_link_libraries(test_addcontent_unmount PRIVATE prosper_core)
  add_test(NAME addcontent_unmount COMMAND test_addcontent_unmount)

  add_executable(test_abnormal_termination_stops tests/hle/service/test_abnormal_termination_stops.cpp)
  target_include_directories(test_abnormal_termination_stops PRIVATE tests)
  target_link_libraries(test_abnormal_termination_stops PRIVATE prosper_core)
  add_test(NAME abnormal_termination_stops COMMAND test_abnormal_termination_stops)

  add_executable(test_addcontent_mount_report tests/hle/service/test_addcontent_mount_report.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_addcontent_mount_report PRIVATE tests)
  target_link_libraries(test_addcontent_mount_report PRIVATE prosper_core)
  add_test(NAME addcontent_mount_report COMMAND test_addcontent_mount_report)

  # libSceSysmodule state query (#2002): IsLoaded answers from prosper's own load history, so a
  # title that gates its initialization on UNLOADED stops skipping that initialization.
  add_executable(test_sysmodule_state tests/hle/test_sysmodule_state.cpp)
  target_link_libraries(test_sysmodule_state PRIVATE prosper_core)
  add_test(NAME sysmodule_state COMMAND test_sysmodule_state)

  # libSceHttp local URI parser (#642): two-pass caller pool sizing, public URI layout, and
  # deterministic error paths. Pure and offline; it never contacts a network.
  add_executable(test_http tests/hle/net/test_http.cpp)
  target_link_libraries(test_http PRIVATE prosper_core)
  add_test(NAME http_hle COMMAND test_http)

  add_executable(test_guard_recursion tests/hle/test_guard_recursion.cpp)
  target_link_libraries(test_guard_recursion PRIVATE prosper_core Threads::Threads)
  add_test(NAME guard_recursion COMMAND test_guard_recursion)
  set_tests_properties(guard_recursion PROPERTIES TIMEOUT 30)

  add_executable(test_font tests/hle/test_font.cpp)
  target_link_libraries(test_font PRIVATE prosper_core)
  add_test(NAME font_hle COMMAND test_font)

  add_executable(test_font_render tests/hle/test_font_render.cpp)
  target_link_libraries(test_font_render PRIVATE prosper_core)
  add_test(NAME font_render_hle COMMAND test_font_render)

  add_executable(test_fiber tests/hle/test_fiber.cpp)
  target_link_libraries(test_fiber PRIVATE prosper_core)
  add_test(NAME fiber_hle COMMAND test_fiber)

  # #1603: libSceUlt has no implementation, so it must fail VISIBLY rather than return 0 = SCE_OK.
  # Pure, no game dump.
  add_executable(test_ult_fail_visible tests/hle/test_ult_fail_visible.cpp)
  target_link_libraries(test_ult_fail_visible PRIVATE prosper_core)
  add_test(NAME ult_fail_visible COMMAND test_ult_fail_visible)

  # #1603 Phase 2: the implemented libSceUlt semantics. Mutual exclusion is DEMONSTRATED with real
  # concurrent threads over a non-atomic counter and an exact expected total — a no-op mutex produces
  # a wrong-but-plausible number rather than a crash, so only the exact total is a real check.
  # Pure, no game dump.
  add_executable(test_ult_semantics tests/hle/test_ult_semantics.cpp)
  target_link_libraries(test_ult_semantics PRIVATE prosper_core Threads::Threads)
  add_test(NAME ult_semantics COMMAND test_ult_semantics)

  add_executable(test_posix_sem tests/hle/test_posix_sem.cpp)
  target_link_libraries(test_posix_sem PRIVATE prosper_core)
  add_test(NAME posix_sem_hle COMMAND test_posix_sem)

  # FreeBSD errno numbering across the libkernel 0x80020000|errno family (#1612). Pins the encoding
  # vocabulary, the host->FreeBSD table, and the guest-observable values on the paths a Linux/macOS
  # host can actually reach (sem trywait, timed mutex, wait-on-address, POSIX setcancelstate).
  add_executable(test_sce_errno tests/hle/kernel/test_sce_errno.cpp)
  target_link_libraries(test_sce_errno PRIVATE prosper_core)
  add_test(NAME sce_errno_freebsd COMMAND test_sce_errno)

  # #1659: every diagnostic that prints "<module>+0x<rva>" agrees with the loader's mapped base, and a
  # printed offset round-trips back through PROSPER_BP. Header-only; no dump, no device.
  add_executable(test_guest_module_label tests/host/image/test_guest_module_label.cpp)
  target_link_libraries(test_guest_module_label PRIVATE prosper_core)
  add_test(NAME guest_module_label COMMAND test_guest_module_label)

  # The frame-pointer walk shared by the fault reporter and PROSPER_HWBP_STACK. It dereferences
  # guest-controlled pointers inside a signal handler, so every stop condition (cycle, unreadable
  # frame, depth cap) is a safety property. Header-only; no dump, no device.
  add_executable(test_rbp_chain tests/host/fault/test_rbp_chain.cpp)
  target_link_libraries(test_rbp_chain PRIVATE prosper_core)
  add_test(NAME rbp_chain COMMAND test_rbp_chain)

  # PlatformUi hook (#347): the app frontend can service the guest's ImeDialog with real UI; with no
  # backend registered the headless auto-complete default is unchanged. Pure, no game dump.
  add_executable(test_platform_ui tests/hle/service/test_platform_ui.cpp)
  target_link_libraries(test_platform_ui PRIVATE prosper_core)
  add_test(NAME platform_ui COMMAND test_platform_ui)

  # SDL3 dialog frontend's guest-struct parsing (#347): pure, SDL-free, so it runs in normal CI even
  # though the SDL_ShowMessageBox glue is display-only. Verifies the MsgDialog/ErrorDialog layouts.
  add_executable(test_dialog_helpers frontends/dialog_sdl3/test_dialog_helpers.cpp
                                     frontends/dialog_sdl3/dialog_helpers.cpp)
                                     # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
                                     target_include_directories(test_dialog_helpers PRIVATE src)
  add_test(NAME dialog_helpers COMMAND test_dialog_helpers)

  # libSceImeDialog lifecycle + libSceSaveData memory API round-trip (#191): auto-completing dialog
  # (no hang) + a real per-(user,slot) Set->Get memory block with verified shadPS4 struct layouts.
  add_executable(test_savedata_ime tests/hle/test_savedata_ime.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_savedata_ime PRIVATE tests)
  target_link_libraries(test_savedata_ime PRIVATE prosper_core)
  add_test(NAME savedata_ime COMMAND test_savedata_ime)

  # Save data is namespaced by title id (#2734): two titles that pick the same guest save name --
  # `OptionSettings`, or SaveDataMemory's (user 1, slot 0) -- must not share one host save. Runs two
  # application roots over one guest name, because a single-title round trip passes on the broken code.
  add_executable(test_savedata_title_namespace tests/hle/test_savedata_title_namespace.cpp)
  target_include_directories(test_savedata_title_namespace PRIVATE tests)
  target_link_libraries(test_savedata_title_namespace PRIVATE prosper_core)
  add_test(NAME savedata_title_namespace COMMAND test_savedata_title_namespace)

  # libSceIme keyboard API (#186): reports a consistent "no keyboard connected" state (empty resource
  # array + disconnected info, bounds-checked) so the per-frame input poll never reads garbage.
  add_executable(test_ime_keyboard tests/hle/service/test_ime_keyboard.cpp)
  target_link_libraries(test_ime_keyboard PRIVATE prosper_core)
  add_test(NAME ime_keyboard COMMAND test_ime_keyboard)

  # pthread scheduling getters fill their out-param (not a shadowing no-op): scePthreadGetprio/
  # Getaffinity must WRITE priority/mask, else the caller reads uninitialized stack (harmful-stub class).
  add_executable(test_pthread_sched_getters tests/hle/test_pthread_sched_getters.cpp)
  target_link_libraries(test_pthread_sched_getters PRIVATE prosper_core)
  add_test(NAME pthread_sched_getters COMMAND test_pthread_sched_getters)

  add_executable(test_pthread_names tests/hle/test_pthread_names.cpp)
  target_link_libraries(test_pthread_names PRIVATE prosper_core)
  add_test(NAME pthread_names COMMAND test_pthread_names)

  add_executable(test_cond_clock tests/hle/test_cond_clock.cpp)
  target_link_libraries(test_cond_clock PRIVATE prosper_core)
  add_test(NAME pthread_cond_clock COMMAND test_cond_clock)

  # libScePad controller input: ScePadData layout + HostPadState mapping + HLE fill contract (pure).
  add_executable(test_pad tests/input/test_pad.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_pad PRIVATE tests)
  target_link_libraries(test_pad PRIVATE prosper_core)
  add_test(NAME pad_input COMMAND test_pad)

  # Render-cadence inertness: when a retained DMA copy overrides EVERY skip PROSPER_RENDER_EVERY
  # asked for, the run is a cadence-1 run wearing an `every=N` label and must say so (#2837).
  add_executable(test_render_cadence tests/hle/test_render_cadence.cpp)
  target_link_libraries(test_render_cadence PRIVATE prosper_core)
  add_test(NAME render_cadence COMMAND test_render_cadence)

  add_executable(test_prosper_app_pad_overlay frontends/prosper-app/test_pad_overlay.cpp)
  target_include_directories(test_prosper_app_pad_overlay PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_pad_overlay PRIVATE prosper_core)
  add_test(NAME prosper_app_pad_overlay COMMAND test_prosper_app_pad_overlay)

  # WHAT the --fps counter says, kept out of the ImGui/Vulkan overlay that draws it. The order of the
  # two rates is the load-bearing part: a HUD that led with the PRESENTED rate would read ~60 fps for
  # a frozen title (#2783).
  add_executable(test_prosper_app_fps_hud frontends/prosper-app/test_fps_hud.cpp
                                          src/gpu/present/present_frame_rate.cpp)
  target_include_directories(test_prosper_app_fps_hud PRIVATE src frontends/prosper-app)
  add_test(NAME prosper_app_fps_hud COMMAND test_prosper_app_fps_hud)

  # Which key is which pad control (#2234). No SDL here: the frontend keeps the Key -> scancode
  # table and the mapping itself lives in a header, so this runs in the core suite on every host.
  add_executable(test_prosper_app_keyboard_pad_map frontends/prosper-app/test_keyboard_pad_map.cpp)
  target_include_directories(test_prosper_app_keyboard_pad_map PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_keyboard_pad_map PRIVATE prosper_core)
  add_test(NAME prosper_app_keyboard_pad_map COMMAND test_prosper_app_keyboard_pad_map)

  add_executable(test_prosper_app_window_controls frontends/prosper-app/test_window_controls.cpp)
  target_include_directories(test_prosper_app_window_controls PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_window_controls COMMAND test_prosper_app_window_controls)

  add_executable(test_prosper_app_snap_author frontends/prosper-app/test_snap_author.cpp)
  target_include_directories(test_prosper_app_snap_author PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_snap_author COMMAND test_prosper_app_snap_author)

  add_executable(test_prosper_app_capture_schedule frontends/prosper-app/test_capture_schedule.cpp)
  target_include_directories(test_prosper_app_capture_schedule PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_capture_schedule COMMAND test_prosper_app_capture_schedule)

  add_executable(test_prosper_app_performance_capture_schedule
    frontends/prosper-app/test_performance_capture_schedule.cpp)
  target_include_directories(test_prosper_app_performance_capture_schedule
    PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_performance_capture_schedule
    COMMAND test_prosper_app_performance_capture_schedule)

  add_executable(test_prosper_app_present_mode frontends/prosper-app/test_present_mode.cpp)
  target_include_directories(test_prosper_app_present_mode PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_present_mode COMMAND test_prosper_app_present_mode)

  # The interactive open path's path-resolution + action decisions (#1469): pure and SDL-free, so
  # the drag-and-drop / folder-picker behaviour is covered in normal CI without a window.
  add_executable(test_prosper_app_game_path frontends/prosper-app/test_game_path.cpp)
  target_include_directories(test_prosper_app_game_path PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_path COMMAND test_prosper_app_game_path)

  # The game library scan, param.json metadata, and settings precedence (#1471): filesystem access is
  # injected, so the whole scan runs in normal CI with no window, no ImGui and no real dump.
  add_executable(test_prosper_app_game_library frontends/prosper-app/test_game_library.cpp)
  target_include_directories(test_prosper_app_game_library PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_library COMMAND test_prosper_app_game_library)

  # F9 frame-grab file naming: one stamp and one exclusively claimed name pair per capture, so two
  # titles in one directory cannot overwrite each other and an aborted grab cannot pair with a
  # different capture. Uses a real directory under the build tree and an injected clock — no SDL, no
  # window, no GPU.
  add_executable(test_prosper_app_frame_grab_naming frontends/prosper-app/test_frame_grab_naming.cpp)
  target_include_directories(test_prosper_app_frame_grab_naming PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_frame_grab_naming COMMAND test_prosper_app_frame_grab_naming)

  # F8's pre-trigger ring and post-trigger timing artifact: pure state machine + filesystem test,
  # including bounded overflow accounting and temporary-to-final atomic publication.
  add_executable(test_performance_capture frontends/shared/tests/test_performance_capture.cpp)
  target_link_libraries(test_performance_capture PRIVATE prosper_performance_capture)
  add_test(NAME performance_capture COMMAND test_performance_capture)

  if(Python3_Interpreter_FOUND)
    add_test(NAME performance_capture_report_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_performance_capture_report.py")
  endif()

  # Library grid selection movement (#1471): arithmetic only, so the navigation rules are covered in
  # normal CI rather than by someone pressing arrow keys. No ImGui, no SDL, no window.
  add_executable(test_prosper_app_library_nav frontends/prosper-app/test_library_nav.cpp)
  target_include_directories(test_prosper_app_library_nav PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_library_nav COMMAND test_prosper_app_library_nav)

  # Library descriptor-pool sizing and the exhaustion rule (#1649): the pool must follow the title count
  # instead of a fixed 256, and a set must never be allocated out of a full pool — ImGui's backend writes
  # to whatever vkAllocateDescriptorSets returned without checking it. Pure arithmetic plus a fake
  # allocator that records whether it ran, so no Vulkan, no ImGui and no window.
  add_executable(test_prosper_app_library_descriptor_budget
                 frontends/prosper-app/test_library_descriptor_budget.cpp)
  target_include_directories(test_prosper_app_library_descriptor_budget PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_library_descriptor_budget
           COMMAND test_prosper_app_library_descriptor_budget)

  # Library background art and focus music (#1630): DDS/BC7 and RIFF/ATRAC9 container validation plus
  # the debounce, two-stage fade and LRU policy. Pure byte and arithmetic work — the containers are
  # synthesized in the test, so no dump, no GPU and no audio device is involved.
  add_executable(test_prosper_app_launcher_media frontends/prosper-app/test_launcher_media.cpp)
  target_include_directories(test_prosper_app_launcher_media PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_launcher_media COMMAND test_prosper_app_launcher_media)

  # Fault-safe guest-pointer read in the PROSPER_AMPRLOG diagnostic (#1154): an unmapped, aligned,
  # > 0xffff pointer (GTA V's 0x302200) must report unreadable, not SIGSEGV. POSIX-only
  # (process_vm_readv / mmap); ampr_arglog lives in the __linux__||__APPLE__ half of hle_kernel_mem.
  if(NOT WIN32)
    add_executable(test_amprlog_safe_read tests/host/platform/test_amprlog_safe_read.cpp)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    target_include_directories(test_amprlog_safe_read PRIVATE src)
    add_test(NAME amprlog_safe_read COMMAND test_amprlog_safe_read)
  endif()

  # Compute-side renderer-owned-RTT scale detection (PROSPER_RENDER_SCALE>1 downscale vs a view alias).
  # Pure integer arithmetic, no deps.
  add_executable(test_rtt_scale frontends/shared/tests/test_rtt_scale.cpp)
  target_include_directories(test_rtt_scale PRIVATE frontends)
  add_test(NAME rtt_scale COMMAND test_rtt_scale)

  # MRT-prefix truncation: which bound colour slot may join a pass, and which extent comparisons are
  # evidence of a real conflict rather than an unmeasured 0x0. Pure integer arithmetic, no deps.
  add_executable(test_mrt_extent frontends/shared/tests/test_mrt_extent.cpp)
  target_include_directories(test_mrt_extent PRIVATE frontends)
  add_test(NAME mrt_extent COMMAND test_mrt_extent)

  # ONE active-colour-binding policy, shared by pass grouping and same-pass feedback detection, built
  # from a real DrawItem. A second, looser copy of this rule classified stale named state as a live
  # binding (#2550 review).
  add_executable(test_mrt_binding frontends/shared/tests/test_mrt_binding.cpp)
  target_include_directories(test_mrt_binding PRIVATE frontends src)
  add_test(NAME mrt_binding COMMAND test_mrt_binding)

  add_executable(test_rtt_injection frontends/shared/tests/test_rtt_injection.cpp)
  target_include_directories(test_rtt_injection PRIVATE frontends)
  add_test(NAME rtt_injection COMMAND test_rtt_injection)

  add_executable(test_readback_policy frontends/shared/tests/test_readback_policy.cpp)
  target_include_directories(test_readback_policy PRIVATE frontends)
  add_test(NAME readback_policy COMMAND test_readback_policy)

  add_executable(test_capture_renderer_policy frontends/shared/tests/test_capture_renderer_policy.cpp)
  target_include_directories(test_capture_renderer_policy PRIVATE frontends)
  add_test(NAME capture_renderer_policy COMMAND test_capture_renderer_policy)

  # Which sampled resource is AvPlayer's NV12 chroma plane (#2005). Descriptor-level classification
  # only — no device — but it reads the real guest-linear pitch registry, so it links prosper_core.
  add_executable(test_avplayer_plane_policy frontends/shared/tests/test_avplayer_plane_policy.cpp)
  target_include_directories(test_avplayer_plane_policy PRIVATE frontends src)
  target_link_libraries(test_avplayer_plane_policy PRIVATE prosper_core)
  add_test(NAME avplayer_plane_policy COMMAND test_avplayer_plane_policy)

  # The publish extent contract (#1986): which rendered pass may become the present source, and what
  # the retained-frame net may retain. Pure byte-count policy, so no device and no renderer.
  add_executable(test_present_extent frontends/shared/tests/test_present_extent.cpp)
  target_include_directories(test_present_extent PRIVATE frontends)
  add_test(NAME present_extent COMMAND test_present_extent)

  # When prosper may publish the guest's OWN flipped display buffer (#1968), including the guest
  # authorship precondition whose absence reverted the first attempt (#2044). Pure policy, no device.
  add_executable(test_guest_scanout_present frontends/shared/tests/test_guest_scanout_present.cpp)
  target_include_directories(test_guest_scanout_present PRIVATE frontends)
  add_test(NAME guest_scanout_present COMMAND test_guest_scanout_present)

  # The census window PROSPER_PASS_LOG / PROSPER_DUMP_PERSISTENT open: an ordinal (unchanged) or
  # `ms:<millis>` (#1968). Pure predicate — no renderer, no device, and the clock is an argument.
  add_executable(test_diagnostic_window frontends/shared/tests/test_diagnostic_window.cpp)
  target_include_directories(test_diagnostic_window PRIVATE frontends)
  add_test(NAME diagnostic_window COMMAND test_diagnostic_window)

  # boot_trace periodic BMPs are opt-in, with the legacy disable taking priority. This exercises the
  # exact environment-name lookup and registrar forwarding without constructing Vulkan or writing files.
  add_executable(test_frame_dump_policy frontends/shared/tests/test_frame_dump_policy.cpp)
  target_include_directories(test_frame_dump_policy PRIVATE frontends)
  add_test(NAME frame_dump_policy COMMAND test_frame_dump_policy)

  add_executable(test_capture_compute_policy tests/gpu/capture/test_capture_compute_policy.cpp)
  target_include_directories(test_capture_compute_policy PRIVATE src)
  add_test(NAME capture_compute_policy COMMAND test_capture_compute_policy)

  add_executable(test_write_watch_policy frontends/shared/tests/test_write_watch_policy.cpp)
  # `src` as well as `frontends`: the knobs' strict parsing lives in diagnostics/env_numeric.hpp,
  # and it is asserted here beside the policy whose sentinels make a typo dangerous (#3253).
  target_include_directories(test_write_watch_policy PRIVATE frontends src)
  add_test(NAME write_watch_policy COMMAND test_write_watch_policy)

  # Pooled texture-decode intermediates. `src` for diagnostics/env_numeric.hpp, which the pool's
  # retention knob parses through -- the equivalence arm is the one that matters (see the file head).
  add_executable(test_decode_scratch frontends/shared/tests/test_decode_scratch.cpp)
  target_include_directories(test_decode_scratch PRIVATE frontends src)
  add_test(NAME decode_scratch COMMAND test_decode_scratch)
  # The two arms the pooling claim rests on, each asserting the COMPLEMENT of the default run so
  # neither can pass by accident: with retention disabled the shared pool must keep nothing (and a
  # ScratchBuffer must therefore not reuse pages), and with poison armed every lease must arrive
  # holding 0xCD rather than the incidental zeros a fresh mmap would have given it.
  add_test(NAME decode_scratch_no_retention COMMAND test_decode_scratch)
  set_tests_properties(decode_scratch_no_retention PROPERTIES
    ENVIRONMENT "PROSPER_DECODE_SCRATCH_MB=0")
  add_test(NAME decode_scratch_poisoned COMMAND test_decode_scratch)
  set_tests_properties(decode_scratch_poisoned PROPERTIES
    ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

  # #3307 compute->graphics image borrow census. Header-only and device-free: the load-bearing arm
  # is the 2048-case equivalence between the importer's precondition classifier and the `||` chain
  # it replaced, which needs no Vulkan and must never be skipped on a host without it.
  add_executable(test_compute_image_borrow_census
                 frontends/shared/tests/test_compute_image_borrow_census.cpp)
  target_include_directories(test_compute_image_borrow_census PRIVATE frontends)
  add_test(NAME compute_image_borrow_census COMMAND test_compute_image_borrow_census)

  # Storage-image alias access union is metadata-only and must run without Vulkan on every platform.
  add_executable(test_storage_image_alias_plan
                 frontends/shared/tests/test_storage_image_alias_plan.cpp)
  target_include_directories(test_storage_image_alias_plan PRIVATE frontends src)
  target_link_libraries(test_storage_image_alias_plan PRIVATE prosper_core)
  add_test(NAME storage_image_alias_plan COMMAND test_storage_image_alias_plan)

  add_executable(test_texture_decode_diagnostic
                 frontends/shared/tests/test_texture_decode_diagnostic.cpp)
  target_include_directories(test_texture_decode_diagnostic PRIVATE frontends src)
  target_link_libraries(test_texture_decode_diagnostic PRIVATE prosper_core)
  add_test(NAME texture_decode_diagnostic COMMAND test_texture_decode_diagnostic)

  # Bounded-acquire present classification (#1182). Needs Vulkan HEADERS only (VkResult enum), no loader
  # call — link Vulkan::Vulkan purely for its include dirs. Guarded so a headers-less host skips it
  # gracefully; every platform that builds prosper-app has Vulkan available.
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(TARGET Vulkan::Vulkan)
    add_executable(test_prosper_app_present_policy frontends/prosper-app/test_present_policy.cpp)
    target_include_directories(test_prosper_app_present_policy PRIVATE frontends/prosper-app)
    target_link_libraries(test_prosper_app_present_policy PRIVATE Vulkan::Vulkan)
    add_test(NAME prosper_app_present_policy COMMAND test_prosper_app_present_policy)

    add_executable(test_present_blit_policy frontends/shared/tests/test_present_blit_policy.cpp)
    target_include_directories(test_present_blit_policy PRIVATE frontends)
    target_link_libraries(test_present_blit_policy PRIVATE Vulkan::Vulkan)
    add_test(NAME present_blit_policy COMMAND test_present_blit_policy)
  endif()

  add_executable(test_sync_on_address tests/hle/test_sync_on_address.cpp)
  target_link_libraries(test_sync_on_address PRIVATE prosper_core)
  add_test(NAME sync_on_address COMMAND test_sync_on_address)

  # PROSPER_SYNC_RING is sized from the environment at static init, so the ring size must be set
  # for the process rather than by the test body. 64 is deliberately tiny: the property under test is
  # what the dump reports once it has DROPPED events.
  add_executable(test_sync_ring tests/hle/sync/test_sync_ring.cpp)
  target_link_libraries(test_sync_ring PRIVATE prosper_core)
  add_test(NAME sync_ring COMMAND test_sync_ring)
  set_tests_properties(sync_ring PROPERTIES ENVIRONMENT "PROSPER_SYNC_RING=64")

  add_executable(test_cond_slot_identity tests/hle/sync/test_cond_slot_identity.cpp)
  target_link_libraries(test_cond_slot_identity PRIVATE prosper_core)
  add_test(NAME cond_slot_identity COMMAND test_cond_slot_identity)

  # sceKernelDlsym honors its module handle (#147): per-module export tables, real handles from
  # LoadStartModule for linked-module paths, global first-wins fallback. Pure, no game dump needed.
  add_executable(test_dlsym_handle tests/hle/test_dlsym_handle.cpp)
  target_link_libraries(test_dlsym_handle PRIVATE prosper_core)
  add_test(NAME dlsym_handle COMMAND test_dlsym_handle)

  # Guest mutex TYPE semantics (#145): settype honored, explicit Sony/FreeBSD ERRORCHECK default,
  # non-recursive static-sentinel self-init, FreeBSD errno values (EDEADLK=11). Pure, no game dump.
  add_executable(test_mutex_types tests/hle/kernel/test_mutex_types.cpp)
  target_link_libraries(test_mutex_types PRIVATE prosper_core)
  add_test(NAME mutex_types COMMAND test_mutex_types)

  add_executable(test_agc_shader tests/gpu/test_agc_shader.cpp)
  target_link_libraries(test_agc_shader PRIVATE prosper_core)
  add_test(NAME agc_shader_create COMMAND test_agc_shader)

  add_executable(test_tile tests/gpu/texture/test_tile.cpp)
  target_link_libraries(test_tile PRIVATE prosper_core)
  add_test(NAME gpu_surface_detile COMMAND test_tile)
  # Run the same byte-identity suite through the row-parallel detile path (#1177), forcing threads so
  # it is exercised even on a low-core CI. Passing here + in gpu_surface_detile (single-threaded control
  # via the < 2-thread gate / PROSPER_DETILE_SINGLE_THREADED) proves threaded == single byte-for-byte.
  add_test(NAME gpu_surface_detile_threaded COMMAND test_tile)
  set_tests_properties(gpu_surface_detile_threaded PROPERTIES
                       ENVIRONMENT "PROSPER_DETILE_THREADS=8")

  # BC1/BC2/BC3 block decompression to RGBA8 (#121): hand-built blocks with known output. Pure.
  add_executable(test_bc_decode tests/gpu/texture/test_bc_decode.cpp)
  target_link_libraries(test_bc_decode PRIVATE prosper_core)
  add_test(NAME bc_decode COMMAND test_bc_decode)

  add_executable(test_stat tests/test_stat.cpp)
  target_link_libraries(test_stat PRIVATE prosper_core)
  add_test(NAME sce_stat_layout COMMAND test_stat)

  add_executable(test_file_binary tests/hle/test_file_binary.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_file_binary PRIVATE tests)
  target_link_libraries(test_file_binary PRIVATE prosper_core)
  add_test(NAME file_binary_roundtrip COMMAND test_file_binary)

  add_executable(test_savedata_dirname tests/hle/test_savedata_dirname.cpp)
  target_link_libraries(test_savedata_dirname PRIVATE prosper_core)
  add_test(NAME savedata_dirname_guard COMMAND test_savedata_dirname)

  add_executable(test_savedata_txres tests/hle/test_savedata_txres.cpp)
  target_link_libraries(test_savedata_txres PRIVATE prosper_core)
  add_test(NAME savedata_tx_resource_ids COMMAND test_savedata_txres)

  add_executable(test_savedata_event_drain tests/hle/service/test_savedata_event_drain.cpp)
  target_link_libraries(test_savedata_event_drain PRIVATE prosper_core)
  add_test(NAME savedata_event_drain_code COMMAND test_savedata_event_drain)

  # libScePsml must report failure rather than the dispatcher's default SCE_OK: the guest reads an
  # out-struct prosper never wrote and allocates from it (PPSA20447). Pure, no dump, no device.
  add_executable(test_psml_unimplemented tests/hle/service/test_psml_unimplemented.cpp)
  target_link_libraries(test_psml_unimplemented PRIVATE prosper_core)
  add_test(NAME psml_unimplemented COMMAND test_psml_unimplemented)

  add_executable(test_translate_sandbox tests/hle/test_translate_sandbox.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_translate_sandbox PRIVATE tests)
  target_link_libraries(test_translate_sandbox PRIVATE prosper_core)
  add_test(NAME translate_sandbox_guard COMMAND test_translate_sandbox)

  add_executable(test_service_logging tests/hle/test_service_logging.cpp)
  target_link_libraries(test_service_logging PRIVATE prosper_core)
  add_test(NAME service_logging_unmapped_args COMMAND test_service_logging)

  if(WIN32)
    add_executable(test_win_exception_delivery tests/misc/test_win_exception_delivery.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_win_exception_delivery PRIVATE tests)
    target_link_libraries(test_win_exception_delivery PRIVATE prosper_core)
    add_test(NAME win_exception_delivery COMMAND test_win_exception_delivery)

    add_executable(test_win_pthread_key_destructor tests/hle/test_win_pthread_key_destructor.cpp)
    target_link_libraries(test_win_pthread_key_destructor PRIVATE prosper_core)
    add_test(NAME win_pthread_key_destructor COMMAND test_win_pthread_key_destructor)

    add_executable(test_win_kernel_is_stack tests/host/image/test_win_kernel_is_stack.cpp)
    target_link_libraries(test_win_kernel_is_stack PRIVATE prosper_core)
    add_test(NAME win_kernel_is_stack COMMAND test_win_kernel_is_stack)
  endif()

  # Boot preload path case correction (#1006): a module present with different casing than the
  # hard-coded preload path (Blasphemous 2 / Evergate's Il2CppUserAssemblies.prx on a case-sensitive
  # host filesystem) must resolve to the real on-disk entry instead of being dropped as absent.
  # Pure std::filesystem against a temp tree; no game dump needed.
  add_executable(test_module_path_case tests/host/image/test_module_path_case.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_module_path_case PRIVATE tests)
  target_link_libraries(test_module_path_case PRIVATE prosper_core)
  add_test(NAME module_path_case COMMAND test_module_path_case)

  # Module path policy: prosper links guest modules only from the dump locations it names
  # (eboot.bin, sce_module/, Media/Modules/, Media/Plugins/), rejects everything else by default,
  # and never links a Sony-named library out of the wholesale-auto-linked Media directories. Some
  # dumps ship third-party replacements of libSceAppContent / libSceNpEntitlementAccess /
  # libSceAmpr; loading one would shadow prosper's own implementation and answer the guest with
  # somebody else's stub. Pure string policy plus one end-to-end arm against a temp tree.
  add_executable(test_module_path_policy tests/host/image/test_module_path_policy.cpp)
  target_include_directories(test_module_path_policy PRIVATE tests)
  target_link_libraries(test_module_path_policy PRIVATE prosper_core)
  add_test(NAME module_path_policy COMMAND test_module_path_policy)

  # dump_hygiene classifies files in a game dump that prosper must never load, and can delete them.
  # Its control arms matter more than its positive ones: during development it twice classified
  # AUTHENTIC dump content as removable. The suite pins both regressions and the guards behind them.
  add_test(NAME dump_hygiene_classifier
           COMMAND ${Python3_EXECUTABLE} -m unittest tests.tools.test_dump_hygiene
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Diagnostics: disabled-path test — verifies stubs are zero-cost (always runs).
  add_executable(test_diagnostics_disabled tests/diagnostics/test_diagnostics_disabled.cpp)
  target_link_libraries(test_diagnostics_disabled PRIVATE prosper_core)
  add_test(NAME diagnostics_disabled_stubs COMMAND test_diagnostics_disabled)

  # The boot-phase LINE is reachable in a default build (no PROSPER_DIAGNOSTICS), driven only by
  # PROSPER_BOOTPHASE. Deliberately registered outside the PROSPER_DIAGNOSTICS block: the property
  # under test is that the default build — the one everybody actually runs — can be made to say
  # where boot_program() stopped.
  # Which bundled support PRXs are preloaded at all. Pure policy: no dump, no SELF parser, no
  # filesystem — so both the DROP and the KEEP direction are checkable in ordinary CI.
  add_executable(test_support_modules tests/loader/test_support_modules.cpp
                                      src/loader/support_modules.cpp)
  target_include_directories(test_support_modules PRIVATE src)
  add_test(NAME loader_support_modules COMMAND test_support_modules)

  # Which preloaded modules are started with a real SCE module-param descriptor. Pure policy over
  # the fixed guest base map: no dump, no SELF parser, no filesystem — so both the inclusion and the
  # exclusion direction are checkable in ordinary CI.
  add_executable(test_module_start_params tests/host/image/test_module_start_params.cpp
                                          src/host/image/module_start_params.cpp)
  target_include_directories(test_module_start_params PRIVATE src)
  add_test(NAME module_start_params COMMAND test_module_start_params)

  add_executable(test_boot_phase_log tests/diagnostics/test_boot_phase_log.cpp)
  target_link_libraries(test_boot_phase_log PRIVATE prosper_core)
  add_test(NAME diagnostics_boot_phase_log COMMAND test_boot_phase_log)

  # PROSPER_ENV_ON / PROSPER_ENV_VALUE sample getenv exactly once (#3094). Header-only, so this
  # needs no library: it mutates the environment between two evaluations of the same macro and
  # asserts the second still reports the first value -- the one behaviour a live getenv cannot
  # produce, so the arms go red if the caching is ever removed.
  add_executable(test_env_cache tests/diagnostics/test_env_cache.cpp)
  target_include_directories(test_env_cache PRIVATE src)
  add_test(NAME diagnostics_env_cache COMMAND test_env_cache)

  # One arm per PROSPER_* knob #3267 converted to diagnostics/env_numeric.hpp. Header-only and
  # environment-free by construction: nearly every converted site caches its read in a
  # function-local static, so an arm that armed the variable at runtime would go VACUOUS rather
  # than red (#2214). Each arm therefore calls the same helper the site calls, and also computes
  # what the site's OLD spelling answered -- so an arm on a knob that never needed converting
  # fails instead of passing quietly.
  add_executable(test_env_numeric_sites tests/diagnostics/test_env_numeric_sites.cpp)
  target_include_directories(test_env_numeric_sites PRIVATE src)
  add_test(NAME diagnostics_env_numeric_sites COMMAND test_env_numeric_sites)

  # Diagnostics: enabled-path and JSON tests (only when feature is built).
  if(PROSPER_DIAGNOSTICS)
    add_executable(test_diagnostics_enabled tests/diagnostics/test_diagnostics_enabled.cpp)
    target_link_libraries(test_diagnostics_enabled PRIVATE prosper_core)
    target_compile_definitions(test_diagnostics_enabled PRIVATE PROSPER_DIAGNOSTICS)
    add_test(NAME diagnostics_enabled_capture COMMAND test_diagnostics_enabled)

    add_executable(test_diagnostics_json tests/diagnostics/test_diagnostics_json.cpp)
    target_link_libraries(test_diagnostics_json PRIVATE prosper_core)
    target_compile_definitions(test_diagnostics_json PRIVATE PROSPER_DIAGNOSTICS)
    add_test(NAME diagnostics_json_format COMMAND test_diagnostics_json)
  endif()

  # Unity native-plugin auto-link (#1609): a title's own Media/Plugins/*.prx that the fixed preload
  # list never named must still be linked, or its first P/Invoke raises a DllNotFoundException that
  # IL2CPP captures into a Task and never prints. Also guards the export-collision rule that keeps
  # auto-linking from introducing a duplicate NID export (Evergate ships libfmod.prx AND libfmodL.prx,
  # the same library's release and logging builds). The discovery and predicate assertions are pure
  # std::filesystem/struct work and need no dump; the dump, when configured, additionally exercises
  # link_program's honouring of the skip flag.
  add_executable(test_plugin_autolink tests/misc/test_plugin_autolink.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_plugin_autolink PRIVATE tests)
  target_link_libraries(test_plugin_autolink PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink "${GAME_DUMP}")
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  else()
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink)
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # link_program's structural contract on SYNTHESIZED modules (#2567). plugin_autolink's remaining
  # 14 assertions — the export table, the collision guard's effect on link_program, and the #1635
  # alias record — sat behind `if (argc >= 2)` plus a real Media/Plugins/PSN.prx, so they never ran
  # in CI and never ran for an agent without that title installed. These two cases register
  # UNCONDITIONALLY: their fixtures are minimal ET_SCE_DYNAMIC PRX modules emitted at test time by
  # tests/fixtures/synth_prx.h and tests/fixtures/handmade_prx.h, so there is no dump, no network and no committed
  # binary. Every byte is synthesized from the published ELF64 layout — nothing is carved out of a
  # dump, which would put Sony-built code in a public repository.
  #
  # Two cases, not one, because "the loader accepts this" and "the loader refuses that" are different
  # claims and the second is the one that gives the first its meaning: _reject holds the corrupted
  # variants, each differing from a linking control in exactly one field.
  add_executable(test_loader_synth_link tests/misc/test_loader_synth_link.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_loader_synth_link PRIVATE tests)
  target_link_libraries(test_loader_synth_link PRIVATE prosper_core)
  add_test(NAME loader_synth_link COMMAND test_loader_synth_link)

  add_executable(test_loader_synth_reject tests/misc/test_loader_synth_reject.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_loader_synth_reject PRIVATE tests)
  target_link_libraries(test_loader_synth_reject PRIVATE prosper_core)
  add_test(NAME loader_synth_reject COMMAND test_loader_synth_reject)

  # ATRAC9 decode (vendored LibAtrac9 + prosper glue): an embedded 16-superframe slice of a real PS5
  # snd0.at9 decodes bit-exact vs the LibAtrac9 reference (CRC32). Pure, no game dump; runs in CI.
  add_executable(test_atrac9 tests/hle/audio/test_atrac9.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_atrac9 PRIVATE tests)
  target_link_libraries(test_atrac9 PRIVATE prosper_core)
  add_test(NAME atrac9_decode COMMAND test_atrac9)

  # AJM batch-2.0 gapless decode semantics (#1097): skip/total trim, exact EOS landing of
  # uiTotalDecodedSamples, post-EOS silence, and program re-arm. FMOD's codec recycling keys off
  # the exact EOS; without it one-shot SFX exhaust the codec pool. Pure, no game dump; runs in CI.
  add_executable(test_ajm2_gapless tests/hle/audio/test_ajm2_gapless.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_ajm2_gapless PRIVATE tests)
  target_link_libraries(test_ajm2_gapless PRIVATE prosper_core)
  add_test(NAME ajm2_gapless COMMAND test_ajm2_gapless)

  # APR container registry + read-source resolution (#62/#1901): stable ids, path dedup, ID-first
  # reads, unique-size fallback, ambiguous fallback refusal, and method diagnostics. Pure; no dump.
  add_executable(test_apr_registry tests/host/memory/test_apr_registry.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_apr_registry PRIVATE tests)
  target_link_libraries(test_apr_registry PRIVATE prosper_core)
  add_test(NAME apr_registry_resolution COMMAND test_apr_registry)

  # #2384: the AMPR tracks-offset predicate. Its NEW case is provable by a boot; what needs a unit
  # test is that the widening is strictly ADDITIVE, because the predicate is shared by every title
  # that constructs a command buffer and none of their dumps are available to CI. Pure function of a
  # 64-bit argument, so it needs no dump and cannot skip silently.
  add_executable(test_ampr_tracks_offset tests/hle/memory/test_ampr_tracks_offset.cpp)
  target_link_libraries(test_ampr_tracks_offset PRIVATE prosper_core)
  add_test(NAME ampr_tracks_offset COMMAND test_ampr_tracks_offset)
  add_executable(test_ampr_amm tests/hle/memory/test_ampr_amm.cpp)
  target_link_libraries(test_ampr_amm PRIVATE prosper_core)
  add_test(NAME ampr_amm COMMAND test_ampr_amm)

  # #2908: a refused Ampr map must hand its direct-memory claim back. POSIX-only because the
  # mapping primitives it exercises are — on Windows sceAmprCommandBufferSetBuffer is still the
  # do-nothing stub, which cannot leak and cannot demonstrate the lever either.
  if(NOT WIN32)
    add_executable(test_ampr_map_refusal_releases_dmem
                   tests/hle/memory/test_ampr_map_refusal_releases_dmem.cpp)
    target_link_libraries(test_ampr_map_refusal_releases_dmem PRIVATE prosper_core)
    add_test(NAME ampr_map_refusal_releases_dmem COMMAND test_ampr_map_refusal_releases_dmem)
  endif()

  # GTA V (#1139/#1142) net-new HLE regression guards: sceKernelAprGetFileStat fills the real stat
  # (#1133) and sceAudioOut2GetSpeakerArrayMemorySize returns a non-zero allocation size (#1134). The
  # AGC GetSize third piece is covered separately by test_agc_getsize (#1208).
  add_executable(test_apr_stat_audio tests/hle/test_apr_stat_audio.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_apr_stat_audio PRIVATE tests)
  target_link_libraries(test_apr_stat_audio PRIVATE prosper_core)
  add_test(NAME apr_stat_audio_gtav COMMAND test_apr_stat_audio)

  # #2926/#2928: the APR read builders' two silent-success shapes -- an unregistered NID answering
  # SCE_OK from the dispatcher's return-0 default, and a gather/scatter segment reporting success
  # after its bytes failed to reach the guest destination -- plus the chain closer set, which the
  # handler's comment used to under-state. Pure: a scratch file and generated import stubs, no dump.
  add_executable(test_apr_gather_scatter tests/hle/test_apr_gather_scatter.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_apr_gather_scatter PRIVATE tests)
  target_link_libraries(test_apr_gather_scatter PRIVATE prosper_core)
  add_test(NAME apr_gather_scatter COMMAND test_apr_gather_scatter)
  # #3245: sceAmprMeasureCommandSizeReadFile's compatibility read fired for every title, so a title
  # that also submits its measured reads paid for the same bytes twice (13,925 duplicate reads /
  # 582 MB in a 3-minute Stray boot) and had them written before it recorded the command. Registered
  # twice on purpose -- the PROSPER_APR_MEASURE_READ=always run asserts the opposite outcome, which
  # is what shows the suppression actually suppressed. Pure: scratch files and generated import
  # stubs, no game dump.
  add_executable(test_apr_measure_duplicate tests/hle/test_apr_measure_duplicate.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_apr_measure_duplicate PRIVATE tests)
  target_link_libraries(test_apr_measure_duplicate PRIVATE prosper_core)
  add_test(NAME apr_measure_duplicate COMMAND test_apr_measure_duplicate)
  add_test(NAME apr_measure_duplicate_forced COMMAND test_apr_measure_duplicate)
  set_tests_properties(apr_measure_duplicate_forced PROPERTIES
                       ENVIRONMENT "PROSPER_APR_MEASURE_READ=always")
  add_test(NAME apr_measure_duplicate_never COMMAND test_apr_measure_duplicate)
  set_tests_properties(apr_measure_duplicate_never PROPERTIES
                       ENVIRONMENT "PROSPER_APR_MEASURE_READ=never")

  add_executable(test_ampr_measure tests/hle/test_ampr_measure.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_ampr_measure PRIVATE tests)
  target_link_libraries(test_ampr_measure PRIVATE prosper_core)
  add_test(NAME ampr_measure COMMAND test_ampr_measure)

  # sceAmprCommandBufferGetSize must not answer for one command buffer with another's recorded
  # capacity: a too-small answer permanently hangs the guest's IoStore append loop, which polls
  # GetSize(cb) - GetCurrentOffset(cb) > 0xff without a bound (The Forgotten City, PPSA03026).
  add_executable(test_ampr_getsize_fallback tests/hle/test_ampr_getsize_fallback.cpp)
  target_link_libraries(test_ampr_getsize_fallback PRIVATE prosper_core)
  add_test(NAME ampr_getsize_fallback COMMAND test_ampr_getsize_fallback)

  # Direct-memory allocate/available (issue #99): AvailableDirectMemorySize reports the largest
  # free aligned block in the search window and writes both out-params (was a total-size stub).
  add_executable(test_dmem tests/host/test_dmem.cpp)
  target_link_libraries(test_dmem PRIVATE prosper_core)
  add_test(NAME dmem_available COMMAND test_dmem)

  # PROSPER_DMEM_CALLER stack walk (issue #1755): the diagnostic scans up to 160 slots above its
  # frame and used to read straight off the top of a thread stack, dying with a SIGSEGV that reads
  # as a guest fault.
  add_executable(test_dmem_caller_walk tests/test_dmem_caller_walk.cpp)
  target_link_libraries(test_dmem_caller_walk PRIVATE prosper_core)
  add_test(NAME dmem_caller_walk COMMAND test_dmem_caller_walk)

  # PROSPER_DMEM_CALLER correlation (issue #1859): the full stack remains one bounded line per
  # caller identity while every MEMLOG allocation can carry the same stable run-local ID. Pure,
  # including explicit unknown/overflow states and concurrent interning.
  add_executable(test_dmem_caller_chain tests/hle/memory/test_dmem_caller_chain.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_dmem_caller_chain PRIVATE src)
  add_test(NAME dmem_caller_chain COMMAND test_dmem_caller_chain)

  # #3057: test_dmem_caller_chain above covers format_dmem_caller_chain_definition (the formatter)
  # only -- it never calls k_alloc_dmem/k_alloc_main_dmem, so deleting the actual
  # attribute_dmem_allocation(...) call sites #3055 added left that suite green. These two bind to
  # the real call sites through the NID dispatch table, one process each (see the header comment in
  # the first file for why they cannot share a process).
  add_executable(test_dmem_caller_attribution_alloc_dmem
                 tests/hle/memory/test_dmem_caller_attribution_alloc_dmem.cpp)
  target_link_libraries(test_dmem_caller_attribution_alloc_dmem PRIVATE prosper_core)
  add_test(NAME dmem_caller_attribution_alloc_dmem
           COMMAND test_dmem_caller_attribution_alloc_dmem)

  add_executable(test_dmem_caller_attribution_alloc_main_dmem
                 tests/hle/memory/test_dmem_caller_attribution_alloc_main_dmem.cpp)
  target_link_libraries(test_dmem_caller_attribution_alloc_main_dmem PRIVATE prosper_core)
  add_test(NAME dmem_caller_attribution_alloc_main_dmem
           COMMAND test_dmem_caller_attribution_alloc_main_dmem)

  # Capped-diagnostic rate limit (issue #1761): a print budget must never be readable as a
  # population. Pins the sparse tail and the per-key budget; header-only, no GPU state.
  add_executable(test_diag_ratelimit tests/gpu/diagnostics/test_diag_ratelimit.cpp)
  target_include_directories(test_diag_ratelimit PRIVATE src)
  add_test(NAME diag_ratelimit COMMAND test_diag_ratelimit)

  # IME keyboard event delivery (issue #1093): sceImeUpdate must invoke the guest handler
  # per queued key event; the old stub delivered none.
  add_executable(test_ime_input tests/hle/input/test_ime_input.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_ime_input PRIVATE tests)
  target_link_libraries(test_ime_input PRIVATE prosper_core)
  add_test(NAME ime_input COMMAND test_ime_input)

  # Huge non-fixed sceKernelReserveVirtualRange placement (#312/#946): the 512 GiB UE arena
  # must be steered off its low hint into the guest auto window (see test for the contract).
  add_executable(test_reserve_placement tests/hle/test_reserve_placement.cpp)
  target_link_libraries(test_reserve_placement PRIVATE prosper_core)
  add_test(NAME reserve_placement COMMAND test_reserve_placement)

  # A protection-only change must retain an uncommitted reservation's state (#343), or guest
  # VirtualQuery skips the later commit and the first access faults. Cross-platform HLE contract.
  add_executable(test_retrack_reservation tests/hle/test_retrack_reservation.cpp)
  target_link_libraries(test_retrack_reservation PRIVATE prosper_core)
  add_test(NAME retrack_reservation COMMAND test_retrack_reservation)

  # Hinted map must not clobber a live mapping (#137): MAP_FIXED_NOREPLACE unless the hint is our
  # own uncommitted reservation. Drives the real map HLE handlers. Linux-only (mmap semantics).
  if(UNIX)
    add_executable(test_map_noreplace tests/hle/test_map_noreplace.cpp)
    target_link_libraries(test_map_noreplace PRIVATE prosper_core)
    add_test(NAME map_noreplace COMMAND test_map_noreplace)

    # A guest mprotect(prot=0) makes the page PROT_NONE, not a default RW (#342): guard pages must
    # actually guard. Drives the real sceKernelMprotect + a SIGSEGV probe. Linux-only.
    add_executable(test_prot_none tests/hle/test_prot_none.cpp)
    target_link_libraries(test_prot_none PRIVATE prosper_core)
    add_test(NAME prot_none COMMAND test_prot_none)
  endif()

  # PM4 command-stream decoder (front of the CommandProcessor): decode a Dcb built by the AGC
  # Dcb functions and assert the ops. Pure/cross-platform, no game dump needed.
  add_executable(test_pm4_decode tests/gpu/pm4/test_pm4_decode.cpp)
  target_link_libraries(test_pm4_decode PRIVATE prosper_core)
  add_test(NAME pm4_decode_stream COMMAND test_pm4_decode)

  # CommandProcessor apply stage: replay a decoded stream into a GpuState (register files + draws).
  add_executable(test_command_processor tests/gpu/pm4/test_command_processor.cpp)
  target_link_libraries(test_command_processor PRIVATE prosper_core)
  add_test(NAME command_processor_apply COMMAND test_command_processor)

  add_executable(test_command_provenance tests/gpu/pm4/test_command_provenance.cpp)
  target_link_libraries(test_command_provenance PRIVATE prosper_core)
  add_test(NAME command_provenance COMMAND test_command_provenance)

  # Stage B: CommandProcessor honors RELEASE_MEM/WRITE_DATA memory writes (synchronous end-of-pipe).
  add_executable(test_eop_write tests/gpu/test_eop_write.cpp)
  target_link_libraries(test_eop_write PRIVATE prosper_core)
  add_test(NAME eop_write COMMAND test_eop_write)

  # #1945: the interior free-chain poison scan — its structural predicate, and the census that keeps
  # a zero from reading as "the chains are clean" when the walk reached nothing. POSIX only: the test
  # needs a MAP_FIXED anonymous mapping inside the guest VA window to build a pool array with real
  # addresses, and mingw-w64 has no <sys/mman.h>. The code under test is built on every platform.
  if(NOT WIN32)
    add_executable(test_mb3_poison_scan tests/gpu/execute/test_mb3_poison_scan.cpp)
    target_link_libraries(test_mb3_poison_scan PRIVATE prosper_core)
    add_test(NAME mb3_poison_scan COMMAND test_mb3_poison_scan)
    # A host that cannot place the mapping must SKIP, not pass: the structural half of the test
    # would otherwise be silently absent from a green run.
    set_tests_properties(mb3_poison_scan PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # #729: the synchronous (PROSPER_EOP_WRITE_SYNC=1) honor_* label writes must probe
  # guest_readable before dereferencing a guest-PM4-supplied address (own executable: the
  # env var latches process-wide on first use).
  add_executable(test_eop_write_sync_guard tests/gpu/pm4/test_eop_write_sync_guard.cpp)
  target_link_libraries(test_eop_write_sync_guard PRIVATE prosper_core)
  add_test(NAME eop_write_sync_guard COMMAND test_eop_write_sync_guard)

  # #312 per-queue WAIT_REG_MEM barrier model: unsatisfied waits gate downstream memory effects,
  # streams release independently, jumps don't un-gate, timeout backstop preserves liveness.
  add_executable(test_wait_barrier tests/gpu/pm4/test_wait_barrier.cpp)
  target_link_libraries(test_wait_barrier PRIVATE prosper_core)
  add_test(NAME wait_barrier COMMAND test_wait_barrier)

  # Render-state extraction: fold a stream into a GpuState, then interpret RDNA2 registers.
  add_executable(test_render_state tests/gpu/state/test_render_state.cpp)
  target_link_libraries(test_render_state PRIVATE prosper_core)
  add_test(NAME render_state_extract COMMAND test_render_state)

  # Messenger recompile guard (#228): committed real Messenger VS/PS RDNA2 binaries must keep
  # recompiling — a recompiler change that re-rejects one of their instructions (the #199/#201
  # blue-screen class) fails here instead of shipping. Pure (no Vulkan); needs no game dump.
  add_executable(test_messenger_recompile_guard tests/gpu/recompiler/test_messenger_recompile_guard.cpp)
  target_link_libraries(test_messenger_recompile_guard PRIVATE prosper_core)
  target_compile_definitions(test_messenger_recompile_guard PRIVATE
      PROSPER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME messenger_recompile_guard COMMAND test_messenger_recompile_guard)

  # libkernel/libScePosix signal + time-conversion surface (#190): NID match, registration, and real
  # behavior (UTC<->localtime round-trip fills its out-param; pthread_setcancelstate old-state + EINVAL).
  add_executable(test_kernel_signal_time tests/hle/test_kernel_signal_time.cpp)
  target_link_libraries(test_kernel_signal_time PRIVATE prosper_core)
  add_test(NAME kernel_signal_time COMMAND test_kernel_signal_time)

  # Pipeline-state resolution: RenderState (RDNA2 enums) -> Vulkan-ready pipeline state (pure, no Vulkan).
  add_executable(test_pipeline_state tests/gpu/state/test_pipeline_state.cpp)
  target_link_libraries(test_pipeline_state PRIVATE prosper_core)
  add_test(NAME pipeline_state_resolve COMMAND test_pipeline_state)

  # Recompiler coverage metric (pure): per-instruction support report used by shader_histo.
  add_executable(test_recompile_coverage tests/gpu/test_recompile_coverage.cpp)
  target_link_libraries(test_recompile_coverage PRIVATE prosper_core)
  add_test(NAME recompile_coverage COMMAND test_recompile_coverage)

  add_executable(test_direct_descriptor_copy tests/gpu/recompiler/test_direct_descriptor_copy.cpp)
  target_link_libraries(test_direct_descriptor_copy PRIVATE prosper_core)
  add_test(NAME direct_descriptor_copy COMMAND test_direct_descriptor_copy)

  add_executable(test_image_descriptor_class_lookup
                 tests/gpu/recompiler/test_image_descriptor_class_lookup.cpp)
  target_link_libraries(test_image_descriptor_class_lookup PRIVATE prosper_core)
  add_test(NAME image_descriptor_class_lookup COMMAND test_image_descriptor_class_lookup)

  # Bounded graphics-shader cache: cached modules must match the direct recompiler byte-for-byte.
  add_executable(test_shader_recompile_cache tests/gpu/execute/test_shader_recompile_cache.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_shader_recompile_cache PRIVATE tests)
  target_link_libraries(test_shader_recompile_cache PRIVATE prosper_core)
  add_test(NAME shader_recompile_cache COMMAND test_shader_recompile_cache)

  # RDNA2 shader instruction-stream walker (first stage of the RDNA2->SPIR-V recompiler).
  add_executable(test_compressed_source_authority tests/gpu/resources/test_compressed_source_authority.cpp)
  target_link_libraries(test_compressed_source_authority PRIVATE prosper_core)
  add_test(NAME compressed_source_authority COMMAND test_compressed_source_authority)

  add_executable(test_metadata_kind_correlation tests/gpu/resources/test_metadata_kind_correlation.cpp)
  target_link_libraries(test_metadata_kind_correlation PRIVATE prosper_core)
  add_test(NAME metadata_kind_correlation COMMAND test_metadata_kind_correlation)

  add_executable(test_watch_list tests/gpu/diagnostics/test_watch_list.cpp)
  target_link_libraries(test_watch_list PRIVATE prosper_core)
  add_test(NAME watch_list COMMAND test_watch_list)

  # PROSPER_SKIP_DRAW_PROGRAM: decline graphics draws by shader PROGRAM identity (the graphics
  # counterpart of PROSPER_COMPUTE_SKIP_PROGRAM). Pure selector logic -- no Vulkan; the wiring into
  # the live renderer is pinned separately by draw_program_skip_render below.
  add_executable(test_draw_program_skip tests/gpu/diagnostics/test_draw_program_skip.cpp)
  target_link_libraries(test_draw_program_skip PRIVATE prosper_core)
  add_test(NAME draw_program_skip COMMAND test_draw_program_skip)

  # PROSPER_DRAW_LINKSCAN: the CPU-side census of the linked lists a draw's scalar buffers contain
  # (the graphics counterpart of PROSPER_COMPUTE_PARENTSCAN). Pure analysis + selector -- no Vulkan.
  add_executable(test_link_list_census tests/gpu/diagnostics/test_link_list_census.cpp)
  target_link_libraries(test_link_list_census PRIVATE prosper_core)
  add_test(NAME link_list_census COMMAND test_link_list_census)

  add_executable(test_shader_dump_filter tests/gpu/diagnostics/test_shader_dump_filter.cpp)
  target_link_libraries(test_shader_dump_filter PRIVATE prosper_core)
  add_test(NAME shader_dump_filter COMMAND test_shader_dump_filter)

  add_executable(test_rdna2_decode tests/gpu/recompiler/test_rdna2_decode.cpp)
  target_link_libraries(test_rdna2_decode PRIVATE prosper_core)
  add_test(NAME rdna2_decode_walk COMMAND test_rdna2_decode)

  # Structural RDNA2->SPIR-V validation that does not require Vulkan runtime support.
  add_executable(test_rdna2_spirv_struct tests/gpu/test_rdna2_spirv_struct.cpp)
  target_link_libraries(test_rdna2_spirv_struct PRIVATE prosper_core)
  add_test(NAME rdna2_spirv_struct COMMAND test_rdna2_spirv_struct)

  # #3203: the entry-M0 token containment's CFG-DISPATCHER re-arm. Its own binary rather than another
  # arm in test_rdna2_spirv_struct, because that one returns at its first failure and a mutation run
  # then cannot tell which arms moved. This reports every arm independently.
  add_executable(test_entry_m0_dispatcher tests/gpu/recompiler/test_entry_m0_dispatcher.cpp)
  target_link_libraries(test_entry_m0_dispatcher PRIVATE prosper_core)
  add_test(NAME entry_m0_dispatcher COMMAND test_entry_m0_dispatcher)

  # #3231: the CFG dispatcher's `missing-entry-vcc` gate and the entry-block-dominance case it may
  # now admit. Both directions, in its own binary for the same reason as the file above.
  add_executable(test_entry_vcc_dead tests/gpu/recompiler/test_entry_vcc_dead.cpp)
  target_link_libraries(test_entry_vcc_dead PRIVATE prosper_core)
  add_test(NAME entry_vcc_dead COMMAND test_entry_vcc_dead)

  # Scalar const-fold interpreter behind bindless-dynamic vertex-fetch resolution (no Vulkan needed).
  add_executable(test_dynfetch_fold tests/gpu/recompiler/test_dynfetch_fold.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_dynfetch_fold PRIVATE tests)
  target_link_libraries(test_dynfetch_fold PRIVATE prosper_core)
  add_test(NAME dynfetch_fold COMMAND test_dynfetch_fold)

  # #3137: the FORMAT-path MUBUF reject diagnostic must name which provenance step failed.
  add_executable(test_vertex_fetch_reject_diagnostic
                 tests/gpu/recompiler/test_vertex_fetch_reject_diagnostic.cpp)
  target_link_libraries(test_vertex_fetch_reject_diagnostic PRIVATE prosper_core)
  add_test(NAME vertex_fetch_reject_diagnostic COMMAND test_vertex_fetch_reject_diagnostic)

  # CPU-only streaming shader properties: exact legacy semantics without per-query vectors.
  add_executable(test_shader_property_scans tests/gpu/recompiler/test_shader_property_scans.cpp)
  target_link_libraries(test_shader_property_scans PRIVATE prosper_core)
  add_test(NAME shader_property_scans COMMAND ${CMAKE_COMMAND} -E env
           --unset=PROSPER_PROLOGLOG $<TARGET_FILE:test_shader_property_scans>)

  add_executable(test_gta5_selected_sbuffer tests/gpu/recompiler/test_gta5_selected_sbuffer.cpp)
  target_link_libraries(test_gta5_selected_sbuffer PRIVATE prosper_core)
  add_test(NAME gta5_selected_sbuffer COMMAND test_gta5_selected_sbuffer)

  add_executable(test_gta5_packed_pointer tests/gpu/test_gta5_packed_pointer.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gta5_packed_pointer PRIVATE tests)
  target_link_libraries(test_gta5_packed_pointer PRIVATE prosper_core)
  add_test(NAME gta5_packed_pointer COMMAND test_gta5_packed_pointer)

  add_executable(test_indirect_buffer_shadow tests/gpu/recompiler/test_indirect_buffer_shadow.cpp)
  target_link_libraries(test_indirect_buffer_shadow PRIVATE prosper_core)
  add_test(NAME indirect_buffer_shadow COMMAND test_indirect_buffer_shadow)

  add_executable(test_exec_population_count
    tests/gpu/test_exec_population_count.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_exec_population_count PRIVATE tests)
  target_link_libraries(test_exec_population_count PRIVATE prosper_core)
  add_test(NAME exec_population_count COMMAND test_exec_population_count)

  add_executable(test_indirect_pointer_static_footprint
    tests/gpu/recompiler/test_indirect_pointer_static_footprint.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_indirect_pointer_static_footprint PRIVATE tests)
  target_link_libraries(test_indirect_pointer_static_footprint PRIVATE prosper_core)
  add_test(NAME indirect_pointer_static_footprint
    COMMAND test_indirect_pointer_static_footprint)

  add_executable(test_indirect_pointer_descriptor_range
    tests/gpu/recompiler/test_indirect_pointer_descriptor_range.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_indirect_pointer_descriptor_range PRIVATE tests)
  target_link_libraries(test_indirect_pointer_descriptor_range PRIVATE prosper_core)
  add_test(NAME indirect_pointer_descriptor_range
    COMMAND test_indirect_pointer_descriptor_range)

  add_executable(test_gta5_cf9200 tests/gpu/recompiler/test_gta5_cf9200.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gta5_cf9200 PRIVATE tests)
  target_link_libraries(test_gta5_cf9200 PRIVATE prosper_core)
  add_test(NAME gta5_cf9200_no_backing COMMAND test_gta5_cf9200)

  add_executable(test_compute_parent_walk tests/gpu/diagnostics/test_compute_parent_walk.cpp)
  target_link_libraries(test_compute_parent_walk PRIVATE prosper_core)
  add_test(NAME compute_parent_walk COMMAND test_compute_parent_walk)

  add_executable(test_compute_tree_watch tests/gpu/diagnostics/test_compute_tree_watch.cpp)
  target_link_libraries(test_compute_tree_watch PRIVATE prosper_core)
  add_test(NAME compute_tree_watch COMMAND test_compute_tree_watch)

  # Cross-submit readability reuse is valid only while the HLE mapping generation is unchanged.
  add_executable(test_guest_memory_map tests/host/memory/test_guest_memory_map.cpp)
  target_link_libraries(test_guest_memory_map PRIVATE prosper_core)
  add_test(NAME guest_memory_map COMMAND test_guest_memory_map)

  # #3142: the chunked guest-memory content scan. The straddling case is the one that earns the
  # test -- a scanner that drops its overlap still returns a plausible hit list, so the blindness
  # is invisible in a live run and turns a "not found" into a wrong answer.
  add_executable(test_guest_memory_search tests/host/memory/test_guest_memory_search.cpp)
  target_link_libraries(test_guest_memory_search PRIVATE prosper_core)
  add_test(NAME guest_memory_search COMMAND test_guest_memory_search)

  add_executable(test_guest_read_cache_disabled tests/host/memory/test_guest_read_cache_disabled.cpp)
  target_link_libraries(test_guest_read_cache_disabled PRIVATE prosper_core)
  add_test(NAME guest_read_cache_disabled COMMAND test_guest_read_cache_disabled)
  # #2387: guest_writable's range cache. The first case is the one that matters -- a read-only
  # mapping, deliberately warmed into the READABLE cache first, must still be refused for write.
  # A shared cache (or a writable path that consulted the readable registry) passes every other
  # assertion here and fails only that one, silently corrupting guest memory in production.
  # POSIX-only: the test builds its own read-only / read-write / revoked mappings with mmap and
  # mprotect, which is the whole point -- it needs to CREATE a page whose permissions it controls,
  # not find one. The Windows equivalent is VirtualAlloc + VirtualProtect(PAGE_READONLY) and is
  # deliberately left to the lane that can run it (docs/THREAD_WAIT_PROFILING.md records the same
  # split for the profiling tools). The cache itself is built and exercised on all platforms; only
  # this test's mapping setup is POSIX.
  if(NOT WIN32)
    add_executable(test_guest_writable_cache tests/host/memory/test_guest_writable_cache.cpp)
    target_link_libraries(test_guest_writable_cache PRIVATE prosper_core)
    add_test(NAME guest_writable_cache COMMAND test_guest_writable_cache)
  endif()

  # #2393: the caches above are guarded ONLY by host::guest_mapping_generation(), so every change
  # to guest page protection must advance it -- including the software watchpoints in the Linux
  # fault handler, which flip a live guest page to PROT_READ and back. Linux-only: it drives the
  # real arm path (prosper_arm_label_watch_force) against a page it maps itself. Both production
  # sources are handed to the wiring check, which confirms no protection change in the handler
  # bypasses the notifying wrapper and that the notifier is still lock-free.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_watch_mapping_generation tests/host/memory/test_watch_mapping_generation.cpp)
    target_link_libraries(test_watch_mapping_generation PRIVATE prosper_core)
    target_compile_definitions(test_watch_mapping_generation PRIVATE
      PROSPER_EXEC_IMAGE_LINUX_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/image/exec_image_linux.cpp"
      PROSPER_GUEST_MEMORY_MAP_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/memory/guest_memory_map.hpp"
      PROSPER_GUEST_WRITE_WATCH_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/memory/guest_write_watch.cpp")
    add_test(NAME watch_mapping_generation COMMAND test_watch_mapping_generation)
  endif()

  if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_guest_write_watch tests/host/memory/test_guest_write_watch.cpp)
    target_link_libraries(test_guest_write_watch PRIVATE prosper_core)
    add_test(NAME guest_write_watch COMMAND test_guest_write_watch)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_dmem_write_trace tests/host/memory/test_dmem_write_trace.cpp)
    target_link_libraries(test_dmem_write_trace PRIVATE prosper_core)
    add_test(NAME dmem_write_trace COMMAND test_dmem_write_trace)
  endif()

  # #1932: SIGTRAP arbitration between PROSPER_HWBP/PROSPER_HWWATCH and PROSPER_BP. Linux-only --
  # the live arm opens a real per-task perf execute breakpoint and patches a real int3 into this
  # test's own hand-assembled stubs. Links nothing: src/host/fault/trap_arbitration.hpp is header-only,
  # so the decision table also builds where the handler does not. The source path is handed to the
  # wiring check, which confirms the production handler still calls the shared predicate.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_trap_arbitration tests/host/fault/test_trap_arbitration.cpp)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    target_include_directories(test_trap_arbitration PRIVATE src)
    target_compile_definitions(test_trap_arbitration PRIVATE
      PROSPER_EXEC_IMAGE_LINUX_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/image/exec_image_linux.cpp")
    add_test(NAME trap_arbitration COMMAND test_trap_arbitration)
  endif()

  # Unannounced-32-bit index-buffer fingerprint (#304 — DOLL's Slate/UMG quads). No Vulkan needed.
  add_executable(test_index_size_detect tests/gpu/execute/test_index_size_detect.cpp)
  target_link_libraries(test_index_size_detect PRIVATE prosper_core)
  add_test(NAME index_size_detect COMMAND test_index_size_detect)

  # shader_histo: data-driven recompiler-coverage report over the game's real embedded RDNA2 shaders.
  add_executable(shader_histo tools/shader_histo/shader_histo.cpp)
  target_link_libraries(shader_histo PRIVATE prosper_core)

  # shader_inspect: bounded offline decode/CFG listing for raw PROSPER_SHADER_DUMP binaries.
  add_executable(shader_inspect tools/shader_inspect/shader_inspect.cpp)
  target_link_libraries(shader_inspect PRIVATE prosper_core)

  # Resource-table honesty (#1571): a raw dump carries no descriptors, so a table-less stage
  # rejection must be reported as an UNDETERMINED tool limitation, never as a shader defect.
  if(Python3_Interpreter_FOUND)
    add_test(NAME shader_inspect_resource_table_honesty
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_shader_inspect.py"
                     "$<TARGET_FILE:shader_inspect>")
  endif()

  # spv_validate: emit a module from EVERY SPIR-V-producing entry point and run spirv-val on each —
  # a strict-validation gate (the render tests only prove llvmpipe accepts them; llvmpipe is
  # lenient). No Vulkan needed, but spirv-val (spirv-tools) IS required: it used to report PASS when
  # the validator was absent, which is how #1711 shipped invalid SPIR-V past a gate CLAUDE.md and
  # both READMEs describe as strict. The source root is passed so the tool can read the emitter
  # headers and fail when a new emitter is added without a module here.
  add_executable(spv_validate tools/spv_validate/spv_validate.cpp)
  target_link_libraries(spv_validate PRIVATE prosper_core)
  add_test(NAME spv_validate
           COMMAND spv_validate ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  # tools/vkprobe/shaders (#2945): the HAND-WRITTEN SPIR-V controls -- the arm that separated
  # "prosper's generated SPIR-V is malformed" from "the driver is wrong", by performing the same
  # storage-buffer loads from a module prosper did not emit. Their entire value is being valid by
  # construction, and this folder's recorded history is a control that reported confident numbers
  # from an INVALID pipeline for weeks because nothing checked it. So the check is a test, not a
  # README line: every .spvasm must assemble, land on SPIR-V 1.3, and pass spirv-val.
  #
  # Skipped, loudly, when the assembler is absent -- a validation gate that passes because its
  # validator is missing is exactly the #1711 failure spv_validate above exists to prevent, and a
  # silent skip is how instrument trap 126 reads a smaller suite as a green one.
  find_program(SPIRV_AS_EXECUTABLE NAMES spirv-as)
  find_program(SPIRV_VAL_EXECUTABLE NAMES spirv-val)
  if(Python3_Interpreter_FOUND AND SPIRV_AS_EXECUTABLE AND SPIRV_VAL_EXECUTABLE)
    add_test(NAME vkprobe_hand_written_controls
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/vkprobe/shaders/check_controls.py"
                     "${SPIRV_AS_EXECUTABLE}" "${SPIRV_VAL_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/vkprobe/shaders"
                     "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    message(STATUS "vkprobe_hand_written_controls NOT registered: "
                   "python3=${Python3_Interpreter_FOUND} spirv-as=${SPIRV_AS_EXECUTABLE} "
                   "spirv-val=${SPIRV_VAL_EXECUTABLE}")
  endif()

  # #2945: a recompiled VS must not declare more output components than the device can carry, and
  # the bound is what the FS actually reads. Carries its own mutation arm. Registered OUTSIDE the
  # Vulkan_FOUND block on purpose -- it inspects SPIR-V words and needs no device, so gating it on
  # Vulkan would silently delete the guard on a configuration without it.
  add_executable(test_vertex_output_budget tests/gpu/recompiler/test_vertex_output_budget.cpp)
  target_include_directories(test_vertex_output_budget PRIVATE tests)
  target_link_libraries(test_vertex_output_budget PRIVATE prosper_core)
  add_test(NAME vertex_output_budget COMMAND test_vertex_output_budget)

  # AGC→Vulkan resource layer: the front-half↔back-half contract (gpu_resources.hpp). Registry
  # behavior is fixed here so both halves integrate against a stable API. Pure/cross-platform.
  add_executable(test_gpu_resources tests/gpu/resources/test_gpu_resources.cpp)
  target_link_libraries(test_gpu_resources PRIVATE prosper_core)
  add_test(NAME gpu_resource_registry COMMAND test_gpu_resources)

  # Resource-binding contract (shader_resources.hpp): format sizing + the provenance/binding lookups
  # the recompiler and pipeline share. Pure/cross-platform. See docs/RESOURCE_BINDING.md.
  add_executable(test_shader_resources tests/gpu/resources/test_shader_resources.cpp)
  # Spells a `shared/...` include, which resolves from `frontends`.
  target_include_directories(test_shader_resources PRIVATE frontends)
  target_link_libraries(test_shader_resources PRIVATE prosper_core)
  add_test(NAME shader_resource_contract COMMAND test_shader_resources)

  # Versioned realized-draw capture: explicit round-trip, overlapping-resource alias preservation,
  # owned replay backing, and corrupt/truncated input rejection. Pure; no Vulkan or game dump.
  add_executable(test_gpu_capture tests/gpu/capture/test_gpu_capture.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_capture PRIVATE tests)
  target_link_libraries(test_gpu_capture PRIVATE prosper_core)
  add_test(NAME gpu_capture_roundtrip COMMAND test_gpu_capture)

  add_executable(test_gpu_capture_bundle tests/gpu/capture/test_gpu_capture_bundle.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_capture_bundle PRIVATE tests)
  target_link_libraries(test_gpu_capture_bundle PRIVATE prosper_core)
  add_test(NAME gpu_capture_bundle_roundtrip COMMAND test_gpu_capture_bundle)

  # Framerate must count DISTINCT guest frames, not publications: the renderer re-serves its retained
  # frame through the ordinary publish path, so a present-counting fps reads full speed for a frozen
  # title (#2783 / instrument trap 90). Pure -- no Vulkan, no dump.
  add_executable(test_present_frame_rate tests/gpu/present/test_present_frame_rate.cpp)
  target_link_libraries(test_present_frame_rate PRIVATE prosper_core)
  add_test(NAME present_frame_rate COMMAND test_present_frame_rate)

  # Native-speed append-only submit/present timeline: portable framing, checksums, and truncated-tail
  # recovery are pure and must not depend on Vulkan or a game dump.
  add_executable(test_gpu_timeline tests/gpu/timeline/test_gpu_timeline.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_timeline PRIVATE tests)
  target_link_libraries(test_gpu_timeline PRIVATE prosper_core)
  add_test(NAME gpu_timeline_roundtrip COMMAND test_gpu_timeline)

  add_executable(test_gpu_timeline_capture_exit tests/gpu/test_gpu_timeline_capture_exit.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_timeline_capture_exit PRIVATE tests)
  target_link_libraries(test_gpu_timeline_capture_exit PRIVATE prosper_core)
  add_test(NAME gpu_timeline_capture_exit COMMAND test_gpu_timeline_capture_exit)

  # Generic guest-log phase gate: exact bounded lines arm the existing seeded whole-frame capture
  # after one completed present. Pure/cross-platform; no game dump or Vulkan device.
  add_executable(test_guest_log_capture tests/gpu/test_guest_log_capture.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_guest_log_capture PRIVATE tests)
  target_link_libraries(test_guest_log_capture PRIVATE prosper_core)
  add_test(NAME guest_log_capture COMMAND test_guest_log_capture)

  # The same gate's SILENT failure: an armed marker that never matches must diagnose itself instead
  # of leaving a missing file as the only evidence (#1684). Separate binary because the gate is a
  # process-wide one-shot and guest_log_capture's marker deliberately fires.
  add_executable(test_guest_log_capture_miss tests/gpu/timeline/test_guest_log_capture_miss.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_guest_log_capture_miss PRIVATE tests)
  target_link_libraries(test_guest_log_capture_miss PRIVATE prosper_core)
  add_test(NAME guest_log_capture_miss COMMAND test_guest_log_capture_miss)

  # The same silence, two selector families over: a malformed capture tunable substituting a
  # different policy (#2565) and a detailed timeline selector that never matched (#2564). Separate
  # binary because both the detailed-capture request and the tunable refusal are process-wide
  # one-shots, and one of its arms deliberately ends a CHILD process during static initialization.
  add_executable(test_capture_tunables tests/gpu/timeline/test_capture_tunables.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_capture_tunables PRIVATE tests)
  target_link_libraries(test_capture_tunables PRIVATE prosper_core)
  add_test(NAME capture_tunables COMMAND test_capture_tunables)

  # Remote/headless F9 equivalent: an external regular-file witness arms the existing seeded
  # whole-frame capture on the next present. Pure/cross-platform; no game dump or Vulkan device.
  add_executable(test_capture_trigger_file tests/gpu/test_capture_trigger_file.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_capture_trigger_file PRIVATE tests)
  target_link_libraries(test_capture_trigger_file PRIVATE prosper_core)
  add_test(NAME capture_trigger_file COMMAND test_capture_trigger_file)

  # gpu_replay output dimensions follow the selected draw/prefix target when its byte count agrees.
  # Pure/cross-platform so focused replay tooling remains available in CI.
  add_executable(test_gpu_replay_extent tests/test_gpu_replay_extent.cpp)
  target_link_libraries(test_gpu_replay_extent PRIVATE prosper_core)
  add_test(NAME gpu_replay_output_extent COMMAND test_gpu_replay_extent)

  add_executable(test_gpu_replay_shader_dump tests/gpu/diagnostics/test_gpu_replay_shader_dump.cpp)
  target_link_libraries(test_gpu_replay_shader_dump PRIVATE prosper_core)
  add_test(NAME gpu_replay_shader_dump COMMAND test_gpu_replay_shader_dump)

  # Draw-scoped captured-resource replacement: strict semantic selector, exact span validation,
  # and copy-on-write isolation when draws share tables/backing. Pure/cross-platform; no Vulkan.
  add_executable(test_gpu_replay_resource_override
                 tests/test_gpu_replay_resource_override.cpp)
                 # Includes a shared fixture, which resolves from the `tests` root.
                 target_include_directories(test_gpu_replay_resource_override PRIVATE tests)
  target_link_libraries(test_gpu_replay_resource_override PRIVATE prosper_core)
  add_test(NAME gpu_replay_resource_override COMMAND test_gpu_replay_resource_override)

  # Selected-prefix compute output inspection must distinguish captured seed bytes from the
  # descriptor-visible post-dispatch image, including tiled 2D and true 3D storage layouts.
  add_executable(test_gpu_replay_post_compute_resource
                 tests/test_gpu_replay_post_compute_resource.cpp)
  target_link_libraries(test_gpu_replay_post_compute_resource PRIVATE prosper_core)
  add_test(NAME gpu_replay_post_compute_resource COMMAND test_gpu_replay_post_compute_resource)

  # Offline SPI_PS_INPUT_CNTL linkage classification: a pixel-shader input wired to the constant
  # form consumes no producer export, and must never be reported as a PARAM consumer.
  add_executable(test_gpu_replay_pixel_inputs tests/test_gpu_replay_pixel_inputs.cpp)
  target_link_libraries(test_gpu_replay_pixel_inputs PRIVATE prosper_core)
  add_test(NAME gpu_replay_pixel_inputs COMMAND test_gpu_replay_pixel_inputs)

  add_executable(test_gpu_dependency_graph tests/gpu/execute/test_gpu_dependency_graph.cpp)
  target_link_libraries(test_gpu_dependency_graph PRIVATE prosper_core)
  add_test(NAME gpu_dependency_graph COMMAND test_gpu_dependency_graph)

  add_executable(gpu_timeline tools/gpu_timeline/gpu_timeline.cpp)
  target_link_libraries(gpu_timeline PRIVATE prosper_core)

  # Front-half resource-table builder + V# descriptor decode (agc_shader_layout.cpp): shader user_data
  # + user-data SGPRs -> ShaderResourceTable (constant buffers, stage 1). Pure/cross-platform.
  add_executable(test_build_shader_resources tests/gpu/agc/test_build_shader_resources.cpp)
  target_link_libraries(test_build_shader_resources PRIVATE prosper_core)
  add_test(NAME build_shader_resources COMMAND test_build_shader_resources)

  # GPU descriptor-bound INVARIANT guard (agc_shader_layout.cpp decode_image/buffer_descriptor): the
  # entire guest-input memory-safety argument relies on width/height <= 16384, depth <= 8192, and V#
  # size_bytes clamped to 32 bits. Pins those exact ceilings so a mask-widening / +1-bias refactor fails
  # here first instead of silently un-bounding every downstream size product. Pure decode; no dump/Vulkan.
  add_executable(test_descriptor_clamp tests/gpu/agc/test_descriptor_clamp.cpp)
  target_link_libraries(test_descriptor_clamp PRIVATE prosper_core)
  add_test(NAME descriptor_clamp COMMAND test_descriptor_clamp)

  # #3205: mip-chain provenance participates in image IDENTITY only when a chain exists. The fields
  # are populated on one construction path and left unset on another, so comparing them
  # unconditionally split identical single-level descriptors into separate Vulkan images -- which
  # dropped three of four quadrant stores in GTA V and rendered a checkerboard. Pure predicate.
  add_executable(test_mip_provenance_identity tests/gpu/agc/test_mip_provenance_identity.cpp)
  target_link_libraries(test_mip_provenance_identity PRIVATE prosper_core)
  add_test(NAME mip_provenance_identity COMMAND test_mip_provenance_identity)

  # #3204: the image-identity predicates extracted from execute_item. They decide whether two
  # bindings alias ONE Vulkan image -- the aliasing GTA V's four quadrant writes depend on, and
  # the thing #3205 broke while it was an unnamed local expression. Pure comparisons.
  add_executable(test_image_identity tests/gpu/agc/test_image_identity.cpp)
  target_link_libraries(test_image_identity PRIVATE prosper_core)
  add_test(NAME image_identity COMMAND test_image_identity)

  # #3195: the atomic-image LOGICAL/PHYSICAL staging extents, extracted from execute_item.
  add_executable(test_atomic_image_staging tests/gpu/agc/test_atomic_image_staging.cpp)
  target_link_libraries(test_atomic_image_staging PRIVATE prosper_core)
  add_test(NAME atomic_image_staging COMMAND test_atomic_image_staging)

  # #3204: SPIR-V/guest storage-format agreement, extracted from execute_item. The permissive
  # direction binds the guest image with the wrong element type, so every arm has a rejection
  # partner -- including atomics, which are never a native storage image however well formats match.
  add_executable(test_spirv_storage_match tests/gpu/agc/test_spirv_storage_match.cpp)
  target_link_libraries(test_spirv_storage_match PRIVATE prosper_core)
  add_test(NAME spirv_storage_match COMMAND test_spirv_storage_match)

  # Descriptor-binding policy (#157): textures/storage images never land on the recompiler's hardwired
  # cbuf bindings 2/3; constant/vertex buffers assigned first. Pure, no game dump / Vulkan needed.
  add_executable(test_convention_bindings tests/gpu/test_convention_bindings.cpp)
  target_link_libraries(test_convention_bindings PRIVATE prosper_core)
  add_test(NAME convention_bindings COMMAND test_convention_bindings)

  add_executable(test_writer_provenance tests/gpu/capture/test_writer_provenance.cpp)
  target_link_libraries(test_writer_provenance PRIVATE prosper_core)
  add_test(NAME writer_provenance_ranges COMMAND test_writer_provenance)

  # The init-fault report must survive a wild/null faulting RIP (#128). On Windows this also drives
  # the VEH recovery thunk and asserts its compiled-helper call obeys the Microsoft-x64 ABI (#633).
  add_executable(test_initfault_dump tests/host/image/test_initfault_dump.cpp)
  target_link_libraries(test_initfault_dump PRIVATE prosper_core)
  add_test(NAME initfault_dump_survives COMMAND test_initfault_dump)

  # Generated guest->host import boundary: preserve all register args, stack args 7-9, return value,
  # and ABI stack alignment. Windows exercises SysV->Microsoft-x64; Linux also drives the FS-swap path.
  add_executable(test_hle_stack_args tests/host/image/test_hle_stack_args.cpp)
  target_link_libraries(test_hle_stack_args PRIVATE prosper_core)
  add_test(NAME hle_stack_args COMMAND test_hle_stack_args)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME hle_stack_args_guest_fs COMMAND test_hle_stack_args --guest-fs)
  endif()

  # #2955: the guest(SysV) -> host(Microsoft x64) import bridge. CROSS-PLATFORM on purpose, and that
  # is the point rather than an accident: the bridge's placement rules are a pure function of a
  # handler's signature, so they can be asserted anywhere, and on any x86-64 host the emitted bytes
  # can additionally be EXECUTED across the SysV/Microsoft boundary via __attribute__((ms_abi)).
  # Gating this on WIN32 would leave the one defect it exists to catch visible only on the platform
  # nobody develops on. Its non-x86-64 arm still checks the tables.
  add_executable(test_sysv_ms_bridge tests/host/abi/test_sysv_ms_bridge.cpp)
  target_link_libraries(test_sysv_ms_bridge PRIVATE prosper_core)
  add_test(NAME sysv_ms_bridge COMMAND test_sysv_ms_bridge)

  # #3246: the one argument list #2955's CallSignature cannot describe — a real C variadic, whose
  # shape the format string decides at run time. CROSS-PLATFORM for the same reason as the bridge
  # above, and then some: reading a guest System V va_list and writing a Microsoft one are pure
  # functions, and BOTH conventions' va_arg are available on any x86-64 host through
  # __builtin_sysv_va_list / __builtin_ms_va_list — so the round trip is executed against the
  # compiler's own readers here, and against the real CRT on Windows.
  add_executable(test_guest_varargs tests/host/abi/test_guest_varargs.cpp)
  target_link_libraries(test_guest_varargs PRIVATE prosper_core)
  add_test(NAME guest_varargs COMMAND test_guest_varargs)

  # sceImeUpdate must invoke the guest key handler on the CALLER's guest %fs, not the host %fs the
  # import stub swapped in (#1286: Evergate SIGSEGV at eboot+0xaf8431 on a key press). Drives the real
  # import path on an activated guest TCB; a no-op pass off Linux.
  add_executable(test_ime_fs tests/hle/input/test_ime_fs.cpp)
  target_link_libraries(test_ime_fs PRIVATE prosper_core)
  add_test(NAME ime_fs COMMAND test_ime_fs)

  # x86-64 low-address READ decoder used by the Windows PROSPER_NULL_PAGE emulation. Header-only, so the
  # decode logic is verified on every platform even though the VEH caller is Windows-only.
  add_executable(test_x86_read_decode tests/host/x86/test_x86_read_decode.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_x86_read_decode PRIVATE src)
  add_test(NAME x86_read_decode COMMAND test_x86_read_decode)

  # #2018: a fatal worker-thread fault report must describe one fault. Header-only snapshot of the
  # handler's own siginfo/ucontext; POSIX signal contexts only (the test self-skips elsewhere).
  add_executable(test_fault_context tests/host/fault/test_fault_context.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_fault_context PRIVATE src)
  # The contention arm spawns real threads. This target links no library, so unlike every other
  # thread-using test here it inherits -pthread from nothing; name Threads::Threads explicitly
  # (found above with prosper_core, whose existence gates this block).
  target_link_libraries(test_fault_context PRIVATE Threads::Threads)
  add_test(NAME fault_context COMMAND test_fault_context)

  # End-to-end AGC submit plus fixed args 7-9 in ReleaseMem/WaitRegMem. This is cross-platform so
  # Windows validates the real fence builders behind the generated import-ABI conformance test.
  add_executable(test_agc_dcb_callback_abi tests/hle/test_agc_dcb_callback_abi.cpp)
  target_link_libraries(test_agc_dcb_callback_abi PRIVATE prosper_core)
  add_test(NAME agc_dcb_callback_abi COMMAND test_agc_dcb_callback_abi)

  add_executable(test_agc_submit tests/gpu/pm4/test_agc_submit.cpp)
  target_link_libraries(test_agc_submit PRIVATE prosper_core)
  add_test(NAME agc_submit_to_gpustate COMMAND test_agc_submit)

  # libSceVideoOut: query/config functions return real 1080p60 values + swapchain-scaffolding
  # display-buffer registry records the game's registered framebuffers.
  #
  # CROSS-PLATFORM on purpose (#1765). This sat inside the if(UNIX) block below, so the one platform
  # whose display pacing is at risk was the one platform that did not build it: sceVideoOutWaitVblank
  # waits on a deadline, and on Windows that wait is quantized by the process timer period. Nothing
  # here is POSIX-specific -- it drives the HLE registry and asserts returned values -- so the gate
  # was incidental rather than a statement about the test.
  add_executable(test_videoout tests/host/platform/test_videoout.cpp)
  target_link_libraries(test_videoout PRIVATE prosper_core)
  add_test(NAME videoout_queries COMMAND test_videoout)

  # #3017: the display prosper advertises must be DERIVED from the host's real mode rather than
  # hardcoded 1080p59.94. Its own executable, not an arm of test_videoout, because the advertised
  # mode resolves ONCE per process on first use -- so the derived case needs a process whose
  # environment is set before any VideoOut call, and test_videoout above is exactly the
  # complementary process where those variables are unset and the untouched `legacy` default is
  # asserted. CROSS-PLATFORM for the same reason test_videoout is: it drives the HLE registry and
  # a pure header, with no OS display access.
  add_executable(test_display_mode tests/hle/graphics/test_display_mode.cpp)
  target_link_libraries(test_display_mode PRIVATE prosper_core)
  add_test(NAME display_mode_derivation COMMAND test_display_mode)

  # precise_sleep's deadline post-condition (#3074): every sleep_until_steady_ns() backend must
  # never return before now_ns() >= deadline_ns. Win32SleepFallback did not re-check after its one
  # OS sleep call, unlike the sibling high-resolution-timer path. CROSS-PLATFORM like test_videoout
  # above and for the same reason: sleep_until_deadline_retry() is the pure-function loop the fix
  # composes, deliberately extracted so its termination condition is testable with a fake clock and
  # a fake sleep on any host, without executing the Windows-only code path or measuring wall time.
  add_executable(test_precise_sleep tests/host/platform/test_precise_sleep.cpp)
  target_link_libraries(test_precise_sleep PRIVATE prosper_core)
  add_test(NAME precise_sleep_deadline_retry COMMAND test_precise_sleep)

  # Windows guest-execution substrate (exec_image_win.cpp): build boot_trace so the whole boot path
  # links (proves the substrate + SysV-ABI HLE boundary resolve). No evdev/Vulkan here. The guest boot
  # itself is not yet verified on Windows — see docs/PORTING.md "Windows".
  if(WIN32)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp tools/boot_trace/first_frame_capture.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(boot_trace PRIVATE tests)
    # Spells a `shared/...` include, which resolves from `frontends`.
    target_include_directories(boot_trace PRIVATE frontends)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    # Live Vulkan renderer on Windows (PROSPER_RENDER=1): with the Vulkan SDK present, wire the same
    # shared DrawItem->Vulkan compositor boot_trace uses on Linux so the guest's SubmitDcb draws
    # execute + present + dump frames. Mirrors the UNIX block below. `gpu_replay` is built here too
    # since #2331: the F9/scheduled bundle grab already worked on Windows, so a Windows investigation
    # could produce a several-hundred-MiB .prgbundle and then have no way to open it -- the documented
    # "capture, then replay offline" loop was half-available, and the half that was missing was the
    # half that answers the question. See docs/PORTING.md "Windows".
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)
      add_library(prosper_live_renderer STATIC frontends/shared/live/live_renderer.cpp
                                               frontends/shared/live/live_compute.cpp
                                               frontends/shared/present/present_blit.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()

      # #3204: the renderer-target format table, extracted from execute_item where it was an
      # `||` chain -- the shape that cost Dead Cells an HDR target (#773) and Syberia its 3D
      # menu. Now an exhaustive switch: the compiler pins exhaustiveness, this test pins the
      # table. Registered HERE, inside if(Vulkan_FOUND), because live_target_format.hpp includes
      # <vulkan/vulkan.h> -- registering it outside built on Linux and broke MinGW and macOS.
      add_executable(test_live_target_format_match tests/gpu/test_live_target_format_match.cpp)
      target_link_libraries(test_live_target_format_match PRIVATE prosper_core Vulkan::Vulkan)
      target_include_directories(test_live_target_format_match PRIVATE frontends src)
      add_test(NAME live_target_format_match COMMAND test_live_target_format_match)
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # Keep the production compute backend covered on native Windows too. This catches host- and
      # driver-specific image upload/writeback failures in the same backend prosper-app uses.
      add_executable(test_game_compute tests/gpu/recompiler/test_game_compute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_game_compute PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_game_compute PRIVATE frontends)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")
      # The compute path's full-surface seed/pack intermediates are pooled rather than freshly
      # allocated. Their old allocation was `make_unique_for_overwrite`, i.e. nominally
      # indeterminate -- but at these sizes it was a fresh mmap, which reads as ZERO, so a call site
      # could be depending on zeroing it never asked for. This arm hands every lease 0xCD instead:
      # it fails exactly when such a dependency exists, which is the only way that claim is testable
      # against the real backend rather than argued from the diff.
      add_test(NAME game_compute_exec_scratch_poison COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_scratch_poison PROPERTIES
        ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

      # Local-only realized-submit replayer (#514), on Windows since #2331. Same target as the UNIX
      # block below, including the -Werror=switch gate: gpu_replay maps LiveTargetPixelFormat itself
      # in inspect_live_target, and an unhandled member there falls through to the seed's Rgba8Unorm
      # default and makes --inspect-only mislabel the seed -- sending an investigation down the wrong
      # path using its own instrument. That failure mode is not platform-specific, so neither is the
      # gate.
      add_executable(gpu_replay tools/gpu_replay/gpu_replay.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(gpu_replay PRIVATE frontends)
      target_include_directories(gpu_replay PRIVATE src tests)
      target_link_libraries(gpu_replay PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -Werror=format for the same reason as -Werror=switch above, and found the same way: by a
        # defect this tool's own diagnostics hid. gpu_replay prints index values with %zu, and #2836
        # gave three of those printf arguments a distinct type (ItemIndex) while the format stayed
        # %zu. Varargs performs no conversion, so the type system has nothing to check, and this
        # target compiled with neither -Wall nor -Wformat -- the mismatch was invisible in the build
        # and was caught only in review. #2840 will type the operation indices too, and gpu_replay.cpp
        # prints those with %zu at roughly sixteen more sites, so the class recurs the moment that
        # lands. Measured clean under -Wformat=2 before this gate was added.
        #
        # THIS GATE BINDS ON GCC ONLY, despite the GNU|Clang guard above. Clang deliberately looks
        # through a scoped enum to its underlying type here and stays silent at -Wformat,
        # -Werror=format, -Wformat=2 AND -Wall -Wextra; only -Wformat-pedantic makes it fire. The
        # same positive control that errors under g++ exits 0 under clang++. It still protects every
        # build this project actually performs -- both the Linux and the MinGW Windows legs are GCC,
        # and the control was verified to fire on both -- but do not read the guard as Clang cover.
        # -Wformat-pedantic on a Clang-only branch was measured clean on this TU if that is ever
        # wanted; it is left out because no build here uses Clang and an unrecognized option is its
        # own hazard. Deliberately NOT widened to -Wformat=2: GCC already diagnoses this class at
        # plain -Wformat, so =2 adds only unrelated surface, and it does not even mean the same thing
        # to the two compilers (GCC gives -Wformat-nonliteral where Clang gives
        # -Wmissing-format-attribute, which GCC lacks entirely).
        target_compile_options(gpu_replay PRIVATE -Werror=switch -Werror=format)
      endif()
    else()
      message(STATUS "Vulkan not found — Windows boot_trace built without the live renderer")
    endif()
  endif()

  # M2+: real memory mapping, trap, boot, and the boot-trace debug tool (Linux + macOS/Rosetta).
  if(UNIX)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp tools/boot_trace/first_frame_capture.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(boot_trace PRIVATE tests)
    # Spells a `shared/...` include, which resolves from `frontends`.
    target_include_directories(boot_trace PRIVATE frontends)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    if(NOT APPLE)
      # Zero-dependency Linux evdev gamepad frontend (outside prosper_core). Always available on Linux;
      # boot_trace installs it (gated by PROSPER_PAD) when the SDL3 gamepad frontend isn't built.
      add_library(prosper_pad_evdev STATIC frontends/pad_evdev/pad_evdev.cpp)
      target_include_directories(prosper_pad_evdev PUBLIC frontends/pad_evdev src)
      target_link_libraries(prosper_pad_evdev PUBLIC prosper_core)
      target_link_libraries(boot_trace PRIVATE prosper_pad_evdev)
      target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_EVDEV=1)
    endif()

    add_executable(imgdump tools/imgdump/imgdump.cpp)
    target_link_libraries(imgdump PRIVATE prosper_core)

    add_executable(test_trap_linux tests/misc/test_trap_linux.cpp)
    target_link_libraries(test_trap_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME trap_identifies_imports
               COMMAND test_trap_linux "${GAME_DUMP}")
    endif()

    add_executable(test_boot_linux tests/misc/test_boot_linux.cpp)
    target_link_libraries(test_boot_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      # Messenger-specific: links THIS title's own Il2cppUserAssemblies.prx / PS5Util.prx (#1573).
      prosper_add_title_gated_test(boot_reaches_first_syscall PPSA24651
               test_boot_linux "${GAME_DUMP}")
    endif()

    if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      # Real runtime PRX loading (#639). Hermetic: the test writes its own synthetic PRX outside
      # Media/Plugins, so it needs no game dump — but it maps that module at the fixed runtime-module
      # base and EXECUTES its module_start, whose bytes the test emits by hand, so it is host-x86-64
      # only. Its scratch root goes in the build tree, never /tmp — no path is passed here, so the
      # test derives one per process from tests/fixtures/test_scratch.h. Passing a fixed
      # "${CMAKE_CURRENT_BINARY_DIR}/runtime-prx-test-root" is what #2582 was about: two ctest runs
      # against one build directory then shared the .prx they each write and map.
      add_executable(test_runtime_prx_load tests/host/image/test_runtime_prx_load.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_runtime_prx_load PRIVATE tests)
      target_link_libraries(test_runtime_prx_load PRIVATE prosper_core)
      add_test(NAME runtime_prx_load COMMAND test_runtime_prx_load)
    endif()

    # Regression test for the real setjmp/longjmp impl (x86-64 asm; used by the GC's root scan).
    add_executable(test_setjmp tests/test_setjmp.cpp)
    target_link_libraries(test_setjmp PRIVATE prosper_core)
    add_test(NAME setjmp_saves_and_restores COMMAND test_setjmp)

    add_executable(test_stack_registry tests/host/image/test_stack_registry.cpp)
    target_link_libraries(test_stack_registry PRIVATE prosper_core)
    add_test(NAME thread_stack_registry COMMAND test_stack_registry)

    add_executable(test_rwlock_once tests/hle/test_rwlock_once.cpp)
    target_link_libraries(test_rwlock_once PRIVATE prosper_core)
    add_test(NAME rwlock_once_semantics COMMAND test_rwlock_once)

    # #2042: a destroyed guest sync object's storage is quarantined, not returned to the allocator,
    # so a Destroy racing a handler cannot corrupt prosper's own heap. The assertions hold on every
    # allocator, so the test itself runs everywhere.
    add_executable(test_sync_destroy_lifetime tests/hle/sync/test_sync_destroy_lifetime.cpp)
    target_link_libraries(test_sync_destroy_lifetime PRIVATE prosper_core)
    add_test(NAME sync_destroy_lifetime COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime PROPERTIES TIMEOUT 120)

    # The window switch must FAIL SAFE. `atof` returns 0.0 on no conversion, so an early version of
    # retire_window_seconds() turned the guard off for `=on`, `=true` or any typo — the exact
    # use-after-free this exists to prevent, reached by misspelling a tuning knob. This arm passes an
    # unparseable value and requires the suite to still PASS, i.e. the default window is in force.
    # It asserts the green outcome, so unlike the counter-arm it is allocator-independent.
    add_test(NAME sync_destroy_lifetime_bad_window COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime_bad_window PROPERTIES
        ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=on" TIMEOUT 120)

    # The counter-arm runs the SAME binary with the quarantine switched off
    # (PROSPER_SYNC_RETIRE_SECONDS=0 is the pre-#2042 immediate free) and requires it to FAIL, so CI
    # keeps proving the assertions can move without anyone rebuilding against an older master.
    #
    # LINUX ONLY, and the restriction is the honest one rather than a convenience: the arm is red
    # because glibc's tcache hands a just-freed block of the same size back to the very next
    # allocation, deterministically and LIFO. That reuse is the mechanism being demonstrated, not an
    # incidental detail — on an allocator that does not reuse promptly the assertions would pass and
    # WILL_FAIL would turn a correct result into a red build. Asserting an allocator's internals as
    # a merge gate on hosts where they were never measured is how a guard starts failing for reasons
    # unrelated to what it guards.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_test(NAME sync_destroy_lifetime_counter_arm COMMAND test_sync_destroy_lifetime)
      set_tests_properties(sync_destroy_lifetime_counter_arm PROPERTIES
          ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=0" WILL_FAIL TRUE TIMEOUT 120)
    endif()
    # #2178: every fallible scePthread* spelling reports the libkernel-encoded error while the POSIX
    # spelling on the same body keeps the bare FreeBSD errno. Asserts both halves per entry point —
    # a single-spelling arm cannot see a missing (or a wrongly added) SCE_PTHREAD_ALIAS.
    add_executable(test_pthread_error_encoding tests/hle/kernel/test_pthread_error_encoding.cpp)
    target_link_libraries(test_pthread_error_encoding PRIVATE prosper_core)
    add_test(NAME pthread_error_encoding COMMAND test_pthread_error_encoding)

    # #2596: the C11 _Mtx_* / _Cnd_* handlers must resolve a guest sync slot the way the libkernel
    # entry point they forward to on hardware resolves it. Asserts what the handler DID (blocked,
    # excluded a second thread, self-initialised the slot), never what it returned -- _Cnd_wait's
    # unconditional 0 is the shipped contract. TIMEOUT is generous because every arm is a bounded
    # spin: a regression here is a thread that never wakes, and each arm reports it as a failure.
    add_executable(test_c11_thread_slots tests/hle/test_c11_thread_slots.cpp)
    target_link_libraries(test_c11_thread_slots PRIVATE prosper_core)
    add_test(NAME c11_thread_slots COMMAND test_c11_thread_slots)
    set_tests_properties(c11_thread_slots PROPERTIES TIMEOUT 120)

    # #2619 / #2623: sharing the slot RESOLUTION was not the same as sharing the per-object
    # BOOKKEEPING. Guards the claim-and-poison of a destroyed slot, the missed-wakeup generation,
    # and the cond-destroy waiter count -- through BOTH spellings, since a divergence between them
    # is the defect. Deliberately separate from c11_thread_slots: every arm there passes with the
    # bookkeeping absent. Same generous TIMEOUT, same reason -- one arm parks a real thread.
    add_executable(test_c11_sync_bookkeeping tests/hle/sync/test_c11_sync_bookkeeping.cpp)
    target_link_libraries(test_c11_sync_bookkeeping PRIVATE prosper_core)
    add_test(NAME c11_sync_bookkeeping COMMAND test_c11_sync_bookkeeping)
    set_tests_properties(c11_sync_bookkeeping PROPERTIES TIMEOUT 120)

    # __tls_get_addr per-thread DTV is purged (blocks freed) on both thread-exit paths (#68), and
    # normal return leaves guest %fs before glibc resumes host pthread cleanup (#644).
    add_executable(test_tls_dtv_purge tests/hle/test_tls_dtv_purge.cpp)
    target_link_libraries(test_tls_dtv_purge PRIVATE prosper_core)
    add_test(NAME tls_dtv_purge_on_thread_exit COMMAND test_tls_dtv_purge)

    # #2613: nothing joins guest threads at shutdown, so thread_trampoline's tail runs while the
    # process is already inside __run_exit_handlers and touches the thread-name / thread-stack / DTV
    # registries. They must be process-lifetime (src/host/platform/immortal.hpp), i.e. register no static
    # destructor. The test parks a guest worker and drives the whole tail from an exit-time probe, so
    # the post-destruction access happens on EVERY run instead of 1-in-300 — and the probe checks the
    # two library canaries first, so a broken ordering assumption fails instead of passing vacuously.
    add_executable(test_exit_registry_lifetime tests/host/image/test_exit_registry_lifetime.cpp)
    target_link_libraries(test_exit_registry_lifetime PRIVATE prosper_core)
    add_test(NAME exit_registry_lifetime COMMAND test_exit_registry_lifetime)
    set_tests_properties(exit_registry_lifetime PROPERTIES TIMEOUT 120)

    add_executable(test_agc_dcb tests/gpu/pm4/test_agc_dcb.cpp)
    target_link_libraries(test_agc_dcb PRIVATE prosper_core)
    add_test(NAME agc_dcb_pm4 COMMAND test_agc_dcb)

    # #1143: each sceAgc*GetSize must equal its builder's emitted dword count — drift guard.
    add_executable(test_agc_getsize tests/hle/test_agc_getsize.cpp)
    target_link_libraries(test_agc_getsize PRIVATE prosper_core)
    add_test(NAME agc_getsize_drift COMMAND test_agc_getsize)

    # #1748: a builder must not emit MORE dwords than the hardware packet it stands for — the guest
    # reserves its command buffer from the real AGC sizes, so an oversized packet overruns it.
    add_executable(test_agc_epilogue_window tests/gpu/pm4/test_agc_epilogue_window.cpp)
    target_link_libraries(test_agc_epilogue_window PRIVATE prosper_core)
    add_test(NAME agc_epilogue_window COMMAND test_agc_epilogue_window)

    # libSceVideoOut flip->present: SubmitFlip scans out the registered buffer; present_readback
    # returns the presented frame's real pixels. The surface the renderer presents frames to.
    add_executable(test_present tests/misc/test_present.cpp)
    target_link_libraries(test_present PRIVATE prosper_core)
    add_test(NAME videoout_present COMMAND test_present)

    # Graphics-phase foundation + agentic-first verification harness: headless offscreen Vulkan
    # (render/clear -> readback -> pixel/CRC assert). Only built if Vulkan dev files are present.
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      # Live game rendering: with Vulkan available, boot_trace can register the live submit renderer
      # (PROSPER_RENDER=1) so the game's SubmitDcb draws execute + present + dump frames to disk.
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)

      # Shared live renderer (frontends/shared): the DrawItem->Vulkan compositor, factored out of
      # boot_trace so BOTH boot_trace and prosper-app register the SAME renderer instead of
      # duplicating it. Needs render_runner.h (tests/) and refvs.inc (tools/boot_trace/).
      add_library(prosper_live_renderer STATIC frontends/shared/live/live_renderer.cpp
                                               frontends/shared/live/live_compute.cpp
                                               frontends/shared/present/present_blit.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # vkprobe (#2945/#2937): a CONTROL that AVOIDS prosper's host-side Vulkan usage rather than
      # reproducing it, so a failure here shows that usage is not NECESSARY for the defect -- never
      # that it is exonerated. Drives a dumped SPIR-V pair through a bare Vulkan pipeline with no
      # prosper code in the process. It does not clear the recompiler (the SPIR-V is prosper's own
      # output) and it does not clear the driver -- run enough times it reproduces #2945 itself.
      # Read tools/vkprobe/README.md before quoting a clean run. Deliberately links nothing from
      # prosper.
      add_executable(vkprobe tools/vkprobe/vkprobe.cpp)
      target_link_libraries(vkprobe PRIVATE Vulkan::Vulkan)

      # vkval_sync_probe (#3248): the positive control for `vk_validation_scan.py --sync`. It
      # commits one deliberate write-after-write so the scanner can prove SYNCHRONIZATION validation
      # is armed before reporting a sync verdict -- loading the layer proves only that CORE checks
      # run, and a sync setting the layer version does not recognise produces exactly the output of
      # a clean sync run. Deliberately NOT registered as a ctest: it exists to be invalid, and the
      # gate it feeds fails on any hazard the ledger does not carry.
      add_executable(vkval_sync_probe tools/vkval/sync_probe.cpp)
      target_link_libraries(vkval_sync_probe PRIVATE Vulkan::Vulkan)

      # Local-only realized-submit replayer (#514): no guest boot, reads a versioned capture whose game
      # data remains gitignored, renders it through the same live backend, and checks the captured hash.
      add_executable(gpu_replay tools/gpu_replay/gpu_replay.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(gpu_replay PRIVATE frontends)
      target_include_directories(gpu_replay PRIVATE src tests)
      target_link_libraries(gpu_replay PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      # gpu_replay maps LiveTargetPixelFormat itself (inspect_live_target), and it is the tool an
      # investigation reaches for FIRST. An unhandled member there falls through to the
      # GpuCaptureRttSeed::format default (Rgba8Unorm) and makes --inspect-only mislabel the seed,
      # sending the next investigation down the wrong path using its own instrument. Same gate as
      # prosper_live_renderer so the exhaustiveness contract holds at every mapping, not just the
      # ones inside frontends/shared/live/live_target_format.hpp.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -Werror=format for the same reason as -Werror=switch above, and found the same way: by a
        # defect this tool's own diagnostics hid. gpu_replay prints index values with %zu, and #2836
        # gave three of those printf arguments a distinct type (ItemIndex) while the format stayed
        # %zu. Varargs performs no conversion, so the type system has nothing to check, and this
        # target compiled with neither -Wall nor -Wformat -- the mismatch was invisible in the build
        # and was caught only in review. #2840 will type the operation indices too, and gpu_replay.cpp
        # prints those with %zu at roughly sixteen more sites, so the class recurs the moment that
        # lands. Measured clean under -Wformat=2 before this gate was added.
        #
        # THIS GATE BINDS ON GCC ONLY, despite the GNU|Clang guard above. Clang deliberately looks
        # through a scoped enum to its underlying type here and stays silent at -Wformat,
        # -Werror=format, -Wformat=2 AND -Wall -Wextra; only -Wformat-pedantic makes it fire. The
        # same positive control that errors under g++ exits 0 under clang++. It still protects every
        # build this project actually performs -- both the Linux and the MinGW Windows legs are GCC,
        # and the control was verified to fire on both -- but do not read the guard as Clang cover.
        # -Wformat-pedantic on a Clang-only branch was measured clean on this TU if that is ever
        # wanted; it is left out because no build here uses Clang and an unrecognized option is its
        # own hazard. Deliberately NOT widened to -Wformat=2: GCC already diagnoses this class at
        # plain -Wformat, so =2 adds only unrelated surface, and it does not even mean the same thing
        # to the two compilers (GCC gives -Wformat-nonliteral where Clang gives
        # -Wmissing-format-attribute, which GCC lacks entirely).
        target_compile_options(gpu_replay PRIVATE -Werror=switch -Werror=format)
      endif()

      # Seed flags are renderer-registration inputs. Exercise the real gpu_replay process so an
      # early --recompile-raw registration cannot pass by proving substitution but failing restore.
      add_executable(test_gpu_replay_seed_order tests/gpu/capture/test_gpu_replay_seed_order.cpp)
      target_include_directories(test_gpu_replay_seed_order PRIVATE src tests)
      target_link_libraries(test_gpu_replay_seed_order PRIVATE prosper_core)
      add_test(NAME gpu_replay_seeded_raw_execution
        COMMAND test_gpu_replay_seed_order $<TARGET_FILE:gpu_replay>)

      # Exercise the real metadata-only CLI path for v46's selected raw resource-input witness.
      # Unit tests cover verdict policy; this process test keeps inspect formatting from silently
      # disappearing or collapsing full-match/mismatch/unavailable into one output state.
      add_test(NAME gpu_replay_resource_input_inspect
        COMMAND test_gpu_capture --inspect-resource-input $<TARGET_FILE:gpu_replay>)

      # Capture -> serialize -> owned replay backing -> SAME live renderer -> pixels. This specifically
      # proves replay textures do not dereference the captured guest VA and still sample exact bytes.
      add_executable(test_gpu_capture_render tests/misc/test_gpu_capture_render.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_gpu_capture_render PRIVATE frontends)
      target_include_directories(test_gpu_capture_render PRIVATE src tests tools/boot_trace)
      target_link_libraries(test_gpu_capture_render PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME gpu_capture_render_replay COMMAND test_gpu_capture_render)

      # PROSPER_SKIP_DRAW_PROGRAM through the REAL live renderer: an unnamed program renders, the
      # named one is withheld, and a mixed submit loses only the named draw. Needs the live renderer
      # (and therefore Vulkan) because the wiring, not the selector logic, is what is under test.
      add_executable(test_draw_program_skip_render tests/gpu/diagnostics/test_draw_program_skip_render.cpp)
      target_include_directories(test_draw_program_skip_render PRIVATE frontends)
      target_include_directories(test_draw_program_skip_render PRIVATE src tests tools/boot_trace)
      target_link_libraries(test_draw_program_skip_render PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME draw_program_skip_render COMMAND test_draw_program_skip_render)

      # #3204: the renderer-target format table, extracted from execute_item where it was an
      # `||` chain -- the shape that cost Dead Cells an HDR target (#773) and Syberia its 3D menu.
      # Now an exhaustive switch: the compiler pins exhaustiveness, this test pins the table.
      # Registered in BOTH the UNIX and WIN32 branches, like the compute-backend tests above:
      # live_target_format.hpp includes <vulkan/vulkan.h>, so it needs an if(Vulkan_FOUND) scope,
      # and this file has one per platform.
      add_executable(test_live_target_format_match tests/gpu/test_live_target_format_match.cpp)
      target_link_libraries(test_live_target_format_match PRIVATE prosper_core Vulkan::Vulkan)
      target_include_directories(test_live_target_format_match PRIVATE frontends src)
      add_test(NAME live_target_format_match COMMAND test_live_target_format_match)
      # Armed from the ENVIRONMENT rather than by a setenv inside the test. The persistent-target
      # opt-out is read through PROSPER_ENV_VALUE, which caches, so arming it from inside the process
      # is exactly the vacuous-arm defect cached_env_arming_logic exists to catch; and the selector
      # itself is read once into a function-local static, so environment arming removes the ordering
      # argument for it too.
      set_tests_properties(draw_program_skip_render PROPERTIES ENVIRONMENT
        "PROSPER_SKIP_DRAW_PROGRAM=0x5006c6a00;PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1")

      add_executable(test_vulkan_offscreen tests/test_vulkan_offscreen.cpp)
      target_link_libraries(test_vulkan_offscreen PRIVATE Vulkan::Vulkan)
      add_test(NAME vulkan_offscreen_clear COMMAND test_vulkan_offscreen)

      # First real rendered output: a VkPipeline whose topology/format come from a RenderState via
      # vk_translate, drawing a triangle (embedded placeholder SPIR-V), verified by pixel readback.
      add_executable(test_vulkan_triangle tests/gpu/state/test_vulkan_triangle.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_vulkan_triangle PRIVATE tests)
      target_link_libraries(test_vulkan_triangle PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vulkan_triangle_render COMMAND test_vulkan_triangle)

      # Execution-differential harness (verification layer 4): run a compute shader with known
      # inputs, assert numeric outputs. The RDNA2->SPIR-V recompiler's output will plug in here.
      add_executable(test_compute_exec tests/test_compute_exec.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_compute_exec PRIVATE tests)
      target_link_libraries(test_compute_exec PRIVATE Vulkan::Vulkan)
      add_test(NAME compute_exec_differential COMMAND test_compute_exec)

      # Execute a real game kernel with its user/system-register ABI and resource table.
      add_executable(test_game_compute tests/gpu/recompiler/test_game_compute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_game_compute PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_game_compute PRIVATE frontends)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")
      # The compute path's full-surface seed/pack intermediates are pooled rather than freshly
      # allocated. Their old allocation was `make_unique_for_overwrite`, i.e. nominally
      # indeterminate -- but at these sizes it was a fresh mmap, which reads as ZERO, so a call site
      # could be depending on zeroing it never asked for. This arm hands every lease 0xCD instead:
      # it fails exactly when such a dependency exists, which is the only way that claim is testable
      # against the real backend rather than argued from the diff.
      add_test(NAME game_compute_exec_scratch_poison COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_scratch_poison PROPERTIES
        ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

      add_executable(test_live_compute_descriptor_array
                     tests/shared/live/test_live_compute_descriptor_array.cpp)
                     # Spells a `shared/...` include, which resolves from `frontends`.
                     target_include_directories(test_live_compute_descriptor_array PRIVATE frontends)
      target_link_libraries(test_live_compute_descriptor_array PRIVATE prosper_live_renderer)
      add_test(NAME live_compute_descriptor_array COMMAND test_live_compute_descriptor_array)

      # #3249: a dispatch result the host maps and reads needs an AVAILABILITY operation, not just a
      # fence. Registered beside the descriptor-array case because both drive the real backend
      # through execute_live_compute_items; it cannot live in the platform-independent block since
      # prosper_live_renderer exists only where Vulkan was found.
      add_executable(test_live_compute_host_read_barrier
                     tests/shared/live/test_live_compute_host_read_barrier.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_live_compute_host_read_barrier PRIVATE frontends)
      target_link_libraries(test_live_compute_host_read_barrier PRIVATE prosper_live_renderer)
      add_test(NAME live_compute_host_read_barrier
               COMMAND test_live_compute_host_read_barrier)

      # Prove our own SPIR-V emitter produces valid, correct, runnable SPIR-V (execution-differential).
      add_executable(test_spirv_builder tests/gpu/recompiler/test_spirv_builder.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_spirv_builder PRIVATE tests)
      target_link_libraries(test_spirv_builder PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME spirv_builder_exec COMMAND test_spirv_builder)

      # End-to-end: recompile real RDNA2 (llvm-mc) -> SPIR-V -> run -> assert numeric result.
      add_executable(test_rdna2_to_spirv tests/gpu/recompiler/test_rdna2_to_spirv.cpp
                 tests/gpu/recompiler/test_rdna2_bvh.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_rdna2_to_spirv PRIVATE tests)
      target_link_libraries(test_rdna2_to_spirv PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME rdna2_to_spirv_exec COMMAND test_rdna2_to_spirv)
      add_executable(test_descriptor_array_emit tests/gpu/test_descriptor_array_emit.cpp)
      target_link_libraries(test_descriptor_array_emit PRIVATE PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME descriptor_array_emit COMMAND test_descriptor_array_emit)


      # The DS layer-stride registry's identity (base+width+height, not base alone) and
      # PersistentDsKey's field membership.
      add_executable(test_ds_layer_stride tests/render/test_ds_layer_stride.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_ds_layer_stride PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_ds_layer_stride PRIVATE frontends)
      target_link_libraries(test_ds_layer_stride PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME ds_layer_stride COMMAND test_ds_layer_stride)

      # The per-draw internal-GDS scan memo (#2334). Two registrations of ONE binary: the second
      # sets the opt-out, so the pair proves the memo both fires and can be turned off, on a single
      # build. Both must PASS -- the binary flips its own expectation from the environment.
      add_executable(test_gds_scan_memo tests/render/test_gds_scan_memo.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_gds_scan_memo PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_gds_scan_memo PRIVATE frontends)
      target_link_libraries(test_gds_scan_memo PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gds_scan_memo COMMAND test_gds_scan_memo)
      add_test(NAME gds_scan_memo_disabled COMMAND test_gds_scan_memo)
      set_tests_properties(gds_scan_memo_disabled PROPERTIES
                           ENVIRONMENT "PROSPER_NO_GDS_SCAN_MEMO=1")

      # The backend's persistent-resource domain lock (#2953). Carries its own positive control --
      # an unguarded arm that proves the overlap detector fires at all -- so a clean zero in the
      # guarded arms cannot be read as "the threads never overlapped".
      add_executable(test_backend_persistent_resource_lock
                     tests/render/test_backend_persistent_resource_lock.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_backend_persistent_resource_lock PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_backend_persistent_resource_lock PRIVATE frontends)
      target_link_libraries(test_backend_persistent_resource_lock
                            PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME backend_persistent_resource_lock
               COMMAND test_backend_persistent_resource_lock)

      # #2944: every GPU readback in the render backend is host-read after a fence wait, and a fence
      # orders execution rather than performing the availability operation into the host domain.
      # Structural by necessity -- on a coherent desktop driver the UNFIXED code returns correct
      # pixels, so the assertion is that the barrier is RECORDED, with a no-readback negative control
      # so the counter cannot be satisfied by any render at all. Same include roots and same real
      # Vulkan device as the render_runner.h tests above.
      add_executable(test_host_read_barrier tests/render/test_host_read_barrier.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_host_read_barrier PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_host_read_barrier PRIVATE frontends)
      target_link_libraries(test_host_read_barrier PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME host_read_barrier COMMAND test_host_read_barrier)

      # #3248: the mip-assembly path cleared the destination image and then copied into it with no
      # memory dependency between the two writes. Structural for a sharper reason than the readback
      # above -- assembly clears to BLACK deliberately, so a clear that lands after a copy produces
      # exactly what a correct run produces for a missing level. No content assertion can separate
      # them; the negative control is an ordinary upload, which has no clear-then-copy pair.
      add_executable(test_mip_assembly_barrier tests/render/test_mip_assembly_barrier.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_mip_assembly_barrier PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_mip_assembly_barrier PRIVATE frontends)
      target_link_libraries(test_mip_assembly_barrier PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME mip_assembly_barrier COMMAND test_mip_assembly_barrier)

      # #3248: PROSPER_GEOM_PROBE and PROSPER_DRAW_ISO record Vulkan commands that no test executed,
      # so tools/vkval could not see them however it was configured. This case runs both once. Its
      # own assertions pin the arming decisions; the SPEC misuse they used to commit is caught by
      # running the suite under the validation layer, which is only possible because this exists.
      add_executable(test_render_diagnostic_paths tests/render/test_render_diagnostic_paths.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_render_diagnostic_paths PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_render_diagnostic_paths PRIVATE frontends)
      target_link_libraries(test_render_diagnostic_paths PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME render_diagnostic_paths COMMAND test_render_diagnostic_paths)
      # Armed HERE rather than with setenv() inside the test. PROSPER_GEOM_PROBE is a cached
      # one-shot read (PROSPER_ENV_VALUE), so a test that sets it after something has read it arms
      # nothing while still passing -- which `cached_env_arming_logic` refuses by name. The test
      # ENVIRONMENT is in place before the process starts, so read order cannot make it vacuous, and
      # the test fails loudly when run without it. PROSPER_ISO_AT must name a pixel inside the
      # 32x32 frame the test renders.
      set_tests_properties(render_diagnostic_paths PROPERTIES ENVIRONMENT
                           "PROSPER_GEOM_PROBE=0;PROSPER_DRAW_ISO=1;PROSPER_ISO_AT=16,16;PROSPER_FRAME_DIR=render_diagnostic_paths_frames")

      # The address-watch matcher's edge cases (exclusive end, empty range, UINT64_MAX wrap).
      add_executable(test_compute_address_watch tests/gpu/test_compute_address_watch.cpp)
      target_link_libraries(test_compute_address_watch PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME compute_address_watch COMMAND test_compute_address_watch)

      # The EXEC-driven pointer-chasing loop shape GTA V hangs on, plus PROSPER_CFG_TRIP_BOUND.
      add_executable(test_cfg_trip_bound tests/shared/diagnostics/test_cfg_trip_bound.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_cfg_trip_bound PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_cfg_trip_bound PRIVATE frontends)
      target_link_libraries(test_cfg_trip_bound PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME cfg_trip_bound_exec COMMAND test_cfg_trip_bound)

      # The same loop shape, but CHASING A REAL BUFFER through a V# -- the half test_cfg_trip_bound
      # deliberately leaves out ("decrements the index instead of chasing a buffer").
      add_executable(test_traversal_chase tests/gpu/recompiler/test_traversal_chase.cpp)
      target_include_directories(test_traversal_chase PRIVATE tests)
      target_link_libraries(test_traversal_chase PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME traversal_chase_exec COMMAND test_traversal_chase)

      # Render a triangle whose PIXEL shader is recompiled from RDNA2 (EXP MRT0 -> color output).
      add_executable(test_recompiled_fragment tests/gpu/recompiler/test_recompiled_fragment.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_recompiled_fragment PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_recompiled_fragment PRIVATE frontends)
      target_link_libraries(test_recompiled_fragment PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_fragment_render COMMAND test_recompiled_fragment)

      # Render a frame where BOTH vertex + pixel shaders are recompiled from RDNA2.
      add_executable(test_recompiled_shaders tests/gpu/recompiler/test_recompiled_shaders.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_recompiled_shaders PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_recompiled_shaders PRIVATE frontends)
      target_link_libraries(test_recompiled_shaders PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_shaders_render COMMAND test_recompiled_shaders)

      # Vertex fetch: a VS that pulls positions from a bound vertex buffer via buffer_load_format,
      # keyed by a ShaderResourceTable — real game vertex shaders work this way.
      add_executable(test_vertex_fetch_render tests/gpu/test_vertex_fetch_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_vertex_fetch_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_vertex_fetch_render PRIVATE frontends)
      target_link_libraries(test_vertex_fetch_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vertex_fetch_render COMMAND test_vertex_fetch_render)

      # Texture sampling: a recompiled pixel shader does MIMG image_sample against a bound 2x2 texture.
      add_executable(test_texture_sample_render tests/gpu/test_texture_sample_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_texture_sample_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_texture_sample_render PRIVATE frontends)
      target_link_libraries(test_texture_sample_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_sample_render COMMAND test_texture_sample_render)

      # Multi-draw backend: N draws composited into ONE cleared-once framebuffer (red opaque + green
      # additive -> yellow center proves accumulation into a single target).
      add_executable(test_multidraw_render tests/gpu/state/test_multidraw_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_multidraw_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_multidraw_render PRIVATE frontends)
      target_link_libraries(test_multidraw_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME multidraw_render COMMAND test_multidraw_render)

      # #3180: a failed color-target vkCreateImage must name its site and the real VkResult. Same
      # block as the other render_runner.h tests -- it needs the `tests` and `frontends` include
      # roots and a real Vulkan device.
      add_executable(test_color_target_create_failure tests/gpu/test_color_target_create_failure.cpp)
      target_include_directories(test_color_target_create_failure PRIVATE tests)
      target_include_directories(test_color_target_create_failure PRIVATE frontends)
      target_link_libraries(test_color_target_create_failure PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME color_target_create_failure COMMAND test_color_target_create_failure)

      # #3210: the render pass, framebuffer, image-view, sampler and shader-module creates in the
      # same pass. Two of them wrote UNINITIALIZED handles that were then used unchecked. Same
      # block and same reason as the test above -- `tests` and `frontends` include roots plus a
      # real Vulkan device.
      add_executable(test_render_object_create_failure
                     tests/gpu/test_render_object_create_failure.cpp)
      target_include_directories(test_render_object_create_failure PRIVATE tests)
      target_include_directories(test_render_object_create_failure PRIVATE frontends)
      target_link_libraries(test_render_object_create_failure PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME render_object_create_failure COMMAND test_render_object_create_failure)

      # Indexed draws (#64): vkCmdDrawIndexed from a BackendDraw's index data — the [0,1,2,2,3,0] quad
      # must match the retired perimeter-fan rendering pixel-exact, and indices must really be applied.
      add_executable(test_indexed_render tests/gpu/state/test_indexed_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_indexed_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_indexed_render PRIVATE frontends)
      target_link_libraries(test_indexed_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME indexed_render COMMAND test_indexed_render)

      add_executable(test_descriptor_array_render tests/gpu/state/test_descriptor_array_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_descriptor_array_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_descriptor_array_render PRIVATE frontends)
      target_link_libraries(test_descriptor_array_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME descriptor_array_render COMMAND test_descriptor_array_render)

      # Storage images: a recompiled compute kernel does MIMG image_load->image_store (no sampler) to
      # copy a 1D storage image bit-exact (the game's blit/copy compute-shader class).
      add_executable(test_storage_image_copy tests/gpu/test_storage_image_copy.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_storage_image_copy PRIVATE tests)
      target_link_libraries(test_storage_image_copy PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME storage_image_copy COMMAND test_storage_image_copy)

      # Interpolation: a recompiled VS exports EXP PARAM0, a recompiled PS reads it via v_interp.
      add_executable(test_interp_render tests/gpu/recompiler/test_interp_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_interp_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_interp_render PRIVATE frontends)
      target_link_libraries(test_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME interp_render COMMAND test_interp_render)

      # Full textured-draw path: VS exports interpolated UVs, PS samples a texture at them (+screenshot).
      add_executable(test_textured_interp_render tests/gpu/test_textured_interp_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_textured_interp_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_textured_interp_render PRIVATE frontends)
      target_link_libraries(test_textured_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME textured_interp_render COMMAND test_textured_interp_render)

      # The sampled-depth bridge's miss classifier must name the reason belonging to the entry it
      # rejected, not the first failing reason across every sibling at that address.
      add_executable(test_dsbridge_miss_classifier tests/render/test_dsbridge_miss_classifier.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_dsbridge_miss_classifier PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_dsbridge_miss_classifier PRIVATE frontends)
      target_link_libraries(test_dsbridge_miss_classifier PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME dsbridge_miss_classifier COMMAND test_dsbridge_miss_classifier)

      # IMAGE_SAMPLE_C_LZ 2D (shadow-map compare, #1271): manual-dref lowering, end-to-end.
      add_executable(test_shadow_compare_render tests/gpu/test_shadow_compare_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_shadow_compare_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_shadow_compare_render PRIVATE frontends)
      target_link_libraries(test_shadow_compare_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME shadow_compare_render COMMAND test_shadow_compare_render)

      # Generated mip chains bounded by the T#'s declared levels (#1272), end-to-end.
      add_executable(test_texture_mip_render tests/gpu/test_texture_mip_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_texture_mip_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_texture_mip_render PRIVATE frontends)
      target_link_libraries(test_texture_mip_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_mip_render COMMAND test_texture_mip_render)

      # Pipeline realization: build a VkPipeline whose fixed-function state comes from
      # resolve_pipeline_state(RenderState) and prove the resolved state drives rendering.
      add_executable(test_pipeline_render tests/gpu/test_pipeline_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_pipeline_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_pipeline_render PRIVATE frontends)
      target_link_libraries(test_pipeline_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME pipeline_render COMMAND test_pipeline_render)

      # Full back-half spine: a GpuState -> recompile shaders from their PGM addresses -> resolve
      # pipeline state -> rendered frame (the path a real submitted Dcb will drive).
      add_executable(test_gpustate_render tests/gpu/test_gpustate_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_gpustate_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_gpustate_render PRIVATE frontends)
      target_link_libraries(test_gpustate_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpustate_render COMMAND test_gpustate_render)

      # #3225: the backend's queue wrappers really consult the shutdown gate. The pure gate test
      # (gpu_submit_gate) asserts the gate's logic; this asserts the wiring, which needs
      # render_runner.h. Its positive arm uses a real queue when one exists and is skipped with a
      # printed note when it does not — the refusal arm needs no device at all.
      add_executable(test_submit_gate_wiring tests/gpu/test_submit_gate_wiring.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_submit_gate_wiring PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_submit_gate_wiring PRIVATE frontends)
      target_link_libraries(test_submit_gate_wiring PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME submit_gate_wiring COMMAND test_submit_gate_wiring)

      # GPU-executor core (Stage A): GpuState -> execute_gpustate() -> render -> present round-trip.
      add_executable(test_gpu_execute tests/gpu/execute/test_gpu_execute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_gpu_execute PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_gpu_execute PRIVATE frontends)
      target_link_libraries(test_gpu_execute PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpu_execute COMMAND test_gpu_execute)

      # Payoff demo: recompile the GAME'S OWN embedded shaders and render them on real Vulkan.
      # Needs both Vulkan and the (gitignored) game dump, so it's gated on HAVE_GAME_DUMP.
      if(HAVE_GAME_DUMP)
        add_executable(test_real_shader_render tests/self/test_real_shader_render.cpp)
        # Includes a shared fixture, which resolves from the `tests` root.
        target_include_directories(test_real_shader_render PRIVATE tests)
        # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
        target_include_directories(test_real_shader_render PRIVATE frontends)
        target_link_libraries(test_real_shader_render PRIVATE prosper_core Vulkan::Vulkan)
        # Messenger-specific: it recompiles and renders the RDNA2 blobs embedded in THIS title's
        # eboot, so a dump whose eboot carries none (or none that fully translate) fails (#1573).
        prosper_add_title_gated_test(real_shader_render PPSA24651
                 test_real_shader_render "${GAME_DUMP}/eboot.bin")
      endif()
      message(STATUS "Vulkan found (${Vulkan_LIBRARY}) — offscreen graphics harness enabled")
    else()
      message(STATUS "Vulkan not found — skipping offscreen graphics harness")
    endif()
  endif()
endif()

# Normal sampled-render capture on every host with the live renderer. Unix writes PNG through zlib;
# Windows uses its built-in WIC PNG encoder so the native tool needs no extra package.
if(TARGET prosper_live_renderer AND TARGET Vulkan::Vulkan)
  if(WIN32)
    add_executable(screenshot tools/screenshot/screenshot.cpp
                              tools/screenshot/capture_manifest.cpp
                              tools/screenshot/overlay_text.cpp)
                              # Spells a `shared/...` include, which resolves from `frontends`.
                              target_include_directories(screenshot PRIVATE frontends)
    target_include_directories(screenshot PRIVATE src tools/screenshot)
    target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan
                                              windowscodecs ole32)
  else()
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
      add_executable(screenshot tools/screenshot/screenshot.cpp
                                tools/screenshot/capture_manifest.cpp
                                tools/screenshot/overlay_text.cpp)
                                # Spells a `shared/...` include, which resolves from `frontends`.
                                target_include_directories(screenshot PRIVATE frontends)
      target_include_directories(screenshot PRIVATE src tools/screenshot)
      target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan ZLIB::ZLIB)
    else()
      message(STATUS "zlib not found — skipping the screenshot tool")
    endif()
  endif()
endif()

# Optional FFmpeg AJM codec backend (#1347). Keep it outside prosper_core so core-only/headless
# builds retain no host codec dependency. The core owns the AJM ABI and copies; this frontend only
# accepts host byte spans and returns interleaved S16 PCM. MP3 is the first implemented codec.
option(PROSPER_AUDIO_FFMPEG "Build the FFmpeg AJM compressed-audio decoder backend" ON)
if(UNIX AND PROSPER_AUDIO_FFMPEG AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_AJM_FFMPEG QUIET IMPORTED_TARGET
                      libavcodec libavutil libswresample)
  endif()
  if(PROSPER_AJM_FFMPEG_FOUND)
    add_library(prosper_ajm_ffmpeg STATIC
                frontends/audio_ffmpeg/ajm_ffmpeg.cpp)
    target_include_directories(prosper_ajm_ffmpeg PUBLIC frontends/audio_ffmpeg PRIVATE src)
    target_link_libraries(prosper_ajm_ffmpeg PUBLIC prosper_core
                          PkgConfig::PROSPER_AJM_FFMPEG)

    add_executable(test_ajm_mp3 tests/hle/audio/test_ajm_mp3.cpp)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    target_include_directories(test_ajm_mp3 PRIVATE src)
    target_link_libraries(test_ajm_mp3 PRIVATE prosper_ajm_ffmpeg)
    add_test(NAME ajm_mp3_decode COMMAND test_ajm_mp3)

    add_executable(test_ajm_opus tests/hle/audio/test_ajm_opus.cpp)
    target_include_directories(test_ajm_opus PRIVATE src tests/hle/audio)
    target_link_libraries(test_ajm_opus PRIVATE prosper_ajm_ffmpeg)
    add_test(NAME ajm_opus_decode COMMAND test_ajm_opus)

    # libSceAudiodec (UE Electra movie audio). Same backend, different guest-facing library.
    add_executable(test_audiodec tests/hle/audio/test_audiodec.cpp)
    target_include_directories(test_audiodec PRIVATE src)
    target_link_libraries(test_audiodec PRIVATE prosper_ajm_ffmpeg)
    add_test(NAME audiodec_aac_decode COMMAND test_audiodec)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(screenshot PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
  else()
    message(WARNING "FFmpeg audio development packages not found - AJM MP3 decoding disabled")
  endif()
endif()

# Windows native AvPlayer backend (#841): Media Foundation performs MP4 demux and video/audio
# decode on a worker thread. Keep it outside prosper_core so core-only/headless builds retain no OS
# media dependency and so software fallback can never be selected merely because no frontend exists.
if(WIN32 AND TARGET prosper_core)
  add_library(prosper_video_mf STATIC
              frontends/video_mf/media_foundation_backend.cpp)
  target_include_directories(prosper_video_mf PUBLIC frontends/video_mf PRIVATE src)
  target_link_libraries(prosper_video_mf PUBLIC prosper_core
                         mfplat mfreadwrite mfuuid ole32 propsys d3d11 dxgi strmiids)

  add_executable(test_video_mf frontends/video_mf/test_video_mf.cpp)
  target_link_libraries(test_video_mf PRIVATE prosper_video_mf)
  add_test(NAME video_media_foundation COMMAND test_video_mf)

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_video_mf)
    target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_MF=1)
  endif()

  # The capture frontend must boot with the same real decoder as boot_trace and prosper-app;
  # otherwise video-gated titles never reach the frames that the tool is meant to preserve.
  if(TARGET screenshot)
    target_link_libraries(screenshot PRIVATE prosper_video_mf)
    target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_MF=1)
  endif()
endif()

# Linux native AvPlayer backend (#841): FFmpeg provides MP4 demux/audio conversion and drives a
# VA-API hardware decoder for video. Keep the dependency outside prosper_core; systems without the
# development packages retain the headless core but do not advertise a native video backend.
option(PROSPER_VIDEO_VAAPI "Build the Linux FFmpeg/VA-API AvPlayer backend" ON)
if(UNIX AND NOT APPLE AND PROSPER_VIDEO_VAAPI AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_FFMPEG QUIET IMPORTED_TARGET
                      libavformat libavcodec libavutil libswresample libswscale libva)
  endif()
  if(PROSPER_FFMPEG_FOUND)
    add_library(prosper_video_vaapi STATIC
                frontends/video_vaapi/vaapi_backend.cpp)
    target_include_directories(prosper_video_vaapi PUBLIC frontends/video_vaapi PRIVATE src)
    target_link_libraries(prosper_video_vaapi PUBLIC prosper_core PkgConfig::PROSPER_FFMPEG)

    add_executable(test_video_vaapi frontends/video_vaapi/test_video_vaapi.cpp)
    target_link_libraries(test_video_vaapi PRIVATE prosper_video_vaapi)
    # Point the test at a committed test-pattern clip so it decodes a real stream. On headless CI
    # (no VA-API hardware) a successful decode exercises the default software-decode fallback (#320).
    # The second asset is a raw Annex-B H.264 ELEMENTARY STREAM, not a container: sceVideodec2's
    # caller demuxes itself and submits access units, so the AU-decode section (#2270) needs a
    # stream with no container around it, framed by access-unit delimiters the way a live guest's
    # is. Its expected per-frame hashes come from an independent host `ffmpeg` decode.
    target_compile_definitions(test_video_vaapi PRIVATE
      PROSPER_TEST_VIDEO_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/testpattern.mp4"
      PROSPER_TEST_H264_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_annexb_testpattern.264")
    add_test(NAME video_vaapi COMMAND test_video_vaapi)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_video_vaapi)
      target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()

    # Match boot_trace and prosper-app: headless captures must install the real Linux decoder too,
    # otherwise video-gated titles only expose an empty scanout to the screenshot frontend.
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_video_vaapi)
      target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()
  else()
    message(WARNING "FFmpeg/libva development packages not found - Linux AvPlayer video disabled")
  endif()
endif()

# --- SDL3 frontends (optional; headless core stays dependency-free) --------------------------
# The sceAudioOut sink (frontends/audio_sdl3) and the libScePad gamepad backend (frontends/pad_sdl3)
# can both use SDL3. They must share ONE SDL3 build: if each fetched independently, the first fetch's
# option set would win — e.g. an AUDIO+PAD build would get audio's joystick-less SDL and the gamepad
# backend would silently have no device. So acquire SDL3 ONCE here, with the UNION of the subsystems
# the enabled frontends need (JOYSTICK/HIDAPI only when the pad frontend is on). Both consumer blocks
# below just use the resulting SDL3::SDL3 target.
option(PROSPER_APP        "Build the prosper-app frontend (SDL3 window + Vulkan present)" OFF)
# The app frontend defaults to having audio, because it is the frontend a USER runs and a silent
# one is indistinguishable from broken audio. With PROSPER_AUDIO_SDL3 off, install_sdl3_audio_sink()
# is compiled out of prosper-app entirely and NOTHING is logged about it -- GTA V appeared to have
# no audio for exactly this reason, while the guest was publishing 99.9%-non-zero PCM that prosper
# mixed correctly into a sink with no device behind it (#2402).
#
# option() honours an existing cache entry, so an explicit -DPROSPER_AUDIO_SDL3=OFF still wins;
# this only changes the DEFAULT, and only when the app is being built.
option(PROSPER_AUDIO_SDL3 "Build the optional SDL3 audio frontend"   ${PROSPER_APP})
option(PROSPER_PAD_SDL3   "Build the optional SDL3 gamepad frontend" OFF)
if((PROSPER_AUDIO_SDL3 OR PROSPER_PAD_SDL3 OR PROSPER_APP) AND TARGET prosper_core AND NOT TARGET SDL3::SDL3)
  # On macOS the build is x86_64 (Rosetta), but a system SDL3 (e.g. Homebrew) is arm64-only and won't
  # link — so skip discovery there and always FetchContent+build SDL3 for the requested architecture.
  if(NOT APPLE)
    find_package(SDL3 QUIET CONFIG)
  endif()
  if(NOT SDL3_FOUND)
    message(STATUS "SDL3 not found — fetching release-3.2.30 (audio=${PROSPER_AUDIO_SDL3} pad=${PROSPER_PAD_SDL3} app=${PROSPER_APP})")
    include(FetchContent)
    set(SDL_TEST OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    # The window frontend (PROSPER_APP) needs a windowing system (Wayland/X11) for the Vulkan
    # surface; audio/pad-only builds do not — use SDL's console (no-video) escape hatch there. This
    # is the same "union of subsystems" logic as the joystick block below: enable VIDEO iff any
    # enabled frontend needs a window.
    if(PROSPER_APP)
      set(SDL_UNIX_CONSOLE_BUILD OFF CACHE BOOL "" FORCE)
      set(SDL_VIDEO   ON CACHE BOOL "" FORCE)
      # Wayland/X11 are the Linux window backends; on macOS SDL uses Cocoa (forcing X11/Wayland ON
      # there makes SDL hunt for X11/Wayland dev headers that don't exist). Only request them on Linux.
      if(NOT APPLE)
        set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
        set(SDL_X11     ON CACHE BOOL "" FORCE)
      endif()
    else()
      set(SDL_UNIX_CONSOLE_BUILD ON CACHE BOOL "" FORCE)
      set(SDL_VIDEO   OFF CACHE BOOL "" FORCE)
      set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
      set(SDL_X11     OFF CACHE BOOL "" FORCE)
    endif()
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    # The app frontend opens a title with the host's native folder picker (#1469); the audio- and
    # pad-only builds have no window to parent a dialog to, so they keep SDL's dialog code out.
    if(PROSPER_APP)
      set(SDL_DIALOG ON CACHE BOOL "" FORCE)
    else()
      set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    endif()
    # Gamepad support is needed ONLY by the pad frontend — enable JOYSTICK + HIDAPI whenever it's on,
    # regardless of which frontend triggers the (single) fetch. This is the union that fixes the
    # combined-build trap: an AUDIO+PAD build now gets a joystick-capable SDL.
    if(PROSPER_PAD_SDL3)
      set(SDL_JOYSTICK ON CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   ON CACHE BOOL "" FORCE)
    else()
      set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(SDL3
      GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
      GIT_TAG release-3.2.30
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
  endif()
endif()

# --- prosper-app: SDL3 window + Vulkan swapchain present frontend (issue #164 P0) -----------
# Presents the frame prosper's renderer hands to the present layer (present_readback) in a real
# window. A SEPARATE Vulkan context from the core's headless render device (docs/FRONTEND_APP.md);
# frames cross as CPU pixels. Entirely outside prosper_core. Default OFF; needs SDL3 (video) + Vulkan.
if(PROSPER_APP AND TARGET SDL3::SDL3)
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(Vulkan_FOUND)
    # Dear ImGui for the game library view (#1471), vendored under third_party/imgui. FRONTEND ONLY:
    # prosper_core neither links nor sees it, so the one-way dependency arrow in docs/FRONTEND_APP.md
    # is unchanged. Built as its own static lib so the vendored sources compile with their own
    # warning expectations rather than prosper's.
    add_library(prosper_imgui STATIC
                third_party/imgui/src/imgui.cpp
                third_party/imgui/src/imgui_draw.cpp
                third_party/imgui/src/imgui_tables.cpp
                third_party/imgui/src/imgui_widgets.cpp
                third_party/imgui/backends/imgui_impl_sdl3.cpp
                third_party/imgui/backends/imgui_impl_vulkan.cpp)
    target_include_directories(prosper_imgui PUBLIC third_party/imgui/src third_party/imgui/backends)
    target_link_libraries(prosper_imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan)
    # Vendored third-party code is not held to prosper's warning settings; keep its noise out of our
    # build output so a real warning in prosper's own code stays visible.
    if(NOT MSVC)
      target_compile_options(prosper_imgui PRIVATE -w)
    endif()

    add_executable(prosper-app frontends/prosper-app/main.cpp frontends/prosper-app/library_ui.cpp
                               frontends/prosper-app/library_media.cpp
                               frontends/prosper-app/fps_overlay.cpp)
    # third_party/renderdoc: the vendored in-application API header (#3321). Header-only and
    # never linked -- librenderdoc.so is dlopen'd at runtime by the capture trigger.
    target_include_directories(prosper-app PRIVATE src third_party/stb third_party/renderdoc
                                                   frontends)
    target_link_libraries(prosper-app PRIVATE prosper_core SDL3::SDL3 Vulkan::Vulkan prosper_imgui
                                              prosper_performance_capture)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIBRARY_UI=1)
    # Share the live renderer with boot_trace so the window shows composited game frames (not blank).
    if(TARGET prosper_live_renderer)
      target_link_libraries(prosper-app PRIVATE prosper_live_renderer)
      target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIVE_RENDERER=1)
    endif()
    # SDL3 dialog frontend (#347): real SDL_ShowMessageBox for the guest's MsgDialog/ErrorDialog via
    # the PlatformUi hook. The layout parsing (dialog_helpers) is pure + unit-tested (see below); this
    # module adds the SDL glue and needs SDL3 video (already required by prosper-app).
    add_library(prosper_dialog_sdl3 STATIC
                frontends/dialog_sdl3/platform_ui_sdl3.cpp frontends/dialog_sdl3/dialog_helpers.cpp)
    target_include_directories(prosper_dialog_sdl3 PUBLIC frontends/dialog_sdl3 src)
    target_link_libraries(prosper_dialog_sdl3 PUBLIC prosper_core SDL3::SDL3)
    target_link_libraries(prosper-app PRIVATE prosper_dialog_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_DIALOG_SDL3=1)
    message(STATUS "prosper-app frontend enabled (SDL3 window + Vulkan present)")
  else()
    message(WARNING "PROSPER_APP=ON but Vulkan not found — skipping prosper-app")
  endif()
endif()

# --- optional SDL3 audio frontend (headless core stays dependency-free) ---------------------
# The sceAudioOut HLE is backend-agnostic (src/hle/audio/audio.hpp). This builds an SDL3-backed sink,
# entirely OUTSIDE prosper_core, so the guest's audio plays on the host.
if(PROSPER_AUDIO_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_audio_sdl3 STATIC frontends/audio_sdl3/audio_sink_sdl3.cpp)
  target_include_directories(prosper_audio_sdl3 PUBLIC frontends/audio_sdl3 src)
  target_link_libraries(prosper_audio_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_audio_sdl3 PUBLIC PROSPER_AUDIO_SDL3=1)

  # Headless self-test of the full SDL3 path (run under the SDL "dummy" audio driver in CI):
  # install the sink, drive the sceAudioOut HLE, assert grains flow through SDL with no device.
  add_executable(test_audio_sdl3 frontends/audio_sdl3/test_audio_sdl3.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_audio_sdl3 PRIVATE src)
  target_link_libraries(test_audio_sdl3 PRIVATE prosper_audio_sdl3)
  add_test(NAME audio_sdl3 COMMAND test_audio_sdl3)
  set_tests_properties(audio_sdl3 PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy")

  # Route the live game's audio to the host when the runner is built (guarded by the macro).
  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_audio_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_SDL3=1)
  endif()

  message(STATUS "SDL3 audio frontend enabled (prosper_audio_sdl3)")
endif()

# --- optional SDL3 gamepad frontend (headless core stays dependency-free) --------------------
# The libScePad HLE is backend-agnostic (src/input/pad.hpp). This builds an SDL_Gamepad-backed
# PadBackend, entirely OUTSIDE prosper_core, so a real controller drives the guest cross-platform.
# Off by default.  Enable with:  -DPROSPER_PAD_SDL3=ON  (SDL3 acquired by the shared step above,
# with JOYSTICK/HIDAPI ON because this option is set).
if(PROSPER_PAD_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_pad_sdl3 STATIC frontends/pad_sdl3/pad_sink_sdl3.cpp)
  target_include_directories(prosper_pad_sdl3 PUBLIC frontends/pad_sdl3 src)
  target_link_libraries(prosper_pad_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_pad_sdl3 PUBLIC PROSPER_PAD_SDL3=1)

  # Headless self-test of the full SDL3 gamepad path (no controller attached in CI): install the
  # backend, poll -> disconnected/neutral, drive the pure mapping. Uses the SDL dummy video driver.
  add_executable(test_pad_sdl3 frontends/pad_sdl3/test_pad_sdl3.cpp)
  target_link_libraries(test_pad_sdl3 PRIVATE prosper_pad_sdl3)
  add_test(NAME pad_sdl3 COMMAND test_pad_sdl3)
  set_tests_properties(pad_sdl3 PROPERTIES ENVIRONMENT "SDL_VIDEO_DRIVER=dummy;SDL_AUDIO_DRIVER=dummy")

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_pad_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_SDL3=1)
  endif()

  message(STATUS "SDL3 gamepad frontend enabled (prosper_pad_sdl3)")
endif()

# --- wire the SDL3 audio/pad frontends into prosper-app (after their targets exist) ---------
# prosper-app is defined above (before these libs), so link them here once they're built. Enable
# both alongside the app for a full experience:  -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
if(TARGET prosper-app)
  if(TARGET prosper_ajm_ffmpeg)
    target_link_libraries(prosper-app PRIVATE prosper_ajm_ffmpeg)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_FFMPEG=1)
    message(STATUS "prosper-app: FFmpeg AJM audio decoding enabled")
  endif()
  if(TARGET prosper_audio_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_audio_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/audio_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_SDL3=1)
    message(STATUS "prosper-app: audio out enabled")
  endif()
  if(TARGET prosper_pad_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_pad_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/pad_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_PAD_SDL3=1)
    message(STATUS "prosper-app: controller input enabled")
  endif()
endif()

if(TARGET prosper-app AND TARGET prosper_video_mf)
  target_link_libraries(prosper-app PRIVATE prosper_video_mf)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_MF=1)
  message(STATUS "prosper-app: Media Foundation video enabled")
endif()

if(TARGET prosper-app AND TARGET prosper_video_vaapi)
  target_link_libraries(prosper-app PRIVATE prosper_video_vaapi)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_VAAPI=1)
  message(STATUS "prosper-app: FFmpeg/VA-API video enabled")
endif()

# Shared-device adoption guard (#1091). prosper_live_renderer only exists in frontend-enabled
# configurations and is defined later in this file than the core test block, so register this
# after the fact and only when the target is actually present.
if(TARGET prosper_live_renderer AND TARGET prosper_core)
  add_executable(test_shared_vulkan_device tests/shared/live/test_shared_vulkan_device.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_shared_vulkan_device PRIVATE tests)
  # Spells a `shared/...` include, which resolves from `frontends`.
  target_include_directories(test_shared_vulkan_device PRIVATE frontends)
  target_link_libraries(test_shared_vulkan_device PRIVATE prosper_live_renderer prosper_core)
  add_test(NAME shared_vulkan_device COMMAND test_shared_vulkan_device)

  # Live-target format round trip: every LiveTargetPixelFormat a compute dispatch can import must
  # survive the write-back notifier. An unnamed format is reported as RGBA8, fails the mirror
  # identity check, and silently leaves the renderer's target invalidated (#773's family).
  add_executable(test_live_target_format tests/shared/live/test_live_target_format.cpp)
  # Spells a `shared/...` include, which resolves from `frontends`.
  target_include_directories(test_live_target_format PRIVATE frontends)
  target_include_directories(test_live_target_format PRIVATE src tests)
  target_link_libraries(test_live_target_format PRIVATE prosper_live_renderer prosper_core
                                                        Vulkan::Vulkan)
  add_test(NAME live_target_format COMMAND test_live_target_format)

  # GPU scanout handoff (#1270): the front-buffer image blits into a scanout slot byte-exactly, and the
  # producer/consumer slot handoff never tears or reuses an image in flight. Headless (no surface).
  add_executable(test_present_blit tests/shared/present/test_present_blit.cpp)
  # Spells a `shared/...` include, which resolves from `frontends`.
  target_include_directories(test_present_blit PRIVATE frontends)
  target_include_directories(test_present_blit PRIVATE src tests)
  target_link_libraries(test_present_blit PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
  add_test(NAME present_blit COMMAND test_present_blit)
endif()

# ---------------------------------------------------------------------------
# Per-case scratch directory for every registered test (#1613).
#
# Tests that must touch the real filesystem take their paths from
# tests/fixtures/test_scratch.h, which reads PROSPER_TEST_SCRATCH_DIR. Giving each ctest
# *case* its own root is what makes `ctest -j` safe: one test binary is
# sometimes registered as several cases (game_compute_exec and
# game_compute_exec_no_adaptive_storage_result are the same executable), so
# per-binary uniqueness is not enough. The helper adds a pid component
# underneath, which separates concurrent ctest invocations from different
# worktrees. The root is inside the build directory, never /tmp — see the
# header for why that matters on this machine.
#
# This must stay at the very end of the file: it snapshots the DIRECTORY TESTS
# property, so any add_test() below it would not be covered. A test that did get
# missed still fails safe rather than silently — test_scratch.h falls back to
# <cwd>/prosper-test-scratch/, and ctest's working directory is this same build
# directory, so it lands off the tmpfs either way; it merely loses the per-case
# split (the pid component still separates concurrent processes).
get_property(prosper_registered_tests DIRECTORY PROPERTY TESTS)
foreach(prosper_test_case IN LISTS prosper_registered_tests)
  set_property(TEST ${prosper_test_case} APPEND PROPERTY ENVIRONMENT
    "PROSPER_TEST_SCRATCH_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-scratch/${prosper_test_case}")
endforeach()
