// test_env_numeric_sites — one arm per PROSPER_* knob converted to the strict parser by #3267.
//
// WHY THIS SHAPE, AND WHY IT IS NOT A TEST OF THE HELPER. `parse_u64_strict` and
// `env_u64_or_default*` are already pinned by frontends/shared/tests/test_write_watch_policy.cpp.
// What that file cannot say is whether a given CALL SITE passes the right fallback and the right
// cap, and that is where the defect this issue is about actually lives: the helper is correct and
// the site still selects a dangerous setting if its fallback is wrong.
//
// So each arm below reproduces ONE site's arithmetic twice -- once as the site is written now
// (`converted`), once as it was written before (`legacy`) -- and asserts:
//
//   converted(malformed) == expected   the refusal keeps the DEFAULT
//   legacy(malformed)    != expected   ...and the old spelling did NOT, i.e. the site really was
//                                      one of the 169, so the arm is not vacuously green
//   converted(good)      == the parsed setting, so the refusal did not break the knob
//
// The second condition is the one that earns the arm. Without it a site that was ALREADY safe --
// there are 23 such in the census, guarded by a range check or a `> 0` -- would produce an
// identical-looking pass, and this file would then be asserting a property it never exercised.
//
// WHY THE ENVIRONMENT IS NEVER TOUCHED. Almost every converted site caches its read in a
// function-local `static`, so a test that armed the variable with setenv and then called the site
// would go VACUOUS rather than red once the static was already initialised (#2214, and
// tools/env/check_cached_env.py is the gate for it). These arms therefore pass the TEXT directly to
// the same helper call the site makes, which is a pure function of its arguments. No PROSPER_*
// variable is set, unset or read here.
//
// WHAT THIS THEREFORE DOES NOT PIN, stated plainly because the limitation is the price of the shape
// above: each arm mirrors its site rather than calling it, so it asserts the CONTRACT (which helper,
// which fallback, which cap) and not the WIRING. A site edited to pass a different fallback while
// its arm here stays put would not redden. tools/env/check_env_numeric_arms.py closes the half of
// that gap which is mechanically checkable -- it fails if a knob is parsed through env_numeric with
// no arm here, or armed here with no call site left -- and it is registered as `env_numeric_arms`.
// The remaining gap is a fallback VALUE changed at one end only, which stays a review matter.
//
// The expected values are the sites' documented defaults; if a default is deliberately changed, the
// arm here must change with it, which is the point.
#include "diagnostics/env_numeric.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

using prosper::diag::env_u64_or_default;
using prosper::diag::env_u64_or_default_capped;
using prosper::diag::parse_u64_strict;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", what);
    if (!ok) ++fails;
}

// One census row: the site's own arithmetic, before and after.
struct Site {
    const char* where;        // file + knob, so a failure names the line to open
    const char* name;         // the real variable name, so the refusal on stderr is the real one
    uint64_t (*converted)(const char* name, const char* text);
    uint64_t (*legacy)(const char* text);
    const char* malformed;    // the input a person plausibly types
    uint64_t expected;        // what the converted site must answer for it
    const char* good;         // a well-formed value...
    uint64_t good_expected;   // ...and what it must still select
};

// --- src/host/memory/guest_write_watch.cpp : PROSPER_WRITE_WATCH_MAX_KB ------------------------
// 0 means UNBOUNDED here, so a malformed value used to remove the cap entirely.
static uint64_t ww_max_new(const char* n, const char* t) {
    const uint64_t kib = env_u64_or_default_capped(n, t, 0ull, UINT64_MAX / 1024ull, "KiB",
                                                   "unbounded: no range is too large to watch");
    return kib ? kib * 1024ull : UINT64_MAX;
}
static uint64_t ww_max_old(const char* t) {
    const uint64_t kib = t ? std::strtoull(t, nullptr, 10) : 0ull;
    if (!kib) return UINT64_MAX;
    return kib > UINT64_MAX / 1024ull ? UINT64_MAX : kib * 1024ull;
}

// --- frontends/shared/live/live_renderer.cpp : PROSPER_ARRAY_DECODE_BUDGET_MIB -----------------
static uint64_t array_budget_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 1024ull, UINT64_MAX >> 20, "MiB") << 20;
}
static uint64_t array_budget_old(const char* t) {
    return static_cast<uint64_t>(t ? std::atoll(t) : 1024) << 20;
}

// --- a byte cap expressed in MiB, the shape shared by six sites --------------------------------
// live_compute.cpp (BUFFER_CACHE_MB 256, MEMORY_POOL_MB 640) and render_runner.h
// (BACKEND_TARGET_CACHE_MB 256, BACKEND_TEXTURE_CACHE_MB 1024, MEMORY_POOL_MB 512).
template <uint64_t Default>
static uint64_t mib_cap_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, Default, UINT64_MAX / (1024ull * 1024ull), "MiB")
           * 1024ull * 1024ull;
}
template <uint64_t Default>
static uint64_t mib_cap_old(const char* t) {
    const uint64_t mib = t ? std::strtoull(t, nullptr, 10) : Default;
    if (mib > UINT64_MAX / (1024ull * 1024ull)) return UINT64_MAX;
    return mib * 1024ull * 1024ull;
}

// --- tests/fixtures/render_runner.h : PROSPER_RENDER_COPY_THREADS -----------------------------
// #3407: worker count for the parallel texture-staging copy. 0 means "derive from the core count",
// so a malformed value must land on 0 rather than a wrapped 32 -- which is what a strtoul parse of
// "-1" produced before this site used the helper.
static uint64_t copy_threads_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 0ull, 32ull, "threads");
}
static uint64_t copy_threads_old(const char* t) {
    if (!t || !*t) return 0ull;
    const unsigned long parsed = std::strtoul(t, nullptr, 10);
    return parsed > 32ul ? 32ull : static_cast<uint64_t>(parsed);
}

// --- tests/fixtures/render_runner.h : PROSPER_BACKEND_BUFFER_ARENA_KB --------------------------
static uint64_t arena_new(const char* n, const char* t) {
    const uint64_t kib = env_u64_or_default_capped(n, t, 1024ull, UINT64_MAX / 1024ull, "KiB");
    const uint64_t bytes = kib * 1024ull;
    return bytes < 4 ? 4 : bytes;
}
static uint64_t arena_old(const char* t) {
    const uint64_t kib = t ? std::strtoull(t, nullptr, 10) : 1024ull;
    if (kib > UINT64_MAX / 1024ull) return UINT64_MAX;
    const uint64_t bytes = kib * 1024ull;
    return bytes < 4 ? 4 : bytes;
}

// --- tests/fixtures/render_runner.h : PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES --------------------
static uint64_t layout_entries_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 256ull, SIZE_MAX, "entries");
}
static uint64_t layout_entries_old(const char* t) {
    return t ? std::strtoull(t, nullptr, 10) : 256ull;
}

// --- src/hle/memory/hle_kernel_mem.cpp : PROSPER_DMEM_BUDGET_MB --------------------------------
// The `>= 1024` floor already refused small typos; what it could not refuse was `-1`, which
// saturated and then WRAPPED the MiB multiply.
static uint64_t dmem_new(const char* n, const char* t) {
    const uint64_t mib = env_u64_or_default_capped(n, t, 16ull * 1024ull,
                                                   UINT64_MAX / (1024ull * 1024ull), "MiB");
    if (mib >= 1024) return mib * 1024ull * 1024ull;
    return 16ull * 1024 * 1024 * 1024;
}
static uint64_t dmem_old(const char* t) {
    const uint64_t mib = t ? std::strtoull(t, nullptr, 10) : 0ull;
    if (mib >= 1024) return mib * 1024ull * 1024ull;   // wraps for UINT64_MAX
    return 16ull * 1024 * 1024 * 1024;
}

// --- src/gpu/execute/gpu_executor.cpp : PROSPER_MAX_DISPATCH_GROUPS ----------------------------
// Unset is 0 = "no cap", which is the right default and the wrong answer to a typo: the knob is
// only ever set in order to impose a cap.
static uint64_t dispatch_cap_new(const char* n, const char* t) {
    if (!t || !*t) return 0;
    return prosper::diag::env_u64_or_default_auto_capped(n, t, 0ull, UINT32_MAX, "workgroups",
                                                         "no cap: nothing is bounded");
}
static uint64_t dispatch_cap_old(const char* t) {
    return t && *t ? static_cast<uint32_t>(std::strtoul(t, nullptr, 0)) : 0u;
}

// --- the four default-ON guards ----------------------------------------------------------------
// PROSPER_REL1_FORGE_GUARD, PROSPER_REL1_STOMP_GUARD, PROSPER_MB3_CENTRAL_SCAN,
// PROSPER_MB3_TRACK_TLS and PROSPER_NETCTL_CB all read `!e || strtol(e, nullptr, 0) != 0`. The
// inversion is the whole finding: `=yes`, `=true` and `=on` are what a person types to make a
// default-ON switch explicit, and strtol answers 0 for every one of them -- i.e. OFF.
static uint64_t default_on_new(const char* n, const char* t) {
    return prosper::diag::env_u64_or_default_auto(n, t, 1ull) != 0 ? 1 : 0;
}
static uint64_t default_on_old(const char* t) {
    return (!t || std::strtol(t, nullptr, 0) != 0) ? 1 : 0;
}

// --- src/hle/kernel/hle_kernel.cpp : PROSPER_MUTEX_FAIR_US -------------------------------------
static uint64_t fair_us_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 3000ull, INT_MAX, "us");
}
static uint64_t fair_us_old(const char* t) { return t ? (uint64_t)(unsigned)std::atoi(t) : 3000ull; }

// --- src/hle/input/hle_pad.cpp : PROSPER_PAD_FRAME_HOLD / PROSPER_PAD_READ_HOLD ----------------
static uint64_t pad_frame_hold_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 8ull, INT64_MAX, "flips");
}
static uint64_t pad_read_hold_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 8ull, INT64_MAX, "reads");
}
static uint64_t pad_hold_old(const char* t) { return t ? (uint64_t)std::atoll(t) : 8ull; }


// --- the six knobs #3253 converted, plus its two siblings ---------------------------------------
// These were already strict before this file existed; the arms are here because
// tools/env/check_env_numeric_arms.py requires one per parsed knob, and because #3253 pinned the
// HELPER's grammar without pinning any individual site's fallback. Their sentinels are the sharpest
// in the tree: on every one of them 0 is the MOST aggressive setting, not "off".
static uint64_t kib_cap_1024(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 1024ull, SIZE_MAX / 1024ull, "KiB") * 1024ull;
}
static uint64_t kib_cap_8192(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, 8192ull, SIZE_MAX / 1024ull, "KiB") * 1024ull;
}
static uint64_t kib_cap_old(uint64_t dflt, const char* t) {
    return (t ? std::strtoull(t, nullptr, 10) : dflt) * 1024ull;
}
static uint64_t kib_1024_old(const char* t) { return kib_cap_old(1024ull, t); }
static uint64_t kib_8192_old(const char* t) { return kib_cap_old(8192ull, t); }
template <uint64_t Default>
static uint64_t mib_cap_size_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, Default, SIZE_MAX / (1024ull * 1024ull), "MiB")
           * 1024ull * 1024ull;
}
template <uint64_t Default>
static uint64_t hits_new(const char* n, const char* t) {
    return env_u64_or_default_capped(n, t, Default, UINT32_MAX, "unchanged validations");
}
template <uint64_t Default>
static uint64_t hits_old(const char* t) { return t ? std::strtoull(t, nullptr, 10) : Default; }

// --- tests/fixtures/render_runner.h : PROSPER_BACKEND_TARGET_CACHE_COUNT (#3267 B1) -------------
// `n ? n : 256` refused a value parsing to 0 and nothing else, so `=-1` saturated, stayed non-zero,
// and removed the resident-target count bound outright.
static uint64_t target_count_new(const char* n, const char* t) {
    const uint64_t v = env_u64_or_default_capped(n, t, 256ull, SIZE_MAX, "targets");
    return v ? v : 256ull;
}
static uint64_t target_count_old(const char* t) {
    const uint64_t v = t ? std::strtoull(t, nullptr, 10) : 256ull;
    return v ? v : 256ull;
}

// --- src/hle/service/hle_service.cpp : PROSPER_VDEC2_DUMP_FRAMES (#3267 B1) ---------------------
// Same shape, and the bound it lost is a DISK bound: the site's own comment puts the 16-frame cap
// at 50 MB against 5 GB uncapped.
static uint64_t vdec_frames_new(const char* n, const char* t) {
    const uint64_t v = prosper::diag::env_u64_or_default_auto_capped(n, t, 16ull, UINT32_MAX,
                                                                     "frames");
    return v ? v : 16ull;
}
static uint64_t vdec_frames_old(const char* t) {
    const uint64_t v = t ? std::strtoul(t, nullptr, 0) : 0ull;
    return v ? v : 16ull;
}

// --- frontends/shared/live/live_compute.cpp : PROSPER_COMPUTE_IMAGE_CACHE_MB (#3267 N2) ---------
// The one converted site whose refusal falls back to a CODE PATH rather than a number: unset means
// "derive the budget from device memory". kSentinelDerived stands for that path so the arm can be
// expressed in the same uint64_t table as the rest.
static const uint64_t kSentinelDerived = ~0ull - 1ull;
static uint64_t image_cache_new(const char* n, const char* t) {
    uint64_t mib = 0;
    if (!prosper::diag::env_u64_or_report(n, t, &mib, "MiB", "the device-derived budget"))
        return kSentinelDerived;
    if (mib > UINT64_MAX / (1024ull * 1024ull)) return UINT64_MAX;
    return mib * 1024ull * 1024ull;
}
static uint64_t image_cache_old(const char* t) {
    // The pre-image had no empty-string check either, so `=""` reached strtoull and selected a
    // ZERO-byte image cache. That half is a fix too, and this mirror keeps it visible.
    if (!t) return kSentinelDerived;
    const uint64_t mib = std::strtoull(t, nullptr, 10);
    if (mib > UINT64_MAX / (1024ull * 1024ull)) return UINT64_MAX;
    return mib * 1024ull * 1024ull;
}

static const uint64_t kMiB = 1024ull * 1024ull;
static const uint64_t kGiB = 1024ull * kMiB;

static const Site kSites[] = {
    // A malformed value used to make the cap 1024x TIGHTER than asked for -- which on this knob
    // means "watch essentially nothing", since every range above 8 KiB is then refused a watch.
    // TWO malformed inputs are deliberately absent, for the same reason: `-1` saturates and is then
    // clamped back to UINT64_MAX, and `eight` parses to 0 which IS this knob's default -- in both
    // cases the old spelling already answered what the refusal keeps, so an arm would be void
    // rather than green (this file's own rule, applied to itself). On those inputs the change here
    // is the MESSAGE alone, and the same is true of PROSPER_MAX_DISPATCH_GROUPS below; both say so
    // in their refusal, which is why both pass a gloss naming what their 0 means.
    {"guest_write_watch.cpp PROSPER_WRITE_WATCH_MAX_KB", "PROSPER_WRITE_WATCH_MAX_KB",
     ww_max_new, ww_max_old,
     "8mb", UINT64_MAX /* the documented 0 == unbounded default */, "65536", 65536ull * 1024ull},
    {"guest_write_watch.cpp PROSPER_WRITE_WATCH_MAX_KB (spaced unit)",
     "PROSPER_WRITE_WATCH_MAX_KB", ww_max_new, ww_max_old,
     "64 MB", UINT64_MAX, "1", 1024ull},

    // A malformed value used to make the decode budget effectively unbounded, or 1024x too small.
    {"live_renderer.cpp PROSPER_ARRAY_DECODE_BUDGET_MIB", "PROSPER_ARRAY_DECODE_BUDGET_MIB",
     array_budget_new, array_budget_old, "-1", 1024ull << 20, "2048", 2048ull << 20},
    {"live_renderer.cpp PROSPER_ARRAY_DECODE_BUDGET_MIB (unit suffix)",
     "PROSPER_ARRAY_DECODE_BUDGET_MIB", array_budget_new, array_budget_old,
     "1gib", 1024ull << 20, "512", 512ull << 20},

    {"live_compute.cpp PROSPER_COMPUTE_BUFFER_CACHE_MB", "PROSPER_COMPUTE_BUFFER_CACHE_MB",
     mib_cap_new<256>, mib_cap_old<256>, "-1", 256ull * kMiB, "512", 512ull * kMiB},
    {"live_compute.cpp PROSPER_COMPUTE_MEMORY_POOL_MB", "PROSPER_COMPUTE_MEMORY_POOL_MB",
     mib_cap_new<640>, mib_cap_old<640>, "1gb", 640ull * kMiB, "1024", 1024ull * kMiB},
    {"render_runner.h PROSPER_BACKEND_TARGET_CACHE_MB", "PROSPER_BACKEND_TARGET_CACHE_MB",
     mib_cap_new<256>, mib_cap_old<256>, "1gb", 256ull * kMiB, "128", 128ull * kMiB},
    {"render_runner.h PROSPER_BACKEND_TEXTURE_CACHE_MB", "PROSPER_BACKEND_TEXTURE_CACHE_MB",
     mib_cap_new<1024>, mib_cap_old<1024>, "eight", 1024ull * kMiB, "2048", 2048ull * kMiB},
    {"render_runner.h PROSPER_MEMORY_POOL_MB", "PROSPER_MEMORY_POOL_MB",
     mib_cap_new<512>, mib_cap_old<512>, "-1", 512ull * kMiB, "256", 256ull * kMiB},
    // #3405: the mapped-staging cache's own budget, accounted separately from the transient pool
    // above so peak retained host memory is the sum of the two rather than one figure standing in
    // for both. A malformed value must keep the default rather than disable the bound.
    {"render_runner.h PROSPER_MAPPED_STAGING_MB", "PROSPER_MAPPED_STAGING_MB",
     mib_cap_new<256>, mib_cap_old<256>, "-1", 256ull * kMiB, "512", 512ull * kMiB},
    {"render_runner.h PROSPER_RENDER_COPY_THREADS", "PROSPER_RENDER_COPY_THREADS",
     copy_threads_new, copy_threads_old, "-1", 0ull, "4", 4ull},

    {"render_runner.h PROSPER_BACKEND_BUFFER_ARENA_KB", "PROSPER_BACKEND_BUFFER_ARENA_KB",
     arena_new, arena_old, "4mb", 1024ull * 1024ull, "2048", 2048ull * 1024ull},
    {"render_runner.h PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES",
     "PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES", layout_entries_new, layout_entries_old,
     "512 entries", 256ull, "512", 512ull},

    {"hle_kernel_mem.cpp PROSPER_DMEM_BUDGET_MB", "PROSPER_DMEM_BUDGET_MB",
     dmem_new, dmem_old, "-1", 16ull * kGiB, "8192", 8192ull * kMiB},

    {"gpu_executor.cpp PROSPER_MAX_DISPATCH_GROUPS", "PROSPER_MAX_DISPATCH_GROUPS",
     dispatch_cap_new, dispatch_cap_old,
     "750,000", 0ull /* refused: no cap, but the refusal SAYS so instead of capping at 750 */,
     "65536", 65536ull},

    // The default-ON family. One arm per spelling a person actually types to make a default
    // explicit -- every one of which strtol answers 0 for, i.e. OFF.
    {"command_processor.cpp PROSPER_REL1_FORGE_GUARD", "PROSPER_REL1_FORGE_GUARD",
     default_on_new, default_on_old, "yes", 1, "0", 0},
    {"command_processor.cpp PROSPER_REL1_STOMP_GUARD", "PROSPER_REL1_STOMP_GUARD",
     default_on_new, default_on_old, "on", 1, "1", 1},
    {"mb3_freelist.cpp PROSPER_MB3_CENTRAL_SCAN", "PROSPER_MB3_CENTRAL_SCAN",
     default_on_new, default_on_old, "true", 1, "0", 0},
    {"mb3_freelist.cpp PROSPER_MB3_TRACK_TLS", "PROSPER_MB3_TRACK_TLS",
     default_on_new, default_on_old, "enabled", 1, "0", 0},
    {"hle_service.cpp PROSPER_NETCTL_CB", "PROSPER_NETCTL_CB",
     default_on_new, default_on_old, "yes", 1, "0", 0},

    {"hle_kernel.cpp PROSPER_MUTEX_FAIR_US", "PROSPER_MUTEX_FAIR_US",
     fair_us_new, fair_us_old, "3ms", 3000ull, "500", 500ull},

    {"hle_pad.cpp PROSPER_PAD_FRAME_HOLD", "PROSPER_PAD_FRAME_HOLD",
     pad_frame_hold_new, pad_hold_old, "16 flips", 8ull, "16", 16ull},
    {"hle_pad.cpp PROSPER_PAD_READ_HOLD", "PROSPER_PAD_READ_HOLD",
     pad_read_hold_new, pad_hold_old, "-1", 8ull, "4", 4ull},

    {"render_runner.h PROSPER_BACKEND_TARGET_CACHE_COUNT", "PROSPER_BACKEND_TARGET_CACHE_COUNT",
     target_count_new, target_count_old, "-1", 256ull, "512", 512ull},
    {"hle_service.cpp PROSPER_VDEC2_DUMP_FRAMES", "PROSPER_VDEC2_DUMP_FRAMES",
     vdec_frames_new, vdec_frames_old, "-1", 16ull, "0x40", 64ull},
    {"live_compute.cpp PROSPER_COMPUTE_IMAGE_CACHE_MB", "PROSPER_COMPUTE_IMAGE_CACHE_MB",
     image_cache_new, image_cache_old, "512mb", kSentinelDerived, "512", 512ull * kMiB},
    {"live_compute.cpp PROSPER_COMPUTE_IMAGE_CACHE_MB (empty string)",
     "PROSPER_COMPUTE_IMAGE_CACHE_MB", image_cache_new, image_cache_old,
     "", kSentinelDerived, "1", 1ull * kMiB},

    // N1: `0x…` was well-formed at these six before the conversion, because they read base 0. The
    // `good` column is the regression: if the _auto family is ever narrowed back to decimal, these
    // fail. On PROSPER_MAX_DISPATCH_GROUPS refusing `0x2000` would mean NO CAP, i.e. this PR's own
    // hazard class introduced by the fix for it.
    {"gpu_executor.cpp PROSPER_MAX_DISPATCH_GROUPS (base-0 spelling kept)",
     "PROSPER_MAX_DISPATCH_GROUPS", dispatch_cap_new, dispatch_cap_old,
     "0x2000 groups", 0ull, "0x2000", 8192ull},
    {"command_processor.cpp PROSPER_REL1_FORGE_GUARD (base-0 off kept)",
     "PROSPER_REL1_FORGE_GUARD", default_on_new, default_on_old, "0xzz", 1, "0x0", 0},

    // #3253's own six, plus the two the same PR added. Every fallback below is that site's
    // documented default; every legacy answer below is the aggressive end of its own policy.
    {"live_renderer.cpp PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB",
     "PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", kib_cap_8192, kib_8192_old,
     "8mb", 8192ull * 1024ull, "4096", 4096ull * 1024ull},
    {"live_renderer.cpp PROSPER_TEXTURE_WRITE_WATCH_MIN_KB", "PROSPER_TEXTURE_WRITE_WATCH_MIN_KB",
     kib_cap_1024, kib_1024_old, "1mb", 1024ull * 1024ull, "512", 512ull * 1024ull},
    {"live_renderer.cpp PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS",
     "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", hits_new<3>, hits_old<3>,
     "three", 3ull, "5", 5ull},
    {"live_renderer.cpp PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB",
     "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", mib_cap_size_new<8>, mib_cap_old<8>,
     "64mb", 8ull * kMiB, "16", 16ull * kMiB},
    {"live_compute.cpp PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS",
     "PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS", hits_new<3>, hits_old<3>,
     "-1", 3ull, "7", 7ull},
    {"live_compute.cpp PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_MB",
     "PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_MB", mib_cap_size_new<8>, mib_cap_old<8>,
     "eight", 8ull * kMiB, "32", 32ull * kMiB},
    // #3309's pooled decode intermediates. The safe sentinel here is 0 ("retain nothing", i.e. the
    // pre-pool allocation pattern), so a malformed value that parsed to 0 would silently REMOVE the
    // optimisation rather than enable a more aggressive one -- the mild direction of this family's
    // hazard, and still a measurement attributed to the wrong setting.
    //
    // The malformed input is "64mb", not "512mb", and the difference is the arm's whole value: the
    // legacy `strtoull` answer for "512mb" is 512, which IS this knob's default, so that spelling
    // made the "...and the old spelling did NOT" leg unprovable and the arm vacuous. The suite
    // caught it. Pick a malformed value whose legacy parse differs from the fallback.
    {"decode_scratch.hpp PROSPER_DECODE_SCRATCH_MB", "PROSPER_DECODE_SCRATCH_MB",
     mib_cap_size_new<512>, mib_cap_old<512>, "64mb", 512ull * kMiB, "64", 64ull * kMiB},
    {"live_compute.cpp PROSPER_COLD_STORAGE_SNAPSHOT_MIN_MB",
     "PROSPER_COLD_STORAGE_SNAPSHOT_MIN_MB", mib_cap_size_new<16>, mib_cap_old<16>,
     "64 MB", 16ull * kMiB, "64", 64ull * kMiB},
    {"live_compute.cpp PROSPER_MAX_GPU_COMPARE_IMAGE_MB", "PROSPER_MAX_GPU_COMPARE_IMAGE_MB",
     mib_cap_size_new<2>, mib_cap_old<2>, "32mb", 2ull * kMiB, "8", 8ull * kMiB},
};
int main() {
    char msg[512];
    for (const Site& s : kSites) {
        const uint64_t got = s.converted(s.name, s.malformed);
        std::snprintf(msg, sizeof msg, "%s: '%s' keeps the default", s.where, s.malformed);
        check(got == s.expected, msg);

        // The discriminator. If this fails, the site did not need converting and the arm above
        // proved nothing.
        const uint64_t was = s.legacy(s.malformed);
        std::snprintf(msg, sizeof msg,
                      "%s: ...and the old spelling did NOT (it selected a different setting)",
                      s.where);
        check(was != s.expected, msg);

        const uint64_t good = s.converted(s.name, s.good);
        std::snprintf(msg, sizeof msg, "%s: '%s' still selects the value asked for", s.where,
                      s.good);
        check(good == s.good_expected, msg);
    }

    // Two properties of the shared grammar that every arm above depends on, asserted once here so a
    // failure points at the helper rather than at twenty sites.
    uint64_t parsed = 12345;
    check(!parse_u64_strict("-1", &parsed) && parsed == 12345,
          "control: a refusal leaves the caller's variable untouched");
    check(env_u64_or_default("PROSPER_UNUSED_IN_TREE", nullptr, 7ull) == 7ull &&
          env_u64_or_default("PROSPER_UNUSED_IN_TREE", "", 7ull) == 7ull,
          "control: unset and empty take the default in silence, and are not typos");

    std::printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
