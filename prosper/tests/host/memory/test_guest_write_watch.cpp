#include "host/memory/guest_write_watch.hpp"
#include "host/image/exec_image.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>

using prosper::host::GuestWriteWatch;
using prosper::host::GuestWriteWatchQuery;

namespace {
int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; } \
} while (0)
} // namespace

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

int main() {
    constexpr uint64_t kSize = 0x10000;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                        0, static_cast<DWORD>(kSize), nullptr);
    CHECK(section != nullptr, "paging-file section created");
    if (!section) return 1;
    auto* first = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    auto* second = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    CHECK(first && second, "two aliases mapped");
    if (!first || !second) {
        if (first) UnmapViewOfFile(first);
        if (second) UnmapViewOfFile(second);
        CloseHandle(section);
        return 1;
    }

    constexpr uint64_t kPhys = 0x10000000;
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(first), kSize, kPhys, PAGE_READWRITE);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(second), kSize, kPhys, PAGE_READWRITE);

    first[0x180] = 0x11;
    GuestWriteWatch watch = GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(first + 0x100), 0x2100);
    CHECK(!static_cast<bool>(watch),
          "Windows rejects page-fault watches that can corrupt the guest SysV red zone");
    CHECK(watch.query() == GuestWriteWatchQuery::Unknown,
          "unsupported watch selects the caller's exact-comparison fallback");
    CHECK(!watch.rearm(), "unsupported watch cannot arm read-only fault pages");

    second[0x180] = 0x22;
    CHECK(first[0x180] == 0x22, "write through second alias updates first alias");
    prosper::host::guest_write_watch_notify_physical_write(kPhys + 0x1000, 0x1000);
    CHECK(watch.query() == GuestWriteWatchQuery::Unknown,
          "physical writes leave an unsupported watch on the exact fallback");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(second), kSize);
    watch.reset();

    first[0x180] = 0x44;
    CHECK(second[0x180] == 0x44, "invalidation restores writable page protections");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(first), kSize);
    UnmapViewOfFile(second);
    UnmapViewOfFile(first);
    CloseHandle(section);

    const auto stats = prosper::host::guest_write_watch_stats();
    CHECK(stats.create_attempts >= 1, "unsupported watch attempts remain observable");
    CHECK(stats.registrations == 0 && stats.faults == 0 && stats.rearms == 0,
          "safe Windows fallback never arms or handles guest page faults");
    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}

#else   // ---- Linux: exercise the real mprotect + SIGSEGV dirty-tracking (#1144) ------------------

#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace {
// dmem is section-backed (one memfd, MAP_SHARED at multiple VAs) — mirror that so the alias handling
// is exercised, not just a single mapping.
constexpr uint32_t kCpuRw = 0x3;   // SCE CPU_READ|CPU_WRITE (see host_prot in guest_write_watch.cpp)

} // namespace

int main() {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t size = page * 4;
    CHECK(setenv("PROSPER_WRITE_WATCH_MAX_KB", "1024", 1) == 0,
          "write-watch size policy fixed for the test");

    int fd = memfd_create("prosper-ww-test", 0);
    CHECK(fd >= 0, "memfd created");
    if (fd < 0) return 1;
    CHECK(ftruncate(fd, static_cast<off_t>(size)) == 0, "memfd sized");

    // Two MAP_SHARED aliases of the same physical (memfd offset 0) range.
    auto* a = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    auto* b = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK(a != MAP_FAILED && b != MAP_FAILED, "two aliases mapped");
    if (a == MAP_FAILED || b == MAP_FAILED) return 1;
    constexpr uint64_t kPhys = 0x20000;   // arbitrary distinct phys base for the test

    // Use the production SIGSEGV handler, not a test-local approximation. Besides covering the
    // sigaltstack gate, the guest-FS regression below must prove the exact signal boundary used by a
    // running title restores host TLS around write-watch handling and guest TLS before resuming.
    CHECK(unsetenv("PROSPER_FAULT_NO_ONSTACK") == 0,
          "production write-watch signal path enabled for the test");
    prosper::install_trap_handler();

    // The real emulator calls set_fault_onstack(true) once at startup, before any dmem map records an
    // alias (and the notify hooks only record while the feature is enabled). Mirror that: enable, record
    // both aliases, THEN prove the gate refuses when the handler is turned back off.
    prosper::host::guest_write_watch_set_fault_onstack(true);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(a), size, kPhys, kCpuRw);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(b), size, kPhys, kCpuRw);

    // Gate check: with the handler NOT on the sigaltstack, create() must refuse (safe fallback).
    prosper::host::guest_write_watch_set_fault_onstack(false);
    GuestWriteWatch off = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), size);
    CHECK(!static_cast<bool>(off), "create refuses to arm when the handler is not on the sigaltstack");

    prosper::host::guest_write_watch_set_fault_onstack(true);

    // Large resources use the renderer's existing exact-comparison fallback instead of creating
    // per-page fault state. Level streaming can otherwise register millions of pages and spend most
    // of its time changing page protections. Exercise the policy with a valid dmem-style mapping.
    {
        const size_t oversized_size = (1u << 20) + page;
        int large_fd = memfd_create("prosper-ww-oversized", 0);
        CHECK(large_fd >= 0, "oversized memfd created");
        if (large_fd >= 0) {
            CHECK(ftruncate(large_fd, static_cast<off_t>(oversized_size)) == 0,
                  "oversized memfd sized");
            auto* large = static_cast<uint8_t*>(
                mmap(nullptr, oversized_size, PROT_READ | PROT_WRITE, MAP_SHARED, large_fd, 0));
            CHECK(large != MAP_FAILED, "oversized mapping");
            if (large != MAP_FAILED) {
                constexpr uint64_t kLargePhys = 0x1000000;
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(large), oversized_size, kLargePhys, kCpuRw);
                GuestWriteWatch oversized = GuestWriteWatch::create(
                    reinterpret_cast<uint64_t>(large), oversized_size);
                CHECK(!static_cast<bool>(oversized),
                      "oversized watch selects exact-comparison fallback");
                CHECK(oversized.query() == GuestWriteWatchQuery::Unknown,
                      "oversized watch reports Unknown for exact comparison");
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(large), oversized_size);
                munmap(large, oversized_size);
            }
            close(large_fd);
        }
    }

    // Arm a watch over the whole range and confirm it starts Unchanged.
    GuestWriteWatch watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), size);
    CHECK(static_cast<bool>(watch), "watch armed over a fully-aliased dmem range");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "freshly armed watch reads Unchanged");

    // Write through alias A -> faults -> handler restores write + marks dirty; the byte must land.
    a[page + 0x40] = 0x5a;
    CHECK(a[page + 0x40] == 0x5a, "write through alias A lands (no lost store)");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty, "watch reads Dirty after a CPU write");

    // Two workers can fault on the same armed page before either handler restores it. Model the
    // second, already-queued delivery deterministically: the first real signal above disarmed the
    // still-live CPU-writable page, so a sibling delivery must be accepted and allowed to retry.
    CHECK(prosper::host::guest_write_watch_handle_fault(
              reinterpret_cast<uint64_t>(a + page + 0x48)),
          "a stale sibling fault on a known writable page retries instead of becoming fatal");
    CHECK(!prosper::host::guest_write_watch_handle_fault(0x9000000000ULL),
          "an unknown stale-fault address remains unhandled");

    // Fail closed when guest topology has genuinely made the known VA read-only. WatchedPage retains
    // its registration until query/reset, so this specifically proves the stale-delivery path consults
    // current mapping topology instead of accepting every known-but-disarmed page.
    CHECK(mprotect(a + page, page, PROT_READ) == 0,
          "known watched alias can be reprotected read-only for stale-fault test");
    prosper::host::guest_write_watch_notify_direct_mapping_protection(
        reinterpret_cast<uint64_t>(a + page), page, 0x1 /* CPU_READ only */);
    CHECK(!prosper::host::guest_write_watch_handle_fault(
              reinterpret_cast<uint64_t>(a + page + 0x48)),
          "a genuine fault on a known read-only alias remains unhandled");
    CHECK(mprotect(a + page, page, PROT_READ | PROT_WRITE) == 0,
          "known watched alias restored writable after stale-fault test");
    prosper::host::guest_write_watch_notify_direct_mapping_protection(
        reinterpret_cast<uint64_t>(a + page), page, kCpuRw);

    // Explicit guest protection changes require fresh alias metadata; ordinary content writes
    // below still rearm the same registration without recreating it.
    CHECK(!watch.rearm() && watch.query() == GuestWriteWatchQuery::Dirty,
          "guest re-protection invalidates the old registration even after restoring permissions");
    watch.reset();
    watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), size);
    CHECK(static_cast<bool>(watch), "recreate the watch after guest re-protection");
    // Re-arm, don't write -> Unchanged again.
    CHECK(watch.rearm(), "rearm succeeds");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "no write since rearm -> Unchanged");

    // ALIAS COVERAGE: write through the OTHER alias B (same phys) must also be caught (both aliases
    // were armed). This is the correctness crux — miss it and the renderer trusts a stale texture.
    b[page * 2 + 0x8] = 0xa5;
    CHECK(b[page * 2 + 0x8] == 0xa5, "write through alias B lands");
    CHECK(a[page * 2 + 0x8] == 0xa5, "aliases share storage");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "write through a SECOND alias of the same phys is caught");

    // A physical-write notification (GPU write) invalidates too.
    CHECK(watch.rearm(), "rearm before physical-write test");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean before physical write");
    prosper::host::guest_write_watch_notify_physical_write(kPhys + page, page);
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty, "physical write marks the watch Dirty");

    // B2: an alias mapped AFTER the watch is armed must be armed too, or a write through it would not
    // fault and the watch would wrongly read Unchanged (stale texture). notify_direct_mapping_added must
    // extend the existing page's coverage in place — without a spurious Dirty for same-phys aliasing.
    CHECK(watch.rearm(), "rearm before late-alias test");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean before late alias");
    auto* c = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK(c != MAP_FAILED, "third alias mapped");
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(c), size, kPhys, kCpuRw);
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged,
          "adding a same-phys alias does not spuriously dirty the watch");
    c[page * 3 + 0x20] = 0x3c;   // write through the LATE alias
    CHECK(c[page * 3 + 0x20] == 0x3c, "write through late alias lands");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "a write through an alias added AFTER arming is caught (B2)");

    // B5: a real kernel write (::read) into an armed read-only page returns EFAULT — the exact failure a
    // texture-streaming sceKernelPread would hit. Pre-announcing the write disarms the range so the store
    // lands and the watch reads Dirty, matching read_full()'s guard.
    CHECK(watch.rearm(), "rearm before kernel-write test");
    {
        int pf[2]; CHECK(pipe(pf) == 0, "pipe A created");
        const char payload[8] = {9, 8, 7, 6, 5, 4, 3, 2};
        CHECK(write(pf[1], payload, sizeof payload) == (ssize_t)sizeof payload, "payload A queued");
        errno = 0;
        const ssize_t blocked = read(pf[0], a + page * 3, sizeof payload);   // page armed RO -> EFAULT
        CHECK(blocked < 0 && errno == EFAULT, "kernel write into an armed page is blocked (EFAULT)");
        close(pf[0]); close(pf[1]);
    }
    {
        int pf[2]; CHECK(pipe(pf) == 0, "pipe B created");
        const char payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        CHECK(write(pf[1], payload, sizeof payload) == (ssize_t)sizeof payload, "payload B queued");
        prosper::host::guest_write_watch_notify_host_write(
            reinterpret_cast<uint64_t>(a + page * 3), sizeof payload);
        const ssize_t got = read(pf[0], a + page * 3, sizeof payload);       // disarmed -> lands
        CHECK(got == (ssize_t)sizeof payload, "kernel read into a pre-announced guest page succeeds (B5)");
        CHECK(a[page * 3] == 1 && a[page * 3 + 7] == 8, "kernel-written bytes landed");
        CHECK(watch.query() == GuestWriteWatchQuery::Dirty, "kernel write marks the watch Dirty (B5)");
        close(pf[0]); close(pf[1]);
    }

    // B4: removing ONE alias of a multi-alias page must drop only that alias and keep the page covered by
    // the survivors (not orphan a stale entry, and not stop faulting). Remove alias c; a write through a
    // must still be caught.
    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(c), size);
    CHECK(watch.rearm(), "rearm after removing one alias");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean after removing one alias");
    a[page + 0x24] = 0x4b;   // write through a surviving alias
    CHECK(a[page + 0x24] == 0x4b, "write through surviving alias lands");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "a page still covered by a surviving alias keeps faulting after a sibling alias is removed (B4)");
    munmap(c, size);

    // A partial remap of the ORIGINAL watched VA must force a new registration. Exercise both
    // an orphaned old page and a surviving old-physical sibling alias: nonempty alias coverage
    // alone cannot prove that this watch still describes the requested address.
    for (bool keep_old_alias : {false, true}) {
        const size_t remap_bytes = page * 3;
        const uint64_t old_phys = keep_old_alias ? 0xc10000 : 0xc00000;
        const uint64_t new_phys = old_phys + 0x8000;
        int old_fd = memfd_create("prosper-ww-remap-old", 0);
        int new_fd = memfd_create("prosper-ww-remap-new", 0);
        CHECK(old_fd >= 0 && new_fd >= 0, "remap memfds created");
        if (old_fd < 0 || new_fd < 0) return 1;
        CHECK(ftruncate(old_fd, static_cast<off_t>(remap_bytes)) == 0 &&
                  ftruncate(new_fd, static_cast<off_t>(page)) == 0,
              "remap memfds sized");
        auto* original = static_cast<uint8_t*>(mmap(
            nullptr, remap_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, old_fd, 0));
        CHECK(original != MAP_FAILED, "original remap range mapped");
        if (original == MAP_FAILED) return 1;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(original), remap_bytes, old_phys, kCpuRw);
        uint8_t* sibling = nullptr;
        if (keep_old_alias) {
            sibling = static_cast<uint8_t*>(mmap(
                nullptr, remap_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, old_fd, 0));
            CHECK(sibling != MAP_FAILED, "old-physical sibling mapped");
            if (sibling == MAP_FAILED) return 1;
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(sibling), remap_bytes, old_phys, kCpuRw);
        }
        auto original_watch = GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(original + 8), remap_bytes - 16);
        CHECK(original_watch && original_watch.query() == GuestWriteWatchQuery::Unchanged,
              "unaligned original range has a clean watch before middle-page replacement");
        auto* middle = original + page;
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(middle), page);
        CHECK(mmap(middle, page, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, new_fd, 0) == middle,
              "middle page remapped to a different backing section at the same VA");
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(middle), page, new_phys, kCpuRw);
        CHECK(!original_watch.rearm(), "original-VA remap cannot rearm an old physical registration");
        CHECK(original_watch.query() == GuestWriteWatchQuery::Dirty,
              "failed remap rearm never clears the obsolete registration's dirty state");
        original_watch.reset();
        auto replacement_watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(middle + 8), 16);
        CHECK(replacement_watch && replacement_watch.query() == GuestWriteWatchQuery::Unchanged,
              "fresh registration resolves the replacement VA backing");
        middle[12] = 0x5a;
        CHECK(replacement_watch.query() == GuestWriteWatchQuery::Dirty,
              "real write to replacement backing dirties the recreated watch");
        CHECK(!sibling || sibling[page + 12] == 0,
              "replacement write does not reach the surviving old physical alias");
        replacement_watch.reset();
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(original), remap_bytes);
        munmap(original, remap_bytes);
        if (sibling) {
            prosper::host::guest_write_watch_notify_direct_mapping_removed(
                reinterpret_cast<uint64_t>(sibling), remap_bytes);
            munmap(sibling, remap_bytes);
        }
        close(old_fd);
        close(new_fd);
    }

    // Protection changes through another VA invalidate the affected PHYSICAL registration only.
    // A read-only alias becoming writable must not remain invisible to a retained watch.
    {
        int protection_fd = memfd_create("prosper-ww-protection", 0);
        CHECK(protection_fd >= 0 && ftruncate(protection_fd, static_cast<off_t>(page * 3)) == 0,
              "protection-transition memfd sized");
        if (protection_fd < 0) return 1;
        auto* writable = static_cast<uint8_t*>(mmap(
            nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_SHARED, protection_fd, 0));
        auto* readonly = static_cast<uint8_t*>(mmap(
            nullptr, page * 3, PROT_READ, MAP_SHARED, protection_fd, 0));
        CHECK(writable != MAP_FAILED && readonly != MAP_FAILED,
              "writable and read-only aliases mapped");
        if (writable == MAP_FAILED || readonly == MAP_FAILED) return 1;
        constexpr uint64_t protection_phys = 0xc20000;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(writable), page * 3, protection_phys, kCpuRw);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(readonly), page * 3, protection_phys, 0x1);
        auto affected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page), page);
        auto unaffected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable), page);
        CHECK(affected && unaffected, "affected and disjoint physical ranges watched");
        CHECK(mprotect(readonly + page, page, PROT_READ | PROT_WRITE) == 0,
              "middle page of read-only alias becomes writable");
        prosper::host::guest_write_watch_notify_direct_mapping_protection(
            reinterpret_cast<uint64_t>(readonly + page), page, kCpuRw);
        CHECK(!affected.rearm() && affected.query() == GuestWriteWatchQuery::Dirty,
              "changed sibling-alias protection requires a fresh registration");
        CHECK(unaffected.rearm() && unaffected.query() == GuestWriteWatchQuery::Unchanged,
              "partial protection change leaves disjoint physical registration rearmable");
        affected.reset();
        affected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page), page);
        CHECK(affected && affected.query() == GuestWriteWatchQuery::Unchanged,
              "sole-owner replacement watch uses current sibling-alias protections");
        readonly[page + 32] = 0x77;
        CHECK(writable[page + 32] == 0x77 && affected.query() == GuestWriteWatchQuery::Dirty,
              "real store through newly writable alias dirties recreated watch");
        affected.reset();
        unaffected.reset();
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(writable), page * 3);
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(readonly), page * 3);
        munmap(writable, page * 3);
        munmap(readonly, page * 3);
        close(protection_fd);
    }

    // N2: a partial re-protect must record ONLY the re-protected sub-range. Re-protect the MIDDLE page of
    // a fresh 3-page mapping read-only; a watch over the FIRST page must still find it writable, arm it,
    // and catch a write. If the whole alias's prot were overwritten, the outer page would read as
    // non-writable, never arm, and the write would be missed (stale texture).
    {
        int fd2 = memfd_create("prosper-ww-n2", 0);
        CHECK(fd2 >= 0 && ftruncate(fd2, static_cast<off_t>(page * 3)) == 0, "n2 memfd sized");
        auto* m = static_cast<uint8_t*>(
            mmap(nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_SHARED, fd2, 0));
        CHECK(m != MAP_FAILED, "n2 mapping");
        constexpr uint64_t kN2Phys = 0x50000;   // distinct from kPhys above
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(m), page * 3, kN2Phys, kCpuRw);
        prosper::host::guest_write_watch_notify_direct_mapping_protection(
            reinterpret_cast<uint64_t>(m) + page, page, 0x1 /* CPU_READ only */);
        GuestWriteWatch outer = GuestWriteWatch::create(reinterpret_cast<uint64_t>(m), page);
        CHECK(static_cast<bool>(outer), "a page outside a partial re-protect is still armable (N2)");
        CHECK(outer.query() == GuestWriteWatchQuery::Unchanged, "outer page clean after arm");
        m[0x30] = 0x9d;
        CHECK(m[0x30] == 0x9d, "write to the outer page lands");
        CHECK(outer.query() == GuestWriteWatchQuery::Dirty,
              "a write to a page outside the re-protected sub-range is caught (N2)");
        outer.reset();
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(m), page * 3);
        munmap(m, page * 3); close(fd2);
    }

    // A watch over a range with NO known mapping must return Unknown (exact fallback), not arm.
    GuestWriteWatch none = GuestWriteWatch::create(0x9000000000ULL, page);
    CHECK(!static_cast<bool>(none), "create over an unmapped range yields Unknown");

    // A title's Job.Worker executes guest code with %fs pointing at its guest TCB. The production
    // SIGSEGV handler must temporarily restore host %fs before GuestWriteWatch enters std::mutex and
    // glibc mprotect, then restore guest %fs before the faulting store resumes. Make that boundary
    // observable by adding a deliberately stale non-faulting alias after arming: the real alias's
    // mprotect succeeds, the stale alias's mprotect fails with ENOMEM, and glibc writes errno through
    // the active %fs. Without the production scope this overwrites the guest-TCB errno slot while the
    // real host errno remains zero.
    {
        int tls_fd = memfd_create("prosper-ww-guest-fs", 0);
        CHECK(tls_fd >= 0 && ftruncate(tls_fd, static_cast<off_t>(page)) == 0,
              "guest-FS watch memfd sized");
        auto* guest_page = static_cast<uint8_t*>(
            mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_SHARED, tls_fd, 0));
        CHECK(guest_page != MAP_FAILED, "guest-FS watched mapping created");
        if (tls_fd >= 0 && guest_page != MAP_FAILED) {
            constexpr uint64_t kGuestFsPhys = 0x80000;
            constexpr uint64_t kStaleAlias = 0x700000000000ull;
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(guest_page), page, kGuestFsPhys, kCpuRw);
            GuestWriteWatch guest_watch = GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(guest_page), page);
            CHECK(static_cast<bool>(guest_watch), "guest-FS watch armed");

            // The fake VA is intentionally not mapped. Adding it after the page is armed records the
            // stale alias and makes its immediate mprotect fail without preventing the real alias from
            // remaining armed. Rearm accepts the updated generation because the physical page itself
            // is still armed.
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                kStaleAlias, page, kGuestFsPhys, kCpuRw);
            CHECK(guest_watch.rearm(), "guest-FS watch rearmed after stale-alias injection");
            CHECK(guest_watch.query() == GuestWriteWatchQuery::Unchanged,
                  "guest-FS watch clean before the fault");

            uint64_t host_fs = 0;
            __asm__ volatile("rdfsbase %0" : "=r"(host_fs));
            const intptr_t errno_offset = reinterpret_cast<intptr_t>(&errno) -
                                          static_cast<intptr_t>(host_fs);
            CHECK(errno_offset < 0 && errno_offset >= -0x20000,
                  "glibc errno fits inside the provisioned guest TLS test block");

            CHECK(unsetenv("PROSPER_NO_GUEST_FS") == 0,
                  "Linux guest-FS path enabled for write-watch test");
            prosper::TlsModuleDesc tls_modules[2]{};  // index zero is reserved
            tls_modules[1].memsz = 0x20000;
            tls_modules[1].align = 64;
            prosper::guest_tls_set_templates(tls_modules, 2);
            CHECK(prosper::guest_tls_enabled(), "guest TLS configured for write-watch fault");

            errno = 0;
            const uint64_t guest_fs = prosper::guest_tls_activate_thread();
            constexpr int kGuestErrnoSentinel = 0x51a7e11;
            auto* guest_errno = reinterpret_cast<volatile int*>(guest_fs + errno_offset);
            *guest_errno = kGuestErrnoSentinel;

            // No libc between activating guest FS and capturing the post-signal state.
            guest_page[0x80] = 0xa7;
            uint64_t fs_at_resume = 0;
            __asm__ volatile("rdfsbase %0" : "=r"(fs_at_resume));
            const int guest_errno_after = *guest_errno;
            const uint8_t landed = guest_page[0x80];
            prosper::guest_fs_enter_host_for_signal();

            CHECK(guest_fs != 0 && guest_fs != host_fs,
                  "guest-FS write-watch test activated a distinct guest TCB");
            CHECK(landed == 0xa7, "guest-FS fault resumed and the original store landed");
            CHECK(fs_at_resume == guest_fs,
                  "write-watch signal return restored the caller's guest FS");
            CHECK(errno == ENOMEM,
                  "failing mprotect updated host errno through host FS");
            CHECK(guest_errno_after == kGuestErrnoSentinel,
                  "host write-watch handling did not overwrite guest TLS errno");

            guest_watch.reset();
            prosper::host::guest_write_watch_notify_direct_mapping_removed(kStaleAlias, page);
            prosper::host::guest_write_watch_notify_direct_mapping_removed(
                reinterpret_cast<uint64_t>(guest_page), page);
            munmap(guest_page, page);
        }
        if (tls_fd >= 0) close(tls_fd);
    }

    watch.reset();
    // After reset the pages are writable and stores fault no more (the handler would _exit(2) if a
    // stale armed page remained). Touch both aliases to prove protections were fully restored.
    a[0x10] = 0x1; b[page + 0x10] = 0x2;
    CHECK(a[0x10] == 0x1 && b[page + 0x10] == 0x2, "reset restored writable protections");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(a), size);
    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(b), size);

    const auto stats = prosper::host::guest_write_watch_stats();
    CHECK(stats.registrations >= 1, "at least one watch armed");
    CHECK(stats.faults >= 2, "both alias-A and alias-B write faults were handled");
    CHECK(stats.stale_faults >= 1, "stale concurrent fault delivery was counted");
    CHECK(stats.rearms >= 2, "rearm path exercised");
    CHECK(stats.create_oversized >= 1, "oversized exact-comparison fallback is observable");

    munmap(a, size); munmap(b, size); close(fd);
    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}

#endif
