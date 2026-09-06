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
#include <csignal>
#include <cstring>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
// dmem is section-backed (one memfd, MAP_SHARED at multiple VAs) — mirror that so the alias handling
// is exercised, not just a single mapping.
constexpr uint32_t kCpuRw = 0x3;   // SCE CPU_READ|CPU_WRITE (see host_prot in guest_write_watch.cpp)

enum class AccessProbe { Allowed, Faulted, Failed };

// Fault in a child, never the test runner. Ordinary probes deliberately bypass prosper's handler:
// they measure actual host permissions, not whether a stale watch could restore permission later.
// The tracked-store arm instead keeps the production handler and checks the child's copied watch
// state there; a fork cannot make the parent's private dirty counters observe the child's store.
AccessProbe probe_access(uint8_t* address, bool write, GuestWriteWatch* tracked = nullptr,
                         GuestWriteWatch* obsolete = nullptr, bool keep_fault_handler = false) {
    const pid_t child = fork();
    if (child < 0) return AccessProbe::Failed;
    if (child == 0) {
        const rlimit no_core{0, 0};
        if (setrlimit(RLIMIT_CORE, &no_core) != 0) _exit(120);
        struct sigaction action{};
        action.sa_handler = SIG_DFL;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGALRM, &action, nullptr) != 0 ||
            sigaction(SIGBUS, &action, nullptr) != 0 ||
            (!tracked && !keep_fault_handler &&
             sigaction(SIGSEGV, &action, nullptr) != 0)) _exit(121);
        sigset_t unblock;
        sigemptyset(&unblock);
        sigaddset(&unblock, SIGALRM);
        sigaddset(&unblock, SIGBUS);
        sigaddset(&unblock, SIGSEGV);
        if (sigprocmask(SIG_UNBLOCK, &unblock, nullptr) != 0) _exit(122);
        alarm(3); // A stale alias protection must fail, not loop forever in the fault handler.
        if (tracked && tracked->query() != GuestWriteWatchQuery::Unchanged) _exit(123);
        if (obsolete && (obsolete->rearm() ||
                         obsolete->query() != GuestWriteWatchQuery::Dirty)) _exit(124);
        auto* byte = reinterpret_cast<volatile uint8_t*>(address);
        if (write) *byte = 0x77;
        else { const uint8_t value = *byte; (void)value; }
        if (tracked && tracked->query() != GuestWriteWatchQuery::Dirty) _exit(125);
        if (obsolete && (obsolete->rearm() ||
                         obsolete->query() != GuestWriteWatchQuery::Dirty)) _exit(126);
        _exit(0);
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited != child) return AccessProbe::Failed;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return AccessProbe::Allowed;
    if (WIFSIGNALED(status) && (WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS))
        return AccessProbe::Faulted;
    return AccessProbe::Failed;
}

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
    for (bool failed_reconciliation : {false, true}) for (bool late_readonly : {false, true}) {
        int protection_fd = memfd_create("prosper-ww-protection", 0);
        CHECK(protection_fd >= 0 && ftruncate(protection_fd, static_cast<off_t>(page * 3)) == 0,
              "protection-transition memfd sized");
        if (protection_fd < 0) return 1;
        auto* writable = static_cast<uint8_t*>(mmap(
            nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_SHARED, protection_fd, 0));
        CHECK(writable != MAP_FAILED, "protection-transition writable alias mapped");
        if (writable == MAP_FAILED) return 1;
        uint8_t* readonly = nullptr;
        const uint64_t protection_phys = (late_readonly ? 0xc30000 : 0xc20000) +
                                         (failed_reconciliation ? 0x20000 : 0);
        constexpr uint64_t stale_alias = 0x700000008000ull;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(writable), page * 3, protection_phys, kCpuRw);
        auto add_readonly = [&] {
            readonly = static_cast<uint8_t*>(mmap(
                nullptr, page * 3, PROT_READ, MAP_SHARED, protection_fd, 0));
            CHECK(readonly != MAP_FAILED, "protection-transition read-only alias mapped");
            if (readonly == MAP_FAILED) return false;
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(readonly), page * 3, protection_phys, 0x1);
            return true;
        };
        if (!late_readonly && !add_readonly()) return 1;
        auto affected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page), page);
        auto retained = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page + 8), 16);
        auto unaffected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable), page);
        CHECK(affected && retained && unaffected, "two middle-page owners and disjoint physical range watched");
        if (late_readonly && !add_readonly()) return 1;
        CHECK(affected.query() == GuestWriteWatchQuery::Unchanged &&
                  retained.query() == GuestWriteWatchQuery::Unchanged,
              "early or late read-only alias leaves both middle-page watches initially clean");
        CHECK(probe_access(readonly + page + 32, false) == AccessProbe::Allowed &&
                  probe_access(readonly + page + 32, true) == AccessProbe::Faulted,
              "read-only alias probe can read but cannot write before upgrade");
        if (failed_reconciliation) {
            CHECK(probe_access(reinterpret_cast<uint8_t*>(stale_alias), false) == AccessProbe::Faulted,
                  "failed-reconciliation control address is inaccessible");
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                stale_alias, page, protection_phys + page, kCpuRw);
        }
        CHECK(mprotect(readonly + page, page, PROT_READ | PROT_WRITE) == 0,
              "middle page of read-only alias becomes writable");
        prosper::host::guest_write_watch_notify_direct_mapping_protection(
            reinterpret_cast<uint64_t>(readonly + page), page, kCpuRw);
        CHECK(!affected.rearm() && affected.query() == GuestWriteWatchQuery::Dirty,
              "changed sibling-alias protection requires a fresh registration");
        CHECK(!retained.rearm() && retained.query() == GuestWriteWatchQuery::Dirty,
              "protection upgrade invalidates every existing owner of the middle physical page");
        CHECK(unaffected.rearm() && unaffected.query() == GuestWriteWatchQuery::Unchanged,
              "partial protection change leaves disjoint physical registration rearmable");
        affected.reset();
        affected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page), page);
        if (failed_reconciliation) {
            CHECK(!affected.rearm() && affected.query() != GuestWriteWatchQuery::Unchanged,
                  "failed sibling protection reconciliation cannot publish a clean replacement watch");
            prosper::host::guest_write_watch_notify_direct_mapping_removed(stale_alias, page);
            if (!affected)
                affected = GuestWriteWatch::create(reinterpret_cast<uint64_t>(writable + page), page);
        }
        CHECK(affected && affected.rearm() && affected.query() == GuestWriteWatchQuery::Unchanged,
              "replacement starts clean with current alias permissions while another owner retains the page");
        CHECK(probe_access(readonly + page + 32, true, &affected, &retained) == AccessProbe::Allowed &&
                  writable[page + 32] == 0x77,
              "raw upgraded-alias store dirties the fresh child watch and leaves the retained owner invalid");
        CHECK(!retained.rearm() && retained.query() == GuestWriteWatchQuery::Dirty,
              "recreating one owner never revives the other obsolete registration");
        CHECK(unaffected.query() == GuestWriteWatchQuery::Unchanged &&
                  probe_access(readonly + 32, true) == AccessProbe::Faulted &&
                  probe_access(readonly + page * 2 + 32, true) == AccessProbe::Faulted &&
                  probe_access(writable + page * 2 + 32, true) == AccessProbe::Allowed,
              "middle-page upgrade preserves neighboring permissions and the disjoint watch");
        affected.reset();
        retained.reset();
        unaffected.reset();
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(writable), page * 3);
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(readonly), page * 3);
        munmap(writable, page * 3);
        munmap(readonly, page * 3);
        close(protection_fd);
    }

    // A permission downgrade must survive disarming, shared-owner replacement, and last release.
    // The writable sibling still supports a watch; it must not grant B permissions the guest revoked.
    for (int requested : {PROT_READ, PROT_NONE}) {
        int downgrade_fd = memfd_create("prosper-ww-downgrade", 0);
        CHECK(downgrade_fd >= 0 && ftruncate(downgrade_fd, static_cast<off_t>(page * 3)) == 0,
              "downgrade memfd sized");
        if (downgrade_fd < 0) return 1;
        auto* first = static_cast<uint8_t*>(mmap(
            nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_SHARED, downgrade_fd, 0));
        auto* second = static_cast<uint8_t*>(mmap(
            nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_SHARED, downgrade_fd, 0));
        CHECK(first != MAP_FAILED && second != MAP_FAILED, "downgrade aliases mapped");
        if (first == MAP_FAILED || second == MAP_FAILED) return 1;
        const uint64_t phys = requested == PROT_READ ? 0xc60000 : 0xc70000;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(first), page * 3, phys, kCpuRw);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(second), page * 3, phys, kCpuRw);
        CHECK(probe_access(second + page + 32, true) == AccessProbe::Allowed,
              "downgrade write probe succeeds on the initial writable alias");
        auto replaced = GuestWriteWatch::create(reinterpret_cast<uint64_t>(first + page), page);
        auto retained = GuestWriteWatch::create(reinterpret_cast<uint64_t>(first + page + 8), 16);
        auto outer = GuestWriteWatch::create(reinterpret_cast<uint64_t>(first), page);
        CHECK(replaced && retained && outer &&
                  replaced.query() == GuestWriteWatchQuery::Unchanged &&
                  retained.query() == GuestWriteWatchQuery::Unchanged,
              "downgrade starts with two clean middle owners and a disjoint outer owner");
        auto check_permissions = [&](const char* stage) {
            const AccessProbe read_result = probe_access(second + page + 32, false);
            const AccessProbe write_result = probe_access(second + page + 32, true);
            const bool correct = read_result == (requested == PROT_READ
                ? AccessProbe::Allowed : AccessProbe::Faulted) && write_result == AccessProbe::Faulted;
            if (!correct) std::fprintf(stderr, "  downgrade prot=%d stage=%s\n", requested, stage);
            CHECK(correct, "downgraded alias retains exact actual read/write permissions");
            CHECK(outer.query() == GuestWriteWatchQuery::Unchanged &&
                      probe_access(second + 32, false) == AccessProbe::Allowed &&
                      probe_access(second + 32, true) == AccessProbe::Faulted &&
                      probe_access(second + page * 2 + 32, true) == AccessProbe::Allowed,
                  "middle-page downgrade preserves the armed left neighbor and writable right neighbor");
        };
        CHECK(mprotect(second + page, page, requested) == 0, "guest downgrades only the middle alias page");
        check_permissions("before notification");
        prosper::host::guest_write_watch_notify_direct_mapping_protection(
            reinterpret_cast<uint64_t>(second + page), page, requested == PROT_READ ? 0x1 : 0x0);
        check_permissions("after notification");
        CHECK(!replaced.rearm() && !retained.rearm() &&
                  replaced.query() == GuestWriteWatchQuery::Dirty &&
                  retained.query() == GuestWriteWatchQuery::Dirty,
              "downgrade invalidates both old middle registrations");
        replaced.reset();
        replaced = GuestWriteWatch::create(reinterpret_cast<uint64_t>(first + page), page);
        if (requested == PROT_NONE) {
            CHECK(!replaced && !replaced.rearm() &&
                      replaced.query() == GuestWriteWatchQuery::Unknown,
                  "an inaccessible physical alias keeps replacement watches on the exact fallback");
        } else {
            CHECK(replaced && replaced.rearm() && replaced.query() == GuestWriteWatchQuery::Unchanged,
                  "fresh middle watch can arm through the still-writable sibling");
        }
        check_permissions("after one-owner replacement");
        replaced.reset();
        retained.reset();
        check_permissions("after last middle owner release");
        CHECK(probe_access(first + page + 32, true) == AccessProbe::Allowed,
              "last middle owner release restores the sibling's legitimate write permission");
        outer.reset();
        CHECK(probe_access(second + 32, true) == AccessProbe::Allowed,
              "outer owner release restores its unchanged writable alias");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(first), page * 3);
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(second), page * 3);
        munmap(first, page * 3);
        munmap(second, page * 3);
        close(downgrade_fd);
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

    // Last-owner release can fail to restore RW when one recorded sibling no longer maps. Its
    // transactional rollback leaves the real alias RO: zero references must not erase the only
    // fault-recovery metadata while that protection remains installed.
    {
        int recovery_fd = memfd_create("prosper-ww-release-recovery", 0);
        CHECK(recovery_fd >= 0 && ftruncate(recovery_fd, static_cast<off_t>(page)) == 0,
              "last-release recovery memfd sized");
        if (recovery_fd < 0) return 1;
        auto* real = static_cast<uint8_t*>(mmap(
            nullptr, page, PROT_READ | PROT_WRITE, MAP_SHARED, recovery_fd, 0));
        CHECK(real != MAP_FAILED, "last-release recovery alias mapped");
        if (real == MAP_FAILED) return 1;
        constexpr uint64_t recovery_phys = 0xc80000;
        constexpr uint64_t missing_alias = 0x700000010000ull;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(real), page, recovery_phys, kCpuRw);
        CHECK(probe_access(real + 32, true) == AccessProbe::Allowed,
              "last-release recovery alias is initially writable without a fault handler");
        auto released = GuestWriteWatch::create(reinterpret_cast<uint64_t>(real), page);
        CHECK(released && released.query() == GuestWriteWatchQuery::Unchanged,
              "last-release recovery watch begins clean");
        CHECK(probe_access(real + 32, true) == AccessProbe::Faulted,
              "last-release recovery control proves the watch actually armed the real page");
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            missing_alias, page, recovery_phys, kCpuRw);
        CHECK(!released.rearm() && released.query() == GuestWriteWatchQuery::Dirty,
              "missing sibling makes the release-recovery watch fail closed");
        released.reset();
        CHECK(!released && released.query() == GuestWriteWatchQuery::Unknown,
              "last-release recovery handle relinquishes its registration");
        CHECK(probe_access(real + 40, true) == AccessProbe::Faulted,
              "failed last-owner disarm leaves a real RO page requiring fault recovery");
        CHECK(probe_access(real + 40, true, nullptr, nullptr, true) == AccessProbe::Allowed &&
                  real[40] == 0x77,
              "production fault handler recovers a raw store after failed last-owner release");
        // The child handled its own copied protection state. Repair the parent's topology too,
        // then require a genuinely fresh clean watch and another raw-store dirty witness.
        prosper::host::guest_write_watch_notify_direct_mapping_removed(missing_alias, page);
        auto repaired = GuestWriteWatch::create(reinterpret_cast<uint64_t>(real), page);
        CHECK(repaired && repaired.rearm() && repaired.query() == GuestWriteWatchQuery::Unchanged,
              "removing the missing sibling permits a fresh exact watch after failed release");
        CHECK(probe_access(real + 48, true, &repaired) == AccessProbe::Allowed && real[48] == 0x77,
              "fresh watch after release recovery catches a real store from an Unchanged baseline");
        repaired.reset();
        CHECK(probe_access(real + 56, true) == AccessProbe::Allowed,
              "successful final release restores actual RW permission after recovery");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(real), page);
        munmap(real, page);
        close(recovery_fd);
    }

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

            // The fake VA is intentionally not mapped. Its failed protection update must prevent a
            // clean verdict, even though the real alias remains armed and can still fault below.
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                kStaleAlias, page, kGuestFsPhys, kCpuRw);
            CHECK(!guest_watch.rearm(), "guest-FS watch refuses incomplete stale-alias coverage");
            CHECK(guest_watch.query() == GuestWriteWatchQuery::Dirty,
                  "guest-FS watch fails closed before the real alias fault");

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
