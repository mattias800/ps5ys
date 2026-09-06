#include "host/memory/guest_memory_query.hpp"

#include <cstdio>
#include <initializer_list>
#include <limits>

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using prosper::host::GuestWritableQueryResult;
using prosper::host::GuestWritableQueryStatus;
using Status = GuestWritableQueryStatus;

struct Checks {
    unsigned count = 0;
    unsigned failures = 0;
    void check(bool condition, const char* name) {
        ++count;
        std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", name);
        if (!condition) ++failures;
    }
};

struct Step {
    uint64_t cursor;
    GuestWritableQueryResult result;
};

void synthetic(Checks& checks, const char* name, uint64_t begin, uint64_t end,
               std::initializer_list<Step> steps, GuestWritableQueryResult expected) {
    std::printf("-- %s --\n", name);
    auto next = steps.begin();
    const auto result = prosper::host::detail::query_contiguous_writable_range(
        begin, end, [&](uint64_t cursor) -> GuestWritableQueryResult {
            const bool available = next != steps.end();
            checks.check(available, "walker requests only expected regions");
            // Also bounds a broken walker in the negative/malformed cases.
            if (!available) return {Status::Unavailable};
            const Step step = *next++;
            checks.check(cursor == step.cursor, "query cursor is exact uncovered boundary");
            return step.result;
        });
    checks.check(next == steps.end(), "walker consumes the expected query sequence");
    checks.check(result.status == expected.status, "result status preserves query contract");
    checks.check(result.begin == expected.begin && result.end == expected.end,
                 "only complete positive proof publishes bounds");
}

void synthetic_tests(Checks& checks) {
    synthetic(checks, "single VMA retains full proven extent", 12, 18,
              {{12, {Status::Writable, 10, 20}}}, {Status::Writable, 10, 20});
    synthetic(checks, "adjacent VMAs cover complete request", 12, 38,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::Writable, 20, 30}},
               {30, {Status::Writable, 30, 40}}}, {Status::Writable, 10, 40});
    synthetic(checks, "exclusive endpoint needs no extra VMA", 12, 20,
              {{12, {Status::Writable, 10, 20}}}, {Status::Writable, 10, 20});
    synthetic(checks, "covering region may begin before subsequent cursor", 12, 30,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::Writable, 15, 35}}}, {Status::Writable, 10, 35});
    synthetic(checks, "NEXT result cannot bridge gap after prefix", 12, 38,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::Writable, 25, 40}}}, {Status::Unavailable});
    synthetic(checks, "NEXT result cannot replace first covering VMA", 12, 18,
              {{12, {Status::Writable, 15, 20}}}, {Status::Unavailable});
    synthetic(checks, "read-only covering VMA refuses request", 12, 18,
              {{12, {Status::NotWritable, 10, 20}}}, {Status::NotWritable});
    synthetic(checks, "non-writable suffix discards valid prefix", 12, 38,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::NotWritable, 20, 40}}}, {Status::NotWritable});
    synthetic(checks, "query error discards valid prefix", 12, 38,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::Unavailable, 20, 40}}}, {Status::Unavailable});
    synthetic(checks, "initial query error publishes nothing", 12, 18,
              {{12, {Status::Unavailable, 10, 20}}}, {Status::Unavailable});
    synthetic(checks, "empty request issues no queries", 20, 20, {}, {Status::NotWritable});
    synthetic(checks, "reversed request issues no queries", 21, 20, {}, {Status::NotWritable});
    synthetic(checks, "zero-width VMA cannot make progress", 12, 18,
              {{12, {Status::Writable, 12, 12}}}, {Status::Unavailable});
    synthetic(checks, "backward VMA endpoint is malformed", 12, 18,
              {{12, {Status::Writable, 10, 11}}}, {Status::Unavailable});
    synthetic(checks, "reversed VMA is malformed", 12, 18,
              {{12, {Status::Writable, 13, 11}}}, {Status::Unavailable});
    synthetic(checks, "repeated prefix cannot loop or publish", 12, 38,
              {{12, {Status::Writable, 10, 20}},
               {20, {Status::Writable, 10, 20}}}, {Status::Unavailable});
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    synthetic(checks, "maximum exact endpoint does not wrap", maximum - 2, maximum,
              {{maximum - 2, {Status::Writable, maximum - 3, maximum - 1}},
               {maximum - 1, {Status::Writable, maximum - 1, maximum}}},
              {Status::Writable, maximum - 3, maximum});
}

#if defined(__linux__) && defined(PROCMAP_QUERY)
struct OwnedMapping {
    void* address;
    size_t size;
    ~OwnedMapping() { if (address != MAP_FAILED) munmap(address, size); }
};

void real_linux_tests(Checks& checks) {
    const long page_size = sysconf(_SC_PAGESIZE);
    checks.check(page_size > 0, "real mapping has a valid host page size");
    if (page_size <= 0) return;
    const size_t page = static_cast<size_t>(page_size);
    OwnedMapping mapping{mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), 3 * page};
    checks.check(mapping.address != MAP_FAILED, "owned three-page mapping created");
    if (mapping.address == MAP_FAILED) return;
    const uint64_t begin = reinterpret_cast<uintptr_t>(mapping.address);
    const uint64_t end = begin + mapping.size;
    void* const middle = static_cast<char*>(mapping.address) + page;

    // Independently probe availability: an adapter that always returns Unavailable
    // must fail on a supporting kernel, rather than turning its own bug into a skip.
    const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    checks.check(fd >= 0, "independent capability probe opens own maps");
    if (fd < 0) return;
    procmap_query probe{};
    probe.size = sizeof(probe);
    probe.query_flags = PROCMAP_QUERY_VMA_WRITABLE;
    probe.query_addr = begin;
    const int probe_result = ioctl(fd, PROCMAP_QUERY, &probe);
    const int probe_errno = errno;
    close(fd);
    if (probe_result != 0) {
        std::printf("REAL PROCMAP_QUERY: unavailable (errno=%d: %s); "
                    "synthetic contracts still executed\n", probe_errno, std::strerror(probe_errno));
        checks.check(probe_errno != ENOENT,
                     "known writable owned mapping is not a definite negative");
        const auto result = prosper::host::query_guest_writable_range(begin, end);
        checks.check(result.status == Status::Unavailable && result.begin == 0 && result.end == 0,
                     "unavailable interface requests fallback without positive bounds");
        return;
    }
    std::puts("REAL PROCMAP_QUERY: available; executing real permission and gap controls");
    checks.check(probe.vma_start <= begin && begin < probe.vma_end &&
                     (probe.vma_flags & PROCMAP_QUERY_VMA_WRITABLE),
                 "independent kernel result covers writable owned address");
    auto expect = [&](uint64_t lo, uint64_t hi, Status status, const char* name) {
        const auto result = prosper::host::query_guest_writable_range(lo, hi);
        checks.check(result.status == status, name);
        if (status == Status::Writable) {
            checks.check(result.begin <= lo && hi <= result.end && result.begin < result.end,
                         "real positive bounds cover the complete requested span");
        } else {
            checks.check(result.begin == 0 && result.end == 0,
                         "real refusal exposes no partial positive bounds");
        }
    };
    expect(begin, end, Status::Writable, "real initial RW span is writable");
    expect(begin, begin, Status::NotWritable, "real empty range refuses");
    expect(end, begin, Status::NotWritable, "real reversed range refuses");

    const int read_only = mprotect(middle, page, PROT_READ);
    checks.check(read_only == 0, "owned middle page downgraded to read-only");
    if (read_only != 0) return;
    expect(begin, begin + page, Status::Writable, "range ending at RO boundary succeeds");
    expect(begin + page, begin + 2 * page, Status::NotWritable, "RO covering VMA refuses");
    expect(begin + page / 2, end - page / 2, Status::NotWritable,
           "RO suffix cannot publish earlier writable prefix");
    expect(begin + 2 * page, end, Status::Writable, "later writable VMA remains queryable");

    const int write_only = mprotect(middle, page, PROT_WRITE);
    checks.check(write_only == 0, "owned middle page changed to write-only");
    if (write_only != 0) return;
    expect(begin + page, begin + 2 * page, Status::Writable,
           "write-only VMA does not require the READABLE query flag");
    expect(begin + page / 2, end - page / 2, Status::Writable,
           "adjacent RW/W/RW VMAs cover the complete range");

    const int removed = munmap(middle, page);
    checks.check(removed == 0, "owned middle page removed to create an actual hole");
    if (removed != 0) return;
    expect(begin + page, begin + 2 * page, Status::NotWritable,
           "unmapped covering address refuses despite later writable VMA");
    expect(begin + page / 2, end - page / 2, Status::NotWritable,
           "unmapped gap discards earlier writable prefix");
    expect(begin + 2 * page, end, Status::Writable, "mapped suffix survives middle-page removal");
    // Linux munmap accepts the now-unmapped middle page during RAII cleanup.
}
#else
void real_linux_tests(Checks& checks) {
    std::puts("REAL PROCMAP_QUERY: unavailable (build platform/UAPI headers); "
              "synthetic contracts still executed");
    const auto result = prosper::host::query_guest_writable_range(0x1000, 0x2000);
    checks.check(result.status == Status::Unavailable && result.begin == 0 && result.end == 0,
                 "build without binary query support explicitly requests fallback");
}
#endif
} // namespace

int main() {
    Checks checks;
    std::puts("== test_guest_memory_query ==");
    synthetic_tests(checks);
    real_linux_tests(checks);
    std::printf("== %s: %u checks executed, %u failures ==\n",
                checks.failures ? "FAIL" : "PASS", checks.count, checks.failures);
    return checks.failures ? 1 : 0;
}
