// Linux integration fixture: link with --wrap=ioctl. Only owned mappings are queried or changed.
#include "gpu/execute/gpu_execute.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "host/memory/guest_memory_query.hpp"

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
int checks = 0;
int failures = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) ++failures;
    std::fprintf(stderr, "%s %s\n", condition ? "[ ok ]" : "[FAIL]", message);
}
void changed_mapping() {
    prosper::host::detail::advance_guest_mapping_generation();
}

struct Mapping {
    void* data;
    size_t size;
    explicit Mapping(size_t bytes, int protection = PROT_READ | PROT_WRITE)
        : data(mmap(nullptr, bytes, protection, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)),
          size(bytes) {
        changed_mapping();
    }
    ~Mapping() {
        if (data != MAP_FAILED) {
            munmap(data, size);
            changed_mapping();
        }
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    uint64_t address() const { return reinterpret_cast<uintptr_t>(data); }
};

enum class Mode { Pass, Error, AdvanceAfterSuccess, ErrorOnSecond, RevokeOnSecond };
Mode mode = Mode::Pass;
int forced_errno = ENOTTY;
unsigned query_calls = 0;
unsigned query_successes = 0;
unsigned generation_advances = 0;
void* revoke_address = nullptr;
size_t revoke_bytes = 0;
int revoke_result = -1;

void reset_probe(Mode next = Mode::Pass, int error = ENOTTY) {
    mode = next;
    forced_errno = error;
    query_calls = query_successes = generation_advances = 0;
    revoke_result = -1;
    changed_mapping(); // Every arm starts with an actually invalidated writable cache.
}
void check_one_unavailable_probe() {
#ifdef PROCMAP_QUERY
    check(query_calls == 1 && query_successes == 0,
          "refusal exercised exactly one injected unavailable adapter call");
#else
    check(query_calls == 0, "old-header refusal used no adapter ioctl");
#endif
}
} // namespace

extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
    // These legacy requests have no third argument. Never fetch a nonexistent vararg.
    // The fixture also tests pointer-bearing legacy FIONREAD forwarding below.
    if (request == FIOCLEX || request == FIONCLEX)
        return __real_ioctl(fd, request);

    va_list arguments;
    va_start(arguments, request);
    void* argument = va_arg(arguments, void*);
    va_end(arguments);
#ifdef PROCMAP_QUERY
    if (request == PROCMAP_QUERY) {
        ++query_calls;
        if (mode == Mode::Error ||
            ((mode == Mode::ErrorOnSecond || mode == Mode::RevokeOnSecond) &&
             query_calls == 2)) {
            if (mode == Mode::RevokeOnSecond) {
                revoke_result = mprotect(revoke_address, revoke_bytes, PROT_READ);
                if (revoke_result == 0) changed_mapping();
            }
            errno = forced_errno;
            return -1;
        }
        const int result = __real_ioctl(fd, request, argument);
        const int saved_errno = errno;
        if (result == 0) {
            ++query_successes;
            if (mode == Mode::AdvanceAfterSuccess && generation_advances == 0) {
                changed_mapping();
                ++generation_advances;
            }
        }
        errno = saved_errno;
        return result;
    }
#endif
    return __real_ioctl(fd, request, argument);
}

int main() {
    std::fprintf(stderr, "== test_guest_writable_query_adapter ==\n");
    // CTest supplies a clean environment before process/TLS initialization. Refuse a conflicting
    // direct invocation rather than silently changing the production bisection setting in main.
    const bool cache_enabled = std::getenv("PROSPER_NO_GUEST_WRITE_CACHE") == nullptr;
    check(cache_enabled, "required cache-enabled environment is established before startup");
    if (!cache_enabled) return 1;
    const long page_value = sysconf(_SC_PAGESIZE);
    check(page_value > 0, "host page size is available");
    if (page_value <= 0) return 1;
    const size_t page = static_cast<size_t>(page_value);

    int pipe_fds[2];
    const bool pipe_ok = pipe(pipe_fds) == 0;
    check(pipe_ok, "owned pipe for unrelated ioctl forwarding");
    if (pipe_ok) {
        check(ioctl(pipe_fds[0], FIOCLEX) == 0, "forward no-argument FIOCLEX without reading varargs");
        check((fcntl(pipe_fds[0], F_GETFD) & FD_CLOEXEC) != 0, "real FIOCLEX changed descriptor flags");
        check(ioctl(pipe_fds[0], FIONCLEX) == 0, "forward no-argument FIONCLEX");
        int available = -1;
        check(ioctl(pipe_fds[0], FIONREAD, &available) == 0 && available == 0,
              "forward pointer-bearing legacy FIONREAD to the real ioctl");
        errno = 0;
        check(ioctl(-1, FIONREAD, &available) == -1 && errno == EBADF,
              "unrelated ioctl forwarding preserves real errno");
        close(pipe_fds[0]);
        close(pipe_fds[1]);
    }

    Mapping pages(3 * page);
    check(pages.data != MAP_FAILED, "owned three-page mapping");
    if (pages.data == MAP_FAILED) return 1;
    auto* base = static_cast<unsigned char*>(pages.data);
    const uint64_t address = pages.address();
    check(mprotect(base + page, page, PROT_READ) == 0, "middle page is read-only");
    changed_mapping();

    reset_probe();
    const auto support = prosper::host::query_guest_writable_range(address, address + 8);
    const bool native = support.status == prosper::host::GuestWritableQueryStatus::Writable;
#ifdef PROCMAP_QUERY
    check(query_calls == 1, "compiled adapter invokes one real covering-VMA ioctl");
    check(native || support.status == prosper::host::GuestWritableQueryStatus::Unavailable,
          "owned RW control is either native writable or explicitly adapter-unavailable");
    if (native)
        check(query_successes == 1, "native capability control completed a real successful ioctl");
    else
        std::fprintf(stderr, "[coverage] Native ioctl unavailable on this kernel/policy; native-success "
                             "and generation-race cases are NOT exercised. Forced-error integration remains tested.\n");
#else
    check(!native && support.status == prosper::host::GuestWritableQueryStatus::Unavailable,
          "old headers explicitly report adapter unavailable");
    check(query_calls == 0, "old-header path performs no adapter ioctl");
    std::fprintf(stderr, "[coverage] PROCMAP_QUERY absent from headers: only existing text fallback "
                         "is exercised; no binary-adapter or injected-error coverage is claimed.\n");
#endif

    reset_probe();
    check(prosper::gpu::guest_writable(address, 8), "normal RW query succeeds");
    const unsigned warmed_calls = query_calls;
    check(prosper::gpu::guest_writable(address + 16, 8), "normal query warms a wider writable span");
    check(query_calls == warmed_calls, "cache hit makes no additional adapter call");
#ifdef PROCMAP_QUERY
    check(warmed_calls == 1, "normal cold query attempted the adapter exactly once");
#else
    check(warmed_calls == 0, "normal old-header fallback makes no adapter calls");
#endif

    for (int error : {ENOTTY, EPERM}) {
        std::fprintf(stderr, "[arm] unavailable errno=%d\n", error);
        reset_probe(Mode::Error, error);
        check(prosper::gpu::guest_writable(address, 8), "unavailable adapter falls back and proves RW");
#ifdef PROCMAP_QUERY
        check(query_calls == 1 && query_successes == 0, "forced unavailable ioctl reached, not bypassed");
#else
        check(query_calls == 0, "old-header fallback does not pretend to inject ioctl errors");
#endif
        reset_probe(Mode::Error, error);
        check(!prosper::gpu::guest_writable(address + page, 8), "text fallback refuses read-only page");
        check_one_unavailable_probe();
        reset_probe(Mode::Error, error);
        check(!prosper::gpu::guest_writable(address + page - 8, 16),
              "text fallback refuses request crossing writable/read-only boundary");
        check_one_unavailable_probe();
    }

#ifdef PROCMAP_QUERY
    reset_probe(Mode::Error, ENOENT);
    check(!prosper::gpu::guest_writable(address, 8), "ENOENT is definitive negative even for real RW mapping");
    check(query_calls == 1 && query_successes == 0, "definitive negative came from injected adapter result");
    check(!prosper::gpu::guest_writable(address, 8), "definitive negative was not cached as writable");
    check(query_calls == 2, "negative request probes again instead of receiving a positive cache hit");

    if (native) {
        reset_probe(Mode::AdvanceAfterSuccess);
        const uint64_t before = prosper::host::guest_mapping_generation();
        check(prosper::gpu::guest_writable(address, 8), "first query succeeds while wrapper advances generation");
        check(query_calls == 1 && query_successes == 1 && generation_advances == 1,
              "race control advanced generation after an actual successful ioctl");
        check(prosper::host::guest_mapping_generation() == before + 1, "race advances exactly one epoch");
        check(prosper::gpu::guest_writable(address, 8), "next query remains writable");
        check(query_calls == 2 && query_successes == 2,
              "pre-probe epoch retained: next call reprobes rather than trusting post-probe retag");
        check(prosper::gpu::guest_writable(address + 16, 8), "stable subsequent subrange succeeds");
        check(query_calls == 2, "stable reprobe warms cache without a third ioctl");
    }
#endif

    // Remove only the middle page of our reservation; surrounding owned pages remain writable.
    check(munmap(base + page, page) == 0, "owned middle page becomes a genuine unmapped gap");
    changed_mapping();
    for (int error : {ENOTTY, EPERM}) {
        reset_probe(Mode::Error, error);
        check(!prosper::gpu::guest_writable(address + page - 8, static_cast<uint32_t>(page + 16)),
              "text fallback refuses an unmapped gap even with writable pages on both sides");
        check_one_unavailable_probe();
    }

#ifdef PROCMAP_QUERY
    if (native) {
        Mapping adjacent(2 * page);
        check(adjacent.data != MAP_FAILED, "owned reservation for distinct adjacent writable VMAs");
        if (adjacent.data != MAP_FAILED) {
            auto* second = static_cast<unsigned char*>(adjacent.data) + page;
            // MAP_FIXED is restricted to the second page of this owned reservation. Shared/private
            // backing distinguishes the VMAs without needing executable memory or external files.
            void* shared = mmap(second, page, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            check(shared == second, "owned second page remapped shared to prevent VMA coalescing");
            changed_mapping();
            if (shared == second) {
                const uint64_t start = adjacent.address() + page - 8;
                reset_probe(Mode::ErrorOnSecond, EPERM);
                check(prosper::gpu::guest_writable(start, 16),
                      "partial adapter error falls back and proves the entire two-VMA request");
                check(query_calls == 2 && query_successes == 1,
                      "partial-error control completed first real VMA query then injected second error");

                revoke_address = adjacent.data;
                revoke_bytes = page;
                reset_probe(Mode::RevokeOnSecond, EPERM);
                check(!prosper::gpu::guest_writable(start, 16),
                      "fallback rechecks ORIGINAL request: now-read-only prefix must refuse writable tail");
                check(query_calls == 2 && query_successes == 1 && revoke_result == 0,
                      "prefix revocation occurred only after first real query and before second-query error");
            }
        }
    }
#endif
    mode = Mode::Pass;
    std::fprintf(stderr, "== %s: %d assertions, %d failures; native-success coverage %s ==\n",
                 failures ? "FAIL" : "PASS", checks, failures, native ? "YES" : "NO");
    return failures ? 1 : 0;
}
