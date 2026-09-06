#include "host/memory/guest_write_watch.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

std::atomic<int64_t> writer_tid{0};
std::atomic<bool> request_contention{false};
std::atomic<bool> contention_held{false};
std::atomic<bool> contention_hook_entered{false};
std::atomic<bool> contention_released{false};
std::atomic<bool> unrelated_lock_held{false};
std::atomic<bool> release_unrelated_lock{false};
std::atomic<bool> unrelated_contention_hook_entered{false};
std::atomic<bool> stepping_rearm_result{true};
std::atomic<bool> stepping_kernel_write_succeeded{false};
std::atomic<bool> pause_invalidation_step{false};
std::atomic<bool> invalidation_step_ready{false};
std::atomic<bool> release_invalidation_step{false};
std::atomic<bool> invalidation_canary_active{false};
std::atomic<bool> pause_reconfigure_step{false};
std::atomic<bool> reconfigure_step_ready{false};
std::atomic<bool> release_reconfigure_step{false};
std::atomic<bool> reconfigure_canary_active{false};
std::atomic<bool> pause_unknown_step{false};
std::atomic<bool> unknown_step_ready{false};
std::atomic<bool> release_unknown_step{false};
std::atomic<bool> preowned_tf_active{false};
std::atomic<bool> preowned_tf_started{false};
std::atomic<uint32_t> preowned_tf_completed{0};
std::atomic<bool> legacy_bool_fault_active{false};
std::atomic<uint32_t> dynamic_protection_helper_entries{0};
std::atomic<int> last_step_action{-1};
std::atomic<uint32_t> trap_count{0};
volatile sig_atomic_t post_store_marker = 0;
volatile sig_atomic_t marker_when_first_completed = -1;

void contention_hook() {
    contention_hook_entered.store(true, std::memory_order_release);
    while (!contention_released.load(std::memory_order_acquire)) sched_yield();
}

void unrelated_contention_hook() {
    unrelated_contention_hook_entered.store(true, std::memory_order_release);
    // Let a mutant that enters the lock path terminate deterministically instead of exhausting the
    // full retry bound. The named assertion below still proves that production never invoked us.
    release_unrelated_lock.store(true, std::memory_order_release);
}

void dynamic_protection_hook() {
    dynamic_protection_helper_entries.fetch_add(1, std::memory_order_relaxed);
}

// Kernel copies obey the real VMA permissions without entering our SIGSEGV handler. Probe both
// directions: rejecting a write alone cannot distinguish the requested RO from accidental NONE.
void check_kernel_permissions(uint8_t* address, bool readable, bool writable) {
    int probe[2] = {-1, -1};
    const int opened = pipe(probe);
    CHECK(opened == 0, "permission probe pipe created");
    if (opened != 0) return;
    const uint8_t payload = 0x31;
    CHECK(write(probe[1], &payload, 1) == 1, "permission probe payload queued");
    errno = 0;
    const ssize_t copied_to_page = read(probe[0], address, 1);
    const int write_error = errno;
    CHECK(writable ? copied_to_page == 1 : copied_to_page == -1 && write_error == EFAULT,
          "kernel write probe matches requested alias permission");
    errno = 0;
    const ssize_t copied_from_page = write(probe[1], address, 1);
    const int read_error = errno;
    CHECK(readable ? copied_from_page == 1 : copied_from_page == -1 && read_error == EFAULT,
          "kernel read probe distinguishes requested RO from NONE");
    close(probe[0]);
    close(probe[1]);
}

void trace_signal_handler(int sig, siginfo_t* info, void* raw_context) {
    auto* context = static_cast<ucontext_t*>(raw_context);
    const int64_t tid = static_cast<int64_t>(syscall(SYS_gettid));
    if (sig == SIGTRAP && preowned_tf_active.load(std::memory_order_acquire) &&
        !preowned_tf_started.load(std::memory_order_acquire) &&
        (!info || info->si_code != TRAP_TRACE)) {
        preowned_tf_started.store(true, std::memory_order_release);
        context->uc_mcontext.gregs[REG_EFL] |= static_cast<greg_t>(0x100);
        return;
    }
    if (sig == SIGSEGV && info && info->si_addr &&
        (context->uc_mcontext.gregs[REG_ERR] & 2)) {
        if (legacy_bool_fault_active.load(std::memory_order_acquire) &&
            prosper::host::guest_write_watch_handle_fault(
                reinterpret_cast<uint64_t>(info->si_addr)))
            return;
        const auto action = prosper::host::guest_write_watch_handle_fault_ex(
            reinterpret_cast<uint64_t>(info->si_addr),
            static_cast<uint64_t>(context->uc_mcontext.gregs[REG_RIP]), tid,
            (context->uc_mcontext.gregs[REG_EFL] & 0x100) != 0);
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep)
            context->uc_mcontext.gregs[REG_EFL] |= 0x100;
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            pause_invalidation_step.exchange(false, std::memory_order_acq_rel)) {
            invalidation_step_ready.store(true, std::memory_order_release);
            while (!release_invalidation_step.load(std::memory_order_acquire)) sched_yield();
        }
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            pause_reconfigure_step.exchange(false, std::memory_order_acq_rel)) {
            reconfigure_step_ready.store(true, std::memory_order_release);
            while (!release_reconfigure_step.load(std::memory_order_acquire)) sched_yield();
        }
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            pause_unknown_step.exchange(false, std::memory_order_acq_rel)) {
            unknown_step_ready.store(true, std::memory_order_release);
            while (!release_unknown_step.load(std::memory_order_acquire)) sched_yield();
        }
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            !request_contention.exchange(true, std::memory_order_acq_rel)) {
            while (!contention_held.load(std::memory_order_acquire)) sched_yield();
        }
        if (action != prosper::host::GuestWriteWatchFaultAction::NotHandled) return;
    }
    if (sig == SIGTRAP && info && info->si_code == TRAP_TRACE) {
        trap_count.fetch_add(1, std::memory_order_relaxed);
        prosper::host::GuestDmemWriteTraceEvent event{};
        const auto action = prosper::host::guest_dmem_write_trace_complete_step(
            tid, static_cast<uint64_t>(context->uc_mcontext.gregs[REG_RIP]), event);
        last_step_action.store(static_cast<int>(action), std::memory_order_release);
        if (action == prosper::host::GuestDmemWriteTraceStepAction::LockTimeout) _exit(92);
        if (action == prosper::host::GuestDmemWriteTraceStepAction::Complete ||
            action == prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) {
            if (action == prosper::host::GuestDmemWriteTraceStepAction::Complete &&
                event.ordinal == 1 && marker_when_first_completed == -1)
                marker_when_first_completed = post_store_marker;
            context->uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(0x100);
            return;
        }
        // A targeted mutation that drops pending ownership must fail the named assertion below rather
        // than trapping forever. Production has no such compensation and would take its fatal path.
        if (action == prosper::host::GuestDmemWriteTraceStepAction::NotHandled &&
            (invalidation_canary_active.load(std::memory_order_acquire) ||
             reconfigure_canary_active.load(std::memory_order_acquire))) {
            context->uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(0x100);
            return;
        }
        if (action == prosper::host::GuestDmemWriteTraceStepAction::NotHandled &&
            preowned_tf_active.load(std::memory_order_acquire) &&
            preowned_tf_started.load(std::memory_order_acquire)) {
            preowned_tf_completed.fetch_add(1, std::memory_order_relaxed);
            context->uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(0x100);
            return;
        }
    }
    _exit(90);
}

void install_altstack() {
    // Intentionally leaked until thread exit: the kernel may still reference it until then.
    auto* storage = new std::vector<uint8_t>(128 * 1024);
    stack_t stack{};
    stack.ss_sp = storage->data();
    stack.ss_size = storage->size();
    if (sigaltstack(&stack, nullptr) != 0) _exit(91);
}

__attribute__((noinline)) void store_byte(volatile uint8_t* address, uint8_t value) {
    *address = value;
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void exchange_byte(volatile uint8_t* address, uint8_t value) {
    __asm__ volatile("xchgb %0, (%1)" : "+q"(value) : "r"(address) : "memory");
}

// The marker store is the exact instruction after the protected selected-byte store. The first
// TRAP_TRACE must capture/re-arm before this instruction is allowed to execute.
__attribute__((noinline)) void store_byte_then_mark(volatile uint8_t* address, uint8_t value,
                                                   volatile sig_atomic_t* marker) {
    __asm__ volatile(
        "movb %b1, (%0)\n\t"
        "movl $1, (%2)\n\t"
        :
        : "r"(address), "q"(value), "r"(marker)
        : "memory");
}

__attribute__((noinline)) void store_qword_then_mark(volatile uint8_t* address, uint64_t value,
                                                     volatile sig_atomic_t* marker) {
    __asm__ volatile(
        "movq %1, (%0)\n\t"
        "movl $1, (%2)\n\t"
        :
        : "r"(address), "r"(value), "r"(marker)
        : "memory");
}

// Models the existing PROSPER_BP/HWBP/STEPWIN ownership sequence: the breakpoint trap sets TF,
// then the one instruction owned by that machinery writes the trace-protected page. The dmem overlay
// must open/invalidate the page without consuming the following completion trap.
__attribute__((noinline)) void breakpoint_then_store_byte(
        volatile uint8_t* address, uint8_t value, volatile sig_atomic_t* marker) {
    __asm__ volatile(
        "int3\n\t"
        "movb %b1, (%0)\n\t"
        "movl $1, (%2)\n\t"
        :
        : "r"(address), "q"(value), "r"(marker)
        : "memory");
}

} // namespace

int main() {
    constexpr uint64_t page = 0x1000;
    constexpr uint64_t allocation_size = page * 3;
    constexpr uint64_t physical = 0x740000;
    constexpr uint64_t offset = page + 0x120;
    constexpr uint32_t bytes = 16;
    constexpr uint32_t chain = 7;

    struct sigaction action{};
    action.sa_sigaction = trace_signal_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    CHECK(sigaction(SIGSEGV, &action, nullptr) == 0, "install SIGSEGV trace handler");
    CHECK(sigaction(SIGTRAP, &action, nullptr) == 0, "install SIGTRAP trace handler");
    install_altstack();
    prosper::host::guest_write_watch_set_fault_onstack(true);

    const uint8_t mov_imm8_mem[] = {0xc6, 0x00, 0x7f};
    const uint8_t mov_imm32_mem[] = {0xc7, 0x00, 1, 0, 0, 0};
    const uint8_t mov_imm16_mem[] = {0x66, 0xc7, 0x00, 1, 0};
    const uint8_t mov_imm64_sign_extended_mem[] = {0x48, 0xc7, 0x00, 1, 0, 0, 0};
    const uint8_t mov_register_only[] = {0x48, 0x89, 0xc0};
    const uint8_t c7_other_extension[] = {0xc7, 0x08, 1, 0, 0, 0};
    const uint8_t c6_other_extension[] = {0xc6, 0x38, 0x7f};
    const uint8_t legacy_prefix_after_rex[] = {0x48, 0x66, 0x89, 0x00};
    const uint8_t lock_prefixed_mov[] = {0xf0, 0x48, 0x89, 0x00};
    CHECK(prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
              mov_imm8_mem) == 1 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm32_mem) == 4 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm16_mem) == 2 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm64_sign_extended_mem) == 8,
          "bounded store decoder accepts C6/C7 /0 memory forms with exact operand widths");
    CHECK(prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
              mov_register_only) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  c7_other_extension) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  c6_other_extension) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  legacy_prefix_after_rex) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  lock_prefixed_mov) == 0,
          "bounded store decoder rejects register ModRM, other C6/C7 extensions, and ambiguous prefixes");

    CHECK(setenv("PROSPER_DMEM_CALLER", "1", 1) == 0 &&
              setenv("PROSPER_DMEM_WRITE_TRACE", "7:0x3000:2:0x1120:16", 1) == 0 &&
              setenv("PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS", "3", 1) == 0,
          "set five-field occurrence-selector environment");
    prosper::host::guest_dmem_write_trace_init_from_environment();
    auto parsed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(parsed.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              parsed.config.allocation_occurrence == 2 && parsed.config.offset == offset &&
              parsed.config.size == bytes && parsed.config.max_events == 3,
          "five-field environment syntax retains explicit occurrence and bounds");
    CHECK(setenv("PROSPER_DMEM_WRITE_TRACE", "7:0x3000:0x1120:16", 1) == 0,
          "set legacy four-field unique-selector environment");
    prosper::host::guest_dmem_write_trace_init_from_environment();
    parsed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(parsed.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              parsed.config.allocation_occurrence == 0 && parsed.config.offset == offset,
          "four-field environment syntax preserves strict-uniqueness mode");
    unsetenv("PROSPER_DMEM_WRITE_TRACE");
    unsetenv("PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS");
    unsetenv("PROSPER_DMEM_CALLER");

    const prosper::host::GuestDmemWriteTraceConfig invalid_config{
        chain, allocation_size, offset,
        static_cast<uint32_t>(prosper::host::kGuestDmemWriteTraceMaxBytes + 1), 3};
    CHECK(!prosper::host::guest_dmem_write_trace_configure(invalid_config) &&
              prosper::host::guest_dmem_write_trace_snapshot().invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::InvalidConfig,
          "invalid selector is fail-visible before any mapping is touched");

    auto invalid_occurrence_config = invalid_config;
    invalid_occurrence_config.size = bytes;
    invalid_occurrence_config.allocation_occurrence =
        prosper::host::kGuestDmemWriteTraceMaxAllocationOccurrence + 1;
    CHECK(!prosper::host::guest_dmem_write_trace_configure(invalid_occurrence_config) &&
              prosper::host::guest_dmem_write_trace_snapshot().invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::InvalidConfig,
          "allocation occurrence above the explicit bound is fail-visible");

    const prosper::host::GuestDmemWriteTraceConfig config{
        chain, allocation_size, offset, bytes, 3};
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "dynamic dmem writer trace configuration is accepted");
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation,
          "configured trace waits for its allocation identity");

    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain + 1);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size - page, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().allocation_matches == 0,
          "wrong chain and wrong allocation size cannot move the selector lever");

    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::WaitingMapping,
          "exact caller-chain and allocation size resolve before mapping");

    auto* mapping = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(mapping != MAP_FAILED, "map canary direct-memory range");
    if (mapping == MAP_FAILED) return 1;
    std::memset(mapping + offset, 0x11, bytes);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(mapping), allocation_size, physical, 0x3);

    auto armed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(armed.status == prosper::host::GuestDmemWriteTraceStatus::Armed,
          "matching map creation arms the selected physical pages");
    CHECK(armed.mapping_matches == 1 && armed.target_addr ==
              reinterpret_cast<uint64_t>(mapping + offset),
          "dynamic selector resolves the mapping-relative virtual address");
    CHECK(armed.initial[0] == 0x11 && armed.initial[bytes - 1] == 0x11,
          "arm records the selected initial bytes");

    // Both the production dirty watch and the diagnostic own the same page protection. Releasing the
    // production owner must leave the page RO for the still-Armed trace. A kernel write gives a
    // deterministic protection probe without consuming the diagnostic's CPU-fault event.
    auto reset_overlap_watch = prosper::host::GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(mapping + offset), bytes);
    CHECK(static_cast<bool>(reset_overlap_watch),
          "production watch can share an already trace-armed page");
    reset_overlap_watch.reset();
    int reset_probe[2] = {-1, -1};
    CHECK(pipe(reset_probe) == 0, "production-reset protection probe pipe created");
    const uint8_t reset_payload = 0x5a;
    CHECK(write(reset_probe[1], &reset_payload, 1) == 1,
          "production-reset protection probe payload queued");
    errno = 0;
    const ssize_t reset_write = read(reset_probe[0], mapping + offset, 1);
    CHECK(reset_write < 0 && errno == EFAULT &&
              prosper::host::guest_dmem_write_trace_snapshot().status ==
                  prosper::host::GuestDmemWriteTraceStatus::Armed,
          "production reset preserves the diagnostic owner's real read-only protection");
    close(reset_probe[0]);
    close(reset_probe[1]);

    auto production_watch = prosper::host::GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(mapping + offset), bytes);
    CHECK(static_cast<bool>(production_watch),
          "production overlap watch arms before the diagnostic step canary");

    const int64_t creator_tid = static_cast<int64_t>(syscall(SYS_gettid));
    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(
        &unrelated_contention_hook);
    std::thread unrelated_locker([] {
        prosper::host::guest_dmem_write_trace_lock_state_for_test();
        unrelated_lock_held.store(true, std::memory_order_release);
        while (!release_unrelated_lock.load(std::memory_order_acquire)) sched_yield();
        prosper::host::guest_dmem_write_trace_unlock_state_for_test();
        unrelated_lock_held.store(false, std::memory_order_release);
    });
    while (!unrelated_lock_held.load(std::memory_order_acquire)) sched_yield();
    prosper::host::GuestDmemWriteTraceEvent unrelated_event{};
    const auto unrelated_step = prosper::host::guest_dmem_write_trace_complete_step(
        creator_tid, 0x1234, unrelated_event);
    const bool unrelated_lock_still_held = unrelated_lock_held.load(std::memory_order_acquire);
    release_unrelated_lock.store(true, std::memory_order_release);
    unrelated_locker.join();
    CHECK(unrelated_step == prosper::host::GuestDmemWriteTraceStepAction::NotHandled &&
              !unrelated_contention_hook_entered.load(std::memory_order_acquire) &&
              unrelated_lock_still_held,
          "unrelated TRAP bypasses contended dmem state through the lock-free pending-TID gate");

    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(&contention_hook);
    std::thread locker([&] {
        while (!request_contention.load(std::memory_order_acquire)) sched_yield();
        stepping_rearm_result.store(production_watch.rearm(), std::memory_order_release);
        int step_probe[2] = {-1, -1};
        if (pipe(step_probe) == 0) {
            const uint8_t step_payload = 0x6b;
            if (write(step_probe[1], &step_payload, 1) == 1) {
                const ssize_t wrote = read(step_probe[0], mapping + offset - 0x40, 1);
                stepping_kernel_write_succeeded.store(wrote == 1, std::memory_order_release);
            }
            close(step_probe[0]);
            close(step_probe[1]);
        }
        prosper::host::guest_dmem_write_trace_lock_state_for_test();
        contention_held.store(true, std::memory_order_release);
        while (!contention_hook_entered.load(std::memory_order_acquire)) sched_yield();
        prosper::host::guest_dmem_write_trace_unlock_state_for_test();
        contention_released.store(true, std::memory_order_release);
    });
    std::thread writer([&] {
        install_altstack();
        writer_tid.store(static_cast<int64_t>(syscall(SYS_gettid)), std::memory_order_release);
        store_byte_then_mark(mapping + offset, 0x42, &post_store_marker);
        exchange_byte(mapping + offset - 0x20, 0x77); // unknown RMW span, outside selected range
        store_byte(mapping + offset + 1, 0x43);      // selected again; proves re-arm
        store_byte(mapping + offset - 0x21, 0x78);   // fourth fault exceeds bounded history
    });
    writer.join();
    locker.join();
    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(nullptr);

    const auto result = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(writer_tid.load(std::memory_order_acquire) != 0 &&
              writer_tid.load(std::memory_order_acquire) != creator_tid,
          "canary writer runs on a thread other than the mapping creator");
    CHECK(result.status == prosper::host::GuestDmemWriteTraceStatus::Overflow &&
              result.overflow_events == 1,
          "the first event beyond the configured bound is explicit overflow");
    CHECK(result.event_count == 3 && result.page_faults >= 3 &&
              result.completed_steps == 3 && result.rearms == 3,
          "fault, single-step, bounded history, and re-arm mechanisms all ran");
    CHECK(contention_hook_entered.load(std::memory_order_acquire) &&
              marker_when_first_completed == 0 && trap_count.load(std::memory_order_relaxed) == 3,
          "contended first TRAP captures and re-arms before the next guest instruction executes");
    CHECK(!stepping_rearm_result.load(std::memory_order_acquire) &&
              stepping_kernel_write_succeeded.load(std::memory_order_acquire) &&
              production_watch.query() == prosper::host::GuestWriteWatchQuery::Dirty,
          "production rearm is refused during the trace RW step and cannot re-protect it early");
    production_watch.reset();

    const auto& first = result.events[0];
    CHECK(first.selected && first.decoded_write_size == 1 &&
              first.coverage_valid_before && first.rearmed && first.post_available &&
              first.changed_during_window,
          "first selected write has continuous process-wide pre-fault coverage");
    CHECK(first.tid == writer_tid.load(std::memory_order_acquire) &&
              first.writer_rip != 0 && first.next_rip != 0,
          "cross-thread canary retains writer thread and instruction identity");
    CHECK(first.before[0] == 0x11 && first.after[0] == 0x42,
          "cross-thread canary retains exact before/after selected bytes");

    const auto& adjacent = result.events[1];
    CHECK(!adjacent.selected && adjacent.selection_uncertain &&
              adjacent.decoded_write_size == 0 &&
              adjacent.before[0] == 0x42 && adjacent.after[0] == 0x42 &&
              result.selection_uncertain_faults == 1,
          "unchanged off-range fault is explicit unknown because a same-value crossing store is possible");

    const auto& repeated = result.events[2];
    CHECK(repeated.selected && repeated.decoded_write_size == 1 && repeated.rearmed &&
              repeated.before[0] == 0x42 &&
              repeated.before[1] == 0x11 && repeated.after[1] == 0x43,
          "a second selected store is captured after the real re-arm path");
    CHECK(!repeated.coverage_valid_before && result.coverage_gaps >= 3,
          "single-step RW windows invalidate later negative coverage instead of claiming completeness");

    // Hold a real writer in the SIGSEGV handler after SingleStep publication. A sibling invalidation
    // must retain that TID's pending ownership until its store executes and TRAP_TRACE is consumed;
    // CompleteInvalid tells the signal caller to clear TF without re-arming stale topology.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for pending-step invalidation control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t invalidation_marker = 0;
    pause_invalidation_step.store(true, std::memory_order_release);
    invalidation_step_ready.store(false, std::memory_order_release);
    release_invalidation_step.store(false, std::memory_order_release);
    invalidation_canary_active.store(true, std::memory_order_release);
    last_step_action.store(-1, std::memory_order_release);
    std::thread invalidated_writer([&] {
        install_altstack();
        store_byte_then_mark(mapping + offset + 2, 0x66, &invalidation_marker);
    });
    while (!invalidation_step_ready.load(std::memory_order_acquire)) sched_yield();
    prosper::host::guest_write_watch_notify_physical_write(physical + offset + 2, 1);
    const auto invalidating = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(invalidating.status == prosper::host::GuestDmemWriteTraceStatus::Stepping &&
              invalidating.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::PhysicalWrite,
          "sibling invalidation retains the published stepping owner and explicit reason");
    release_invalidation_step.store(true, std::memory_order_release);
    invalidated_writer.join();
    invalidation_canary_active.store(false, std::memory_order_release);
    const auto invalidated = prosper::host::guest_dmem_write_trace_snapshot();
    char invalidated_post[32] = {};
    prosper::host::guest_dmem_write_trace_format_post(
        invalidated.events[0], invalidated_post, sizeof(invalidated_post));
    CHECK(last_step_action.load(std::memory_order_acquire) == static_cast<int>(
              prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) &&
              invalidation_marker == 1 &&
              invalidated.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              invalidated.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::PhysicalWrite &&
              invalidated.completed_steps == 1 && invalidated.event_count == 1 &&
              !invalidated.events[0].rearmed && !invalidated.events[0].post_available &&
              std::strcmp(invalidated_post, "unavailable") == 0,
          "invalidated pending TF trap is consumed without fabricating a zero post-image");

    // Configuration is a normal-context seam, but cannot replace the trace while a faulting thread
    // owns TF. Hold a real writer after SingleStep publication: the configure attempt must be refused
    // without altering the pending TID, and the original event must complete normally.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for pending-step reconfiguration control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t reconfigure_marker = 0;
    pause_reconfigure_step.store(true, std::memory_order_release);
    reconfigure_step_ready.store(false, std::memory_order_release);
    release_reconfigure_step.store(false, std::memory_order_release);
    reconfigure_canary_active.store(true, std::memory_order_release);
    last_step_action.store(-1, std::memory_order_release);
    std::thread reconfigured_writer([&] {
        install_altstack();
        store_byte_then_mark(mapping + offset + 3, 0x67, &reconfigure_marker);
    });
    while (!reconfigure_step_ready.load(std::memory_order_acquire)) sched_yield();
    const auto before_reconfigure = prosper::host::guest_dmem_write_trace_snapshot();
    const bool reconfigured = prosper::host::guest_dmem_write_trace_configure(config);
    const auto refused_reconfigure = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(!reconfigured &&
              before_reconfigure.status == prosper::host::GuestDmemWriteTraceStatus::Stepping &&
              refused_reconfigure.status == prosper::host::GuestDmemWriteTraceStatus::Stepping &&
              refused_reconfigure.event_count == 0,
          "configure refuses a live TF owner without replacing its trace state");
    release_reconfigure_step.store(true, std::memory_order_release);
    reconfigured_writer.join();
    reconfigure_canary_active.store(false, std::memory_order_release);
    const auto after_reconfigure = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(last_step_action.load(std::memory_order_acquire) == static_cast<int>(
              prosper::host::GuestDmemWriteTraceStepAction::Complete) &&
              reconfigure_marker == 1 &&
              after_reconfigure.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
              after_reconfigure.completed_steps == 1 && after_reconfigure.event_count == 1 &&
              after_reconfigure.events[0].post_available,
          "refused reconfiguration leaves the exact pending owner able to complete and re-arm");

    // An unknown off-range instruction executes while all aliases are RW. Change a selected byte from
    // a sibling during that window: the delta is retained, but cannot be attributed to the faulting RIP.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for unknown-window sibling-write control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    pause_unknown_step.store(true, std::memory_order_release);
    unknown_step_ready.store(false, std::memory_order_release);
    release_unknown_step.store(false, std::memory_order_release);
    std::thread unknown_writer([&] {
        install_altstack();
        exchange_byte(mapping + offset - 0x22, 0x79);
    });
    while (!unknown_step_ready.load(std::memory_order_acquire)) sched_yield();
    store_byte(mapping + offset + 4, 0xa4);
    release_unknown_step.store(true, std::memory_order_release);
    unknown_writer.join();
    const auto unknown_window = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(unknown_window.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
              unknown_window.event_count == 1 && unknown_window.selected_faults == 0 &&
              unknown_window.selection_uncertain_faults == 1 &&
              !unknown_window.events[0].selected &&
              unknown_window.events[0].selection_uncertain &&
              unknown_window.events[0].changed_during_window &&
              unknown_window.events[0].post_available &&
              std::memcmp(unknown_window.events[0].before.data(),
                          unknown_window.events[0].after.data(), bytes) != 0,
          "unknown off-range RIP stays unattributed when a sibling changes selected bytes in the RW window");

    // A real int3-owned step reaches the selected store with TF already set, matching the ownership
    // shape of BP/HWBP/STEPWIN. Dmem must invalidate/open the page and let that owner consume the trap.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for pre-owned TF control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t preowned_marker = 0;
    preowned_tf_started.store(false, std::memory_order_release);
    preowned_tf_completed.store(0, std::memory_order_release);
    preowned_tf_active.store(true, std::memory_order_release);
    dynamic_protection_helper_entries.store(0, std::memory_order_release);
    prosper::host::guest_dmem_write_trace_set_dynamic_protection_hook_for_test(
        &dynamic_protection_hook);
    breakpoint_then_store_byte(mapping + offset + 6, 0xb6, &preowned_marker);
    prosper::host::guest_dmem_write_trace_set_dynamic_protection_hook_for_test(nullptr);
    preowned_tf_active.store(false, std::memory_order_release);
    const auto preowned = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(preowned_tf_started.load(std::memory_order_acquire) &&
              preowned_tf_completed.load(std::memory_order_acquire) == 1 &&
              preowned_marker == 1 && mapping[offset + 6] == 0xb6 &&
              preowned.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              preowned.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::TrapFlagAlreadyOwned &&
              preowned.event_count == 0 && preowned.selected_faults == 0 &&
              preowned.page_faults == 1 && preowned.coverage_gaps == 1,
          "pre-existing TF owner receives its completion and is never stolen or stranded");
    CHECK(dynamic_protection_helper_entries.load(std::memory_order_acquire) == 0,
          "pre-owned TF signal invalidation bypasses the dynamic protection helper");

    // Strict-unique mode may observe a selected event from occurrence 1 before occurrence 2 arrives.
    // Retain that history, but the terminal result must become undetermined for the ambiguous family.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for post-event strict-uniqueness control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    store_byte(mapping + offset + 7, 0xc7);
    const auto unique_before_ambiguity = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(unique_before_ambiguity.result ==
              prosper::host::GuestDmemWriteTraceResult::WriterObserved &&
              unique_before_ambiguity.selected_faults == 1,
          "strict-unique control first proves a selected event on occurrence 1");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical + allocation_size, allocation_size, chain);
    const auto ambiguous_after_event = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(ambiguous_after_event.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              ambiguous_after_event.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::AmbiguousAllocation &&
              ambiguous_after_event.result ==
                  prosper::host::GuestDmemWriteTraceResult::Undetermined &&
              ambiguous_after_event.allocation_matches == 2 &&
              ambiguous_after_event.selected_faults == 1 &&
              ambiguous_after_event.event_count == 1,
          "occurrence 2 invalidates the writer verdict while retaining occurrence-1 history");

    // CR2 identifies the first faulting byte, not the qword's full span. Start four bytes before the
    // selected interval and change four selected bytes with one instruction; post-step evidence must
    // upgrade this crossing store from unknown to selected.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for crossing-store control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t crossing_marker = 0;
    store_qword_then_mark(mapping + offset - 4, 0x24232221ddccbbaaull, &crossing_marker);
    const auto crossing = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(crossing_marker == 1 && crossing.status ==
              prosper::host::GuestDmemWriteTraceStatus::Armed &&
              crossing.event_count == 1 && crossing.selected_faults == 1 &&
              crossing.selection_uncertain_faults == 0 && crossing.events[0].selected &&
              crossing.events[0].decoded_write_size == 8 &&
              !crossing.events[0].selection_uncertain &&
              crossing.events[0].fault_addr == reinterpret_cast<uint64_t>(mapping + offset - 4) &&
              std::memcmp(crossing.events[0].before.data(), crossing.events[0].after.data(), bytes) != 0,
          "qword beginning before the range is selected when its completed store changes selected bytes");

    // A decoded width does not turn CR2 into an operand start. Begin a qword four bytes before a page
    // boundary while selecting P+4 onward. The fault reports P (the first inaccessible byte), and
    // inventing [P,P+8) would falsely overlap the selection even though the real store ends at P+3.
    auto split_page_config = config;
    split_page_config.offset = page + 4;
    CHECK(prosper::host::guest_dmem_write_trace_configure(split_page_config),
          "selector can be reset for split-page CR2 control");
    std::memset(mapping + split_page_config.offset, 0x31, bytes);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t split_page_marker = 0;
    store_qword_then_mark(mapping + page - 4, 0x8877665544332211ull,
                          &split_page_marker);
    const auto split_page = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(split_page_marker == 1 &&
              split_page.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
              split_page.result == prosper::host::GuestDmemWriteTraceResult::Undetermined &&
              split_page.event_count == 1 && split_page.selected_faults == 0 &&
              split_page.selection_uncertain_faults == 1 &&
              !split_page.events[0].selected &&
              split_page.events[0].selection_uncertain &&
              split_page.events[0].decoded_write_size == 8 &&
              split_page.events[0].fault_addr ==
                  reinterpret_cast<uint64_t>(mapping + page) &&
              std::memcmp(split_page.events[0].before.data(),
                          split_page.events[0].after.data(), bytes) == 0,
          "split-page store cannot attribute CR2-plus-width as the selected writer span");

    // The retained bool handler cannot set TF or route TRAP_TRACE. Exercise a real write through that
    // interface while a production watch overlaps the diagnostic: it must preserve Dirty/Resume, but
    // invalidate and open the trace without publishing sentinel TID 0 as a step owner. complete_step(0)
    // is called only as mutant cleanup; it must be NotHandled in the real implementation.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for legacy bool fault-interface control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    auto legacy_production_watch = prosper::host::GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(mapping + offset), bytes);
    CHECK(static_cast<bool>(legacy_production_watch),
          "production watch overlaps the legacy bool trace control");
    volatile sig_atomic_t legacy_bool_marker = 0;
    legacy_bool_fault_active.store(true, std::memory_order_release);
    dynamic_protection_helper_entries.store(0, std::memory_order_release);
    prosper::host::guest_dmem_write_trace_set_dynamic_protection_hook_for_test(
        &dynamic_protection_hook);
    store_byte_then_mark(mapping + offset + 8, 0xd8, &legacy_bool_marker);
    prosper::host::guest_dmem_write_trace_set_dynamic_protection_hook_for_test(nullptr);
    legacy_bool_fault_active.store(false, std::memory_order_release);
    const auto legacy_bool = prosper::host::guest_dmem_write_trace_snapshot();
    prosper::host::GuestDmemWriteTraceEvent legacy_mutant_cleanup_event{};
    const auto legacy_mutant_cleanup = prosper::host::guest_dmem_write_trace_complete_step(
        0, 0, legacy_mutant_cleanup_event);
    CHECK(legacy_bool_marker == 1 && mapping[offset + 8] == 0xd8 &&
              legacy_production_watch.query() ==
                  prosper::host::GuestWriteWatchQuery::Dirty &&
              legacy_bool.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              legacy_bool.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::SingleStepUnavailable &&
              legacy_bool.page_faults == 1 && legacy_bool.coverage_gaps == 1 &&
              legacy_bool.event_count == 0 && legacy_bool.selected_faults == 0 &&
              legacy_mutant_cleanup ==
                  prosper::host::GuestDmemWriteTraceStepAction::NotHandled,
          "legacy bool fault path completes the store without stranding a zero-TID trace step");
    CHECK(dynamic_protection_helper_entries.load(std::memory_order_acquire) == 0,
          "legacy bool signal invalidation bypasses the dynamic protection helper");
    legacy_production_watch.reset();

    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for mapping-topology control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::Armed,
          "existing complete mapping re-arms the reset selector");
    auto* second_alias = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(second_alias != MAP_FAILED, "map second-alias topology control");
    if (second_alias != MAP_FAILED) {
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(second_alias), allocation_size, physical, 0x3);
        const auto topology = prosper::host::guest_dmem_write_trace_snapshot();
        CHECK(topology.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                  topology.invalid_reason ==
                      prosper::host::GuestDmemWriteTraceInvalidReason::MappingTopologyChanged,
              "a writable alias created after arm invalidates the trace instead of hiding its gap");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(second_alias), allocation_size);
        munmap(second_alias, allocation_size);
    }

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(mapping), allocation_size);
    munmap(mapping, allocation_size);

    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for ambiguity control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical + allocation_size, allocation_size, chain);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical + allocation_size * 2, allocation_size, chain);
    const auto ambiguous = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(ambiguous.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              ambiguous.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::AmbiguousAllocation &&
              ambiguous.allocation_matches == 3 && ambiguous.selected_occurrence == 1,
          "default selector invalidates on occurrence 2 but keeps an honest observed census");

    auto occurrence_config = config;
    occurrence_config.allocation_occurrence = 2;
    CHECK(prosper::host::guest_dmem_write_trace_configure(occurrence_config),
          "explicit bounded occurrence selector is accepted");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    const auto insufficient = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(insufficient.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              insufficient.config.allocation_occurrence == 2 &&
              insufficient.allocation_matches == 1 && insufficient.selected_occurrence == 0 &&
              insufficient.target_phys == 0,
          "insufficient occurrence stays waiting and proves requested/observed/selected state");

    auto* occurrence_two_mapping = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(occurrence_two_mapping != MAP_FAILED, "map occurrence-2 selection control");
    if (occurrence_two_mapping != MAP_FAILED) {
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(occurrence_two_mapping), allocation_size,
            physical + allocation_size, 0x3);
        prosper::host::guest_dmem_write_trace_notify_allocation(
            physical + allocation_size, allocation_size, chain);
        prosper::host::guest_dmem_write_trace_notify_allocation(
            physical + allocation_size * 2, allocation_size, chain);
        const auto selected_two = prosper::host::guest_dmem_write_trace_snapshot();
        CHECK(selected_two.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
                  selected_two.config.allocation_occurrence == 2 &&
                  selected_two.allocation_matches == 3 && selected_two.selected_occurrence == 2 &&
                  selected_two.target_phys == physical + allocation_size + offset &&
                  selected_two.target_addr ==
                      reinterpret_cast<uint64_t>(occurrence_two_mapping + offset),
              "occurrence 2 moves the lever to the second allocation while observing later matches");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(occurrence_two_mapping), allocation_size);
        munmap(occurrence_two_mapping, allocation_size);
    }
    // #3387: a successful guest mprotect precedes the notification on POSIX. Neither the diagnostic
    // nor a co-owning production watch may restore the changed alias's OLD writable protection.
    // Real shared backing, not merely equal registry IDs, also makes the stepping store meaningful.
    for (const bool stepping : {false, true}) {
        for (const bool with_production : {false, true}) {
            for (const int downgraded_prot : {PROT_READ, PROT_NONE}) {
                std::fprintf(stderr, "[trace-protection] stepping=%d production=%d protection=%d\n",
                             stepping, with_production, downgraded_prot);
                const int backing = static_cast<int>(syscall(SYS_memfd_create,
                                                             "trace-protection", 0));
                CHECK(backing >= 0, "create shared trace-protection backing");
                if (backing < 0) continue;
                const int sized = ftruncate(backing, allocation_size);
                CHECK(sized == 0, "size shared trace-protection backing");
                if (sized != 0) { close(backing); continue; }
                auto* writer_alias = static_cast<uint8_t*>(mmap(
                    nullptr, allocation_size, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0));
                auto* changed_alias = static_cast<uint8_t*>(mmap(
                    nullptr, allocation_size, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0));
                close(backing);
                CHECK(writer_alias != MAP_FAILED && changed_alias != MAP_FAILED,
                      "map two actual shared aliases for protection control");
                if (writer_alias == MAP_FAILED || changed_alias == MAP_FAILED) {
                    if (writer_alias != MAP_FAILED) munmap(writer_alias, allocation_size);
                    if (changed_alias != MAP_FAILED) munmap(changed_alias, allocation_size);
                    continue;
                }
                std::memset(writer_alias, 0x31, allocation_size);
                CHECK(changed_alias[offset] == 0x31, "protection control aliases share bytes");
                CHECK(prosper::host::guest_dmem_write_trace_configure(config),
                      "reset selector for alias-protection control");
                // Record both aliases before selecting the allocation, avoiding a late-map gap.
                constexpr uint64_t protection_physical = 0x980000;
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(writer_alias), allocation_size,
                    protection_physical, 0x3);
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(changed_alias), allocation_size,
                    protection_physical, 0x3);
                prosper::host::guest_dmem_write_trace_notify_allocation(
                    protection_physical, allocation_size, chain);
                CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
                          prosper::host::GuestDmemWriteTraceStatus::Armed,
                      "complete alias set arms protection-control trace");
                prosper::host::GuestWriteWatch shared_owner;
                if (with_production) {
                    shared_owner = prosper::host::GuestWriteWatch::create(
                        reinterpret_cast<uint64_t>(writer_alias + offset), bytes);
                    CHECK(static_cast<bool>(shared_owner),
                          "production owner shares the diagnostic physical page");
                }
                std::thread pending_writer;
                std::atomic<int64_t> pending_tid{0};
                volatile sig_atomic_t protection_marker = 0;
                if (stepping) {
                    pause_invalidation_step.store(true, std::memory_order_release);
                    invalidation_step_ready.store(false, std::memory_order_release);
                    release_invalidation_step.store(false, std::memory_order_release);
                    invalidation_canary_active.store(true, std::memory_order_release);
                    last_step_action.store(-1, std::memory_order_release);
                    pending_writer = std::thread([&] {
                        install_altstack();
                        pending_tid.store(static_cast<int64_t>(syscall(SYS_gettid)),
                                          std::memory_order_release);
                        store_byte_then_mark(writer_alias + offset, 0x67, &protection_marker);
                    });
                    while (!invalidation_step_ready.load(std::memory_order_acquire)) sched_yield();
                }
                // Downgrade only the NON-faulting sibling. The pending writer's original store
                // remains legal; revoking its own VA would test a different guest protection fault.
                const int protected_result = mprotect(changed_alias + page, page, downgraded_prot);
                CHECK(protected_result == 0, "apply actual sibling alias permission downgrade");
                if (protected_result == 0)
                    prosper::host::guest_write_watch_notify_direct_mapping_protection(
                        reinterpret_cast<uint64_t>(changed_alias + page), page,
                        downgraded_prot == PROT_READ ? 0x1 : 0x0);
                const auto pending = prosper::host::guest_dmem_write_trace_snapshot();
                CHECK(pending.status == (stepping ? prosper::host::GuestDmemWriteTraceStatus::Stepping
                                                 : prosper::host::GuestDmemWriteTraceStatus::Invalid) &&
                          pending.invalid_reason ==
                              prosper::host::GuestDmemWriteTraceInvalidReason::MappingProtectionChanged &&
                          pending.event_count == 0,
                      "protection change invalidates trace without losing a pending TF owner");
                check_kernel_permissions(changed_alias + offset, downgraded_prot == PROT_READ, false);
                if (stepping) {
                    // The retained trace alias must not turn a new, genuine permission fault into
                    // a stale queued-fault Resume while another thread owns the original TF window.
                    const auto revoked_fault = prosper::host::guest_write_watch_handle_fault_ex(
                        reinterpret_cast<uint64_t>(changed_alias + offset), 0,
                        static_cast<int64_t>(syscall(SYS_gettid)), false);
                    const auto after_revoked_fault = prosper::host::guest_dmem_write_trace_snapshot();
                    CHECK(revoked_fault == prosper::host::GuestWriteWatchFaultAction::NotHandled &&
                              after_revoked_fault.status ==
                                  prosper::host::GuestDmemWriteTraceStatus::Stepping &&
                              after_revoked_fault.invalid_reason ==
                                  prosper::host::GuestDmemWriteTraceInvalidReason::MappingProtectionChanged &&
                              after_revoked_fault.event_count == 0,
                          "revoked sibling fault is rejected without consuming the pending TF owner");
                    check_kernel_permissions(changed_alias + offset,
                                             downgraded_prot == PROT_READ, false);
                    release_invalidation_step.store(true, std::memory_order_release);
                    pending_writer.join();
                    invalidation_canary_active.store(false, std::memory_order_release);
                    const auto completed = prosper::host::guest_dmem_write_trace_snapshot();
                    CHECK(last_step_action.load(std::memory_order_acquire) == static_cast<int>(
                              prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) &&
                              protection_marker == 1 && writer_alias[offset] == 0x67 &&
                              completed.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                              completed.invalid_reason ==
                                  prosper::host::GuestDmemWriteTraceInvalidReason::MappingProtectionChanged &&
                              completed.completed_steps == 1 && completed.event_count == 1 &&
                              completed.events[0].tid == pending_tid.load(std::memory_order_acquire) &&
                              !completed.events[0].rearmed && !completed.events[0].post_available &&
                              completed.rearms == 0,
                          "downgraded sibling preserves exact TF owner and completes without rearm");
                    check_kernel_permissions(changed_alias + offset,
                                             downgraded_prot == PROT_READ, false);
                }
                shared_owner.reset();
                check_kernel_permissions(changed_alias + offset, downgraded_prot == PROT_READ, false);
                check_kernel_permissions(writer_alias + offset, true, true);
                check_kernel_permissions(changed_alias, true, true);
                check_kernel_permissions(changed_alias + 2 * page, true, true);
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(writer_alias), allocation_size);
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(changed_alias), allocation_size);
                munmap(writer_alias, allocation_size);
                munmap(changed_alias, allocation_size);
            }
        }
    }

    // An initially RO alias never enters the trace's writable-only page vectors. Its upgrade still
    // breaks physical-page coverage, so invalidation must consult live mapping topology as well.
    for (const bool with_production : {false, true}) {
        std::fprintf(stderr, "[trace-protection-upgrade] production=%d\n", with_production);
        const int backing = static_cast<int>(syscall(SYS_memfd_create, "trace-upgrade", 0));
        CHECK(backing >= 0, "create shared trace-upgrade backing");
        if (backing < 0) continue;
        const int sized = ftruncate(backing, allocation_size);
        CHECK(sized == 0, "size shared trace-upgrade backing");
        if (sized != 0) { close(backing); continue; }
        auto* writable_alias = static_cast<uint8_t*>(mmap(
            nullptr, allocation_size, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0));
        auto* readonly_alias = static_cast<uint8_t*>(mmap(
            nullptr, allocation_size, PROT_READ, MAP_SHARED, backing, 0));
        close(backing);
        CHECK(writable_alias != MAP_FAILED && readonly_alias != MAP_FAILED,
              "map pre-existing RW and RO aliases for upgrade control");
        if (writable_alias == MAP_FAILED || readonly_alias == MAP_FAILED) {
            if (writable_alias != MAP_FAILED) munmap(writable_alias, allocation_size);
            if (readonly_alias != MAP_FAILED) munmap(readonly_alias, allocation_size);
            continue;
        }
        std::memset(writable_alias, 0x31, allocation_size);
        CHECK(readonly_alias[offset] == 0x31, "upgrade aliases share actual backing bytes");
        CHECK(prosper::host::guest_dmem_write_trace_configure(config),
              "reset selector for pre-existing RO alias upgrade");
        constexpr uint64_t upgrade_physical = 0x9c0000;
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(writable_alias), allocation_size, upgrade_physical, 0x3);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(readonly_alias), allocation_size, upgrade_physical, 0x1);
        prosper::host::guest_dmem_write_trace_notify_allocation(
            upgrade_physical, allocation_size, chain);
        CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
                  prosper::host::GuestDmemWriteTraceStatus::Armed,
              "trace arms with a pre-existing non-writable sibling");
        prosper::host::GuestWriteWatch shared_owner;
        if (with_production) {
            shared_owner = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(writable_alias + offset), bytes);
            CHECK(static_cast<bool>(shared_owner), "production owner shares upgrade-control page");
        }
        check_kernel_permissions(readonly_alias + offset, true, false);
        const int upgraded = mprotect(readonly_alias + page, page, PROT_READ | PROT_WRITE);
        CHECK(upgraded == 0, "apply actual pre-existing sibling RO-to-RW upgrade");
        if (upgraded == 0) {
            prosper::host::guest_write_watch_notify_direct_mapping_protection(
                reinterpret_cast<uint64_t>(readonly_alias + page), page, 0x3);
            const auto invalidated = prosper::host::guest_dmem_write_trace_snapshot();
            CHECK(invalidated.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                      invalidated.invalid_reason ==
                          prosper::host::GuestDmemWriteTraceInvalidReason::MappingProtectionChanged,
                  "physical coverage invalidates even when changed RO alias was absent from trace pages");
            store_byte(readonly_alias + offset, 0x6a);
            const auto after_store = prosper::host::guest_dmem_write_trace_snapshot();
            CHECK(writable_alias[offset] == 0x6a &&
                      after_store.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                      after_store.invalid_reason ==
                          prosper::host::GuestDmemWriteTraceInvalidReason::MappingProtectionChanged &&
                      after_store.event_count == 0 && after_store.completed_steps == 0 &&
                      after_store.result != prosper::host::GuestDmemWriteTraceResult::NoSelectedWriteObserved,
                  "upgraded alias accepts a real store without claiming negative diagnostic coverage");
        }
        shared_owner.reset();
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(writable_alias), allocation_size);
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(readonly_alias), allocation_size);
        munmap(writable_alias, allocation_size);
        munmap(readonly_alias, allocation_size);
    }

    // ---- host-write rebaseline (#3146) --------------------------------------------------------
    //
    // Three review rounds found three defects in this feature and no test could reach any of them,
    // because the mode was a cached getenv read on first use. The arms below are the two that would
    // have caught them: a rebaseline that never happens (the global-depth regression, which silently
    // turned the feature off and was found only by re-running an emulator measurement), and one that
    // happens while a host write is still in flight (the EFAULT the guard exists to prevent).
    {
        constexpr uint64_t rebase_physical = 0x860000;
        constexpr uint32_t rebase_chain = 11;
        prosper::host::guest_dmem_write_trace_set_rebase_enabled_for_test(1);
        CHECK(setenv("PROSPER_DMEM_CALLER", "1", 1) == 0 &&
                  setenv("PROSPER_DMEM_WRITE_TRACE", "11:0x3000:0x1120:16", 1) == 0,
              "configure the rebaseline trace selector");
        prosper::host::guest_dmem_write_trace_init_from_environment();
        prosper::host::guest_dmem_write_trace_notify_allocation(
            rebase_physical, allocation_size, rebase_chain);
        auto* rebase_mapping = static_cast<uint8_t*>(
            mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        CHECK(rebase_mapping != MAP_FAILED, "map the rebaseline canary range");
        if (rebase_mapping != MAP_FAILED) {
            std::memset(rebase_mapping + offset, 0x22, bytes);
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(rebase_mapping), allocation_size, rebase_physical, 0x3);
            const uint64_t traced = reinterpret_cast<uint64_t>(rebase_mapping) + offset;
            // A page of the same mapping that the trace does NOT cover.
            const uint64_t elsewhere = reinterpret_cast<uint64_t>(rebase_mapping);
            CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
                      prosper::host::GuestDmemWriteTraceStatus::Armed,
                  "rebaseline trace arms");

            // A host write over the traced page disarms it and leaves it pending.
            prosper::host::guest_write_watch_notify_host_write(traced, bytes);
            auto after_write = prosper::host::guest_dmem_write_trace_snapshot();
            CHECK(after_write.status == prosper::host::GuestDmemWriteTraceStatus::Invalid,
                  "a host write over the traced page disarms the trace");

            // While that write is STILL IN FLIGHT, an unrelated host write must not re-arm it --
            // re-arming now would make the pending store hit a read-only page and EFAULT the read.
            prosper::host::guest_write_watch_notify_host_write(elsewhere, 16);
            auto during = prosper::host::guest_dmem_write_trace_snapshot();
            CHECK(during.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                      during.host_write_rebases == 0 && during.rebase_in_flight_refusals > 0,
                  "no rebaseline while a host write over the traced page is in flight");
            prosper::host::guest_write_watch_notify_host_write_done(elsewhere, 16);

            // Once it completes, the next unrelated host write DOES rebaseline. This is the arm the
            // global-depth version failed: it refused forever and the feature went silently dead.
            prosper::host::guest_write_watch_notify_host_write_done(traced, bytes);
            prosper::host::guest_write_watch_notify_host_write(elsewhere, 16);
            auto after = prosper::host::guest_dmem_write_trace_snapshot();
            CHECK(after.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
                      after.host_write_rebases == 1,
                  "the completed host write lets the next unrelated one rebaseline exactly once");
            prosper::host::guest_write_watch_notify_host_write_done(elsewhere, 16);

            // B8: the caller-range half of the guard, which the arms above cannot reach.
            //
            // The scenario needs the second alias to appear WHILE THE TRACE IS PENDING: the trace's
            // own mapping hook ignores an Invalid trace, so V2 lands in the watch's alias list and
            // not in trace.pages. The caller-side overlap test then runs against the stale page list
            // and misses it, the rebuild pulls it in, and only the caller-range clause is left to
            // notice that the write about to run covers a page the rebase is about to arm read-only.
            // Without that clause this is the EFAULT the whole guard exists to prevent.
            prosper::host::guest_write_watch_notify_host_write(traced, bytes);
            prosper::host::guest_write_watch_notify_host_write_done(traced, bytes);
            auto* second_alias = static_cast<uint8_t*>(
                mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            CHECK(second_alias != MAP_FAILED, "map a second alias of the traced physical range");
            if (second_alias != MAP_FAILED) {
                const uint64_t rebases_before =
                    prosper::host::guest_dmem_write_trace_snapshot().host_write_rebases;
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(second_alias), allocation_size, rebase_physical, 0x3);
                prosper::host::guest_write_watch_notify_host_write(
                    reinterpret_cast<uint64_t>(second_alias) + offset, bytes);
                auto aliased = prosper::host::guest_dmem_write_trace_snapshot();
                CHECK(aliased.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                          aliased.host_write_rebases == rebases_before,
                      "a write into an alias added during the pending window cannot rebaseline");
                prosper::host::guest_write_watch_notify_host_write_done(
                    reinterpret_cast<uint64_t>(second_alias) + offset, bytes);
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(second_alias), allocation_size);
                munmap(second_alias, allocation_size);
            }

            prosper::host::guest_write_watch_notify_direct_mapping_removed(
                reinterpret_cast<uint64_t>(rebase_mapping), allocation_size);
            munmap(rebase_mapping, allocation_size);
        }
        prosper::host::guest_dmem_write_trace_set_rebase_enabled_for_test(-1);
        // Paired like the earlier blocks: this one is last in main today, so leaking would be
        // harmless right up until somebody appends an arm and inherits a configured selector.
        unsetenv("PROSPER_DMEM_WRITE_TRACE");
        unsetenv("PROSPER_DMEM_CALLER");
    }

    prosper::host::guest_write_watch_set_fault_onstack(false);

    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}
