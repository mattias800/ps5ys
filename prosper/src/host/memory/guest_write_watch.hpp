#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::host {

enum class GuestWriteWatchQuery {
    Unchanged,
    Dirty,
    Unknown,
};

struct GuestWriteWatchStats {
    uint64_t create_attempts = 0;
    uint64_t registrations = 0;
    uint64_t registered_pages = 0;
    uint64_t create_no_mapping = 0;
    uint64_t create_incomplete_aliases = 0;
    uint64_t create_oversized = 0;
    uint64_t create_bytes_le_1m = 0;
    uint64_t create_bytes_le_8m = 0;
    uint64_t create_bytes_le_32m = 0;
    uint64_t create_bytes_gt_32m = 0;
    uint64_t create_protect_failures = 0;
    uint64_t queries = 0;
    uint64_t unchanged = 0;
    uint64_t dirty = 0;
    uint64_t unknown = 0;
    uint64_t faults = 0;
    uint64_t stale_faults = 0;
    uint64_t physical_writes = 0;
    uint64_t rearms = 0;
    // #3307: whether a host-write notification actually changes page protections, or merely walks
    // the page index and finds nothing armed. The compute writeback issues one of these per storage
    // image and its `watch_ms` stage was attributed to "write-watch registration" without either
    // case being distinguished -- and they differ by an order of magnitude. Measured on a Strix Halo
    // for a 64 MiB range: walking the page index costs 0.05 ms even when every page is watched,
    // while one mprotect over it costs 0.39 ms idle and 0.63 ms under load. So `pages_hit` (and the
    // mprotect volume below) is what says which of the two `watch_ms` is.
    uint64_t host_write_notifies = 0;      // reached the state lock (the feature is armed)
    uint64_t host_write_no_alias = 0;      // ...and returned early: not a dmem range at all
    uint64_t host_write_pages_hit = 0;     // watched page aliases found inside the notified range
    // The remaining candidate once `pages_hit` and `protect_calls` are known. This function takes
    // the one global state mutex, and `GuestWriteWatch::query()` holds that same mutex while walking
    // every page of a registration -- 193,248 watched pages across 86 registrations were measured on
    // Stray, queried 89,938 times. A compute writeback whose notification simply WAITS would report
    // exactly the same `watch_ms` as one doing work, and no other counter here can tell them apart.
    uint64_t host_write_lock_contended = 0;
    // Every mprotect this file issues, in both directions. Names the machinery's real syscall
    // volume rather than leaving it inferred from arm/disarm counts of unknown extent.
    uint64_t protect_calls = 0;
    uint64_t protect_bytes = 0;
};

// Bounded, diagnostic-only provenance overlay for direct-memory CPU writes.  Unlike GuestWriteWatch,
// which is a production dirty bit, this records the faulting instruction and the selected bytes on
// both sides of one store.  Selection is run-relative: caller-chain + exact allocation size + offset.
// Fixed guest virtual addresses are deliberately not accepted.
inline constexpr size_t kGuestDmemWriteTraceMaxBytes = 128;
inline constexpr size_t kGuestDmemWriteTraceMaxEvents = 64;
inline constexpr uint32_t kGuestDmemWriteTraceMaxAllocationOccurrence = 4096;

struct GuestDmemWriteTraceConfig {
    uint32_t caller_chain = 0;
    uint64_t allocation_size = 0;
    uint64_t offset = 0;
    uint32_t size = 0;
    uint32_t max_events = 0;
    // Zero preserves the four-field selector's strict-uniqueness contract. A nonzero value selects
    // that one-based occurrence within the exact caller-chain/allocation-size family.
    uint32_t allocation_occurrence = 0;
};

enum class GuestDmemWriteTraceStatus : uint8_t {
    Disabled,
    WaitingAllocation,
    WaitingMapping,
    Armed,
    Stepping,
    Invalid,
    Overflow,
};

enum class GuestDmemWriteTraceInvalidReason : uint8_t {
    None,
    InvalidConfig,
    CallerDiagnosticDisabled,
    UnsupportedPlatform,
    NoAlternateSignalStack,
    AmbiguousAllocation,
    AddressOverflow,
    MappingTopologyChanged,
    MappingProtectionChanged,
    HostWrite,
    PhysicalWrite,
    ProtectFailed,
    RearmFailed,
    SingleStepUnavailable,
    TrapFlagAlreadyOwned,
};

enum class GuestDmemWriteTraceResult : uint8_t {
    Undetermined,
    WriterObserved,
    NoSelectedWriteObserved,
};

struct GuestDmemWriteTraceEvent {
    uint64_t ordinal = 0;
    uint64_t fault_addr = 0;
    uint64_t fault_phys = 0;
    uint64_t writer_rip = 0;
    uint64_t next_rip = 0;
    int64_t tid = 0;
    uint32_t size = 0;
    // Nonzero only when the signal-safe decoder proves one contiguous memory-write width for the
    // faulting instruction. CR2 identifies the first inaccessible byte, not necessarily the operand
    // start: a multi-byte store faulting exactly at a page boundary therefore remains uncertain when
    // that fault byte is outside the selected interval.
    uint32_t decoded_write_size = 0;
    bool selected = false;
    // A page fault reports the first faulting byte, not the memory operand's full write extent. When
    // that byte begins outside the selected range and the bounded x86 decoder does not recognize the
    // opcode, a crossing store cannot be ruled out. A changed post-image cannot attribute the write
    // either because siblings share the process-wide RW step window. Such events are always reported
    // as selection-uncertain rather than falsely claiming selected=yes/no.
    bool selection_uncertain = false;
    // The selected bytes changed while every alias was process-wide RW. This is useful evidence about
    // the window, but does not attribute that change to an unrecognized faulting instruction: a sibling
    // may have written during the same interval.
    bool changed_during_window = false;
    bool coverage_valid_before = false;
    bool rearmed = false;
    // False when an invalidation removed the retained mapping before completion. `after` remains
    // value-initialized in that case and must never be presented as observed bytes.
    bool post_available = false;
    std::array<uint8_t, kGuestDmemWriteTraceMaxBytes> before{};
    std::array<uint8_t, kGuestDmemWriteTraceMaxBytes> after{};
};

struct GuestDmemWriteTraceSnapshot {
    GuestDmemWriteTraceConfig config{};
    GuestDmemWriteTraceStatus status = GuestDmemWriteTraceStatus::Disabled;
    GuestDmemWriteTraceInvalidReason invalid_reason = GuestDmemWriteTraceInvalidReason::None;
    GuestDmemWriteTraceResult result = GuestDmemWriteTraceResult::Undetermined;
    uint64_t allocation_matches = 0;
    uint64_t mapping_matches = 0;
    uint64_t page_faults = 0;
    uint64_t selected_faults = 0;
    uint64_t selection_uncertain_faults = 0;
    uint64_t completed_steps = 0;
    uint64_t rearms = 0;
    uint64_t host_write_rebases = 0;
    uint64_t rebase_in_flight_refusals = 0;
    uint64_t coverage_gaps = 0;
    uint64_t overflow_events = 0;
    uint64_t selected_occurrence = 0;
    uint64_t target_phys = 0;
    uint64_t target_addr = 0;
    std::array<uint8_t, kGuestDmemWriteTraceMaxBytes> initial{};
    uint32_t event_count = 0;
    std::array<GuestDmemWriteTraceEvent, kGuestDmemWriteTraceMaxEvents> events{};
};

enum class GuestWriteWatchFaultAction : uint8_t {
    NotHandled,
    Resume,
    SingleStep,
};

enum class GuestDmemWriteTraceStepAction : uint8_t {
    NotHandled,
    Complete,
    // The owned TF trap was consumed and must be cleared, but a sibling normal-context operation
    // invalidated the diagnostic during its one-instruction RW window. The event is retained without
    // re-arming and the snapshot carries the explicit invalid reason.
    CompleteInvalid,
    LockTimeout,
};

// Windows direct memory is section-backed so MEM_WRITE_WATCH cannot observe it. Page-protection
// watches are deliberately unsupported: Windows writes its exception-dispatch frame into the guest
// SysV red zone before a vectored handler can restore the page, corrupting valid guest locals.
// Callers receive Unknown and retain their exact byte-comparison fallback.
class GuestWriteWatch {
public:
    GuestWriteWatch() = default;
    ~GuestWriteWatch();
    GuestWriteWatch(const GuestWriteWatch&) = delete;
    GuestWriteWatch& operator=(const GuestWriteWatch&) = delete;
    GuestWriteWatch(GuestWriteWatch&& other) noexcept;
    GuestWriteWatch& operator=(GuestWriteWatch&& other) noexcept;

    static GuestWriteWatch create(uint64_t addr, uint64_t size);
    explicit operator bool() const { return id_ != 0; }
    GuestWriteWatchQuery query() const;
    // A removed/replaced original mapping or a changed alias protection requires reset/create;
    // rearm cannot retarget a watch to new physical memory or bless stale protection metadata.
    bool rearm();
    void reset();

private:
    explicit GuestWriteWatch(uint64_t id) : id_(id) {}
    uint64_t id_ = 0;
};

GuestWriteWatchStats guest_write_watch_stats();

// Direct-memory mapping notifications. A topology/protection change invalidates existing watches;
// their next query is Dirty/Unknown and the exact path may establish a fresh complete registration.
void guest_write_watch_notify_direct_mapping_added(uint64_t addr, uint64_t size, uint64_t phys,
                                                    uint32_t protection);
void guest_write_watch_notify_direct_mapping_removed(uint64_t addr, uint64_t size);
void guest_write_watch_notify_direct_mapping_protection(uint64_t addr, uint64_t size,
                                                        uint32_t protection);
void guest_write_watch_notify_physical_write(uint64_t phys, uint64_t size);
void guest_write_watch_invalidate_all();

// PROSPER_TARGET_PHYS (#2932): resolve a guest VA to its physical address through the DIRECT-MAPPING
// ALIAS TABLE -- the ranges recorded by guest_write_watch_notify_direct_mapping_added above, not the
// dmem write-trace, which is a narrower thing armed for one investigation at a time.
//
// Written to test one hypothesis about Stray (#2932): that a flipped scanout buffer and the render
// targets were two virtual mappings of ONE physical allocation, which would have explained a missing
// composite with no further defect. It exposes resolution the write-watch already performs -- it
// models exactly this aliasing in `WatchedPage { phys; aliases; }` -- so the idea could be TESTED
// before anything was built on it. It was FALSIFIED (`docs/STRAY_STATUS.md`, § Ruled out): nothing
// is aliased, and the scanout buffers turn out to be ordinary render targets in their own right.
//
// Kept because the question recurs and the answer should cost one run, not a rediscovery. Do NOT
// restate the framing that motivated it -- "prosper renders into one VA range while the guest flips
// a distant one, so no VA-keyed lookup can connect them" -- which earlier revisions of this comment
// asserted as fact and which the census disproved.
//
// Diagnostic only; no production path calls it. Two limits a caller MUST respect:
//   * It returns the FIRST range containing `addr` and does NOT check that the range extends far
//     enough for the intended access. Fine for a probe; a production caller needs the size check.
//   * `false` means "no alias covers this VA" OR "the alias table holds nothing to search", which
//     are very different facts. Pass `alias_count` and report it beside a false, or the reader will
//     take an unarmed instrument for a negative result.
//
// `alias_count` (optional) receives the SEARCHED DOMAIN -- how many direct-mapping alias ranges were
// available to match against -- read under the SAME lock acquisition as the lookup, so the pair is
// mutually consistent rather than two samples of a table that can change between them. Zero on
// Windows (page-protection watches never arm there) and on any run that never reached
// guest_write_watch_set_fault_onstack.
bool guest_write_watch_va_to_phys(uint64_t addr, uint64_t& phys, size_t* alias_count = nullptr);

// Called by the HLE, by guest VA range, immediately BEFORE a host/kernel store into guest memory (e.g. a
// read()/pread() that streams bytes straight into a guest dmem buffer). Restores write on any armed pages
// the range overlaps and marks them Dirty. No-op on Windows/macOS (no pages are ever armed there).
void guest_write_watch_notify_host_write(uint64_t addr, uint64_t size);
// Paired completion for the call above. Optional: not calling it only disables rebaselining.
void guest_write_watch_notify_host_write_done(uint64_t addr, uint64_t size);

// Device/DMA writes already carry an exact guest VA range. Mark only registrations whose logical
// source overlaps that range; unlike a CPU protection fault, an adjacent write on the same host page
// need not create a false dirty result.
void guest_write_watch_notify_gpu_write(uint64_t addr, uint64_t size);

// Retained for callers that can only resume a handled fault and cannot own a TF completion. On Linux
// it preserves production dirty-watch behavior but fail-visibly invalidates/opens an overlapping dmem
// diagnostic rather than publishing a step nobody can complete. Windows returns false because
// page-fault write watches are unsafe for SysV guest code.
bool guest_write_watch_handle_fault(uint64_t addr);

// Extended Linux signal-path interface. The SIGSEGV caller sets TF only for SingleStep. On the
// following TRAP_TRACE it calls complete_step, clears TF, and may log the returned bounded event.
// complete_step has a lock-free pending-TID gate, so unrelated TRAP_TRACE users never wait for or
// enter the dmem state lock. Other platforms return NotHandled/false.
GuestWriteWatchFaultAction guest_write_watch_handle_fault_ex(uint64_t addr, uint64_t writer_rip,
                                                             int64_t tid,
                                                             bool trap_flag_already_owned = false);
GuestDmemWriteTraceStepAction guest_dmem_write_trace_complete_step(
    int64_t tid, uint64_t next_rip, GuestDmemWriteTraceEvent& event);

// Signal-safe bounded decoder used by the provenance trace. Returns a nonzero ascending contiguous
// write width only for recognized single-destination MOV encodings; zero is explicit unknown.
uint32_t guest_dmem_write_trace_decode_contiguous_store_size(const uint8_t* instruction);

// Formats only the event's post-image payload. Unavailable observations produce the literal
// `unavailable`, never the value-initialized contents of `after`. The returned count excludes NUL.
size_t guest_dmem_write_trace_format_post(const GuestDmemWriteTraceEvent& event,
                                          char* output, size_t capacity);

// Normal-context setup/inspection. Production uses init_from_environment; configure is also the
// deterministic test seam. Reconfiguration while a one-instruction TF step is pending is refused
// without changing the live trace; invalidity never abandons signal ownership. Environment syntax is
// Four fields preserve strict uniqueness:
//   PROSPER_DMEM_WRITE_TRACE=<caller-chain>:<allocation-size>:<offset>:<bytes>
// An explicit run-relative occurrence (1..4096) can disambiguate one exact allocation family:
//   PROSPER_DMEM_WRITE_TRACE=<caller-chain>:<allocation-size>:<occurrence>:<offset>:<bytes>
// PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS is optional (1..64). PROSPER_DMEM_CALLER must also be enabled.
void guest_dmem_write_trace_init_from_environment();
bool guest_dmem_write_trace_configure(const GuestDmemWriteTraceConfig& config);
bool guest_dmem_write_trace_enabled();
GuestDmemWriteTraceSnapshot guest_dmem_write_trace_snapshot();
void guest_dmem_write_trace_report();

// Called immediately after the same bounded interner that prints PROSPER_DMEM_CALLER's allocation
// record. chain_id==0 is an explicit unknown/overflow chain and cannot match a configured selector.
void guest_dmem_write_trace_notify_allocation(uint64_t phys, uint64_t size, uint32_t chain_id);

// Deterministic contention seam for the signal-path canary. Production never sets the hook or calls
// the explicit lock helpers. The hook runs once after complete_step proves the state lock is busy.
using GuestDmemWriteTraceContentionHookForTest = void (*)();
void guest_dmem_write_trace_set_contention_hook_for_test(
    GuestDmemWriteTraceContentionHookForTest hook);
// Fires on entry to the normal-context protection helper that builds a dynamic rollback vector. A
// SIGSEGV invalidation must use its allocation-free precomputed-alias path and never invoke this hook.
using GuestDmemWriteTraceDynamicProtectionHookForTest = void (*)();
void guest_dmem_write_trace_set_dynamic_protection_hook_for_test(
    GuestDmemWriteTraceDynamicProtectionHookForTest hook);
// Force the host-write rebaseline mode on/off regardless of the environment. The mode is otherwise a
// cached getenv read on first use, which makes it order-dependent and effectively untestable in
// process -- and three rounds of review found three defects in this feature with no test able to
// reach any of them. -1 restores the environment's answer.
void guest_dmem_write_trace_set_rebase_enabled_for_test(int enabled);
void guest_dmem_write_trace_lock_state_for_test();
void guest_dmem_write_trace_unlock_state_for_test();

// The Linux page-protection watch (mprotect + SIGSEGV) is only red-zone-safe when the SIGSEGV handler
// runs on a sigaltstack (SA_ONSTACK): without it the kernel writes the signal frame into the guest's
// SysV red zone, corrupting live locals of the faulting store. exec_image_linux reports whether the
// default-on handler has SA_ONSTACK; a diagnostic opt-out makes `create` refuse to arm so callers
// retain the exact byte-comparison fallback. No-op on other platforms.
void guest_write_watch_set_fault_onstack(bool on_altstack);

} // namespace prosper::host
