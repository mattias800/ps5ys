// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/diagnostics/watch_list.hpp"   // strict 0x-only watch parsing (shared with the RTT watch)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not silently drop an operator-set cap
#include <cstdint>
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/timeline/gpu_timeline.hpp"
#include "gpu/capture/capture_compute_policy.hpp"
#include "gpu/diagnostics/compute_parent_walk.hpp"
#include "gpu/diagnostics/shader_dump_filter.hpp"  // PROSPER_SHADER_DUMP_PROGRAM address filter
#include "gpu/diagnostics/compute_tree_watch.hpp"
#include "gpu/present/videoout_present.hpp"   // present_write_frame
#include "gpu/agc/agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "gpu/resources/mip_chain_plan.hpp"  // shader_resource_compute_mip_chain_levels (#3048)
#include "gpu/pm4/pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include "gpu/recompiler/rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"     // recompile_compute
#include "gpu/capture/writer_provenance.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "host/memory/guest_memory_query.hpp"
#include "host/memory/guest_write_watch.hpp"
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <filesystem>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <set>
#include <tuple>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#endif

// Look up a registered AGC shader header by its bound code address (hle_agc.cpp). Layout-compatible
// with gpu::AgcShaderHeader (file_header@0, user_data@0x08, code@0x10, type@0x5a).
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);
// #305 instrument (hle_agc.cpp): every registered shader bound to one code address, oldest first.
extern "C" size_t prosper_agc_shader_headers_for_code(uint64_t code_addr, const void** out,
                                                      size_t max);
extern "C" size_t prosper_agc_shader_count();
extern "C" const void* prosper_agc_shader_at(size_t index);

namespace prosper::gpu {

bool should_log_recompile_reject(uint64_t es_addr, uint64_t ps_addr,
                                 size_t vs_words, size_t gs_words, size_t fs_words,
                                 uint64_t* occurrence) {
    using Key = std::tuple<uint64_t, uint64_t, size_t, size_t, size_t>;
    static std::mutex mutex;
    static std::map<Key, uint64_t> counts;
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t& count = counts[{es_addr, ps_addr, vs_words, gs_words, fs_words}];
    if (count != UINT64_MAX) ++count;
    if (occurrence) *occurrence = count;
    return count != 0 && (count & (count - 1)) == 0;
}

namespace {
LiveRenderFn g_live;   // empty until the runtime/test registers a device-backed renderer
LiveComputeFn g_compute;   // synchronous compute backend, registered with the live Vulkan frontend
GuestGpuWriteObserver g_guest_gpu_write_observer;
ComputeAuthorityBoundaryObserver g_compute_authority_boundary_observer;
std::mutex g_compute_authority_boundary_mutex;
std::atomic<bool> g_compute_authority_boundary_enabled{false};
thread_local LiveRenderPhase g_live_phase;
std::array<uint8_t, 64 * 1024> g_compute_gds{};

struct GuestGpuWriteRange {
    uint64_t addr = 0;
    uint64_t size = 0;
};
struct GuestGpuWriteJournal {
    bool active = false;
    bool overflowed = false;
    uint64_t submit_serial = 0;
    std::vector<GuestGpuWriteRange> writes;
};
thread_local GuestGpuWriteJournal g_guest_gpu_writes;
std::atomic<uint64_t> g_next_guest_gpu_submit_serial{0};
std::atomic<uint64_t> g_next_shader_analysis_identity{0};

// PROSPER_NULL_IMAGE_SOURCE_PROBE=<hex-program-address> pairs an exactly-null sampled T# with the
// mapped bytes that produced it, then reads those bytes again after the retained submit executes.
// This answers whether eager graphics realization sampled a descriptor before an earlier ordered
// producer in the same submit. Program + MIMG pc + source address travel as one record: neither a
// numeric SRT offset nor adjacent log lines identify a descriptor uniquely (#2422, trap 156).
struct NullImageSourceProbe {
    uint64_t submit_no = 0;
    uint64_t program_addr = 0;
    uint64_t source_addr = 0;
    uint64_t draw_order = 0;
    uint64_t initial_writer_sequence = 0;
    uint32_t use_pc = UINT32_MAX;
    std::array<uint32_t, 8> initial{};
};
std::mutex g_null_image_source_probe_mutex;
std::mutex g_null_image_source_probe_run_mutex;
std::vector<NullImageSourceProbe> g_null_image_source_probes;
std::atomic<uint64_t> g_collect_null_image_source_probe_submit{0};

uint64_t null_image_source_probe_program() {
    static const uint64_t program = [] {
        const char* value = std::getenv("PROSPER_NULL_IMAGE_SOURCE_PROBE");
        if (!value || !*value) return uint64_t{0};
        return static_cast<uint64_t>(std::strtoull(value, nullptr, 16));
    }();
    return program;
}

void record_null_image_source_probe(uint64_t program_addr, uint32_t use_pc,
                                    uint64_t source_addr, uint64_t draw_order,
                                    const std::array<uint32_t, 8>& initial) {
    const uint64_t filter = null_image_source_probe_program();
    const uint64_t submit_no =
        g_collect_null_image_source_probe_submit.load(std::memory_order_relaxed);
    if (!submit_no ||
        !filter || program_addr != filter || !source_addr || !draw_order) return;
    static std::once_flag banner;
    std::call_once(banner, [filter] {
        std::fprintf(stderr,
                     "[null-source] armed program=0x%llx; exact-null x8 sources will be "
                     "re-read after their eager submit (writer history %s)\n",
                     static_cast<unsigned long long>(filter),
                     writer_provenance_enabled() ? "armed" : "NOT armed");
    });
    const auto initial_writer = last_guest_write_overlap(source_addr, sizeof(initial));
    std::lock_guard lock(g_null_image_source_probe_mutex);
    const auto same = [&](const NullImageSourceProbe& probe) {
        return probe.submit_no == submit_no && probe.program_addr == program_addr &&
               probe.use_pc == use_pc &&
               probe.source_addr == source_addr && probe.draw_order == draw_order;
    };
    if (std::none_of(g_null_image_source_probes.begin(),
                     g_null_image_source_probes.end(), same)) {
        g_null_image_source_probes.push_back({
            submit_no, program_addr, source_addr, draw_order,
            initial_writer ? initial_writer->sequence : 0u, use_pc, initial});
    }
}

std::vector<NullImageSourceProbe> take_null_image_source_probes(
        const std::vector<DrawItem>& draws, uint64_t submit_no) {
    if (!null_image_source_probe_program()) return {};
    std::unordered_set<uint64_t> orders;
    orders.reserve(draws.size());
    for (const DrawItem& draw : draws) orders.insert(draw.command_order);
    std::vector<NullImageSourceProbe> result;
    std::lock_guard lock(g_null_image_source_probe_mutex);
    auto it = g_null_image_source_probes.begin();
    while (it != g_null_image_source_probes.end()) {
        if (it->submit_no != submit_no) {
            ++it;
            continue;
        }
        if (orders.count(it->draw_order)) result.push_back(*it);
        // A stage table may have been built before a later pipeline failure prevented DrawItem
        // publication. Retire every record from this completed collection generation so a reused
        // command-order value in another submit cannot pick up stale provenance.
        it = g_null_image_source_probes.erase(it);
    }
    return result;
}

void check_null_image_source_probes(const std::vector<NullImageSourceProbe>& probes,
                                    uint64_t submit_no) {
    if (probes.empty()) return;
    size_t checked = 0, changed = 0, unreadable = 0;
    for (const NullImageSourceProbe& probe : probes) {
        if (!guest_readable(probe.source_addr, sizeof(probe.initial))) {
            ++unreadable;
            continue;
        }
        std::array<uint32_t, 8> current{};
        std::memcpy(current.data(), reinterpret_cast<const void*>(probe.source_addr),
                    sizeof(current));
        ++checked;
        if (current == probe.initial) continue;
        ++changed;
        const auto writer = last_guest_write_overlap(probe.source_addr, sizeof(current));
        std::fprintf(stderr,
                     "[null-source] CHANGED submit=%llu draw-order=%llu program=0x%llx pc=%u "
                     "source=0x%llx now=",
                     static_cast<unsigned long long>(submit_no),
                     static_cast<unsigned long long>(probe.draw_order),
                     static_cast<unsigned long long>(probe.program_addr), probe.use_pc,
                     static_cast<unsigned long long>(probe.source_addr));
        for (uint32_t word : current) std::fprintf(stderr, "%08x ", word);
        if (writer) {
            const bool new_writer = writer->sequence > probe.initial_writer_sequence;
            const bool ordered_predecessor = new_writer && writer->order < probe.draw_order &&
                writer->kind != GuestWriterKind::ColorTarget;
            std::fprintf(stderr,
                         "writer=%s writer-submit=%llu writer-order=%llu identity=0x%llx seq=%llu%s%s\n",
                         guest_writer_kind_name(writer->kind),
                         static_cast<unsigned long long>(writer->submit),
                         static_cast<unsigned long long>(writer->order),
                         static_cast<unsigned long long>(writer->identity),
                         static_cast<unsigned long long>(writer->sequence),
                         new_writer ? " NEW-SINCE-FOLD" : "",
                         ordered_predecessor ? " ORDERED-BEFORE-DRAW" : "");
        } else {
            std::fprintf(stderr,
                         "writer=none (history=%zu recorded: %s)%s\n",
                         guest_write_history_size(), guest_write_recorder_summary(),
                         guest_write_history_size() ? "" : " HISTORY-EMPTY");
        }
    }
    std::fprintf(stderr,
                 "[null-source] submit=%llu probes=%zu checked=%zu changed=%zu unreadable=%zu\n",
                 static_cast<unsigned long long>(submit_no), probes.size(), checked, changed,
                 unreadable);
}

void dispatch_compute_authority_boundary(const ComputeAuthorityBoundary& boundary) {
    // Default path: one relaxed load and no mutex/std::function copy when the opt-in diagnostic is
    // absent. This hook sits on every ordered draw, so making an unarmed census measurable would
    // defeat the performance investigation it exists to support.
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    ComputeAuthorityBoundaryObserver observer;
    {
        std::lock_guard lock(g_compute_authority_boundary_mutex);
        observer = g_compute_authority_boundary_observer;
    }
    if (observer) observer(boundary);
}

void notify_compute_authority_range(ComputeAuthorityBoundaryKind kind,
                                    uint64_t submit_no, uint64_t command_order,
                                    uint64_t address, uint64_t bytes) {
    const bool known = address != 0 && bytes != 0 &&
        address <= UINT64_MAX - (bytes - 1);
    dispatch_compute_authority_boundary(
        {kind, submit_no, command_order, address, bytes, known});
}

void notify_compute_authority_unknown(ComputeAuthorityBoundaryKind kind,
                                      uint64_t submit_no, uint64_t command_order = 0) {
    dispatch_compute_authority_boundary(
        {kind, submit_no, command_order, 0, 0, false});
}

void notify_compute_authority_draw_resources(const DrawItem& item, uint64_t submit_no) {
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    for (const ComputeAuthorityBoundary& boundary :
         compute_authority_draw_resource_boundaries(item, submit_no))
        dispatch_compute_authority_boundary(boundary);
}

void notify_compute_authority_draw_unrealized(uint64_t submit_no,
                                              uint64_t command_order) {
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    dispatch_compute_authority_boundary({
        ComputeAuthorityBoundaryKind::DrawResourceEnd,
        submit_no,
        command_order,
        0,
        0,
        false,
        UINT32_MAX,
        UINT32_MAX,
        false,
    });
}

bool ranges_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
    if (!a_size || !b_size) return false;
    return a <= b ? b - a < a_size : a - b < b_size;
}

class GuestGpuWriteSubmitScope {
public:
    GuestGpuWriteSubmitScope() {
        // Ordered execution is synchronous and non-recursive. If that contract is ever broken,
        // force conservative validation rather than silently losing the outer write history.
        if (g_guest_gpu_writes.active) {
            nested_ = true;
            g_guest_gpu_writes.overflowed = true;
            return;
        }
        g_guest_gpu_writes.active = true;
        g_guest_gpu_writes.overflowed = false;
        g_guest_gpu_writes.submit_serial =
            g_next_guest_gpu_submit_serial.fetch_add(1, std::memory_order_relaxed) + 1;
        g_guest_gpu_writes.writes.clear();
    }
    ~GuestGpuWriteSubmitScope() {
        if (!nested_) g_guest_gpu_writes.active = false;
    }

private:
    bool nested_ = false;
};

// Read a 32-dword user-data SGPR block from a stage's register file. `base` = the stage's
// SPI_SHADER_USER_DATA_*_0 register offset; absent registers read as 0. 32 (not 16) because NGG merged
// shaders place descriptors in the extended user SGPRs s16..s31 (e.g. vertex buffers at s16/s18).
static constexpr uint32_t kUserSgprs = 32;
void read_user_sgprs(const RegisterFile& sh, uint32_t base, uint32_t out[kUserSgprs]) {
    for (uint32_t i = 0; i < kUserSgprs; i++) { auto it = sh.find(base + i); out[i] = it == sh.end() ? 0u : it->second; }
}

} // namespace (guest_readable below has external linkage — declared in gpu_execute.hpp, shared with
  // the HLE diagnostic probes that chase raw guest pointers)

// PROSPER_DYNTRACE_FAIL support: while true, resolve_dynamic_fetch traces its walk and
// build_stage_table dumps the user-data blocks, regardless of the PROSPER_DYNTRACE/RESDUMP
// envs. Set (and cleared) by realize_draw_item's failure replay — the submit path is serialized
// by the HLE submit mutex, so a plain global is safe there.
bool g_dyntrace_force = false;

// PROSPER_DESCR_COHERENCE=1 — stage 0 of the runtime-selected-descriptor lift (#2412). Records the
// bytes seen at each descriptor address and counts how often the same address later yields DIFFERENT
// bytes, i.e. whether the guest rewrites descriptor-table slots under us. See the call site for why
// this gates the whole design rather than being a checkpoint inside it.
//
// Reports `checked` alongside `changed` because a bare zero cannot distinguish "the table is stable"
// from "the probe never ran"; `checked` is the control line. Also reports distinct addresses, so a
// large `checked` with tiny `distinct` (the same slot re-read every dispatch) is not mistaken for
// broad coverage.
bool descr_coherence_enabled() {
    static const bool on = std::getenv("PROSPER_DESCR_COHERENCE") != nullptr;
    return on;
}

namespace {
struct DescriptorCoherenceState {
    std::mutex mx;
    // Grows with DISTINCT addresses and is never pruned — ~125k entries on a 210 s GTA V route. Fine for
    // an opt-in diagnostic at that scale; a much longer route would want a cap or an eviction policy.
    std::unordered_map<uint64_t, uint64_t> seen;   // descriptor address -> last observed hash
    uint64_t checked = 0, changed = 0;
    // Summary is emitted PERIODICALLY, not at exit. The first version of this probe reported from a
    // destructor with a comment explaining that a summary must not depend on reaching a particular exit
    // path — and a destructor is exactly such a path: every routed run here is terminated by `timeout`,
    // whose SIGTERM runs no static destructors, so the run that produced the finding printed the
    // per-slot lines and no totals at all. Cost: one wasted 220 s run and a result with no denominator.
    //
    // Periodic emission has no such dependency: whatever the run does, the last line printed is a valid
    // running total. `checked` is the control (a bare `changed` cannot separate "stable" from "never
    // ran") and `distinct_addrs` guards against reading one slot re-read many times as broad coverage.
    static constexpr uint64_t kReportEvery = 20000;
};
DescriptorCoherenceState& descr_coherence_state() {
    static DescriptorCoherenceState s;
    return s;
}
}  // namespace

void record_descriptor_observation(uint64_t addr, uint64_t hash) {
    DescriptorCoherenceState& s = descr_coherence_state();
    std::lock_guard<std::mutex> lk(s.mx);
    ++s.checked;
    // Reported BEFORE the first-sight early return below, so emission depends only on `checked`.
    // Placed after it, this line could fire only on a REPEAT observation — and in a run whose slots are
    // rarely re-read inside the window that means `checked` in the hundreds of thousands, zero RUNNING
    // lines, and zero CHANGED lines (a first sight can never mismatch), which is indistinguishable from
    // the probe never having run. That is exactly the ambiguity the control exists to remove, and the
    // mechanism is trap 147: the report was not unarmed, it was STARVED by an early return consuming the
    // path before the trigger. Found in review; the published 8.5% figure was unaffected because both
    // measured runs reached repeat observations, but the next run need not have.
    if ((s.checked % DescriptorCoherenceState::kReportEvery) == 0)
        std::fprintf(stderr,
                     "[descr-coherence] RUNNING checked=%llu distinct_addrs=%zu changed=%llu\n",
                     (unsigned long long)s.checked, s.seen.size(), (unsigned long long)s.changed);
    auto it = s.seen.find(addr);
    if (it == s.seen.end()) { s.seen.emplace(addr, hash); return; }
    if (it->second != hash) {
        // Report the first few individually — an aggregate alone cannot say WHICH slot moves, and a
        // table whose entries all churn is a different problem from one slot being rewritten.
        if (s.changed < 8)
            std::fprintf(stderr, "[descr-coherence] CHANGED addr=0x%llx %016llx -> %016llx\n",
                         (unsigned long long)addr, (unsigned long long)it->second,
                         (unsigned long long)hash);
        ++s.changed;
        it->second = hash;
    }
}

bool dyntrace_failed_shader_enabled(uint64_t code_addr) {
    if (!std::getenv("PROSPER_DYNTRACE_FAIL")) return false;
    const char* filter = std::getenv("PROSPER_DYNTRACE_FAIL_ADDR");
    if (!filter) return true;

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(filter, &end, 16);
    return errno == 0 && end != filter && *end == '\0' && parsed == code_addr;
}

// PROSPER_COMPUTE_MEMPROBE — read the SAME guest dwords twice for one dispatch: once when the
// CPU-side const-fold resolves its descriptors, and once at command-ordered realization, just before
// the dispatch executes. It answers one question and only that one:
//
//     did the bytes this dispatch's descriptor was derived from change in between?
//
// Same bytes at both points means fold-time staleness cannot explain a wrong descriptor, whatever
// else might. Different bytes means it demonstrably can, for that dispatch, by observation rather
// than by a ratio over runs. It therefore fails informatively in both directions, which a
// success-rate comparison across routed runs does not: those runs reach different game state and
// their rates are computed over different populations (#2516).
//
// The address is derived per dispatch rather than fixed, because the interesting slot is at a
// constant offset from a POINTER the guest passes in. Format, all fields hex except the dword count:
//
//     PROSPER_COMPUTE_MEMPROBE=<code_addr>:<user_sgpr_index>:<byte_offset>:<dwords>
//     e.g. 413ce6000:0:a8:4   -> (user_sgprs[0] | user_sgprs[1]<<32) + 0xa8, 4 dwords
//
// Read-only and opt-in; a malformed value disables the probe rather than probing something else.
struct ComputeMemProbeSpec {
    bool on = false;
    uint64_t code_addr = 0;
    uint32_t sgpr_index = 0;
    uint64_t offset = 0;
    uint32_t dwords = 0;
};

const ComputeMemProbeSpec& compute_memprobe_spec() {
    static const ComputeMemProbeSpec spec = [] {
        ComputeMemProbeSpec s;
        const char* raw = std::getenv("PROSPER_COMPUTE_MEMPROBE");
        if (!raw || !*raw) return s;
        unsigned long long code = 0, off = 0;
        unsigned idx = 0, n = 0;
        if (std::sscanf(raw, "%llx:%u:%llx:%u", &code, &idx, &off, &n) != 4 || n == 0 || n > 64) {
            std::fprintf(stderr, "[memprobe] malformed PROSPER_COMPUTE_MEMPROBE=\"%s\" "
                                 "(want <code_addr>:<sgpr_index>:<byte_offset>:<dwords>) — probe disabled\n",
                         raw);
            return s;
        }
        s.on = true; s.code_addr = code; s.sgpr_index = idx; s.offset = off; s.dwords = n;
        std::fprintf(stderr, "[memprobe] armed: program=0x%llx addr=(user_sgprs[%u..%u])+0x%llx dwords=%u\n",
                     code, idx, idx + 1, off, n);
        return s;
    }();
    return spec;
}

struct ComputeMemProbeSample {
    uint64_t addr = 0;
    std::vector<uint32_t> words;
    bool readable = false;
};

// Keyed by dispatch identity, not by address: two dispatches of one program have different base
// pointers, and pairing them by address would silently compare different structures.
std::mutex g_memprobe_mu;
std::map<uint64_t /*addr*/, ComputeMemProbeSample> g_memprobe_fold;

bool compute_memprobe_read(const std::vector<uint32_t>& user_sgprs, ComputeMemProbeSample& out) {
    const ComputeMemProbeSpec& spec = compute_memprobe_spec();
    if (user_sgprs.size() <= spec.sgpr_index + 1) return false;
    const uint64_t base = static_cast<uint64_t>(user_sgprs[spec.sgpr_index]) |
                          (static_cast<uint64_t>(user_sgprs[spec.sgpr_index + 1]) << 32);
    if (!base) return false;
    out.addr = base + spec.offset;
    const uint32_t bytes = spec.dwords * 4u;
    out.readable = guest_readable(out.addr, bytes);
    out.words.assign(spec.dwords, 0u);
    if (out.readable)
        std::memcpy(out.words.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(out.addr)), bytes);
    return true;
}

std::string compute_memprobe_words(const ComputeMemProbeSample& s) {
    if (!s.readable) return "<unreadable>";
    std::string out;
    char buf[16];
    for (size_t i = 0; i < s.words.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%s%08x", i ? ":" : "", s.words[i]);
        out += buf;
    }
    return out;
}

void compute_memprobe_flush(const char* whence);

void compute_memprobe_at_fold(uint64_t code_addr, uint64_t dispatch_index,
                              const std::vector<uint32_t>& user_sgprs) {
    const ComputeMemProbeSpec& spec = compute_memprobe_spec();
    if (!spec.on || code_addr != spec.code_addr) return;
    compute_memprobe_flush("next-fold");
    ComputeMemProbeSample s;
    if (!compute_memprobe_read(user_sgprs, s)) {
        // Fail visibly. A probe that records nothing looks identical to a probe whose subject never
        // ran, and distinguishing those by absence of output is exactly the silent-instrument trap.
        std::fprintf(stderr, "[memprobe] program=0x%llx dispatch=%llu SKIPPED "
                             "(user_sgprs=%zu need>%u, base=%s)\n",
                     (unsigned long long)code_addr, (unsigned long long)dispatch_index,
                     user_sgprs.size(), compute_memprobe_spec().sgpr_index + 1,
                     user_sgprs.size() > compute_memprobe_spec().sgpr_index + 1 ? "zero" : "n/a");
        return;
    }
    std::fprintf(stderr, "[memprobe] program=0x%llx dispatch=%llu addr=0x%llx FOLD %s\n",
                 (unsigned long long)code_addr, (unsigned long long)dispatch_index,
                 (unsigned long long)s.addr, compute_memprobe_words(s).c_str());
    std::lock_guard<std::mutex> lock(g_memprobe_mu);
    g_memprobe_fold.emplace(s.addr, s);   // first fold of an address wins; later folds compare against it
}

// Re-read every recorded fold sample at its OWN address and report which changed.
//
// This is called at the ordered-execution boundary rather than from the dispatch path, and that is
// the whole point: a dispatch whose descriptors failed to resolve is SKIPPED before execution, so a
// re-read hung off the dispatch itself can only ever observe dispatches that succeeded -- exactly
// the population the question is not about. Keying on the address recorded at fold time keeps the
// comparison per dispatch while removing any need for the dispatch to run.
void compute_memprobe_flush(const char* whence) {
    const ComputeMemProbeSpec& spec = compute_memprobe_spec();
    if (!spec.on) return;
    std::map<uint64_t, ComputeMemProbeSample> pending;
    {
        std::lock_guard<std::mutex> lock(g_memprobe_mu);
        pending.swap(g_memprobe_fold);
    }
    const uint32_t bytes = spec.dwords * 4u;
    for (const auto& [key, folded] : pending) {
        (void)key;
        ComputeMemProbeSample now;
        now.addr = folded.addr;
        now.readable = guest_readable(now.addr, bytes);
        now.words.assign(spec.dwords, 0u);
        if (now.readable)
            std::memcpy(now.words.data(),
                        reinterpret_cast<const void*>(static_cast<uintptr_t>(now.addr)), bytes);
        const bool changed = folded.readable != now.readable || folded.words != now.words;
        std::fprintf(stderr, "[memprobe] program=0x%llx addr=0x%llx %s at=%s fold=%s later=%s\n",
                     (unsigned long long)compute_memprobe_spec().code_addr,
                     (unsigned long long)now.addr, changed ? "CHANGED" : "SAME", whence,
                     compute_memprobe_words(folded).c_str(), compute_memprobe_words(now).c_str());
    }
}

namespace {

struct ShaderResourceCompileKey {
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t img_dim = 0;
    uint32_t sample_count = 1;
    uint32_t declared_mip_levels = 1;
    // #3048: how many levels the compute backend materializes for this resource. The emitted module
    // bakes it in (the IMAGE_LOAD_MIP LOD clamp is a literal), so two descriptors that differ only
    // in their allocation-wide mip placement must not share a compiled module.
    uint32_t materialized_mip_levels = 1;
    bool in_mip_tail = false;
    bool compression_enabled = false;
    bool proven_zero_mip = false;
    uint32_t binding = 0;
    uint32_t stride = 0;
    uint32_t one_record_tail_semantic = 0;
    uint32_t atomic_x2_record_count = 0;
    uint32_t scalar_buffer_dword_count = 0;
    bool scalar_buffer_contract_valid = true;
    uint32_t srt_offset = 0;
    uint32_t sgpr_base = 0;
    uint32_t fetch_pc = 0;
    uint32_t fetch_index_mode = 0;
    uint32_t table_index_count = 0;
    uint32_t table_entry_stride = 0;
    uint32_t table_index_sgpr = UINT32_MAX;
    uint32_t table_selector_mode = 0;
    uint32_t table_load_pc = UINT32_MAX;
    bool table_contract_valid = true;
    uint32_t flat_base_sgpr = 0;
    uint32_t bvh_box_grow = 0;
    bool bvh_sort_enabled = false;
    bool null_bvh = false;
    bool zero_record_raw = false;
    bool optional_null_raw_load = false;
    bool proven_null_guarded_raw_store = false;
    bool proven_null_nullable_raw_buffer = false;
    bool gta5_cf9200_no_backing = false;
    uint32_t selected_sbuffer_soffset = UINT32_MAX;
    std::array<uint32_t, 4> selected_sbuffer_words{};
    uint32_t indirect_buffer_contract_tag = 0;
    uint32_t indirect_buffer_binding_bytes = 0;
    uint32_t indirect_buffer_slot_count = 0;
    uint32_t indirect_buffer_header_bytes = 0;
    uint32_t indirect_buffer_slot_bytes = 0;
    uint32_t indirect_pointer_carrier_version = 0;
    uint32_t indirect_pointer_proof_schema = 0;
    uint32_t indirect_pointer_binding_bytes = 0;
    uint32_t indirect_pointer_record_count = 0;
    uint32_t indirect_pointer_segment_count = 0;
    uint32_t indirect_pointer_segment_directory_byte_offset = 0;
    uint64_t indirect_pointer_proof_fingerprint = 0;
    bool srgb = false;
    bool depth_compare = false;
    uint32_t depth_compare_func = 0;
    uint32_t mag_filter = 0;
    uint32_t addr_u = 0;
    uint32_t addr_v = 0;
    uint32_t border_color_type = 0;
    bool normalize_unnormalized_coordinates = false;

    bool operator==(const ShaderResourceCompileKey&) const = default;
};

struct ShaderCompileKey {
    ShaderProgramStage stage = ShaderProgramStage::Vertex;
    bool has_resource_table = false;
    bool force_position_w = false;
    bool capture_position = false;
    bool has_pixel_inputs = false;
    PixelInputMapping pixel_inputs{};
    bool has_system_inputs = false;
    PixelSystemInputMapping system_inputs{};
    bool fragment_wave32 = false;
    bool has_pcrel_dispatch = false;
    uint32_t pcrel_dispatch_target = UINT32_MAX;
    // Compute modules also depend on launch ABI shape. User SGPR VALUES are push constants and stay
    // runtime data; only their count changes declarations. Exact thread extents matter because a
    // partial final workgroup emits a literal invocation guard into SPIR-V.
    bool has_compute_config = false;
    uint32_t compute_user_sgpr_count = 0;
    uint32_t compute_local_x = 1, compute_local_y = 1, compute_local_z = 1;
    bool compute_exact_thread_extent = false;
    uint32_t compute_threads_x = 0, compute_threads_y = 0, compute_threads_z = 0;
    uint32_t compute_wave_size = 64;
    uint32_t compute_pgm_rsrc1 = kDefaultComputePgmRsrc1;
    uint32_t compute_tidig_comp_cnt = 0;
    bool compute_tgid_x_en = false, compute_tgid_y_en = false, compute_tgid_z_en = false;
    bool compute_tg_size_en = false;
    uint32_t compute_lds_bytes = 0;
    uint32_t compute_native_subgroup_size = 0;
    uint32_t compute_native_storage_format_support = 0;
    bool compute_storage_buffer_int64_atomics = false;
    bool compute_packed_r11_storage = true;
    uint32_t vertex_lds_dwords = 0;
    uint32_t vertices_per_instance = 0;
    // Aliases ShaderCodeAnalysis::code and keeps that immutable analysis alive. Warm lookups used to
    // copy and re-hash the complete raw program for every draw before reaching the shader cache.
    std::shared_ptr<const std::vector<uint32_t>> code;
    uint64_t code_hash = 0;
    // Optional main program reached by a separately-installed vertex-fetch prolog. It remains a
    // distinct immutable analysis so cache identity covers both allocations without constructing a
    // transient concatenated buffer on every warm draw.
    std::shared_ptr<const std::vector<uint32_t>> chain_code;
    uint64_t chain_code_hash = 0;
    std::vector<ShaderResourceCompileKey> resources;
    // Diagnostic identity. All-default in production (PROSPER_CFG_TRIP_BOUND unset), which leaves
    // every key byte-identical to what it was before this field existed -- so caching behaviour is
    // provably unchanged unless the diagnostic is armed. When it IS armed, the emitted module depends
    // on the program's ADDRESS and on the selector state, neither of which any other key field can
    // see: the rest of the key is code bytes and launch shape, so a target and a non-target sharing
    // a body would otherwise share one compiled module and defeat the targeting.
    uint64_t trip_bound_program_address = 0;
    ComputeTripBoundSettings trip_bound{};
    size_t cached_hash = 0;

    bool operator==(const ShaderCompileKey& other) const {
        const bool same_code = code == other.code ||
            (code && other.code && *code == *other.code);
        const bool same_chain_code = chain_code == other.chain_code ||
            (chain_code && other.chain_code && *chain_code == *other.chain_code);
        return stage == other.stage &&
               trip_bound_program_address == other.trip_bound_program_address &&
               trip_bound.bound == other.trip_bound.bound &&
               trip_bound.only_program == other.trip_bound.only_program &&
               trip_bound.only_phase == other.trip_bound.only_phase &&
               trip_bound.only_ordinal == other.trip_bound.only_ordinal &&
               has_resource_table == other.has_resource_table &&
               force_position_w == other.force_position_w &&
               capture_position == other.capture_position &&
               has_pixel_inputs == other.has_pixel_inputs &&
               pixel_inputs == other.pixel_inputs &&
               has_system_inputs == other.has_system_inputs &&
               system_inputs == other.system_inputs &&
               fragment_wave32 == other.fragment_wave32 &&
               has_pcrel_dispatch == other.has_pcrel_dispatch &&
               pcrel_dispatch_target == other.pcrel_dispatch_target &&
               has_compute_config == other.has_compute_config &&
               compute_user_sgpr_count == other.compute_user_sgpr_count &&
               compute_local_x == other.compute_local_x &&
               compute_local_y == other.compute_local_y &&
               compute_local_z == other.compute_local_z &&
               compute_exact_thread_extent == other.compute_exact_thread_extent &&
               compute_threads_x == other.compute_threads_x &&
               compute_threads_y == other.compute_threads_y &&
               compute_threads_z == other.compute_threads_z &&
               compute_wave_size == other.compute_wave_size &&
               compute_pgm_rsrc1 == other.compute_pgm_rsrc1 &&
               compute_tidig_comp_cnt == other.compute_tidig_comp_cnt &&
               compute_tgid_x_en == other.compute_tgid_x_en &&
               compute_tgid_y_en == other.compute_tgid_y_en &&
               compute_tgid_z_en == other.compute_tgid_z_en &&
               compute_tg_size_en == other.compute_tg_size_en &&
               compute_lds_bytes == other.compute_lds_bytes &&
               compute_native_subgroup_size == other.compute_native_subgroup_size &&
               compute_native_storage_format_support ==
                   other.compute_native_storage_format_support &&
               compute_storage_buffer_int64_atomics ==
                   other.compute_storage_buffer_int64_atomics &&
               compute_packed_r11_storage == other.compute_packed_r11_storage &&
               vertex_lds_dwords == other.vertex_lds_dwords &&
               vertices_per_instance == other.vertices_per_instance &&
               resources == other.resources && same_code && same_chain_code;
    }
};

static uint64_t hash_mix(uint64_t hash, uint64_t value) {
    // FNV-1a over fixed-width values. Equality still compares the full key, so collisions are benign.
    for (unsigned i = 0; i < 8; ++i) {
        hash ^= static_cast<uint8_t>(value >> (i * 8));
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t hash_shader_code(const std::vector<uint32_t>& code) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : code) hash = hash_mix(hash, word);
    return hash;
}

struct ShaderCompileKeyHash {
    static size_t compute(const ShaderCompileKey& key) {
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, static_cast<uint32_t>(key.stage));
        hash = hash_mix(hash, key.trip_bound_program_address);
        hash = hash_mix(hash, key.trip_bound.bound);
        hash = hash_mix(hash, key.trip_bound.only_program);
        hash = hash_mix(hash, key.trip_bound.only_phase);
        hash = hash_mix(hash, key.trip_bound.only_ordinal);
        hash = hash_mix(hash, key.has_resource_table);
        hash = hash_mix(hash, key.force_position_w);
        hash = hash_mix(hash, key.capture_position);
        hash = hash_mix(hash, key.has_pixel_inputs);
        if (key.has_pixel_inputs) {
            hash = hash_mix(hash, key.pixel_inputs.valid_mask);
            hash = hash_mix(hash, key.pixel_inputs.passthrough_mask);
            hash = hash_mix(hash, key.pixel_inputs.consumed_mask);   // #2945
            hash = hash_mix(hash, key.pixel_inputs.consumed_known);
            for (uint32_t control : key.pixel_inputs.controls)
                hash = hash_mix(hash, control);
        }
        hash = hash_mix(hash, key.has_system_inputs);
        if (key.has_system_inputs) {
            hash = hash_mix(hash, key.system_inputs.ena);
            hash = hash_mix(hash, key.system_inputs.addr);
        }
        hash = hash_mix(hash, key.fragment_wave32);
        hash = hash_mix(hash, key.has_pcrel_dispatch);
        if (key.has_pcrel_dispatch) hash = hash_mix(hash, key.pcrel_dispatch_target);
        hash = hash_mix(hash, key.has_compute_config);
        if (key.has_compute_config) {
            hash = hash_mix(hash, key.compute_user_sgpr_count);
            hash = hash_mix(hash, key.compute_local_x);
            hash = hash_mix(hash, key.compute_local_y);
            hash = hash_mix(hash, key.compute_local_z);
            hash = hash_mix(hash, key.compute_exact_thread_extent);
            hash = hash_mix(hash, key.compute_threads_x);
            hash = hash_mix(hash, key.compute_threads_y);
            hash = hash_mix(hash, key.compute_threads_z);
            hash = hash_mix(hash, key.compute_wave_size);
            hash = hash_mix(hash, key.compute_pgm_rsrc1);
            hash = hash_mix(hash, key.compute_tidig_comp_cnt);
            hash = hash_mix(hash, key.compute_tgid_x_en);
            hash = hash_mix(hash, key.compute_tgid_y_en);
            hash = hash_mix(hash, key.compute_tgid_z_en);
            hash = hash_mix(hash, key.compute_tg_size_en);
            hash = hash_mix(hash, key.compute_lds_bytes);
            hash = hash_mix(hash, key.compute_native_subgroup_size);
            hash = hash_mix(hash, key.compute_native_storage_format_support);
            hash = hash_mix(hash, key.compute_storage_buffer_int64_atomics);
            hash = hash_mix(hash, key.compute_packed_r11_storage);
        }
        hash = hash_mix(hash, key.vertex_lds_dwords);
        hash = hash_mix(hash, key.vertices_per_instance);
        hash = hash_mix(hash, key.code ? key.code->size() : 0u);
        hash = hash_mix(hash, key.code_hash);
        hash = hash_mix(hash, key.chain_code ? key.chain_code->size() : 0u);
        hash = hash_mix(hash, key.chain_code_hash);
        hash = hash_mix(hash, key.resources.size());
        for (const auto& resource : key.resources) {
            hash = hash_mix(hash, resource.cls);
            hash = hash_mix(hash, resource.format);
            hash = hash_mix(hash, resource.num_components);
            hash = hash_mix(hash, resource.width);
            hash = hash_mix(hash, resource.height);
            hash = hash_mix(hash, resource.depth);
            hash = hash_mix(hash, resource.img_dim);
            hash = hash_mix(hash, resource.sample_count);
            hash = hash_mix(hash, resource.declared_mip_levels);
            hash = hash_mix(hash, resource.materialized_mip_levels);
            hash = hash_mix(hash, resource.in_mip_tail);
            hash = hash_mix(hash, resource.compression_enabled);
            hash = hash_mix(hash, resource.proven_zero_mip);
            hash = hash_mix(hash, resource.binding);
            hash = hash_mix(hash, resource.stride);
            hash = hash_mix(hash, resource.one_record_tail_semantic);
            hash = hash_mix(hash, resource.atomic_x2_record_count);
            hash = hash_mix(hash, resource.scalar_buffer_dword_count);
            hash = hash_mix(hash, resource.scalar_buffer_contract_valid);
            hash = hash_mix(hash, resource.srt_offset);
            hash = hash_mix(hash, resource.sgpr_base);
            hash = hash_mix(hash, resource.fetch_pc);
            hash = hash_mix(hash, resource.fetch_index_mode);
            hash = hash_mix(hash, resource.table_index_count);
            hash = hash_mix(hash, resource.table_entry_stride);
            hash = hash_mix(hash, resource.table_index_sgpr);
            hash = hash_mix(hash, resource.table_selector_mode);
            hash = hash_mix(hash, resource.table_load_pc);
            hash = hash_mix(hash, resource.table_contract_valid);
            hash = hash_mix(hash, resource.flat_base_sgpr);
            hash = hash_mix(hash, resource.bvh_box_grow);
            hash = hash_mix(hash, resource.bvh_sort_enabled);
            hash = hash_mix(hash, resource.null_bvh);
            hash = hash_mix(hash, resource.zero_record_raw);
            hash = hash_mix(hash, resource.optional_null_raw_load);
            hash = hash_mix(hash, resource.proven_null_guarded_raw_store);
            hash = hash_mix(hash, resource.proven_null_nullable_raw_buffer);
            hash = hash_mix(hash, resource.gta5_cf9200_no_backing);
            hash = hash_mix(hash, resource.selected_sbuffer_soffset);
            for (const uint32_t word : resource.selected_sbuffer_words)
                hash = hash_mix(hash, word);
            hash = hash_mix(hash, resource.indirect_buffer_contract_tag);
            hash = hash_mix(hash, resource.indirect_buffer_binding_bytes);
            hash = hash_mix(hash, resource.indirect_buffer_slot_count);
            hash = hash_mix(hash, resource.indirect_buffer_header_bytes);
            hash = hash_mix(hash, resource.indirect_buffer_slot_bytes);
            hash = hash_mix(hash, resource.indirect_pointer_carrier_version);
            hash = hash_mix(hash, resource.indirect_pointer_proof_schema);
            hash = hash_mix(hash, resource.indirect_pointer_binding_bytes);
            hash = hash_mix(hash, resource.indirect_pointer_record_count);
            hash = hash_mix(hash, resource.indirect_pointer_segment_count);
            hash = hash_mix(
                hash, resource.indirect_pointer_segment_directory_byte_offset);
            hash = hash_mix(hash, resource.indirect_pointer_proof_fingerprint);
            hash = hash_mix(hash, resource.srgb);
            hash = hash_mix(hash, resource.depth_compare);
            hash = hash_mix(hash, resource.depth_compare_func);
            hash = hash_mix(hash, resource.mag_filter);
            hash = hash_mix(hash, resource.addr_u);
            hash = hash_mix(hash, resource.addr_v);
            hash = hash_mix(hash, resource.border_color_type);
            hash = hash_mix(hash, resource.normalize_unnormalized_coordinates);
        }
        return static_cast<size_t>(hash);
    }

    size_t operator()(const ShaderCompileKey& key) const { return key.cached_hash; }
};

struct CachedShader {
    SharedShaderWords spirv;
    uint64_t identity = 0;
    mutable std::atomic<uint64_t> last_use{0};
    uint64_t bytes = 0;

    CachedShader() = default;
    CachedShader(const CachedShader& other)
        : spirv(other.spirv), identity(other.identity),
          last_use(other.last_use.load(std::memory_order_relaxed)),
          bytes(other.bytes) {}
    CachedShader& operator=(const CachedShader& other) {
        if (this != &other) {
            spirv = other.spirv;
            identity = other.identity;
            last_use.store(other.last_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
            bytes = other.bytes;
        }
        return *this;
    }
};

struct ShaderCache {
    std::shared_mutex mutex;
    std::unordered_map<ShaderCompileKey, CachedShader, ShaderCompileKeyHash> entries;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> bypasses{0};
    std::atomic<uint64_t> use_counter{0};
    ShaderRecompileCacheStats stats;
    uint64_t next_identity = 1;
};

static_assert(std::is_same_v<decltype(ShaderCache::mutex), std::shared_mutex>,
              "ShaderCache::mutex must be std::shared_mutex for concurrent read scaling");

ShaderCache& shader_cache() {
    static ShaderCache cache;
    return cache;
}

struct DecodedShader {
    std::vector<uint32_t> code;
    std::vector<Rdna2Inst> instructions;
    std::vector<Rdna2Inst> shader_constant_instructions;
    size_t source_dwords = 0;
    bool shader_constant_specialized = false;
    bool terminated = false;
    uint64_t bytes = 0;
};

struct DecodedShaderEntry {
    std::shared_ptr<const DecodedShader> shader;
    uint64_t last_use = 0;
};

struct ShaderDecodeCache {
    std::mutex mutex;
    std::unordered_map<uintptr_t, DecodedShaderEntry> entries;
    ShaderDecodeCacheStats stats;
    uint64_t use_counter = 0;
};

struct ShaderCodeAnalysis {
    std::vector<uint32_t> code;
    uint64_t code_hash = 0;
    PcrelDispatchInfo pcrel_dispatch;
    uint64_t identity = 0;
    size_t source_dwords = 0;
    bool bounded_span = false;
    uint64_t bytes = 0;
};

struct ShaderCodeAnalysisEntry {
    std::shared_ptr<const ShaderCodeAnalysis> analysis;
    uint64_t last_use = 0;
};

struct ShaderAnalysisCache {
    std::mutex mutex;
    std::unordered_map<uintptr_t, ShaderCodeAnalysisEntry> entries;
    ShaderAnalysisCacheStats stats;
    uint64_t use_counter = 0;
};

struct InterpolationCacheKey {
    uint64_t analysis_identity = 0;
    PixelSystemInputMapping system_inputs{};
    bool has_system_inputs = false;
    uint32_t passthrough_mask = 0;
    // #3051: FragmentInterpolationLayout now also carries flat_mask, derived from pixel_inputs's raw
    // control words exactly as passthrough_mask is (PixelInputMapping::effective_flat_mask()). It
    // must be part of this key for the same reason passthrough_mask is: SPI_PS_INPUT_CNTL is a
    // context register that can carry a different FLAT_SHADE bit across two draws sharing identical
    // shader bytes, and this cache is keyed on shader identity alone otherwise -- omitting it would
    // let a flat-shaded draw's layout silently answer a later non-flat draw of the same shader, or
    // vice versa.
    uint32_t flat_mask = 0;

    bool operator==(const InterpolationCacheKey&) const = default;
};

struct InterpolationCacheKeyHash {
    size_t operator()(const InterpolationCacheKey& key) const {
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, key.analysis_identity);
        hash = hash_mix(hash, key.has_system_inputs);
        if (key.has_system_inputs) {
            hash = hash_mix(hash, key.system_inputs.ena);
            hash = hash_mix(hash, key.system_inputs.addr);
        }
        hash = hash_mix(hash, key.passthrough_mask);
        hash = hash_mix(hash, key.flat_mask);
        return static_cast<size_t>(hash);
    }
};

struct CachedInterpolationLayout {
    FragmentInterpolationLayout layout;
    uint64_t last_use = 0;
};

struct InterpolationCache {
    std::mutex mutex;
    std::unordered_map<InterpolationCacheKey, CachedInterpolationLayout,
                       InterpolationCacheKeyHash> entries;
    uint64_t use_counter = 0;
};

struct StageFoldProfileEntry {
    uint64_t code = 0;
    uint32_t user_base = 0;
    uint64_t calls = 0;
    uint64_t instructions = 0;
    uint64_t dynamic_fetches = 0;
    uint64_t srt_uses = 0;
    uint64_t code_dwords = 0;
    uint64_t guest_probes = 0;
    double total_ms = 0.0;
    double decode_ms = 0.0;
    double guest_probe_ms = 0.0;
    double max_ms = 0.0;
};

struct StageFoldProfileKey {
    uint64_t code = 0;
    uint32_t user_base = 0;
    bool operator==(const StageFoldProfileKey&) const = default;
};

struct StageFoldProfileKeyHash {
    size_t operator()(const StageFoldProfileKey& key) const {
        const uint64_t mixed = key.code ^ (static_cast<uint64_t>(key.user_base) << 48);
        return static_cast<size_t>(mixed ^ (mixed >> 32));
    }
};

struct StageFoldProfiler {
    std::mutex mutex;
    std::unordered_map<StageFoldProfileKey, StageFoldProfileEntry,
                       StageFoldProfileKeyHash> window;
    uint64_t calls = 0;
};

StageFoldProfiler& stage_fold_profiler() {
    static StageFoldProfiler profiler;
    return profiler;
}

void record_stage_fold_profile(uint64_t code, uint32_t user_base, size_t code_dwords,
                               size_t instructions, size_t dynamic_fetches, size_t srt_uses,
                               uint64_t guest_probes, double elapsed_ms, double decode_ms,
                               double guest_probe_ms) {
    static const uint64_t interval = [] {
        const char* value = std::getenv("PROSPER_STAGE_FOLD_PROFILE_CALLS");
        if (!value || !*value) return 4096ull;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end != value && parsed > 0 ? parsed : 4096ull;
    }();
    StageFoldProfiler& profiler = stage_fold_profiler();
    std::lock_guard lock(profiler.mutex);
    StageFoldProfileEntry& entry = profiler.window[{code, user_base}];
    entry.code = code;
    entry.user_base = user_base;
    ++entry.calls;
    entry.instructions += instructions;
    entry.dynamic_fetches += dynamic_fetches;
    entry.srt_uses += srt_uses;
    entry.code_dwords += code_dwords;
    entry.guest_probes += guest_probes;
    entry.total_ms += elapsed_ms;
    entry.decode_ms += decode_ms;
    entry.guest_probe_ms += guest_probe_ms;
    entry.max_ms = std::max(entry.max_ms, elapsed_ms);
    if (++profiler.calls < interval) return;

    std::vector<StageFoldProfileEntry> ranked;
    ranked.reserve(profiler.window.size());
    for (const auto& item : profiler.window) ranked.push_back(item.second);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.total_ms > b.total_ms;
    });
    std::fprintf(stderr, "[stage-fold-profile] calls=%llu shaders=%zu top-by-total-ms:\n",
                 (unsigned long long)profiler.calls, ranked.size());
    for (size_t i = 0; i < std::min<size_t>(ranked.size(), 12); ++i) {
        const StageFoldProfileEntry& item = ranked[i];
        const double calls = static_cast<double>(item.calls);
        std::fprintf(stderr,
                     "[stage-fold-profile] code=0x%llx base=%u calls=%llu total=%.3f avg=%.3f "
                     "max=%.3f decode=%.3f probe=%.3f body=%.3f dw/call=%.1f ins/call=%.1f "
                     "probes/call=%.1f dyn/call=%.1f srt/call=%.1f\n",
                     (unsigned long long)item.code, item.user_base,
                     (unsigned long long)item.calls, item.total_ms, item.total_ms / calls,
                     item.max_ms, item.decode_ms / calls, item.guest_probe_ms / calls,
                     (item.total_ms - item.decode_ms - item.guest_probe_ms) / calls,
                     item.code_dwords / calls, item.instructions / calls,
                     item.guest_probes / calls, item.dynamic_fetches / calls, item.srt_uses / calls);
    }
    profiler.window.clear();
    profiler.calls = 0;
}

ShaderDecodeCache& shader_decode_cache() {
    static ShaderDecodeCache cache;
    return cache;
}

ShaderAnalysisCache& shader_analysis_cache() {
    static ShaderAnalysisCache cache;
    return cache;
}

InterpolationCache& interpolation_cache() {
    static InterpolationCache cache;
    return cache;
}

uint64_t shader_decode_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 64ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_DECODE_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 1024ull) * 1024 * 1024;
}

std::shared_ptr<const DecodedShader> decode_shader_cached(const uint32_t* code, size_t dwords) {
    auto decode = [&] {
        auto result = std::make_shared<DecodedShader>();
        result->source_dwords = dwords;
        std::vector<Rdna2Inst> decoded;
        const size_t consumed = rdna2_walk(code, dwords, decoded);
        result->code.assign(code, code + consumed);
        if (!decoded.empty()) {
            const Rdna2Inst& last = decoded.back();
            result->terminated = last.is_end || last.fmt == Rdna2Format::Unknown ||
                                 last.len_dwords == 0;
        }
        // The fold ignores most vector/control/export instructions unless the decoder reports an SGPR
        // destination (the conservative unknown-value invalidation). Scalar lane spills are the exception:
        // retain their spill VGPR writes too, so an ordinary VGPR write invalidates any saved lane values.
        std::set<int> scalar_spill_vgprs;
        std::set<int> fetch_vaddr_vgprs;
        std::set<int> zero_mip_vgprs;
        for (const Rdna2Inst& instruction : decoded) {
            if ((instruction.fmt == Rdna2Format::MUBUF ||
                 instruction.fmt == Rdna2Format::MTBUF) &&
                instruction.src[0].kind == OperandKind::VGPR)
                fetch_vaddr_vgprs.insert(instruction.src[0].value);
            uint32_t mip_vgpr = 0;
            if (rdna2_mimg_zero_mip_shape(instruction, &mip_vgpr))
                zero_mip_vgprs.insert(static_cast<int>(mip_vgpr));
            if (instruction.fmt == Rdna2Format::VOP3) {
                if (instruction.opcode == 0x361 && instruction.dst.kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.dst.value);       // v_writelane_b32
                else if (instruction.opcode == 0x360 && instruction.src[0].kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.src[0].value);    // v_readlane_b32
            }
        }
        auto retain_fold_instructions = [&](const std::vector<Rdna2Inst>& source,
                                            std::vector<Rdna2Inst>& retained) {
            retained.reserve(source.size());
            for (const Rdna2Inst& instruction : source) {
                if (instruction.is_end) break;
                const bool scalar_spill = instruction.fmt == Rdna2Format::VOP3 &&
                                          (instruction.opcode == 0x360 || instruction.opcode == 0x361);
                const bool vector_index_select =
                    ((instruction.fmt == Rdna2Format::VOP3 && instruction.opcode == 0x101) ||
                     (instruction.fmt == Rdna2Format::VOP2 && instruction.opcode == 0x01)) &&
                    instruction.dst.kind == OperandKind::VGPR &&
                    fetch_vaddr_vgprs.contains(instruction.dst.value);
                // Index provenance begins at the hardware ABI VGPRs and is killed by any later shader
                // computation of a register used as VADDR. Keep only writes to actual fetch-address
                // registers; retaining every VALU instruction would make the otherwise-small scalar fold
                // walk large UE shaders in full. This is what distinguishes DQ's first v5=vertex_id fetch
                // from its later v5=3*vertex_id+1 packed-attribute fetch.
                const bool fetch_vaddr_write = instruction.dst.kind == OperandKind::VGPR &&
                                               fetch_vaddr_vgprs.contains(instruction.dst.value);
                const bool scalar_spill_invalidation = instruction.dst.kind == OperandKind::VGPR &&
                                                       scalar_spill_vgprs.contains(instruction.dst.value);
                // The zero-mip proof needs the unambiguous reaching definition of one exact address
                // VGPR. Retain every possible writer whose (at most four-dword) result overlaps it;
                // false-positive retention is cheap, while dropping one would accept a stale v_mov.
                bool zero_mip_write = false;
                if (instruction.dst.kind == OperandKind::VGPR) {
                    for (int reg : zero_mip_vgprs)
                        if (instruction.dst.value <= reg && instruction.dst.value + 3 >= reg) {
                            zero_mip_write = true;
                            break;
                        }
                }
                const bool zero_mip_definition = zero_mip_write &&
                    instruction.fmt == Rdna2Format::VOP1 && instruction.opcode == 0x01u &&
                    instruction.len_dwords == 1u && !instruction.has_modifier &&
                    instruction.src[0].kind == OperandKind::SGPR;
                const bool zero_mip_intervening_write =
                    zero_mip_write && !zero_mip_definition;
                const bool fold_format = instruction.fmt == Rdna2Format::SOP1 ||
                                         instruction.fmt == Rdna2Format::SOP2 ||
                                         instruction.fmt == Rdna2Format::SOPC ||
                                         instruction.fmt == Rdna2Format::SOPK ||
                                         instruction.fmt == Rdna2Format::SOPP ||
                                         instruction.fmt == Rdna2Format::SMEM ||
                                         instruction.fmt == Rdna2Format::MIMG ||
                                         instruction.fmt == Rdna2Format::MUBUF ||
                                         instruction.fmt == Rdna2Format::MTBUF;
                if (fold_format || rdna2_instruction_may_change_exec(instruction) ||
                    scalar_spill || vector_index_select || fetch_vaddr_write ||
                    scalar_spill_invalidation || zero_mip_definition ||
                    zero_mip_intervening_write ||
                    instruction.dst.kind == OperandKind::SGPR)
                    retained.push_back(instruction);
            }
        };
        // Retain only instructions that can affect fold state or emit a descriptor use, preserving
        // their original order and PCs. Prove shader-constant branches against the FULL decoded
        // stream first: the compact fold stream intentionally omits most VALU, including implicit
        // VCC writers that must invalidate a scalar-data proof through s106:s107.
        retain_fold_instructions(decoded, result->instructions);
        std::vector<Rdna2Inst> shader_constant_decoded = decoded;
        result->shader_constant_specialized =
            rdna2_specialize_shader_constant_branches(shader_constant_decoded) != 0;
        if (result->shader_constant_specialized)
            retain_fold_instructions(shader_constant_decoded,
                                     result->shader_constant_instructions);
        result->bytes = static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
                        static_cast<uint64_t>(result->instructions.size() +
                                              result->shader_constant_instructions.size()) *
                            sizeof(Rdna2Inst);
        return result;
    };

    if (!code || !dwords) return decode();
    auto& cache = shader_decode_cache();
    if (getenv("PROSPER_NO_SHADER_DECODE_CACHE")) {
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        return decode();
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(code);
    {
        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(address);
        if (found != cache.entries.end()) {
            const auto& cached = found->second.shader;
            const bool compatible_length = cached->terminated || cached->source_dwords == dwords;
            const uint64_t code_bytes = static_cast<uint64_t>(cached->code.size()) * sizeof(uint32_t);
            const bool readable = code_bytes == 0 ||
                                  (code_bytes <= UINT32_MAX && guest_readable(address, (uint32_t)code_bytes));
            if (compatible_length && cached->code.size() <= dwords && readable &&
                (code_bytes == 0 || memcmp(code, cached->code.data(), (size_t)code_bytes) == 0)) {
                ++cache.stats.hits;
                found->second.last_use = ++cache.use_counter;
                return cached;
            }
            cache.stats.bytes -= cached->bytes;
            cache.entries.erase(found);
            ++cache.stats.invalidations;
        }
        ++cache.stats.misses;
    }

    auto decoded = decode();
    std::lock_guard lock(cache.mutex);
    // Another cold worker may have populated this address while decode ran. Replace it only after
    // removing its accounted bytes; returned shared_ptrs remain valid independently of the map entry.
    auto concurrent = cache.entries.find(address);
    if (concurrent != cache.entries.end()) {
        cache.stats.bytes -= concurrent->second.shader->bytes;
        cache.entries.erase(concurrent);
    }
    constexpr size_t max_entries = 4096;
    const uint64_t limit = shader_decode_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + decoded->bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.shader->bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (decoded->bytes <= limit && max_entries != 0) {
        cache.entries[address] = {decoded, ++cache.use_counter};
        cache.stats.bytes += decoded->bytes;
    }
    cache.stats.entries = cache.entries.size();
    return decoded;
}

uint64_t shader_analysis_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 64ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_ANALYSIS_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 1024ull) * 1024 * 1024;
}

std::shared_ptr<const ShaderCodeAnalysis> analyze_shader_code_cached(const uint32_t* code,
                                                                      size_t dwords) {
    auto analyze = [&] {
        auto result = std::make_shared<ShaderCodeAnalysis>();
        result->identity = ++g_next_shader_analysis_identity;
        result->source_dwords = dwords;
        const size_t span = rdna2_recompile_code_span(code, dwords);
        result->bounded_span = span < dwords;
        if (code && span) result->code.assign(code, code + span);
        result->code_hash = hash_shader_code(result->code);
        result->pcrel_dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        result->bytes = static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
                        static_cast<uint64_t>(result->pcrel_dispatch.target_pcs.size()) *
                            sizeof(uint32_t) +
                        static_cast<uint64_t>(result->pcrel_dispatch.setup_pcs.size()) *
                            sizeof(uint32_t);
        return result;
    };

    if (!code || !dwords) return analyze();
    auto& cache = shader_analysis_cache();
    // #2395: was a getenv PER LOOKUP -- 1,697,925 of them in one Blue Prince gameplay run,
    // and `getenv` showed at 1.34% of total CPU in a profile of that run. The environment
    // cannot change under a running process, so reading it once is not merely an
    // optimisation, it is the correct semantics. Same hoist as #2214 did for the live
    // renderer's per-resource reads.
    static const bool bypass_analysis_cache = getenv("PROSPER_NO_SHADER_ANALYSIS_CACHE") != nullptr;
    if (bypass_analysis_cache) {
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        return analyze();
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(code);
    // Copy the immutable candidate under the mutex, then validate guest bytes without it. Dense draw
    // realization performs these exact checks from several workers; holding one global cache mutex
    // across guest_readable + memcmp otherwise serializes every warm lookup.
    for (;;) {
        std::shared_ptr<const ShaderCodeAnalysis> cached;
        {
            std::lock_guard lock(cache.mutex);
            auto found = cache.entries.find(address);
            if (found == cache.entries.end()) {
                ++cache.stats.misses;
                break;
            }
            cached = found->second.analysis;
        }
        const bool compatible_length = cached->bounded_span
            ? cached->code.size() <= dwords : cached->source_dwords == dwords;
        const uint64_t code_bytes = static_cast<uint64_t>(cached->code.size()) * sizeof(uint32_t);
        const bool readable = code_bytes == 0 ||
                              (code_bytes <= UINT32_MAX &&
                               guest_readable(address, static_cast<uint32_t>(code_bytes)));
        const bool matches = compatible_length && readable &&
            (code_bytes == 0 ||
             memcmp(code, cached->code.data(), static_cast<size_t>(code_bytes)) == 0);

        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(address);
        if (found == cache.entries.end() || found->second.analysis != cached)
            continue; // Evicted/replaced while unlocked: validate the current version instead.
        if (matches) {
            ++cache.stats.hits;
            found->second.last_use = ++cache.use_counter;
            return cached;
        }
        cache.stats.bytes -= cached->bytes;
        cache.entries.erase(found);
        ++cache.stats.invalidations;
        ++cache.stats.misses;
        break;
    }

    auto analysis = analyze();
    std::lock_guard lock(cache.mutex);
    // Another cold worker may have populated this address while analysis ran. Keep exact byte
    // accounting when replacing it; each returned shared_ptr remains independently valid.
    auto concurrent = cache.entries.find(address);
    if (concurrent != cache.entries.end()) {
        cache.stats.bytes -= concurrent->second.analysis->bytes;
        cache.entries.erase(concurrent);
    }
    constexpr size_t max_entries = 4096;
    const uint64_t limit = shader_analysis_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + analysis->bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.analysis->bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (analysis->bytes <= limit && max_entries != 0) {
        cache.entries[address] = {analysis, ++cache.use_counter};
        cache.stats.bytes += analysis->bytes;
    }
    cache.stats.entries = cache.entries.size();
    return analysis;
}

uint64_t shader_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 128ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 4096ull) * 1024 * 1024;
}

uint64_t shader_cache_entry_bytes(const ShaderCompileKey& key, const std::vector<uint32_t>& spirv) {
    return static_cast<uint64_t>(key.code ? key.code->size() : 0u) * sizeof(uint32_t) +
           static_cast<uint64_t>(key.chain_code ? key.chain_code->size() : 0u) * sizeof(uint32_t) +
           static_cast<uint64_t>(key.resources.size()) * sizeof(ShaderResourceCompileKey) +
           (key.has_pixel_inputs ? sizeof(PixelInputMapping) : 0u) +
           (key.has_system_inputs ? sizeof(PixelSystemInputMapping) : 0u) +
           static_cast<uint64_t>(spirv.size()) * sizeof(uint32_t);
}

struct PcrelDispatchSelection {
    PcrelDispatchInfo dispatch;
    const ShaderResource* resource = nullptr;
    uint32_t raw_selector = 0;
    uint32_t target = UINT32_MAX;
    bool readable = false;
};

PcrelDispatchSelection select_pcrel_dispatch(const uint32_t* code, size_t dwords,
                                              const ShaderResourceTable* resources,
                                              const ShaderCodeAnalysis* analysis = nullptr) {
    PcrelDispatchSelection selection;
    if (!code || !dwords || !resources) return selection;
    selection.dispatch = analysis ? analysis->pcrel_dispatch
                                  : rdna2_pcrel_dispatch_info(code, dwords);
    if (!selection.dispatch.valid) return selection;

    selection.resource = resources->by_sgpr_base_cls(
        selection.dispatch.selector_sgpr_base, ResourceClass::ConstantBuffer);
    if (!selection.resource ||
        selection.dispatch.selector_byte_offset > selection.resource->size ||
        selection.resource->size - selection.dispatch.selector_byte_offset <
            sizeof(selection.raw_selector)) return selection;

    if (selection.resource->host_data &&
        selection.dispatch.selector_byte_offset <= selection.resource->host_data_size &&
        selection.resource->host_data_size - selection.dispatch.selector_byte_offset >=
            sizeof(selection.raw_selector)) {
        memcpy(&selection.raw_selector,
               selection.resource->host_data + selection.dispatch.selector_byte_offset,
               sizeof(selection.raw_selector));
        selection.readable = true;
    } else if (selection.resource->gpu_addr <=
               UINT64_MAX - selection.dispatch.selector_byte_offset) {
        const uint64_t address =
            selection.resource->gpu_addr + selection.dispatch.selector_byte_offset;
        if (guest_readable(address, sizeof(selection.raw_selector))) {
            memcpy(&selection.raw_selector,
                   reinterpret_cast<const void*>(static_cast<uintptr_t>(address)),
                   sizeof(selection.raw_selector));
            selection.readable = true;
        }
    }
    if (!selection.readable) return selection;

    const uint32_t adjusted = selection.raw_selector +
                              static_cast<uint32_t>(selection.dispatch.selector_addend);
    const uint32_t index = std::min(adjusted, selection.dispatch.selector_max);
    if (index < selection.dispatch.target_pcs.size())
        selection.target = selection.dispatch.target_pcs[index];
    return selection;
}

ShaderCompileKey make_shader_compile_key(ShaderProgramStage stage, const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* resources,
                                         const PixelInputMapping* pixel_inputs,
                                         const PixelSystemInputMapping* system_inputs,
                                         const uint32_t* chain_code = nullptr,
                                         size_t chain_dwords = 0,
                                         uint32_t vertex_lds_dwords = 0,
                                         const ComputeShaderConfig* compute_config = nullptr,
                                         bool fragment_wave32 = false,
                                         bool capture_position = false) {
    ShaderCompileKey key;
    key.stage = stage;
    key.vertex_lds_dwords = stage == ShaderProgramStage::Vertex
        ? std::min(vertex_lds_dwords, 16384u) : 0u;
    key.vertices_per_instance = stage == ShaderProgramStage::Vertex && resources
        ? resources->vertices_per_instance : 0u;
    key.has_resource_table = resources != nullptr;
    // #2395: also once, not per key. make_shader_compile_key is 4.52% of total CPU in Blue
    // Prince gameplay and runs per draw per stage.
    static const bool force_position_w = getenv("PROSPER_FORCE_W") != nullptr;
    key.force_position_w = force_position_w;
    key.capture_position = stage == ShaderProgramStage::Vertex && capture_position;
    key.has_pixel_inputs = stage != ShaderProgramStage::Compute && pixel_inputs != nullptr;
    if (key.has_pixel_inputs) key.pixel_inputs = *pixel_inputs;
    key.has_system_inputs = stage == ShaderProgramStage::Fragment && system_inputs != nullptr;
    if (key.has_system_inputs) key.system_inputs = *system_inputs;
    key.fragment_wave32 = stage == ShaderProgramStage::Fragment && fragment_wave32;
    key.has_compute_config = stage == ShaderProgramStage::Compute && compute_config;
    if (key.has_compute_config) {
        key.compute_user_sgpr_count = static_cast<uint32_t>(compute_config->user_sgprs.size());
        key.compute_local_x = compute_config->local_x;
        key.compute_local_y = compute_config->local_y;
        key.compute_local_z = compute_config->local_z;
        key.compute_exact_thread_extent = compute_config->exact_thread_extent;
        key.compute_threads_x = compute_config->threads_x;
        key.compute_threads_y = compute_config->threads_y;
        key.compute_threads_z = compute_config->threads_z;
        key.compute_wave_size = compute_config->wave_size;
        key.compute_pgm_rsrc1 = compute_config->compute_pgm_rsrc1;
        key.compute_tidig_comp_cnt = compute_config->tidig_comp_cnt;
        key.compute_tgid_x_en = compute_config->tgid_x_en;
        key.compute_tgid_y_en = compute_config->tgid_y_en;
        key.compute_tgid_z_en = compute_config->tgid_z_en;
        key.compute_tg_size_en = compute_config->tg_size_en;
        key.compute_lds_bytes = compute_config->lds_bytes;
        key.compute_native_subgroup_size = compute_config->native_subgroup_size;
        key.compute_native_storage_format_support =
            compute_config->native_storage_format_support;
        key.compute_storage_buffer_int64_atomics =
            compute_config->storage_buffer_int64_atomics;
        key.compute_packed_r11_storage = compute_config->packed_r11_storage;
    }
    const std::shared_ptr<const ShaderCodeAnalysis> analysis =
        code && dwords ? analyze_shader_code_cached(code, dwords) : nullptr;
    if (stage == ShaderProgramStage::Fragment && code && dwords && resources) {
        const PcrelDispatchSelection selection =
            select_pcrel_dispatch(code, dwords, resources, analysis.get());
        const PcrelDispatchInfo& dispatch = selection.dispatch;
        if (dispatch.valid) {
            if (selection.target != UINT32_MAX) {
                key.has_pcrel_dispatch = true;
                key.pcrel_dispatch_target = selection.target;
            }
            if (getenv("PROSPER_DBG")) {
                static std::mutex dispatch_log_mutex;
                static std::set<uintptr_t> dispatch_logged;
                std::lock_guard lock(dispatch_log_mutex);
                if (dispatch_logged.insert(reinterpret_cast<uintptr_t>(code)).second) {
                    fprintf(stderr,
                            "[pcrel-dispatch] code=%p selector=s%u+0x%x resource=%s readable=%u "
                            "raw=%u target=%u resources=%zu\n",
                            static_cast<const void*>(code), dispatch.selector_sgpr_base,
                            dispatch.selector_byte_offset,
                            selection.resource ? "found" : "missing", selection.readable,
                            selection.raw_selector, key.pcrel_dispatch_target,
                            resources->resources.size());
                    for (const auto& candidate : resources->resources)
                        fprintf(stderr,
                                "[pcrel-dispatch]   cls=%u binding=%u sgpr=%u srt=%u addr=%llx "
                                "size=%u host=%llu\n",
                                static_cast<unsigned>(candidate.cls), candidate.binding,
                                candidate.sgpr_base, candidate.srt_offset,
                                static_cast<unsigned long long>(candidate.gpu_addr), candidate.size,
                                static_cast<unsigned long long>(candidate.host_data_size));
                }
            }
        }
    }
    if (analysis) {
        // Most shaders end at S_ENDPGM. A compiler-generated s_getpc_b64 V# may instead address an
        // embedded lookup table after ENDPGM; retain that proven tail so cached recompilation sees the
        // same blob as the direct path and table contents participate in the cache identity.
        key.code = std::shared_ptr<const std::vector<uint32_t>>(analysis, &analysis->code);
        key.code_hash = analysis->code_hash;
    }
    if (stage == ShaderProgramStage::Vertex && chain_code && chain_dwords) {
        const std::shared_ptr<const ShaderCodeAnalysis> chain_analysis =
            analyze_shader_code_cached(chain_code, chain_dwords);
        if (chain_analysis) {
            key.chain_code = std::shared_ptr<const std::vector<uint32_t>>(
                chain_analysis, &chain_analysis->code);
            key.chain_code_hash = chain_analysis->code_hash;
        }
    }
    if (resources) {
        key.resources.reserve(resources->resources.size());
        for (const auto& resource : resources->resources) {
            const bool texture = resource.cls == ResourceClass::Texture;
            const bool storage_image = resource.cls == ResourceClass::StorageImage;
            const bool manual_compare = texture && resource.depth_compare;
            const bool normalize_unnormalized = texture && resource.unnormalized &&
                !std::getenv("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE");
            const bool atomic_extent = storage_image &&
                resource.format == DataFormat::Uint32 && resource.num_components == 1;
            ShaderResourceCompileKey compiled;
            compiled.cls = static_cast<uint32_t>(resource.cls);
            compiled.format = static_cast<uint32_t>(resource.format);
            compiled.num_components = resource.num_components;
            compiled.width = (atomic_extent || normalize_unnormalized) ? resource.width : 0u;
            compiled.height = (atomic_extent || normalize_unnormalized) ? resource.height : 0u;
            compiled.depth = (storage_image || normalize_unnormalized) ? resource.depth : 0u;
            compiled.img_dim = (texture || storage_image) ? resource.img_dim : 0u;
            compiled.sample_count = (texture || storage_image) ? resource.sample_count : 1u;
            compiled.declared_mip_levels = (texture || storage_image)
                ? resource.declared_mip_levels : 1u;
            compiled.materialized_mip_levels = texture
                ? shader_resource_compute_mip_chain_levels(resource) : 1u;
            compiled.in_mip_tail = (texture || storage_image) && resource.in_mip_tail;
            compiled.compression_enabled =
                (texture || storage_image) && resource.compression_enabled;
            compiled.proven_zero_mip = (texture || storage_image) && resource.proven_zero_mip;
            compiled.binding = resource.binding;
            compiled.stride = resource.stride;
            const bool one_record_tail = resource.num_components == 1u &&
                resource.stride == 2u && resource.size == 2u;
            compiled.one_record_tail_semantic = static_cast<uint32_t>(
                one_record_tail && resource.format == DataFormat::Uint16
                    ? StorageBufferTailSemantic::Uint16
                    : one_record_tail && resource.format == DataFormat::Float16
                        ? StorageBufferTailSemantic::Float16
                        : StorageBufferTailSemantic::None);
            // A warm cache hit bypasses emitter-side resource revalidation. Partition on the
            // complete resource-side admission result, not the serialized marker alone, so an
            // opaque/corrupt marker cannot borrow a valid module across size, alignment, or array
            // shape changes. Address identity remains intentionally absent; only alignment matters.
            const bool atomic_x2_exact_admission =
                resource.atomic_x2_record_count != 0u &&
                resource.atomic_x2_record_count <= 0x02000000u &&
                static_cast<uint64_t>(resource.atomic_x2_record_count) * 8u == resource.size &&
                resource.stride == 8u &&
                (resource.gpu_addr & 7u) == 0u && resource.table_index_count == 0u;
            compiled.atomic_x2_record_count =
                atomic_x2_exact_admission ? resource.atomic_x2_record_count : 0u;
            compiled.scalar_buffer_dword_count = resource.scalar_buffer_dword_count;
            compiled.scalar_buffer_contract_valid =
                !resource.scalar_buffer_dword_count ||
                shader_resource_buffer_binding_bytes(resource) != 0u;
            compiled.srt_offset = resource.srt_offset;
            compiled.sgpr_base = resource.sgpr_base;
            compiled.fetch_pc = resource.fetch_pc;
            compiled.fetch_index_mode = static_cast<uint32_t>(resource.fetch_index_mode);
            compiled.table_index_count = resource.table_index_count;
            compiled.table_entry_stride = resource.table_entry_stride;
            compiled.table_index_sgpr = resource.table_index_sgpr;
            compiled.table_selector_mode =
                static_cast<uint32_t>(resource.table_selector_mode);
            compiled.table_load_pc = resource.table_load_pc;
            compiled.table_contract_valid = valid_shader_buffer_table_contract(resource);
            compiled.flat_base_sgpr = resource.flat_base_sgpr;
            compiled.bvh_box_grow = resource.bvh_box_grow;
            compiled.bvh_sort_enabled = resource.bvh_sort_enabled;
            compiled.null_bvh = is_proven_null_bvh(resource);
            // gpu_addr/size intentionally stay out of the module key, but this semantic is compiled
            // into zero-producing/no-op instructions. Partition exactly this proven marker so a
            // later nonzero descriptor at the same pc cannot reuse the specialized module.
            compiled.zero_record_raw = is_zero_record_raw_buffer(resource);
            compiled.optional_null_raw_load = is_optional_null_raw_load_buffer(resource);
            // This marker elides a real one-record store, unlike NUM_RECORDS=0. Keep the admission
            // result explicit even though its impossible V# stride also differs, so cache identity
            // documents and enforces the semantic boundary directly.
            compiled.proven_null_guarded_raw_store =
                is_proven_null_guarded_raw_store(resource);
            compiled.proven_null_nullable_raw_buffer =
                is_proven_null_nullable_raw_buffer(resource);
            compiled.gta5_cf9200_no_backing =
                is_proven_gta5_cf9200_no_backing(resource);
            if (is_gta5_selected_sbuffer_descriptor(resource)) {
                compiled.selected_sbuffer_soffset = resource.selected_sbuffer_soffset;
                compiled.selected_sbuffer_words = resource.selected_sbuffer_words;
            }
            if (is_gta5_packed_pointer_resource(resource)) {
                compiled.indirect_buffer_contract_tag =
                    resource.indirect_buffer_contract_tag;
                compiled.indirect_buffer_binding_bytes =
                    resource.indirect_buffer_binding_bytes;
                compiled.indirect_buffer_slot_count = resource.indirect_buffer_slot_count;
                compiled.indirect_buffer_header_bytes = resource.indirect_buffer_header_bytes;
                compiled.indirect_buffer_slot_bytes = resource.indirect_buffer_slot_bytes;
            }
            if (is_indirect_pointer_relocation_resource(resource)) {
                const auto& relocation = resource.indirect_pointer_relocation;
                compiled.indirect_pointer_carrier_version = relocation.carrier_version;
                compiled.indirect_pointer_proof_schema = relocation.proof_schema;
                compiled.indirect_pointer_binding_bytes = relocation.binding_bytes;
                compiled.indirect_pointer_record_count = relocation.record_count;
                compiled.indirect_pointer_segment_count = relocation.segment_count;
                compiled.indirect_pointer_segment_directory_byte_offset =
                    relocation.segment_directory_byte_offset;
                compiled.indirect_pointer_proof_fingerprint =
                    relocation.proof_fingerprint;
            }
            compiled.srgb = storage_image && resource.srgb;
            compiled.depth_compare = (manual_compare || storage_image) && resource.depth_compare;
            compiled.depth_compare_func = manual_compare ? resource.depth_compare_func : 0u;
            compiled.mag_filter = manual_compare ? resource.mag_filter : 0u;
            compiled.addr_u = manual_compare ? resource.addr_uvw[0] : 0u;
            compiled.addr_v = manual_compare ? resource.addr_uvw[1] : 0u;
            compiled.border_color_type = manual_compare ? resource.border_color_type : 0u;
            compiled.normalize_unnormalized_coordinates = normalize_unnormalized;
            key.resources.push_back(compiled);
        }
    }
    // All three callers still attach their program's trip-bound settings. Hash only after that
    // finalization: a provisional hash here is unconditionally overwritten before any cache lookup.
    return key;
}

// `program_address` is the guest address of the shader this key was built from. It is DIAGNOSTIC
// PROVENANCE ONLY and deliberately arrives as a separate argument rather than inside the key: the
// same program bytes produce the same SPIR-V at every guest address, so folding it into
// ShaderCompileKey would give each address its own cache entry and quietly defeat the cache. The
// key's own `code` pointer cannot serve here -- it points into the key's owned COPY of the words,
// not at guest memory (#3130).
std::vector<uint32_t> compile_graphics_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                              const ShaderResourceTable* resources,
                                              uint64_t program_address) {
    const uint32_t* code = !key.code || key.code->empty() ? nullptr : key.code->data();
    const size_t code_size = key.code ? key.code->size() : 0u;
    if (stage == ShaderProgramStage::Vertex)
        // Geometry probe: decorate gl_Position only when the caller proved the VS is the last
        // pre-rasterization stage. Generated interpolation geometry stages own XFB themselves.
        // Same provenance the fragment stage received in #3130: without the guest address the
        // vertex stage's terminal reject is never RECORDED (record_terminal_reject_reason()
        // early-returns on zero), so every vertex skip printed `reason=unrecorded`.
        return key.chain_code
            ? recompile_vertex_chain(code, code_size, key.chain_code->data(),
                                     key.chain_code->size(), resources,
                                     key.has_pixel_inputs ? &key.pixel_inputs : nullptr,
                                     key.capture_position,
                                     key.vertex_lds_dwords,
                                     {RecompileDiagnosticStage::Vertex, program_address})
            : recompile_vertex(code, code_size, resources,
                               key.has_pixel_inputs ? &key.pixel_inputs : nullptr,
                               key.capture_position,
                               key.vertex_lds_dwords,
                               {RecompileDiagnosticStage::Vertex, program_address});
    if (stage == ShaderProgramStage::Fragment) {
        const FragmentInterpolationLayout interpolation = fragment_interpolation_layout(
            code, code_size,
            key.has_system_inputs ? &key.system_inputs : nullptr,
            key.has_pixel_inputs ? &key.pixel_inputs : nullptr);
        return recompile_fragment(code, code_size, resources,
                                  key.has_system_inputs ? &key.system_inputs : nullptr,
                                  key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX,
                                  &interpolation, key.fragment_wave32,
                                  {RecompileDiagnosticStage::Fragment, program_address});
    }
    return {};
}

// `program_address` and `chain_address` are the guest addresses the dumped words came from, and
// carrying them into the FILENAME is the point of #3196. A content hash names WHAT was compiled and
// never WHERE the guest put it, so an investigation holding an address -- which is how
// `PROSPER_SKIP_DRAW_PROGRAM`, `PROSPER_COMPUTE_SKIP_PROGRAM`, `PROSPER_DRAW_PROGRAM_CENSUS` and the
// `[buf-op]` / `[mubuf-unresolved]` lines all name programs -- could not ask this dump for its
// program, and had to hash-match the directory by hand. `chain_address` is the NGG main
// continuation of a vertex chain, 0 when there is none. Either may be 0 when the caller genuinely
// has no address; that is written as `0000000000000000` rather than omitted, so a reader can tell
// "no address was available" from "the address is unusual".
void maybe_dump_successful_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                  const std::vector<uint32_t>& spirv, uint64_t program_address,
                                  uint64_t chain_address) {
    // Detect-its-own-invalidity, against #2149's TWO-GATE shape. `PROSPER_SHADER_DUMP_PROGRAM` is a
    // modifier of `PROSPER_SHADER_DUMP_SUCCESS`, not a switch of its own, so arming only the one
    // whose NAME matches the question -- "dump the program at 0x..." -- produces an empty directory
    // and not one line, which reads as "that program never compiled". Said once, from the first
    // shader that reaches here; the function-local static keeps the steady-state cost at one guard
    // load, because this runs per draw realization on a cache hit.
    static const bool announced_filter_without_directory = [] {
        if (!getenv("PROSPER_SHADER_DUMP_PROGRAM") || getenv("PROSPER_SHADER_DUMP_SUCCESS"))
            return false;
        fprintf(stderr,
                "[shader-dump] PROSPER_SHADER_DUMP_PROGRAM is set but PROSPER_SHADER_DUMP_SUCCESS=DIR "
                "is NOT -- nothing will be dumped. The filter narrows a dump; it does not start one\n");
        return true;
    }();
    (void)announced_filter_without_directory;

    const char* directory = getenv("PROSPER_SHADER_DUMP_SUCCESS");
    if (!directory || !*directory || !key.code || key.code->empty() || spirv.empty()) return;

    // PROSPER_SHADER_DUMP_PROGRAM: the other half of #3196. Default OFF, in which case `allows` is
    // true for everything and this costs one env read per dumped shader -- and that read only
    // happens once the dump directory is set, so a default run never reaches here at all.
    // Ordered BEFORE the dedup set on purpose: a withheld program must not be recorded as dumped,
    // or re-arming the filter (which a test does, and a long-running process could) would then
    // silently write nothing. The consequence is that the ordinal below counts dump OPPORTUNITIES,
    // not distinct programs -- a cache hit on a withheld program counts again. Read it as a lower
    // bound on volume avoided, never as a program count.
    auto& dump_filter = shader_dump_program_filter();
    if (!dump_filter.allows(program_address, chain_address)) {
        const auto withheld = dump_filter.note_withheld();
        if (withheld.print)
            fprintf(stderr,
                    "[shader-dump] withheld addr=0x%llx chain=0x%llx (not named by "
                    "PROSPER_SHADER_DUMP_PROGRAM) opportunities=%llu\n",
                    static_cast<unsigned long long>(program_address),
                    static_cast<unsigned long long>(chain_address),
                    static_cast<unsigned long long>(withheld.ordinal));
        return;
    }

    const uint64_t spirv_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(spirv.data()), spirv.size() * sizeof(uint32_t));
    const uint64_t raw_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(key.code->data()), key.code->size() * sizeof(uint32_t));
    const uint64_t chain_hash = key.chain_code && !key.chain_code->empty()
        ? gpu_capture_hash(reinterpret_cast<const uint8_t*>(key.chain_code->data()),
                           key.chain_code->size() * sizeof(uint32_t))
        : 0;
    static std::mutex dump_mutex;
    // The address joins the dedup key now that it is part of the filename. Without it the FIRST
    // address to compile a given body would claim the pair and every later address sharing those
    // bytes would silently write nothing -- which is exactly the false negative this whole change
    // exists to remove, just moved one step downstream.
    static std::set<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> dumped;
    std::lock_guard lock(dump_mutex);
    if (!dumped.emplace(static_cast<uint32_t>(stage), program_address, chain_address, spirv_hash,
                        raw_hash, chain_hash).second)
        return;

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        fprintf(stderr, "[shader-dump] cannot create %s: %s\n", directory, ec.message().c_str());
        return;
    }
    const char* tag = stage == ShaderProgramStage::Vertex ? "vs" :
                      stage == ShaderProgramStage::Fragment ? "ps" : "cs";
    char raw_path[1024], chain_path[1024], spirv_path[1024];
    // `success_<stage>_at_<addr>_<spirv-hash>_<raw-hash>` -- the address leads the hash fields so
    // that recovering a known program is one glob (`success_vs_at_00000005008efd00_*`) instead of a
    // hash match. The hashes stay: they are what deduplicates, and one program compiled against
    // different resource tables legitimately yields several variants under the same address.
    snprintf(raw_path, sizeof(raw_path), "%s/success_%s_at_%016llx_%016llx_%016llx.bin", directory,
             tag, static_cast<unsigned long long>(program_address),
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    snprintf(chain_path, sizeof(chain_path),
             "%s/success_%s_at_%016llx_%016llx_%016llx_main_%016llx_%016llx.bin",
             directory, tag, static_cast<unsigned long long>(program_address),
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash),
             static_cast<unsigned long long>(chain_address),
             static_cast<unsigned long long>(chain_hash));
    snprintf(spirv_path, sizeof(spirv_path), "%s/success_%s_at_%016llx_%016llx_%016llx.spv",
             directory, tag, static_cast<unsigned long long>(program_address),
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    FILE* raw = fopen(raw_path, "wb");
    FILE* chain = chain_hash ? fopen(chain_path, "wb") : nullptr;
    FILE* translated = fopen(spirv_path, "wb");
    const size_t raw_bytes = key.code->size() * sizeof(uint32_t);
    const size_t chain_bytes = chain_hash ? key.chain_code->size() * sizeof(uint32_t) : 0;
    const size_t spirv_bytes = spirv.size() * sizeof(uint32_t);
    const bool ok = raw && translated &&
        fwrite(key.code->data(), 1, raw_bytes, raw) == raw_bytes &&
        (!chain_hash || (chain && fwrite(key.chain_code->data(), 1, chain_bytes, chain) ==
                                     chain_bytes)) &&
        fwrite(spirv.data(), 1, spirv_bytes, translated) == spirv_bytes;
    if (raw) fclose(raw);
    if (chain) fclose(chain);
    if (translated) fclose(translated);
    fprintf(stderr,
            "[shader-dump] %s addr=0x%llx chain=0x%llx spv=%016llx raw=%016llx main=%016llx "
            "words=%zu+%zu/%zu result=%s\n", tag,
            static_cast<unsigned long long>(program_address),
            static_cast<unsigned long long>(chain_address),
            static_cast<unsigned long long>(spirv_hash),
            static_cast<unsigned long long>(raw_hash),
            static_cast<unsigned long long>(chain_hash), key.code->size(),
            key.chain_code ? key.chain_code->size() : 0u, spirv.size(),
            ok ? "written" : "failed");
}

SharedShaderWords cache_compiled_graphics_shader(ShaderProgramStage stage, ShaderCompileKey key,
                                                  const ShaderResourceTable* resources,
                                                  uint64_t* cache_identity,
                                                  uint64_t program_address,
                                                  uint64_t chain_address) {
    if (cache_identity) *cache_identity = 0;
    if (getenv("PROSPER_NO_SHADER_CACHE")) {
        auto& cache = shader_cache();
        cache.bypasses.fetch_add(1, std::memory_order_relaxed);
        auto spirv = std::make_shared<const std::vector<uint32_t>>(
            compile_graphics_shader(stage, key, resources, program_address));
        maybe_dump_successful_shader(stage, key, *spirv, program_address, chain_address);
        return spirv;
    }

    auto& cache = shader_cache();
    {
        std::shared_lock lock(cache.mutex);
        auto found = cache.entries.find(key);
        if (found != cache.entries.end()) {
            cache.hits.fetch_add(1, std::memory_order_relaxed);
            found->second.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                                         std::memory_order_relaxed);
            if (cache_identity) *cache_identity = found->second.identity;
            maybe_dump_successful_shader(stage, key, *found->second.spirv, program_address,
                                        chain_address);
            return found->second.spirv;
        }
    }

    std::unique_lock lock(cache.mutex);
    auto double_check = cache.entries.find(key);
    if (double_check != cache.entries.end()) {
        cache.hits.fetch_add(1, std::memory_order_relaxed);
        double_check->second.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                                            std::memory_order_relaxed);
        if (cache_identity) *cache_identity = double_check->second.identity;
        maybe_dump_successful_shader(stage, key, *double_check->second.spirv, program_address,
                                    chain_address);
        return double_check->second.spirv;
    }

    const auto start = std::chrono::steady_clock::now();
    auto spirv = std::make_shared<const std::vector<uint32_t>>(
        compile_graphics_shader(stage, key, resources, program_address));
    maybe_dump_successful_shader(stage, key, *spirv, program_address, chain_address);
    const auto end = std::chrono::steady_clock::now();
    ++cache.stats.misses;
    cache.stats.compile_ms += std::chrono::duration<double, std::milli>(end - start).count();

    constexpr size_t max_entries = 4096;
    const uint64_t bytes = shader_cache_entry_bytes(key, *spirv);
    const uint64_t limit = shader_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use.load(std::memory_order_relaxed) <
                oldest->second.last_use.load(std::memory_order_relaxed))
                oldest = it;
        cache.stats.bytes -= oldest->second.bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (bytes <= limit && max_entries != 0) {
        CachedShader value;
        value.spirv = spirv;
        value.identity = cache.next_identity++;
        value.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                             std::memory_order_relaxed);
        value.bytes = bytes;
        if (cache_identity) *cache_identity = value.identity;
        cache.stats.bytes += bytes;
        cache.entries.emplace(std::move(key), std::move(value));
    }
    cache.stats.entries = cache.entries.size();
    return spirv;
}

} // namespace

ShaderDecodeCacheStats shader_decode_cache_stats() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    ShaderDecodeCacheStats stats = cache.stats;
    stats.entries = cache.entries.size();
    return stats;
}

void clear_shader_decode_cache() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
    cache.stats = {};
    cache.use_counter = 0;
}

ShaderAnalysisCacheStats shader_analysis_cache_stats() {
    auto& cache = shader_analysis_cache();
    std::lock_guard lock(cache.mutex);
    ShaderAnalysisCacheStats stats = cache.stats;
    stats.entries = cache.entries.size();
    return stats;
}

// Keyed on the shader-analysis identity, so it is cleared with that cache rather than outliving it.
// Internal linkage: it is a private detail of this translation unit, like the caches around it.
namespace {
struct ConsumedAttributeMaskCache {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint32_t> masks;
};
ConsumedAttributeMaskCache& consumed_attribute_mask_cache() {
    static ConsumedAttributeMaskCache cache;
    return cache;
}
}  // namespace

void clear_shader_analysis_cache() {
    {
        auto& cache = shader_analysis_cache();
        std::lock_guard lock(cache.mutex);
        cache.entries.clear();
        cache.stats = {};
        cache.use_counter = 0;
    }
    {
        // #2945: the consumed-attribute memo is keyed on an analysis identity, so it must die with
        // the analysis cache. Leaving it would answer from an identity nothing can produce again.
        auto& cache = consumed_attribute_mask_cache();
        std::lock_guard lock(cache.mutex);
        cache.masks.clear();
    }
    {
        auto& cache = interpolation_cache();
        std::lock_guard lock(cache.mutex);
        cache.entries.clear();
        cache.use_counter = 0;
    }
}

FragmentInterpolationLayout fragment_interpolation_layout_cached(
        const uint32_t* code, size_t dwords,
        const PixelSystemInputMapping* system_inputs,
        const PixelInputMapping* pixel_inputs) {
    const auto analysis = analyze_shader_code_cached(code, dwords);
    if (!analysis || getenv("PROSPER_NO_SHADER_ANALYSIS_CACHE"))
        return fragment_interpolation_layout(code, dwords, system_inputs, pixel_inputs);

    InterpolationCacheKey key;
    // Use the immutable analysis version, without retaining its shader-byte allocation beyond the
    // analysis cache's own memory bound. A same-address shader mutation receives a new identity.
    key.analysis_identity = analysis->identity;
    key.has_system_inputs = system_inputs != nullptr;
    if (system_inputs) key.system_inputs = *system_inputs;
    key.passthrough_mask = pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    key.flat_mask = pixel_inputs ? pixel_inputs->effective_flat_mask() : 0u;   // #3051
    auto& cache = interpolation_cache();
    std::lock_guard lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        found->second.last_use = ++cache.use_counter;
        return found->second.layout;
    }
    const FragmentInterpolationLayout layout =
        fragment_interpolation_layout(code, dwords, system_inputs, pixel_inputs);
    constexpr size_t max_entries = 4096;
    while (cache.entries.size() >= max_entries && !cache.entries.empty()) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.entries.erase(oldest);
    }
    cache.entries.emplace(std::move(key), CachedInterpolationLayout{layout, ++cache.use_counter});
    return layout;
}

uint32_t fragment_consumed_attribute_mask_cached(const uint32_t* code, size_t dwords) {
    static const bool no_cache = getenv("PROSPER_NO_SHADER_ANALYSIS_CACHE") != nullptr;
    const auto analysis = no_cache ? nullptr : analyze_shader_code_cached(code, dwords);
    if (!analysis) return fragment_consumed_attribute_mask(code, dwords);
    // Keyed on the immutable analysis identity, not on the address: a same-address shader mutation
    // receives a new identity, which is the property the interpolation cache next door relies on.
    auto& state = consumed_attribute_mask_cache();
    auto& mutex = state.mutex;
    auto& masks = state.masks;
    {
        std::lock_guard lock(mutex);
        const auto found = masks.find(analysis->identity);
        if (found != masks.end()) return found->second;
    }
    const uint32_t mask = fragment_consumed_attribute_mask(code, dwords);
    std::lock_guard lock(mutex);
    // Cleared wholesale rather than aged; the entries are four bytes each and the bound exists only
    // so a pathological run cannot grow it without limit.
    constexpr size_t max_entries = 4096;
    if (masks.size() >= max_entries) masks.clear();
    masks.emplace(analysis->identity, mask);
    return mask;
}

SharedShaderWords recompile_graphics_shader_cached_shared(
        ShaderProgramStage stage, const uint32_t* code, size_t dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        const PixelSystemInputMapping* system_inputs, uint64_t* cache_identity,
        bool fragment_wave32, uint32_t vertex_lds_dwords,
        bool vertex_capture_position) {
    ShaderCompileKey key = make_shader_compile_key(stage, code, dwords, resources, pixel_inputs,
                                                   system_inputs, nullptr, 0,
                                                   vertex_lds_dwords, nullptr,
                                                   fragment_wave32, vertex_capture_position);
    // Guest memory is 1:1-mapped, so the caller's code pointer IS the guest program address; it must
    // be captured here because the key owns a copy of the words rather than pointing at them.
    const uint64_t program_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(code));
    // The trip-bound selector has to be in the key, for exactly the reason the compute path already
    // mixes it in (`rdna2_cfg_support.hpp`: the cache is keyed on CODE BYTES and never on the
    // address, so a target and a non-target with the same body collide on one entry, and whichever
    // compiled first serves the other). Graphics never did this, and it was harmless only because
    // fragment diagnostics were pinned at address 0 — the selector could not match a program that
    // had no address. This change gives them their real address, which makes the hazard real for
    // graphics exactly as it already is for compute, so the guard has to arrive with it.
    key.trip_bound = compute_trip_bound_settings();
    if (key.trip_bound.bound) key.trip_bound_program_address = program_address;
    key.cached_hash = ShaderCompileKeyHash::compute(key);
    return cache_compiled_graphics_shader(stage, std::move(key), resources, cache_identity,
                                          program_address, /*chain_address=*/0);
}

SharedShaderWords recompile_vertex_chain_cached_shared(
        const uint32_t* prolog, size_t prolog_dwords,
        const uint32_t* main, size_t main_dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        uint64_t* cache_identity, uint32_t vertex_lds_dwords,
        bool capture_position) {
    ShaderCompileKey key = make_shader_compile_key(
        ShaderProgramStage::Vertex, prolog, prolog_dwords, resources, pixel_inputs, nullptr,
        main, main_dwords, vertex_lds_dwords, nullptr, false, capture_position);
    if (!key.chain_code) {
        if (cache_identity) *cache_identity = 0;
        return {};
    }
    // The prolog address is the program's identity everywhere else -- `DrawItem::vs_guest_addr` is
    // `rs.es_addr` for a chained vertex program, and that is the address a census or a skip selector
    // reports. The main continuation gets its own address so the dumped `_main_` file can be found
    // by either half, matching what gpu_capture already does when it resolves a wanted code address.
    const uint64_t prolog_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(prolog));
    const uint64_t main_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(main));
    key.trip_bound = compute_trip_bound_settings();
    if (key.trip_bound.bound) key.trip_bound_program_address = prolog_address;
    key.cached_hash = ShaderCompileKeyHash::compute(key);
    return cache_compiled_graphics_shader(
        ShaderProgramStage::Vertex, std::move(key), resources, cache_identity, prolog_address,
        main_address);
}

std::vector<uint32_t> recompile_graphics_shader_cached(
        ShaderProgramStage stage, const uint32_t* code, size_t dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        const PixelSystemInputMapping* system_inputs, uint64_t* cache_identity,
        bool fragment_wave32, uint32_t vertex_lds_dwords,
        bool vertex_capture_position) {
    SharedShaderWords words = recompile_graphics_shader_cached_shared(
        stage, code, dwords, resources, pixel_inputs, system_inputs, cache_identity,
        fragment_wave32, vertex_lds_dwords, vertex_capture_position);
    return words ? *words : std::vector<uint32_t>{};
}

std::vector<uint32_t> recompile_compute_shader_cached(
        const uint32_t* code, size_t dwords, const ShaderResourceTable* resources,
        const ComputeShaderConfig& config, uint64_t* cache_identity,
        RecompileDiagnosticContext diagnostic) {
    if (cache_identity) *cache_identity = 0;
    const bool has_null_guarded_raw_store = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_proven_null_guarded_raw_store);
    const bool has_nullable_output_raw_buffer = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_nullable_raw_buffer_marker_candidate);
    const bool has_selected_sbuffer_descriptor = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_gta5_selected_sbuffer_marker_candidate);
    const bool has_gta5_packed_pointer = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_gta5_packed_pointer_marker_candidate);
    const bool has_indirect_pointer_relocation = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_indirect_pointer_relocation_marker_candidate);
    const bool has_gta5_cf9200_no_backing = resources &&
        std::any_of(resources->resources.begin(), resources->resources.end(),
                    is_gta5_cf9200_no_backing_marker_candidate);
    // Push-constant values intentionally stay out of the general compute cache key. Conditional
    // store elision is the exception: validate its raw shader and exact dispatch before even looking
    // in the cache, so a warm null-dispatch module cannot be returned for changed s2:s3.
    if (has_null_guarded_raw_store &&
        !rdna2_gta5_null_guarded_raw_store_dispatch(
            code, dwords, config.user_sgprs.data(), config.user_sgprs.size()))
        return {};
    if (has_nullable_output_raw_buffer &&
        !rdna2_gta5_nullable_output_dispatch(code, dwords, config, *resources))
        return {};
    if (has_selected_sbuffer_descriptor &&
        !rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, *resources))
        return {};
    if (has_gta5_packed_pointer &&
        !rdna2_gta5_packed_pointer_dispatch(code, dwords, config, *resources))
        return {};
    if (has_indirect_pointer_relocation &&
        !validate_rdna2_indirect_pointer_relocations(
            code, dwords, config, *resources))
        return {};
    if (has_gta5_cf9200_no_backing &&
        !rdna2_gta5_cf9200_no_backing_dispatch(code, dwords, config, *resources))
        return {};
    ShaderCompileKey key = make_shader_compile_key(
        ShaderProgramStage::Compute, code, dwords, resources, nullptr, nullptr,
        nullptr, 0, 0, &config);
    // Only meaningful while the trip-bound diagnostic is armed; disarmed it leaves the key exactly as
    // it was. The program address is carried on the diagnostic context because nothing else in the
    // key identifies WHICH program these code bytes belong to, and that is precisely the distinction
    // the selector makes. Finalize the hash only after attaching those diagnostic settings.
    key.trip_bound = compute_trip_bound_settings();
    if (key.trip_bound.bound) key.trip_bound_program_address = diagnostic.program_address;
    key.cached_hash = ShaderCompileKeyHash::compute(key);
    auto compile = [&] {
        const uint32_t* owned_code = !key.code || key.code->empty() ? nullptr : key.code->data();
        const size_t owned_dwords = key.code ? key.code->size() : 0u;
        return recompile_compute(owned_code, owned_dwords, resources, config, diagnostic);
    };
    if (getenv("PROSPER_NO_SHADER_CACHE")) {
        auto& cache = shader_cache();
        cache.bypasses.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint32_t> spirv = compile();
        maybe_dump_successful_shader(ShaderProgramStage::Compute, key, spirv,
                                     diagnostic.program_address, 0);
        return spirv;
    }

    auto& cache = shader_cache();
    {
        std::shared_lock lock(cache.mutex);
        auto found = cache.entries.find(key);
        if (found != cache.entries.end()) {
            cache.hits.fetch_add(1, std::memory_order_relaxed);
            found->second.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                                         std::memory_order_relaxed);
            if (cache_identity) *cache_identity = found->second.identity;
            maybe_dump_successful_shader(ShaderProgramStage::Compute, key, *found->second.spirv,
                                        diagnostic.program_address, 0);
            return *found->second.spirv;
        }
    }

    std::unique_lock lock(cache.mutex);
    auto double_check = cache.entries.find(key);
    if (double_check != cache.entries.end()) {
        cache.hits.fetch_add(1, std::memory_order_relaxed);
        double_check->second.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                                            std::memory_order_relaxed);
        if (cache_identity) *cache_identity = double_check->second.identity;
        maybe_dump_successful_shader(ShaderProgramStage::Compute, key, *double_check->second.spirv,
                                    diagnostic.program_address, 0);
        return *double_check->second.spirv;
    }

    const auto start = std::chrono::steady_clock::now();
    auto spirv = std::make_shared<const std::vector<uint32_t>>(compile());
    maybe_dump_successful_shader(ShaderProgramStage::Compute, key, *spirv,
                                 diagnostic.program_address, 0);
    const auto end = std::chrono::steady_clock::now();
    ++cache.stats.misses;
    cache.stats.compile_ms += std::chrono::duration<double, std::milli>(end - start).count();

    constexpr size_t max_entries = 4096;
    const uint64_t bytes = shader_cache_entry_bytes(key, *spirv);
    const uint64_t limit = shader_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use.load(std::memory_order_relaxed) <
                oldest->second.last_use.load(std::memory_order_relaxed))
                oldest = it;
        cache.stats.bytes -= oldest->second.bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (bytes <= limit && max_entries != 0) {
        CachedShader value;
        value.spirv = spirv;
        value.identity = cache.next_identity++;
        value.last_use.store(cache.use_counter.fetch_add(1, std::memory_order_relaxed),
                             std::memory_order_relaxed);
        value.bytes = bytes;
        if (cache_identity) *cache_identity = value.identity;
        cache.stats.bytes += bytes;
        cache.entries.emplace(std::move(key), std::move(value));
    }
    cache.stats.entries = cache.entries.size();
    return *spirv;
}

// PROSPER_COMPUTE_DISPATCH_LOG=0xADDR[,0xADDR...]: one line per DISPATCH of the named programs,
// with what actually happened to it. Every existing signal is once-per-program:
// `[compute] skip unsupported program` fires once ever, and the subgroup, resource-map and
// selected-sbuffer lines are all deduped. That is right for their purposes and wrong for this one,
// and reading a once-per-program line as a per-dispatch property is exactly how a root-cause claim
// in this investigation was published and had to be retracted -- the program it said "never runs"
// ran 26 times.
//
// Answers "did THIS dispatch execute", which is what a frame-by-frame correlation between a decline
// and a corrupted result needs.
bool compute_dispatch_log_selected(uint64_t program) {
    const char* spec = std::getenv("PROSPER_COMPUTE_DISPATCH_LOG");
    if (!spec || !*spec || !program) return false;
    for (const char* cursor = spec; cursor && *cursor;) {
        char* end = nullptr;
        const uint64_t one = std::strtoull(cursor, &end, 0);
        if (end == cursor) return false;
        if (one == program) return true;
        if (*end != ',') return false;
        cursor = end + 1;
    }
    return false;
}

void log_compute_dispatch(uint64_t program, uint64_t submit_no, size_t dispatch_index,
                          uint64_t order, const char* outcome,
                          const ComputeLaunchDimensions* launch = nullptr) {
    if (!compute_dispatch_log_selected(program)) return;
    // The launch travels with the outcome. GTA V's tree builder emits a PERFECT tree for five
    // consecutive submits and then a cyclic one for the rest of the route, so the question is what
    // differs between those two regimes -- and the guest's active-record count reaches the shader as
    // the thread extent. A dispatch record without it cannot answer that.
    char geometry[128] = "";
    if (launch)
        std::snprintf(geometry, sizeof(geometry),
                      " groups=%ux%ux%u local=%ux%ux%u threads=%ux%ux%u",
                      launch->groups_x, launch->groups_y, launch->groups_z,
                      launch->local_x, launch->local_y, launch->local_z,
                      launch->threads_x, launch->threads_y, launch->threads_z);
    std::fprintf(stderr,
                 "[compute-dispatch] program=0x%llx submit=%llu dispatch=%zu order=%llu outcome=%s%s\n",
                 static_cast<unsigned long long>(program),
                 static_cast<unsigned long long>(submit_no), dispatch_index,
                 static_cast<unsigned long long>(order), outcome, geometry);
}

// PROSPER_COMPUTE_PROGRAM_CENSUS=1 — per program, how many dispatches EXECUTED and how many were
// skipped.
//
// `[compute] skip unsupported program` prints once per program address by design, so a run showing
// twelve skip lines has been read as "twelve programs are disabled". That does not follow: the skip
// is a `continue` inside the per-dispatch loop, so every dispatch is re-decided, and a program whose
// descriptors are garbage on its first fold and valid afterwards is skipped once and runs from then
// on. One line cannot distinguish 1-of-438 from 438-of-438, and those are opposite states of the
// title. Ratios, not first sightings.
inline void note_compute_program_outcome(uint64_t code_addr, bool executed,
                                        uint32_t gx = 0, uint32_t gy = 0, uint32_t gz = 0,
                                        uint32_t lx = 0, uint32_t ly = 0) {
    static const bool on = std::getenv("PROSPER_COMPUTE_PROGRAM_CENSUS") != nullptr;
    if (!on) return;
    static std::mutex mutex;
    struct Outcome { uint64_t executed = 0, skipped = 0; uint32_t gx = 0, gy = 0, gz = 0, lx = 0, ly = 0; };
    static std::map<uint64_t, Outcome> outcomes;
    static std::atomic<uint64_t> total{0};
    const uint64_t n = total.fetch_add(1) + 1;
    std::lock_guard lock(mutex);
    auto& entry = outcomes[code_addr];
    (executed ? entry.executed : entry.skipped) += 1;
    // The dispatch grid is what identifies a kernel's ROLE without resolving its descriptors, which
    // is the whole difficulty for a program that never executes. A full-screen 4K pass at 8x8 groups
    // is 480x270; anything much smaller is not writing the scene.
    if (gx) { entry.gx = gx; entry.gy = gy; entry.gz = gz; entry.lx = lx; entry.ly = ly; }
    // Emit at every power of two, with NO floor. The floor was 512, so any run making fewer
    // decisions than that printed nothing at all -- a null indistinguishable from "the census is
    // off", "the hook is not on this path" and "this build lacks the diagnostic". A small population
    // is a legitimate answer to "how much compute does this screen run", and the floor made it the
    // one answer the instrument could not give.
    //
    // Honest provenance, since the first version of this comment claimed a Stray run had hit the
    // floor: it had not. That run recorded 65,536 decisions and printed eight census blocks; the
    // author had grepped for a string that does not occur ("program-census") and then for one that
    // matched the run's own DIRECTORY NAME, and read the resulting silence as a measurement. The
    // floor did not cause that and removing it would not have prevented it. It is removed because
    // an unreadable null is a real hazard on its own terms, not because it was the culprit here.
    if ((n & (n - 1)) != 0) return;
    std::fprintf(stderr, "[compute-census] %llu dispatch decisions over %zu program(s)\n",
                 (unsigned long long)n, outcomes.size());
    for (const auto& [addr, counts] : outcomes)
        if (counts.skipped)   // only programs that skipped at least once are interesting
            std::fprintf(stderr,
                         "[compute-census]   program=0x%llx executed=%llu skipped=%llu "
                         "groups=%ux%ux%u local=%ux%u threads=%ux%u\n",
                         (unsigned long long)addr, (unsigned long long)counts.executed,
                         (unsigned long long)counts.skipped, counts.gx, counts.gy, counts.gz,
                         counts.lx, counts.ly, counts.gx * counts.lx, counts.gy * counts.ly);
}

// PROSPER_COMPUTE_BINDS=<hex addr>[,<hex addr>…] — name every compute program that binds one of these
// guest addresses, with the binding's class and size and what became of the dispatch.
//
// The question it answers is "who was supposed to write this surface". A render-target census sees
// draws, the write watches see DMA/WRITE_DATA/RELEASE_MEM, and between them a surface produced by a
// compute STORE is invisible: its address appears only inside a dispatch's resource table, which
// nothing enumerates. GTA V's 4K HDR scene colour 0x2063380000 is the case -- sampled 24x per
// composite, written by nothing any existing instrument can see.
//
// WHAT ITS NULL DOES AND DOES NOT COVER. The instrument reads the *resource table*, so it can only
// see bindings the resource build resolved. For a program whose recompile failed, that table is
// built from a walk that is itself incomplete -- PROSPER_DYNTRACE_FAIL measured 15 of 21 image ops
// recovered for 0x205b5e8600 and 4 of 6 for 0x205b657200 -- so a `partial-recompile-empty` row is a
// lower bound on that program's bindings, and the ABSENCE of a row for such a program is not
// evidence that it does not bind the address. Only `executed` and `skipped-descriptors` rows come
// from a fully resolved table. Read a null across a failing-shader population as "unproven", never
// as "cleared". CONFIDENCE: HIGH (the coverage gap is measured, not assumed).
// `ComputeBindOutcome`, `ComputeBindWatchKey`, `compute_bind_watch_key` and the selection/dedup seam
// `compute_bind_watch_rows` all live in gpu_execute.hpp, where the regression reaches them. This
// function is deliberately only environment parsing and formatting: every decision about WHICH rows
// exist -- range filtering, key construction, dedup -- is in the tested function, so a narrowing
// anywhere in that logic fails the regression instead of passing it.
inline void report_compute_binding_watch(uint64_t code_addr, const ShaderResourceTable* resources,
                                         ComputeBindOutcome outcome) {
    static const std::vector<uint64_t> watch = [] {
        std::vector<uint64_t> addrs;
        const char* spec = std::getenv("PROSPER_COMPUTE_BINDS");
        if (!spec || !*spec) return addrs;
        for (const char* cursor = spec; *cursor;) {
            char* end = nullptr;
            const uint64_t value = std::strtoull(cursor, &end, 16);
            if (end == cursor) break;
            addrs.push_back(value);
            cursor = (*end == ',') ? end + 1 : end;
        }
        return addrs;
    }();
    if (watch.empty() || !resources) return;
    static std::mutex mutex;
    static std::set<ComputeBindWatchKey> reported;
    std::vector<ComputeBindWatchKey> rows;
    {
        std::lock_guard lock(mutex);
        rows = compute_bind_watch_rows(reported, watch, code_addr, resources, outcome);
    }
    for (const ComputeBindWatchKey& row : rows)
        std::fprintf(stderr,
                     "[compute-binds] 0x%llx bound by program=0x%llx binding=%u class=%u "
                     "fetch_pc=%u addr=0x%llx size=%llu %ux%u fmt=%u outcome=%s\n",
                     (unsigned long long)row.watched, (unsigned long long)row.program, row.binding,
                     row.cls, row.fetch_pc, (unsigned long long)row.resource_addr,
                     (unsigned long long)row.size, row.width, row.height, row.format,
                     compute_bind_outcome_name(static_cast<ComputeBindOutcome>(row.outcome)));
}

bool report_compute_recompile_skip_once(RecompileDiagnosticContext diagnostic) {
    static std::mutex mutex;
    static std::set<uint64_t> logged;
    {
        std::lock_guard lock(mutex);
        if (!logged.insert(diagnostic.program_address).second) return false;
    }
    if (std::getenv("PROSPER_DBG")) {
        log_compute_recompile_skip_diagnostic(diagnostic);
        return true;
    }
    // Name the reason, not only the address. Every skip is meant to be the next thing implemented,
    // and until now the only way to learn why one happened was PROSPER_DBG -- which on a routed run
    // desyncs the pad script badly enough that the route never reaches the phase being diagnosed.
    const std::string reason = last_terminal_reject_reason(diagnostic.program_address);
    std::fprintf(stderr, "[compute] skip unsupported program 0x%llx reason=%s\n",
                 static_cast<unsigned long long>(diagnostic.program_address),
                 reason.empty() ? "unrecorded" : reason.c_str());
    return true;
}

ShaderRecompileCacheStats shader_recompile_cache_stats() {
    auto& cache = shader_cache();
    std::shared_lock lock(cache.mutex);
    ShaderRecompileCacheStats result = cache.stats;
    result.hits = cache.hits.load(std::memory_order_relaxed);
    result.bypasses = cache.bypasses.load(std::memory_order_relaxed);
    result.entries = cache.entries.size();
    return result;
}

void clear_shader_recompile_cache() {
    auto& cache = shader_cache();
    std::unique_lock lock(cache.mutex);
    cache.entries.clear();
    cache.stats = {};
    cache.hits.store(0, std::memory_order_relaxed);
    cache.bypasses.store(0, std::memory_order_relaxed);
    cache.use_counter.store(0, std::memory_order_relaxed);
}

thread_local DrawRealizationPhaseStats g_draw_realization_phases;

void record_draw_realization_phases(double table_ms, double shader_ms) {
    ++g_draw_realization_phases.draws;
    g_draw_realization_phases.table_ms += table_ms;
    g_draw_realization_phases.shader_ms += shader_ms;
}

DrawRealizationPhaseStats draw_realization_phase_stats() {
    return g_draw_realization_phases;
}

thread_local StageTablePhaseStats g_stage_table_phases;

void record_stage_table_phases(double metadata_ms, double dynamic_fold_ms, double resources_ms) {
    ++g_stage_table_phases.calls;
    g_stage_table_phases.metadata_ms += metadata_ms;
    g_stage_table_phases.dynamic_fold_ms += dynamic_fold_ms;
    g_stage_table_phases.resources_ms += resources_ms;
}

StageTablePhaseStats stage_table_phase_stats() {
    return g_stage_table_phases;
}

namespace {

struct GuestReadableCacheState {
    bool enabled = getenv("PROSPER_NO_GUEST_READ_CACHE") == nullptr;
    bool active = false;
    host::GuestReadableRangeCache persistent_ranges;
    host::GuestReadableRangeCache submit_ranges;
    uint64_t calls = 0, hits = 0, os_probes = 0;
};
thread_local GuestReadableCacheState g_guest_readable_cache;

bool guest_range_cache_hit(uint64_t begin, uint64_t end) {
    if (!g_guest_readable_cache.enabled) return false;
    if (g_guest_readable_cache.active) ++g_guest_readable_cache.calls;

    // Completion-label writes are folded on the guest draw thread before/after a renderer submit
    // scope. On Windows, falling through to VirtualQuery for every 8-byte label read made Astro's
    // otherwise headless frame loop take ~16 seconds per flip. The kernel-memory HLE already owns a
    // generation-guarded registry of fully committed readable guest mappings; reuse it here on every
    // call, not only while a renderer scope happens to be active. Sparse/lazy mappings are deliberately
    // absent from that registry and retain the OS probe/commit path below.
    g_guest_readable_cache.persistent_ranges.sync_generation(host::guest_mapping_generation());
    bool hit = g_guest_readable_cache.persistent_ranges.contains(begin, end) ||
               (g_guest_readable_cache.active &&
                g_guest_readable_cache.submit_ranges.contains(begin, end));
    if (!hit) {
        host::GuestReadableRange mapping{};
        if (host::guest_readable_mapping_containing(begin, end, mapping)) {
            g_guest_readable_cache.persistent_ranges.insert(mapping.begin, mapping.end);
            hit = true;
        }
    }
    if (hit && g_guest_readable_cache.active) ++g_guest_readable_cache.hits;
    return hit;
}

void cache_guest_readable_range(uint64_t begin, uint64_t end,
                                uint64_t query_begin, uint64_t query_end) {
    if (!g_guest_readable_cache.enabled || !g_guest_readable_cache.active || begin >= end) return;
    g_guest_readable_cache.submit_ranges.insert(begin, end);
    host::GuestReadableRange mapping{};
    if (host::guest_readable_mapping_containing(query_begin, query_end, mapping))
        g_guest_readable_cache.persistent_ranges.insert(mapping.begin, mapping.end);
}

// #2387: the same treatment for guest_writable, which had none. Its Linux arm answers by
// opening /proc/self/maps and running a fgets/sscanf loop over the process's whole mapping
// table -- per call, uncached -- while Windows and macOS each pay a syscall per region. Stack
// sampling of a Blue Prince frame loop put the main thread's stdio parsing (fgets,
// __vfscanf_internal, _IO_default_uflow) well above anything else it was doing.
//
// A SEPARATE range set, deliberately. Writability is not implied by readability, so this cache
// must never consult the readable registry (`guest_readable_mapping_containing`) or share its
// sets: doing so would let a read-only mapping satisfy a write probe, which turns a would-be
// refusal into a silent guest memory corruption. Every range in here was proven writable by an
// actual probe on this platform's own path.
//
// Generation-guarded exactly like the readable cache: `sync_generation` drops everything when the
// guest mapping generation moves, so an unmap/remap cannot leave a stale writable range behind.
// That mechanism is not new here -- it is the one the readable cache has already been relying on.
//
// THE INVARIANT THIS CACHE DEPENDS ON, written down because it is now load-bearing and was not
// stated anywhere: *every* change to guest page protection must advance
// host::guest_mapping_generation(). If a page is made read-only without advancing it, this cache
// keeps answering "writable" for it until something else moves the generation.
//
// The guest-facing path satisfies this: hle_kernel_mem.cpp implements protect as
// notify_guest_mapping_removed + notify_guest_mapping_added(..., committed && (prot & 0x1)), and
// both advance the generation.
//
// Watchpoint protection changes must obey the same generation contract (#2393/#2601).
// Their signal-safe notification is notify_guest_page_protection_changed(), not the
// mutex-taking mapping-registry operations. Never retain positives across those changes.
struct GuestWritableCacheState {
    bool enabled = getenv("PROSPER_NO_GUEST_WRITE_CACHE") == nullptr;   // bisection lever
    host::GuestReadableRangeCache ranges;    // the range-set container, not the readable DATA
    uint64_t calls = 0, hits = 0, os_probes = 0;
};
thread_local GuestWritableCacheState g_guest_writable_cache;

static bool guest_writable_cache_hit(uint64_t begin, uint64_t end) {
    if (!g_guest_writable_cache.enabled) return false;
    ++g_guest_writable_cache.calls;
    g_guest_writable_cache.ranges.sync_generation(host::guest_mapping_generation());
    const bool hit = g_guest_writable_cache.ranges.contains(begin, end);
    if (hit) ++g_guest_writable_cache.hits;
    return hit;
}

// `begin`/`end` must bound a span this call PROVED writable end to end, never merely the query.
static void cache_guest_writable_range(uint64_t begin, uint64_t end) {
    if (!g_guest_writable_cache.enabled || begin >= end) return;
    g_guest_writable_cache.ranges.insert(begin, end);
}

struct GuestReadableSubmitScope {
    GuestReadableSubmitScope() {
        g_guest_readable_cache.active = g_guest_readable_cache.enabled;
        g_guest_readable_cache.calls = 0;
        g_guest_readable_cache.hits = 0;
        g_guest_readable_cache.os_probes = 0;
        g_guest_readable_cache.submit_ranges.clear();
        if (g_guest_readable_cache.active)
            g_guest_readable_cache.persistent_ranges.sync_generation(
                host::guest_mapping_generation());
    }
    ~GuestReadableSubmitScope() {
        g_guest_readable_cache.active = false;
    }
};

} // namespace

// Readability probe (guest memory is 1:1-mapped, but a mis-decoded address could be unmapped),
// so guarded derefs on the render/submit thread don't risk a SIGSEGV. NOTE: /dev/null does NOT
// work for this — the kernel's null_write returns count without ever touching the source buffer,
// so the old probe reported EVERY address >= 0x1000 "readable" and all guards built on it were
// no-ops (verified empirically on this project's WSL kernel). A pipe write actually imports the
// user pages and returns EFAULT for unmapped memory. Readability is page-granular: probe one byte
// in each page the range touches, draining after every write so the pipe can never fill. Windows
// uses VirtualQuery for the same guard. Dynamic fetch resolution now runs in the native
// live renderer too, so an always-true probe turns a guest null descriptor-table pointer into a
// host access violation (#688).
#ifndef _WIN32
static int g_probe_pipe[2] = {-1, -1};
// One-time pipe creation via a C++11 magic static (thread-safe). guest_readable is shared with
// the multi-threaded HLE pointer probes: an unguarded `if (fd < 0) pipe2(...)` lazy init let two
// first-callers each pipe2 into the array — a torn pair (write-end of pipe B, read-end of pipe A)
// never drains, fills, and EAGAINs: VALID memory reported unreadable, silently. (PR #61 review.)
static bool make_probe_pipe() {
#ifdef __linux__
    return pipe2(g_probe_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
#else   // Darwin/BSD: no pipe2 — pipe + fcntl (still inside the once-only magic static, so no race)
    if (pipe(g_probe_pipe) != 0) return false;
    for (int fd : g_probe_pipe) {
        if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) return false;
        if (fcntl(fd, F_SETFL, O_NONBLOCK) != 0) return false;
    }
    return true;
#endif
}
static bool probe_pipe_ok() {
    static const bool ok = make_probe_pipe();
    return ok;
}
static bool probe_byte(uint64_t a) {
    ssize_t w = write(g_probe_pipe[1], (const void*)(uintptr_t)a, 1);
    if (w == 1) { char c; (void)!read(g_probe_pipe[0], &c, 1); return true; }
    return false;   // EFAULT: unmapped (EAGAIN can't happen — we drain after every byte)
}
bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0) return false;
    if (a + n < a) return false;   // wrap
    const uint64_t end = a + n;
    if (guest_range_cache_hit(a, end)) return true;
    if (!probe_pipe_ok()) return false;
    uint64_t last_page = (a + n - 1) & ~0xfffull;
    for (uint64_t p = a & ~0xfffull; p <= last_page; p += 0x1000) {
        if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
        if (!probe_byte(p < a ? a : p)) return false;
    }
    cache_guest_readable_range(a & ~0xfffull, last_page + 0x1000, a, end);
    return true;
}
#else
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);

bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0 || a + n < a) return false;
    const uint64_t end = a + n;
    if (guest_range_cache_hit(a, end)) return true;
    uint64_t cursor = a;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
        if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
        if (mbi.State != MEM_COMMIT) {
            if (!prosper_try_commit_dmem(cursor, end - cursor, 0)) return false;
            if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
            if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        }
        if (mbi.State != MEM_COMMIT || (mbi.Protect & blocked)) return false;
        const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                               PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (!(mbi.Protect & readable)) return false;
        const uint64_t region_end = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (region_end <= cursor) return false;
        const uint64_t query_end = std::min(end, region_end);
        cache_guest_readable_range((uint64_t)(uintptr_t)mbi.BaseAddress, region_end,
                                   cursor, query_end);
        cursor = query_end;
    }
    return true;
}
#endif

bool guest_writable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0 || a + n < a) return false;
    const uint64_t end = a + n;
    if (guest_writable_cache_hit(a, end)) return true;   // #2387
    // Widest span this call proves writable, so a later query anywhere inside the same
    // contiguous run of writable regions hits. Each arm below walks contiguous regions and
    // fails the moment one is missing or non-writable, so [span_lo, span_hi) is writable
    // end to end whenever the arm returns true. Recorded only on that success path.
    uint64_t span_lo = a, span_hi = a;
#ifdef _WIN32
    uint64_t cursor = a;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        ++g_guest_writable_cache.os_probes;
        if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        if (mbi.State != MEM_COMMIT) {
            if (!prosper_try_commit_dmem(cursor, end - cursor, 1) ||
                !VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        }
        const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
        const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & blocked) || !(mbi.Protect & writable))
            return false;
        const uint64_t region_end = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (region_end <= cursor) return false;
        const uint64_t region_lo = (uint64_t)(uintptr_t)mbi.BaseAddress;
        if (cursor == a) span_lo = std::max(region_lo, (uint64_t)0x1000);
        span_hi = region_end;
        cursor = std::min(end, region_end);
    }
    cache_guest_writable_range(span_lo, span_hi);   // #2387
    return true;
#elif defined(__APPLE__)
    uint64_t cursor = a;
    while (cursor < end) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const kern_return_t result = mach_vm_region(
            mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (result != KERN_SUCCESS || cursor < region || !(info.protection & VM_PROT_WRITE))
            return false;
        if (region > UINT64_MAX - size || region + size <= cursor) return false;
        if (cursor == a) span_lo = std::max((uint64_t)region, (uint64_t)0x1000);
        span_hi = (uint64_t)(region + size);
        cursor = std::min(end, static_cast<uint64_t>(region + size));
    }
    cache_guest_writable_range(span_lo, span_hi);   // #2387
    return true;
#else
    // Preserve generation-before-probe ordering. Exact covering queries avoid formatting
    // every unrelated VMA when write-watch activity frequently invalidates this cache.
    // Unsupported headers/kernels or ioctl access retain the text enumeration below.
    const uint64_t query_generation = host::guest_mapping_generation();
    const auto query = host::query_guest_writable_range(a, end);
    if (query.status != host::GuestWritableQueryStatus::Unavailable) {
        ++g_guest_writable_cache.os_probes;
        if (query.status == host::GuestWritableQueryStatus::Writable) {
            if (g_guest_writable_cache.enabled) {
                g_guest_writable_cache.ranges.sync_generation(query_generation);
                cache_guest_writable_range(std::max(query.begin, uint64_t{0x1000}), query.end);
            }
            return true;
        }
        return false;
    }
    // Compatibility path: enumerate all writable ranges in the text maps table and
    // retain them only when the original whole request succeeds. Unlike the binary
    // path, this warms unrelated VMAs too; their later queries may hit until invalidation.
    const uint64_t gen = host::guest_mapping_generation();
    ++g_guest_writable_cache.os_probes;
    int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;

    std::vector<host::GuestReadableRange> writable_ranges;
    writable_ranges.reserve(256);
    char buf[16384];
    char line[4096];
    size_t line_len = 0;
    ssize_t bytes_read = 0;

    while ((bytes_read = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t i = 0; i < bytes_read; ++i) {
            char c = buf[i];
            if (c == '\n') {
                line[line_len] = '\0';
                const char* p = line;
                uint64_t begin = 0, finish = 0;
                while (*p && *p != '-') {
                    char h = *p++;
                    uint64_t v = (h >= '0' && h <= '9') ? (h - '0') : (h >= 'a' && h <= 'f') ? (h - 'a' + 10) : 0;
                    begin = (begin << 4) | v;
                }
                if (*p == '-') {
                    ++p;
                    while (*p && *p != ' ') {
                        char h = *p++;
                        uint64_t v = (h >= '0' && h <= '9') ? (h - '0') : (h >= 'a' && h <= 'f') ? (h - 'a' + 10) : 0;
                        finish = (finish << 4) | v;
                    }
                    if (*p == ' ') {
                        ++p;
                        if (p[0] && p[1] == 'w' && finish > begin) {
                            begin = std::max(begin, (uint64_t)0x1000);
                            if (finish > begin) {
                                if (!writable_ranges.empty() && writable_ranges.back().end == begin) {
                                    writable_ranges.back().end = finish;
                                } else {
                                    writable_ranges.push_back({begin, finish});
                                }
                            }
                        }
                    }
                }
                line_len = 0;
            } else {
                if (line_len < sizeof(line) - 1) {
                    line[line_len++] = c;
                }
            }
        }
    }
    close(fd);

    bool found = false;
    for (const auto& r : writable_ranges) {
        if (r.begin <= a && end <= r.end) {
            found = true;
            break;
        }
    }

    if (found && g_guest_writable_cache.enabled) {
        g_guest_writable_cache.ranges.assign_sorted_ranges(std::move(writable_ranges), gen);
    }
    return found;
#endif
}

// Registered AGC headers publish the shader blob size in bytes. Dynamic descriptor folding used to
// ignore it and hand the decoder a fixed 0x4000-dword window, allowing the walk to read up to 64 KiB
// past a short shader. Retain that historical 64 KiB ceiling as a WORK bound too: CreateShader accepts
// guest metadata, so a corrupt multi-gigabyte shader_size must not become a page-probe/decode/OOM budget.
// Truncate a non-dword-aligned tail rather than reading one byte beyond the blob, and refuse the bounded
// span if it is not wholly readable. A zero result safely disables only the optional fold;
// metadata-described resources remain available to build_stage_table.
size_t dynamic_fold_shader_dwords(uint32_t shader_size_bytes) {
    constexpr size_t kMaxDynamicFoldDwords = 0x4000;
    return std::min<size_t>(shader_size_bytes / sizeof(uint32_t), kMaxDynamicFoldDwords);
}

size_t registered_shader_dwords(const AgcShaderHeader& header, uint64_t code_addr) {
    const size_t dwords = dynamic_fold_shader_dwords(header.shader_size);
    const uint32_t bounded_bytes = static_cast<uint32_t>(dwords * sizeof(uint32_t));
    if (!bounded_bytes || !guest_readable(code_addr, bounded_bytes)) return 0;
    return dwords;
}

// A null BVH marker is only safe when the static ray instruction is inside a real EXEC region. Prove
// the narrow compiler shape `EXEC writer; s_cbranch_execz MERGE; ... ray ...; MERGE`, and reject any
// external branch edge into the region. This is intentionally stronger than merely observing an
// earlier forward branch: the prior no-hit experiment did not establish dominance and could hide an
// unguarded missing descriptor.
// SOPP direct branches: s_branch (0x02) and the s_cbranch_* family (0x04..0x09). Hoisted so the two
// CFG proofs in this file share one definition of "is a branch" and one target computation — #2181
// unified four private copies of the VOPC cmpx windows for the same reason, and #2120 is the
// cautionary tale for a forked predicate.
bool sopp_is_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
           (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
}
bool sopp_is_unconditional_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && in.opcode == 0x02;
}
// GFX10 branch target: PC of the branch + its own length + the signed dword displacement. Direction
// is deliberately NOT filtered here — a predecessor tally that only counts forward edges is not a
// predecessor tally (#2202 review, B2).
int64_t sopp_branch_target(const Rdna2Inst& in) {
    return static_cast<int64_t>(in.pc) + static_cast<int64_t>(in.len_dwords) +
           static_cast<int64_t>(in.simm16);
}
// Indirect control transfer: s_setpc_b64 / s_swappc_b64 / s_rfe_b64 (SOP1 0x20/0x21/0x22) and
// s_call_b64 (SOPK 0x16). Encodings round-tripped through llvm-mc -mcpu=gfx1030, never read off a
// table (see SONIC_CROSSWORLDS_STATUS.md § Ruled out, the 0x305 trap). A shader containing one has a
// CFG no static scan over SOPP displacements can represent.
bool has_indirect_control_flow(const std::vector<Rdna2Inst>& instructions) {
    for (const Rdna2Inst& in : instructions) {
        if (in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == 0x20 || in.opcode == 0x21 || in.opcode == 0x22))
            return true;
        if (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16) return true;
    }
    return false;
}

bool guarded_bvh_use(const std::vector<Rdna2Inst>& instructions, uint32_t use_pc) {
    // An indirect target can enter the guarded interval after its EXECZ check, or leave and later
    // re-enter it. No scan over direct SOPP displacements can prove dominance in that program.
    if (has_indirect_control_flow(instructions)) return false;

    const auto& is_branch = sopp_is_branch;
    const auto& target = sopp_branch_target;

    for (size_t i = 1; i < instructions.size(); ++i) {
        const Rdna2Inst& guard = instructions[i];
        if (guard.fmt != Rdna2Format::SOPP || guard.opcode != 0x08 ||
            guard.simm16 <= 0 || guard.pc >= use_pc)
            continue;
        const int64_t merge64 = target(guard);
        if (merge64 <= static_cast<int64_t>(use_pc) || merge64 > UINT32_MAX)
            continue;
        const uint32_t merge = static_cast<uint32_t>(merge64);
        const Rdna2Inst& exec_writer = instructions[i - 1];
        if (exec_writer.pc + exec_writer.len_dwords != guard.pc ||
            !rdna2_instruction_may_change_exec(exec_writer))
            continue;
        if (std::none_of(instructions.begin(), instructions.end(),
                         [&](const Rdna2Inst& in) { return in.pc == merge; }))
            continue;

        bool external_entry = false;
        for (const Rdna2Inst& branch : instructions) {
            if (!is_branch(branch) || &branch == &guard) continue;
            const int64_t branch_target = target(branch);
            const bool target_inside = branch_target > static_cast<int64_t>(guard.pc) &&
                                       branch_target < static_cast<int64_t>(merge);
            const bool source_inside = branch.pc > guard.pc && branch.pc < merge;
            if (target_inside && !source_inside) { external_entry = true; break; }
        }
        if (!external_entry) return true;
    }
    return false;
}

bool gta5_null_raw_store_descriptor(const std::array<uint32_t, 4>& descriptor) {
    // Exact live null V#: base=0, stride=56, one record, 56 bytes, OOB_SELECT=0, TYPE=0 and no
    // swizzled-address controls. A non-null dispatch changes the base words and stays on the ordinary
    // materialized-buffer path.
    return descriptor == std::array<uint32_t, 4>{
        0x00000000u, 0x00380000u, 0x00000001u, 0x00016204u};
}

// The x16-header null proof deliberately accepts one observed straight-line descriptor builder, not
// a general scalar phi analysis. Every instruction that carries the root from its mapped load to the
// ray must therefore execute whenever the ray does. Reject a branch inside that interval, or any
// edge entering it after the load; branches that skip both the load and the ray remain harmless.
bool straight_line_null_chain_dominates(const std::vector<Rdna2Inst>& instructions,
                                        uint32_t definition_pc, uint32_t use_pc) {
    if (definition_pc >= use_pc) return false;
    bool found_definition = false, found_use = false;
    for (const Rdna2Inst& in : instructions) {
        found_definition |= in.pc == definition_pc;
        found_use |= in.pc == use_pc;
        if (!sopp_is_branch(in)) continue;
        const int64_t branch_target = sopp_branch_target(in);
        if (in.pc > definition_pc && in.pc < use_pc) return false;
        if (branch_target > static_cast<int64_t>(definition_pc) &&
            branch_target <= static_cast<int64_t>(use_pc))
            return false;
    }
    return found_definition && found_use;
}

// --- Bindless-dynamic vertex-fetch resolution (const-fold the scalar setup) ---------------------------
// This game's NGG vertex shader loads its vertex-buffer V# from a descriptor table at a RUNTIME-computed
// offset (e.g. `s_load_dwordx4 s[8:11], s[24:25], vcc_hi` where `vcc_hi = (s64<<4)&0x1f0` and
// `s64 = *[s26:27]`). The recompiler resolves descriptors by a STATIC provenance key, so it can't match a
// computed offset. We const-fold the wave-uniform scalar setup here: seed the concrete user-data SGPR
// values, interpret the scalar ALU + scalar loads (reading the 1:1-mapped guest memory), and snapshot the
// V# each descriptor load produces AT LOAD TIME (before the shader's later dynamic stride patch). The
// result maps each buffer_load_format's SRSRC SGPR -> its decoded V#, which build_stage_table emits as a
// VertexBuffer keyed by sgpr_base so the recompiler's by_sgpr_base() resolves it. Uniform-scalar only: any
// value that would depend on a VGPR/lane is left unknown (the op's dest becomes unknown), so we never
// fabricate a per-lane-dependent descriptor. CONFIDENCE: MED (covers this game's fetch-shader shape).
// External linkage (DynFetch + declaration in gpu_execute.hpp) so the fold is unit-testable.
static uint64_t scalar_buffer_dword_count(const DecodedBufferDescriptor& descriptor) {
    // A scalar address is a BYTE offset into the descriptor's span:
    //
    //     m_size = (m_stride == 0 ? 1 : m_stride) * m_num_records      [bytes]
    //
    // which is exactly decode_buffer_descriptor::size_bytes. A dword access needs its whole four
    // bytes inside that span, so the dword bound truncates. CONFIDENCE: HIGH.
    //
    // Read the RDNA2 guide's own §7.2.1 with care: it prints `m_size = (m_stride == 0) ? 1 :
    // m_num_records`, dropping the stride factor, even though its buffer-descriptor table (§8.1.8)
    // states NUM_RECORDS is in units of stride when strided and bytes otherwise. The RDNA3 guide
    // §8.1.2 and the RDNA4 guide §8.1.2 both print the corrected product, and RDNA3 §8.4.1 defines
    // out-of-range as `offset >= (stride == 0 ? 1 : stride) * num_records` — a byte-address bound.
    // The RDNA2 line is a documentation omission; do not re-derive the bound from it alone (#2528).
    //
    // #2528: reading NUM_RECORDS as a DWORD count (and STRIDE==0 as one dword) makes every ordinary
    // constant buffer a single dword long. Two independent live falsifications, one per branch:
    //   * STRIDE==0 — The Messenger, whose first level is checked against PS5 hardware, rendered
    //     111,118 consecutive byte-identical fully black frames; Dead Cells collapsed the same way.
    //   * STRIDE!=0 — GTA V's 0x413ced900 indexes a STRIDE=120/NUM_RECORDS=5 descriptor table with
    //     `s_mul_i32 vcc_hi, s9, 0x78` + `s_buffer_load_dwordx4 …, vcc_hi offset:0x8` for s9 in
    //     0..4. A NUM_RECORDS-dword bound admits only 20 of those 600 bytes, so four of the five
    //     records load an all-zero V#; 269 routed dispatches resolved every record to its real
    //     vertex buffer before that bound existed.
    //
    // An EMPTY V# keeps eece6f84's separately reviewed stride-zero rule, which is where the
    // one-dword reading came from and the only domain it was ever exercised in: NUM_RECORDS==0 with
    // STRIDE==0 still admits its first dword, so only a strictly positive offset is proven OOB.
    if (descriptor.num_records == 0u && descriptor.size_bytes == 0u)
        return descriptor.stride == 0u ? 1ull : 0ull;
    return static_cast<uint64_t>(descriptor.size_bytes) / sizeof(uint32_t);
}

static bool sbuffer_access_is_fully_oob(const Rdna2Inst& in,
                                        const DecodedBufferDescriptor& descriptor,
                                        bool exact_byte_offset_known = false,
                                        uint32_t exact_byte_offset = 0u) {
    if (in.fmt != Rdna2Format::SMEM || in.opcode < 0x8u || in.opcode > 0xCu)
        return false;

    const uint64_t scalar_dwords = scalar_buffer_dword_count(descriptor);
    if (exact_byte_offset_known) {
        // The fold already preserved the 32-bit SOFFSET+OFFSET result. Scalar memory converts that
        // effective byte offset to its first dword index.
        const uint32_t first_dword = exact_byte_offset >> 2u;
        return static_cast<uint64_t>(first_dword) >= scalar_dwords;
    }

    // An unknown SOFFSET can only be specialized by the established empty-descriptor proof. SOFFSET
    // is unsigned, so a non-negative immediate names its minimum possible dword. Do not generalize
    // this minimum argument to a non-empty descriptor: the exact-offset and emitted dynamic paths
    // below handle those without assuming a runtime value.
    if (static_cast<int32_t>(in.literal) < 0 || descriptor.num_records != 0u ||
        descriptor.size_bytes != 0u)
        return false;
    return static_cast<uint64_t>(in.literal >> 2u) >= scalar_dwords;
}

static uint32_t validated_scalar_buffer_dword_count(
        const SrtUse& use, const DecodedBufferDescriptor& descriptor,
        const uint32_t* code, size_t dwords) {
    if (!use.scalar_buffer_dword_count) return 0u;
    // An EMPTY V# carries no resource-level scalar bound. `scalar_buffer_dword_count` still answers
    // one dword for the stride-zero empty case, because eece6f84's fold specialization is about
    // whether a given access is provably out of range — but a RESOURCE built from that descriptor
    // has `size == 0`, and a carried count of one then contradicts its own footprint. #2529 made
    // `shader_resource_buffer_binding_bytes` authenticate the count against `size`, which turned
    // that contradiction into a hard rejection: every GTA V compute capture failed to write with
    // "invalid scalar-buffer bound metadata", losing the capture/replay tool entirely on that title.
    // The empty descriptor's zero semantics are carried by `zero_record_raw`, not by this field.
    if (descriptor.size_bytes == 0u) return 0u;
    if (use.kind != 1 || use.instruction_format != UINT32_MAX || use.use_pc >= dwords)
        return 0u;
    const Rdna2Inst consumer = rdna2_decode_one(code + use.use_pc, dwords - use.use_pc);
    if (consumer.fmt != Rdna2Format::SMEM || consumer.opcode < 0x8u ||
        consumer.opcode > 0xCu ||
        scalar_buffer_dword_count(descriptor) != use.scalar_buffer_dword_count)
        return 0u;
    const uint64_t backing_bytes =
        static_cast<uint64_t>(use.scalar_buffer_dword_count) * sizeof(uint32_t);
    return backing_bytes <= 0x10000000ull ? use.scalar_buffer_dword_count : 0u;
}

static bool zero_record_format_selectors_are_zero(const Rdna2Inst& in,
                                                   const uint32_t descriptor_words[4]) {
    if (in.fmt != Rdna2Format::MUBUF || in.opcode > 0x3u)
        return false;
    const uint32_t components = in.opcode + 1u;
    for (uint32_t component = 0; component < components; ++component) {
        const uint32_t selector = (descriptor_words[3] >> (component * 3u)) & 0x7u;
        // SQ_SEL_0 and X/Y/Z/W all yield zero for an OOB record. SQ_SEL_1 yields one; 2/3 are
        // reserved and stay fail-closed rather than acquiring an invented constant value.
        if (selector != 0u && selector < 4u)
            return false;
    }
    return true;
}

// GTA V rebuilds the same linear qword-atomic V# shape with dispatch-sized record counts (the routed
// scene exercises 25, 2, 3, and 1). RDNA2 range-checks INDEX against NUM_RECORDS and OFFSET against
// STRIDE, so the exact concrete count can be compiled as the all-or-nothing qword bound. Keep every
// other packet/descriptor control deliberately narrow; SLC/TFE/reserved dword1 variants have
// different or invalid contracts and the decoder does not otherwise retain those bits.
static uint32_t exact_atomic_x2_record_count(
        const Rdna2Inst& in, const DecodedBufferDescriptor& descriptor,
        const uint32_t descriptor_words[4]) {
    const uint32_t offset = in.literal & 0xfffu;
    const bool offen = ((in.literal >> 12u) & 1u) != 0;
    const bool idxen = ((in.literal >> 13u) & 1u) != 0;
    const bool zero_soffset =
        (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
        (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
    const bool supported_opcode = in.opcode == 0x50u || in.opcode == 0x5au;
    const bool exact = in.fmt == Rdna2Format::MUBUF && supported_opcode &&
           idxen && !offen && offset == 0u && zero_soffset &&
           !in.mubuf_dlc && !in.mubuf_lds && (in.words[0] & 0x00020000u) == 0u &&
           (in.words[1] & 0x00e00000u) == 0u &&
           in.src[0].kind == OperandKind::VGPR && descriptor.base > 0x10000u &&
           (descriptor.base & 7u) == 0u && descriptor.stride == 8u &&
           descriptor.num_records != 0u && descriptor.num_records <= 0x02000000u &&
           descriptor.size_bytes == static_cast<uint64_t>(descriptor.num_records) * 8u &&
           (descriptor_words[1] & 0xc0000000u) == 0u && // no swizzled addressing
           // Exact observed upper controls: no INDEX_STRIDE/ADD_TID/RESOURCE_LEVEL/reserved bits,
           // OOB_SELECT=0, and buffer TYPE=0. The normalized ShaderResource retains none of these.
           (descriptor_words[3] & 0xfff80000u) == 0u;
    return exact ? descriptor.num_records : 0u;
}

static bool gta_optional_null_descriptor_shape(
        const DecodedBufferDescriptor& descriptor, const uint32_t words[4]) {
    return words[0] == 0u && words[1] == kGtaOptionalBufferStrideWord &&
           words[2] != 0u && words[3] == kGtaOptionalBufferConfigWord &&
           descriptor.base == 0u && descriptor.stride == kGtaOptionalBufferStride &&
           descriptor.num_records == words[2] &&
           descriptor.size_bytes == static_cast<uint64_t>(words[2]) *
                                        kGtaOptionalBufferStride &&
           descriptor.size_bytes <= 0x10000000u;
}

static bool gta_nullable_output_descriptor_shape(
        const DecodedBufferDescriptor& descriptor, const uint32_t words[4],
        uint32_t record_count) {
    return words[0] == 0u && words[1] == kGtaNullableOutputStrideWord &&
           record_count != 0u && record_count <= kGtaNullableOutputMaxRecordCount &&
           words[2] == record_count &&
           words[3] == kGtaNullableOutputConfigWord &&
           descriptor.base == 0u && descriptor.stride == kGtaNullableOutputStride &&
           descriptor.num_records == record_count &&
           descriptor.size_bytes == static_cast<uint64_t>(record_count) *
                                        kGtaNullableOutputStride;
}

std::vector<DynFetch>
resolve_dynamic_fetch(const uint32_t* code, size_t dwords, const uint32_t* user_sgprs, uint32_t nsgpr,
                      uint32_t user_sgpr_base, std::vector<SrtUse>* srt_uses,
                      uint32_t pcrel_dispatch_target,
                      const PcrelDispatchInfo* pcrel_dispatch,
                      const uint32_t* system_sgprs, uint32_t nsystem_sgprs) {
    using FoldClock = std::chrono::steady_clock;
    static const bool profile_fold = std::getenv("PROSPER_STAGE_FOLD_PROFILE") != nullptr;
    const auto fold_start = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    const size_t srt_before = srt_uses ? srt_uses->size() : 0;
    uint64_t guest_probe_calls = 0;
    double guest_probe_ms = 0.0;
    std::vector<DynFetch> out;
    const auto decoded = decode_shader_cached(code, dwords);
    const auto decode_done = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    std::vector<Rdna2Inst> specialized;
    const std::vector<Rdna2Inst>* fold_instructions = &decoded->instructions;
    if (pcrel_dispatch_target != UINT32_MAX) {
        specialized = decoded->instructions;
        const PcrelDispatchInfo dispatch = pcrel_dispatch
            ? *pcrel_dispatch : rdna2_pcrel_dispatch_info(code, dwords);
        if (!rdna2_specialize_pcrel_dispatch(specialized, dispatch,
                                             pcrel_dispatch_target)) {
            // The fragment recompiler will reject the same unprovable specialization. Do not walk all
            // alternatives here: doing so would fabricate resource provenance for code that cannot run.
            specialized.clear();
        }
        fold_instructions = &specialized;
    }
    // The cache specializes the full decoded stream before compacting it for the scalar fold. A
    // PC-relative dispatch is already a separate explicit specialization and keeps its historical
    // filtered stream here; combining the two requires proving the selected full-stream CFG first.
    if (fold_instructions == &decoded->instructions && decoded->shader_constant_specialized)
        fold_instructions = &decoded->shader_constant_instructions;
    const auto& ins = *fold_instructions;
    const bool gta5_null_raw_store_guard = pcrel_dispatch_target == UINT32_MAX &&
        rdna2_gta5_null_guarded_raw_store_shader(code, dwords);

    auto readable = [&](uint64_t addr, uint32_t bytes) {
        if (!profile_fold) return guest_readable(addr, bytes);
        const auto start = FoldClock::now();
        const bool result = guest_readable(addr, bytes);
        guest_probe_ms += std::chrono::duration<double, std::milli>(
            FoldClock::now() - start).count();
        ++guest_probe_calls;
        return result;
    };

    // PROSPER_DYNTRACE traces the whole const-fold walk; PROSPER_DYNTRACE_ADDR=<hex code addr>
    // narrows it to ONE shader (a full run otherwise traces every draw's walk — unusable volume).
    // g_dyntrace_force: set by the PROSPER_DYNTRACE_FAIL failure-replay path (gpu_execute.hpp) so
    // the walk of a shader that just FAILED to recompile is traced without knowing its address.
    bool trc = g_dyntrace_force || getenv("PROSPER_DYNTRACE") != nullptr;
    if (trc && !g_dyntrace_force)
        if (const char* fa = getenv("PROSPER_DYNTRACE_ADDR"))
            trc = strtoull(fa, nullptr, 16) == (uint64_t)(uintptr_t)code;
    // A full scalar-fold trace is intentionally verbose. Live shaders can rebuild their stage table
    // thousands of times per scene, so permit a targeted diagnostic run to capture the first matching
    // fold without turning every subsequent frame into gigabytes of duplicate logging.
    if (trc && !g_dyntrace_force && getenv("PROSPER_DYNTRACE_ONCE")) {
        static std::mutex once_mx;
        static std::set<const uint32_t*> traced;
        std::lock_guard<std::mutex> lk(once_mx);
        trc = traced.insert(code).second;
    }
    // Scalar operands encode at most 128 SGPRs. Fixed register files avoid hundreds of tiny hash/tree
    // allocations per submit while retaining exactly the same known/unknown state model.
    constexpr size_t kFoldSgprs = 128;
    std::array<uint32_t, kFoldSgprs> val{};                 // concrete SGPR values
    std::bitset<kFoldSgprs> val_known;
    // Entry-user-data identity attached to an otherwise ordinary scalar value. This is narrower
    // than descriptor provenance: it only proves that s_mov_b32/s_mov_b64 copies reassembled
    // consecutive entry dwords, without arithmetic or a load changing any word.
    std::array<uint32_t, kFoldSgprs> val_seed_origin{};
    std::bitset<kFoldSgprs> val_seed_origin_known;
    // Descriptor provenance attached to the CURRENT scalar value. Unlike the load-time snapshots
    // below, this follows s_mov shuffles and is cleared by arithmetic/data writes. A key-less value
    // uses exact consuming-pc provenance, matching the recompiler after scalar spills.
    std::array<uint32_t, kFoldSgprs> val_srt_key{};
    std::bitset<kFoldSgprs> val_srt_key_known;
    std::unordered_map<uint32_t, uint32_t> scalar_spill_slots; // (VGPR << 6) | lane -> scalar value
    std::array<std::array<uint32_t, 4>, kFoldSgprs> descr{}; // load-time V# snapshots by base SGPR
    std::bitset<kFoldSgprs> descr_known;
    // RUNTIME-SELECTED descriptor table (#2481). A `s_buffer_load_dwordx4` whose scalar offset the
    // fold cannot resolve still names a descriptor that is one OF a bounded table, because the outer
    // V# it reads through is fully known. The selected element is not knowable here — GTA V derives
    // it from `v_readfirstlane` inside a waterfall loop, so it genuinely varies per wave — but the
    // TABLE is, and that is what a Vulkan descriptor array binds. Record the producer so a later
    // MUBUF consumer of these SGPRs can publish the whole declared table plus the exact instruction
    // whose live scalar value selects within it. Deliberately kept apart from `descr`: that promises
    // "the descriptor IS these words", this promises "the descriptor is one of these N".
    struct FoldTableSource {
        uint32_t load_pc = 0xFFFFFFFFu;
        uint64_t base = 0;
        uint32_t stride = 0;
        uint32_t records = 0;
        uint32_t element_offset = 0;   // byte offset of the V# inside each table record
    };
    std::array<FoldTableSource, kFoldSgprs> table_source{};
    std::bitset<kFoldSgprs> table_source_known;
    // Descriptor-TABLE provenance (#294): for each snapshotted 4/8-dword s_load, the load's IMMEDIATE
    // byte offset — the recompiler's sreg_srt/by_srt_offset key. 0xFFFFFFFF = not provenance-usable
    // (register-SOFFSET or negative-immediate load, which emit_alu doesn't tag).
    std::array<uint32_t, kFoldSgprs> descr_key{};
    std::bitset<kFoldSgprs> descr_key_known;
    std::array<std::array<uint32_t, 8>, kFoldSgprs> descr8{};  // load-time T# snapshots by base SGPR
    std::bitset<kFoldSgprs> descr8_known;
    std::array<uint32_t, kFoldSgprs> descr8_key{};
    std::bitset<kFoldSgprs> descr8_key_known;
    // Exact source address of each CURRENT scalar word from a successful mapped x8/x16 load. The
    // address follows only bit-preserving scalar moves and is cleared by every other write. A live
    // T# receives one contiguous source only when word k still comes from base + 4*k; reorders and
    // duplicates therefore fail closed even when every word came from the same load window.
    std::array<uint64_t, kFoldSgprs> descriptor_word_source_addr{};
    std::bitset<kFoldSgprs> descriptor_word_source_known;
    // Exact null-pointer dataflow for guarded BVHs. Each mapped zero qword load receives a unique
    // origin; failed dereferences and scalar address/descriptor ALU retain that origin. A null BVH is
    // published only when all four live descriptor words carry the SAME origin, so an unrelated
    // unknown/zero SGPR cannot turn an unresolved ray instruction into a synthetic result.
    std::array<uint32_t, kFoldSgprs> null_chain_origin{};
    std::bitset<kFoldSgprs> null_chain_known;
    uint32_t next_null_chain_origin = 0;
    // Semantic provenance for GTA V's mapped optional table entry. Generic mapped zero qwords keep
    // their existing null-chain identity, but only the exact s0:s1 + 0x58 producer may attach these
    // descriptor roles and eventually publish a nonzero-record zero-read candidate.
    enum class OptionalNullRole : uint8_t { None, BaseLo, BaseHi, RecordCount, Config };
    std::array<OptionalNullRole, kFoldSgprs> optional_null_role{};
    // Producer PC per tagged origin. The exact location lets the optional proof impose a narrow
    // dominance contract over direct CFG instead of trusting this fold's linear walk.
    std::vector<uint32_t> optional_table_null_origin_pc(1u, UINT32_MAX);
    // Same mapped-qword proof for GTA V's distinct +0x20 output/work pointer convention. Keeping a
    // separate origin table prevents the existing +0x58 load-only contract from authorizing stores.
    std::vector<uint32_t> nullable_output_null_origin_pc(1u, UINT32_MAX);
    // Only x16-header roots need the narrow straight-line dominance contract added for GTA V. The
    // generic mapped-qword null proof predates this shape and has separate loop/CFG validation.
    // Origin ids are monotonic and never restored, so this metadata is immutable once published.
    std::vector<uint32_t> x16_null_origin_pc(1u, UINT32_MAX);
    // Null-pointer provenance carried by SCC between the low and high halves of an exact
    // s_add_u32/s_addc_u32 pair. The concrete carry can be unknown after a failed null dereference;
    // this records only that the high result still belongs to that null chain.
    uint32_t null_count_carry_origin = 0;
    // Exact provenance for GTA V's RTIP 1.1 descriptor builder. A successful mapped x16 header
    // load seeds candidate qword origins; only the observed mask/lshl/load-count/bfe/carry/header
    // chain can turn one into a complete descriptor. This is deliberately separate from concrete
    // value knowledge: four literal-built words that merely decode as TYPE=8 remain unresolved.
    enum class BvhBuildRole : uint8_t {
        None,
        HeaderLo, HeaderHi,
        PointerLo, PointerHi,
        CountLo, CountHi, CountPlusOne,
        SizeLo, SizeHi, SizeHiMasked, DescriptorHi,
        BaseLo, BaseHi, SortedBaseHi,
    };
    std::array<uint32_t, kFoldSgprs> bvh_build_origin{};
    std::array<BvhBuildRole, kFoldSgprs> bvh_build_role{};
    uint32_t next_bvh_build_origin = 0;
    uint32_t bvh_count_carry_origin = 0;
    // SGPRs overwritten by an s_load since seeding — the seed-V# MUBUF fallback below must not use a
    // stale user-data snapshot once the register was RELOADED from memory (ALU patches deliberately
    // don't count: descriptor snapshots are load-time semantics, pre-patch, like `descr`).
    std::bitset<kFoldSgprs> reloaded;
    enum class FoldMask : uint8_t { Unknown, None, All };
    std::array<FoldMask, kFoldSgprs> mask_state{};
    constexpr size_t kFoldVgprs = 256;
    std::array<VertexFetchIndexMode, kFoldVgprs> vector_index_mode{};
    const bool explicit_ngg_index_provenance = user_sgpr_base == 8;
    if (explicit_ngg_index_provenance) {
        // GFX10 NGG merged VS/GS ABI: the VS inputs follow five GS VGPRs.
        vector_index_mode[5] = VertexFetchIndexMode::Vertex;
        vector_index_mode[8] = VertexFetchIndexMode::Instance;
    }
    int scc = -1;   // tracked SCC (-1 unknown): set by s_cmp_*, consumed by s_cselect (the format patch's tail)
    bool gta5_null_pointer_at_guard = false;
    // The SPI loads the user-data block starting at shader SGPR `user_sgpr_base` (s0..s7 are NGG system
    // SGPRs). So user-data block index k lands in shader SGPR (user_sgpr_base + k).
    auto valid_reg = [](int r) { return r >= 0 && r < (int)kFoldSgprs; };
    // PROSPER_DYNTRACE_SGPR=<n>: report every fold-visible transition of one scalar register. The
    // existing traces cover SMEM/MIMG/MUBUF/BVH and deliberately not scalar ALU, so a register the fold
    // ends up not knowing is invisible in them by construction -- which is #2412's remaining question
    // ("what makes the fold lose the descriptor-table pointer") and nothing in the log could answer it.
    //
    // The pc is trustworthy, and that was checked rather than assumed. Every set_value/forget call that
    // can fire during the walk is inside the ONE instruction loop below; the only calls outside it are
    // the pre-loop seeding of system/user SGPRs, which report the sentinel 0xffffffff and so cannot be
    // mistaken for an instruction. (The two earlier scanning loops in this function build branch edges
    // and mutate no fold state.)
    uint32_t watch_pc = 0xffffffffu;   // 0xffffffff = pre-loop seeding, not an instruction
    // Raw words of the instruction being folded, so a FORGOTTEN line can be decoded directly with
    // `llvm-mc -disassemble` instead of matched back to a dumped shader by pc -- pcs are
    // program-local and several of this title's shaders share a prologue, which makes that match
    // unreliable (#2412).
    uint32_t watch_w0 = 0, watch_w1 = 0; uint32_t watch_len = 0;

    // PROSPER_DYNTRACE_SCC=1 — report every transition of the tracked SCC, with the pc and words that
    // caused it. SCC is a single bit and it gates whole descriptors: GTA V's ray-tracing pass builds
    // its BVH descriptor's dword3 with `s_addc_u32 s19, -1, 0`, whose operands are both inline
    // constants, so the ONLY thing that can leave it unknown is an unknown SCC. Without this, "the
    // descriptor did not resolve" gives no way to find which earlier instruction cost it.
    //
    // Deliberately mirrors the register watch, including printing the program identity rather than
    // the sh= signature -- which is not unique (#2548).
    const bool watch_scc = std::getenv("PROSPER_DYNTRACE_SCC") != nullptr;
    int last_reported_scc = -2;
    auto report_scc = [&](const char* why) {
        if (!watch_scc || scc == last_reported_scc) return;
        last_reported_scc = scc;
        fprintf(stderr, "[sccwatch] program=0x%llx pc=%u scc=%s %s words=%08x:%08x len=%u\n",
                (unsigned long long)(uintptr_t)code, watch_pc,
                scc < 0 ? "UNKNOWN" : (scc ? "1" : "0"), why, watch_w0, watch_w1, watch_len);
    };
    // Per-register record of the instruction that most recently made it UNKNOWN. Sized like the other
    // per-register fold state; `0xffffffff` means "never forgotten in this walk", which is a distinct
    // and useful answer from "forgotten by <instruction>" — a V# word that was never written at all
    // fails for a different reason than one a modeled instruction gave up on.
    std::array<uint32_t, kFoldSgprs> forget_pc{}; forget_pc.fill(0xffffffffu);
    std::array<uint32_t, kFoldSgprs> forget_w0{}, forget_w1{};
    const int watch_sgpr = [] {
        const char* w = std::getenv("PROSPER_DYNTRACE_SGPR");
        return w ? (int)strtol(w, nullptr, 0) : -1;
    }();
    auto set_value = [&](int r, uint32_t v) {
        // `sh=` identifies WHICH PROGRAM this transition belongs to, derived exactly as the reject
        // lines derive it (first code dword + span; rdna2_to_spirv.cpp:14974). Without it the watch
        // is a bare register number across every program folded in the run, and register numbers are
        // program-local — so two lanes tracing "s16" can legitimately observe different registers and
        // reach contradictory conclusions about the same chain. That happened: a Windows trace of s16
        // showed ZERO forget sites while the Linux fold attributed the loss of the descriptor-table
        // pointer to a v_cmp writing s16, and neither observation was wrong. (#2412)
        // `program=` alongside `sh=`, because sh= IS NOT UNIQUE. It is the first code dword plus the
        // span, and the first dword of a great many GTA V shaders is `bfa00003` -- `s_branch +3`, an
        // ordinary prologue. Two different 276-dword programs therefore share `sh=bfa00003/276`, and
        // filtering a watch by it silently mixes them: a trace filtered that way showed s106 being
        // FORGOTTEN at a pc whose instruction words do not appear anywhere in the program being
        // studied. The comment below claims this signature "identifies one cheaply and stably" -- it
        // is cheap and stable, and it is not an identity. The code pointer is.
        if (r == watch_sgpr)
            fprintf(stderr,
                    "[sgprwatch] program=0x%llx sh=%08x/%zu pc=%u s%d <- KNOWN 0x%08x\n",
                    (unsigned long long)(uintptr_t)code,
                    dwords ? code[0] : 0u, dwords, watch_pc, r, v);
        if (valid_reg(r)) {
            val[(size_t)r] = v;
            val_known.set((size_t)r);
            table_source_known.reset((size_t)r);
            val_srt_key_known.reset((size_t)r);
            val_seed_origin_known.reset((size_t)r);
            descriptor_word_source_known.reset((size_t)r);
            null_chain_known.reset((size_t)r);
            optional_null_role[(size_t)r] = OptionalNullRole::None;
            bvh_build_origin[(size_t)r] = 0;
            bvh_build_role[(size_t)r] = BvhBuildRole::None;
            mask_state[(size_t)r] = FoldMask::Unknown;
        }
    };
    auto forget = [&](int r) {
        if (r == watch_sgpr)
            fprintf(stderr,
                    "[sgprwatch] program=0x%llx sh=%08x/%zu pc=%u s%d <- FORGOTTEN "
                    "words=%08x:%08x len=%u\n",
                    (unsigned long long)(uintptr_t)code,
                    dwords ? code[0] : 0u, dwords, watch_pc, r, watch_w0, watch_w1, watch_len);
        if (valid_reg(r)) {
            val_known.reset((size_t)r);
            val_srt_key_known.reset((size_t)r);
            val_seed_origin_known.reset((size_t)r);
            table_source_known.reset((size_t)r);
            descriptor_word_source_known.reset((size_t)r);
            null_chain_known.reset((size_t)r);
            optional_null_role[(size_t)r] = OptionalNullRole::None;
            bvh_build_origin[(size_t)r] = 0;
            bvh_build_role[(size_t)r] = BvhBuildRole::None;
            mask_state[(size_t)r] = FoldMask::Unknown;
            // Remember WHICH instruction made this register unknown (#2412). `[mubuf-unknown]` names
            // the V# word that is not provable; it could not name the instruction responsible, so the
            // cause had to be recovered by hand-tracing one shader at a time. Three such traces
            // produced three different mechanisms — a pointer-assembled V#, a VCC-as-soffset load, and
            // a mask-recycled register — which is exactly the point at which per-site archaeology stops
            // converging and the aggregate has to be asked instead.
            //
            // Cost is two stores on a path that already writes five bitsets, and it is read only by the
            // opt-in diagnostics below.
            forget_pc[(size_t)r] = watch_pc;
            forget_w0[(size_t)r] = watch_w0;
            forget_w1[(size_t)r] = watch_w1;
        }
    };
    for (uint32_t i = 0; system_sgprs && i < nsystem_sgprs && i < kFoldSgprs; ++i)
        set_value(static_cast<int>(i), system_sgprs[i]);
    for (uint32_t i = 0; i < nsgpr; i++) {
        const int reg = (int)(user_sgpr_base + i);
        set_value(reg, user_sgprs[i]);
        if (valid_reg(reg)) {
            val_seed_origin[(size_t)reg] = i;
            val_seed_origin_known.set((size_t)reg);
        }
    }
    // BRANCH-EXCLUSIVE WRITES (#2132). This fold walks the decoded stream STRAIGHT-LINE — it models
    // no control flow at all. That is fine while every block is on one path, and silently wrong the
    // moment a shader guards its descriptor setup with a branch: the walk executes the not-taken
    // side and carries its register writes into a block that, on hardware, is only ever entered from
    // the branch. CrossWorlds' failing NGG vertex stage is exactly that shape —
    //
    //     pc=21   s_cbranch_vccz 151                  -> pc=173
    //     pc=45   s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60    (not-taken side)
    //     pc=113  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60    (not-taken side)
    //     pc=172  s_branch 47                         -> pc=220     (so pc=173 has NO fall-through)
    //     pc=173  s_load_dword     s20, s[18:19]
    //     pc=188  s_load_dwordx4   s[4:7], s[16:17], vcc_lo
    //
    // s16..s19 are the stage's two DECLARED DIRECT POINTERS, seeded above and both readable. The
    // two dwordx16 loads overwrite them with float uniforms (0x3f800000 = 1.0f and zeros), so by
    // pc=173 the walk dereferences 0 and reports a "null bindless-table pointer" that no writer ever
    // wrote and no guest ever had. Four investigations hunted that phantom writer.
    //
    // THE QUALIFYING SHAPE, stated as what the code actually tests (#2202 review, B1/B2):
    //
    //   * `ins` is the COMPACTED fold stream (`retain_fold_instructions`), which drops ordinary
    //     VALU/EXP/DS/FLAT while preserving original PCs. So "the previous element" is the previous
    //     RETAINED instruction, and a dropped VALU block between it and the target would be
    //     invisible. The check is therefore PHYSICAL: `prev.pc + prev.len_dwords == ins[k].pc`
    //     proves prev immediately precedes the target in the real program, whatever was compacted
    //     away. Without it a target with a genuine fall-through predecessor can qualify, and the
    //     restore then installs a *known* wrong value — a worse failure than the bug being fixed.
    //   * prev must be an UNCONDITIONAL `s_branch`, so no fall-through edge enters the target.
    //   * EXACTLY ONE branch in the DECODED STREAM targets it, counted over BOTH directions. A
    //     backward edge into the target is a second predecessor and disqualifies it. "Decoded
    //     stream" rather than "program" is deliberate: the decode stops at the first s_endpgm, so
    //     anything past it is not scanned. B1 existed because a property of a stream was described
    //     as a property of a program, so the distinction is spelled out rather than assumed.
    //   * The program contains no indirect control transfer at all; one makes the CFG
    //     unrepresentable by any scan over SOPP displacements, so the rule declines to fire.
    //
    // This is the same standard `guarded_bvh_use` above already holds itself to, and for the same
    // reason its comment gives: observing an earlier forward branch does not establish dominance.
    //
    // The target's only predecessor is then the branch at B, and since a SOPP branch writes no
    // register, THE STATE AT THE TARGET IS EXACTLY THE STATE AT B. So the rule is a whole-state
    // save at B and restore at the target, rather than a per-register rollback: every write in
    // (B, target) belongs to a path that cannot reach the target, including the ones that live in
    // `scc`, the scalar spill slots and the vector index-mode table, which a register-only rollback
    // left behind. Published uses (`out`, `srt_uses`) are deliberately NOT rolled back: an
    // instruction inside the skipped region is genuinely reachable on its own path and the
    // recompiler still needs its descriptor.
    //
    // What this does NOT claim: that the state at B is itself right. If branches precede B the walk
    // may already be a chimera, and this only makes the target agree with the walk's own state at B.
    // That is strictly closer to one real path than mixing both arms of a branch.
    // CONFIDENCE: HIGH on the CFG rule; the shape is proved from the guest's own encodings.
    //
    // Kept allocation-free for the overwhelmingly common shader (no branch at all): this fold runs
    // thousands of times per submit. The scan is linear — one pass collecting every branch target,
    // one sort, then a binary search per candidate.
    std::vector<std::pair<uint32_t, uint32_t>> restore_at;   // (target pc, its sole predecessor branch pc)
    size_t restore_cursor = 0;
    {
        // PROSPER_NO_BRANCH_EXCLUSIVE=<any value> restores the old straight-line walk. It exists so
        // the A/B that established this — CrossWorlds' two dropped vertex pipelines and its
        // `[mubuf-unresolved]` line appear with it set and are absent without it — stays reproducible
        // for the next reader, in the same spirit as PROSPER_UD_TAIL_ALIGN. It must stay OFF in every
        // normal run: with it set the fold reports descriptors built from a path the wave cannot be on.
        static const bool branch_exclusive_disabled =
            std::getenv("PROSPER_NO_BRANCH_EXCLUSIVE") != nullptr;
        bool any_branch = false;
        for (const Rdna2Inst& in : ins)
            if (sopp_is_branch(in)) { any_branch = true; break; }
        if (any_branch && !branch_exclusive_disabled && !has_indirect_control_flow(ins)) {
            // Every branch edge, both directions — the predecessor tally is only a tally if it
            // counts them all.
            std::vector<std::pair<uint32_t, uint32_t>> edges;   // (target, branch pc)
            for (const Rdna2Inst& in : ins) {
                if (!sopp_is_branch(in)) continue;
                const int64_t t = sopp_branch_target(in);
                if (t < 0 || t > (int64_t)UINT32_MAX) continue;
                edges.emplace_back((uint32_t)t, in.pc);
            }
            std::sort(edges.begin(), edges.end());
            for (size_t k = 1; k < ins.size(); ++k) {
                const Rdna2Inst& prev = ins[k - 1];
                const uint32_t t = ins[k].pc;
                if (!sopp_is_unconditional_branch(prev)) continue;      // a fall-through edge exists
                if (prev.pc + prev.len_dwords != t) continue;           // ...unless prev physically ends here
                const auto lo = std::lower_bound(edges.begin(), edges.end(),
                                                 std::make_pair(t, uint32_t{0}));
                const auto hi = std::upper_bound(edges.begin(), edges.end(),
                                                 std::make_pair(t, UINT32_MAX));
                if (std::distance(lo, hi) != 1) continue;               // 0 or >=2 predecessors
                const uint32_t source = lo->second;
                if (source >= t) continue;                              // a lone backward edge: no region
                restore_at.emplace_back(t, source);
            }
        }
    }
    // The interpreter state that a restore has to put back. Everything the walk mutates and that can
    // be read after the target — deliberately excluding the published uses, and excluding
    // `next_null_chain_origin`, which is a monotonic id allocator: rewinding it would let two
    // distinct null chains share an origin, which is worse than leaking ids.
    struct FoldStateSnapshot {
        std::array<uint32_t, kFoldSgprs> val;
        std::bitset<kFoldSgprs> val_known;
        std::array<uint32_t, kFoldSgprs> val_seed_origin;
        std::bitset<kFoldSgprs> val_seed_origin_known;
        std::array<uint32_t, kFoldSgprs> val_srt_key;
        std::bitset<kFoldSgprs> val_srt_key_known;
        std::unordered_map<uint32_t, uint32_t> scalar_spill_slots;
        std::array<std::array<uint32_t, 4>, kFoldSgprs> descr;
        std::bitset<kFoldSgprs> descr_known;
        std::array<FoldTableSource, kFoldSgprs> table_source;
        std::bitset<kFoldSgprs> table_source_known;
        std::array<uint32_t, kFoldSgprs> descr_key;
        std::bitset<kFoldSgprs> descr_key_known;
        std::array<std::array<uint32_t, 8>, kFoldSgprs> descr8;
        std::bitset<kFoldSgprs> descr8_known;
        std::array<uint32_t, kFoldSgprs> descr8_key;
        std::bitset<kFoldSgprs> descr8_key_known;
        std::array<uint64_t, kFoldSgprs> descriptor_word_source_addr;
        std::bitset<kFoldSgprs> descriptor_word_source_known;
        std::array<uint32_t, kFoldSgprs> null_chain_origin;
        std::bitset<kFoldSgprs> null_chain_known;
        std::array<OptionalNullRole, kFoldSgprs> optional_null_role;
        uint32_t null_count_carry_origin;
        std::array<uint32_t, kFoldSgprs> bvh_build_origin;
        std::array<BvhBuildRole, kFoldSgprs> bvh_build_role;
        uint32_t bvh_count_carry_origin;
        std::bitset<kFoldSgprs> reloaded;
        std::array<FoldMask, kFoldSgprs> mask_state;
        std::array<VertexFetchIndexMode, kFoldVgprs> vector_index_mode;
        int scc;
    };
    // One slot per qualifying target, indexed rather than searched: each target has exactly one
    // predecessor branch, so `restore_slot[i]` is the slot that `restore_at[i]`'s branch fills.
    // `.first` is "this slot has been written", so a target reached before its branch (impossible for
    // a forward edge, but cheap to be safe about) restores nothing rather than garbage.
    std::vector<std::pair<bool, FoldStateSnapshot>> saved_at_branch;
    std::vector<size_t> restore_slot;
    std::unordered_map<uint32_t, size_t> save_slot_of_pc;
    if (!restore_at.empty()) {
        restore_slot.reserve(restore_at.size());
        saved_at_branch.reserve(restore_at.size());
        for (const auto& [target, branch_pc] : restore_at) {
            (void)target;
            auto existing = save_slot_of_pc.find(branch_pc);
            if (existing == save_slot_of_pc.end()) {
                existing = save_slot_of_pc.emplace(branch_pc, saved_at_branch.size()).first;
                saved_at_branch.emplace_back();
                saved_at_branch.back().first = false;
            }
            restore_slot.push_back(existing->second);
        }
    }
    auto capture_fold_state = [&]() {
        FoldStateSnapshot s;
        s.val = val; s.val_known = val_known;
        s.val_seed_origin = val_seed_origin; s.val_seed_origin_known = val_seed_origin_known;
        s.val_srt_key = val_srt_key; s.val_srt_key_known = val_srt_key_known;
        s.scalar_spill_slots = scalar_spill_slots;
        s.descr = descr; s.descr_known = descr_known;
        s.table_source = table_source; s.table_source_known = table_source_known;
        s.descr_key = descr_key; s.descr_key_known = descr_key_known;
        s.descr8 = descr8; s.descr8_known = descr8_known;
        s.descr8_key = descr8_key; s.descr8_key_known = descr8_key_known;
        s.descriptor_word_source_addr = descriptor_word_source_addr;
        s.descriptor_word_source_known = descriptor_word_source_known;
        s.null_chain_origin = null_chain_origin; s.null_chain_known = null_chain_known;
        s.optional_null_role = optional_null_role;
        s.null_count_carry_origin = null_count_carry_origin;
        s.bvh_build_origin = bvh_build_origin; s.bvh_build_role = bvh_build_role;
        s.bvh_count_carry_origin = bvh_count_carry_origin;
        s.reloaded = reloaded; s.mask_state = mask_state;
        s.vector_index_mode = vector_index_mode; s.scc = scc;
        return s;
    };
    auto restore_fold_state = [&](const FoldStateSnapshot& s) {
        val = s.val; val_known = s.val_known;
        val_seed_origin = s.val_seed_origin; val_seed_origin_known = s.val_seed_origin_known;
        val_srt_key = s.val_srt_key; val_srt_key_known = s.val_srt_key_known;
        scalar_spill_slots = s.scalar_spill_slots;
        descr = s.descr; descr_known = s.descr_known;
        table_source = s.table_source; table_source_known = s.table_source_known;
        descr_key = s.descr_key; descr_key_known = s.descr_key_known;
        descr8 = s.descr8; descr8_known = s.descr8_known;
        descr8_key = s.descr8_key; descr8_key_known = s.descr8_key_known;
        descriptor_word_source_addr = s.descriptor_word_source_addr;
        descriptor_word_source_known = s.descriptor_word_source_known;
        null_chain_origin = s.null_chain_origin; null_chain_known = s.null_chain_known;
        optional_null_role = s.optional_null_role;
        null_count_carry_origin = s.null_count_carry_origin;
        bvh_build_origin = s.bvh_build_origin; bvh_build_role = s.bvh_build_role;
        bvh_count_carry_origin = s.bvh_count_carry_origin;
        reloaded = s.reloaded; mask_state = s.mask_state;
        vector_index_mode = s.vector_index_mode; scc = s.scc;
    };
    if (explicit_ngg_index_provenance) {
        // s3 is the merged GS/ES wave info: s3[7:0] is the active ES-vertex count and s3[15:8]
        // the GS-primitive count.  The Vulkan vertex shell represents one active ES vertex and no
        // GS primitive, so s3=1.  Fetch prologues use this value to choose and patch V#
        // descriptors before their MUBUF loads; leaving it unknown made the dynamic resource walk
        // drop otherwise valid scene-geometry fetches.
        // WARNING: this does NOT match recompile_vertex for most programs, and this comment used to
        // claim it did.  There, s3=1 is seeded only under exact_ngg_projection, i.e. only for the
        // seven wrappers is_astro_bot_ngg_one_lane_wrapper() admits; every other NGG program gets
        // 0x40004040|(wave<<24).  The two models differ in exactly the high bits a prologue reads
        // when it builds a SOFFSET from s3 — 0 here against 0x40000000 there.  #2072.
        set_value(3, 1u);
    }

    // A direct sharp lives in the initial user-data SGPR block rather than arriving through an
    // s_load. It remains a usable load-time descriptor only while none of its SGPRs has subsequently
    // been reloaded from memory. ALU descriptor patches deliberately retain the same pre-patch
    // snapshot semantics as `descr`/`descr8` and the seed-V# fallback below.
    auto untouched_seed_range = [&](int first, int count) {
        if (!user_sgprs || first < (int)user_sgpr_base ||
            first + count > (int)(user_sgpr_base + nsgpr)) return false;
        for (int r = first; r < first + count; r++)
            if (!valid_reg(r) || reloaded.test((size_t)r)) return false;
        return true;
    };

    auto known = [&](int r, uint32_t& v) {
        if (!valid_reg(r) || !val_known.test((size_t)r)) return false;
        v = val[(size_t)r];
        return true;
    };
    auto null_origin = [&](int r) -> uint32_t {
        return valid_reg(r) && null_chain_known.test((size_t)r)
            ? null_chain_origin[(size_t)r] : 0u;
    };
    auto mark_null_origin = [&](int r, uint32_t origin) {
        if (!origin || !valid_reg(r)) return;
        null_chain_origin[(size_t)r] = origin;
        null_chain_known.set((size_t)r);
    };
    auto mark_optional_null_role = [&](int r, OptionalNullRole role) {
        if (role != OptionalNullRole::None && valid_reg(r))
            optional_null_role[(size_t)r] = role;
    };
    auto mark_bvh_build = [&](int r, uint32_t origin, BvhBuildRole role) {
        if (!origin || role == BvhBuildRole::None || !valid_reg(r)) return;
        bvh_build_origin[(size_t)r] = origin;
        bvh_build_role[(size_t)r] = role;
    };
    auto operand_bvh_build = [&](const Operand& operand, BvhBuildRole role,
                                 uint32_t& origin) {
        if ((operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) ||
            !valid_reg(operand.value) || bvh_build_role[(size_t)operand.value] != role)
            return false;
        origin = bvh_build_origin[(size_t)operand.value];
        return origin != 0;
    };
    auto operand_null_origin = [&](const Operand& operand) -> uint32_t {
        if ((operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
            valid_reg(operand.value))
            return null_origin(operand.value);
        return 0u;
    };
    auto operand_optional_null_role = [&](const Operand& operand) -> OptionalNullRole {
        if ((operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
            valid_reg(operand.value))
            return optional_null_role[(size_t)operand.value];
        return OptionalNullRole::None;
    };
    auto unchanged_seed_range = [&](int first, int count) {
        if (!untouched_seed_range(first, count)) return false;
        for (int r = first; r < first + count; ++r) {
            uint32_t current = 0;
            if (!known(r, current) ||
                current != user_sgprs[r - (int)user_sgpr_base]) return false;
        }
        return true;
    };
    auto consecutive_seed_copy_range = [&](int first, int count) {
        if (!user_sgprs || count <= 0 || !valid_reg(first) || !valid_reg(first + count - 1) ||
            !val_seed_origin_known.test((size_t)first)) return false;
        const uint32_t first_origin = val_seed_origin[(size_t)first];
        if (first_origin + (uint32_t)count > nsgpr) return false;
        for (int k = 0; k < count; ++k) {
            const int reg = first + k;
            uint32_t current = 0;
            if (!val_seed_origin_known.test((size_t)reg) ||
                val_seed_origin[(size_t)reg] != first_origin + (uint32_t)k ||
                !known(reg, current) || current != user_sgprs[first_origin + (uint32_t)k])
                return false;
        }
        return true;
    };
    // Resolve an ALU source operand to a concrete value (SGPR / inline int / literal / a vcc Special).
    // vcc_lo/hi (106/107) are written by ALU dsts as SGPR 106/107 but read back as Special operands with
    // the same field value, so map them onto the same val[] keys. Other Specials (EXEC/M0/...) stay unknown.
    auto srcval = [&](const Operand& o, uint32_t& v) -> bool {
        switch (o.kind) {
            case OperandKind::SGPR:      return known(o.value, v);
            case OperandKind::InlineInt: v = (uint32_t)o.value; return true;
            case OperandKind::Special:
                if (o.value == 106 || o.value == 107) return known(o.value, v);
                // SCC (source field 253) is the one architectural bit this fold already MAINTAINS:
                // `scc` is written by every modelled SOP2/SOPC/SOPK above and consumed by
                // s_cselect. Reading it as a SOURCE was simply never wired up, so an instruction
                // whose operand the fold holds in hand still gave up. Measured cost: Sonic
                // Frontiers' most-repeated compute blocker is `s_or_b32 vcc_lo, scc, vcc_hi`
                // (word 886a6bfd) on three separate programs -- the OPCODE is modelled, and the
                // fold failed on the operand (#2790).
                //
                // -1 means unknown and must stay unreadable; SCC is a single bit, so a known value
                // is exactly 0 or 1.
                if (o.value == 253) {
                    if (scc < 0) return false;
                    v = static_cast<uint32_t>(scc);
                    return true;
                }
                return false;
            case OperandKind::Literal:   v = 0; return false;   // literal is in in.literal; handled per-op
            default: return false;
        }
    };
    auto srcmask = [&](const Operand& o) -> FoldMask {
        if (o.kind == OperandKind::Special && (o.value == 126 || o.value == 127))
            return FoldMask::All;       // EXEC is full at this fetch-prologue boundary
        if (o.kind == OperandKind::InlineInt)
            return o.value == 0 ? FoldMask::None
                 : o.value == -1 ? FoldMask::All : FoldMask::Unknown;
        if ((o.kind == OperandKind::SGPR || o.kind == OperandKind::Special) &&
            valid_reg(o.value))
            return mask_state[(size_t)o.value];
        return FoldMask::Unknown;
    };

    // A zero mip is useful only when it is a property of the exact use, not a global VGPR guess.
    // Track plain scalar-to-vector zero moves inside one basic block and clear the proof on every
    // block boundary or possible overlapping write. Branch targets and branch fall-throughs both
    // begin blocks; indirect control flow makes this finite edge inventory incomplete and disables
    // the proof for the whole program. Any EXEC writer also disables later proofs globally: without
    // a mask dataflow model, a subsequent v_mov may still execute on only a subset of lanes.
    const bool zero_mip_cfg_known = !has_indirect_control_flow(ins);
    std::set<uint32_t> zero_mip_block_starts;
    if (!ins.empty()) zero_mip_block_starts.insert(ins.front().pc);
    if (zero_mip_cfg_known) {
        for (const Rdna2Inst& candidate : ins) {
            if (!sopp_is_branch(candidate)) continue;
            const int64_t target = sopp_branch_target(candidate);
            if (target >= 0 && target <= static_cast<int64_t>(UINT32_MAX))
                zero_mip_block_starts.insert(static_cast<uint32_t>(target));
            zero_mip_block_starts.insert(candidate.pc + candidate.len_dwords);
        }
    }
    std::bitset<256> same_block_zero_vgprs;
    uint32_t current_zero_mip_block = UINT32_MAX;
    bool zero_mip_exec_pristine = true;

    for (size_t instruction_index = 0; instruction_index < ins.size(); ++instruction_index) {
        const auto& in = ins[instruction_index];
        watch_pc = in.pc;
        watch_w0 = in.words[0]; watch_w1 = in.len_dwords > 1 ? in.words[1] : 0u;
        watch_len = in.len_dwords;
        if (in.is_end) break;
        uint32_t zero_mip_block = UINT32_MAX;
        if (zero_mip_cfg_known) {
            // The retained fold stream may omit the first physical instruction of a block (EXP is
            // a common example). Classify by the greatest start not after this PC instead of
            // waiting to observe the start itself, or a pre-branch proof could leak across a block
            // whose first instruction was compacted out.
            auto block = zero_mip_block_starts.upper_bound(in.pc);
            if (block != zero_mip_block_starts.begin())
                zero_mip_block = *std::prev(block);
        }
        if (!zero_mip_cfg_known ||
            (instruction_index != 0 && zero_mip_block != current_zero_mip_block))
            same_block_zero_vgprs.reset();
        current_zero_mip_block = zero_mip_block;
        // A plain v_mov is still lane-predicated by EXEC. Without tracking the active mask and its
        // later restoration, no pre-mutation all-lanes zero fact can survive any explicit or
        // implicit EXEC write (including every v_cmpx encoding).
        if (rdna2_instruction_may_change_exec(in)) {
            same_block_zero_vgprs.reset();
            zero_mip_exec_pristine = false;
        }
        uint32_t mip_vgpr = 0;
        const bool proven_zero_mip_at_use =
            rdna2_mimg_zero_mip_shape(in, &mip_vgpr) &&
            same_block_zero_vgprs.test(mip_vgpr);
        // Why the proof failed, at the site that knows. `[mimg-mip]` downstream reports
        // `proven_zero_mip=0` and stops there, which is the bool this line produced -- it cannot say
        // whether the mip register was never written in this block, was written from a scalar the
        // fold could not evaluate, or was discarded by an EXEC write. Those are three different
        // pieces of work and the difference is only visible here. Deduped per pc, ungated for the
        // same reason the downstream line is: PROSPER_DBG desyncs the routed repro.
        if (rdna2_mimg_zero_mip_shape(in, &mip_vgpr)) {
            // Dedup by (PROGRAM, pc). A pc-only key collides across programs -- every shader has a
            // pc=16 -- so the first program to reach a pc silently speaks for every later one, and
            // the line then describes a kernel the reader is not looking at.
            static std::mutex why_mutex;
            static std::set<std::pair<const uint32_t*, uint32_t>> why_reported;
            bool first = false;
            {
                std::lock_guard<std::mutex> lock(why_mutex);
                first = why_reported.insert({code, in.pc}).second;
            }
            if (first)
                std::fprintf(stderr,
                             "[mimg-mip-why] program=0x%llx pc=%u proven_at_use=%d mip_vgpr=v%u "
                             "in_zero_set=%d exec_pristine=%d cfg_known=%d\n",
                             (unsigned long long)(uintptr_t)code, in.pc,
                             (int)proven_zero_mip_at_use, mip_vgpr,
                             mip_vgpr < 256 ? (int)same_block_zero_vgprs.test(mip_vgpr) : -1,
                             (int)zero_mip_exec_pristine, (int)zero_mip_cfg_known);
        }
        // #2132. RESTORE FIRST, THEN SAVE — the order is load-bearing and getting it wrong
        // reproduces the very defect this rule exists to fix (#2202 review, B3). One instruction can
        // be BOTH a qualifying target and the sole-predecessor branch of a later target: a block
        // whose first instruction is a branch, which is ordinary compiler output. Saving first would
        // store the pre-restore chimera and reinstate it one block later, so the second target would
        // be handed exactly the mixed-path state the rule is meant to remove.
        //
        // Arrival at a target: put back the state its only predecessor left, discarding everything
        // the walk did on the path that branch skips. `restore_at` is ascending by target, as is the
        // walk, so a cursor suffices.
        while (restore_cursor < restore_at.size() && restore_at[restore_cursor].first < in.pc)
            ++restore_cursor;
        if (restore_cursor < restore_at.size() && restore_at[restore_cursor].first == in.pc) {
            const size_t slot = restore_slot[restore_cursor];
            if (slot < saved_at_branch.size() && saved_at_branch[slot].first) {
                restore_fold_state(saved_at_branch[slot].second);
                if (trc)
                    fprintf(stderr, "[dyntrace]   branch-exclusive: pc=%u restored the fold state to "
                                    "pc=%u (its only predecessor); every write in between is on the "
                                    "path that branch skips\n",
                            in.pc, restore_at[restore_cursor].second);
            }
        }
        // ...and only now record the state at this instruction, if it is itself some later target's
        // only predecessor. Each source appears at most once in `restore_at`, so this is an indexed
        // store rather than a scan.
        if (!save_slot_of_pc.empty()) {
            const auto slot = save_slot_of_pc.find(in.pc);
            if (slot != save_slot_of_pc.end()) {
                saved_at_branch[slot->second].first = true;
                saved_at_branch[slot->second].second = capture_fold_state();
            }
        }
        if (gta5_null_raw_store_guard && in.pc == 42u) {
            uint32_t pointer_lo = 0, pointer_hi = 0;
            gta5_null_pointer_at_guard = known(2, pointer_lo) && known(3, pointer_hi) &&
                                         pointer_lo == 0u && pointer_hi == 0u;
        }
        const bool scalar_spill = in.fmt == Rdna2Format::VOP3 &&
                                  (in.opcode == 0x360 || in.opcode == 0x361);
        const bool vector_select =
            (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101) ||
            (in.fmt == Rdna2Format::VOP2 && in.opcode == 0x01);
        // Buffer instructions read VADDR before writing their payload destination. Some prologues
        // intentionally reuse the same VGPR for both (Messenger's v0 attribute fetch), so retain the
        // source provenance before the generic destination-write invalidation below.
        VertexFetchIndexMode fetch_index_mode_before_write = VertexFetchIndexMode::Automatic;
        if (explicit_ngg_index_provenance &&
            (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
            in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 &&
            in.src[0].value < static_cast<int>(kFoldVgprs))
            fetch_index_mode_before_write = vector_index_mode[(size_t)in.src[0].value];
        if (!scalar_spill && in.dst.kind == OperandKind::VGPR) {
            // Any ordinary write replaces the whole vector result. Drop scalar values previously
            // packed into that VGPR's lanes rather than restoring stale descriptor words later.
            for (uint32_t lane = 0; lane < 64; ++lane)
                scalar_spill_slots.erase(((uint32_t)in.dst.value << 6) | lane);
            // Address selectors are scalar VGPR values; ordinary VALU writes replace their exact
            // destination. Multi-dword memory results are payload here, not later ABI selectors.
            if (explicit_ngg_index_provenance && !vector_select && in.dst.value >= 0 &&
                in.dst.value < static_cast<int>(kFoldVgprs)) {
                vector_index_mode[(size_t)in.dst.value] = VertexFetchIndexMode::Shader;
            }
        }
        if (in.dst.kind == OperandKind::VGPR && in.dst.value >= 0) {
            // Invalidate the exact decoded result range. GTA V writes v0 between its v2=zero
            // definition and IMAGE_LOAD_MIP; treating that scalar VOP2 as a four-dword memory
            // payload erased the unrelated live mip proof. Wide MIMG/buffer/VALU results still
            // clear every register in their shared audited result-width inventory.
            if (in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32) {
                // v_movreld_b32 writes VGPR[VDST+M0]. M0 is not tracked by this proof, so no
                // individual zero fact is safe across the dynamic destination write. Keep this
                // separate from the consecutive-width inventory; CFG phi discovery models the
                // same instruction dynamically through loop_written_regs.
                same_block_zero_vgprs.reset();
            } else {
                const uint32_t write_count = rdna2_vgpr_write_count(in);
                for (int reg = in.dst.value;
                     reg < in.dst.value + static_cast<int>(write_count) && reg < 256; ++reg)
                    same_block_zero_vgprs.reset(static_cast<size_t>(reg));
                const int tfe_status = rdna2_tfe_status_vgpr(in);
                if (tfe_status >= 0 && tfe_status < 256)
                    same_block_zero_vgprs.reset(static_cast<size_t>(tfe_status));
            }
        }
        if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x01u &&
            in.len_dwords == 1u && !in.has_modifier &&
            in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 && in.dst.value < 256 &&
            in.src[0].kind == OperandKind::SGPR) {
            uint32_t scalar = 1;
            if (zero_mip_exec_pristine && known(in.src[0].value, scalar) && scalar == 0u)
                same_block_zero_vgprs.set(static_cast<size_t>(in.dst.value));
        }
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == kSop1OpcodeMovB32) {            // s_mov_b32
                    uint32_t v, source_key = 0;
                    uint64_t source_descriptor_word_addr = 0;
                    const uint32_t source_null_origin = operand_null_origin(in.src[0]);
                    const bool source_key_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        val_srt_key_known.test((size_t)in.src[0].value) &&
                        (source_key = val_srt_key[(size_t)in.src[0].value], true);
                    uint32_t source_origin = 0;
                    const bool source_origin_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        val_seed_origin_known.test((size_t)in.src[0].value) &&
                        (source_origin = val_seed_origin[(size_t)in.src[0].value], true);
                    const bool source_descriptor_word_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        descriptor_word_source_known.test((size_t)in.src[0].value) &&
                        (source_descriptor_word_addr =
                             descriptor_word_source_addr[(size_t)in.src[0].value], true);
                    if (in.src[0].kind == OperandKind::Literal ? (v = in.literal, true) : srcval(in.src[0], v)) {
                        set_value(in.dst.value, v);
                        if (source_key_known && valid_reg(in.dst.value)) {
                            val_srt_key[(size_t)in.dst.value] = source_key;
                            val_srt_key_known.set((size_t)in.dst.value);
                        }
                        if (source_origin_known && valid_reg(in.dst.value)) {
                            val_seed_origin[(size_t)in.dst.value] = source_origin;
                            val_seed_origin_known.set((size_t)in.dst.value);
                        }
                        if (source_descriptor_word_known && valid_reg(in.dst.value)) {
                            descriptor_word_source_addr[(size_t)in.dst.value] =
                                source_descriptor_word_addr;
                            descriptor_word_source_known.set((size_t)in.dst.value);
                        }
                    } else forget(in.dst.value);
                    mark_null_origin(in.dst.value, source_null_origin);
                    if (in.dst.kind == OperandKind::SGPR &&
                        in.src[0].kind == OperandKind::Literal &&
                        in.literal == kGtaOptionalBufferConfigWord)
                        mark_optional_null_role(in.dst.value, OptionalNullRole::Config);
                } else if (in.opcode == 0x34) {                 // s_abs_i32
                    // This is a 32-bit destination. Treating every unmodeled SOP1 as a possible
                    // 64-bit pair write erased the untouched adjacent SGPR; Astro Bot keeps the
                    // high half of a descriptor-table pointer there immediately before an x8 load.
                    // Compute abs in unsigned arithmetic so INT_MIN remains 0x80000000 without C++
                    // signed-overflow undefined behaviour. Arithmetic deliberately drops seed/SRT
                    // provenance through set_value().
                    uint32_t v = 0;
                    const bool source_known = in.src[0].kind == OperandKind::Literal
                        ? (v = in.literal, true) : srcval(in.src[0], v);
                    if (source_known)
                        set_value(in.dst.value, static_cast<int32_t>(v) < 0 ? 0u - v : v);
                    else
                        forget(in.dst.value);
                } else if ((in.opcode == 0x1c || in.opcode == 0x1d) &&
                           in.dst.kind == OperandKind::SGPR) {  // s_bitset{0,1}_b32
                    // These are in-place read/modify/write operations: SDST is both the old value
                    // and destination, while SRC0 is only the bit index. Astro sets the descriptor's
                    // SORT bit after extracting a zero BVH base, so preserve the exact null-root
                    // provenance while applying that constant field patch.
                    uint32_t old_value = 0, bit_index = 0;
                    const uint32_t old_null_origin = null_origin(in.dst.value);
                    const uint32_t old_bvh_origin = valid_reg(in.dst.value)
                        ? bvh_build_origin[(size_t)in.dst.value] : 0u;
                    const BvhBuildRole old_bvh_role = valid_reg(in.dst.value)
                        ? bvh_build_role[(size_t)in.dst.value] : BvhBuildRole::None;
                    const OptionalNullRole old_optional_role = valid_reg(in.dst.value)
                        ? optional_null_role[(size_t)in.dst.value] : OptionalNullRole::None;
                    const bool old_known = known(in.dst.value, old_value);
                    const bool bit_known = in.src[0].kind == OperandKind::Literal
                        ? (bit_index = in.literal, true) : srcval(in.src[0], bit_index);
                    if (old_known && bit_known) {
                        const uint32_t bit = 1u << (bit_index & 31u);
                        set_value(in.dst.value, in.opcode == 0x1c
                            ? old_value & ~bit : old_value | bit);
                    } else {
                        forget(in.dst.value);
                    }
                    mark_null_origin(in.dst.value, old_null_origin);
                    if (in.opcode == kSop1OpcodeBitset1B32 && bit_known && bit_index == 18u &&
                        old_optional_role == OptionalNullRole::BaseHi)
                        mark_optional_null_role(in.dst.value, OptionalNullRole::BaseHi);
                    if (in.opcode == 0x1d && bit_known && bit_index == 31u &&
                        old_bvh_role == BvhBuildRole::BaseHi)
                        mark_bvh_build(in.dst.value, old_bvh_origin,
                                       BvhBuildRole::SortedBaseHi);
                } else if (in.opcode == 0x04 &&                 // s_mov_b64
                           in.dst.kind == OperandKind::SGPR &&
                           in.src[0].kind == OperandKind::SGPR) {
                    // Capture both source lanes before touching either destination: source and
                    // destination pairs may overlap. Only an entirely known SGPR pair is modeled;
                    // special/inline sources and partial pairs retain the previous fail-closed erase.
                    // A saved wave MASK is a separate domain from scalar data and survives even when
                    // the pair's value does not: `s_mov_b64 sDST, sSRC` copies it (see the EXEC form
                    // below for why the fetch prologue depends on this).
                    const FoldMask source_mask_state = valid_reg(in.src[0].value)
                        ? mask_state[(size_t)in.src[0].value] : FoldMask::Unknown;
                    std::array<uint32_t, 2> source_values{};
                    std::array<uint32_t, 2> source_keys{};
                    std::array<uint32_t, 2> source_origins{};
                    std::array<uint64_t, 2> source_descriptor_word_addrs{};
                    std::array<uint32_t, 2> source_null_origins{};
                    std::array<bool, 2> source_key_known{};
                    std::array<bool, 2> source_origin_known{};
                    std::array<bool, 2> source_descriptor_word_known{};
                    bool source_known = true;
                    for (int k = 0; k < 2; ++k) {
                        const int src = in.src[0].value + k;
                        source_known &= known(src, source_values[(size_t)k]);
                        source_key_known[(size_t)k] = valid_reg(src) &&
                            val_srt_key_known.test((size_t)src);
                        if (source_key_known[(size_t)k])
                            source_keys[(size_t)k] = val_srt_key[(size_t)src];
                        source_origin_known[(size_t)k] = valid_reg(src) &&
                            val_seed_origin_known.test((size_t)src);
                        if (source_origin_known[(size_t)k])
                            source_origins[(size_t)k] = val_seed_origin[(size_t)src];
                        source_descriptor_word_known[(size_t)k] = valid_reg(src) &&
                            descriptor_word_source_known.test((size_t)src);
                        if (source_descriptor_word_known[(size_t)k])
                            source_descriptor_word_addrs[(size_t)k] =
                                descriptor_word_source_addr[(size_t)src];
                        source_null_origins[(size_t)k] = null_origin(src);
                    }
                    for (int k = 0; k < 2; ++k) {
                        const int dst = in.dst.value + k;
                        if (!source_known) {
                            forget(dst);
                            continue;
                        }
                        set_value(dst, source_values[(size_t)k]);
                        if (source_key_known[(size_t)k] && valid_reg(dst)) {
                            val_srt_key[(size_t)dst] = source_keys[(size_t)k];
                            val_srt_key_known.set((size_t)dst);
                        }
                        if (source_origin_known[(size_t)k] && valid_reg(dst)) {
                            val_seed_origin[(size_t)dst] = source_origins[(size_t)k];
                            val_seed_origin_known.set((size_t)dst);
                        }
                        if (source_descriptor_word_known[(size_t)k] && valid_reg(dst)) {
                            descriptor_word_source_addr[(size_t)dst] =
                                source_descriptor_word_addrs[(size_t)k];
                            descriptor_word_source_known.set((size_t)dst);
                        }
                        mark_null_origin(dst, source_null_origins[(size_t)k]);
                    }
                    if (valid_reg(in.dst.value))
                        mask_state[(size_t)in.dst.value] = source_mask_state;
                } else if (in.opcode == 0x04 &&                 // s_mov_b64 sDST, exec
                           in.dst.kind == OperandKind::SGPR &&
                           in.src[0].kind == OperandKind::Special &&
                           (in.src[0].value == 126 || in.src[0].value == 127)) {
                    // Saving EXEC is not a scalar-data move — the pair's VALUE stays fail-closed —
                    // but it does carry a wave MASK the NGG fetch prologue later consumes. R-Type
                    // Delta's AGC prologue saves EXEC once and then selects vertex-vs-instance
                    // indexing per attribute through that saved copy
                    // (`s_cselect_b64 sSEL, sSAVED, 0` -> `v_cndmask_b32 vIDX, v8, v5, sSEL`).
                    // Losing the mask made s_cselect_b64 fold to Unknown, which classified a
                    // per-vertex attribute as shader-computed. Two things then go wrong at once, and
                    // each alone still reads zero: every vertex takes the *instance* index (0 for a
                    // one-instance draw), and the recompiler retains the load's runtime SOFFSET,
                    // which for that shader is `s3 & 0xfff80000`. This fold seeds s3 = 1 below (see
                    // set_value(3, 1u)), so that term is 0 and fetch_off is 0; recompile_vertex seeds
                    // s3 = 1 only for the seven wrappers is_astro_bot_ngg_one_lane_wrapper() admits,
                    // and otherwise 0x40004040|(wave<<24), whose
                    // masked value is 0x40000000. The shader therefore adds ~1 GiB to a 96-byte
                    // descriptor and robustBufferAccess returns 0. That s3 mismatch is #2072 (the
                    // vertex-rate bit's separate SMEM divergence is #2069); NEITHER is repaired here.
                    // This only removes the shader's address expression from the path for every
                    // attribute the fold classifies Vertex or Instance.
                    //
                    // `All` is this fold's standing assumption about EXEC, not a new one: `srcmask()`
                    // already answers `All` for a directly named EXEC operand, because this walk is a
                    // linearised prologue scan with no EXEC model at all. The one thing that changes
                    // here is that the assumption becomes durable in a register and copyable, so
                    // record it: if a future prologue narrows EXEC before saving it, this fold will
                    // still read the saved copy as all-lanes. That is tolerable for the only consumer
                    // — selecting *which ABI register* holds the element index is a wave-uniform
                    // question either way — and would need an explicit EXEC model to do better.
                    // CONFIDENCE: HIGH.
                    forget(in.dst.value);
                    forget(in.dst.value + 1);
                    if (valid_reg(in.dst.value))
                        mask_state[(size_t)in.dst.value] = FoldMask::All;
                } else if (in.dst.kind == OperandKind::SGPR &&
                           sop1_opcode_writes_exec_b32(in.opcode)) {
                    // The Wave32 SAVEEXEC/WREXEC encodings write exactly one physical SGPR.
                    // Treating every unmodeled SOP1 as a possible pair write erased an untouched
                    // descriptor word when the destination immediately preceded its SRSRC. The
                    // scalar mask value remains outside this fold, but only SDST is unknown.
                    forget(in.dst.value);
                } else if (in.dst.kind == OperandKind::SGPR) {
                    // Not a modeled scalar move -> the dest is unknown. Erase the PAIR: 64-bit SOP1 ops
                    // (s_getpc_b64, s_and/or/xor/not_b64, s_*_saveexec_b64, …) write
                    // S[dst:dst+1], so leaving a stale "known" val[dst+1] let a later instruction fold a
                    // confidently-wrong 64-bit base/offset -> a wrong V#/T# read from the wrong guest
                    // address (#460). Over-erasing dst+1 for a 32-bit SOP1 only loses a fold opportunity
                    // (never fabricates a value) — matching the SOP2 s_bfe_u64 pair-erase.
                    forget(in.dst.value);
                    forget(in.dst.value + 1);
                }
                // Several SOP1 ops write SCC (s_abs_i32, s_not_b32, s_and_saveexec_*, …). Only the moves
                // (s_mov_b32 0x03 / s_mov_b64 0x04) are known not to — anything else invalidates the
                // tracked SCC, or a later s_cselect folds with a stale compare result.
                if (in.opcode != kSop1OpcodeMovB32 && in.opcode != kSop1OpcodeMovB64) {
                    scc = -1;
                    report_scc("invalidated");
                    null_count_carry_origin = 0;
                    bvh_count_carry_origin = 0;
                }
                break;
            case Rdna2Format::SOP2: {
                // s_cselect_b64 is commonly used to make an all-lanes/zero mask for the following
                // NGG v_cndmask ABI selector. It does not produce ordinary scalar data, but SCC and
                // both source masks are wave-uniform here, so retain the exact All/None result.
                if (in.opcode == 0x0B) {
                    const FoldMask m0 = srcmask(in.src[0]);
                    const FoldMask m1 = srcmask(in.src[1]);
                    const FoldMask result = scc < 0 ? FoldMask::Unknown : (scc ? m0 : m1);
                    forget(in.dst.value);
                    forget(in.dst.value + 1);
                    if (valid_reg(in.dst.value)) mask_state[(size_t)in.dst.value] = result;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   SOP2 pc=%u s_cselect_b64 dst=s%d scc=%d mask=%u\n",
                                in.pc, in.dst.value, scc, (unsigned)result);
                    break;
                }
                uint32_t a, c; bool ka, kc;
                ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                int d = in.dst.value; bool ok = ka && kc; uint32_t r = 0;
                uint32_t hi64 = 0; bool wrote_pair = false;
                int next_scc = scc;
                const uint32_t null0 = operand_null_origin(in.src[0]);
                const uint32_t null1 = operand_null_origin(in.src[1]);
                const OptionalNullRole optional0 = operand_optional_null_role(in.src[0]);
                const OptionalNullRole optional1 = operand_optional_null_role(in.src[1]);
                uint32_t null0_hi = 0;
                if ((in.src[0].kind == OperandKind::SGPR ||
                     in.src[0].kind == OperandKind::Special) &&
                    valid_reg(in.src[0].value + 1))
                    null0_hi = null_origin(in.src[0].value + 1);
                auto source_bvh_state = [&](const Operand& operand, int add,
                                            uint32_t& origin, BvhBuildRole& role) {
                    const int reg = operand.value + add;
                    if ((operand.kind != OperandKind::SGPR &&
                         operand.kind != OperandKind::Special) || !valid_reg(reg)) {
                        origin = 0;
                        role = BvhBuildRole::None;
                        return;
                    }
                    origin = bvh_build_origin[(size_t)reg];
                    role = bvh_build_role[(size_t)reg];
                };
                uint32_t bvh0_origin = 0, bvh1_origin = 0, bvh0_hi_origin = 0;
                BvhBuildRole bvh0_role = BvhBuildRole::None;
                BvhBuildRole bvh1_role = BvhBuildRole::None;
                BvhBuildRole bvh0_hi_role = BvhBuildRole::None;
                source_bvh_state(in.src[0], 0, bvh0_origin, bvh0_role);
                source_bvh_state(in.src[1], 0, bvh1_origin, bvh1_role);
                source_bvh_state(in.src[0], 1, bvh0_hi_origin, bvh0_hi_role);
                const uint32_t previous_null_count_carry_origin = null_count_carry_origin;
                const uint32_t previous_bvh_count_carry_origin = bvh_count_carry_origin;
                // s_addc_u32 remains derived from the same pointer chain even when its concrete
                // carry is unknown; the taint is provenance, not a claim about the folded value.
                const bool null_chain_opcode =
                    in.opcode <= 0x04 || in.opcode == 0x0E || in.opcode == 0x10 ||
                    in.opcode == 0x12 || in.opcode == 0x1E || in.opcode == 0x20 ||
                    in.opcode == 0x26 || (in.opcode >= 0x2E && in.opcode <= 0x31) ||
                    in.opcode == 0x27 || in.opcode == 0x1F || in.opcode == 0x21 ||
                    in.opcode == 0x29;
                uint32_t propagated_null_origin = 0;
                const bool null_chain_64bit_opcode =
                    in.opcode == 0x1Fu || in.opcode == 0x21u || in.opcode == 0x29u;
                if (null_chain_64bit_opcode) {
                    // A 64-bit value is one provenance unit. Looking only at its low SGPR lets two
                    // independently loaded zero qwords be spliced with scalar moves and then
                    // collapsed onto one origin by the shift/BFE result.
                    if (null0 && null0_hi == null0 && kc)
                        propagated_null_origin = null0;
                } else if (null_chain_opcode) {
                    if (null0 && null1 && null0 == null1) propagated_null_origin = null0;
                    else if (null0 && !null1 && kc) propagated_null_origin = null0;
                    else if (null1 && !null0 && ka) propagated_null_origin = null1;
                }
                // GTA V forms the high descriptor count with `s_add_u32 -1,null_value` followed by
                // `s_addc_u32 -1,0`. The failed dereference intentionally makes the concrete SCC
                // unknown, but the carry still depends exclusively on the same proven null chain.
                // Keep this narrower than general SCC taint: accept only that exact constant pair.
                if (in.opcode == 0x04u && previous_null_count_carry_origin && ka && kc &&
                    ((a == UINT32_MAX && c == 0u) || (c == UINT32_MAX && a == 0u)))
                    propagated_null_origin = previous_null_count_carry_origin;
                if (ok) switch (in.opcode) {
                    case 0x00: {                                           // s_add_u32
                        const uint64_t sum = static_cast<uint64_t>(a) + c;
                        r = static_cast<uint32_t>(sum);
                        next_scc = static_cast<int>(sum >> 32);
                        break;
                    }
                    case 0x01:                                             // s_sub_u32
                        r = a - c; next_scc = a < c; break;
                    case 0x02: {                                           // s_add_i32
                        r = a + c;
                        next_scc = ((~(a ^ c) & (a ^ r)) >> 31) != 0;
                        break;
                    }
                    case 0x03:                                             // s_sub_i32
                        r = a - c; next_scc = (((a ^ c) & (a ^ r)) >> 31) != 0; break;
                    case 0x04: {                                           // s_addc_u32
                        if (scc < 0) { ok = false; break; }
                        const uint64_t sum = static_cast<uint64_t>(a) + c +
                                             static_cast<uint32_t>(scc);
                        r = static_cast<uint32_t>(sum);
                        next_scc = static_cast<int>(sum >> 32);
                        break;
                    }
                    case kSop2OpcodeAndB32:
                        r = a & c; next_scc = r != 0; break;
                    case kSop2OpcodeOrB32:
                        r = a | c; next_scc = r != 0; break;
                    case kSop2OpcodeXorB32:
                        r = a ^ c; next_scc = r != 0; break;
                    case kSop2OpcodeAndn2B32:
                        r = a & ~c; next_scc = r != 0; break;
                    case 0x1E:                                             // s_lshl_b32
                        r = a << (c & 31); next_scc = r != 0; break;
                    case 0x20:                                             // s_lshr_b32
                        r = a >> (c & 31); next_scc = r != 0; break;
                    case 0x26: r = a * c; break;                            // s_mul_i32
                    case 0x2E: case 0x2F: case 0x30: case 0x31: {           // s_lshl{1,2,3,4}_add_u32
                        const uint32_t shift = in.opcode - 0x2Du;
                        const uint64_t sum = (static_cast<uint64_t>(a) << shift) + c;
                        r = static_cast<uint32_t>(sum);
                        next_scc = sum >= (uint64_t{1} << 32);
                        break;
                    }
                    case kSop2OpcodePackLlB32B16:                           // s_pack_ll_b32_b16
                        r = (a & 0xffffu) | ((c & 0xffffu) << 16);
                        break;
                    case 0x27: { uint32_t off = c & 0x1f, wid = (c >> 16) & 0x7f;   // s_bfe_u32
                                 r = wid == 0 ? 0 : (wid >= 32 ? (a >> off) : ((a >> off) & ((1u << wid) - 1)));
                                 next_scc = r != 0; break; }
                    case kSop2OpcodeCselectB32: // dst = SCC ? src0 : src1
                        if (scc < 0) ok = false; else r = scc ? a : c;
                        break;
                    case 0x1F: case 0x21: {  // s_lshl_b64 / s_lshr_b64
                        uint32_t ahi = 0;
                        bool khi = false;
                        // VCC as a 64-bit SOURCE reaches here as Special(106), not SGPR — the same
                        // asymmetry `srcval` documents above (":2314"): vcc_lo/hi are WRITTEN as SGPR
                        // 106/107 and READ BACK as Specials with the same field value. `srcval`
                        // already maps the LOW half onto val[106]; this high-half read did not, so
                        // `s_lshl_b64 vcc, vcc, N` resolved its low dword and then bailed on the
                        // high one — two halves of one handler disagreeing about whether VCC is
                        // addressable.
                        //
                        // Only base 106 is accepted. A 64-bit operand names its LOW register, so
                        // VCC is 106:107; a base of 107 would index 108, which is not the pair's
                        // high half and is not a valid scalar key.
                        //
                        // This does not widen what may be folded: `known()` still has to have a
                        // value for 107, and returns false if the high dword was never tracked, in
                        // which case the handler bails exactly as before. It only stops the lookup
                        // failing for a register the fold demonstrably tracks. (#2412)
                        if (in.src[0].kind == OperandKind::SGPR ||
                            (in.src[0].kind == OperandKind::Special && in.src[0].value == 106))
                            khi = known(in.src[0].value + 1, ahi);
                        else if (in.src[0].kind == OperandKind::InlineInt) {
                            ahi = in.src[0].value < 0 ? UINT32_MAX : 0u;
                            khi = true;
                        } else if (in.src[0].kind == OperandKind::Literal) {
                            ahi = 0;
                            khi = true;
                        }
                        if (!khi) { ok = false; wrote_pair = true; break; }
                        const uint64_t src64 = static_cast<uint64_t>(a) |
                                               (static_cast<uint64_t>(ahi) << 32);
                        const uint32_t shift = c & 63u;
                        const uint64_t result = in.opcode == 0x1Fu
                            ? src64 << shift : src64 >> shift;
                        r = static_cast<uint32_t>(result);
                        hi64 = static_cast<uint32_t>(result >> 32);
                        wrote_pair = true;
                        next_scc = result != 0;
                        break;
                    }
                    case 0x29: {  // s_bfe_u64: dst[63:0] = bitfield of src0[63:0] (format patch reads a small field)
                        uint32_t off = c & 0x3f, wid = (c >> 16) & 0x7f, ahi = 0;
                        // src0's high dword (RDNA2 ISA 64-bit scalar-operand rules, #155): the next SGPR
                        // of the pair; SIGN-extension of an integer inline constant (-1..-16 read as a
                        // 64-bit operand are all-ones in the high dword); or 0 only for a 32-bit literal
                        // (zero-extended). Inline FLOAT constants read as 64-bit doubles (a different bit
                        // pattern entirely) — srcval() already leaves those unknown, so they never reach
                        // here. Only an SGPR operand may index val[value+1] — a literal's `value` is not
                        // an SGPR number. And an UNTRACKED high dword must not silently fold as 0: if the
                        // field reaches bits >= 32 the result is unknown.
                        bool khi;
                        // Special(106) = VCC read back as a 64-bit source; see the note on the
                        // s_lshl_b64/s_lshr_b64 handler above for why this is not an SGPR here.
                        if (in.src[0].kind == OperandKind::SGPR ||
                            (in.src[0].kind == OperandKind::Special && in.src[0].value == 106))
                                                                          khi = known(in.src[0].value + 1, ahi);
                        else if (in.src[0].kind == OperandKind::InlineInt) { ahi = in.src[0].value < 0 ? 0xFFFFFFFFu : 0u; khi = true; }
                        // 32-bit literal -- and, since SCC became readable, also `Special{253}`.
                        // Zero is right for both: a literal's upper half is zero-extended, and SCC
                        // is a single bit.
                        else                                               { ahi = 0; khi = true; }
                        if (!khi && wid != 0 && off + wid > 32) { ok = false; wrote_pair = true; break; }
                        uint64_t src64 = (uint64_t)a | ((uint64_t)ahi << 32);
                        uint64_t res = wid == 0 ? 0 : (wid >= 64 ? (src64 >> off) : ((src64 >> off) & (((uint64_t)1 << wid) - 1)));
                        r = (uint32_t)res; hi64 = (uint32_t)(res >> 32); wrote_pair = true;
                        next_scc = res != 0; break;
                    }
                    default: ok = false; break;                            // SCC-dependent / unmodeled -> unknown
                }
                if (trc)   // unfiltered like the SMEM/MUBUF traces (one shader walk — volume is bounded)
                    fprintf(stderr, "[dyntrace]   SOP2 pc=%u op=0x%x dst=s%d src0=%d(k%d) src1=%d(k%d) ok=%d r=0x%x\n",
                            in.pc, in.opcode, d, in.src[0].value, ka, in.src[1].value, kc, ok, r);
                scc = ok ? next_scc : -1;
                if (ok) {
                    set_value(d, r);
                    if (wrote_pair) set_value(d + 1, hi64);
                }
                // A 64-bit-dst op invalidates BOTH dwords even when its source was unknown (the
                // opcode switch may not have reached the point that marks wrote_pair).
                else    { forget(d); if (wrote_pair || in.opcode == 0x1F || in.opcode == 0x21 ||
                                         in.opcode == 0x29) forget(d + 1); }
                mark_null_origin(d, propagated_null_origin);
                if (wrote_pair || in.opcode == 0x1F || in.opcode == 0x21 || in.opcode == 0x29)
                    mark_null_origin(d + 1, propagated_null_origin);
                if (ok && in.opcode == kSop2OpcodeAddI32 &&
                    in.src[0].kind == OperandKind::SGPR && in.src[0].value == 2 &&
                    c == UINT32_MAX &&
                    ((in.src[1].kind == OperandKind::InlineInt && in.src[1].value == -1) ||
                     in.src[1].kind == OperandKind::Literal)) {
                    mark_optional_null_role(d, OptionalNullRole::RecordCount);
                } else if (ok && in.opcode == kSop2OpcodeOrB32 &&
                           ((optional0 == OptionalNullRole::BaseHi &&
                             in.src[0].kind == OperandKind::SGPR && in.src[0].value == d &&
                             in.src[1].kind == OperandKind::Literal &&
                             in.literal == kGtaOptionalBufferStrideWord) ||
                            (optional1 == OptionalNullRole::BaseHi &&
                             in.src[1].kind == OperandKind::SGPR && in.src[1].value == d &&
                             in.src[0].kind == OperandKind::Literal &&
                             in.literal == kGtaOptionalBufferStrideWord))) {
                    mark_optional_null_role(d, OptionalNullRole::BaseHi);
                }
                // Every SOP2 writes SCC, so a count carry survives only when this exact instruction
                // creates it. Capture the old origin above for the following s_addc_u32.
                null_count_carry_origin = 0;
                if (in.opcode == 0x00u && propagated_null_origin &&
                    ((null0 && kc && c == UINT32_MAX) ||
                     (null1 && ka && a == UINT32_MAX)))
                    null_count_carry_origin = propagated_null_origin;
                bvh_count_carry_origin = 0;
                if (ok) {
                    auto role_plus_constant = [&](BvhBuildRole wanted, uint32_t constant,
                                                  uint32_t& origin) {
                        if (bvh0_role == wanted && c == constant) {
                            origin = bvh0_origin;
                            return origin != 0;
                        }
                        if (bvh1_role == wanted && a == constant) {
                            origin = bvh1_origin;
                            return origin != 0;
                        }
                        return false;
                    };
                    uint32_t origin = 0;
                    if (in.opcode == 0x0Eu &&
                        role_plus_constant(BvhBuildRole::HeaderHi, 0x003fffffu, origin)) {
                        mark_bvh_build(d, origin, BvhBuildRole::HeaderHi);
                    } else if (in.opcode == 0x1Fu && c == 3u &&
                               bvh0_role == BvhBuildRole::HeaderLo &&
                               bvh0_hi_role == BvhBuildRole::HeaderHi &&
                               bvh0_origin && bvh0_origin == bvh0_hi_origin) {
                        mark_bvh_build(d, bvh0_origin, BvhBuildRole::PointerLo);
                        mark_bvh_build(d + 1, bvh0_origin, BvhBuildRole::PointerHi);
                    } else if (in.opcode == 0x29u && c == 0x00280008u &&
                               bvh0_role == BvhBuildRole::PointerLo &&
                               bvh0_hi_role == BvhBuildRole::PointerHi &&
                               bvh0_origin && bvh0_origin == bvh0_hi_origin) {
                        mark_bvh_build(d, bvh0_origin, BvhBuildRole::BaseLo);
                        mark_bvh_build(d + 1, bvh0_origin, BvhBuildRole::BaseHi);
                    } else if (in.opcode == 0x02u &&
                               role_plus_constant(BvhBuildRole::CountLo, 1u, origin)) {
                        mark_bvh_build(d, origin, BvhBuildRole::CountPlusOne);
                    } else if (in.opcode == 0x00u &&
                               role_plus_constant(BvhBuildRole::CountPlusOne,
                                                  UINT32_MAX, origin)) {
                        mark_bvh_build(d, origin, BvhBuildRole::SizeLo);
                        bvh_count_carry_origin = origin;
                    } else if (in.opcode == 0x04u &&
                               previous_bvh_count_carry_origin &&
                               ((a == UINT32_MAX && c == 0u) ||
                                (c == UINT32_MAX && a == 0u))) {
                        mark_bvh_build(d, previous_bvh_count_carry_origin,
                                       BvhBuildRole::SizeHi);
                    } else if (in.opcode == 0x0Eu &&
                               role_plus_constant(BvhBuildRole::SizeHi, 0x3ffu, origin)) {
                        mark_bvh_build(d, origin, BvhBuildRole::SizeHiMasked);
                    } else if (in.opcode == 0x10u &&
                               role_plus_constant(BvhBuildRole::SizeHiMasked,
                                                  0x81000000u, origin)) {
                        mark_bvh_build(d, origin, BvhBuildRole::DescriptorHi);
                    }
                }
                break;
            }
            case Rdna2Format::SOPC: {   // scalar compare -> SCC (feeds the format patch's s_cselect)
                null_count_carry_origin = 0;
                bvh_count_carry_origin = 0;
                uint32_t a, c;
                bool ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                bool kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                if (ka && kc) switch (in.opcode) {
                    case 0x00: scc = (a == c); break;                            // s_cmp_eq_i32
                    case 0x01: scc = (a != c); break;                            // s_cmp_lg_i32
                    case 0x02: scc = ((int32_t)a >  (int32_t)c); break;          // s_cmp_gt_i32
                    case 0x03: scc = ((int32_t)a >= (int32_t)c); break;          // s_cmp_ge_i32
                    case 0x04: scc = ((int32_t)a <  (int32_t)c); break;          // s_cmp_lt_i32
                    case 0x05: scc = ((int32_t)a <= (int32_t)c); break;          // s_cmp_le_i32
                    case 0x06: scc = (a == c); break;                            // s_cmp_eq_u32
                    case 0x07: scc = (a != c); break;                            // s_cmp_lg_u32
                    case 0x08: scc = (a >  c); break;                            // s_cmp_gt_u32
                    case 0x09: scc = (a >= c); break;                            // s_cmp_ge_u32
                    case 0x0A: scc = (a <  c); break;                            // s_cmp_lt_u32
                    case 0x0B: scc = (a <= c); break;                            // s_cmp_le_u32
                    default: scc = -1; break;
                } else scc = -1;
                report_scc("alu");
                break;
            }
            case Rdna2Format::SMEM: {
                // SBASE (src[0]) is a 2-dword pointer (s_load, op<8) or a 4-dword V# (s_buffer_load, op>=8).
                // Address = base + immediate OFFSET (in.literal) + SOFFSET register value (decoded here from
                // words[1][31:25]; the shared decoder doesn't expose it). Dword count from the opcode.
                uint32_t n = 0; bool is_buffer = in.opcode >= 8;
                switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                         case 3: n = 8; break; case 4: n = 16; break; default: n = 0; }
                int sbase = in.src[0].value, sdst = in.dst.value;
                const uint32_t null_base_lo = null_origin(sbase);
                const uint32_t null_base_hi = null_origin(sbase + 1);
                const uint32_t null_base_origin =
                    null_base_lo && null_base_lo == null_base_hi ? null_base_lo : 0u;
                uint32_t soff_field = (in.words[1] >> 25) & 0x7Fu;         // SOFFSET SGPR (125 = null)
                uint32_t soff_val = 0; bool soff_ok = true;
                if (soff_field < 106) soff_ok = known((int)soff_field, soff_val);          // SGPR
                else if (soff_field == 106 || soff_field == 107) soff_ok = known((int)soff_field, soff_val); // vcc lo/hi
                else if (soff_field == 125) soff_val = 0;                   // SGPR_NULL -> const-0 offset (ok)
                else soff_ok = false;                                       // m0(124)/exec(126,127)/reserved:
                                                                            // untracked -> mark UNKNOWN, not 0. The
                                                                            // old `else soff_val=0` claimed ok=true
                                                                            // for these, snapshotting a descriptor
                                                                            // from base+imm+0 instead of +m0/exec
                                                                            // -> wrong V#/T#, silently (#398).
                // Descriptor-table use (#294): an s_buffer_load's SBASE is a V# — if that V# was
                // snapshotted from a table load, report it as a ConstantBuffer use keyed by its load
                // immediate (matching the recompiler's sreg_srt tag). Recorded BEFORE the dest write
                // below (SBASE and SDST ranges may overlap).
                SrtUse pending_srt_use;
                bool have_pending_srt_use = false;
                bool live_sbase_vsharp_known = false;
                if (srt_uses && is_buffer) {
                    bool current_known = true, current_key_known = true;
                    uint32_t current_key = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (!valid_reg(sbase + k) || !val_known.test((size_t)(sbase + k)))
                            current_known = false;
                        if (!valid_reg(sbase + k) || !val_srt_key_known.test((size_t)(sbase + k))) {
                            current_key_known = false;
                        } else if (k == 0) {
                            current_key = val_srt_key[(size_t)sbase];
                        } else if (val_srt_key[(size_t)(sbase + k)] != current_key) {
                            current_key_known = false;
                        }
                    }
                    if (current_known) {
                        pending_srt_use.kind = 1;
                        // A V# may live directly in the entry user SGPRs without an AGC sharp or
                        // preceding s_load (Astro's title PS uses s[24:27] this way). Its four live
                        // dwords are still exact; only the table-offset provenance is absent. Publish
                        // that descriptor by the unambiguous consuming PC, matching direct MIMG/MUBUF
                        // discovery, instead of forcing the recompiler onto an unbound fallback cbuf.
                        pending_srt_use.key = current_key_known ? current_key : 0xFFFFFFFFu;
                        for (int k = 0; k < 4; ++k)
                            pending_srt_use.v4[(size_t)k] = val[(size_t)(sbase + k)];
                        pending_srt_use.use_pc = in.pc;
                        have_pending_srt_use = true;
                    } else if (valid_reg(sbase) && descr_known.test((size_t)sbase) &&
                               descr_key_known.test((size_t)sbase)) {
                        pending_srt_use.kind = 1;
                        pending_srt_use.key = descr_key[(size_t)sbase];
                        pending_srt_use.v4 = descr[(size_t)sbase];
                        pending_srt_use.use_pc = in.pc;
                        have_pending_srt_use = true;
                    }
                }
                // The zero-record decision below must be based on the four words live at this
                // instruction, never only on an older descriptor snapshot. Callers that do not
                // request SRT uses still need the known-zero result to propagate to later consumers.
                std::array<uint32_t, 4> live_sbase_vsharp{};
                DecodedBufferDescriptor live_sbase_descriptor{};
                if (is_buffer) {
                    live_sbase_vsharp_known = true;
                    for (int k = 0; k < 4; ++k)
                        live_sbase_vsharp_known &= known(sbase + k, live_sbase_vsharp[(size_t)k]);
                    if (live_sbase_vsharp_known) {
                        live_sbase_descriptor =
                            decode_buffer_descriptor(live_sbase_vsharp.data());
                        if (have_pending_srt_use) {
                            const uint64_t scalar_dwords =
                                scalar_buffer_dword_count(live_sbase_descriptor);
                            if (scalar_dwords <= UINT32_MAX)
                                pending_srt_use.scalar_buffer_dword_count =
                                    static_cast<uint32_t>(scalar_dwords);
                        }
                    }
                }
                // RAW-POINTER scalar load (#2412). `s_load_dword`/`x2` (op < 8) take SBASE as a plain
                // 64-bit pointer rather than a V#, and no use was recorded for them at all — the
                // branch above is gated on `is_buffer`. GTA V's compute shaders read their constants
                // exactly this way: `user_sgprs=6` holding three pointers and no descriptor anywhere,
                // so `by_sgpr_base_cls(..., ConstantBuffer)` has nothing to find, the load rejects as
                // `unresolved-cbuf`, and the whole program is skipped.
                //
                // The fold already proves what is needed: both SBASE dwords are const-folded, so the
                // pointer is exact and wave-uniform, and the commit below measures the exact accessed
                // span into `required_size`. Publish a key-less ConstantBuffer use keyed by the
                // consuming pc — the same per-instruction provenance direct MIMG/MUBUF discovery uses,
                // and the key the recompiler's SMEM path already tries FIRST (`by_fetch_pc(in.pc)`).
                //
                // `v4` carries only the pointer; words 2/3 stay zero so `decode_buffer_descriptor`
                // yields size 0 and the consumer takes its documented `required_size` path, which is
                // gated on exactly this key-less shape (`u.key != 0xFFFFFFFFu -> continue`).
                //
                // Deliberately USE-DRIVEN: it fires only for a scalar load the shader actually
                // performs, never for a slot that merely parses. A declaration-driven version of this
                // idea was tried in build_shader_resources and regressed the frame (#2412) precisely
                // because it lacked that condition.
                // SINGLE-DWORD loads only. `s_load_dword` reads one scalar of DATA; every wider form
                // is a pointer or descriptor fetch whose result the fold already tracks as V#/T#/BVH
                // provenance, and publishing a ConstantBuffer for those shadows the uses those paths
                // exist to produce. `dynfetch_fold` establishes the bound empirically: at `n <= 2` its
                // three BVH-root cases fail (a BVH root is a two-dword pointer load), and at `n <= 16`
                // six cases fail including the x16 T#/S# publications. This is the widest form that
                // is unambiguously data.
                if (srt_uses && !is_buffer && n == 1 &&
                    valid_reg(sbase) && valid_reg(sbase + 1) &&
                    val_known.test((size_t)sbase) && val_known.test((size_t)(sbase + 1))) {
                    const uint64_t ptr = (uint64_t)val[(size_t)sbase] |
                                         ((uint64_t)val[(size_t)(sbase + 1)] << 32);
                    if (ptr > 0x10000) {   // matches the consumer's own null/low-pointer guard
                        pending_srt_use = SrtUse{};
                        pending_srt_use.kind = 1;
                        pending_srt_use.key = 0xFFFFFFFFu;         // no SRT tag: resolved by use_pc
                        pending_srt_use.v4[0] = (uint32_t)(ptr & 0xFFFFFFFFu);
                        pending_srt_use.v4[1] = (uint32_t)(ptr >> 32);
                        pending_srt_use.v4[2] = 0;                 // size comes from required_size
                        pending_srt_use.v4[3] = 0;
                        pending_srt_use.use_pc = in.pc;
                        have_pending_srt_use = true;
                        if (trc)
                            fprintf(stderr,
                                    "[dyntrace] rawptr-use pc=%u sbase=s%d ptr=0x%llx imm=0x%x\n",
                                    in.pc, sbase, (unsigned long long)ptr, in.literal);
                    }
                }
                for (uint32_t k = 0; k < n; k++)
                    if (valid_reg(sdst + (int)k)) reloaded.set((size_t)(sdst + (int)k));
                uint64_t base = 0; bool base_ok;
                // `known()` leaves its out-param UNTOUCHED when it returns false, and both halves are
                // composed into `base` regardless of `base_ok` — so an unknown SBASE used to read
                // uninitialized stack and print it. That is undefined behaviour, and it produced a
                // diagnostic that actively lied: a half-unknown pointer rendered as a plausible guest
                // VA, because stale stack here is full of real addresses whose high dword is 0x20.
                // It cost a session's worth of chasing a table pointer that was never known (#2412).
                // Zero-initialise so an unknown half reads as an obvious 0 rather than as evidence.
                if (is_buffer) { uint32_t b0 = 0, b1 = 0; base_ok = known(sbase, b0) && known(sbase + 1, b1);
                                 base = ((uint64_t)b0 | ((uint64_t)b1 << 32)) & 0xFFFFFFFFFFFFull; }   // V#.Base48
                else { uint32_t p0 = 0, p1 = 0; base_ok = known(sbase, p0) && known(sbase + 1, p1);
                       base = (uint64_t)p0 | ((uint64_t)p1 << 32); }        // raw pointer
                if (trc) fprintf(stderr, "[dyntrace] SMEM op=0x%x %s sdst=s%d sbase=s%d base=0x%llx base_ok=%d "
                                 "soff_field=%u soff_val=0x%x soff_ok=%d imm=0x%x n=%u\n", in.opcode,
                                 is_buffer ? "bufload" : "load", sdst, sbase, (unsigned long long)base, base_ok,
                                 soff_field, soff_val, soff_ok, in.literal, n);
                if (n != 0 && live_sbase_vsharp_known) {
                    if (sbuffer_access_is_fully_oob(in, live_sbase_descriptor)) {
                        // This exact scalar access begins beyond the descriptor's effective M_SIZE,
                        // including at the minimum value of a wave-derived SOFFSET. Publish the outer
                        // descriptor at this pc so the SPIR-V emitter can remove the memory access,
                        // and make the result exact zero so a descriptor loaded through it remains
                        // provably empty at a later MUBUF consumer.
                        if (srt_uses && have_pending_srt_use) {
                            pending_srt_use.zero_record_raw = true;
                            pending_srt_use.required_size = 0;
                            srt_uses->push_back(pending_srt_use);
                        }
                        for (uint32_t k = 0; k < n; ++k)
                            set_value(sdst + static_cast<int>(k), 0u);
                        if (n == 4 && valid_reg(sdst) && valid_reg(sdst + 3)) {
                            descr[(size_t)sdst] = {0u, 0u, 0u, 0u};
                            descr_known.set((size_t)sdst);
                            descr_key[(size_t)sdst] = 0xFFFFFFFFu;
                            descr_key_known.set((size_t)sdst);
                        }
                        break;
                    }
                }
                if (n == 0 || !base_ok || !soff_ok) {
                    // A wave-derived scalar SOFFSET (for example v_readfirstlane -> VCC_LO) is
                    // deliberately not concrete in this CPU-side fold. The V# can still be exact,
                    // though: publish its full bounded buffer by consuming-PC provenance and let
                    // the SPIR-V recompiler evaluate the dynamic dword index at runtime. The stage
                    // resource builder below accepts this zero-required-size use only when V# itself
                    // carries a conventional valid size, so an unknown offset never invents a range
                    // or gets folded to zero.
                    if (have_pending_srt_use && n != 0 && base_ok && !soff_ok)
                        srt_uses->push_back(pending_srt_use);
                    for (uint32_t k = 0; k < n; k++) {
                        forget(sdst + (int)k);
                        mark_null_origin(sdst + (int)k, null_base_origin);
                    }
                    // #2481: the offset is unknown, but the OUTER V# is fully known and bounded, so
                    // this x4 load names one entry of a declared descriptor table. Record that
                    // provenance for the destination quad; a MUBUF consumer republishes it as a
                    // table-indexed binding and the emitter selects with this instruction's own
                    // runtime scalar value. `forget` above already cleared any stale entry, so this
                    // runs after it. Every bound here is the guest's own declaration — never an
                    // inferred one — and an element that does not fit its record rejects.
                    if (is_buffer && n == 4u && live_sbase_vsharp_known &&
                        valid_reg(sdst) && valid_reg(sdst + 3)) {
                        const DecodedBufferDescriptor& outer = live_sbase_descriptor;
                        const uint32_t element = in.literal;
                        const bool bounded_table =
                            outer.base > 0x10000u && outer.stride >= 16u &&
                            outer.size_bytes != 0u &&
                            outer.size_bytes % outer.stride == 0u &&
                            static_cast<int32_t>(in.literal) >= 0 &&
                            (element % 4u) == 0u &&
                            static_cast<uint64_t>(element) + 16u <= outer.stride;
                        const uint32_t records =
                            bounded_table ? outer.size_bytes / outer.stride : 0u;
                        if (bounded_table && records >= 1u && records <= kMaxSelectedTableRecords) {
                            FoldTableSource source;
                            source.load_pc = in.pc;
                            source.base = outer.base;
                            source.stride = outer.stride;
                            source.records = records;
                            source.element_offset = element;
                            table_source[(size_t)sdst] = source;
                            table_source_known.set((size_t)sdst);
                            if (trc)
                                fprintf(stderr,
                                        "[dyntrace] selected-table pc=%u s%d base=0x%llx stride=%u "
                                        "records=%u element=+0x%x\n",
                                        in.pc, sdst, (unsigned long long)outer.base, outer.stride,
                                        records, element);
                        }
                    }
                    break;
                }
                // in.literal is the SIGN-EXTENDED 21-bit immediate (#149) — add it as signed so a
                // negative offset subtracts from the base instead of wrapping to a huge address.
                const uint32_t scalar_byte_off = soff_val + in.literal;
                const int64_t byte_off = is_buffer
                    ? static_cast<int64_t>(scalar_byte_off)
                    : (int64_t)(int32_t)in.literal + (int64_t)soff_val;
                uint64_t addr = (base + (uint64_t)byte_off) & ~3ull;
                uint32_t scalar_in_range_dwords = n;
                if (is_buffer && live_sbase_vsharp_known) {
                    const uint64_t scalar_dwords =
                        scalar_buffer_dword_count(live_sbase_descriptor);
                    const uint32_t first_dword = scalar_byte_off >> 2u;
                    scalar_in_range_dwords = static_cast<uint64_t>(first_dword) >= scalar_dwords
                        ? 0u
                        : static_cast<uint32_t>(std::min<uint64_t>(
                              n, scalar_dwords - static_cast<uint64_t>(first_dword)));
                }
                if (have_pending_srt_use) {
                    const int64_t required = byte_off + (int64_t)n * 4;
                    bool measured_required_size = false;
                    if (required > 0 && required <= (int64_t)UINT32_MAX) {
                        pending_srt_use.required_size = (uint32_t)required;
                        measured_required_size = true;
                    }
                    // A bounded V# is publishable from its own declaration even when the requested
                    // scalar offset lies past it. An unbounded raw-pointer convention still requires
                    // the measured access span before it can acquire backing.
                    if (live_sbase_vsharp_known && live_sbase_descriptor.size_bytes != 0u) {
                        srt_uses->push_back(pending_srt_use);
                    } else if (measured_required_size) {
                        srt_uses->push_back(pending_srt_use);
                    }
                }
                if (is_buffer && scalar_in_range_dwords == 0u &&
                    live_sbase_vsharp_known) {
                    // The complete scalar result is architectural zero even if base+offset happens
                    // to name mapped memory. Retain a real bounded resource for a non-empty V# so the
                    // emitted dynamic path can implement the same bound; only the established empty
                    // V# shape uses the no-backing marker.
                    if (srt_uses && have_pending_srt_use &&
                        live_sbase_descriptor.num_records == 0u &&
                        live_sbase_descriptor.size_bytes == 0u) {
                        if (!srt_uses->empty() &&
                            srt_uses->back().use_pc == pending_srt_use.use_pc)
                            srt_uses->pop_back();
                        pending_srt_use.zero_record_raw = true;
                        pending_srt_use.scalar_oob_offset_known = true;
                        pending_srt_use.scalar_oob_byte_offset = scalar_byte_off;
                        pending_srt_use.required_size = 0u;
                        srt_uses->push_back(pending_srt_use);
                    }
                    for (uint32_t k = 0; k < n; ++k)
                        set_value(sdst + static_cast<int>(k), 0u);
                    if (n == 4 && valid_reg(sdst) && valid_reg(sdst + 3)) {
                        descr[(size_t)sdst] = {0u, 0u, 0u, 0u};
                        descr_known.set((size_t)sdst);
                        descr_key[(size_t)sdst] = 0xFFFFFFFFu;
                        descr_key_known.set((size_t)sdst);
                    }
                    break;
                }
                // AGC scalar-pointer user data can carry aperture/tag bits above the title's usable
                // GPU VA. Real GFX10 S_LOAD addresses are canonicalized by the memory system; a raw
                // host dereference is not. Prefer the exact 64-bit address, then (only when it is
                // unreadable) try the architectural Base48 and the PS5 process' exercised 40-bit VA
                // aperture. Every candidate must be mapped for the complete load, so this never turns
                // an unreadable pointer into an unchecked dereference. The 4-byte alignment matches
                // scalar-memory dword addressing (SharpEmu's evaluator applies the same alignment).
                bool addr_readable = is_buffer ? false : readable(addr, n * 4);
                if (!is_buffer && !addr_readable) {
                    for (uint64_t mask : {0xFFFFFFFFFFFFull, 0xFFFFFFFFFFull}) {
                        const uint64_t candidate = (((base & mask) + (uint64_t)byte_off) & ~3ull);
                        if (candidate != addr && readable(candidate, n * 4)) {
                            if (trc) fprintf(stderr, "[dyntrace]   canonical S_LOAD addr 0x%llx -> 0x%llx (mask=0x%llx)\n",
                                             (unsigned long long)addr, (unsigned long long)candidate,
                                             (unsigned long long)mask);
                            addr = candidate;
                            addr_readable = true;
                            break;
                        }
                    }
                }
                if (is_buffer)
                    addr_readable = scalar_in_range_dwords != 0u &&
                                    readable(addr, scalar_in_range_dwords * 4u);
                if (!addr_readable) { if (trc) fprintf(stderr, "[dyntrace]   addr 0x%llx unreadable\n", (unsigned long long)addr);
                                      for (uint32_t k = 0; k < n; k++) {
                                          if (is_buffer && k >= scalar_in_range_dwords)
                                              set_value(sdst + (int)k, 0u);
                                          else {
                                              forget(sdst + (int)k);
                                              mark_null_origin(sdst + (int)k, null_base_origin);
                                          }
                                      }
                                      break; }
                if (trc && writer_provenance_enabled()) {
                    const uint32_t observed_bytes = is_buffer
                        ? scalar_in_range_dwords * 4u : n * 4u;
                    const auto writer = last_guest_write_overlap(addr, observed_bytes);
                    if (writer) {
                        fprintf(stderr,
                                "[dyntrace]   latest GPU writer kind=%s seq=%llu "
                                "range=[0x%llx,+0x%llx) submit=%llu item=%llu order=%llu "
                                "identity=0x%llx\n",
                                guest_writer_kind_name(writer->kind),
                                (unsigned long long)writer->sequence,
                                (unsigned long long)writer->addr,
                                (unsigned long long)writer->size,
                                (unsigned long long)writer->submit,
                                (unsigned long long)writer->item,
                                (unsigned long long)writer->order,
                                (unsigned long long)writer->identity);
                    } else {
                        // Quote the history's own state with every negative, because a negative can
                        // mean "nothing wrote this" OR "the recorder that would have caught it was
                        // never armed", and the two are otherwise identical in the log. All four
                        // recorders now follow writer_provenance_enabled() — colour targets did not
                        // until this diagnostic's own change (they were gated on the unrelated
                        // PROSPER_PROVENANCE_DIM, which is what made a negative unreadable and is
                        // why the counts are printed at all).
                        //
                        // Three limits the counts do NOT remove, all load-bearing:
                        //   * a SKIPPED dispatch records nothing however this is configured, so a
                        //     negative never rules out a skipped compute program;
                        //   * a guest CPU write is recorded by no kind at all, so this history only
                        //     ever describes GPU-side writers;
                        //   * a kind can read non-zero here while still having DISCARDED the write
                        //     that matters. Without PROSPER_WRITER_PROVENANCE (i.e. when only a
                        //     dimension probe armed the history) DmaData keeps writes >= 256 bytes
                        //     and WriteData >= 64 dwords, and the compute recorders skip any
                        //     host-data buffer. A small write to a narrow window is then missing
                        //     from a history whose summary looks healthy — set
                        //     PROSPER_WRITER_PROVENANCE for unfiltered retention before trusting a
                        //     negative over a span of a few dwords.
                        fprintf(stderr,
                                "[dyntrace]   no recorded GPU writer overlaps [0x%llx,+0x%x) "
                                "(history=%zu recorded: %s)%s\n",
                                (unsigned long long)addr, observed_bytes,
                                guest_write_history_size(), guest_write_recorder_summary(),
                                guest_write_history_size() ? ""
                                    : "  <- HISTORY EMPTY: this is VOID, not negative");
                    }
                }
                // WHOLE-BUFFER DUMP for a traced scalar buffer load (#2132). When a bindless chain
                // dies on a pointer field that reads zero, the loaded dwords alone cannot separate
                // "this buffer legitimately carries no table pointer here" from "we read the wrong
                // bytes of the right buffer" from "this is the wrong buffer" — all three produce the
                // same null. So dump the buffer the descriptor itself bounds, and flag every dword
                // pair inside it that is a mapped guest address. That makes the negative checkable
                // against the buffer's own contents instead of against the one offset the chain
                // happened to dereference: a table pointer present at some OTHER offset is a
                // positive instance built independently of the failing load, which is the only kind
                // that can validate this null (CLAUDE.md, same-source positive controls).
                // Bounded by the V#'s own size and a hard cap; probes readability before touching.
                if (trc && is_buffer) {
                    uint32_t r2 = 0, r3 = 0;
                    const bool size_ok = known(sbase + 2, r2) && known(sbase + 3, r3);
                    uint32_t stride = 0, b1v = 0;
                    if (known(sbase + 1, b1v)) stride = (b1v >> 16) & 0x3FFFu;
                    uint64_t bytes = 0;
                    if (size_ok) bytes = stride ? (uint64_t)r2 * stride : (uint64_t)r2;
                    const uint64_t kCap = 512;
                    if (bytes > kCap) bytes = kCap;
                    fprintf(stderr, "[dyntrace]   V# buffer base=0x%llx stride=%u num_records=%u "
                                    "dword3=0x%08x -> %llu bytes%s\n",
                            (unsigned long long)base, stride, r2, r3,
                            (unsigned long long)bytes, size_ok ? "" : " (size UNKNOWN)");
                    if (bytes >= 4 && readable(base, (uint32_t)bytes)) {
                        const uint32_t* buf = (const uint32_t*)(uintptr_t)base;
                        // NOT `dwords` — that names the shader length in this scope (#2202 N4).
                        const uint32_t buf_dwords = (uint32_t)(bytes / 4);
                        for (uint32_t i = 0; i < buf_dwords; i += 8) {
                            fprintf(stderr, "[dyntrace]   +0x%03x:", i * 4);
                            for (uint32_t j = i; j < i + 8 && j < buf_dwords; ++j)
                                fprintf(stderr, " %08x", buf[j]);
                            fprintf(stderr, "\n");
                        }
                        fprintf(stderr, "[dyntrace]   mapped 64-bit pointer candidates in buffer:");
                        bool any_ptr = false;
                        for (uint32_t i = 0; i + 1 < buf_dwords; ++i) {
                            const uint64_t cand =
                                (uint64_t)buf[i] | ((uint64_t)buf[i + 1] << 32);
                            if (cand <= 0x10000 || !readable(cand, 8)) continue;
                            fprintf(stderr, " +0x%x=0x%llx", i * 4, (unsigned long long)cand);
                            any_ptr = true;
                        }
                        fprintf(stderr, "%s\n", any_ptr ? "" : " none");
                    } else {
                        fprintf(stderr, "[dyntrace]   buffer body not readable -> dump skipped\n");
                    }
                }
                std::array<uint32_t, 16> bounded_scalar_words{};
                const uint32_t* mem = (const uint32_t*)(uintptr_t)addr;
                if (is_buffer && scalar_in_range_dwords < n) {
                    std::memcpy(bounded_scalar_words.data(), mem,
                                scalar_in_range_dwords * sizeof(uint32_t));
                    mem = bounded_scalar_words.data();
                }
                const bool imm_only = (soff_field == 125) && (int32_t)in.literal >= 0;   // SGPR_NULL soffset
                const bool optional_table_source =
                    !is_buffer && in.opcode == kSmemOpcodeLoadDwordX2 && n == 2u &&
                    imm_only && in.literal == kGtaOptionalBufferPointerOffset &&
                    sbase == 0 && consecutive_seed_copy_range(sbase, 2) &&
                    addr == base + kGtaOptionalBufferPointerOffset &&
                    readable(base, kGtaOptionalBufferTableBytes);
                const bool nullable_output_table_source =
                    !is_buffer && in.opcode == kSmemOpcodeLoadDwordX2 && n == 2u &&
                    imm_only && in.literal == kGtaNullableOutputPointerOffset &&
                    sbase == 0 && consecutive_seed_copy_range(sbase, 2) &&
                    addr == base + kGtaNullableOutputPointerOffset &&
                    readable(base, kGtaNullableOutputWitnessBytes);
                uint32_t bvh_count_origin = 0;
                if (!is_buffer && n == 4 && imm_only && in.literal == 0x58u &&
                    valid_reg(sbase) && valid_reg(sbase + 1) &&
                    bvh_build_role[(size_t)sbase] == BvhBuildRole::PointerLo &&
                    bvh_build_role[(size_t)(sbase + 1)] == BvhBuildRole::PointerHi &&
                    bvh_build_origin[(size_t)sbase] &&
                    bvh_build_origin[(size_t)sbase] ==
                        bvh_build_origin[(size_t)(sbase + 1)])
                    bvh_count_origin = bvh_build_origin[(size_t)sbase];
                for (uint32_t k = 0; k < n; k++) set_value(sdst + (int)k, mem[k]);
                if (bvh_count_origin) {
                    mark_bvh_build(sdst, bvh_count_origin, BvhBuildRole::CountLo);
                    mark_bvh_build(sdst + 1, bvh_count_origin, BvhBuildRole::CountHi);
                }
                if (!is_buffer && n == 16 && imm_only && in.literal == 0u) {
                    // The title's object header is fetched as one mapped 16-dword block. Seed each
                    // aligned qword independently: only the qword later consumed by the exact
                    // descriptor builder can reach a ray instruction, and unrelated header fields
                    // must not share provenance with it.
                    for (uint32_t first = 0; first < n; first += 2) {
                        uint32_t origin = ++next_bvh_build_origin;
                        if (!origin) origin = ++next_bvh_build_origin;
                        mark_bvh_build(sdst + (int)first, origin, BvhBuildRole::HeaderLo);
                        mark_bvh_build(sdst + (int)first + 1, origin, BvhBuildRole::HeaderHi);

                        // A zero qword in the same successful header read is also an exact null
                        // pointer. Give each pair its own origin so adjacent zero fields cannot be
                        // spliced into one descriptor, then let only dependent scalar ALU retain it.
                        if (mem[first] == 0u && mem[first + 1] == 0u) {
                            uint32_t null_origin = ++next_null_chain_origin;
                            if (!null_origin) null_origin = ++next_null_chain_origin;
                            if (x16_null_origin_pc.size() <= null_origin)
                                x16_null_origin_pc.resize((size_t)null_origin + 1u, UINT32_MAX);
                            x16_null_origin_pc[(size_t)null_origin] = in.pc;
                            mark_null_origin(sdst + (int)first, null_origin);
                            mark_null_origin(sdst + (int)first + 1, null_origin);
                        }
                    }
                }
                if (!is_buffer && n == 2 && mem[0] == 0 && mem[1] == 0) {
                    uint32_t origin = ++next_null_chain_origin;
                    if (!origin) origin = ++next_null_chain_origin;
                    mark_null_origin(sdst, origin);
                    mark_null_origin(sdst + 1, origin);
                    if (optional_table_source) {
                        if (optional_table_null_origin_pc.size() <= origin)
                            optional_table_null_origin_pc.resize(
                                (size_t)origin + 1u, UINT32_MAX);
                        optional_table_null_origin_pc[(size_t)origin] = in.pc;
                        mark_optional_null_role(sdst, OptionalNullRole::BaseLo);
                        mark_optional_null_role(sdst + 1, OptionalNullRole::BaseHi);
                    }
                    if (nullable_output_table_source) {
                        if (nullable_output_null_origin_pc.size() <= origin)
                            nullable_output_null_origin_pc.resize(
                                static_cast<size_t>(origin) + 1u, UINT32_MAX);
                        nullable_output_null_origin_pc[static_cast<size_t>(origin)] = in.pc;
                        mark_optional_null_role(sdst, OptionalNullRole::BaseLo);
                        mark_optional_null_role(sdst + 1, OptionalNullRole::BaseHi);
                    }
                }
                if ((n == 8 || n == 16) && scalar_in_range_dwords == n) {
                    // The per-register tags were cleared by set_value(); install exact lanes only
                    // after this successful mapped read. Aligned x8 windows inside x16 loads then
                    // naturally retain distinct contiguous base addresses.
                    for (uint32_t k = 0; k < n; ++k) {
                        const int reg_value = sdst + static_cast<int>(k);
                        if (!valid_reg(reg_value)) continue;
                        descriptor_word_source_addr[(size_t)reg_value] =
                            addr + k * sizeof(uint32_t);
                        descriptor_word_source_known.set((size_t)reg_value);
                    }
                }
                if ((n == 4 || n == 8) && valid_reg(sdst) && valid_reg(sdst + (int)n - 1)) {
                    const uint32_t key = (imm_only && !is_buffer) ? in.literal : 0xFFFFFFFFu;
                    for (uint32_t k = 0; k < n; ++k) {
                        val_srt_key[(size_t)(sdst + (int)k)] = key;
                        val_srt_key_known.set((size_t)(sdst + (int)k));
                    }
                }
                // Provenance key: the recompiler tags an IMMEDIATE-only descriptor load's dest SGPRs
                // with the load immediate (sreg_srt = in.literal); register-SOFFSET / negative loads
                // are not tagged, so mark those snapshots key-less.
                // A 4-dword load is a V# candidate — snapshot it now (before any later stride patch) so a
                // vertex fetch using these SGPRs resolves to the descriptor as loaded. An 8-dword load is
                // a T# candidate (image_sample SRSRC), snapshotted the same way (#294).
                // PROSPER_DESCR_COHERENCE=1 — stage 0 of the runtime-selected-descriptor lift (#2412).
                //
                // The planned design pre-materialises a descriptor TABLE on the CPU and lets the shader
                // select an entry by runtime index. That is only sound if the table's contents at the
                // moment we read them are the contents the dispatch expects. A bindless renderer
                // rewriting table slots per frame is not exotic — it is the ordinary reason to be
                // bindless — and if it happens here the failure is SILENT: the index is in range, the
                // descriptor is valid, and the bytes are a previous frame's. Nothing else in the plan
                // detects that, which is why it is measured before any of the five layers are touched.
                //
                // The probe: remember the bytes seen at each descriptor ADDRESS and report when the same
                // address later yields different bytes. That answers "does the guest rewrite descriptor
                // slots" directly, without plumbing a hash through to submit.
                //
                // BOTH counters are reported, deliberately. A bare "0 mismatches" is indistinguishable
                // from "the probe never ran" — the ambiguity that has cost this project several false
                // zeros — so `checked` is the control line and is printed alongside.
                // Gated on `is_buffer` (s_buffer_load, op >= 8) — a V#-RELATIVE load, which is what
                // reading a descriptor table looks like. The first version hooked every 4-dword scalar
                // load and so measured ordinary constant data too: 2.22M observations over 404,343
                // addresses, 21% of which changed at least once. That number is real but its population
                // is far broader than "descriptor tables", and the design question is specifically about
                // table slots. Narrowing costs one run and makes the answer actionable instead of
                // suggestive.
                if (descr_coherence_enabled() && n == 4 && is_buffer &&
                    scalar_in_range_dwords == n) {
                    uint64_t h = 1469598103934665603ull;              // FNV-1a over the four dwords
                    for (int k = 0; k < 4; ++k) {
                        h ^= mem[k]; h *= 1099511628211ull;
                    }
                    record_descriptor_observation(addr, h);
                }
                if (n == 4 && valid_reg(sdst)) {
                              descr[(size_t)sdst] = { mem[0], mem[1], mem[2], mem[3] };
                              descr_known.set((size_t)sdst);
                              // only s_load (not s_buffer_load) dests get the recompiler's sreg_srt tag
                              descr_key[(size_t)sdst] = (imm_only && !is_buffer)
                                  ? in.literal : 0xFFFFFFFFu;
                              descr_key_known.set((size_t)sdst);
                }
                if (n == 8 && valid_reg(sdst)) {
                              descr8[(size_t)sdst] = {
                                  mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7] };
                              descr8_known.set((size_t)sdst);
                              descr8_key[(size_t)sdst] = (imm_only && !is_buffer)
                                  ? in.literal : 0xFFFFFFFFu;
                              descr8_key_known.set((size_t)sdst);
                              // SGPR loads are typeless: a later scalar buffer load may consume the
                              // first four words of this eight-dword result as a V#. Keep both views;
                              // only an actual buffer consumer reports the V# candidate.
                              descr[(size_t)sdst] = { mem[0], mem[1], mem[2], mem[3] };
                              descr_known.set((size_t)sdst);
                              descr_key[(size_t)sdst] = descr8_key[(size_t)sdst];
                              descr_key_known.set((size_t)sdst);
                }
                if (n == 16) {
                    // NGG back halves commonly fetch a compact stage-data block containing a mix of
                    // one 8-dword T# and 4-dword V# descriptors with one s_load_dwordx16. SGPR loads
                    // are typeless, so retain both aligned interpretations and let the eventual MIMG
                    // or MUBUF consumer select the appropriate one. The load's one byte-offset key
                    // cannot distinguish the resources inside the block, so retain key-less
                    // descriptor provenance; each consumer is resolved by its exact instruction pc.
                    for (uint32_t first = 0; first < n; first += 8) {
                        const int base_reg = sdst + static_cast<int>(first);
                        if (!valid_reg(base_reg) || !valid_reg(base_reg + 7)) continue;
                        descr8[(size_t)base_reg] = {
                            mem[first], mem[first + 1], mem[first + 2], mem[first + 3],
                            mem[first + 4], mem[first + 5], mem[first + 6], mem[first + 7] };
                        descr8_known.set((size_t)base_reg);
                        descr8_key[(size_t)base_reg] = 0xFFFFFFFFu;
                        descr8_key_known.set((size_t)base_reg);
                    }
                    for (uint32_t first = 0; first < n; first += 4) {
                        const int base_reg = sdst + static_cast<int>(first);
                        if (!valid_reg(base_reg) || !valid_reg(base_reg + 3)) continue;
                        descr[(size_t)base_reg] = {
                            mem[first], mem[first + 1], mem[first + 2], mem[first + 3] };
                        descr_known.set((size_t)base_reg);
                        descr_key[(size_t)base_reg] = 0xFFFFFFFFu;
                        descr_key_known.set((size_t)base_reg);
                    }
                }
                break;
            }
            case Rdna2Format::MIMG: {
                // IMAGE_BVH_INTERSECT_RAY consumes a four-dword BVH descriptor, not an eight-dword
                // image T#. Preserve the words live at this exact instruction and materialize the
                // acceleration-structure bytes as a raw read-only SSBO for software lowering.
                if (in.opcode == 0xE6u) {
                    if (srt_uses) {
                        const int bbase = in.src[1].value;
                        std::array<uint32_t, 4> live_bvh{};
                        bool live_known = valid_reg(bbase) && valid_reg(bbase + 3);
                        for (int k = 0; live_known && k < 4; ++k)
                            live_known &= known(bbase + k, live_bvh[(size_t)k]);
                        const DecodedBvhDescriptor d = live_known
                            ? decode_bvh_descriptor(live_bvh.data()) : DecodedBvhDescriptor{};
                        // GTA V's sorted builder rewrites an aligned four-word window from its x16
                        // header load. A historical snapshot proves only the bytes it actually loaded;
                        // if the sorted live words differ, require the exact builder chain below. Keep
                        // the established snapshot semantics for unsorted descriptors, whose supported
                        // builders predate the narrow BOX_SORT_EN provenance proof.
                        const bool snapshot_provenance = valid_reg(bbase) &&
                            descr_known.test((size_t)bbase) &&
                            (!d.sort_enabled ||
                             (live_known && live_bvh == descr[(size_t)bbase]));
                        const bool seed_provenance = !snapshot_provenance &&
                            (untouched_seed_range(bbase, 4) ||
                             consecutive_seed_copy_range(bbase, 4));
                        uint32_t builder_origin = valid_reg(bbase)
                            ? bvh_build_origin[(size_t)bbase] : 0u;
                        const BvhBuildRole expected_builder_roles[4] = {
                            BvhBuildRole::BaseLo, BvhBuildRole::SortedBaseHi,
                            BvhBuildRole::SizeLo, BvhBuildRole::DescriptorHi,
                        };
                        bool builder_provenance = builder_origin != 0 && valid_reg(bbase + 3);
                        for (int k = 0; builder_provenance && k < 4; ++k)
                            builder_provenance =
                                bvh_build_origin[(size_t)(bbase + k)] == builder_origin &&
                                bvh_build_role[(size_t)(bbase + k)] ==
                                    expected_builder_roles[k];
                        const bool plausible = live_known &&
                            (snapshot_provenance || seed_provenance || builder_provenance) &&
                            d.type == 8u && d.base > 0x10000u && d.size_bytes != 0;
                        uint32_t proven_null_origin = valid_reg(bbase) ? null_origin(bbase) : 0u;
                        for (int k = 1; proven_null_origin && k < 4; ++k)
                            if (null_origin(bbase + k) != proven_null_origin)
                                proven_null_origin = 0;
                        bool null_control_proven = true;
                        if (proven_null_origin < x16_null_origin_pc.size() &&
                            x16_null_origin_pc[(size_t)proven_null_origin] != UINT32_MAX)
                            null_control_proven = straight_line_null_chain_dominates(
                                ins, x16_null_origin_pc[(size_t)proven_null_origin], in.pc);
                        const bool guarded_null = !plausible && proven_null_origin &&
                            null_control_proven && guarded_bvh_use(ins, in.pc);
                        if (trc) {
                            fprintf(stderr,
                                    "[dyntrace] BVH pc=%u srsrc=s%d snapshot=%d seed=%d builder=%d "
                                    "live_known=%d bvh=",
                                    in.pc, bbase, snapshot_provenance, seed_provenance,
                                    builder_provenance, live_known);
                            if (live_known) {
                                for (uint32_t word : live_bvh) fprintf(stderr, "%08x ", word);
                            } else {
                                fprintf(stderr, "<unknown> ");
                            }
                            if (plausible) {
                                fprintf(stderr, "-> base=0x%llx size=0x%llx type=%u tri_mode=%u grow=%u",
                                        (unsigned long long)d.base,
                                        (unsigned long long)d.size_bytes, d.type,
                                        d.triangle_return_mode, d.box_grow);
                            } else if (guarded_null) {
                                fprintf(stderr, "<guarded-null origin=%u>", proven_null_origin);
                            } else if (live_known) {
                                fprintf(stderr, "<insufficient-provenance>");
                            }
                            fputc('\n', stderr);
                        }
                        if (plausible) {
                            SrtUse u;
                            u.kind = 2;
                            u.key = 0xFFFFFFFFu;
                            u.bvh4 = live_bvh;
                            u.use_pc = in.pc;
                            srt_uses->push_back(u);
                        } else if (guarded_null) {
                            SrtUse u;
                            u.kind = 3;
                            u.key = 0xFFFFFFFFu;
                            u.use_pc = in.pc;
                            srt_uses->push_back(u);
                        }
                    }
                    break;
                }
                // Descriptor-table use (#294): an image op's SRSRC (src[1]) is an 8-dword T#; if it was
                // snapshotted from a table load, report it as a Texture use — with the paired SSAMP
                // (src[2]) S# when that 4-dword load also resolved. VGPR-only dest: no SGPR state.
                // Key-less snapshots (register-SOFFSET loads) are reported too (#273): the use carries
                // its instruction pc, which the recompiler resolves via ShaderResource::fetch_pc when
                // the immediate-key model fails or collides.
                if (srt_uses) {
                    const int tbase = in.src[1].value;
                    const int samp_base = in.src[2].value;
                    const bool have_t8 = valid_reg(tbase) && descr8_known.test((size_t)tbase);
                    const bool have_key = valid_reg(tbase) && descr8_key_known.test((size_t)tbase);
                    // MIMG itself proves that its eight SRSRC words are a T#. A table snapshot or
                    // direct user-data range establishes provenance, but the RESOURCE MUST use the
                    // words live at this instruction: modeled scalar patches are architectural, and
                    // an unmodeled write makes one word unknown and therefore rejects. Falling back
                    // to descr8's load-time bytes after either case would bind a stale texture.
                    std::array<uint32_t, 8> live_t8{};
                    bool live_t8_known = true;
                    for (int k = 0; k < 8; ++k)
                        live_t8_known &= known(tbase + k, live_t8[(size_t)k]);
                    const bool seed_provenance = !have_t8 &&
                        (untouched_seed_range(tbase, 8) ||
                         consecutive_seed_copy_range(tbase, 8));
                    bool plausible_seed = true;
                    if (seed_provenance && live_t8_known) {
                        const DecodedImageDescriptor d = decode_image_descriptor(live_t8.data());
                        plausible_seed = d.base > 0x10000 && d.width && d.height &&
                            d.width <= 16384 && d.height <= 16384 &&
                            d.type >= 8 && d.type <= 15;
                    }
                    // An all-zero direct T# is architectural null state, not a malformed image that
                    // needs to pass the ordinary descriptor plausibility gate. Preserve it as an
                    // exact-PC use just like an all-zero table-loaded T#: build_stage_table applies
                    // the operation-class rule and admits only sampled reads as null Textures, while
                    // stores and atomics remain fail-visible. A partially-zero/nonzero seed still has
                    // to satisfy the normal image checks and cannot enter through this exception.
                    const bool exact_null_seed = seed_provenance && live_t8_known &&
                        std::all_of(live_t8.begin(), live_t8.end(),
                                    [](uint32_t word) { return word == 0; });
                    const std::array<uint32_t, 8>* t8 =
                        live_t8_known &&
                                (have_t8 || (seed_provenance &&
                                             (plausible_seed || exact_null_seed)))
                            ? &live_t8 : nullptr;
                    uint32_t tkey = 0xFFFFFFFFu;
                    if (t8 && have_key) {
                        uint32_t common_key = 0;
                        bool common_key_known = true;
                        for (int k = 0; k < 8; ++k) {
                            const size_t reg = (size_t)(tbase + k);
                            if (!valid_reg(tbase + k) || !val_srt_key_known.test(reg)) {
                                common_key_known = false;
                                break;
                            }
                            if (k == 0) common_key = val_srt_key[reg];
                            else if (val_srt_key[reg] != common_key) {
                                common_key_known = false;
                                break;
                            }
                        }
                        if (common_key_known) tkey = common_key;
                    }
                    uint64_t t8_source_addr = 0;
                    if (t8) {
                        bool contiguous_source_known = true;
                        for (int k = 0; k < 8; ++k) {
                            const int reg_value = tbase + k;
                            if (!valid_reg(reg_value) ||
                                !descriptor_word_source_known.test((size_t)reg_value)) {
                                contiguous_source_known = false;
                                break;
                            }
                            const uint64_t word_addr =
                                descriptor_word_source_addr[(size_t)reg_value];
                            if (k == 0) t8_source_addr = word_addr;
                            else if (word_addr != t8_source_addr +
                                                    static_cast<uint64_t>(k) * sizeof(uint32_t)) {
                                contiguous_source_known = false;
                                break;
                            }
                        }
                        if (!contiguous_source_known) t8_source_addr = 0;
                    }
                    const bool from_seed = t8 && seed_provenance;
                    if (trc) {
                        fprintf(stderr, "[dyntrace] MIMG pc=%u op=0x%x srsrc=s%d ssamp=s%d "
                                        "have_t8=%d seed_t8=%d key=0x%x t8=",
                                in.pc, in.opcode, tbase, samp_base, have_t8, from_seed, tkey);
                        if (t8) {
                            for (uint32_t word : *t8) fprintf(stderr, "%08x ", word);
                            const DecodedImageDescriptor td = decode_image_descriptor(t8->data());
                            fprintf(stderr,
                                    "-> base=0x%llx %ux%ux%u type=%u fmt=%u tile=%u mip=%u:%u",
                                    (unsigned long long)td.base, td.width, td.height, td.depth,
                                    td.type, td.format, td.tile_mode, td.base_level, td.max_mip);
                        } else {
                            fprintf(stderr, "<unknown>");
                        }
                        fputc('\n', stderr);
                    }
                    if (t8) {
                        SrtUse u; u.kind = 0; u.t8 = *t8;
                        u.descriptor_source_addr = t8_source_addr;
                        u.key = tkey;
                        u.use_pc = in.pc;
                        // image_store plus EVERY integer image atomic -- an atomic is a read-modify-write, so a
                        // resource one targets can never be a sampled texture. This list used to be 0x08/0x0f/0x11,
                        // which is exactly the set the RECOMPILER emitted: classifier and emitter kept in lockstep,
                        // neither generalised. Opcodes verified with llvm-mc, positive control image_atomic_add ->
                        // word0 0xf0440128, byte-identical to the guest's own dword (#2275).
                        u.is_storage_image = in.opcode == 0x08 || in.opcode == 0x09 ||
                                             in.opcode == 0x0f ||
                                             (in.opcode >= 0x11 && in.opcode <= 0x1a && in.opcode != 0x13);
                        u.proven_zero_mip = proven_zero_mip_at_use;
                        u.is_depth_compare = (in.opcode >= 0x28 && in.opcode <= 0x2f) ||
                                             (in.opcode >= 0x38 && in.opcode <= 0x3f) ||
                                             (in.opcode >= 0x58 && in.opcode <= 0x5f);
                        const bool have_s4 = valid_reg(samp_base) &&
                                             descr_known.test((size_t)samp_base);
                        const bool seed_s4 = !have_s4 && untouched_seed_range(samp_base, 4);
                        bool live_s4_known = true;
                        for (int k = 0; k < 4; ++k)
                            live_s4_known &= known(samp_base + k, u.s4[(size_t)k]);
                        if (live_s4_known && (have_s4 || seed_s4)) {
                            // Like the T#, sampler words are read live. This avoids retaining a stale
                            // x16 load snapshot if scalar code patches or invalidates the paired S#.
                            u.has_samp = true;
                        }
                        srt_uses->push_back(u);
                    }
                }
                break;
            }
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF: {
                const bool is_mtbuf = in.fmt == Rdna2Format::MTBUF;
                // Buffer stores, raw loads/stores, and supported atomics need a kind-1 resource use.
                // Nonempty format loads are intentionally handled only by DynFetch below: it snapshots
                // the V# live at the instruction and resolves by exact pc, avoiding a duplicate stale
                // SRT use. An empty format load is the exception because it has no materializable
                // DynFetch resource, yet its exact zero result needs the same no-backing marker.
                const bool format_load_use = !is_mtbuf && in.opcode <= 0x03;
                const bool raw_buffer_use = !is_mtbuf &&
                    ((in.opcode >= 0x08 && in.opcode <= 0x0F) ||
                     (in.opcode >= kMubufOpcodeStoreDword &&
                      in.opcode <= kMubufOpcodeStoreDwordX3));
                const bool format_store_use = in.opcode >= 0x04 && in.opcode <= 0x07;
                // Keep the generic set aligned with the 32-bit RMW opcodes the SPIR-V emitter lowers.
                // Opcodes 0x50/0x5a reach descriptor inspection separately: only GTA V's exact
                // linear stride-8 qword proof below is published; every sibling x2 shape stays
                // fail-closed rather than inheriting ordinary 32-bit bounds.
                const bool atomic_buffer_use_32 =
                    !is_mtbuf &&
                    (in.opcode == kMubufOpcodeAtomicSwap ||
                     in.opcode == kMubufOpcodeAtomicAdd ||
                     in.opcode == kMubufOpcodeAtomicSub ||
                     (in.opcode >= kMubufOpcodeAtomicSmin &&
                      in.opcode <= kMubufOpcodeAtomicXor) ||
                     in.opcode == kMubufOpcodeAtomicFmin ||
                     in.opcode == kMubufOpcodeAtomicFmax);
                const bool atomic_x2_candidate =
                    !is_mtbuf && (in.opcode == kMubufOpcodeAtomicSwapX2 ||
                                  in.opcode == kMubufOpcodeAtomicOrX2);
                if (srt_uses &&
                    (format_load_use || format_store_use || raw_buffer_use ||
                     atomic_buffer_use_32 || atomic_x2_candidate)) {
                    const int srsrc = in.src[1].value;
                    std::array<uint32_t, 4> current{};
                    bool current_known = true;
                    for (int k = 0; k < 4; ++k)
                        current_known &= known(srsrc + k, current[(size_t)k]);
                    const bool loaded_provenance = valid_reg(srsrc) &&
                                                   descr_known.test((size_t)srsrc);
                    // #2412: WHY a MUBUF publishes no descriptor use. A use is created only when the
                    // fold knows all four SRSRC words; ~1.8% of sites fail that, and those are exactly
                    // the instructions whose shaders then reject. This names which word was unknown, so
                    // the causes can be enumerated per shader instead of guessed one at a time -- one
                    // cause (VCC consumed as scalar data) was traced and fixed for 27 draws, so there
                    // are demonstrably others.
                    if (!current_known && std::getenv("PROSPER_MUBUF_UNKNOWN_LOG")) {
                        int unknown_word = -1;
                        for (int k = 0; k < 4 && unknown_word < 0; ++k) {
                            uint32_t probe = 0;
                            if (!known(srsrc + k, probe)) unknown_word = k;
                        }
                        // Name the instruction that made the unknown word unknown, and identify the
                        // program so lines can be joined to `[recompile-reject]`'s `sh=` (#2436's
                        // scheme: first code dword + span). `writer_pc=4294967295` means the register
                        // was NEVER forgotten in this walk — i.e. never written at all — which is a
                        // different defect from a modeled instruction giving up on it, and the two
                        // were previously indistinguishable in this output.
                        const int unk_reg = unknown_word >= 0 ? srsrc + unknown_word : -1;
                        const bool unk_valid = unk_reg >= 0 && valid_reg(unk_reg);
                        fprintf(stderr,
                                "[mubuf-unknown] sh=%08x/%zu pc=%u op=0x%x srsrc=s%d unknown_word=%d "
                                "loaded_provenance=%d writer_pc=%u writer_words=%08x:%08x\n",
                                dwords ? code[0] : 0u, dwords,
                                in.pc, in.opcode, srsrc, unknown_word, (int)loaded_provenance,
                                unk_valid ? forget_pc[(size_t)unk_reg] : 0xffffffffu,
                                unk_valid ? forget_w0[(size_t)unk_reg] : 0u,
                                unk_valid ? forget_w1[(size_t)unk_reg] : 0u);
                    }
                    // #2481: the SRSRC words are NOT known, but the fold proved they were produced
                    // by an x4 scalar load of one element of a bounded descriptor table. Publish the
                    // table itself; the emitter resolves the element at run time. Restricted to the
                    // untyped load family the table-indexed emitter already admits — a typed fetch,
                    // store or atomic through a selected descriptor needs per-entry treatment that
                    // does not exist yet, and must keep failing visibly rather than binding element
                    // zero. Note this is checked BEFORE `current_known`, which is false here by
                    // construction: `forget` cleared those words when the offset proved dynamic.
                    if (!current_known && srt_uses && valid_reg(srsrc) &&
                        table_source_known.test((size_t)srsrc) &&
                        in.fmt == Rdna2Format::MUBUF && !is_mtbuf &&
                        (in.opcode == kMubufOpcodeLoadDword ||
                         in.opcode == kMubufOpcodeLoadDwordX2 ||
                         in.opcode == kMubufOpcodeLoadDwordX3 ||
                         in.opcode == kMubufOpcodeLoadDwordX4)) {
                        const FoldTableSource& source = table_source[(size_t)srsrc];
                        SrtUse u;
                        u.kind = 1;
                        u.key = 0xFFFFFFFFu;
                        u.use_pc = in.pc;
                        u.table_record_count = source.records;
                        u.table_entry_stride = source.stride;
                        u.table_element_offset = source.element_offset;
                        u.table_load_pc = source.load_pc;
                        u.table_base = source.base;
                        srt_uses->push_back(u);
                        if (trc)
                            fprintf(stderr,
                                    "[dyntrace] MUBUF pc=%u selected-table s%d producer=%u "
                                    "records=%u stride=%u element=+0x%x\n",
                                    in.pc, srsrc, source.load_pc, source.records, source.stride,
                                    source.element_offset);
                    }
                    if (current_known) {
                        // MUBUF/MTBUF itself is definitive that its four SRSRC words are a V#. Publish
                        // the values LIVE at the consumer whenever the scalar fold knows all of them.
                        // This also covers a direct descriptor whose NUM_RECORDS/stride is patched by
                        // modeled scalar ALU: such a patch intentionally breaks entry-seed provenance,
                        // but does not make the now-concrete V# any less valid. An unmodeled write makes
                        // a word unknown and still fails closed; the shape/range checks below reject
                        // malformed concrete values before any resource is emitted.
                        SrtUse u; u.kind = 1; u.v4 = current; u.key = 0xFFFFFFFFu; u.use_pc = in.pc;
                        if (is_mtbuf) u.instruction_format = in.mtbuf_format;
                        if (loaded_provenance) {
                            uint32_t common_key = 0;
                            bool have_common_key = true;
                            for (int k = 0; k < 4; ++k) {
                                const size_t r = (size_t)(srsrc + k);
                                if (!valid_reg(srsrc + k) || !val_srt_key_known.test(r)) {
                                    have_common_key = false;
                                    break;
                                }
                                if (k == 0) common_key = val_srt_key[r];
                                else if (val_srt_key[r] != common_key) {
                                    have_common_key = false;
                                    break;
                                }
                            }
                            // Rewrites clear the recompiler's complete sreg_srt tag, so the live V#
                            // must then resolve through this consuming instruction's exact pc.
                            if (have_common_key) u.key = common_key;
                        }
                        DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                        const uint32_t atomic_x2_record_count = atomic_x2_candidate
                            ? exact_atomic_x2_record_count(in, d, u.v4.data()) : 0u;
                        if (atomic_x2_record_count)
                            u.atomic_x2_record_count = atomic_x2_record_count;
                        DataFormat inst_format = DataFormat::Unknown;
                        uint32_t inst_components = 0;
                        if (is_mtbuf)
                            rdna2_buffer_format(in.mtbuf_format, &inst_format, &inst_components);
                        // FORMAT=INVALID is the architectural unbound-resource marker. MTBUF's
                        // instruction format controls conversion, but does not turn an unbound V#
                        // into a valid resource.
                        const bool descriptor_bound = !is_mtbuf ||
                            (((u.v4[3] >> 12) & 0x7Fu) != 0);
                        const bool format_supported = !format_store_use ||
                            (is_mtbuf ? (descriptor_bound && inst_format != DataFormat::Unknown &&
                                         inst_components != 0)
                                      : (d.format != DataFormat::Unknown && d.num_components != 0 &&
                                         !d.forbid_unknown_fallback));
                        // Byte-addressed raw/atomic V#s validly use stride zero: NUM_RECORDS is bytes.
                        // Typed format stores retain the strided record requirement.
                        const bool stride_supported = !format_store_use || d.stride != 0;
                        // A fully-known admitted V# with NUM_RECORDS=0 is not a missing descriptor:
                        // hardware returns zero for loads, drops stores, and range-checks an atomic
                        // as one all-or-nothing access. Preserve that proof at this exact consumer pc.
                        // MTBUF, format stores, unsupported atomics, and an addr0 descriptor with
                        // nonzero records remain outside the marker contract.
                        const bool zero_record_format =
                            format_load_use && zero_record_format_selectors_are_zero(in, u.v4.data());
                        const bool zero_record_raw =
                            (zero_record_format || raw_buffer_use || atomic_buffer_use_32) &&
                            d.num_records == 0u && d.size_bytes == 0u;
                        const uint32_t optional_low_origin = null_origin(srsrc);
                        const uint32_t optional_high_origin = null_origin(srsrc + 1);
                        const bool optional_origin = optional_low_origin &&
                            optional_low_origin == optional_high_origin &&
                            optional_low_origin < optional_table_null_origin_pc.size() &&
                            optional_table_null_origin_pc[(size_t)optional_low_origin] != UINT32_MAX;
                        const uint32_t optional_producer_pc = optional_origin
                            ? optional_table_null_origin_pc[(size_t)optional_low_origin]
                            : UINT32_MAX;
                        const bool optional_cfg_dominates = optional_origin &&
                            !has_indirect_control_flow(decoded->instructions) &&
                            straight_line_null_chain_dominates(
                                decoded->instructions, optional_producer_pc, in.pc);
                        const bool optional_null_raw_load =
                            optional_cfg_dominates &&
                            rdna2_optional_null_raw_load_shape(in) &&
                            valid_reg(srsrc + 3) &&
                            optional_null_role[(size_t)srsrc] == OptionalNullRole::BaseLo &&
                            optional_null_role[(size_t)(srsrc + 1)] == OptionalNullRole::BaseHi &&
                            optional_null_role[(size_t)(srsrc + 2)] == OptionalNullRole::RecordCount &&
                            optional_null_role[(size_t)(srsrc + 3)] == OptionalNullRole::Config &&
                            gta_optional_null_descriptor_shape(d, u.v4.data());
                        const bool nullable_origin = optional_low_origin &&
                            optional_low_origin == optional_high_origin &&
                            optional_low_origin < nullable_output_null_origin_pc.size() &&
                            nullable_output_null_origin_pc[static_cast<size_t>(
                                optional_low_origin)] != UINT32_MAX;
                        const bool proven_null_nullable_raw_buffer =
                            nullable_origin &&
                            rdna2_gta5_nullable_output_shader(code, dwords) &&
                            rdna2_gta5_nullable_output_site(in) !=
                                Gta5NullableOutputAccess::None &&
                            valid_reg(srsrc + 3) &&
                            user_sgprs && nsgpr > 7u &&
                            gta_nullable_output_descriptor_shape(
                                d, u.v4.data(), user_sgprs[7]);
                        if (trc && d.base == 0u && d.num_records != 0u)
                            fprintf(stderr,
                                    "[dyntrace] optional-null-candidate pc=%u origin=%u/%u tagged=%d "
                                    "producer=%u cfg=%d roles=%u/%u/%u/%u packet=%d descriptor=%d\n",
                                    in.pc, optional_low_origin, optional_high_origin,
                                    (int)optional_origin, optional_producer_pc,
                                    (int)optional_cfg_dominates,
                                    (unsigned)optional_null_role[(size_t)srsrc],
                                    (unsigned)optional_null_role[(size_t)(srsrc + 1)],
                                    (unsigned)optional_null_role[(size_t)(srsrc + 2)],
                                    (unsigned)optional_null_role[(size_t)(srsrc + 3)],
                                    (int)rdna2_optional_null_raw_load_shape(in),
                                    (int)gta_optional_null_descriptor_shape(d, u.v4.data()));
                        const bool proven_null_guarded_raw_store =
                            gta5_null_pointer_at_guard &&
                            gta5_null_raw_store_descriptor(u.v4) &&
                            rdna2_gta5_null_guarded_raw_store_site(in);
                        if (proven_null_guarded_raw_store || zero_record_raw ||
                            optional_null_raw_load || proven_null_nullable_raw_buffer ||
                            (!format_load_use &&
                            (d.base > 0x10000 && d.size_bytes != 0 &&
                             d.size_bytes <= 0x10000000u && stride_supported && format_supported &&
                             (!atomic_x2_candidate || atomic_x2_record_count != 0u)))) {
                            u.zero_record_raw = zero_record_raw;
                            u.optional_null_raw_load = optional_null_raw_load;
                            u.proven_null_guarded_raw_store =
                                proven_null_guarded_raw_store;
                            u.proven_null_nullable_raw_buffer =
                                proven_null_nullable_raw_buffer;
                            if (trc) fprintf(stderr,
                                             "[dyntrace] MUBUF pc=%u live V# s%d base=0x%llx "
                                             "stride=%u size=%u zero-record=%d optional-null=%d "
                                             "null-guarded-store=%d\n",
                                             in.pc, srsrc, (unsigned long long)d.base,
                                             d.stride, d.size_bytes, (int)zero_record_raw,
                                             (int)optional_null_raw_load,
                                             (int)proven_null_guarded_raw_store);
                            srt_uses->push_back(u);
                        } else if (trc) {
                            // Fail-visible: say WHY a raw buffer use was not published. Without this
                            // the site simply vanishes from the trace, which reads as "the fold never
                            // walked it" and sends the reader upstream to control flow. Three separate
                            // static derivations of why GTA V 0x413ce6000's pc70 is missing were wrong
                            // before this line existed (#2481).
                            fprintf(stderr,
                                    "[dyntrace] MUBUF pc=%u NOT-PUBLISHED s%d v4=%08x:%08x:%08x:%08x "
                                    "base=0x%llx stride=%u records=%u size=%u "
                                    "fmt-load=%d raw=%d base>64k=%d size!=0=%d size<=256M=%d "
                                    "stride-ok=%d format-ok=%d\n",
                                    in.pc, srsrc, u.v4[0], u.v4[1], u.v4[2], u.v4[3],
                                    (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                                    (int)format_load_use, (int)raw_buffer_use,
                                    (int)(d.base > 0x10000), (int)(d.size_bytes != 0),
                                    (int)(d.size_bytes <= 0x10000000u),
                                    (int)stride_supported, (int)format_supported);
                        }
                    }
                }
                // buffer_load_format_* (vertex fetch): opcodes 0..3. Resolve the SRSRC (src[1]) SGPR to the
                // V# most-recently loaded into it.
                if (in.opcode <= 3) {
                    int srsrc = in.src[1].value;
                    // Prefer the FETCH-TIME V#: the fetch shader patches the descriptor's format field (v[3])
                    // between load and fetch — so read the CURRENT SGPR values (which the interpreter has
                    // tracked through the patch, incl. the s_cselect tail) to get the real data format (e.g.
                    // UNORM8 for a packed vertex color, vs the load-time Unknown). Fall back to the load-time
                    // snapshot if the patched dwords aren't fully known.
                    uint32_t vv[4]; bool k0 = known(srsrc, vv[0]), k1 = known(srsrc + 1, vv[1]),
                                         k2 = known(srsrc + 2, vv[2]), k3 = known(srsrc + 3, vv[3]);
                    bool patched = k0 && k1 && k2 && k3;
                    const bool have_descr = valid_reg(srsrc) && descr_known.test((size_t)srsrc);
                    // Fold the fetch's CONSTANT byte offset into the emitted V# base (#273 item 1, the
                    // "solid banner" bug): the recompiler's per-fetch (by_fetch_pc) address model is
                    // exactly gl_VertexIndex*stride from the resolved base — it assumes the attribute's
                    // in-record byte offset is already IN the base. Unity-style fetch shaders satisfy
                    // that by patching each attribute's V# base; DOLL's UE4 Slate VS instead uses ONE
                    // un-patched V# and carries each attribute's offset in the MUBUF SOFFSET register
                    // (+ the 12-bit inst offset). Without the fold, all four Slate attributes (pos, uv,
                    // material-uv, color) read the position bytes -> the loading-banner widget rendered
                    // as a solid bar. The walk knows the SOFFSET value (it computed it from the attr-spec
                    // words), so add soffset+inst_offset to the descriptor base; an UNKNOWN soffset keeps
                    // the un-offset base (previous behavior). CONFIDENCE: HIGH (fetch-time values traced
                    // live; Messenger's fetches carry SOFFSET=0 so they are byte-identical).
                    uint32_t soff = 0; bool soff_known = true;
                    if (in.src[2].kind == OperandKind::Special && in.src[2].value == 125)     // SGPR_NULL -> 0
                        soff = 0;
                    else if (!srcval(in.src[2], soff))
                        soff_known = false;                          // real but untracked SOFFSET (#398) — see below
                    const uint32_t inst_off = in.literal & 0xFFFu;
                    const uint32_t fetch_off = soff + inst_off;
                    auto with_off = [&](DecodedBufferDescriptor d) {
                        d.base += fetch_off;
                        // size_bytes stays num_records*stride: the hardware bound is INDEX < num_records
                        // (record granularity), so from the offset base the last record's attribute still
                        // lies within (num_records-1)*stride + fetch_off + attr bytes — trimming the size
                        // by fetch_off cut the LAST vertex's attribute off the upload (guarded reads made
                        // it zeros -> a collapsed final vertex).
                        return d;
                    };
                    auto append_fetch = [&](DecodedBufferDescriptor d, uint32_t desc_v3,
                                            bool from_seed = false) {
                        if (is_mtbuf && ((desc_v3 >> 12) & 0x7Fu) == 0) {
                            if (trc) fprintf(stderr,
                                "[dyntrace]   MTBUF pc=%u unbound V# v3=0x%x -> unresolved\n",
                                in.pc, desc_v3);
                            return;
                        }
                        // decode_buffer_descriptor deliberately rejects packed formats whose selector
                        // or conversion semantics are unsupported. Do not let build_stage_table's
                        // legacy Unknown->Float32 fallback resurrect that descriptor as four raw dwords.
                        if (!is_mtbuf && d.forbid_unknown_fallback) {
                            if (trc) fprintf(stderr,
                                "[dyntrace]   MUBUF pc=%u packed V# v3=0x%x unsupported -> unresolved\n",
                                in.pc, desc_v3);
                            return;
                        }
                        DynFetch fetch{ in.pc, srsrc, with_off(d), desc_v3 };
                        fetch.from_seed = from_seed;
                        // A legal instruction/SOFFSET transform changes the normalized base without
                        // changing the four raw seed dwords.  That transformed resource no longer has
                        // byte-identical descriptor identity, so it cannot name the entry V# as a
                        // direct raw witness (doing so would manufacture a mismatch in #1853).
                        if (fetch_off == 0 && consecutive_seed_copy_range(srsrc, 4))
                            fetch.direct_user_data_index =
                                val_seed_origin[static_cast<size_t>(srsrc)];
                        fetch.unshifted_desc = d;
                        if (is_mtbuf) fetch.instruction_format = in.mtbuf_format;
                        fetch.index_mode = fetch_index_mode_before_write;
                        if (trc)
                            fprintf(stderr,
                                    "[dyntrace]   fetch pc=%u -> base=0x%llx stride=%u "
                                    "num_records=%u size=%u fmt=%u nc=%u index=%u\n",
                                    in.pc, (unsigned long long)fetch.desc.base, fetch.desc.stride,
                                    fetch.desc.num_records, fetch.desc.size_bytes,
                                    (unsigned)fetch.desc.format, fetch.desc.num_components,
                                    (unsigned)fetch.index_mode);
                        out.push_back(fetch);
                    };
                    if (trc) fprintf(stderr, "[dyntrace] MUBUF fetch pc=%u op=0x%x SRSRC=s%d patched=%d (k=%d%d%d%d v3=0x%x) have_descr=%d off=+0x%x soff_known=%d\n",
                                     in.pc, in.opcode, srsrc, patched, k0, k1, k2, k3,
                                     k3 ? vv[3] : 0, have_descr, fetch_off, (int)soff_known);
                    // A real (non-NULL) SOFFSET the fold cannot resolve would silently collapse fetch_off's
                    // in-record component to 0 — every attribute reads base+inst_off (the "solid banner"
                    // collapse this fold was written to fix) or a wrong descriptor address. Leave the fetch
                    // UNRESOLVED (a loud recompile-coverage miss) rather than fabricating offset 0 (#398).
                    if (!soff_known) {
                        if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u SOFFSET untracked -> fetch left unresolved (not folded to 0)\n", in.pc);
                        break;
                    }
                    // Per-fetch: record THIS fetch's live V# (the SRSRC SGPR is reloaded with a different V#
                    // per vertex attribute — position, uv, color…). Keyed by the fetch's pc so the recompiler
                    // resolves each buffer_load_format to the descriptor as loaded at that instruction.
                    if (patched) {
                        append_fetch(decode_buffer_descriptor(vv), vv[3]);
                    } else if (have_descr) {
                        append_fetch(decode_buffer_descriptor(descr[(size_t)srsrc].data()),
                                     descr[(size_t)srsrc][3]);
                    }
                    else if (srsrc >= (int)user_sgpr_base && srsrc + 4 <= (int)(user_sgpr_base + nsgpr) &&
                             valid_reg(srsrc + 3) && !reloaded.test((size_t)srsrc) &&
                             !reloaded.test((size_t)(srsrc + 1)) &&
                             !reloaded.test((size_t)(srsrc + 2)) &&
                             !reloaded.test((size_t)(srsrc + 3))) {
                        // SEED fallback (#294): the SRSRC V# was placed directly in the user-data SGPRs
                        // by the driver (never s_loaded — so no `descr` snapshot) and the shader's
                        // stride/format patch left the CURRENT dwords partially unknown (its s_cselect
                        // condition reads an NGG system SGPR we don't model). Use the SEED values — the
                        // same load-time/pre-patch semantics as the `descr` fallback above. Refused if
                        // any of the 4 SGPRs was RELOADED from memory since seeding (a stale seed then
                        // no longer describes the register). DOLL's scene-geometry VS fetches resolve
                        // through exactly this path. CONFIDENCE: MED (patch-ignoring, like `descr`).
                        const uint32_t sv[4] = { user_sgprs[srsrc - (int)user_sgpr_base],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 1],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 2],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 3] };
                        DecodedBufferDescriptor d = decode_buffer_descriptor(sv);
                        // Plausibility: only emit a real-looking V# (mirrors the direct-resource guard).
                        if (d.base > 0x10000 && d.size_bytes != 0 && d.size_bytes <= 0x10000000u &&
                            (is_mtbuf || !d.forbid_unknown_fallback)) {
                            if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u seed-V# fallback SRSRC=s%d base=0x%llx\n",
                                             in.pc, srsrc, (unsigned long long)d.base);
                            append_fetch(d, sv[3], /*from_seed=*/true);
                        }
                    }
                }
                break;
            }
            case Rdna2Format::VOP2:
                // The compact e32 v_cndmask form uses VCC implicitly. Unity's NGG fetch prologue
                // reissues this selector before each attribute load; treating the write as ordinary
                // shader arithmetic loses the vertex-id proof after the first position fetch and
                // collapses Evergate's later UV/color attributes.
                if (explicit_ngg_index_provenance && in.opcode == 0x01 &&
                    in.dst.kind == OperandKind::VGPR) {
                    const FoldMask mask = mask_state[106]; // implicit VCC_LO/HI pair
                    const Operand& selected = mask == FoldMask::All ? in.src[1] : in.src[0];
                    VertexFetchIndexMode mode = VertexFetchIndexMode::Shader;
                    if (mask != FoldMask::Unknown && selected.kind == OperandKind::VGPR &&
                        selected.value >= 0 && selected.value < static_cast<int>(kFoldVgprs))
                        mode = vector_index_mode[(size_t)selected.value];
                    if (in.dst.value >= 0 && in.dst.value < static_cast<int>(kFoldVgprs))
                        vector_index_mode[(size_t)in.dst.value] = mode;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   VOP2 pc=%u cndmask v%d mask=%u selected=v%d index=%u\n",
                                in.pc, in.dst.value, (unsigned)mask, selected.value, (unsigned)mode);
                }
                break;
            case Rdna2Format::VOP3:
                if (explicit_ngg_index_provenance && in.opcode == 0x101 &&
                    in.dst.kind == OperandKind::VGPR) {
                    const FoldMask mask = srcmask(in.src[2]);
                    const Operand& selected = mask == FoldMask::All ? in.src[1] : in.src[0];
                    VertexFetchIndexMode mode = VertexFetchIndexMode::Shader;
                    if (mask != FoldMask::Unknown && selected.kind == OperandKind::VGPR &&
                        selected.value >= 0 && selected.value < static_cast<int>(kFoldVgprs))
                        mode = vector_index_mode[(size_t)selected.value];
                    if (in.dst.value >= 0 && in.dst.value < static_cast<int>(kFoldVgprs))
                        vector_index_mode[(size_t)in.dst.value] = mode;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   VOP3 pc=%u cndmask v%d mask=%u selected=v%d index=%u\n",
                                in.pc, in.dst.value, (unsigned)mask, selected.value, (unsigned)mode);
                }
                // Scalar spill slots used by large UE shaders: v_writelane_b32 packs wave-uniform
                // SGPR values into fixed VGPR lanes, then v_readlane_b32 restores them later. The
                // recompiler deliberately drops SRT tags on restore, so recovered descriptors use
                // key-less, exact-pc provenance.
                if ((in.opcode == 0x361 || in.opcode == 0x360) &&
                    in.src[1].kind == OperandKind::InlineInt &&
                    in.src[1].value >= 0 && in.src[1].value < 64) {
                    const uint32_t slot =
                        ((uint32_t)(in.opcode == 0x361 ? in.dst.value : in.src[0].value) << 6) |
                        (uint32_t)in.src[1].value;
                    if (in.opcode == 0x361) {                         // v_writelane_b32 vDST, sSRC, lane
                        uint32_t v;
                        if (srcval(in.src[0], v)) scalar_spill_slots[slot] = v;
                        else scalar_spill_slots.erase(slot);
                    } else {                                          // v_readlane_b32 sDST, vSRC, lane
                        auto it = scalar_spill_slots.find(slot);
                        if (it == scalar_spill_slots.end()) {
                            forget(in.dst.value);
                        } else {
                            set_value(in.dst.value, it->second);
                            if (valid_reg(in.dst.value)) {
                                val_srt_key[(size_t)in.dst.value] = 0xFFFFFFFFu;
                                val_srt_key_known.set((size_t)in.dst.value);
                            }
                        }
                    }
                }
                break;
            case Rdna2Format::SOPK:
                // s_movk_i32 (op 0x00) is `SDST = signext(SIMM16)` — a compile-time constant, and the one
                // SOPK this fold has direct evidence it must not discard. Measured on PPSA04263 (GTA V) at
                // gameplay, a single register watch (`PROSPER_DYNTRACE_SGPR=16`) reports **66,507** forgets
                // of s16 to this encoding — `b0100092` = `s_movk_i32 s16, 0x92`, round-trip-decoded with
                // llvm-mc gfx1030 against a positive control, not read off a table. Forgetting a literal
                // assignment is pure loss: the value is in the instruction.
                //
                // It also does NOT write SCC. The comment below already said so and the code invalidated
                // SCC anyway, which is safe but costs every later s_cselect that depended on a live SCC.
                //
                // Deliberately ONE opcode. The rest of the format stays exactly as before — conservative
                // and correct — because s_addk_i32/s_cmpk_* genuinely do write SCC, and a stale SCC
                // consumed by a later s_cselect fabricates a confidently-wrong V# patch. Widening this
                // needs its own evidence per opcode; the sibling ops are not free just because they are
                // adjacent in the encoding.
                if (in.opcode == 0x00) {                         // s_movk_i32: constant, no SCC write
                    if (in.dst.kind == OperandKind::SGPR)
                        set_value(in.dst.value, (uint32_t)(int32_t)in.simm16);
                    break;
                }
                // s_mulk_i32: `D.i = D.i * signext(SIMM16)`, and it does NOT write SCC -- the comment
                // below already says so while the code invalidated SCC for it anyway. Two costs, both
                // paid on GTA V: the product is discarded, and every later s_cselect that depended on
                // a live SCC loses it.
                //
                // The per-opcode evidence this widening requires, since the note below rightly refuses
                // to take sibling opcodes for free: 0x413ce6000 pc149 is `s_mulk_i32 s106, 120`, where
                // 120 is the descriptor array's exact record stride. It feeds
                // `s_buffer_load_dwordx4 s[8:11], s[4:7], s106` at pc153, which selects one V# from
                // that array, which is the descriptor for `buffer_load_dwordx3` at pc156 -- the
                // instruction the program is rejected on, with mode=unresolved-operand. Forgetting
                // s106 here is what makes that descriptor unresolvable.
                //
                // Multiplies a KNOWN destination only. An unknown destination still forgets, exactly
                // as before; the change is that it no longer takes SCC with it.
                if (in.opcode == kSopkOpcodeMulkI32) {
                    if (in.dst.kind == OperandKind::SGPR) {
                        uint32_t current = 0;
                        if (known(in.dst.value, current))
                            set_value(in.dst.value,
                                      static_cast<uint32_t>(current *
                                                            static_cast<uint32_t>(
                                                                static_cast<int32_t>(in.simm16))));
                        else
                            forget(in.dst.value);
                    }
                    // A product is not an add-carry chain; the tracked origins cannot survive it.
                    null_count_carry_origin = 0;
                    bvh_count_carry_origin = 0;
                    break;
                }
                // s_cmpk_* / s_addk_i32 write SCC (only s_movk/s_version/s_cmovk/s_mulk don't); this
                // interpreter doesn't model the rest of SOPK, so they conservatively invalidate the
                // tracked SCC — a stale SCC consumed by a later s_cselect would fabricate a
                // confidently-wrong V# patch.
                // Three SOPK encodings account for every conservative SCC loss measured on GTA V's
                // ray-tracing pass (0x205b654a00, PROSPER_DYNTRACE_SCC), and TWO OF THEM DO NOT
                // WRITE SCC AT ALL:
                //
                //   188x  s_setreg_b32     (0x13) -- writes a HARDWARE REGISTER, not SCC
                //    43x  s_waitcnt_vscnt  (0x17, sdst=NULL) -- a wait; register-transparent
                //    94x  s_addk_i32       (0x0f) -- genuinely writes SCC on overflow
                //
                // This matters because SCC gates whole descriptors here: the BVH descriptor's dword3
                // is `s_addc_u32 s19, -1, 0`, whose operands are both inline constants, so an unknown
                // SCC is the ONLY thing that can leave it unresolved -- and an unresolved dword3 is
                // `mode=unresolved-operand` at the image_bvh_intersect_ray that consumes it.
                //
                // s_addk_i32 keeps invalidating SCC, because it really does write it. Its VALUE is
                // now folded when the destination is known, for the same reason s_mulk_i32's is.
                if (in.opcode == kSopkOpcodeSetregB32) {
                    // Its encoded "dst" field is a hwreg id, not an SGPR, so nothing scalar is
                    // written and nothing needs forgetting.
                    break;
                }
                if (in.opcode == kSopkOpcodeWaitcntVscnt) {
                    // The sibling case above already passes the sdst=NULL(125) form through. Any
                    // other destination is unmodelled, so forget it -- but never SCC.
                    if (in.dst.kind == OperandKind::SGPR && in.dst.value != 125)
                        forget(in.dst.value);
                    break;
                }
                if (in.opcode == kSopkOpcodeAddkI32) {
                    if (in.dst.kind == OperandKind::SGPR) {
                        uint32_t current = 0;
                        if (known(in.dst.value, current))
                            set_value(in.dst.value,
                                      current + static_cast<uint32_t>(
                                                    static_cast<int32_t>(in.simm16)));
                        else
                            forget(in.dst.value);
                    }
                    scc = -1;                    // s_addk_i32 writes SCC on signed overflow
                    report_scc("sopk-addk");
                    null_count_carry_origin = 0;
                    bvh_count_carry_origin = 0;
                    break;
                }
                scc = -1;
                report_scc("sopk-conservative");
                null_count_carry_origin = 0;
                bvh_count_carry_origin = 0;
                if (in.dst.kind == OperandKind::SGPR) forget(in.dst.value);
                break;
            default:
                // Remaining formats (SOPP, VALU, memory, …) don't write SCC, so the tracked SCC survives.
                if (in.dst.kind == OperandKind::SGPR) forget(in.dst.value);   // unmodeled scalar write -> unknown
                break;
        }
    }
    if (profile_fold)
        record_stage_fold_profile(
            (uint64_t)(uintptr_t)code, user_sgpr_base, decoded->code.size(), ins.size(), out.size(),
            srt_uses ? srt_uses->size() - srt_before : 0, guest_probe_calls,
            std::chrono::duration<double, std::milli>(FoldClock::now() - fold_start).count(),
            std::chrono::duration<double, std::milli>(decode_done - fold_start).count(),
            guest_probe_ms);
    return out;
}

// Bound the common compiler-generated linear clear kernel without trusting an oversized/formatless
// V#. Its first instruction forms GlobalInvocationId.x from TGID.x and TID.x, then an idxen-only
// single-component store writes literal zero at that descriptor record. Since every supported numeric
// format represents zero identically, Uint32 is a lossless backing view even when FORMAT is INVALID.
// This recognizes the complete four-instruction kernel, not a general "large buffer" fallback: any
// other value, address shape, or side effect stays rejected.
static uint32_t linear_dispatch_raw_store_size(const uint32_t* code, size_t dwords,
                                               uint32_t fetch_pc,
                                               const DecodedBufferDescriptor& descriptor,
                                               uint32_t local_x, uint32_t threads_x,
                                               uint32_t tgid_x_sgpr) {
    if (!code || !dwords || !local_x || !threads_x || tgid_x_sgpr == UINT32_MAX ||
        !descriptor.stride || (local_x & (local_x - 1u)) != 0)
        return 0;
    uint32_t local_shift = 0;
    for (uint32_t width = local_x; width > 1; width >>= 1) ++local_shift;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    if (instructions.size() != 4) return 0;
    const Rdna2Inst* index_writer = &instructions[0];
    const Rdna2Inst* zero_writer = &instructions[1];
    const Rdna2Inst* access = &instructions[2];
    const Rdna2Inst* end = &instructions[3];
    if (access->pc != fetch_pc || index_writer->pc != 0 || zero_writer->pc != 2 ||
        end->pc != 5 || end->fmt != Rdna2Format::SOPP || end->opcode != 0x001 ||
        access->fmt != Rdna2Format::MUBUF || access->opcode != 0x004 ||
        access->dst.kind != OperandKind::VGPR || access->dst.value != 1 ||
        access->src[0].kind != OperandKind::VGPR || access->src[0].value != 0 ||
        access->src[2].kind != OperandKind::InlineInt || access->src[2].value != 0 ||
        (access->literal & 0x3FFFu) != 0x2000u ||
        zero_writer->fmt != Rdna2Format::VOP1 || zero_writer->opcode != 0x001 ||
        zero_writer->dst.kind != OperandKind::VGPR || zero_writer->dst.value != 1 ||
        zero_writer->src[0].kind != OperandKind::InlineInt || zero_writer->src[0].value != 0 ||
        index_writer->fmt != Rdna2Format::VOP3 || index_writer->opcode != 0x346 ||
        index_writer->dst.kind != OperandKind::VGPR || index_writer->dst.value != 0 ||
        index_writer->src[0].kind != OperandKind::SGPR ||
        static_cast<uint32_t>(index_writer->src[0].value) != tgid_x_sgpr ||
        index_writer->src[1].kind != OperandKind::InlineInt ||
        static_cast<uint32_t>(index_writer->src[1].value) != local_shift ||
        index_writer->src[2].kind != OperandKind::VGPR || index_writer->src[2].value != 0)
        return 0;

    const uint64_t required =
        static_cast<uint64_t>(threads_x - 1u) * descriptor.stride + sizeof(uint32_t);
    constexpr uint32_t kMaxProvenLinearStore = 16u << 20;
    if (!required || required > descriptor.size_bytes || required > kMaxProvenLinearStore)
        return 0;
    return static_cast<uint32_t>(required);
}

static bool gta_optional_null_linear_load_launch(
        const uint32_t* code, size_t dwords, uint32_t use_pc,
        const DecodedBufferDescriptor& descriptor, const uint32_t descriptor_words[4],
        uint32_t local_x, uint32_t threads_x, uint32_t tgid_x_sgpr) {
    if (!code || !dwords || local_x != kGtaOptionalBufferLocalSize ||
        tgid_x_sgpr != kGtaOptionalBufferTgidSgpr || !threads_x ||
        threads_x != descriptor.num_records ||
        !gta_optional_null_descriptor_shape(descriptor, descriptor_words))
        return false;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    if (has_indirect_control_flow(instructions)) return false;

    const Rdna2Inst* consumer = nullptr;
    for (const Rdna2Inst& in : instructions)
        if (in.pc == use_pc) { consumer = &in; break; }
    if (!consumer || !rdna2_optional_null_raw_load_shape(*consumer)) return false;

    const int srsrc = consumer->src[1].value;
    const Rdna2Inst* descriptor_producer = nullptr;
    for (const Rdna2Inst& in : instructions) {
        if (in.pc >= use_pc) break;
        if (in.fmt == Rdna2Format::SMEM &&
            in.opcode == kSmemOpcodeLoadDwordX2 && in.len_dwords == 2u &&
            in.dst.kind == OperandKind::SGPR && in.dst.value == srsrc &&
            in.src[0].kind == OperandKind::SGPR && in.src[0].value == 0 &&
            in.src[1].kind == OperandKind::Special && in.src[1].value == 125 &&
            in.literal == kGtaOptionalBufferPointerOffset)
            descriptor_producer = &in;
    }
    if (!descriptor_producer ||
        !straight_line_null_chain_dominates(
            instructions, descriptor_producer->pc, use_pc))
        return false;

    const int vaddr = consumer->src[0].value;
    const Rdna2Inst* index_writer = nullptr;
    for (const Rdna2Inst& in : instructions) {
        if (in.pc >= use_pc) break;
        // M0-relative writes have no statically enumerable destination. One after the candidate
        // index definition could overwrite VADDR, so the narrow proof declines the whole program.
        if (in.fmt == Rdna2Format::VOP1 &&
            in.opcode == kVop1OpcodeMovreldB32 && in.pc > 1u)
            return false;
        if (in.dst.kind == OperandKind::VGPR && in.dst.value >= 0) {
            const uint32_t writes = rdna2_vgpr_write_count(in);
            if (vaddr >= in.dst.value &&
                vaddr < in.dst.value + static_cast<int>(writes))
                index_writer = &in;
        }
        if (rdna2_tfe_status_vgpr(in) == vaddr)
            index_writer = &in;
    }
    if (!index_writer || index_writer->pc != 1u ||
        index_writer->fmt != Rdna2Format::VOP3 ||
        index_writer->opcode != kVop3OpcodeLshlAddU32 ||
        index_writer->len_dwords != 2u || index_writer->has_modifier ||
        index_writer->dst.kind != OperandKind::VGPR ||
        index_writer->dst.value != vaddr ||
        index_writer->src[0].kind != OperandKind::SGPR ||
        static_cast<uint32_t>(index_writer->src[0].value) != tgid_x_sgpr ||
        index_writer->src[1].kind != OperandKind::InlineInt ||
        index_writer->src[1].value != 6 ||
        index_writer->src[2].kind != OperandKind::VGPR ||
        index_writer->src[2].value != 0)
        return false;
    return true;
}

bool shader_resource_allows_zero_mip_specialization(
    const SrtUse& use, const DecodedImageDescriptor& descriptor,
    const DecodedImageView& view) {
    return use.proven_zero_mip && descriptor.sample_count == 1u &&
        descriptor.base_level == 0u && descriptor.last_level == 0u &&
        descriptor.max_mip == 0u && !view.in_mip_tail &&
        // RESTORED. The rendering series removed this term, but only here -- the recompiler's
        // IMAGE_LOAD_MIP acceptance (rdna2_emit_alu.cpp, `res->compression_enabled`) and the compute
        // cache's partitioning both still refuse a DCC resource, so removing it here alone leaves the
        // two halves of one contract disagreeing: realization marks the descriptor proven and the
        // recompiler then declines it as `unresolved-operand`, which drops the whole dispatch.
        //
        // Three existing tests enforce the invariant -- test_dynfetch_fold ("DCC-backed
        // IMAGE_LOAD_MIP remains fail-visible"), test_rdna2_spirv_struct ("IMAGE_LOAD_MIP accepted an
        // unproven, DCC, or multilevel resource") and test_shader_recompile_cache ("compute cache
        // partitions zero-mip proof, levels, DCC, and texture-tail safety") -- against one new
        // assertion that expects the opposite. Dropping the term in all three places was tried and
        // makes those three fail, so the change is incomplete rather than merely unlanded here.
        //
        // The argument for eventually removing it is real and recorded in the recompiler beside its
        // own copy of the term: compression describes how bytes are ENCODED while this marker is
        // about which LOD is ADDRESSED, and GTA V's 0x2042f49a00 is declined partly for bytes that
        // resource never reads. Making that change coherently means moving the refusal to the point
        // that knows whether authoritative pixels were acquired, and doing it in all three places at
        // once. That is its own piece of work, not a side effect of this series.
        !descriptor.compression_enabled;
}

std::vector<SrtUse> add_compute_buffer_resources(ShaderResourceTable& table,
                                                 const uint32_t* code, size_t dwords,
                                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                                 uint32_t linear_local_x,
                                                 uint32_t linear_threads_x,
                                                 uint32_t tgid_x_sgpr,
                                                 const ComputeResourceDispatchContext*
                                                     dispatch_context) {
    std::vector<SrtUse> srt_uses;
    const std::vector<DynFetch> direct_fetches = resolve_dynamic_fetch(
        code, dwords, user_sgprs, nsgpr, /*user_sgpr_base*/0, &srt_uses);

    // A format-load resource has one identity: the descriptor live at its exact instruction pc.
    // Keep its original base because the compute ConstantBuffer address path applies the MUBUF
    // OFFSET/SOFFSET itself; `fetch.desc` is shifted for graphics' special vertex-index path.
    for (const auto& fetch : direct_fetches) {
        const DecodedBufferDescriptor& d = fetch.unshifted_desc;
        if (fetch.instruction_format != UINT32_MAX &&
            ((fetch.desc_v3 >> 12) & 0x7Fu) == 0)
            continue;
        DataFormat format = d.format;
        uint32_t components = d.num_components;
        if (fetch.instruction_format != UINT32_MAX)
            rdna2_buffer_format(fetch.instruction_format, &format, &components);
        const uint32_t resource_size = d.size_bytes;
        if (d.base <= 0x10000 || resource_size == 0 || resource_size > 0x10000000u ||
            format == DataFormat::Unknown || !components ||
            (fetch.instruction_format == UINT32_MAX && d.forbid_unknown_fallback))
            continue;
        bool mapped = false;
        for (auto& r0 : table.resources) {
            if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                r0.size != resource_size || r0.stride != d.stride || r0.format != format ||
                r0.num_components != components)
                continue;
            if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = fetch.fetch_pc;
            if (r0.fetch_pc == fetch.fetch_pc) { mapped = true; break; }
        }
        if (mapped) continue;
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        r.format = format;
        r.num_components = components;
        r.gpu_addr = d.base;
        r.size = resource_size;
        r.stride = d.stride;
        r.fetch_pc = fetch.fetch_pc;
        table.resources.push_back(r);
    }

    // Oversized FORMAT=INVALID store descriptors are deliberately absent from the generic fold.
    // Recover only the exact zero-clear kernel proven above, taking its live direct V# from the
    // user-data SGPRs and bounding the upload to this dispatch's one-dimensional invocation extent.
    // A nonzero FORMAT stays on the generic width-aware path: treating an R8/R16 zero as Uint32 would
    // overwrite adjacent components even though zero itself has the same bit pattern.
    uint32_t proven_linear_store_pc = UINT32_MAX;
    if (linear_local_x && linear_threads_x && tgid_x_sgpr != UINT32_MAX && user_sgprs) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(code, dwords, instructions);
        if (instructions.size() == 4 && instructions[2].src[1].kind == OperandKind::SGPR) {
            const uint32_t srsrc = static_cast<uint32_t>(instructions[2].src[1].value);
            if (srsrc + 4u <= nsgpr) {
                const DecodedBufferDescriptor d = decode_buffer_descriptor(user_sgprs + srsrc);
                const uint32_t raw_format = (user_sgprs[srsrc + 3] >> 12) & 0x7Fu;
                const uint32_t resource_size = linear_dispatch_raw_store_size(
                    code, dwords, instructions[2].pc, d, linear_local_x, linear_threads_x,
                    tgid_x_sgpr);
                const bool already_materialized = std::any_of(
                    table.resources.begin(), table.resources.end(), [&](const ShaderResource& r) {
                        return r.cls == ResourceClass::ConstantBuffer && r.gpu_addr == d.base &&
                               r.fetch_pc == instructions[2].pc;
                    });
                if (raw_format == 0 && d.base > 0x10000 && resource_size &&
                    !already_materialized) {
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    r.format = DataFormat::Uint32;
                    r.num_components = 1;
                    r.gpu_addr = d.base;
                    r.size = resource_size;
                    r.stride = d.stride;
                    r.fetch_pc = instructions[2].pc;
                    table.resources.push_back(r);
                    proven_linear_store_pc = instructions[2].pc;
                }
            }
        }
    }

    bool null_guarded_store_shape = false;
    if (std::any_of(srt_uses.begin(), srt_uses.end(), [](const SrtUse& use) {
            return use.proven_null_guarded_raw_store;
        })) {
        null_guarded_store_shape =
            rdna2_gta5_null_guarded_raw_store_dispatch(
                code, dwords, user_sgprs, nsgpr);
    }

    bool nullable_output_shape = false;
    uint64_t nullable_output_table_root = 0;
    const bool has_nullable_output_candidate =
        std::any_of(srt_uses.begin(), srt_uses.end(), [](const SrtUse& use) {
            return use.proven_null_nullable_raw_buffer;
        });
    if (dispatch_context && user_sgprs && nsgpr >= 2u &&
        has_nullable_output_candidate) {
        const bool nullable_output_launch = rdna2_gta5_nullable_output_launch(
            code, dwords, user_sgprs, nsgpr,
            dispatch_context->local_x, dispatch_context->local_y,
            dispatch_context->local_z, dispatch_context->threads_x,
            dispatch_context->threads_y, dispatch_context->threads_z,
            dispatch_context->exact_thread_extent, dispatch_context->wave_size,
            dispatch_context->tgid_x_en, dispatch_context->tgid_y_en,
            dispatch_context->tgid_z_en, dispatch_context->tidig_comp_cnt);
        nullable_output_shape = nullable_output_launch;
        nullable_output_table_root = static_cast<uint64_t>(user_sgprs[0]) |
            (static_cast<uint64_t>(user_sgprs[1]) << 32u);
        if (nullable_output_shape) {
            if (!guest_readable(nullable_output_table_root,
                                kGtaNullableOutputWitnessBytes)) {
                nullable_output_shape = false;
            } else {
                uint64_t witnessed_pointer = UINT64_MAX;
                std::memcpy(&witnessed_pointer,
                    reinterpret_cast<const uint8_t*>(
                        static_cast<uintptr_t>(nullable_output_table_root)) +
                        kGtaNullableOutputPointerOffset,
                    sizeof(witnessed_pointer));
                nullable_output_shape = witnessed_pointer == 0u;
            }
        }
    }

    std::set<uint64_t> seen;
    for (const auto& u : srt_uses) {
        if (u.kind == 3) {
            const uint64_t dk = 0x8000000300000000ull | u.use_pc;
            if (!seen.insert(dk).second) continue;
            alignas(256) static std::array<uint8_t, 256> null_bvh{};
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Uint32;
            r.num_components = 1;
            r.size = static_cast<uint32_t>(null_bvh.size());
            r.fetch_pc = u.use_pc;
            r.host_data = null_bvh.data();
            r.host_data_size = null_bvh.size();
            table.resources.push_back(r);
            continue;
        }
        if (u.kind == 2) {
            const DecodedBvhDescriptor d = decode_bvh_descriptor(u.bvh4.data());
            if (d.type != 8u || d.base <= 0x10000u || d.size_bytes == 0 ||
                d.size_bytes > 0x10000000u || d.size_bytes > UINT32_MAX ||
                !d.triangle_return_mode || d.box_node_64b)
                continue;
            const uint64_t dk = 0x8000000200000000ull | u.use_pc;
            if (!seen.insert(dk).second) continue;
            bool mapped = false;
            for (auto& r0 : table.resources) {
                if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                    r0.size != static_cast<uint32_t>(d.size_bytes) ||
                    r0.bvh_box_grow != d.box_grow ||
                    r0.bvh_sort_enabled != d.sort_enabled)
                    continue;
                if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = u.use_pc;
                if (r0.fetch_pc == u.use_pc) { mapped = true; break; }
            }
            if (mapped) continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Uint32;
            r.num_components = 1;
            r.gpu_addr = d.base;
            r.size = static_cast<uint32_t>(d.size_bytes);
            r.fetch_pc = u.use_pc;
            r.bvh_box_grow = d.box_grow;
            r.bvh_sort_enabled = d.sort_enabled;
            table.resources.push_back(r);
            continue;
        }
        if (u.kind != 1) continue;
        if (u.table_record_count) {
            // PROSPER_SRTTABLE_LOG=1 -- report why a runtime-selected descriptor table was declined.
            // The three decline paths below already logged, but only under PROSPER_DBG, which this
            // project records as producing a ~1.5 GB log and desyncing the pad script badly enough
            // that the route never reaches the phase being diagnosed. A diagnostic reachable only by
            // a switch that destroys the repro is not reachable -- the same reasoning that ungated
            // the `[compute-table]` block in this file.
            //
            // Volume, stated correctly: `realize_compute_dispatches` reaches this PER DISPATCH, so
            // the output scales with dispatches that publish a selected table, not with the number
            // of distinct tables. Measured on a 500 s routed GTA V route: 138 lines. That is small
            // because few programs use the path, not because anything bounds it -- a title that used
            // it widely would produce far more.
            //
            // EVERY report in this block is routed through it, including the SUCCESS line -- not
            // only the declines. A switch that speaks solely on failure cannot distinguish "this
            // table was accepted" from "this table was never published", which is the same
            // silent-`continue` ambiguity that motivated the switch in the first place.
            const bool table_log =
                std::getenv("PROSPER_SRTTABLE_LOG") || std::getenv("PROSPER_DBG");
            // #2481: a runtime-selected descriptor table. The element is chosen on the GPU, so the
            // binding declares every entry and the emitter indexes it. Re-derive the whole contract
            // from guest memory here rather than trusting the use: a forged SrtUse must not be able
            // to name an arbitrary address range as a descriptor array.
            const uint64_t table_key = 0x8000000600000000ull | u.use_pc;
            if (!seen.insert(table_key).second) continue;
            if (u.instruction_format != UINT32_MAX || u.zero_record_raw ||
                u.table_entry_stride < 16u || u.table_record_count == 0u ||
                u.table_record_count > kMaxSelectedTableRecords ||
                static_cast<uint64_t>(u.table_element_offset) + 16u > u.table_entry_stride ||
                (u.table_element_offset % 4u) != 0u || u.table_base <= 0x10000u ||
                u.table_load_pc == 0xFFFFFFFFu || u.table_load_pc >= dwords) {
                // Previously a silent `continue`, which is indistinguishable from the table never
                // having been published at all -- the two have completely different causes and the
                // log could not tell them apart.
                if (table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u REJECTED shape fmt=%u zero_record=%d "
                                 "stride=%u records=%u element=%u base=0x%llx load_pc=%u dwords=%zu\n",
                                 u.use_pc, u.instruction_format, (int)u.zero_record_raw,
                                 u.table_entry_stride, u.table_record_count, u.table_element_offset,
                                 (unsigned long long)u.table_base, u.table_load_pc, dwords);
                continue;
            }
            // Every element must be a readable, individually valid V#. One malformed entry rejects
            // the whole table: binding a partially populated descriptor array would let the guest's
            // own index select a slot prosper never resolved, which renders confidently wrong
            // content instead of failing visibly.
            // The guest only guarantees the slots it actually selects. A record it never indexes on
            // this dispatch may legitimately hold anything — GTA V's arena reuses the same 600-byte
            // allocation, so unselected slots have been observed holding ordinary float data — and
            // requiring EVERY slot to decode is a stronger contract than the guest's, which declined
            // real tables. Bind a NULL descriptor for a slot that does not decode: RDNA2's own
            // out-of-range contract returns zero through one, `ShaderBufferTableEntry` already models
            // it, and it keeps the array's arity exact so the runtime index cannot land on a slot
            // that was silently dropped. A table where NOTHING decodes is still declined — that is
            // not a descriptor table at all.
            std::vector<ShaderBufferTableEntry> entries;
            entries.reserve(u.table_record_count);
            uint32_t resolved = 0;
            bool unsupported_record = false;
            for (uint32_t index = 0; index < u.table_record_count; ++index) {
                const uint64_t entry_addr =
                    u.table_base + static_cast<uint64_t>(index) * u.table_entry_stride +
                    u.table_element_offset;
                std::array<uint32_t, 4> words{};
                DecodedBufferDescriptor entry{};
                // TWO different conditions, deliberately kept apart. Nulling a slot is only sound
                // when the record is not a descriptor at all -- a stale arena slot the guest never
                // indexes. A record that decodes as a plausible V# prosper merely cannot SUPPORT is
                // prosper's gap, and nulling it would let an in-range guest index read zeros with
                // full confidence. The RDNA2 out-of-range argument does not cover that case: an
                // out-of-range index returns zero by hardware rule, but an in-range index into a
                // descriptor we declined to decode is not out of range.
                // An unreadable record is NOT a record known not to be a descriptor -- we simply
                // could not look at it. Nulling it would be the same fail-visible-to-silent-zero
                // inversion as the unsupported case below, one category over, so it declines the
                // whole table exactly as it did before this change.
                if (!guest_readable(entry_addr, static_cast<uint32_t>(sizeof(words)))) {
                    if (table_log)
                        std::fprintf(stderr,
                                     "[srt] selected-table pc=%u REJECTED unreadable-record "
                                     "index=%u addr=0x%llx\n",
                                     u.use_pc, index, (unsigned long long)entry_addr);
                    unsupported_record = true;
                    break;
                }
                std::memcpy(words.data(),
                            reinterpret_cast<const void*>(static_cast<uintptr_t>(entry_addr)),
                            sizeof(words));
                entry = decode_buffer_descriptor(words.data());
                // Only a record we READ and that does not look like a descriptor at all may be
                // nulled: a stale arena slot the guest never indexes.
                // DO NOT add a `guest_readable(entry.base, …)` term here. It looks obviously right
                // -- a record whose base is unmapped cannot be a descriptor, because selecting it
                // would fault -- and it is WRONG on this platform, in two independent ways:
                //
                //  * `exec_image_linux.cpp`'s SIGSEGV handler maps a fresh RW page for ANY unmapped
                //    SEGV_MAPERR in [GPU_VA_LO, GPU_VA_HI) = [4 GiB, 64 GiB), unconditionally and
                //    not env-gated. Faulting is the TRIGGER FOR LAZY BACKING, not evidence of
                //    invalidity -- and `0x413ce6000`, the program that motivated the idea, sits
                //    inside that window at ~16.3 GiB.
                //  * `guest_readable` cannot see through that anyway: its Linux arm probes with a
                //    `write()` to a pipe, so the kernel returns EFAULT via copy_from_user and no
                //    SIGSEGV is ever delivered to the process. The probe reports "unmapped" for an
                //    address the guest can legitimately touch.
                //
                // Page RESIDENCY is not the oracle; region IDENTITY is. `prosper_reserved_range_state`
                // exists on both platforms for that. Whatever replaces this needs a mutation arm too:
                // the existing coverage in `test_dynfetch_fold.cpp` builds its garbage with `base = 0`,
                // so a residency term can be deleted with the whole suite still green (#2481).
                const bool not_a_descriptor = entry.base <= 0x10000u || entry.size_bytes == 0u;
                const bool unsupported_descriptor = !not_a_descriptor &&
                    (entry.size_bytes > 0x10000000u || entry.forbid_unknown_fallback);
                if (unsupported_descriptor) {
                    // Decline the WHOLE table, as before this change. A >256 MiB buffer or an
                    // unrepresentable DST_SEL / USCALED-SSCALED 2_10_10_10 is a descriptor the guest
                    // may well select.
                    if (table_log)
                        std::fprintf(stderr,
                                     "[srt] selected-table pc=%u REJECTED unsupported-record "
                                     "index=%u base=0x%llx size=%u stride=%u records=%u fmt=%u "
                                     "comps=%u fallback=%d words=%08x:%08x:%08x:%08x\n",
                                     u.use_pc, index, (unsigned long long)entry.base,
                                     entry.size_bytes, entry.stride, entry.num_records,
                                     static_cast<unsigned>(entry.format), entry.num_components,
                                     (int)entry.forbid_unknown_fallback,
                                     words[0], words[1], words[2], words[3]);
                    unsupported_record = true;
                    break;
                }
                const bool usable = !not_a_descriptor;
                if (!usable && table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u NULL-SLOT index=%u why=%s "
                                 "base=0x%llx size=%u stride=%u records=%u fmt=%u comps=%u "
                                 "words=%08x:%08x:%08x:%08x\n",
                                 u.use_pc, index,
                                 entry.base <= 0x10000u ? "low-base" : "zero-size",
                                 (unsigned long long)entry.base, entry.size_bytes, entry.stride,
                                 entry.num_records, static_cast<unsigned>(entry.format),
                                 entry.num_components,
                                 words[0], words[1], words[2], words[3]);
                // The ACCEPTED population. Without it this report has only records that already
                // FAILED some filter, and comparing two post-filter populations cannot produce a
                // counterexample -- which is precisely how three successive predicates for "is this
                // record a descriptor" were proposed and refuted on #2481. Every record inspected is
                // now reported with the same decoded fields, from one run, with no filter between
                // them.
                if (usable && table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u ACCEPT-SLOT index=%u "
                                 "base=0x%llx size=%u stride=%u records=%u fmt=%u comps=%u "
                                 "words=%08x:%08x:%08x:%08x\n",
                                 u.use_pc, index, (unsigned long long)entry.base, entry.size_bytes,
                                 entry.stride, entry.num_records,
                                 static_cast<unsigned>(entry.format), entry.num_components,
                                 words[0], words[1], words[2], words[3]);
                ShaderBufferTableEntry slot;
                if (usable) {
                    slot.vsharp = words;
                    slot.gpu_addr = entry.base;
                    slot.size = entry.size_bytes;
                    slot.stride = entry.stride;
                    ++resolved;
                }
                entries.push_back(slot);
            }
            if (unsupported_record) continue;
            if (resolved == 0u) {
                if (table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u REJECTED base=0x%llx stride=%u "
                                 "records=%u resolved=%u\n",
                                 u.use_pc, (unsigned long long)u.table_base, u.table_entry_stride,
                                 u.table_record_count, resolved);
                continue;
            }
            // A descriptor array is ONE Vulkan binding, so its declared element stride, format and
            // component count are shared by every entry — `valid_shader_buffer_table_contract`
            // requires exactly that. Derive them from the entries and reject a heterogeneous table
            // rather than picking one entry's shape and binding the rest through it.
            // Null slots carry no shape, so they neither define nor contradict the array's element
            // contract; compare only the slots that resolved.
            DecodedBufferDescriptor first{};
            bool have_first = false;
            bool homogeneous = true;
            uint32_t widest = 0u;
            for (const ShaderBufferTableEntry& slot : entries) {
                if (!slot.gpu_addr && !slot.size) continue;      // null slot
                const DecodedBufferDescriptor entry =
                    decode_buffer_descriptor(slot.vsharp.data());
                if (!have_first) { first = entry; have_first = true; }
                else
                    homogeneous &= entry.stride == first.stride &&
                                   entry.format == first.format &&
                                   entry.num_components == first.num_components;
                widest = std::max(widest, slot.size);
            }
            homogeneous &= have_first;
            if (!homogeneous) {
                if (table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u REJECTED heterogeneous entries\n",
                                 u.use_pc);
                continue;
            }
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = first.format;
            r.num_components = first.num_components ? first.num_components : 1u;
            // `gpu_addr` names the table so dependency closure and capture see the guest range the
            // selection reads through; `size`/`stride` describe the ELEMENTS, because that is what
            // the binding's descriptors are.
            r.gpu_addr = u.table_base;
            r.size = widest;
            r.stride = first.stride;
            r.srt_offset = 0xFFFFFFFFu;
            r.sgpr_base = 0xFFFFFFFFu;
            r.fetch_pc = u.use_pc;
            r.table_index_count = u.table_record_count;
            r.table_entry_stride = u.table_entry_stride;
            r.table_selector_mode = BufferTableSelectorMode::DynamicSbufferByteOffset;
            r.table_load_pc = u.table_load_pc;
            r.table_entries = std::move(entries);
            // Publish only a table the whole pipeline can actually honour. An invalid contract makes
            // `declare_cbufs` poison the ENTIRE shader (`invalid_cbuf_array_access`), so publishing
            // one optimistically would turn shaders that compile today into silent empty modules.
            // Failing to publish costs exactly what the previous behaviour cost: an unresolved
            // descriptor at the consumer, fail-visible.
            if (!valid_shader_buffer_table_contract(r)) {
                if (table_log)
                    std::fprintf(stderr,
                                 "[srt] selected-table pc=%u REJECTED contract stride=%u fmt=%u "
                                 "comps=%u\n",
                                 u.use_pc, r.stride, (unsigned)r.format, r.num_components);
                continue;
            }
            if (table_log)
                std::fprintf(stderr,
                             "[srt] selected-table pc=%u producer=%u base=0x%llx stride=%u "
                             "records=%u resolved=%u element=+0x%x\n",
                             u.use_pc, u.table_load_pc, (unsigned long long)u.table_base,
                             u.table_entry_stride, u.table_record_count, resolved,
                             u.table_element_offset);
            table.resources.push_back(r);
            continue;
        }
        if (u.use_pc == proven_linear_store_pc) continue;
        if (u.instruction_format != UINT32_MAX && ((u.v4[3] >> 12) & 0x7Fu) == 0)
            continue;
        const bool exact_mtbuf = u.instruction_format != UINT32_MAX;
        // Keep an exact alias for every consumer pc, including an otherwise keyed table load. The
        // arbitrary-CFG compute dispatcher persists scalar values across basic blocks, but descriptor
        // identity is compile-time provenance and is intentionally not stored in those Function
        // variables. Exact aliases therefore remain the only unambiguous lookup after a block join.
        const uint64_t dk = 0x8000000100000000ull | u.use_pc;
        if (!seen.insert(dk).second) continue;
        bool clash = exact_mtbuf || u.key == 0xFFFFFFFFu;
        if (!clash)
            for (const auto& r0 : table.resources)
                if (r0.srt_offset == u.key) { clash = true; break; }

        const DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
        if (u.proven_null_nullable_raw_buffer) {
            // Repeat the complete program/launch/witness proof before preserving a source-table
            // range whose only semantic effect is to authorize exact zero-load/drop-store emission.
            // Ordinary non-null buffers never acquire this impossible-stride marker.
            if (u.zero_record_raw || u.optional_null_raw_load ||
                u.proven_null_guarded_raw_store || u.instruction_format != UINT32_MAX ||
                u.key != 0xFFFFFFFFu || u.required_size != 0u ||
                !nullable_output_shape ||
                !gta_nullable_output_descriptor_shape(
                    d, u.v4.data(), user_sgprs[7]) ||
                u.use_pc >= dwords)
                continue;
            Rdna2Inst marker_consumer =
                rdna2_decode_one(code + u.use_pc, dwords - u.use_pc);
            marker_consumer.pc = u.use_pc;
            if (rdna2_gta5_nullable_output_site(marker_consumer) ==
                Gta5NullableOutputAccess::None)
                continue;
            const uint64_t marker_key = 0x8000000500000000ull | u.use_pc;
            if (!seen.insert(marker_key).second) continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Unknown;
            r.num_components = 0;
            r.gpu_addr = nullable_output_table_root;
            r.size = kGtaNullableOutputWitnessBytes;
            r.stride = kProvenNullNullableRawBufferStride;
            r.srt_offset = 0xFFFFFFFFu;
            r.sgpr_base = 0xFFFFFFFFu;
            r.fetch_pc = u.use_pc;
            table.resources.push_back(r);
            continue;
        }
        if (u.optional_null_raw_load) {
            // The fold proves the mapped +0x58 producer and exact scalar V# construction. Repeat
            // the packet/launch half here so a forged SrtUse cannot manufacture zero semantics.
            if (u.zero_record_raw || u.instruction_format != UINT32_MAX ||
                u.key != 0xFFFFFFFFu || u.required_size != 0u ||
                !gta_optional_null_linear_load_launch(
                    code, dwords, u.use_pc, d, u.v4.data(), linear_local_x,
                    linear_threads_x, tgid_x_sgpr))
                continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Uint32;
            r.num_components = 1;
            r.gpu_addr = 0;
            r.size = 0;
            r.stride = kGtaOptionalBufferStride;
            r.srt_offset = 0xFFFFFFFFu;
            r.sgpr_base = 0xFFFFFFFFu;
            // Not a valid SGPR index. This field is serialized even for ConstantBuffers, where it
            // is otherwise inert, so capture/replay preserves the load-only marker without a
            // format extension.
            r.sampler_sgpr_base = kOptionalNullRawLoadMarkerSamplerBase;
            r.fetch_pc = u.use_pc;
            table.resources.push_back(r);
            continue;
        }
        if (u.proven_null_guarded_raw_store) {
            // Re-check every retained part of the dispatch proof at materialization. In particular,
            // this is not a base-zero shortcut: the observed descriptor has one in-bounds record and
            // becomes an ordinary writable resource when the entry pointer is non-null.
            if (u.zero_record_raw || u.instruction_format != UINT32_MAX ||
                !null_guarded_store_shape ||
                !gta5_null_raw_store_descriptor(u.v4) || u.use_pc >= dwords)
                continue;
            Rdna2Inst marker_consumer =
                rdna2_decode_one(code + u.use_pc, dwords - u.use_pc);
            marker_consumer.pc = u.use_pc;
            if (!rdna2_gta5_null_guarded_raw_store_site(marker_consumer)) continue;
            const uint64_t marker_key = 0x8000000400000000ull | u.use_pc;
            if (!seen.insert(marker_key).second) continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Unknown;
            r.num_components = 0;
            r.gpu_addr = 0;
            r.size = 0;
            r.stride = kProvenNullGuardedRawStoreStride;
            r.srt_offset = 0xFFFFFFFFu;
            r.sgpr_base = 0xFFFFFFFFu;
            r.fetch_pc = u.use_pc;
            table.resources.push_back(r);
            continue;
        }
        if (u.zero_record_raw) {
            // Re-check the producer's proof at materialization so a malformed/manually-constructed
            // SrtUse cannot manufacture zero semantics. The resource deliberately carries only
            // exact-PC identity; the emitter never accesses its assigned dummy binding.
            if (u.instruction_format != UINT32_MAX || d.num_records != 0u ||
                d.size_bytes != 0u || u.required_size != 0u)
                continue;
            if (u.use_pc >= dwords)
                continue;
            const Rdna2Inst marker_consumer =
                rdna2_decode_one(code + u.use_pc, dwords - u.use_pc);
            if (marker_consumer.fmt == Rdna2Format::SMEM &&
                !sbuffer_access_is_fully_oob(marker_consumer, d,
                                             u.scalar_oob_offset_known,
                                             u.scalar_oob_byte_offset))
                continue;
            if (marker_consumer.fmt == Rdna2Format::MUBUF && marker_consumer.opcode <= 0x3u &&
                !zero_record_format_selectors_are_zero(marker_consumer, u.v4.data()))
                continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Unknown;
            r.num_components = 0;
            r.gpu_addr = 0;
            r.size = 0;
            r.stride = 0;
            r.srt_offset = 0xFFFFFFFFu;
            r.sgpr_base = 0xFFFFFFFFu;
            r.fetch_pc = u.use_pc;
            for (uint32_t component = 0; component < 4u; ++component)
                r.swizzle[component] = (u.v4[3] >> (component * 3u)) & 0x7u;
            table.resources.push_back(r);
            continue;
        }
        if (d.base <= 0x10000 || d.size_bytes > 0x10000000u) continue;
        const uint32_t scalar_buffer_dwords = validated_scalar_buffer_dword_count(
            u, d, code, dwords);
        if (u.scalar_buffer_dword_count && !scalar_buffer_dwords) continue;
        // A scalar consumer whose base is a RAW POINTER has no bounded V# size to report, so
        // `size_bytes == 0` here means "unbounded", not "empty" (#2412). The graphics builder already
        // makes exactly this distinction and documents it — *"Scalar SMEM only needs V#.Base plus the
        // consuming load's exact byte span"* — and caps the pc-keyed upload to the measured access.
        // Compute dropped the use instead, which is why GTA V's compute programs, whose entire user
        // data is pointers, resolved nothing. Mirror the graphics rule rather than inventing a second
        // one: key-less only, a real measured span, and the same 1 MiB ceiling.
        uint32_t scalar_size = d.size_bytes;
        uint32_t scalar_stride = d.stride;
        if (scalar_size == 0 && !scalar_buffer_dwords) {
            if (u.key != 0xFFFFFFFFu || u.required_size == 0 || u.required_size > (1u << 20))
                continue;
            scalar_size = u.required_size;
            scalar_stride = 0;
        } else {
            // A bounded V# remains authoritative even when a scalar request lies beyond M_SIZE.
            // Widening it to required_size turns an architectural zero into an unrelated host read
            // and was the source of GTA V's spurious 3.70/3.44 GiB compute bindings.
            scalar_size = d.size_bytes;
        }
        if (clash && !exact_mtbuf && !u.atomic_x2_record_count) {
            bool piggybacked = false;
            for (auto& r0 : table.resources) {
                if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                    r0.size != scalar_size ||
                    r0.scalar_buffer_dword_count != scalar_buffer_dwords)
                    continue;
                if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = u.use_pc;
                piggybacked = r0.fetch_pc == u.use_pc;
                if (piggybacked) break;
            }
            if (piggybacked) continue;
        }
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        if (u.instruction_format != UINT32_MAX) {
            rdna2_buffer_format(u.instruction_format, &r.format, &r.num_components);
            if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
        } else if (d.format != DataFormat::Unknown) {
            r.format = d.format;
            r.num_components = d.num_components ? d.num_components : 1;
        } else {
            // A raw-pointer scalar consumer carries no format bits — its V# words 2 and 3 are zero
            // because there is no descriptor, only an address. The access is a run of dwords, which
            // is exactly what the scalar path reads, so describe it as such rather than propagating
            // Unknown into a resource the renderer would then have to guess about (#2412).
            r.format = DataFormat::Uint32;
            r.num_components = 1;
        }
        r.gpu_addr = d.base;
        r.size = scalar_size;
        r.stride = scalar_stride;
        r.scalar_buffer_dword_count = scalar_buffer_dwords;
        if (d.size_bytes == 0u && u.key == UINT32_MAX && !scalar_buffer_dwords)
            r.scalar_raw_pointer_word_hi = u.v4[1];
        r.atomic_x2_record_count = u.atomic_x2_record_count;
        r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
        r.fetch_pc = u.use_pc;
        table.resources.push_back(r);
    }
    return srt_uses;
}

namespace { constexpr uint32_t kPsBindingBase = 32; }

// Preserve the exact resource-path proof across the decoded analysis and raw-byte translation
// passes. Ordinary instruction-scoped resources disappear with their consumers, but the dispatch-
// scoped null marker must survive because the raw translator independently repeats the proof.
ComputeResourcePathSpecializationReport specialize_compute_resource_paths(
        std::vector<Rdna2Inst>& instructions, ShaderResourceTable& resources,
        uint32_t wave_size) {
    ComputeResourcePathSpecializationReport report;
    std::vector<uint32_t> original_pcs;
    original_pcs.reserve(instructions.size());
    for (const Rdna2Inst& instruction : instructions)
        original_pcs.push_back(instruction.pc);

    report.proven_null_exits = rdna2_specialize_proven_null_bvh_paths(
        instructions, &resources, wave_size);
    report.zero_record_execz_exits = rdna2_specialize_zero_record_execz_paths(
        instructions, &resources, wave_size);
    if (!report.proven_null_exits && !report.zero_record_execz_exits) return report;
    report.shader_constant_branches =
        rdna2_specialize_shader_constant_branches(instructions);

    std::unordered_set<uint32_t> live_pcs;
    live_pcs.reserve(instructions.size());
    for (const Rdna2Inst& instruction : instructions)
        live_pcs.insert(instruction.pc);
    for (uint32_t pc : original_pcs)
        if (!live_pcs.contains(pc)) report.removed_pcs.push_back(pc);

    const size_t resources_before = resources.resources.size();
    std::erase_if(resources.resources, [&](const ShaderResource& resource) {
        // The raw-byte translator repeats resource-path specialization. Keep its dispatch-scoped
        // null proof even though the marked IMAGE_BVH instruction disappeared from this analysis
        // stream; all ordinary instruction-scoped resources can be pruned with their consumer.
        return resource.fetch_pc != 0xFFFFFFFFu &&
               !is_proven_null_bvh(resource) &&
               !live_pcs.contains(resource.fetch_pc);
    });
    report.removed_resources = resources_before - resources.resources.size();
    return report;
}

// Assign each resource its OWN descriptor binding, starting at `first` (0/1 reserved). The N-buffer
// model: the shader reads several distinct constant buffers (Unity's per-draw transform, per-frame,
// …) + vertex buffers + textures, and each must land at a separate binding so they don't collapse.
// The recompiler declares a storage buffer (cbuf/vbuf) or image sampler (texture) at each binding and
// resolves an s_buffer_load/image_sample to its resource's binding via provenance.
//
// The VS and PS tables are bound together in ONE descriptor set by the live renderer, so the two
// stages' binding ranges MUST be disjoint (VS keeps 2.., PS starts at kPsBindingBase). Within a
// stage, constant/vertex BUFFERS are assigned first (from `first`), then TEXTURES / storage images —
// but never on binding 2 or 3, which the recompiler's declare_cbufs always occupies with its two
// hardwired storage-buffer cbufs (v_cbuf/v_cbuf1). A shader whose FIRST resource is a texture used to
// land it on binding 2, declaring BOTH a combined image sampler AND that storage buffer at one
// binding (two descriptor types -> layout-creation failure, the draw disappears) (#157). Buffers-
// first keeps the common cbufs-first shaders' bindings byte-identical (cbufs 2/3, textures 4+).
// External linkage (declared in gpu_execute.hpp) so the binding policy is unit-testable.
//
// ONE binding is NOT this policy's to choose: `kComputeInternalGdsBinding`. The recompiler HARD-CODES
// it whenever a program touches GDS (`declare_internal_gds` in rdna2_to_spirv_internal.hpp), so it is
// part of the emitted module's ABI, not a slot to be handed out. A resource that already carries it
// therefore keeps it, and the running counters step over it so nothing else can be given it either.
//
// Without that, a GDS-using COMPUTE program was renumbered out of its own contract: the module
// declared set 0 binding 127 while the table carried that buffer at some sequential number, and the
// descriptor-contract check correctly rejected the pair with `missing runtime binding binding=127`.
// The dispatch was then DECLINED -- every time, for the life of the process. Measured on Astro Bot
// (#3214): five programs skipped for this reason on every boot, including the world-map light-list
// producer `0x5006e8500` at 0 executed of 918 dispatches. Its output is the per-tile light list the
// world-map pixel shader walks, so the shader was handed an all-zero arena, and because that walk
// terminates on 0xffffffff -- not on zero -- an unwritten arena is an INFINITE list. The device hung
// and RADV reset it. A binding-assignment policy is exactly the kind of code whose defects surface a
// very long way from home.
//
// The gap was already visible and was patched one instance at a time: the trip-bound witness's own
// binding-127 resource is injected AFTER this call, at table finalization, with a comment describing
// this same symptom. That fixed the witness and not the guest's own GDS buffer, which is pushed
// beside the `uses_gds` test and so still passed through here.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first) {
    const auto reserved = [](const ShaderResource& r) {
        return r.binding == kComputeInternalGdsBinding;
    };
    uint32_t next = first;
    const auto claim = [&next]() {
        if (next == kComputeInternalGdsBinding) ++next;
        return next++;
    };
    for (auto& r : t.resources)
        if (!reserved(r) &&
            (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer))
            r.binding = claim();
    uint32_t tex_next = next > first + 2 ? next : first + 2;   // reserve the two hardwired cbuf slots
    for (auto& r : t.resources)
        if (!reserved(r) &&
            r.cls != ResourceClass::ConstantBuffer && r.cls != ResourceClass::VertexBuffer) {
            if (tex_next == kComputeInternalGdsBinding) ++tex_next;
            r.binding = tex_next++;
        }
}

std::shared_ptr<ShaderResourceTable> merge_vertex_chain_resource_tables(
        const std::shared_ptr<ShaderResourceTable>& prolog,
        const std::shared_ptr<ShaderResourceTable>& main,
        uint32_t main_pc_offset) {
    if (!prolog && !main) return nullptr;
    auto merged = std::make_shared<ShaderResourceTable>();
    if (prolog) {
        merged->resources = prolog->resources;
        merged->vertices_per_instance = prolog->vertices_per_instance;
    }
    if (main) {
        if (!merged->vertices_per_instance)
            merged->vertices_per_instance = main->vertices_per_instance;
        merged->resources.reserve(merged->resources.size() + main->resources.size());
        for (ShaderResource resource : main->resources) {
            if (resource.fetch_pc != 0xFFFFFFFFu) {
                if (resource.fetch_pc > UINT32_MAX - main_pc_offset) return nullptr;
                resource.fetch_pc += main_pc_offset;
            }
            merged->resources.push_back(std::move(resource));
        }
    }
    assign_convention_bindings(*merged, 2u);
    return merged;
}

std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr,
                                                       bool is_ps, uint32_t draw_vertex_count,
                                                       uint64_t draw_command_order) {
    if (!code_addr) return nullptr;
    const auto* hdr = (const AgcShaderHeader*)prosper_agc_shader_header_for_code(code_addr);
    if (!hdr) return nullptr;
    using StageClock = std::chrono::steady_clock;
    const bool phase_timing = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto metadata_start = phase_timing ? StageClock::now() : StageClock::time_point{};
    namespace P = prosper::agc::Pm4;
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;
    const size_t shader_dwords = registered_shader_dwords(*hdr, code_addr);

    // The V#/T# descriptors live in the stage's user-data SGPR block. The pixel stage uses PS user
    // data; the vertex/geometry stage under NGG merges the ES program's descriptors into GS user data.
    // Try the stage's expected base, then the alternates — the exact stage-merge layout varies, so use
    // whichever base actually yields resources.
    uint32_t bases[3];
    if (is_ps) { bases[0] = P::SPI_SHADER_USER_DATA_PS_0; bases[1] = P::SPI_SHADER_USER_DATA_GS_0; bases[2] = P::SPI_SHADER_USER_DATA_VS_0; }
    else       { bases[0] = P::SPI_SHADER_USER_DATA_GS_0; bases[1] = P::SPI_SHADER_USER_DATA_VS_0; bases[2] = P::SPI_SHADER_USER_DATA_PS_0; }

    // Per-shader user-data RANGE: the shader blob's "specials" block declares which DWORD range of
    // the stage's USER_DATA register block holds this shader's SGPR-visible user data
    // (user_data_range_start/end — the range SetSource programs; e.g. DOLL's UE4 Slate VS declares
    // [0,8) matching its 8-dword {V#, ptr, ptr} block). Header sharp/direct offsets are relative to
    // range_start, so seed the SGPR block from USER_DATA_<stage>_<range_start>. Every shader
    // observed live (DOLL + Messenger) declares start=0, so this is currently behavior-identical —
    // LATENT support for a start!=0 shader, guarded back to 0 on insane metadata.
    // CONFIDENCE: LOW on start!=0 semantics (no live example yet); zero risk for start==0.
    uint32_t range_start = 0;
    if (hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials, sizeof(AgcShaderSpecials))) {
        const uint32_t s = hdr->specials->user_data_range_start;
        const uint32_t e = hdr->specials->user_data_range_end;
        if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        if (log && range_start) {   // once per shader: non-zero ranges are the rare/interesting case
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                fprintf(stderr, "[agc] %s 0x%llx user_data_range=[%u,%u) -> seeding from USER_DATA_%u\n",
                        is_ps ? "PS" : "VS", (unsigned long long)code_addr, s, e, range_start);
        }
    }

    // PROSPER_UD_TAIL_ALIGN (#305 A/B) — FALSIFIED, retained as a documented negative result.
    //
    // The hypothesis was that a stage's user data is the TAIL of the block the pipeline programmed:
    // on Nikoderiko the failing stages' declared descriptors do land on clean guest pointers exactly
    // `programmed - user_data_range_end` dwords above USER_DATA_*_0, and the stages that resolve
    // today are exactly those where `programmed == range_end`. Two independent measurements kill it:
    //
    //  * SPI_SHADER_PGM_RSRC2_GS.USER_SGPR equals `user_data_range_end` for every stage measured
    //    (8/8, 12/12, 20/20, ...). That field is the count of user SGPRs the hardware loads, starting
    //    at USER_DATA_GS_0 — so a stage with range_end=12 physically cannot see GS_12..GS_31, and no
    //    tail alignment can be what the guest intended. It also confirms the existing seeding base
    //    and the 8 leading system SGPRs of the merged-stage ABI are correct.
    //  * A live A/B with the semantics-derived prefix raised Nikoderiko's exec-recompile rejects
    //    well above the unmodified baseline rather than clearing them.
    //
    // Keep the switch so the measurement is reproducible; it must stay off. CONFIDENCE: HIGH that
    // tail alignment is wrong.
    static const char* const tail_mode = std::getenv("PROSPER_UD_TAIL_ALIGN");
    if (tail_mode) {
        const uint32_t e = (hdr->specials &&
                            guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                           sizeof(AgcShaderSpecials)))
                               ? hdr->specials->user_data_range_end : 0u;
        const uint32_t rsrc2 = [&] {
            const auto it = st.sh.find(is_ps ? P::SPI_SHADER_PGM_RSRC2_PS : P::SPI_SHADER_PGM_RSRC2_GS);
            return it == st.sh.end() ? 0u : it->second;
        }();
        uint32_t user_sgprs = (rsrc2 >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT) &
                              P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MASK;
        if (!is_ps)
            user_sgprs |= ((rsrc2 >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_SHIFT) &
                           P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_MASK) << 5;
        // Two candidate sources for the size of the block that precedes this stage's own user data:
        // the hardware user-SGPR count the pipeline programmed, and the shader's own declared vertex
        // input count (one 4-dword V# per input, the fetch block). Report both, drive with one.
        const uint32_t sem_prefix = is_ps ? 0u : hdr->num_input_semantics * 4u;
        const uint32_t rsrc2_prefix = (user_sgprs >= e) ? user_sgprs - e : UINT32_MAX;
        const bool use_sem = tail_mode[0] == 's';
        const uint32_t prefix = use_sem ? sem_prefix : rsrc2_prefix;
        if (log || g_dyntrace_force) {
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                fprintf(stderr,
                        "[udtail] %s 0x%llx rsrc2=0x%08x user_sgprs=%u range_end=%u num_in_sem=%u "
                        "prefix_rsrc2=%d prefix_sem=%u mode=%s\n",
                        is_ps ? "PS" : "VS", (unsigned long long)code_addr, rsrc2, user_sgprs, e,
                        hdr->num_input_semantics, (int)rsrc2_prefix, sem_prefix,
                        use_sem ? "sem" : "rsrc2");
        }
        if (e && prefix != UINT32_MAX && prefix + e <= kUserSgprs) range_start = prefix;
    }

    // PROSPER_RESDUMP: raw dump of the user-data struct + SGPR block per base, so the EUD layout
    // (which sharps have offset_dw>=16, and where the EUD pointer sits) can be read empirically.
    bool resdump = getenv("PROSPER_RESDUMP") != nullptr;
    if (resdump)   // PROSPER_RESDUMP_ADDR=<hex code addr>: narrow the dump to one shader
        if (const char* fa = getenv("PROSPER_RESDUMP_ADDR"))
            resdump = strtoull(fa, nullptr, 16) == code_addr;
    if (g_dyntrace_force) resdump = true;   // failure replay: always dump the failing stage's blocks
    if (resdump) {
        const AgcShaderUserData* ud = hdr->user_data;
        fprintf(stderr, "[resdump] %s code=0x%llx type=%u ud=%p range_start=%u (end=%u)\n",
                is_ps ? "PS" : "VS", (unsigned long long)code_addr, hdr->type, (const void*)ud,
                range_start, hdr->specials ? (uint32_t)hdr->specials->user_data_range_end : 0u);
        if (ud) {
            fprintf(stderr, "[resdump]   eud_size_dw=%u srt_size_dw=%u direct_count=%u sharp_counts={%u,%u,%u,%u}\n",
                    ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                    ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                    ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
            for (int cat = 0; cat < 4; cat++) {
                const AgcShaderSharp* sh = ud->sharp_resource_offset[cat];
                if (!sh || !ud->sharp_resource_count[cat]) continue;
                fprintf(stderr, "[resdump]   sharp[%d] offset_dw:", cat);
                for (uint16_t s = 0; s < ud->sharp_resource_count[cat] && s < 12; s++)
                    fprintf(stderr, " %u%s", sh[s].offset_dw(), sh[s].empty() ? "(empty)" : "");
                fprintf(stderr, "\n");
            }
            if (ud->direct_resource_offset && ud->direct_resource_count) {
                fprintf(stderr, "[resdump]   direct offset_dw:");
                for (uint16_t t2 = 0; t2 < ud->direct_resource_count && t2 < 16; t2++)
                    fprintf(stderr, " [%u]=%u", t2, ud->direct_resource_offset[t2]);
                fprintf(stderr, "\n");
            }
        }
        for (uint32_t base : bases) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
            fprintf(stderr, "[resdump] %s code=0x%llx sgprs@0x%x:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base);
            for (uint32_t i = 0; i < kUserSgprs; i++) fprintf(stderr, " %08x", sgprs[i]);
            fprintf(stderr, "\n");
        }
        // USER-DATA POINTER MAP. A stage's descriptor-table pointers are ordinary 64-bit guest
        // addresses sitting in consecutive user-data dwords, and every bindless descriptor chain
        // starts by dereferencing one. When a stage fails to resolve its V#/T#, the first question
        // is whether the seeded block even CONTAINS the pointers the shader loads from — the AGC
        // header's direct-resource offsets and the shader's own SBASE registers both name dword
        // positions, so a block whose readable pointers sit elsewhere is not the block that shader
        // ran with. Report every dword pair that is a mapped guest address, plus the readability of
        // each header-declared direct offset, so that question is answered by data rather than by
        // eye. Probing is bounded to the 32-dword window and uses the fault-safe readability probe.
        for (uint32_t base : bases) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
            fprintf(stderr, "[udmap] %s code=0x%llx base=0x%x readable-ptr dwords:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base);
            bool any = false;
            for (uint32_t i = 0; i + 1 < kUserSgprs; i++) {
                const uint64_t candidate =
                    (uint64_t)sgprs[i] | ((uint64_t)sgprs[i + 1] << 32);
                if (candidate <= 0x10000 || !guest_readable(candidate, 8)) continue;
                fprintf(stderr, " [%u]=0x%llx", i, (unsigned long long)candidate);
                any = true;
            }
            if (!any) fprintf(stderr, " none");
            fprintf(stderr, "\n");
        }
        // A stage pointer may carry aperture/tag bits above the title's usable GPU VA, exactly as the
        // scalar fold's S_LOAD canonicalization assumes. Accept a dword pair as a pointer when the
        // raw 64-bit value or either canonical form is mapped, so a tagged-but-valid table pointer is
        // never reported as unmapped.
        const auto pointer_is_mapped = [](uint64_t value) {
            for (uint64_t mask : {~uint64_t{0}, uint64_t{0xFFFFFFFFFFFF}, uint64_t{0xFFFFFFFFFF}}) {
                const uint64_t candidate = value & mask;
                if (candidate > 0x10000 && guest_readable(candidate, 8)) return true;
            }
            return false;
        };
        if (const AgcShaderUserData* ud = hdr->user_data) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
            fprintf(stderr, "[udmap] %s code=0x%llx declared direct:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr);
            for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; t++) {
                const uint32_t off = ud->direct_resource_offset[t];
                if (off == 0xFFFFu || off + 1 >= kUserSgprs) continue;
                const uint64_t candidate =
                    (uint64_t)sgprs[off] | ((uint64_t)sgprs[off + 1] << 32);
                fprintf(stderr, " [%u]@dw%u=0x%llx(%s)", t, off,
                        (unsigned long long)candidate,
                        pointer_is_mapped(candidate) ? "readable" : "UNMAPPED");
            }
            // Self-validating half: search the 32-dword window for the ONE seed offset at which
            // EVERY declared direct pointer becomes a mapped guest address. If that offset equals
            // the shader's raw user_data_range_start, the block is simply seeded from the wrong
            // register and the metadata already said so; if no offset works, the block genuinely
            // does not contain this shader's pointers and the seeding origin is not the defect.
            uint32_t implied = UINT32_MAX;
            uint32_t declared_pointers = 0;
            for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; t++)
                if (ud->direct_resource_offset[t] != 0xFFFFu) ++declared_pointers;
            for (uint32_t seed = 0; declared_pointers && seed < kUserSgprs; ++seed) {
                uint32_t probe[kUserSgprs];
                read_user_sgprs(st.sh, bases[0] + seed, probe);
                bool all_mapped = true;
                for (uint16_t t = 0; all_mapped && t < ud->direct_resource_count && t < 16; t++) {
                    const uint32_t off = ud->direct_resource_offset[t];
                    if (off == 0xFFFFu) continue;
                    if (off + 1 >= kUserSgprs) { all_mapped = false; break; }
                    const uint64_t value =
                        (uint64_t)probe[off] | ((uint64_t)probe[off + 1] << 32);
                    all_mapped = pointer_is_mapped(value);
                }
                if (all_mapped) { implied = seed; break; }
            }
            const uint32_t raw_start = hdr->specials ? hdr->specials->user_data_range_start : 0u;
            const uint32_t raw_end = hdr->specials ? hdr->specials->user_data_range_end : 0u;
            fprintf(stderr, " | specials raw=[%u,%u) seeded=%u implied=",
                    raw_start, raw_end, range_start);
            if (implied == UINT32_MAX) fprintf(stderr, "none");
            else fprintf(stderr, "%u", implied);
            fprintf(stderr, "\n");
        }
        // #305 instrument: the code address alone does not identify a shader. CreateShader's registry
        // is append-only and the header lookup returns the FIRST registration bound to an address, so
        // a guest that re-registers a different shader over a recycled code allocation resolves to the
        // OLDEST layout while the register block holds the newest bind. Enumerate every candidate and
        // report, for each, whether ALL of its declared direct offsets land on mapped pointers in the
        // block this draw actually has — the candidate that does is the layout the guest programmed.
        {
            const void* cands[16] = {};
            const size_t total =
                prosper_agc_shader_headers_for_code(code_addr, cands, std::size(cands));
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
            fprintf(stderr, "[udcand] %s code=0x%llx registrations=%zu\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, total);
            for (size_t ci = 0; ci < total && ci < std::size(cands); ci++) {
                const auto* ch = static_cast<const AgcShaderHeader*>(cands[ci]);
                // This loop exists to enumerate EXTRA registrations at one code address — i.e. the
                // recycled-allocation case — so its entries are the ones most likely to point into
                // a blob the guest has already freed. Probe every hop, exactly as the whole-registry
                // scan below does.
                if (!ch || !guest_readable((uint64_t)(uintptr_t)ch, sizeof(AgcShaderHeader)))
                    continue;
                const bool sp_ok = ch->specials && guest_readable((uint64_t)(uintptr_t)ch->specials,
                                                                  sizeof(AgcShaderSpecials));
                fprintf(stderr, "[udcand]   #%zu hdr=%p type=%u range=[%u,%u) direct:", ci,
                        (const void*)ch, ch->type,
                        sp_ok ? (uint32_t)ch->specials->user_data_range_start : 0u,
                        sp_ok ? (uint32_t)ch->specials->user_data_range_end : 0u);
                bool all = true, any = false;
                const AgcShaderUserData* cud = ch->user_data;
                if (cud && !guest_readable((uint64_t)(uintptr_t)cud, sizeof(AgcShaderUserData)))
                    cud = nullptr;
                uint16_t cud_count = 0;
                const uint16_t* cud_offsets = nullptr;
                if (cud) {
                    cud_count = cud->direct_resource_count;
                    cud_offsets = cud->direct_resource_offset;
                    if (!cud_offsets || !cud_count || cud_count > 64 ||
                        !guest_readable((uint64_t)(uintptr_t)cud_offsets,
                                        cud_count * (uint32_t)sizeof(uint16_t)))
                        cud = nullptr;
                }
                if (cud) {
                    for (uint16_t t = 0; t < cud_count && t < 16; t++) {
                        const uint32_t off = cud_offsets[t];
                        if (off == 0xFFFFu) continue;
                        any = true;
                        if (off + 1 >= kUserSgprs) { all = false; fprintf(stderr, " [%u]@dw%u=OOB", t, off); continue; }
                        const uint64_t v = (uint64_t)sgprs[off] | ((uint64_t)sgprs[off + 1] << 32);
                        const bool ok = pointer_is_mapped(v);
                        all = all && ok;
                        fprintf(stderr, " [%u]@dw%u=0x%llx(%s)", t, off, (unsigned long long)v,
                                ok ? "readable" : "UNMAPPED");
                    }
                    fprintf(stderr, " sharps={%u,%u,%u,%u}", cud->sharp_resource_count[0],
                            cud->sharp_resource_count[1], cud->sharp_resource_count[2],
                            cud->sharp_resource_count[3]);
                }
                fprintf(stderr, " -> %s\n", !any ? "no-direct" : (all ? "FITS-BLOCK" : "misfit"));
            }
            // Whole-registry search: the block this draw programmed is a fact; the header is an
            // inference from the PGM register. If the resolved header does not fit the block but
            // some OTHER registered shader's declared layout does — and its declared size equals the
            // contiguous extent this draw's own bind freshly wrote — then the PGM register, not the
            // register block, is what is stale. `fresh` is that extent, from the write-provenance
            // instrument (dwords sharing the newest write order, from dw0 up).
            uint32_t fresh = 0;
            if (prosper::gpu::udprov_enabled()) {
                uint64_t newest = 0;
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const auto it = st.sh_prov.find(bases[0] + i);
                    if (it != st.sh_prov.end())
                        newest = std::max(newest, it->second & ~GpuState::kProvIndirect);
                }
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const auto it = st.sh_prov.find(bases[0] + i);
                    if (it == st.sh_prov.end() ||
                        (it->second & ~GpuState::kProvIndirect) != newest) break;
                    fresh = i + 1;
                }
            }
            const size_t registry = prosper_agc_shader_count();
            fprintf(stderr, "[udcand]   block: fresh_extent=%u registry=%zu fitting:", fresh, registry);
            size_t fits = 0;
            for (size_t si = 0; si < registry; si++) {
                const auto* sh = static_cast<const AgcShaderHeader*>(prosper_agc_shader_at(si));
                // This walks the WHOLE registry (thousands of entries), and every pointer in it is
                // guest-owned metadata that may point into a blob the guest has since freed. Probe
                // each hop before dereferencing it — the resolved-header path above only ever
                // touches one entry, this one touches all of them.
                if (!sh || !guest_readable((uint64_t)(uintptr_t)sh, sizeof(AgcShaderHeader)))
                    continue;
                if (!sh->user_data ||
                    !guest_readable((uint64_t)(uintptr_t)sh->user_data, sizeof(AgcShaderUserData)))
                    continue;
                const uint16_t direct_count = sh->user_data->direct_resource_count;
                const uint16_t* direct_offsets = sh->user_data->direct_resource_offset;
                if (!direct_offsets || !direct_count || direct_count > 64 ||
                    !guest_readable((uint64_t)(uintptr_t)direct_offsets,
                                    direct_count * (uint32_t)sizeof(uint16_t)))
                    continue;
                if (!sh->specials ||
                    !guest_readable((uint64_t)(uintptr_t)sh->specials, sizeof(AgcShaderSpecials)))
                    continue;
                const uint32_t end = sh->specials->user_data_range_end;
                if (!fresh || end != fresh) continue;   // must match what this bind actually wrote
                bool all = true, any = false;
                for (uint16_t t = 0; all && t < direct_count && t < 16; t++) {
                    const uint32_t off = direct_offsets[t];
                    if (off == 0xFFFFu) continue;
                    any = true;
                    if (off + 1 >= kUserSgprs) { all = false; break; }
                    all = pointer_is_mapped((uint64_t)sgprs[off] |
                                            ((uint64_t)sgprs[off + 1] << 32));
                }
                if (!any || !all) continue;
                if (fits++ < 8)
                    fprintf(stderr, " 0x%llx(type=%u,end=%u)",
                            (unsigned long long)(uintptr_t)sh->code, sh->type, end);
            }
            if (!fits) fprintf(stderr, " none");
            fprintf(stderr, " total=%zu\n", fits);
        }
        // PROSPER_UDPROV (#305): write provenance for the block this stage was seeded from, plus the
        // program-address registers that named the stage. "The block holds a previous pipeline's
        // user data" is a claim about WHEN each dword was last written relative to the draw — values
        // alone cannot express it. Print, per dword, the value and the command_order of its last
        // write (i = indirect path, d = direct), the orders of the PGM registers that selected this
        // shader, and the draw's own order.
        if (prosper::gpu::udprov_enabled()) {
            // order + path + SOURCE. `q` is the queue origin the packet folded under
            // (0=unknown/graphics, 1=Dcb, 2=Acb, 3=DcbFinal), `f` the top-level fold (submit
            // stream) id, `j` the sceAgcDcbJump recursion depth. A write whose q/f differs from
            // the draw's own is one that did not arrive in this submit's inline position.
            const auto prov = [&](uint32_t reg) -> std::string {
                const auto it = st.sh_prov.find(reg);
                if (it == st.sh_prov.end()) return "never";
                char buf[64];
                const auto sit = st.sh_prov_src.find(reg);
                const uint64_t src = sit == st.sh_prov_src.end() ? 0 : sit->second;
                snprintf(buf, sizeof(buf), "%c%llu/q%u,f%llu,j%u",
                         (it->second & GpuState::kProvIndirect) ? 'i' : 'd',
                         (unsigned long long)(it->second & ~GpuState::kProvIndirect),
                         (unsigned)(src & 0xFFu), (unsigned long long)(src >> 16),
                         (unsigned)((src >> 8) & 0xFFu));
                return buf;
            };
            const auto shv = [&](uint32_t reg) {
                const auto it = st.sh.find(reg);
                return it == st.sh.end() ? 0u : it->second;
            };
            fprintf(stderr,
                    "[udprov] %s code=0x%llx draw_order=%llu pgm: ES_LO=0x%x@%s ES_HI=0x%x@%s "
                    "GS_LO=0x%x@%s GS_HI=0x%x@%s PS_LO=0x%x@%s\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr,
                    (unsigned long long)st.command_order,
                    shv(P::SPI_SHADER_PGM_LO_ES), prov(P::SPI_SHADER_PGM_LO_ES).c_str(),
                    shv(P::SPI_SHADER_PGM_HI_ES), prov(P::SPI_SHADER_PGM_HI_ES).c_str(),
                    shv(P::SPI_SHADER_PGM_LO_GS), prov(P::SPI_SHADER_PGM_LO_GS).c_str(),
                    shv(P::SPI_SHADER_PGM_HI_GS), prov(P::SPI_SHADER_PGM_HI_GS).c_str(),
                    shv(P::SPI_SHADER_PGM_LO_PS), prov(P::SPI_SHADER_PGM_LO_PS).c_str());
            {
                const auto cxv = [&](uint32_t reg) {
                    const auto it = st.cx.find(reg);
                    return it == st.cx.end() ? 0u : it->second;
                };
                fprintf(stderr,
                        "[udprov]   stages: HS_LO=0x%x@%s LS_LO=0x%x@%s RSRC2_GS=0x%x@%s "
                        "VGT_SHADER_STAGES_EN=0x%x\n",
                        shv(P::SPI_SHADER_PGM_LO_HS), prov(P::SPI_SHADER_PGM_LO_HS).c_str(),
                        shv(P::SPI_SHADER_PGM_LO_LS), prov(P::SPI_SHADER_PGM_LO_LS).c_str(),
                        shv(P::SPI_SHADER_PGM_RSRC2_GS), prov(P::SPI_SHADER_PGM_RSRC2_GS).c_str(),
                        cxv(P::VGT_SHADER_STAGES_EN));
            }
            for (uint32_t bi = 0; bi < std::size(bases); ++bi) {
                fprintf(stderr, "[udprov]   base=0x%x:", bases[bi]);
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const uint32_t reg = bases[bi] + i;
                    fprintf(stderr, " dw%u=0x%08x@%s", i, shv(reg), prov(reg).c_str());
                }
                fprintf(stderr, "\n");
            }
        }
        // ALL set sh registers (sorted) — finds where the user-data SGPRs actually landed, including
        // any at unexpected offsets (a wrong indirect-register decode would scatter them).
        fprintf(stderr, "[resdump]   %zu sh regs set; lowest 48 offsets:", st.sh.size());
        std::vector<uint32_t> keys; for (auto& kv : st.sh) if (kv.second) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size() && i < 48; i++) fprintf(stderr, " 0x%x", keys[i]);
        fprintf(stderr, "\n");
    }

    // Bindless-dynamic buffer fetch: const-fold the scalar setup to resolve each buffer_load_format's
    // V#, then emit it as a VertexBuffer keyed by its SRSRC SGPR so the recompiler's by_sgpr_base resolves
    // it. Despite the historical vertex-fetch name, pixel shaders use the same instructions for UE4
    // structured/material buffers. Dropping the PS results leaves those loads with no storage-buffer
    // binding, so the recompiler falls back to its legacy binding 2 and reads zeros (#719).
    // The SAME const-fold also recovers descriptor-TABLE uses for BOTH stages (#294): UE4 shaders
    // s_load their T#/S#/V# descriptors from a table pointer in the user-data SGPRs and consume them
    // via image_sample / s_buffer_load — srt_uses reports each with its load-immediate key, which
    // becomes the resource's srt_offset (the recompiler's by_srt_offset provenance).
    std::vector<DynFetch> dyn_vb;
    std::vector<SrtUse> srt_uses;
    // Build the primary metadata table before dynamic folding so a proven PS dispatch can read its
    // direct selector resource. The fold then follows the same selected arm as recompilation.
    const uint32_t user_sgpr_base = is_ps ? 0u : 8u;
    uint32_t primary_sgprs[kUserSgprs];
    read_user_sgprs(st.sh, bases[0] + range_start, primary_sgprs);
    ShaderResourceTable primary_resources =
        build_shader_resources(*hdr, primary_sgprs, kUserSgprs, user_sgpr_base);
    PcrelDispatchSelection dispatch_selection;
    std::shared_ptr<const ShaderCodeAnalysis> shader_analysis;
    if (is_ps) {
        shader_analysis = analyze_shader_code_cached(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)), shader_dwords);
        dispatch_selection = select_pcrel_dispatch(
            (const uint32_t*)(uintptr_t)code_addr, shader_dwords, &primary_resources,
            shader_analysis.get());
    }
    const auto metadata_done = phase_timing ? StageClock::now() : StageClock::time_point{};
    if (is_ps) {
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                       primary_sgprs, kUserSgprs, 0, &srt_uses,
                                       dispatch_selection.target, &dispatch_selection.dispatch);
    } else {
        // NGG merged VS/GS: s0..s7 are system SGPRs, user data starts at s8 (confirmed by matching the
        // shader's s[8:11]/s[24:25] descriptor pointers to the register file at GS_0+offset).
        uint32_t system_sgprs[2] = {};
        uint32_t system_count = 0;
        if (hdr->type == 6) { // fused GS back: s[0:1] points at the driver stage-data table
            const auto sh_value = [&](uint32_t reg) {
                const auto found = st.sh.find(reg);
                return found == st.sh.end() ? 0u : found->second;
            };
            system_sgprs[0] = sh_value(P::SPI_SHADER_USER_DATA_ADDR_LO_GS);
            system_sgprs[1] = sh_value(P::SPI_SHADER_USER_DATA_ADDR_HI_GS);
            system_count = (system_sgprs[0] || system_sgprs[1]) ? 2u : 0u;
        }
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                       primary_sgprs, kUserSgprs, 8, &srt_uses,
                                       UINT32_MAX, nullptr, system_sgprs, system_count);
        if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_RESDUMP")) {
            fprintf(stderr, "[dynvb] VS resolved %zu dynamic vertex-fetch descriptor(s):\n", dyn_vb.size());
            for (auto& kv : dyn_vb) {
                const auto& d = kv.desc;
                fprintf(stderr, "[dynvb]   pc=%u SRSRC s%d -> base=0x%llx stride=%u num_records=%u size=%u fmt=%u nc=%u\n",
                        kv.fetch_pc, kv.srsrc, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                        (unsigned)d.format, d.num_components);
                if (guest_readable(d.base, 64)) {   // raw dwords of vertex 0 (to read packed formats) + floats
                    const uint32_t* u = (const uint32_t*)(uintptr_t)d.base;
                    fprintf(stderr, "[dynvb]     v3fmt=0x%x  raw v0: %08x %08x %08x %08x\n",
                            kv.desc_v3, u[0], u[1], u[2], u[3]);
                    // Print the first 3 vertex records (the DrawIndexAuto count=3 triangle) as floats, so the
                    // actual on/off-screen span can be computed offline. Each record is `stride` bytes.
                    for (int rec = 0; rec < 3 && d.stride; rec++) {
                        uint64_t a = d.base + (uint64_t)rec * d.stride;
                        if (!guest_readable(a, 16)) break;
                        const float* f = (const float*)(uintptr_t)a;
                        fprintf(stderr, "[dynvb]     rec%d: %.3f %.3f %.3f %.3f\n", rec, f[0], f[1], f[2], f[3]);
                    }
                }
            }
        }
    }

    const auto fold_done = phase_timing ? StageClock::now() : StageClock::time_point{};
    auto record_phases = [&] {
        if (!phase_timing) return;
        const auto resources_done = StageClock::now();
        record_stage_table_phases(
            std::chrono::duration<double, std::milli>(metadata_done - metadata_start).count(),
            std::chrono::duration<double, std::milli>(fold_done - metadata_done).count(),
            std::chrono::duration<double, std::milli>(resources_done - fold_done).count());
    };

    // NGG VS/GS loads user data at shader s8 (s0..s7 = system SGPRs); PS at s0. The resource sgpr_base
    // (an s_buffer_load/image_sample's SBASE/SRSRC register) is in that shader-SGPR space.
    for (size_t base_index = 0; base_index < std::size(bases); ++base_index) {
        const uint32_t base = bases[base_index];
        ShaderResourceTable t;
        if (base_index == 0) {
            t = std::move(primary_resources);
        } else {
            uint32_t sgprs[kUserSgprs];
            read_user_sgprs(st.sh, base + range_start, sgprs);
            t = build_shader_resources(*hdr, sgprs, kUserSgprs, user_sgpr_base);
        }
        // Add the const-fold-resolved dynamic buffers, keyed by their SRSRC SGPR so the
        // recompiler's by_sgpr_base() resolves each buffer_load_format. The V#'s data format is patched
        // at runtime by the fetch shader (so the load-time snapshot reads Unknown) — default to Float32
        // (a raw 32-bit-per-component fetch, correct for float attributes like positions).
        for (auto& kv : dyn_vb) {
            // Only ABI-proven vertex/instance fetches use the special address path that replaces
            // VADDR with a built-in index and therefore needs OFFSET/SOFFSET folded into the bound
            // base. A shader-computed VADDR keeps the instruction's complete address expression,
            // including idxen+offen's second VGPR, so bind the original base for that mode.
            const auto& d = kv.index_mode == VertexFetchIndexMode::Shader
                          ? kv.unshifted_desc : kv.desc;
            if (kv.instruction_format != UINT32_MAX &&
                ((kv.desc_v3 >> 12) & 0x7Fu) == 0)
                continue;
            // Belt-and-suspenders: resolve_dynamic_fetch filters these before emitting DynFetch, but
            // never allow a deliberately rejected packed descriptor to reach the generic Float32
            // fallback if another producer constructs a DynFetch in the future.
            if (kv.instruction_format == UINT32_MAX && d.forbid_unknown_fallback) continue;
            // A SEED-fallback entry must not shadow a metadata-described DIRECT vertex buffer at the
            // same SGPRs (see DynFetch::from_seed): the direct resource resolves the fetch through
            // the faithful address path, which is the correct model for a single un-patched V#.
            if (kv.from_seed && kv.instruction_format == UINT32_MAX) {
                bool direct_exists = false;
                for (const auto& r0 : t.resources)
                    if (r0.cls == ResourceClass::VertexBuffer && r0.sgpr_base == (uint32_t)kv.srsrc)
                        { direct_exists = true; break; }
                if (direct_exists) continue;
            }
            ShaderResource r;
            // A pixel-stage buffer_load_format_* is a structured/material-buffer read, not a
            // vertex attribute fetch. Keeping it as ConstantBuffer preserves the instruction's
            // computed VADDR/stride address; labeling it VertexBuffer makes the recompiler take
            // the gl_VertexIndex shortcut and rejects valid stride-2 uint16 tables (#719).
            r.cls           = is_ps ? ResourceClass::ConstantBuffer : ResourceClass::VertexBuffer;
            if (kv.instruction_format != UINT32_MAX) {
                rdna2_buffer_format(kv.instruction_format, &r.format, &r.num_components);
                if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
            } else {
                r.format = (d.format == DataFormat::Unknown) ? DataFormat::Float32 : d.format;
                r.num_components = d.num_components ? d.num_components : 4;
            }
            r.gpu_addr      = d.base;
            if (d.size_bytes) {
                r.size = d.size_bytes;
            } else if (is_ps) {
                // A pixel-stage buffer_load_format_* retains its real VADDR/stride addressing as a
                // ConstantBuffer. Fragment/material indices are unrelated to submitted vertex count,
                // so applying the VS draw-derived bound here can truncate an access that the previous
                // compatibility allocation covered. Keep that allocation for PS until the fold can
                // prove an access range, and surface the uncertainty below.
                r.size = d.stride ? d.stride * 4u : 128u;
            } else {
                const bool packed_word =
                    r.format == DataFormat::Float10_11_11 ||
                    r.format == DataFormat::Unorm2_10_10_10 ||
                    r.format == DataFormat::Snorm2_10_10_10 ||
                    r.format == DataFormat::Uint2_10_10_10 ||
                    r.format == DataFormat::Sint2_10_10_10;
                const uint64_t element_bytes = packed_word ? 4u :
                    static_cast<uint64_t>(data_format_bytes(r.format)) * r.num_components;
                const uint64_t record_bytes = d.stride ? d.stride : element_bytes;
                const uint64_t draw_bytes = record_bytes * draw_vertex_count;
                r.size = static_cast<uint32_t>(std::min<uint64_t>(draw_bytes, 0x10000000ull));
            }
            r.stride        = d.stride;
            r.sgpr_base     = kv.srsrc;           // DIRECT provenance = the fetch's SRSRC SGPR (fallback)
            r.fetch_pc      = kv.fetch_pc;        // PER-FETCH provenance = the exact fetch instruction
            r.fetch_index_mode = kv.index_mode;
            r.srt_offset    = 0xFFFFFFFFu;
            // The dynamic fold may resolve a descriptor loaded or patched inside the shader. Only
            // an unchanged four-dword entry seed has one exact raw PM4 SH source. Keep the primary
            // entry range that actually fed the fold; do not search other stage bases by whether
            // their decoded output happens to equal this normalized resource (#1853).
            if (kv.direct_user_data_index <= kUserSgprs - 4u) {
                r.direct_vsharp_sh_register_base =
                    bases[0] + range_start + kv.direct_user_data_index;
            } else {
                r.direct_vsharp_sh_register_base =
                    ShaderResource::kDirectVSharpOriginAmbiguous;
            }
            if (!d.size_bytes && is_ps) {
                static std::mutex zero_ps_mx;
                static std::set<std::tuple<uint64_t, uint32_t, uint32_t>> zero_ps_seen;
                std::lock_guard<std::mutex> lk(zero_ps_mx);
                if (zero_ps_seen.emplace(code_addr, kv.fetch_pc, kv.desc_v3).second)
                    std::fprintf(stderr,
                                 "[dynvb] PS code=0x%llx fetch pc=%u has zero-record V#; "
                                 "VADDR range is not derivable from draw vertices, preserving "
                                 "compatibility size=%u\n",
                                 (unsigned long long)code_addr, kv.fetch_pc, r.size);
            }
            if (kv.instruction_format == UINT32_MAX && d.format == DataFormat::Unknown) {
                static std::mutex unknown_mx;
                static std::set<std::tuple<uint64_t, bool, uint32_t, uint32_t>> unknown_seen;
                std::lock_guard<std::mutex> lk(unknown_mx);
                if (unknown_seen.emplace(code_addr, is_ps, kv.fetch_pc, kv.desc_v3).second)
                    std::fprintf(stderr,
                                 "[dynvb] %s code=0x%llx fetch pc=%u has unknown V# format 0x%x; "
                                 "using Float32x4 (draw_vertices=%u size=%u)\n",
                                 is_ps ? "PS" : "VS", (unsigned long long)code_addr,
                                 kv.fetch_pc, (kv.desc_v3 >> 12) & 0x7fu,
                                 draw_vertex_count, r.size);
            }
            t.resources.push_back(r);
        }
        // Descriptor-TABLE resources (#294): one ShaderResource per distinct table use, keyed by the
        // s_load immediate (srt_offset) so the recompiler's sreg_srt/by_srt_offset provenance resolves
        // the consuming image_sample / s_buffer_load. Never shadow an existing resource at the same
        // srt_offset (the EUD-sharp path may already have emitted it — first match wins in
        // by_srt_offset, and two DIFFERENT tables reusing one immediate would be ambiguous anyway).
        {
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                // Dedupe: a KEYED cbuf use per key (the s_buffer_load resolves by key); texture and
                // key-less buffer uses per CONSUMING INSTRUCTION (#273 — several image ops may share
                // one key, or have none; a key-less V# fetch resolves by its pc).
                // Distinct namespaces: pc keys must never collide with byte-offset keys.
                const bool exact_mtbuf = u.kind == 1 && u.instruction_format != UINT32_MAX;
                uint64_t dk = (u.kind == 0 || u.key == 0xFFFFFFFFu || exact_mtbuf)
                                  ? (0x8000000000000000ull | ((uint64_t)(uint32_t)u.kind << 32) | u.use_pc)
                                  : ((uint64_t)(uint32_t)u.kind << 32) | u.key;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = exact_mtbuf || u.key == 0xFFFFFFFFu;
                if (!clash)
                    for (const auto& r0 : t.resources) if (r0.srt_offset == u.key) { clash = true; break; }
                if (u.kind == 1) {                       // constant buffer / structured-buffer V#
                    DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                    if (u.instruction_format != UINT32_MAX &&
                        ((u.v4[3] >> 12) & 0x7Fu) == 0)
                        continue;
                    if (d.base <= 0x10000) continue;
                    const uint32_t scalar_buffer_dwords =
                        validated_scalar_buffer_dword_count(
                            u, d,
                            reinterpret_cast<const uint32_t*>(
                                static_cast<uintptr_t>(code_addr)),
                            shader_dwords);
                    if (u.scalar_buffer_dword_count && !scalar_buffer_dwords) continue;
                    uint32_t resource_size = d.size_bytes;
                    uint32_t resource_stride = d.stride;
                    if (scalar_buffer_dwords && resource_size > 0x10000000u) {
                        continue;
                    } else if (!scalar_buffer_dwords &&
                               (resource_size == 0 || resource_size > 0x10000000u)) {
                        // Scalar SMEM only needs V#.Base plus the consuming load's exact byte span.
                        // DOLL uses base-valid sharps whose other words do not describe a conventional
                        // bounded buffer. Cap the pc-keyed upload to the observed access; the renderer's
                        // safe copy preserves its usual zero-fill behavior for an unavailable guest page.
                        if (u.key != 0xFFFFFFFFu || u.required_size == 0 ||
                            u.required_size > (1u << 20)) continue;
                        resource_size = u.required_size;
                        resource_stride = 0;
                    }
                    // A keyed use whose key already resolves keeps the existing resource; a key-less
                    // (or key-clashed) use still needs a pc-provenance entry — piggyback the pc onto
                    // an existing resource describing the SAME buffer, else create one (#273).
                    if (clash && !exact_mtbuf) {
                        bool piggybacked = false;
                        for (auto& r0 : t.resources)
                            if ((r0.cls == ResourceClass::ConstantBuffer || r0.cls == ResourceClass::VertexBuffer) &&
                                r0.gpu_addr == d.base && r0.size == resource_size &&
                                r0.stride == resource_stride &&
                                r0.scalar_buffer_dword_count == scalar_buffer_dwords) {
                                if (r0.fetch_pc == 0xFFFFFFFFu && r0.cls == ResourceClass::ConstantBuffer)
                                    r0.fetch_pc = u.use_pc;
                                piggybacked = r0.fetch_pc == u.use_pc || r0.cls == ResourceClass::VertexBuffer;
                                if (piggybacked) break;
                            }
                        if (piggybacked) continue;
                    }
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    if (u.instruction_format != UINT32_MAX) {
                        rdna2_buffer_format(u.instruction_format, &r.format, &r.num_components);
                        if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
                    } else {
                        r.format = d.format;
                        r.num_components = d.num_components ? d.num_components : 1;
                    }
                    r.gpu_addr = d.base; r.size = resource_size; r.stride = resource_stride;
                    r.scalar_buffer_dword_count = scalar_buffer_dwords;
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    if (clash) r.fetch_pc = u.use_pc;    // pc-only provenance (key-less/collided V#)
                    if (log) fprintf(stderr, "[srt] %s cbuf key=0x%x pc=%u base=0x%llx size=%u\n", is_ps ? "PS" : "VS",
                                     u.key, u.use_pc, (unsigned long long)d.base, resource_size);
                    t.resources.push_back(r);
                } else {                                  // sampled texture or storage image (T# [+ paired S#])
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    const char* reject = image_descriptor_reject_reason(d);
                    if (g_dyntrace_force)
                        fprintf(stderr, "[dynfail] tex use pc=%u key=0x%x base=0x%llx %ux%u type=%u "
                                        "base_array=%u fmt=%u tile=%u -> %s\n",
                                u.use_pc, u.key, (unsigned long long)d.base, d.width, d.height,
                                d.type, d.base_array, d.format, d.tile_mode,
                                reject ? reject : "materialize");
                    // An EXACTLY all-zero eight-dword T# is not garbage: it is the guest explicitly
                    // binding a null image, and `validate_shader_resources` already calls that valid
                    // state -- "Explicit null descriptors are valid guest state: resource reads return
                    // zero and the backend binds a zero-filled dummy. An absent table entry is still an
                    // error above." (shader_resources.cpp). Dropping it here made the entry ABSENT,
                    // which is the error case, so the consuming MIMG found nothing by fetch_pc and the
                    // whole shader was rejected.
                    //
                    // Measured on GTA V routed gameplay: 11 of the 18 terminal-MIMG fragment pairs
                    // reach this line with an exactly-known `t8 = 00000000 x8`, carrying 1,328 of the
                    // rejected-draw occurrences (#2422).
                    //
                    // Deliberately narrow, and each clause is load-bearing:
                    //   * EXACTLY zero, all eight dwords -- a nonzero-but-implausible base stays
                    //     fail-visible, which is what the base-zero/low-base screen exists for.
                    //   * proven descriptor provenance only -- either one table load or exact direct
                    //     user-data provenance. An UNKNOWN T# must not be silently turned into a
                    //     null bind.
                    //   * sampled reads only. A storage image or atomic reaching a null descriptor is
                    //     a WRITE to nowhere; that stays rejected rather than being made to look
                    //     handled.
                    const bool exact_null_t8 =
                        reject && std::string_view(reject) == "base-zero" && !u.is_storage_image &&
                        std::all_of(u.t8.begin(), u.t8.end(), [](uint32_t w) { return w == 0u; });
                    if (exact_null_t8) {
                        ShaderResource rn;
                        rn.cls      = ResourceClass::Texture;
                        rn.gpu_addr = 0;            // the three fields validate_shader_resources reads
                        rn.size     = 0;            // to recognise an explicit null descriptor
                        rn.fetch_pc = u.use_pc;     // exact-PC provenance: only THIS use resolves to it
                        rn.srt_offset = 0xFFFFFFFFu;
                        rn.sgpr_base  = 0xFFFFFFFFu;
                        // Gated on PROSPER_DBG, not PROSPER_GFXLOG: GFXLOG must not be added to the
                        // timing-dependent GTA V route without re-establishing its baseline, so an
                        // instrument only visible under it cannot be read in the run being measured.
                        // Publishing a null image is rare and consequential enough to say so.
                        if (std::getenv("PROSPER_DBG"))
                            fprintf(stderr, "[srt] %s null-image pc=%u key=0x%x (exact all-zero T#)\n",
                                    is_ps ? "PS" : "VS", u.use_pc, u.key);
                        record_null_image_source_probe(
                            code_addr, u.use_pc, u.descriptor_source_addr,
                            draw_command_order, u.t8);
                        t.resources.push_back(rn);
                        continue;
                    }
                    if (reject) continue;                                    // garbage/degenerate T#
                    // A previous use already produced a resource for this SAME selected view (address +
                    // extent): don't duplicate the binding/upload — give it this use's pc provenance
                    // if it has none yet (#273). If it already carries a DIFFERENT use's pc, fall
                    // through and create a second resource for this pc (fetch_pc holds one pc; a
                    // sample whose pc has no mapping would stay unresolved).
                    const ResourceClass wanted = u.is_storage_image
                        ? ResourceClass::StorageImage : ResourceClass::Texture;
                    Gen5ImageFormatInfo fi;
                    if (!gen5_image_format(d.format, &fi)) {
                        if (wanted == ResourceClass::StorageImage) continue;
                        // Same policy as build_shader_resources: the normal per-target renderer can
                        // bind this as RGBA8 for RTT injection; legacy single-target mode skips it.
                        static const bool rtt_bind = getenv("PROSPER_RTT") != nullptr ||
                                                     getenv("PROSPER_RTT_PERTARGET") != nullptr;
                        if (!rtt_bind) continue;
                        fi.format = DataFormat::Unorm8; fi.num_components = 4; fi.bytes_per_block = 4;
                        fi.block_width = fi.block_height = 1; fi.srgb = false; fi.snorm = false;
                    }
                    const bool is_bcn = fi.block_width > 1;
                    if (is_bcn && fi.snorm) continue;   // signed BCn (SNORM / BC6H SF16): decode not wired
                    const DecodedImageView view = image_base_level_view(d, fi);
                    if (!view.supported) {
                        warn_unsupported_image_view(d);
                        continue;
                    }
                    const uint32_t img_dim = image_type_to_dim(d.type);
                    const bool proven_zero_mip =
                        shader_resource_allows_zero_mip_specialization(u, d, view);
                    {
                        bool mapped = false;
                        for (auto& r0 : t.resources)
                            if (r0.cls == wanted && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == fi.format &&
                                r0.img_dim == img_dim && r0.sample_count == d.sample_count &&
                                r0.depth_compare == u.is_depth_compare &&
                                r0.proven_zero_mip == proven_zero_mip &&
                                r0.in_mip_tail == view.in_mip_tail &&
                                r0.mip_tail_offset == (view.in_mip_tail ? view.mip_offset : 0) &&
                                r0.mip_tail_x == view.mip_tail_x &&
                                r0.mip_tail_y == view.mip_tail_y &&
                                r0.layer_stride_bytes == view.layer_stride &&
                                r0.layer_mip_offset_bytes == view.layer_mip_offset) {
                                if (attach_image_use(
                                        r0, u.use_pc,
                                        wanted == ResourceClass::Texture && u.has_samp
                                            ? u.s4.data() : nullptr)) {
                                    mapped = true;
                                    break;
                                }
                            }
                        if (mapped) continue;
                    }
                    ShaderResource r;
                    r.cls = wanted;
                    r.format = fi.format; r.num_components = fi.num_components;
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.sample_count = d.sample_count;
                    r.declared_mip_levels = d.sample_count > 1u ? 1u :
                        (d.last_level >= d.base_level ?
                            (uint32_t)(d.last_level - d.base_level) + 1u : 1u);
                    shader_resource_apply_mip_chain_provenance(r, view);
                    r.tile_mode = d.tile_mode; r.srgb = fi.srgb;
                    r.in_mip_tail = view.in_mip_tail;
                    r.mip_tail_offset = view.in_mip_tail
                        ? static_cast<uint32_t>(view.mip_offset) : 0;
                    r.mip_tail_bytes = view.mip_tail_bytes;
                    r.mip_tail_x = view.mip_tail_x;
                    r.mip_tail_y = view.mip_tail_y;
                    r.layer_stride_bytes = static_cast<uint32_t>(view.layer_stride);
                    r.layer_mip_offset_bytes = static_cast<uint32_t>(view.layer_mip_offset);
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0 || view.in_mip_tail;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    // T# TYPE -> MIMG dim (GFX10: 9=2D, 10=3D, 11=CUBE, 13=2D_ARRAY); a cube
                    // uploads as six vertically-stacked faces (#273 — see agc_shader_layout).
                    r.img_dim = img_dim;
                    r.depth_compare = u.is_depth_compare;
                    r.proven_zero_mip = proven_zero_mip;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    const uint64_t backing_bytes_per_sample = is_bcn
                        ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                        : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                    if (!d.sample_count ||
                        backing_bytes_per_sample > UINT32_MAX / d.sample_count) continue;
                    const uint64_t backing_bytes = backing_bytes_per_sample * d.sample_count;
                    if (!backing_bytes || backing_bytes > UINT32_MAX) continue;
                    r.size = static_cast<uint32_t>(backing_bytes);
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;   // ambiguous/absent key: pc-only provenance
                    r.fetch_pc   = u.use_pc;                       // per-instruction provenance (#273)
                    if (wanted == ResourceClass::Texture && u.has_samp) {
                        // Paired S# (same SQ_IMG_SAMP decode as the sharp path). Storage operations do
                        // not consume a sampler even when their SSAMP bits alias known user SGPRs.
                        apply_sampler_descriptor(r, u.s4.data());
                    }
                    if (log) fprintf(stderr, "[srt] %s %s key=0x%x %ux%u fmt=%u base=0x%llx tile=%u samp=%d\n",
                                     is_ps ? "PS" : "VS",
                                     wanted == ResourceClass::StorageImage ? "storage-image" : "tex",
                                     u.key, d.width, d.height, d.format,
                                     (unsigned long long)d.base, d.tile_mode,
                                     (int)(wanted == ResourceClass::Texture && u.has_samp));
                    t.resources.push_back(r);
                }
            }
        }
        // Metadata-described DIRECT buffers came from this exact candidate user-data range.
        // Dynamic-fold resources already carry either their proven primary seed or the explicit
        // ambiguous sentinel and are therefore never overwritten here.
        for (auto& resource : t.resources) {
            if (resource.direct_vsharp_sh_register_base !=
                    ShaderResource::kDirectVSharpOriginUnavailable ||
                resource.srt_offset != 0xFFFFFFFFu ||
                resource.sgpr_base == 0xFFFFFFFFu ||
                (resource.cls != ResourceClass::ConstantBuffer &&
                 resource.cls != ResourceClass::VertexBuffer))
                continue;
            if (resource.sgpr_base < user_sgpr_base ||
                resource.sgpr_base - user_sgpr_base > kUserSgprs - 4u) {
                resource.direct_vsharp_sh_register_base =
                    ShaderResource::kDirectVSharpOriginAmbiguous;
                continue;
            }
            resource.direct_vsharp_sh_register_base = base + range_start +
                (resource.sgpr_base - user_sgpr_base);
        }
        if (t.resources.empty()) continue;
        t.vertices_per_instance = is_ps ? 0u : draw_vertex_count;
        assign_convention_bindings(t, is_ps ? kPsBindingBase : 2u);
        if (log) {
            fprintf(stderr, "[restab] %s code=0x%llx base=0x%x -> %zu resources:\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base, t.resources.size());
            for (auto& r : t.resources) {
                fprintf(stderr, "[restab]   cls=%u binding=%u addr=0x%llx size=%u %ux%u fmt=%u stride=%u\n",
                        (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                        r.width, r.height, (unsigned)r.format, r.stride);
                if (r.cls == ResourceClass::ConstantBuffer && guest_readable(r.gpu_addr, 32)) {
                    const float* f = (const float*)(uintptr_t)r.gpu_addr;
                    fprintf(stderr, "[restab]     cbuf@0 floats: %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
                    if (r.size >= 64 && guest_readable(r.gpu_addr, 64))
                        fprintf(stderr, "[restab]     cbuf@0 8..15:   %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                                f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                    if (r.size >= 0x150 && guest_readable(r.gpu_addr + 0x110, 64)) {
                        const float* g = (const float*)(uintptr_t)(r.gpu_addr + 0x110);
                        // Full 4x4 projection matrix at 0x110 (16 floats). Row 3's w element (g[15]) must be
                        // ~1.0 for clip.w to be nonzero — a zero here collapses the perspective divide.
                        fprintf(stderr, "[restab]     mtx@0x110 r0: %.4f %.4f %.4f %.4f\n", g[0], g[1], g[2], g[3]);
                        fprintf(stderr, "[restab]     mtx@0x110 r1: %.4f %.4f %.4f %.4f\n", g[4], g[5], g[6], g[7]);
                        fprintf(stderr, "[restab]     mtx@0x110 r2: %.4f %.4f %.4f %.4f\n", g[8], g[9], g[10], g[11]);
                        fprintf(stderr, "[restab]     mtx@0x110 r3: %.4f %.4f %.4f %.4f  <- g[15]=clip.w src\n", g[12], g[13], g[14], g[15]);
                    }
                }
            }
        }
        auto result = std::make_shared<ShaderResourceTable>(std::move(t));
        record_phases();
        return result;
    }
    if (log) fprintf(stderr, "[restab] %s code=0x%llx -> no resources in any user-data base\n",
                     is_ps ? "PS" : "VS", (unsigned long long)code_addr);
    record_phases();
    return nullptr;
}

bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage) {
    // The read stays LIVE here (not PROSPER_ENV_VALUE): two tests arm this variable at runtime and
    // a process-lifetime cache would make their arms vacuous rather than failing -- #2214. Callers
    // on a per-draw path pass the mode in via the overload below instead of paying this per call.
    return validate_runtime_descriptor_contract(stage_name, spirv, runtime, expected_set,
                                                expected_stage,
                                                getenv("PROSPER_DESCRIPTOR_VALIDATE"));
}

bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage,
                                           const char* mode) {
    if (!mode || !*mode || !strcmp(mode, "off") || !strcmp(mode, "0")) return true;

    DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, runtime, expected_set, expected_stage, true);
    const bool verbose = !strcmp(mode, "all");
    if (!report.issues.empty() || verbose) {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t word : spirv) { hash ^= word; hash *= 1099511628211ull; }
        static std::set<uint64_t> logged;
        uint64_t key = hash ^ (static_cast<uint64_t>(expected_set) << 56);
        auto mix = [&](uint64_t value) { key ^= value; key *= 1099511628211ull; };
        for (const auto& issue : report.issues) {
            mix(static_cast<uint32_t>(issue.code)); mix(issue.binding); mix(issue.set);
            mix(static_cast<uint32_t>(issue.actual));
            // Contract errors with different proven ranges are distinct. Warning-only unused
            // resources are not: their guest address/size can change every draw and must not flood
            // a long diagnostic run with the same module/binding warning.
            // Counts are mixed alongside the byte ranges so the property survives codes that report
            // counts instead of ranges: two arity mismatches at one binding differing only in how many
            // descriptors each side declared stay distinct rather than deduping into one line.
            if (issue.error) {
                mix(issue.required_bytes); mix(issue.available_bytes);
                mix(issue.shader_count); mix(issue.runtime_count);
            }
        }
        if (logged.insert(key).second) {
            fprintf(stderr, "[descriptor] %s module=%016llx used=%zu runtime=%zu result=%s mode=%s\n",
                    stage_name, (unsigned long long)hash, report.descriptors.size(),
                    runtime ? runtime->resources.size() : 0, report.ok() ? "accept" : "reject", mode);
            for (const auto& d : report.descriptors)
                fprintf(stderr, "[descriptor]   set=%u binding=%u type=%s required=%llu%s\n",
                        d.set, d.binding, spirv_descriptor_kind_name(d.kind),
                        (unsigned long long)d.required_bytes, d.dynamic_access ? "+dynamic" : "");
            for (const auto& issue : report.issues)
                fprintf(stderr, "[descriptor]   %s: %s set=%u binding=%u expected=%s actual=%s "
                                "required=%llu available=%llu\n",
                        issue.error ? "ERROR" : "warn", descriptor_issue_name(issue.code),
                        issue.set, issue.binding, spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual),
                        (unsigned long long)issue.required_bytes,
                        (unsigned long long)issue.available_bytes);
            for (const auto& issue : report.issues)
                if (issue.code == DescriptorIssueCode::ArrayBindingArityMismatch)
                    fprintf(stderr, "[descriptor]     arity set=%u binding=%u shader_declares=%u "
                                    "runtime_supplies=%u\n",
                            issue.set, issue.binding, issue.shader_count, issue.runtime_count);
            if (runtime) for (const auto& r : runtime->resources)
                fprintf(stderr, "[descriptor]   runtime binding=%u cls=%u addr=0x%llx size=%u "
                                "stride=%u fmt=%u comps=%u srt=0x%x sgpr=%u pc=%u\n",
                        r.binding, (unsigned)r.cls, (unsigned long long)r.gpu_addr, r.size,
                        r.stride, (unsigned)r.format, r.num_components,
                        r.srt_offset, r.sgpr_base, r.fetch_pc);
        }
    }
    return strcmp(mode, "strict") != 0 || report.ok();
}

ComputeLaunchDimensions resolve_compute_launch(const GpuState::Dispatch& d) {
    namespace P = prosper::agc::Pm4;
    ComputeLaunchDimensions out;
    const GpuState* ds = d.state.get();
    auto reg = [&](uint32_t off) {
        if (!ds) {
            return 0u;
        }
        auto it = ds->sh.find(off);
        return it == ds->sh.end() ? 0u : it->second;
    };
    // The workgroup local size is the NUM_THREAD_FULL field [15:0]; masking prevents a nonzero
    // NUM_THREAD_PARTIAL ([31:16]) from being folded into the dimension (#911).
    auto num_thread = [&](uint32_t off) {
        return (reg(off) >> P::COMPUTE_NUM_THREAD_FULL_SHIFT) & P::COMPUTE_NUM_THREAD_FULL_MASK;
    };
    out.local_x = num_thread(P::COMPUTE_NUM_THREAD_X);
    out.local_y = num_thread(P::COMPUTE_NUM_THREAD_Y);
    out.local_z = num_thread(P::COMPUTE_NUM_THREAD_Z);
    if (!out.local_x) out.local_x = 1;
    if (!out.local_y) out.local_y = 1;
    if (!out.local_z) out.local_z = 1;
    auto groups = [](uint32_t threads, uint32_t local) {
        return threads ? 1u + (threads - 1u) / local : 0u;
    };
    const bool use_thread_dimensions = ((d.modifier >>
        P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
        P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0;
    if (use_thread_dimensions) {
        out.threads_x = d.threads_x;
        out.threads_y = d.threads_y;
        out.threads_z = d.threads_z;
        out.groups_x = groups(out.threads_x, out.local_x);
        out.groups_y = groups(out.threads_y, out.local_y);
        out.groups_z = groups(out.threads_z, out.local_z);
    } else {
        out.groups_x = d.threads_x;
        out.groups_y = d.threads_y;
        out.groups_z = d.threads_z;
        auto total_threads = [](uint32_t group_count, uint32_t local) {
            return static_cast<uint32_t>(std::min<uint64_t>(
                static_cast<uint64_t>(group_count) * local, UINT32_MAX));
        };
        out.threads_x = total_threads(out.groups_x, out.local_x);
        out.threads_y = total_threads(out.groups_y, out.local_y);
        out.threads_z = total_threads(out.groups_z, out.local_z);
    }
    // PROSPER_MAX_DISPATCH_GROUPS=N — DIAGNOSTIC ONLY, default off. Caps the workgroup count of every
    // dispatch. This deliberately computes WRONG results (a clamped kernel processes part of its
    // domain), so it is never a correctness or progression mode; it exists to test whether an
    // enormous dispatch is what loses the Vulkan device (#1742 -> #1743). Without it, Astro Bot
    // reaches 752,646 workgroups in one dispatch and the device is lost permanently a few submits
    // later; with it, the same route can be run to the same point with the size bounded.
    static const uint32_t group_cap = [] {
        // 0 disables the cap. That is the right DEFAULT and the wrong answer to a typo: this knob
        // is only ever set to bound a dispatch that loses the device (#1742/#1743), so `=750000`
        // mistyped as `=750,000` unbounded the very thing being bounded (#3267).
        const char* value = std::getenv("PROSPER_MAX_DISPATCH_GROUPS");
        if (!value || !*value) return 0u;
        return static_cast<uint32_t>(prosper::diag::env_u64_or_default_auto_capped(
            "PROSPER_MAX_DISPATCH_GROUPS", value, 0ull, UINT32_MAX, "workgroups",
            "no cap: nothing is bounded"));
    }();
    if (group_cap) {
        auto clamp_groups = [&](uint32_t& groups, uint32_t& threads, uint32_t local) {
            if (groups <= group_cap) return;
            static std::atomic<int> reported{0};
            if (reported.fetch_add(1) < 16)
                std::fprintf(stderr, "[dispatch-cap] clamping %u workgroups to %u\n",
                             groups, group_cap);
            groups = group_cap;
            threads = static_cast<uint32_t>(std::min<uint64_t>(
                static_cast<uint64_t>(groups) * local, UINT32_MAX));
        };
        clamp_groups(out.groups_x, out.threads_x, out.local_x);
        clamp_groups(out.groups_y, out.threads_y, out.local_y);
        clamp_groups(out.groups_z, out.threads_z, out.local_z);
    }
    return out;
}

// A direct dispatch with zero workgroups on any axis launches no waves and therefore cannot read
// its program, descriptors, or backing memory. Cull it before realization rather than reporting an
// unreachable shader/resource failure. Indirect dimensions are not available at this boundary and
// remain on the ordered argument-resolution path, which applies the same hardware no-op rule after
// reading their three argument dwords.
static bool direct_compute_dispatch_is_noop(const GpuState::Dispatch& dispatch) {
    if (dispatch.indirect) return false;
    const ComputeLaunchDimensions launch = resolve_compute_launch(dispatch);
    return !launch.groups_x || !launch.groups_y || !launch.groups_z;
}

ComputeCpuFastPath classify_compute_cpu_fast_path(const uint32_t* code, size_t dwords) {
    // Compiler-generated 16-byte buffer fill:
    //   record = (tgid.x << 6) + tid.x; buffer[record] = {s4, s5, s6, s7}
    // This exact stream is used by Dead Cells and very frequently by Astro Bot. Keep the match
    // byte-exact so a different address calculation, descriptor, component mask, or predicate can
    // never be mistaken for a fill.
    static constexpr uint32_t kFillSgprUvec4[] = {
        0xd7460004u, 0x04010c08u, 0x7e000204u, 0x7e020205u, 0x7e040206u,
        0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u,
    };
    if (code && dwords >= std::size(kFillSgprUvec4) &&
        std::equal(std::begin(kFillSgprUvec4), std::end(kFillSgprUvec4), code))
        return ComputeCpuFastPath::FillSgprUvec4;
    return ComputeCpuFastPath::None;
}

uint64_t compute_dispatch_code_addr(const GpuState& submit, const GpuState::Dispatch& dispatch) {
    namespace P = prosper::agc::Pm4;
    const GpuState& state = dispatch.state ? *dispatch.state : submit;
    auto reg = [&](uint32_t offset) {
        const auto it = state.sh.find(offset);
        return it == state.sh.end() ? 0u : it->second;
    };
    return (static_cast<uint64_t>(reg(P::COMPUTE_PGM_LO)) << 8) |
           (static_cast<uint64_t>(reg(P::COMPUTE_PGM_HI) & 0xffu) << 40);
}

std::vector<ComputeItem> realize_compute_dispatches(
    const GpuState& st, uint64_t submit_no,
    std::vector<OperationRealizationFailure>* failures) {
    if (failures) failures->clear();
    if (st.dispatches.empty()) return {};
    namespace P = prosper::agc::Pm4;
    auto rd = [](const RegisterFile& regs, uint32_t off) {
        auto it = regs.find(off);
        return it == regs.end() ? 0u : it->second;
    };

    std::vector<ComputeItem> items;
    items.reserve(st.dispatches.size());
    for (size_t dispatch_index = 0; dispatch_index < st.dispatches.size(); dispatch_index++) {
        // Set where the program's GDS usage is known (see uses_gds below); read where the item is
        // finalized. Declared at loop scope because those two points are in different blocks.
        // Set where the program's GDS usage is known (see uses_gds below); read where the item is
        // finalized, alongside the post-translation emission result. Declared at loop scope because
        // those two points are in different blocks.
        bool program_uses_guest_gds_for_item = false;
        const auto& dispatch = st.dispatches[dispatch_index];
        if (direct_compute_dispatch_is_noop(dispatch)) continue;
        const GpuState& ds = dispatch.state ? *dispatch.state : st;
        const uint64_t code_addr = compute_dispatch_code_addr(st, dispatch);
        OperationRealizationFailure failure;
        failure.kind = SubmitOperationKind::Dispatch;
        failure.index = dispatch_index;
        failure.command_order = dispatch.command_order;
        failure.compute_launch = resolve_compute_launch(dispatch);
        auto record_failure = [&](RealizationFailureReason reason,
                                  const std::shared_ptr<ShaderResourceTable>& resources,
                                  const std::vector<uint32_t>& spirv,
                                  const ComputeShaderConfig* recompile_config = nullptr) {
            if (!failures) return;
            failure.reason = reason;
            ShaderRealizationDiagnostic stage;
            stage.stage = ShaderProgramStage::Compute;
            stage.program_addr = code_addr;
            stage.resources = resources;
            stage.recompiled = !spirv.empty();
            if (recompile_config) {
                stage.recompile_config = *recompile_config;
                stage.recompile_config_available = true;
            }
            if (!spirv.empty()) {
                const DescriptorValidationReport diagnostic_report =
                    validate_spirv_descriptor_interface(spirv, resources.get(), 0,
                                                        SpirvShaderStage::Compute, true);
                stage.descriptor_issue_count =
                    static_cast<uint32_t>(diagnostic_report.issues.size());
                auto issue = std::find_if(diagnostic_report.issues.begin(),
                                          diagnostic_report.issues.end(),
                                          [](const auto& candidate) { return candidate.error; });
                if (issue == diagnostic_report.issues.end() && !diagnostic_report.issues.empty())
                    issue = diagnostic_report.issues.begin();
                if (issue != diagnostic_report.issues.end())
                    stage.first_descriptor_issue = static_cast<uint32_t>(issue->code);
            }
            failure.stages.push_back(std::move(stage));
            failures->push_back(std::move(failure));
        };
        const auto* header = static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(code_addr));
        if (!header || !code_addr || !guest_readable(code_addr, sizeof(uint32_t))) {
            record_failure(RealizationFailureReason::MissingProgram, {}, {});
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                std::fprintf(stderr, "[compute] skip unregistered/unreadable program 0x%llx\n",
                             (unsigned long long)code_addr);
            continue;
        }
        const size_t shader_dwords = registered_shader_dwords(*header, code_addr);

        uint32_t range_start = 0;
        if (header->specials && guest_readable((uint64_t)(uintptr_t)header->specials,
                                               sizeof(AgcShaderSpecials))) {
            const uint32_t start = header->specials->user_data_range_start;
            const uint32_t end = header->specials->user_data_range_end;
            if (start < kUserSgprs && end > start && end <= 2 * kUserSgprs) range_start = start;
        }
        uint32_t sgprs[kUserSgprs] = {};
        read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
        const uint32_t rsrc2 = rd(ds.sh, P::COMPUTE_PGM_RSRC2);
        auto field = [&](uint32_t shift, uint32_t mask) { return (rsrc2 >> shift) & mask; };
        const uint32_t user_count = field(P::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT,
                                          P::COMPUTE_PGM_RSRC2_USER_SGPR_MASK);
        const uint32_t compute_wave_size = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_MASK) ? 32u : 64u;
        const ComputeLaunchDimensions launch = resolve_compute_launch(dispatch);
        const bool tgid_x_en = field(P::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0;
        const bool tgid_y_en = field(P::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0;
        const bool tgid_z_en = field(P::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0;
        // System SGPRs follow the user SGPR block. Only provide dispatch geometry to the bounded
        // linear-store recognizer when TGID.x is the only enabled group id and fits in the
        // push-constant SGPR block. Extra Y/Z groups merely repeat the exact same zero stores because
        // the proven kernel cannot observe them; all other oversized/formatless descriptors stay rejected.
        const bool linear_store_proof_context =
            user_count < kUserSgprs && tgid_x_en && !tgid_y_en && !tgid_z_en;
        auto table = std::make_shared<ShaderResourceTable>(
            build_shader_resources(*header, sgprs, kUserSgprs, 0));
        // Descriptor-TABLE uses (#590, mirroring the graphics fold in build_stage_table): UE4 compute
        // kernels s_load their V#/T# descriptors from tables pointed to by the user-data SGPRs and
        // consume them via s_buffer_load / image ops. build_shader_resources only sees the DIRECT
        // sharp/user-data layout, so those table descriptors were thrown away — the const-fold walk
        // recovers them here with the same keyed (srt_offset) / per-use (fetch_pc) provenance the
        // recompiler resolves. Compute-specific delta: an image_store use becomes a STORAGE image
        // (the recompiler's storage path requires ResourceClass::StorageImage), a sampled use a
        // Texture. Never shadow an existing resource at the same srt_offset (first-match-wins).
        // Seed the fold with the registers the dispatch ACTUALLY initializes, not all 32 (#590).
        // COMPUTE_PGM_RSRC2.USER_SGPR declares how many user SGPRs the hardware loads; s[user_count..31]
        // are undefined. Passing kUserSgprs made the fold mark every one of those as a KNOWN value of
        // zero, because `sgprs` is a zero-filled kUserSgprs array. "Known zero" and "never established"
        // are not the same thing to a const-fold: a descriptor chain rooted in an undefined register
        // then folds to base=0, the address becomes a bare immediate offset (0x38, 0xb0, 0x128 ...),
        // `readable()` fails, and the use is dropped — or worse, four zero dwords are published as a V#
        // that shadows a real descriptor under first-match-wins.
        //
        // Measured on PPSA04263 (GTA V) at gameplay: ALL 52 failing compute programs declare fewer than
        // 32 (range 2..15), so 17-30 phantom registers were marked known in every one of them; the same
        // run reports 1,988 SMEM loads with `base=0x0 base_ok=1` and 475 unreadable bare-offset
        // addresses.
        //
        // This function already encodes the same bound ~180 lines below, where the push-constant path
        // computes `fw_user_count = min(user_count, kUserSgprs)` and skips a fold result whose
        // `base_sgpr` lands in [user_count, 31] — because indexing past the block would be wrong. The
        // fold call had not been given that bound, so the two disagreed about which registers exist.
        const uint32_t fold_user_count = std::min<uint32_t>(user_count, kUserSgprs);
        {
            const ComputeResourceDispatchContext resource_dispatch_context{
                launch.local_x, launch.local_y, launch.local_z,
                launch.threads_x, launch.threads_y, launch.threads_z,
                ((dispatch.modifier >>
                    P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
                    P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0,
                compute_wave_size,
                tgid_x_en, tgid_y_en, tgid_z_en,
                field(P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT,
                      P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK),
            };
            const std::vector<SrtUse> srt_uses = add_compute_buffer_resources(
                *table, (const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                sgprs, fold_user_count,
                launch.local_x, launch.threads_x,
                linear_store_proof_context ? user_count : UINT32_MAX,
                &resource_dispatch_context);
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                if (u.kind != 0) continue;                 // buffers were materialized by the shared helper
                // Texture/storage uses are instruction-specific even when their table keys repeat.
                const uint64_t dk = 0x8000000000000000ull | u.use_pc;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = u.key == 0xFFFFFFFFu;
                if (!clash)
                    for (const auto& r0 : table->resources)
                        if (r0.srt_offset == u.key) { clash = true; break; }
                {                                         // T# — storage image (store) or sampled texture
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    const char* reject = image_descriptor_reject_reason(d);
                    if (g_dyntrace_force)
                        fprintf(stderr, "[dynfail] compute tex use pc=%u key=0x%x base=0x%llx %ux%u "
                                        "type=%u base_array=%u fmt=%u tile=%u -> %s\n",
                                u.use_pc, u.key, (unsigned long long)d.base, d.width, d.height,
                                d.type, d.base_array, d.format, d.tile_mode,
                                reject ? reject : "materialize");
                    if (reject) continue;                            // garbage/degenerate T#
                    Gen5ImageFormatInfo fi;
                    const bool mapped_fmt = gen5_image_format(d.format, &fi);
                    // Unknown sampled formats cannot be decoded. Unknown storage formats may still
                    // recompile (format-free SPIR-V), but remain explicitly Unknown so the live backend
                    // rejects them instead of silently treating arbitrary bytes as RGBA8.
                    if (!mapped_fmt && !u.is_storage_image) continue;
                    if (mapped_fmt && fi.block_width > 1 && fi.snorm) continue;   // signed BCn: not wired
                    // The unmapped-format fallback builds a view by hand and therefore never applies
                    // a slice offset. It must fail closed on a non-zero BASE_ARRAY for the same
                    // reason image_base_level_view's early returns do: an unshifted base under a
                    // selected slice binds slice ZERO's texels silently. (Storage images are the only
                    // consumer here — a sampled use with an unmapped format was already dropped.)
                    const DecodedImageView view = mapped_fmt
                        ? image_base_level_view(d, fi)
                        : unmapped_format_image_view(d);
                    if (!view.supported) {
                        warn_unsupported_image_view(d);
                        continue;
                    }
                    const ResourceClass wanted = u.is_storage_image ? ResourceClass::StorageImage
                                                                   : ResourceClass::Texture;
                    const DataFormat view_format = mapped_fmt ? fi.format : DataFormat::Unknown;
                    const uint32_t img_dim = image_type_to_dim(d.type);
                    // This is deliberately narrower than "the backend currently uploads one mip".
                    // The instruction proof belongs to this exact use, and the descriptor must also
                    // declare one base level. Compression is deliberately not part of this marker:
                    // it proves only that discarding the mip operand is exact. The live backend must
                    // independently acquire authoritative pixels from a renderer image, an
                    // all-uncompressed metadata plane, or a supported DCC decode before execution.
                    const bool proven_zero_mip =
                        shader_resource_allows_zero_mip_specialization(u, d, view);
                    {
                        bool mapped = false;
                        for (auto& r0 : table->resources)
                            if (r0.cls == wanted && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == view_format &&
                                r0.img_dim == img_dim && r0.sample_count == d.sample_count &&
                                r0.depth_compare == u.is_depth_compare &&
                                r0.proven_zero_mip == proven_zero_mip &&
                                r0.in_mip_tail == view.in_mip_tail &&
                                r0.mip_tail_offset == (view.in_mip_tail ? view.mip_offset : 0) &&
                                r0.mip_tail_x == view.mip_tail_x &&
                                r0.mip_tail_y == view.mip_tail_y &&
                                r0.layer_stride_bytes == view.layer_stride &&
                                r0.layer_mip_offset_bytes == view.layer_mip_offset) {
                                if (attach_image_use(
                                        r0, u.use_pc,
                                        wanted == ResourceClass::Texture && u.has_samp
                                            ? u.s4.data() : nullptr)) {
                                    mapped = true;
                                    break;
                                }
                            }
                        if (mapped) continue;
                    }
                    ShaderResource r;
                    r.cls = wanted;
                    if (mapped_fmt) {
                        r.format = fi.format; r.num_components = fi.num_components;
                        const bool is_bcn = fi.block_width > 1;
                        const uint64_t bytes_per_sample = is_bcn
                            ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                            : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                        if (!d.sample_count || bytes_per_sample > UINT32_MAX / d.sample_count)
                            continue;
                        const uint64_t bytes = bytes_per_sample * d.sample_count;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.size = static_cast<uint32_t>(bytes);
                        r.srgb = fi.srgb;
                    } else {
                        const uint64_t bytes_per_sample =
                            static_cast<uint64_t>(d.width) * d.height * d.depth * 4;
                        if (!d.sample_count || bytes_per_sample > UINT32_MAX / d.sample_count)
                            continue;
                        const uint64_t bytes = bytes_per_sample * d.sample_count;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.format = DataFormat::Unknown; r.num_components = 4;
                        r.size = static_cast<uint32_t>(bytes);
                    }
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.sample_count = d.sample_count;
                    shader_resource_apply_mip_chain_provenance(r, view);
                    r.tile_mode = d.tile_mode;
                    r.declared_mip_levels = d.sample_count > 1u ? 1u :
                        (d.last_level >= d.base_level ?
                            (uint32_t)(d.last_level - d.base_level) + 1u : 1u);
                    r.in_mip_tail = view.in_mip_tail;
                    r.mip_tail_offset = view.in_mip_tail
                        ? static_cast<uint32_t>(view.mip_offset) : 0;
                    r.mip_tail_bytes = view.mip_tail_bytes;
                    r.mip_tail_x = view.mip_tail_x;
                    r.mip_tail_y = view.mip_tail_y;
                    r.layer_stride_bytes = static_cast<uint32_t>(view.layer_stride);
                    r.layer_mip_offset_bytes = static_cast<uint32_t>(view.layer_mip_offset);
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0 || view.in_mip_tail;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    r.img_dim = img_dim;
                    r.depth_compare = u.is_depth_compare;
                    r.proven_zero_mip = proven_zero_mip;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    r.fetch_pc = u.use_pc;               // per-use pc provenance (the image op)
                    if (u.has_samp) {
                        apply_sampler_descriptor(r, u.s4.data());
                    }
                    table->resources.push_back(r);
                }
            }
        }
        // A direct compute "constant buffer" can actually be an SRT address pair consumed by
        // s_load_dword[xN]. Its adjacent SGPRs are not V# NUM_RECORDS/format words, so decoding them
        // as a four-dword V# can produce a nonsense one-byte bound. Size these pointer-backed tables
        // from the shader's immediate s_load accesses, but only when that exact guest range is mapped.
        for (auto& resource : table->resources) {
            if (resource.cls != ResourceClass::ConstantBuffer ||
                resource.sgpr_base == 0xFFFFFFFFu || !resource.gpu_addr)
                continue;
            const uint32_t required = rdna2_sload_required_bytes(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, resource.sgpr_base);
            if (required > resource.size && required <= 0x10000000u &&
                guest_readable(resource.gpu_addr, required)) {
                if (std::getenv("PROSPER_DBG"))
                    std::fprintf(stderr,
                                 "[compute-sload-range] s%u addr=0x%llx decoded=%u inferred=%u\n",
                                 resource.sgpr_base, (unsigned long long)resource.gpu_addr,
                                 resource.size, required);
                resource.size = required;
            }
        }
        // FLAT-window resources (#1171): a general flat_load reads a byte/dword from a raw 64-bit guest
        // pointer held in user SGPRs (not a V# descriptor, so build/add above never saw it). Resolve each
        // such load to its base SGPR pair, read the base pointer from THIS dispatch's user SGPRs, bind the
        // containing guest allocation as an SSBO, and key it by the load's pc so the recompiler lowers the
        // load to an indexed read at (address - base). Pushed before assign_convention_bindings so each
        // window gets a distinct binding; an unmapped base leaves the load unresolved (fail-visible).
        {
            // Bound the analysis by the ACTUAL user-SGPR count, not kUserSgprs (32). The emit indexes the
            // push-constant block, which is sized to config.user_sgprs.size() = min(user_count, kUserSgprs);
            // a base resolved in [user_count, 31] would make the emit's load_push_constant index past that
            // block (spirv-val would fail-visible, but keep the analysis and push-constant widths consistent).
            const uint32_t fw_user_count = std::min<uint32_t>(
                (rd(ds.sh, P::COMPUTE_PGM_RSRC2) >> P::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT) &
                    P::COMPUTE_PGM_RSRC2_USER_SGPR_MASK,
                kUserSgprs);
            const FlatLoadAnalysis fla = analyze_flat_loads(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, fw_user_count);
            // 256 MiB cap on the per-dispatch copy. TODO(#1183): the decode loop reads only base..base+N;
            // size the window from that access footprint (or the destination image extent) once the loop
            // executes, instead of the whole containing-allocation remainder.
            constexpr uint32_t kFlatWindowMax = 0x10000000u;
            for (const auto& fl : fla.loads) {
                if (fl.base_sgpr < 0 || static_cast<uint32_t>(fl.base_sgpr) + 1u >= fw_user_count) continue;
                const uint64_t base = static_cast<uint64_t>(sgprs[fl.base_sgpr]) |
                                      (static_cast<uint64_t>(sgprs[fl.base_sgpr + 1]) << 32);
                host::GuestReadableRange range{};
                if (!host::guest_readable_mapping_containing(base, base + 1, range) || range.end <= base)
                    continue;                                  // unmapped base -> stay fail-visible
                const uint64_t avail = range.end - base;
                ShaderResource w;
                w.cls = ResourceClass::ConstantBuffer;         // read-only SSBO (declare_cbufs + cbuf_load)
                w.gpu_addr = base;
                w.size = static_cast<uint32_t>(std::min<uint64_t>(avail, kFlatWindowMax));
                w.fetch_pc = fl.load_pc;
                w.flat_base_sgpr = static_cast<uint32_t>(fl.base_sgpr);
                table->resources.push_back(w);
                if (std::getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[flat-window] pc=%u base=s%d addr=0x%llx size=%u\n",
                                 fl.load_pc, fl.base_sgpr, (unsigned long long)base, w.size);
            }
        }
        bool native_multiwave_wave_work = false;
        const RecompileDiagnosticContext recompile_diagnostic{
            RecompileDiagnosticStage::Compute, code_addr};
        {
            std::vector<Rdna2Inst> decoded;
            rdna2_walk(reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                       shader_dwords, decoded);
            // Keep dispatch-scoped resource discovery and translation on the same specialized
            // instruction stream. A proven-null BVH can collapse only the exact no-hit exit and a
            // fully matched empty-stack traversal cycle; shader-byte constant folding may then
            // remove any remaining unreachable arm.
            // Drop only instruction-scoped resources whose consumers disappeared. Direct resources
            // (fetch_pc == UINT32_MAX) remain available to every surviving use of their SGPR/SRT key.
            std::vector<Rdna2Inst> resource_paths = decoded;
            const ComputeResourcePathSpecializationReport path_report =
                specialize_compute_resource_paths(resource_paths, *table, compute_wave_size);
            if (std::getenv("PROSPER_DBG") &&
                (path_report.proven_null_exits || path_report.zero_record_execz_exits)) {
                std::fprintf(stderr,
                             "[compute-resource-specialization] code=0x%llx null-exits=%zu zero-record-execz=%zu constants=%zu "
                             "removed-resources=%zu removed-pcs=",
                             static_cast<unsigned long long>(code_addr),
                             path_report.proven_null_exits,
                             path_report.zero_record_execz_exits,
                             path_report.shader_constant_branches,
                             path_report.removed_resources);
                for (size_t index = 0; index < path_report.removed_pcs.size(); ++index)
                    std::fprintf(stderr, "%s%u", index ? "," : "",
                                 path_report.removed_pcs[index]);
                std::fprintf(stderr, "\n");
            }
            native_multiwave_wave_work = compute_shader_prefers_native_multiwave(
                decoded, reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, recompile_diagnostic);
            const bool uses_gds = std::any_of(decoded.begin(), decoded.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::DS && in.ds_gds &&
                       (in.opcode == 0x0d || in.opcode == 0x3d || in.opcode == 0x3e);
            });
            // A bounded dispatcher needs the internal GDS buffer as its witness destination even when
            // the guest program never touches GDS — see kComputeTripWitnessDword.
            //
            // A program that uses GDS ITSELF is never instrumented: the witness would overwrite the
            // guest's own data and so change the input to the behaviour under measurement. This is
            // the only place both facts are known, so the decision is made here and carried on the
            // item rather than re-derived by the host.
            // Intent, used only to decide whether the buffer must be BOUND. Whether a witness was
            // actually emitted is not knowable here -- translation has not run yet -- so the item's
            // token is set after it, from spirv_writes_trip_witness() on the compiled module.
            program_uses_guest_gds_for_item = uses_gds;
            if (uses_gds || compute_trip_witness_active(code_addr)) {
                ShaderResource gds;
                gds.cls = ResourceClass::ConstantBuffer;
                gds.format = DataFormat::Uint32;
                gds.num_components = 1;
                gds.binding = kComputeInternalGdsBinding;
                gds.size = static_cast<uint32_t>(g_compute_gds.size());
                gds.stride = 4;
                gds.host_data = g_compute_gds.data();
                gds.host_data_size = g_compute_gds.size();
                table->resources.push_back(gds);
            }
        }

        ComputeShaderConfig config;
        config.compute_pgm_rsrc1 = rd(ds.sh, P::COMPUTE_PGM_RSRC1);
        config.user_sgprs.assign(sgprs, sgprs + std::min(user_count, kUserSgprs));
        config.local_x = launch.local_x;
        config.local_y = launch.local_y;
        config.local_z = launch.local_z;
        config.exact_thread_extent = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0;
        config.threads_x = launch.threads_x;
        config.threads_y = launch.threads_y;
        config.threads_z = launch.threads_z;
        config.wave_size = compute_wave_size;
        const SharedVulkanContext shared_vulkan = shared_vulkan_context();
        const bool shared_compute_adoptable = shared_vulkan.valid() &&
            shared_vulkan.compute_queue_supported &&
            shared_vulkan.storage_image_read_without_format &&
            shared_vulkan.storage_image_write_without_format;
        config.native_storage_format_support = shared_compute_adoptable
            ? shared_vulkan.native_storage_format_support : 0;
        config.storage_buffer_int64_atomics = shared_compute_adoptable &&
            shared_vulkan.storage_buffer_int64_atomics;
        const uint32_t replay_native_storage_format_support =
            config.native_storage_format_support;
        config.packed_r11_storage =
            std::getenv("PROSPER_NO_PACKED_R11_STORAGE") == nullptr;
        // Keep the stored capture module portable by compiling capture-bound dispatches through
        // device-independent storage paths (raw uvec4 or exact packed R32ui), so optional format
        // support never becomes an artifact ABI. Capture v39 also retains the raw shader and
        // semantic launch ABI, allowing --recompile-raw to reconstruct a device-specific module.
        static const bool timeline_capture_requested =
            std::getenv("PROSPER_GPU_TIMELINE_CAPTURE") != nullptr;
        static const bool timeline_capture_after_compute_gated =
            timeline_capture_requested && gpu_timeline_capture_is_after_compute_gated();
        const bool timeline_capture_after_compute_armed =
            timeline_capture_after_compute_gated &&
            gpu_timeline_capture_after_compute_gate_armed();
        const bool timeline_capture_bound = timeline_capture_requires_portable_compute(
            timeline_capture_requested, timeline_capture_after_compute_gated,
            timeline_capture_after_compute_armed);
        const bool capture_bound = std::getenv("PROSPER_GPU_CAPTURE") ||
            timeline_capture_bound ||
            interactive_gpu_capture_armed() || interactive_capture_bundle_active();
        if (capture_bound)
            config.native_storage_format_support = 0;
        // Full exact-size subgroups let the translator assign guest local coordinates in
        // SubgroupId/SubgroupLocalInvocationId order. That avoids assuming any relationship between
        // Vulkan's implementation-defined LocalInvocationIndex order and subgroup lane order while
        // still making each native subgroup exactly one RDNA wave. Capture v37 records this exact
        // module's required-subgroup/full-subgroups pipeline contract for faithful replay.
        // Repeated scratch-emulated wave scans/votes are a structural exception to the conservative
        // multi-wave default: the exact subgroup shell removes their workgroup barriers while
        // preserving one native subgroup per guest wave. Keep the environment switch as an explicit
        // experiment for every other multi-wave shape; this automatic path is shader-address/title
        // independent and remains subject to all device and workgroup bounds below.
        // PROSPER_NO_NATIVE_COMPUTE_MULTIWAVE: refuse the multi-wave native contract while leaving
        // the ordinary one-wave native path alone. The symmetric counterpart of the switch above,
        // and it exists because the broad PROSPER_NO_NATIVE_COMPUTE_SUBGROUP is unusable as an A/B
        // arm: on a routed GTA V run it additionally declines fifteen OTHER programs, so the program
        // under test never dispatches and the arm returns a zero that means "never ran" rather than
        // "ran clean". This switch changes only shaders whose workgroup spans several guest waves,
        // which is exactly the population the multiwave lowering is responsible for.
        const bool native_multiwave_disabled =
            getenv("PROSPER_NO_NATIVE_COMPUTE_MULTIWAVE") != nullptr;
        const bool native_multiwave_requested = !native_multiwave_disabled &&
            (native_multiwave_wave_work || getenv("PROSPER_NATIVE_COMPUTE_MULTIWAVE") != nullptr);
        const bool native_subgroup_disabled =
            getenv("PROSPER_NO_NATIVE_COMPUTE_SUBGROUP") != nullptr;
        config.native_subgroup_size = select_native_compute_subgroup_size(
            shared_vulkan, config, native_multiwave_requested, native_subgroup_disabled);
        // PROSPER_SUBGROUP_LOG (#2429): the resolved native-subgroup contract is a GUARD CONDITION in
        // several recompiler paths and was printable NOWHERE -- so any claim that turned on it could
        // only be inferred from the device's reported subgroup size, which is a different value with a
        // different derivation. Conflating the two is what this line exists to stop.
        //
        // Prints the INPUTS beside the answer, deliberately. `select_native_compute_subgroup_size`
        // returns 0 -- meaning "no native contract, use the portable shell" -- from **three `return 0`
        // sites spanning 22 clauses** (25 if `adoptable`'s four ANDed checks are counted individually
        // rather than as the one `!adoptable` clause). **Exactly two of those clauses concern the
        // device's subgroup range** (`wave_size` below `min_compute_subgroup_size` or above
        // `max_compute_subgroup_size`); the rest are capability bits, workgroup bounds, and the
        // kernel's own local size. Measured on GTA V, the causes were entirely the last of those:
        // `invocations % wave_size != 0` and the one-wave rule, never the adapter. So a bare
        // `native=0` would send a reader hunting the GPU when the answer is the dispatch's local size,
        // which is why local/invocations and the device bounds are all on the line.
        if (getenv("PROSPER_SUBGROUP_LOG")) {
            const uint64_t local_invocations = static_cast<uint64_t>(config.local_x) *
                config.local_y * config.local_z;
            // Once per (program, resolved contract), not once per dispatch. Per-dispatch this line
            // is emitted thousands of times on a routed run, and the volume is not merely untidy:
            // it slows the subject enough to desync a timing-dependent pad script, so the route
            // never reaches the phase you armed the diagnostic to observe. A run of this diagnostic
            // on GTA V produced 1,464 lines and never dispatched the program under test, which made
            // its zero read as a negative result when it was a run that never happened.
            //
            // Keyed on the resolved contract as well as the address, so a program whose contract
            // legitimately changes between dispatches still reports every distinct outcome.
            static std::mutex subgroup_log_mutex;
            static std::set<std::pair<uint64_t, uint64_t>> subgroup_logged;
            const uint64_t contract = (static_cast<uint64_t>(config.native_subgroup_size) << 32) |
                                      (static_cast<uint64_t>(config.wave_size) << 8) |
                                      (native_multiwave_requested ? 2u : 0u) |
                                      (native_subgroup_disabled ? 1u : 0u);
            bool first_report = false;
            {
                std::lock_guard lock(subgroup_log_mutex);
                first_report = subgroup_logged.emplace(code_addr, contract).second;
            }
            if (first_report)
            std::fprintf(stderr,
                         "[subgroup] cs=0x%llx guest-wave=%u native=%u local=%ux%ux%u "
                         "invocations=%llu device-range=%u..%u max-wg-subgroups=%u "
                         "size-control=%d full-subgroups=%d multiwave=%d disabled=%d\n",
                         (unsigned long long)code_addr, config.wave_size,
                         config.native_subgroup_size, config.local_x, config.local_y, config.local_z,
                         (unsigned long long)local_invocations,
                         shared_vulkan.min_compute_subgroup_size,
                         shared_vulkan.max_compute_subgroup_size,
                         shared_vulkan.max_compute_workgroup_subgroups,
                         (int)shared_vulkan.compute_subgroup_size_control,
                         (int)shared_vulkan.compute_full_subgroups,
                         (int)native_multiwave_requested, (int)native_subgroup_disabled);
        }
        config.tgid_x_en = tgid_x_en;
        config.tgid_y_en = tgid_y_en;
        config.tgid_z_en = tgid_z_en;
        config.tg_size_en = field(P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT,
                                  P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0;
        config.tidig_comp_cnt = field(P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT,
                                      P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK);
        config.lds_bytes = field(P::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK) * 512u;

        // GTA V 0x413ce6000 selects one V# from a scalar-buffer table with a wave-uniform value
        // derived from live source records. Generic const-folding intentionally cannot manufacture
        // that dynamic descriptor. Certify this dispatch's complete selector domain and materialize
        // the sole in-bounds record before assigning Vulkan bindings.
        const bool selected_sbuffer_candidate = rdna2_gta5_selected_sbuffer_shader(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords);
        const bool selected_sbuffer_valid = discover_rdna2_gta5_selected_sbuffer(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords, config, *table);
        if (selected_sbuffer_candidate &&
            (selected_sbuffer_valid || config.threads_x == kGtaSelectedSbufferThreads) &&
            std::getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[gta-selected-sbuffer] code=0x%llx valid=%d resources=%zu\n",
                         static_cast<unsigned long long>(code_addr),
                         static_cast<int>(selected_sbuffer_valid), table->resources.size());

        // GTA V 0x413cf9d00 loads one lane-zero pointer per dispatched workgroup through pc26, then
        // dereferences only a bounded 368-byte footprint at 17 exact GLOBAL sites. Snapshot the table and
        // pointees at this command-ordered realization point; preceding dispatches in the submit
        // have already completed, so the owned shadow cannot observe a stale producer epoch.
        const bool packed_pointer_candidate = rdna2_gta5_packed_pointer_shader(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords);
        const bool packed_pointer_valid = discover_rdna2_gta5_packed_pointer(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords, config, *table);
        if (packed_pointer_candidate && std::getenv("PROSPER_DBG")) {
            const ShaderResource* source = table->by_fetch_pc(kGta5PackedPointerSourcePc);
            std::fprintf(stderr,
                         "[gta-packed-pointer] code=0x%llx valid=%d resources=%zu "
                         "threads=%u source-bytes=%llu shadow-bytes=%llu slots=%u\n",
                         static_cast<unsigned long long>(code_addr),
                         static_cast<int>(packed_pointer_valid),
                         table->resources.size(), config.threads_x,
                         static_cast<unsigned long long>(source ? source->size : 0u),
                         static_cast<unsigned long long>(source ? source->host_data_size : 0u),
                         source ? source->indirect_buffer_slot_count : 0u);
        }

        // Generic dispatch-indexed pointer proofs preserve the source records verbatim and append
        // only the selector-/consumer-bounded pointee intervals. Discovery happens at the same
        // command-ordered realization point as the fixed-slot predecessor: preceding GPU producers
        // are complete, and convention bindings have not yet hidden fetch provenance.
        const bool indirect_pointer_valid = discover_rdna2_indirect_pointer_relocations(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords, config, *table);
        // Report the DECLINE as well as the success, and name the program in both. Printing only
        // successes made a declined proof indistinguishable from a program the analyzers were never
        // offered, and the line carried no address, so a run's several hundred records could not be
        // attributed to any shader at all. Both gaps cost real time on #2481: GTA V's
        // `0x413d14100` is byte-for-byte the StaticFootprint analyzer's own 386-dword target and
        // still rejects at its GLOBAL consumer, and nothing in the log said whether the analyzer
        // had declined it or matched and failed during discovery.
        if (std::getenv("PROSPER_DBG")) {
            const auto source_it = std::find_if(
                table->resources.begin(), table->resources.end(),
                is_indirect_pointer_relocation_resource);
            const ShaderResource* source = source_it == table->resources.end()
                ? nullptr : &*source_it;
            const auto marker = source
                ? source->indirect_pointer_relocation
                : IndirectPointerRelocationBinding{};
            std::fprintf(stderr,
                         "[indirect-pointer-relocation] program=0x%llx valid=%d records=%u "
                         "segments=%u source-bytes=%llu binding-bytes=%u\n",
                         static_cast<unsigned long long>(code_addr),
                         static_cast<int>(indirect_pointer_valid),
                         marker.record_count, marker.segment_count,
                         static_cast<unsigned long long>(source ? source->size : 0u),
                         marker.binding_bytes);
        }

        // GTA V 0x413cf9200 overlays descriptor-shaped root fields with application records for
        // inactive source/output variants. Inspect the current root only here, after preceding GPU
        // producers have completed, and retain that same 224-byte root as every marker's witness.
        const bool cf9200_candidate = rdna2_gta5_cf9200_shader(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords);
        const bool cf9200_valid = discover_rdna2_gta5_cf9200_no_backing(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
            shader_dwords, config, *table);
        if (cf9200_candidate && std::getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[gta-cf9200-no-backing] code=0x%llx valid=%d resources=%zu\n",
                         static_cast<unsigned long long>(code_addr),
                         static_cast<int>(cf9200_valid), table->resources.size());
        assign_convention_bindings(*table, 2);

        // PROSPER_COMPUTEMAP=1 -- every storage-image destination a compute dispatch binds. The
        // graphics-side DRAWMAP cannot see these, so without it a surface written only by compute
        // looks like a surface nothing writes at all -- which is exactly the wrong turn #3126 took.
        if (getenv("PROSPER_COMPUTEMAP") && table) {
            static std::mutex kmx; static std::set<std::pair<uint64_t, uint64_t>> kseen;
            std::lock_guard<std::mutex> lk(kmx);
            for (const ShaderResource& r : table->resources) {
                if (r.cls != ResourceClass::StorageImage) continue;
                if (!kseen.insert({code_addr, r.gpu_addr}).second) continue;
                fprintf(stderr, "[computemap] program=0x%llx writes 0x%llx %ux%u binding=%u\n",
                        (unsigned long long)code_addr, (unsigned long long)r.gpu_addr,
                        r.width, r.height, r.binding);
            }
        }
        ComputeItem item;
        // Fold-time half of the paired read: the descriptors this dispatch will use were just
        // resolved from guest memory, so sample that memory NOW, before anything else runs.
        compute_memprobe_at_fold(code_addr, dispatch_index, config.user_sgprs);
        item.spirv = recompile_compute_shader_cached(
            (const uint32_t*)(uintptr_t)code_addr, 0x10000, table.get(), config, nullptr,
            recompile_diagnostic);
        item.user_sgprs = config.user_sgprs;
        item.required_subgroup_size = config.native_subgroup_size;
        item.cpu_fast_path = classify_compute_cpu_fast_path(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)), shader_dwords);
        item.recompile_config = config;
        // Capture-bound SPIR-V deliberately uses the device-independent storage path, but v39 raw
        // replay needs the capability mask that a normal live dispatch would have used.
        item.recompile_config.native_storage_format_support =
            replay_native_storage_format_support;
        item.recompile_config_available = true;
        item.null_guarded_raw_store_validated = item.spirv.size() &&
            table && std::any_of(table->resources.begin(), table->resources.end(),
                                 is_proven_null_guarded_raw_store) &&
            rdna2_gta5_null_guarded_raw_store_dispatch(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, config.user_sgprs.data(), config.user_sgprs.size());
        item.nullable_output_raw_buffer_validated = item.spirv.size() &&
            table && std::any_of(table->resources.begin(), table->resources.end(),
                                 is_proven_null_nullable_raw_buffer) &&
            rdna2_gta5_nullable_output_dispatch(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, config, *table);
        // Independent of the resource table by design: an `s_endpgm`-only program has no memory
        // effect no matter what its V#s declare, so this proof reads the code and nothing else.
        // AFTER translation, so this is an emission result rather than an intention. A program the
        // selectors accept can still emit nothing -- a structured loop, or a phase ordinal that does
        // not exist -- and the host must not read or clear guest-visible dwords no shader writes.
        // Read from the MODULE the backend will run, so it is this dispatch's answer rather than
        // this address's history: a structured loop or a phase ordinal the program lacks compiles to
        // a module with no witness, and a cache hit returns whichever module the key selected.
        item.trip_witness_instrumented = !program_uses_guest_gds_for_item &&
            spirv_writes_trip_witness(item.spirv);
        item.terminator_only_program_validated = rdna2_program_is_terminator_only(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)), shader_dwords);
        item.gta5_cf9200_no_backing_validated = item.spirv.size() &&
            table && std::any_of(table->resources.begin(), table->resources.end(),
                                 is_proven_gta5_cf9200_no_backing) &&
            rdna2_gta5_cf9200_no_backing_dispatch(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, config, *table);
        // The trip-bound witness writes into the internal GDS buffer, so that binding must exist
        // whenever the bound instruments this program — including for a program that uses no GDS of
        // its own. Injected HERE, at table finalization, rather than beside the `uses_gds` test:
        // that test sits inside a conditional resource-build region which a program can skip, and
        // when it did, the emitted module declared binding 127 while the table did not carry it. The
        // contract check then rejected the module and the dispatch was DECLINED — so the device
        // survived because the program never ran, which is indistinguishable from "the bound worked"
        // in every artifact except the one line that says `skip invalid descriptor contract`.
        if (compute_trip_witness_active(code_addr) && table &&
            std::none_of(table->resources.begin(), table->resources.end(),
                         [](const ShaderResource& r) {
                             return r.binding == kComputeInternalGdsBinding;
                         })) {
            ShaderResource witness;
            witness.cls = ResourceClass::ConstantBuffer;
            witness.format = DataFormat::Uint32;
            witness.num_components = 1;
            witness.binding = kComputeInternalGdsBinding;
            witness.size = static_cast<uint32_t>(g_compute_gds.size());
            witness.stride = 4;
            witness.host_data = g_compute_gds.data();
            witness.host_data_size = g_compute_gds.size();
            table->resources.push_back(witness);
        }
        item.resources = std::move(table);
        item.launch = launch;
        item.code_addr = code_addr;
        item.dispatch_index = dispatch_index;
        item.submit_no = submit_no;
        item.command_order = dispatch.command_order;
        if (item.spirv.empty()) {
            log_compute_dispatch(code_addr, submit_no, dispatch_index, dispatch.command_order,
                                 "recompile-empty");
            record_failure(RealizationFailureReason::ShaderRecompile, item.resources, item.spirv,
                           &config);
            // Report the bindings a recompile-failed program DID resolve. Without this call the one
            // population the watch is most often pointed at -- the failing shaders -- produced no rows
            // at all, so its silence read as "these programs do not bind the address" when the
            // instrument had simply never looked at them. The row is labelled partial because the
            // table behind it is incomplete by construction (see the contract above the function).
            report_compute_binding_watch(code_addr, item.resources.get(),
                                         ComputeBindOutcome::RecompileEmpty);
            // PROSPER_DYNTRACE_FAIL=1: replay the FAILED compute program's resource build with the
            // const-fold walk trace + user-data dump forced on (once per distinct program) — the
            // compute analog of the graphics VS/PS fail-replay (gpu_execute.hpp).
            //
            // STALE TEXT REMOVED, and what it cost is why this note replaces it rather than deleting
            // it silently. This comment used to assert that compute dispatches "only run
            // build_shader_resources (the DIRECT front-half table) and NEVER the const-fold
            // resolve_dynamic_fetch that graphics stages use". That stopped being true when
            // add_compute_buffer_resources landed: the compute table build calls it (see the #590
            // block above), and it calls resolve_dynamic_fetch directly. An agent investigating GTA V's
            // black world read this comment, cited it with a file:line as the mechanism, and published
            // "the fold is never run for compute, so run it" as the fix — for a fold that had been
            // running the whole time. A stale comment is read as verified fact precisely because it
            // sits next to the code it describes.
            //
            // What the trace is actually for: it reveals whether a failing dispatch's descriptors are
            // const-fold-resolvable (i.e. loaded via a foldable s_load chain), which is the ground
            // truth for #590. Read-only: it does not change what the executor binds.
            if (dyntrace_failed_shader_enabled(code_addr)) {
                static std::set<uint64_t> traced_cs;
                if (traced_cs.insert(code_addr).second) {
                    std::fprintf(stderr,
                                 "[dynfail] replaying PRE-SPECIALIZATION raw COMPUTE 0x%llx "
                                 "resource build with trace:\n",
                                 (unsigned long long)code_addr);
                    std::fprintf(stderr, "[dynfail]   compute user-data SGPRs (s0..s%u):\n", kUserSgprs - 1);
                    std::fprintf(stderr,
                                 "[dynfail]   launch groups=%ux%ux%u threads=%ux%ux%u local=%ux%ux%u "
                                 "user_sgprs=%u tgid=%u/%u/%u\n",
                                 launch.groups_x, launch.groups_y, launch.groups_z,
                                 launch.threads_x, launch.threads_y, launch.threads_z,
                                 launch.local_x, launch.local_y, launch.local_z, user_count,
                                 (unsigned)config.tgid_x_en, (unsigned)config.tgid_y_en,
                                 (unsigned)config.tgid_z_en);
                    for (uint32_t i = 0; i < kUserSgprs; i += 4)
                        std::fprintf(stderr, "[dynfail]     s%-2u: %08x %08x %08x %08x\n", i,
                                     sgprs[i], sgprs[i + 1], sgprs[i + 2], sgprs[i + 3]);
                    std::vector<SrtUse> cs_uses;
                    g_dyntrace_force = true;
                    // Same bound as the production call above. A diagnostic seeded differently from
                    // the path it exists to explain reports a resolution the executor never had —
                    // which is how this fold's phantom-register seeding stayed invisible: the trace
                    // and production agreed with each other while both over-claimed.
                    resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                          sgprs, std::min<uint32_t>(user_count, kUserSgprs), 0,
                                          &cs_uses);
                    g_dyntrace_force = false;
                    std::fprintf(stderr,
                                 "[dynfail]   pre-specialization raw const-fold recovered %zu "
                                 "descriptor use(s):\n",
                                 cs_uses.size());
                    for (const auto& u : cs_uses) {
                        if (u.kind == 0) {
                            std::fprintf(stderr,
                                         "[dynfail]     TEX/IMG(t8) key=0x%x use_pc=%u "
                                         "t8=%08x:%08x:%08x:%08x:%08x:%08x:%08x:%08x\n",
                                         u.key, u.use_pc, u.t8[0], u.t8[1], u.t8[2], u.t8[3],
                                         u.t8[4], u.t8[5], u.t8[6], u.t8[7]);
                        } else if (u.kind == 1) {
                            const DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                            std::fprintf(stderr,
                                         "[dynfail]     BUF(v4) key=0x%x use_pc=%u "
                                         "v4=%08x:%08x:%08x:%08x base=0x%llx stride=%u "
                                         "records=%u size=%u required=%u\n",
                                         u.key, u.use_pc, u.v4[0], u.v4[1], u.v4[2], u.v4[3],
                                         (unsigned long long)d.base, d.stride, d.num_records,
                                         d.size_bytes, u.required_size);
                        } else {
                            const DecodedBvhDescriptor d = decode_bvh_descriptor(u.bvh4.data());
                            std::fprintf(stderr,
                                         "[dynfail]     BVH(bvh4) use_pc=%u "
                                         "bvh4=%08x:%08x:%08x:%08x base=0x%llx size=%llu "
                                         "type=%u tri_mode=%u box64=%u sort=%u grow=%u\n",
                                         u.use_pc, u.bvh4[0], u.bvh4[1], u.bvh4[2], u.bvh4[3],
                                         (unsigned long long)d.base,
                                         (unsigned long long)d.size_bytes, d.type,
                                         (unsigned)d.triangle_return_mode,
                                         (unsigned)d.box_node_64b, (unsigned)d.sort_enabled,
                                         (unsigned)d.box_grow);
                        }
                    }
                }
            }
            if (report_compute_recompile_skip_once(recompile_diagnostic)) {
                // #2412: print the table this program was offered, next to the keys the shader looks
                // up by. Every one of GTA V's 951 recompiler rejects is `mode=unresolved-operand`
                // (#2416) — the lowering exists and the descriptor does not resolve — so the useful
                // question is no longer "which instruction" but "which KEY missed". The recompiler
                // matches a resource three ways: by fetch pc, by SRT offset, and by SGPR base; a
                // resource carrying `srt=0xffffffff sgpr=0xffffffff` can be matched by none of them
                // and is invisible to every lookup, however correct its address and size are.
                // Ungated. This block runs at most ONCE PER PROGRAM (it is inside
                // report_compute_recompile_skip_once), so its cost is twelve programs' worth of
                // lines on a routed GTA V boot -- while PROSPER_DBG, the gate it used to sit
                // behind, produces a ~1.5 GB log and desyncs the pad script badly enough that the
                // route never reaches the phase being diagnosed. A diagnostic reachable only by a
                // switch that destroys the repro is not reachable. Same reasoning as the skip line
                // itself, which was ungated for exactly this in an earlier commit.
                {
                    if (!item.resources || item.resources->resources.empty()) {
                        std::fprintf(stderr,
                                     "[compute-table] program 0x%llx has NO resources at all\n",
                                     (unsigned long long)code_addr);
                    } else {
                        size_t keyless = 0;
                        for (const auto& r : item.resources->resources) {
                            const bool no_key = r.srt_offset == 0xffffffffu &&
                                                r.sgpr_base == 0xffffffffu &&
                                                r.fetch_pc == 0xffffffffu;
                            keyless += no_key ? 1 : 0;
                            std::fprintf(stderr,
                                         "[compute-table] rt=%p program 0x%llx binding=%u class=%u "
                                         "addr=0x%llx size=%llu stride=%u srt=0x%x sgpr=%u pc=%u%s\n",
                                         (const void*)item.resources.get(),
                                         (unsigned long long)code_addr, r.binding,
                                         static_cast<unsigned>(r.cls),
                                         (unsigned long long)r.gpu_addr,
                                         (unsigned long long)r.size, r.stride, r.srt_offset,
                                         r.sgpr_base, r.fetch_pc, no_key ? "  <-- UNMATCHABLE" : "");
                        }
                        std::fprintf(stderr,
                                     "[compute-table] program 0x%llx: %zu resource(s), %zu with no "
                                     "lookup key\n",
                                     (unsigned long long)code_addr,
                                     item.resources->resources.size(), keyless);
                    }
                }
                // PROSPER_SHADER_DUMP=<dir>: write the failed COMPUTE program's raw bytes for offline
                // shader_inspect, mirroring the graphics VS/PS dump (gpu_execute.hpp). The graphics
                // path was the only dumper, so a failing dispatch's CFG could not be mapped offline.
                // Keep the established 64 KiB diagnostic window: some compiler-generated branches
                // jump thousands of dwords forward even when the first rejection is near the entry.
                // Bound the raw fwrite to the bytes the decoder actually PROVED readable
                // (registered_shader_dwords is guest_readable-checked and capped at 0x4000 dwords ==
                // 64 KiB), not a fixed 0x10000 — a short shader at the tail of its mapping would
                // otherwise over-read past the mapped page into a SIGSEGV inside the dump (#1209).
                if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                    const size_t dump_bytes = std::min(shader_dwords * sizeof(uint32_t), size_t(0x10000));
                    char fn[512];
                    snprintf(fn, sizeof fn, "%s/exec_cs_%llx.bin", dd,
                             (unsigned long long)code_addr);
                    if (FILE* f = fopen(fn, "wb")) {
                        fwrite(reinterpret_cast<const void*>(static_cast<uintptr_t>(code_addr)),
                               1, dump_bytes, f);
                        fclose(f);
                    } else if (getenv("PROSPER_DBG")) {
                        std::fprintf(stderr, "[shader-dump] cannot open %s: %s\n",
                                     fn, std::strerror(errno));
                    }
                }
            }
            note_compute_program_outcome(code_addr, false, launch.groups_x, launch.groups_y,
                                        launch.groups_z, launch.local_x, launch.local_y);
            continue;
        }
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
        if (!report.ok()) {
            record_failure(RealizationFailureReason::DescriptorContract, item.resources, item.spirv,
                           &config);
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second) {
                std::fprintf(stderr, "[compute] skip invalid descriptor contract for program 0x%llx\n",
                             (unsigned long long)code_addr);
                // WHICH binding, and why. The detail below existed already and was reachable only
                // through PROSPER_DBG, whose output on a routed title runs to hundreds of thousands
                // of lines -- so the one thing a reader needs in order to FIX a declined dispatch
                // cost a firehose, and the bare skip line above (once per program, by design) was
                // what people quoted instead. `PROSPER_COMPUTE_DESCRIPTOR_DETAIL=1` is the same
                // report under its own switch; PROSPER_DBG keeps working unchanged. It fires at most
                // once per program, so it cannot become a firehose itself.
                // Motivating case: Astro Bot's world-map light-list producer 0x5006e8500 is declined
                // here, which is why the pixel shader's per-tile walk chases an all-zero pool and
                // hangs the device (#3214).
                static const bool descriptor_detail =
                    std::getenv("PROSPER_DBG") != nullptr ||
                    std::getenv("PROSPER_COMPUTE_DESCRIPTOR_DETAIL") != nullptr;
                if (descriptor_detail) for (const auto& issue : report.issues) {
                    std::fprintf(stderr,
                                 "[compute-descriptor] %s binding=%u expected=%s actual=%s required=%llu available=%llu error=%d\n",
                                 descriptor_issue_name(issue.code), issue.binding,
                                 spirv_descriptor_kind_name(issue.expected),
                                 spirv_descriptor_kind_name(issue.actual),
                                 (unsigned long long)issue.required_bytes,
                                 (unsigned long long)issue.available_bytes, issue.error ? 1 : 0);
                    if (issue.code == DescriptorIssueCode::ArrayBindingArityMismatch)
                        std::fprintf(stderr,
                                     "[compute-descriptor]   arity binding=%u shader_declares=%u "
                                     "runtime_supplies=%u\n",
                                     issue.binding, issue.shader_count, issue.runtime_count);
                    if (item.resources) for (const auto& resource : item.resources->resources)
                        if (resource.binding == issue.binding)
                            std::fprintf(stderr,
                                         "[compute-resource] binding=%u class=%u addr=0x%llx size=%llu host=%zu stride=%u fmt=%u comps=%u dims=%ux%ux%u srt=0x%x sgpr=%u pc=%u\n",
                                         resource.binding, static_cast<unsigned>(resource.cls),
                                         (unsigned long long)resource.gpu_addr,
                                         (unsigned long long)resource.size, resource.host_data_size,
                                         resource.stride, static_cast<unsigned>(resource.format),
                                         resource.num_components, resource.width, resource.height,
                                         resource.depth, resource.srt_offset, resource.sgpr_base,
                                         resource.fetch_pc);
                }
            }
            note_compute_program_outcome(code_addr, false, launch.groups_x, launch.groups_y,
                                        launch.groups_z, launch.local_x, launch.local_y);
            report_compute_binding_watch(code_addr, item.resources.get(), ComputeBindOutcome::SkippedDescriptors);
            continue;
        }
        // Image bindings (sampled textures + storage images) execute through the live backend's
        // image paths (#590, live_compute.cpp); shapes it cannot bind correctly are skipped there,
        // loudly and per-item, without aborting the rest of the batch.
        note_compute_program_outcome(code_addr, true, launch.groups_x, launch.groups_y,
                                    launch.groups_z, launch.local_x, launch.local_y);
        report_compute_binding_watch(code_addr, item.resources.get(), ComputeBindOutcome::Executed);
        items.push_back(std::move(item));
    }
    return items;
}

bool execute_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    if (!g_compute) return false;
    std::vector<ComputeItem> items = realize_compute_dispatches(st, submit_no);
    return !items.empty() && g_compute(items);
}

struct OrderedGpustateCaptureTrace {
    std::vector<DrawItem> draws;
    std::vector<ComputeItem> computes;
    std::vector<OperationRealizationFailure> failures;
    std::vector<SubmitOperation> noops;
    PendingGpuCapture* pending_capture = nullptr;
};

static void omit_noop_dispatches(std::vector<SubmitOperation>& operations,
                                 const std::vector<SubmitOperation>& noops) {
    std::erase_if(operations, [&](const SubmitOperation& operation) {
        return operation.kind == SubmitOperationKind::Dispatch &&
            std::any_of(noops.begin(), noops.end(), [&](const SubmitOperation& noop) {
                return noop.kind == operation.kind && noop.index == operation.index &&
                       noop.command_order == operation.command_order;
            });
    });
}

static OrderedSubmitResult execute_ordered_gpustate(
    const GpuState& st, uint32_t width, uint32_t height, uint64_t submit_no,
    const LiveRenderFn& render, const LiveComputeFn& compute,
    OrderedGpustateCaptureTrace* capture_trace = nullptr,
    const std::vector<DrawItem>* eager_draws = nullptr);

bool execute_nonrender_submit_work(const GpuState& st, uint64_t submit_no) {
    if (st.dma_copies.empty() && (!g_compute || st.dispatches.empty())) return false;
    GuestReadableSubmitScope guest_readable_scope;
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitBegin, submit_no);
    // Compute-only submits never reach execute_ordered_and_present(), but they are exactly where an
    // unsupported dispatch can disappear before there is a realized ComputeItem to select.  Give
    // the environment capture path the same semantic pre-submit hook as rendering submits so
    // PROSPER_GPU_CAPTURE_COMPUTE_ADDR can retain the raw program, table snapshots, and explicit
    // realization failure.  Interactive/F9 captures remain draw-triggered: begin_requested_gpu_capture
    // deliberately does not consume an armed interactive request when semantic_draw_count is zero.
    std::unique_ptr<PendingGpuCapture> pending_capture;
    std::vector<SubmitOperation> capture_operations;
    const bool has_indirect_dispatch = std::any_of(
        st.dispatches.begin(), st.dispatches.end(),
        [](const GpuState::Dispatch& dispatch) { return dispatch.indirect; });
    // Preserve the established exact-trace path for ordinary no-DMA compute submits, and extend it
    // to DMA-backed indirect consumers whose arguments cannot be realized until after the copy.
    const bool can_defer_capture = st.dma_copies.empty() || has_indirect_dispatch;
    if (const char* capture_path = std::getenv("PROSPER_GPU_CAPTURE");
        capture_path && *capture_path) {
        capture_operations = plan_submit_operations(st);
        pending_capture = begin_requested_gpu_capture(
            {}, {}, capture_operations, present_width(), present_height(), &st, submit_no,
            static_cast<uint64_t>(st.draws.size()), nullptr, can_defer_capture);
    }
    snapshot_pending_gpu_capture_compute_gds(
        pending_capture.get(), g_compute_gds.data(), g_compute_gds.size());
    OrderedGpustateCaptureTrace capture_trace;
    capture_trace.pending_capture = pending_capture.get();
    const OrderedSubmitResult result = execute_ordered_gpustate(
        st, 0, 0, submit_no, {}, g_compute,
        pending_capture && can_defer_capture ? &capture_trace : nullptr);
    if (pending_capture) {
        std::string error;
        notify_compute_authority_unknown(
            ComputeAuthorityBoundaryKind::Capture, submit_no);
        omit_noop_dispatches(capture_operations, capture_trace.noops);
        if (!finish_requested_gpu_capture(
                std::move(pending_capture), {}, error,
                can_defer_capture ? &capture_trace.draws : nullptr,
                can_defer_capture ? &capture_trace.computes : nullptr,
                can_defer_capture ? &capture_operations : nullptr,
                can_defer_capture ? &st : nullptr,
                can_defer_capture ? &capture_trace.failures : nullptr))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitEnd, submit_no);
    return !st.dma_execution_rejected &&
           (result.compute_executed || !st.dma_copies.empty() ||
            !st.ordered_memory_effects.empty());
}

void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    const char* enabled = getenv("PROSPER_COMPUTELOG");
    const char* dim_env = getenv("PROSPER_COMPUTELOG_DIM");
    // PROSPER_SRTDUMP arms the shader-resource-table dump below and NOTHING else. It needs the same
    // per-dispatch walk as COMPUTELOG, which is why it enters this function, but it must not turn on
    // COMPUTELOG's per-dispatch prose -- that prose is what makes a COMPUTELOG run on this title's
    // gameplay route unreadable, and it carries a 4 KiB FNV hash of the shader code per dispatch.
    //
    // `prose` is what the two PRE-EXISTING switches ask for, and it gates the tail exactly as they
    // did before this diagnostic existed. Getting this wrong is not cosmetic: the first version of
    // this switch let `srt_only` un-gate the tail, so an "SRTDUMP-only" run emitted 352,940
    // `[compute]` lines it never claimed to, and the resulting log was read as if it were the small
    // targeted one the comment promised.
    const char* srt_env = getenv("PROSPER_SRTDUMP");
    const bool srt_only = srt_env && *srt_env;
    const bool prose = (enabled && *enabled) || (dim_env && *dim_env);
    if (!prose && !srt_only) return;

    uint32_t want_w = 0, want_h = 0;
    if (dim_env && *dim_env && sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[compute] invalid PROSPER_COMPUTELOG_DIM='%s' (expected WxH)\n", dim_env);
        }
        want_w = want_h = 0;
    }

    namespace P = prosper::agc::Pm4;
    auto rd = [](const RegisterFile& regs, uint32_t off) {
        auto it = regs.find(off); return it == regs.end() ? 0u : it->second;
    };
    size_t matched = 0;
    for (size_t i = 0; i < st.dispatches.size(); ++i) {
        const auto& d = st.dispatches[i];
        const GpuState& ds = d.state ? *d.state : st;
        const ComputeLaunchDimensions launch = resolve_compute_launch(d);
        const uint64_t code_addr = compute_dispatch_code_addr(st, d);
        const auto* hdr = static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(code_addr));

        uint32_t range_start = 0;
        if (hdr && hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                                   sizeof(AgcShaderSpecials))) {
            uint32_t s = hdr->specials->user_data_range_start;
            uint32_t e = hdr->specials->user_data_range_end;
            if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        }

        ShaderResourceTable table;
        uint32_t sgprs[kUserSgprs] = {};
        // `user_data` can be NON-NULL yet point at unmapped guest memory -- build_shader_resources
        // probes for exactly this and documents the observed case (#713, PPSA02664), returning an
        // empty table rather than faulting. TWO readers below dereference it, in sibling scopes, so
        // the probe is hoisted here and shared: a diagnostic that SIGSEGVs where the renderer does
        // not turns an investigation into a crash report about the instrument.
        const AgcShaderUserData* ud = hdr ? hdr->user_data : nullptr;
        const bool ud_ok = ud && guest_readable((uint64_t)(uintptr_t)ud, sizeof(AgcShaderUserData));
        if (hdr) {
            read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
            table = build_shader_resources(*hdr, sgprs, kUserSgprs, 0);
            assign_convention_bindings(table, 2);

            // SRT CONTENTS. `build_shader_resources` reads sharps and the EUD and has no
            // shader-resource-table path, and every SRT-declaring header in a routed GTA V gameplay
            // run declares neither -- `sharps={0,0,0,0} eud=0` on 138,034 of 138,034 headers across
            // 88 of 88 programs (#2705). So the AGC-header path resolves nothing for them and the
            // table's own bytes are the only way to see what they describe. Nothing dumps them; this
            // does.
            //
            // WHAT THIS DOES NOT MEAN, because an earlier version of this comment said it and it is
            // false: prosper is NOT blind to this channel. `add_compute_buffer_resources` (:5665,
            // called :7570) const-folds descriptors loaded with `s_load_dwordx4/x8 sN, s[ptr:ptr+1],
            // <imm>` from a user-data table -- see the SrtUse contract in gpu_execute.hpp -- and it
            // fires on these very programs, e.g. `[compute-table] program 0x205b657200 …
            // addr=0x20037cf620 stride=32 srt=0x10`. The gap is narrower and stranger than "no path
            // exists": the recovered key set starts at 0x10 and byte offset 0 never appears in it,
            // while the shaders do load descriptors there (#2757).
            //
            // Recorded at this length because the false version travelled: it was written here, then
            // into a PR body, and would have been the wording the next agent inherited -- from code,
            // which outlives the PR that a correction lives in.
            //
            // Deciding that needs the table's own bytes, and nothing dumps them. This does, and
            // deliberately dumps CANDIDATES rather than naming one pointer as the SRT: its position
            // in the user-data block is not established, and a diagnostic that picks one and labels
            // it "the SRT" would manufacture the fact it exists to measure. Every readable dword
            // pair in the window is shown with the first `srt_size_dw` dwords behind it. Whichever
            // holds descriptors will be evident; if none does, that is equally the answer.
            if (srt_only && ud_ok && ud->srt_size_dw) {
                const uint32_t want = ud->srt_size_dw;
                const uint32_t shown = want < 64 ? want : 64;   // srt_size_dw is a uint16
                fprintf(stderr, "[srtdump] code=0x%llx srt_size_dw=%u (showing %u) "
                                "sharps={%u,%u,%u,%u} eud=%u\n",
                        (unsigned long long)code_addr, want, shown,
                        ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                        ud->sharp_resource_count[2], ud->sharp_resource_count[3], ud->eud_size_dw);
                bool any = false;
                for (uint32_t k = 0; k + 1 < kUserSgprs; k++) {
                    const uint64_t raw = (uint64_t)sgprs[k] | ((uint64_t)sgprs[k + 1] << 32);
                    if (raw <= 0x10000) continue;
                    uint64_t addr = 0;
                    for (uint64_t mask : {~uint64_t{0}, uint64_t{0xFFFFFFFFFFFF},
                                          uint64_t{0xFFFFFFFFFF}}) {
                        const uint64_t cand = raw & mask;
                        if (cand > 0x10000 && guest_readable(cand, shown * 4u)) { addr = cand; break; }
                    }
                    if (!addr) continue;
                    any = true;
                    fprintf(stderr, "[srtdump]   dw%u -> 0x%llx:", k, (unsigned long long)addr);
                    const uint32_t* words = reinterpret_cast<const uint32_t*>(addr);
                    uint32_t nonzero = 0;
                    for (uint32_t w = 0; w < shown; w++) {
                        fprintf(stderr, " %08x", words[w]);
                        nonzero += words[w] != 0;
                    }
                    fprintf(stderr, "  (nz=%u/%u)\n", nonzero, shown);
                }
                if (!any)
                    fprintf(stderr, "[srtdump]   no readable pointer in the user-data window\n");
            }
        }

        // A compute shader can carry only an inline direct type-1 V# and no sharp descriptors. Dump
        // its metadata and bound SGPRs once per program if resource decoding still returns empty.
        // This turns the next unsupported layout into a reproducible decode problem instead of
        // another blind `resources=0` investigation.
        if (hdr && table.resources.empty() && enabled && *enabled) {
            static std::set<uint64_t> logged_empty;
            if (logged_empty.insert(code_addr).second) {
                fprintf(stderr, "[compute] empty-resource metadata code=0x%llx type=%u ud=%p%s",
                        (unsigned long long)code_addr, hdr->type, (const void*)ud,
                        ud && !ud_ok ? " (UNMAPPED)" : "");
                if (ud_ok) {
                    fprintf(stderr, " eud=%u srt=%u direct_count=%u sharp={%u,%u,%u,%u}",
                            ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                            ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
                }
                fprintf(stderr, "\n[compute]   user_sgprs:");
                for (uint32_t s = 0; s < kUserSgprs; ++s) fprintf(stderr, " %08x", sgprs[s]);
                fprintf(stderr, "\n");

                if (ud_ok && ud->direct_resource_offset && ud->direct_resource_count) {
                    fprintf(stderr, "[compute]   direct offsets:");
                    for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; ++t)
                        fprintf(stderr, " [%u]=%u", t, ud->direct_resource_offset[t]);
                    fprintf(stderr, "\n");
                    const uint32_t reg = ud->direct_resource_count > 1 ? ud->direct_resource_offset[1] : 0xffffu;
                    if (reg != 0xffffu && reg + 4 <= kUserSgprs) {
                        const DecodedBufferDescriptor d = decode_buffer_descriptor(&sgprs[reg]);
                        fprintf(stderr, "[compute]   type1 V# reg=%u base=0x%llx stride=%u records=%u "
                                        "size=%u fmt=%u comps=%u\n",
                                reg, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                                (unsigned)d.format, d.num_components);
                    }
                }
            }
        }

        bool dim_match = !want_w || !want_h;
        if (!dim_match) {
            for (const auto& r : table.resources)
                if (r.width == want_w && r.height == want_h) { dim_match = true; break; }
        }
        if (!dim_match) continue;
        matched++;
        // Everything past here is COMPUTELOG/COMPUTELOG_DIM's output, including the 4 KiB code hash.
        // An SRTDUMP-only run has already printed what it came for.
        if (!prose) continue;

        uint64_t code_hash = 1469598103934665603ull;
        if (code_addr && guest_readable(code_addr, 4096)) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(code_addr));
            for (size_t n = 0; n < 4096; ++n) { code_hash ^= p[n]; code_hash *= 1099511628211ull; }
        } else {
            code_hash = 0;
        }
        fprintf(stderr,
                "[compute] submit=%llu dispatch=%zu threads=%ux%ux%u local=%ux%ux%u "
                "groups=%ux%ux%u modifier=0x%llx "
                "code=0x%llx hash4k=%016llx header=%s resources=%zu\n",
                (unsigned long long)submit_no, i,
                launch.threads_x, launch.threads_y, launch.threads_z,
                launch.local_x, launch.local_y, launch.local_z,
                launch.groups_x, launch.groups_y, launch.groups_z,
                (unsigned long long)d.modifier, (unsigned long long)code_addr,
                (unsigned long long)code_hash, hdr ? "yes" : "no", table.resources.size());
        for (const auto& r : table.resources) {
            fprintf(stderr,
                    "[compute]   cls=%u binding=%u addr=0x%llx size=%u dims=%ux%u "
                    "fmt=%u comps=%u tile=%u sgpr=%u srt=0x%x\n",
                    (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                    r.width, r.height, (unsigned)r.format, r.num_components, r.tile_mode,
                    r.sgpr_base, r.srt_offset);
        }
    }

    if (want_w && want_h && !st.dispatches.empty() && matched == 0 && enabled && enabled[0] == 'a') {
        fprintf(stderr, "[compute] submit=%llu dispatches=%zu: no resource matched %ux%u\n",
                (unsigned long long)submit_no, st.dispatches.size(), want_w, want_h);
    }
}

void diagnose_resource_provenance(const GpuState& st, uint64_t submit_no) {
    // This function does two separable jobs: it RECORDS colour-target writes into the shared
    // guest-write history, and it PRINTS a consumer/producer diagnostic for one target extent. Only
    // the second needs PROSPER_PROVENANCE_DIM. Returning early for both made the history silently
    // kind-incomplete under PROSPER_WRITER_PROVENANCE=1 — the switch documented as the explicit
    // "retain everything" request — so every `last_guest_write_overlap` consumer answered "no
    // writer" for a range a colour target had in fact written, with nothing in the log to say the
    // recorder had never been armed. Measured on CrossWorlds: `color=0` recorded across a whole boot
    // with writer provenance explicitly on, against `compute-buffer=54` and `write-data=7`.
    const char* dim_env = getenv("PROSPER_PROVENANCE_DIM");
    uint32_t want_w = 0, want_h = 0;
    bool record_only = !dim_env || !*dim_env;
    if (!record_only && (sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2 || !want_w || !want_h)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[provenance] invalid PROSPER_PROVENANCE_DIM='%s' (expected WxH)"
                            " — recording colour targets anyway\n", dim_env);
        }
        // Fall back to RECORD-ONLY instead of returning. A malformed value must not disarm the
        // colour recorder, because the address watch's banner states unconditionally that colour is
        // armed — and a banner that denies a void zero is worse than the void zero it replaced.
        // The reachable spelling is `PROSPER_PROVENANCE_DIM=1`: every other switch in this project
        // is a boolean `=1`, this one is `WxH`, `sscanf` returns 1, and before this the function
        // returned before recording anything. See #2149 for the general shape.
        record_only = true;
        want_w = want_h = 0;
    }
    if (record_only && !writer_provenance_enabled()) return;
    const size_t min_draws = [] {
        const char* e = getenv("PROSPER_PROVENANCE_MIN_DRAWS");
        return e ? static_cast<size_t>(strtoull(e, nullptr, 0)) : size_t{0};
    }();

    struct ColorWrite {
        uint64_t submit = 0;
        uint64_t draw_submit = 0;
        size_t draw = 0;
        uint64_t vs = 0, ps = 0;
        uint32_t width = 0, height = 0;
        GpuState::Draw draw_record{};
    };
    static std::unordered_map<uint64_t, ColorWrite> last_color_write;
    static std::set<uint64_t> recorded_color_ranges;
    static uint64_t draw_submit_ordinal = 0;
    const uint64_t this_draw_submit = st.draws.empty() ? draw_submit_ordinal : draw_submit_ordinal++;

    // Record-only mode still walks every draw to capture its colour targets; it just does not
    // inspect consumers, which is the half that needs a target extent to select on.
    const bool inspect_consumers = !record_only && st.draws.size() >= min_draws;
    for (size_t i = 0; i < st.draws.size(); ++i) {
        const GpuState& ds = st.draws[i].state ? *st.draws[i].state : st;
        const RenderState rs = extract_render_state(ds);

        // Query before recording this draw's target: a feedback draw should resolve to the preceding
        // writer, not identify itself as its own producer.
        if (inspect_consumers && rs.ps_addr) {
            auto prt = build_stage_table(ds, rs.ps_addr, true, st.draws[i].index_count);
            if (prt) for (const auto& r : prt->resources) {
                if (r.width != want_w || r.height != want_h) continue;
                const uint64_t resource_size = r.size ? r.size :
                    static_cast<uint64_t>(r.width) * r.height * 4;
                auto writer = last_guest_write_overlap(r.gpu_addr, resource_size);
                if (writer) {
                    fprintf(stderr,
                            "[provenance]   latest-recorded-overlap kind=%s seq=%llu "
                            "range=[0x%llx,+0x%llx) "
                            "submit=%llu item=%llu order=%llu identity=0x%llx dims=%ux%u\n",
                            guest_writer_kind_name(writer->kind),
                            (unsigned long long)writer->sequence,
                            (unsigned long long)writer->addr,
                            (unsigned long long)writer->size,
                            (unsigned long long)writer->submit,
                            (unsigned long long)writer->item,
                            (unsigned long long)writer->order,
                            (unsigned long long)writer->identity,
                            writer->width, writer->height);
                } else {
                    fprintf(stderr,
                            "[provenance]   no recorded color/compute/DMA/WRITE_DATA overlap for "
                            "[0x%llx,+0x%llx)\n",
                            (unsigned long long)r.gpu_addr,
                            (unsigned long long)resource_size);
                }
                auto it = last_color_write.find(r.gpu_addr);
                if (it == last_color_write.end()) {
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu order=%llu: "
                            "no prior color-target write\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit,
                            (unsigned long long)st.draws[i].command_order);
                } else {
                    const ColorWrite& w = it->second;
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu order=%llu: "
                            "last color write submit=%llu "
                            "draw_submit=%llu draw=%zu target_extent=%ux%u "
                            "vs=0x%llx ps=0x%llx\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit,
                            (unsigned long long)st.draws[i].command_order,
                            (unsigned long long)w.submit,
                            (unsigned long long)w.draw_submit, w.draw, w.width, w.height,
                            (unsigned long long)w.vs, (unsigned long long)w.ps);
                    static std::set<uint64_t> probed;
                    if (probed.insert(r.gpu_addr).second && w.draw_record.state) {
                        DrawItem producer;
                        bool realized = realize_draw_item(*w.draw_record.state, &w.draw_record,
                                                         w.draw_record.index_count, 0x10000, true, producer);
                        fprintf(stderr,
                                "[provenance] producer-realize addr=0x%llx result=%s extent=%ux%u "
                                "items-target=0x%llx\n",
                                (unsigned long long)r.gpu_addr, realized ? "success" : "dropped",
                                producer.color0_width, producer.color0_height,
                                (unsigned long long)producer.color0_base);
                    }
                }
            }
        }

        if (rs.color0_base) {
            // Only the consumer diagnostic reads last_color_write, and its value retains a draw
            // record (which pins a GpuState) for the process lifetime, one per distinct colour base.
            // Record-only mode has no reader, so populating it there would leak that retention onto
            // every writer-provenance run for nothing.
            if (!record_only)
                last_color_write[rs.color0_base] = {
                    submit_no, this_draw_submit, i, rs.es_addr, rs.ps_addr,
                    rs.color0_width, rs.color0_height, st.draws[i]
                };
            // The exact-address map above retains every latest color writer. The generic overlap
            // history needs only one representative event per target range; recording every draw
            // adds millions of mutex/hash operations during Dead Cells' submit-heavy startup.
            if (recorded_color_ranges.insert(rs.color0_base).second) {
                const uint64_t bytes = static_cast<uint64_t>(rs.color0_width) * rs.color0_height * 4;
                record_guest_write(GuestWriterKind::ColorTarget, rs.color0_base, bytes,
                                   submit_no, i, st.draws[i].command_order, rs.ps_addr,
                                   rs.color0_width, rs.color0_height);
            }
        }
        if (rs.color1_base) {
            if (!record_only)
                last_color_write[rs.color1_base] = {
                    submit_no, this_draw_submit, i, rs.es_addr, rs.ps_addr,
                    rs.color1_width, rs.color1_height, st.draws[i]
                };
            if (recorded_color_ranges.insert(rs.color1_base).second) {
                const uint64_t bytes = static_cast<uint64_t>(rs.color1_width) * rs.color1_height * 4;
                record_guest_write(GuestWriterKind::ColorTarget, rs.color1_base, bytes,
                                   submit_no, i, st.draws[i].command_order, rs.ps_addr,
                                   rs.color1_width, rs.color1_height);
            }
        }
    }
}

std::vector<SubmitOperation> plan_submit_operations(const GpuState& st) {
    std::vector<SubmitOperation> operations;
    operations.reserve(st.draws.size() + st.dispatches.size() + st.dma_copies.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        operations.push_back({SubmitOperationKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        if (!direct_compute_dispatch_is_noop(st.dispatches[i]))
            operations.push_back({SubmitOperationKind::Dispatch, i,
                                  st.dispatches[i].command_order});
    for (size_t i = 0; i < st.dma_copies.size(); ++i)
        operations.push_back({SubmitOperationKind::DmaCopy, i, st.dma_copies[i].command_order});
    std::stable_sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });
    return operations;
}

namespace {

template <typename DmaCopyRecord, typename ExecuteDma>
OrderedSubmitResult execute_ordered_items_impl(const std::vector<SubmitOperation>& operations,
                                               const std::vector<DrawItem>& draws,
                                               const std::vector<ComputeItem>& computes,
                                               const std::vector<DmaCopyRecord>& dma_copies,
                                               const LiveRenderFn& render,
                                               const LiveComputeFn& compute,
                                               uint32_t width, uint32_t height,
                                               ExecuteDma&& execute_dma) {
    GuestGpuWriteSubmitScope guest_gpu_write_scope;
    std::unordered_map<size_t, size_t> draw_by_index, compute_by_index;
    for (size_t i = 0; i < draws.size(); ++i) draw_by_index[draws[i].draw_index] = i;
    for (size_t i = 0; i < computes.size(); ++i)
        compute_by_index[static_cast<size_t>(computes[i].dispatch_index)] = i;

    enum class ExecutableKind : uint8_t { Draw, Dispatch, DmaCopy };
    struct ExecutableOperation {
        ExecutableKind kind;
        size_t item;
        uint64_t command_order;
    };
    std::vector<ExecutableOperation> executable;
    bool explicit_dma_operations = false;
    for (const auto& operation : operations) {
        if (operation.kind == SubmitOperationKind::Draw) {
            auto it = draw_by_index.find(operation.index);
            if (it != draw_by_index.end())
                executable.push_back({ExecutableKind::Draw, it->second, operation.command_order});
        } else if (operation.kind == SubmitOperationKind::Dispatch) {
            auto it = compute_by_index.find(operation.index);
            if (it != compute_by_index.end())
                executable.push_back({ExecutableKind::Dispatch, it->second, operation.command_order});
        } else {
            explicit_dma_operations = true;
            if (operation.index < dma_copies.size())
                executable.push_back({ExecutableKind::DmaCopy, operation.index,
                                      operation.command_order});
        }
    }
    // Compatibility for pre-v14 callers whose operation list predates the DMA kind.
    if (!explicit_dma_operations)
        for (size_t i = 0; i < dma_copies.size(); ++i)
            executable.push_back({ExecutableKind::DmaCopy, i, dma_copies[i].command_order});
    std::stable_sort(executable.begin(), executable.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });

    size_t total_spans = 0;
    bool in_draw_span = false;
    for (const auto& operation : executable) {
        if (operation.kind == ExecutableKind::Draw) {
            if (!in_draw_span) ++total_spans;
            in_draw_span = true;
        } else {
            in_draw_span = false;
        }
    }

    OrderedSubmitResult result;
    std::vector<DrawItem> span;
    auto flush_span = [&](bool authoritative_readback = false) {
        if (span.empty() || !render) return;
        LiveRenderPhase saved = g_live_phase;
        g_live_phase = {result.render_spans == 0, result.render_spans + 1 == total_spans,
                        authoritative_readback};
        RenderedFrame rendered = render(span, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
        span.clear();
        ++result.render_spans;
    };
    for (const auto& operation : executable) {
        if (operation.kind == ExecutableKind::Draw) {
            span.push_back(draws[operation.item]);
        } else if (operation.kind == ExecutableKind::Dispatch) {
            flush_span();
            result.compute_executed |= compute && compute({computes[operation.item]});
        } else {
            flush_span(true);
            execute_dma(dma_copies[operation.item]);
        }
    }
    flush_span();
    return result;
}

} // namespace

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<GpuState::DmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items_impl(
        operations, draws, computes, dma_copies, render, compute, width, height,
        [](const GpuState::DmaCopy& copy) {
            std::vector<uint8_t> current_source;
            const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                copy.src, copy.bytes, current_source);
            if (source_result == LiveTargetByteReadResult::InvalidRange) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                 static_cast<unsigned long long>(copy.src), copy.bytes);
                return;
            }
            execute_ordered_dma_copy(
                copy, source_result == LiveTargetByteReadResult::Success
                          ? current_source.data() : nullptr);
        });
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<ReplayDmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items_impl(
        operations, draws, computes, dma_copies, render, compute, width, height,
        [](const ReplayDmaCopy& copy) {
            const uint32_t source_selector = (copy.sels >> 8u) & 0xffu;
            const uint32_t destination_selector = copy.sels & 0xffu;
            const bool source_gds = source_selector == 1u;
            const bool destination_gds = destination_selector == 1u;
            if (source_gds) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA GDS source is unsupported: "
                                 "src=0x%llx dst=0x%llx bytes=%u sels=0x%x\n",
                                 static_cast<unsigned long long>(copy.src),
                                 static_cast<unsigned long long>(copy.dst), copy.bytes,
                                 copy.sels);
                return;
            }
            const bool address_source = (copy.sels & kDmaDataAddressSource) != 0 ||
                                        copy.src > UINT32_MAX;
            if (destination_gds && (source_selector != 3u || !address_source)) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA memory-to-GDS selector form is "
                                 "unsupported: src=0x%llx dst=0x%llx bytes=%u sels=0x%x\n",
                                 static_cast<unsigned long long>(copy.src),
                                 static_cast<unsigned long long>(copy.dst), copy.bytes,
                                 copy.sels);
                return;
            }
            std::vector<uint8_t> current_source;
            const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                copy.src, copy.bytes, current_source);
            if (source_result == LiveTargetByteReadResult::InvalidRange) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                 static_cast<unsigned long long>(copy.src), copy.bytes);
                return;
            }
            const uint8_t* source = source_result == LiveTargetByteReadResult::Success
                ? current_source.data() : copy.source_data;
            const uint64_t source_size = source_result == LiveTargetByteReadResult::Success
                ? current_source.size() : copy.source_size;
            if (!copy.destination_data || !source ||
                copy.bytes > copy.destination_size || copy.bytes > source_size)
                return;
            std::memmove(copy.destination_data, source, copy.bytes);
            if (!destination_gds) {
                set_guest_gpu_write_origin("DMA_DATA(live-target)");
                notify_guest_gpu_write(copy.dst, copy.bytes);
                set_guest_gpu_write_origin(nullptr);
            }
        });
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items(operations, draws, computes,
                                 std::vector<GpuState::DmaCopy>{},
                                 render, compute, width, height);
}

namespace {

struct DrawRealizationBatch {
    using Callback = void (*)(void*, size_t);
    using DrawCallback = void (*)(void*, size_t, size_t);

    void* context = nullptr;
    Callback begin = nullptr;
    DrawCallback draw = nullptr;
    Callback end = nullptr;
    size_t count = 0;
    size_t worker_participants = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> remaining{0};
    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::mutex error_mutex;
    std::exception_ptr error;
};

// One process-lifetime worker set avoids creating dozens of host threads for every guest submit.
// The pool deliberately supports one batch at a time; callers are serialized at run() even when
// independent submit threads arrive concurrently. The submit thread participates as slot zero;
// each persistent worker owns a stable measurement slot.
class DrawRealizationPool {
public:
    explicit DrawRealizationPool(size_t workers) {
        workers_.reserve(workers);
        try {
            for (size_t i = 0; i < workers; ++i)
                workers_.emplace_back([this, i] { worker_main(i); });
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
                ++generation_;
            }
            work_cv_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }

    ~DrawRealizationPool() {
        std::lock_guard run_lock(run_mutex_);
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    size_t worker_count() const { return workers_.size(); }

    void run(const std::shared_ptr<DrawRealizationBatch>& batch, size_t workers_to_use) {
        std::lock_guard run_lock(run_mutex_);
        workers_to_use = std::min(workers_to_use, workers_.size());
        batch->worker_participants = workers_to_use;
        batch->remaining.store(workers_to_use, std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex_);
            active_ = batch;
            ++generation_;
        }
        work_cv_.notify_all();

        run_participant(batch, 0);
        if (workers_to_use) {
            std::unique_lock lock(batch->done_mutex);
            batch->done_cv.wait(lock, [&] {
                return batch->remaining.load(std::memory_order_acquire) == 0;
            });
        }
        // Do not retain the caller-owned context through the process-lifetime pool. A worker that
        // woke for this generation but was not selected may still hold its own shared batch reference;
        // clearing the pool reference is safe after every selected worker has completed.
        {
            std::lock_guard lock(mutex_);
            if (active_ == batch) active_.reset();
        }
        if (batch->error) std::rethrow_exception(batch->error);
    }

private:
    static void remember_error(const std::shared_ptr<DrawRealizationBatch>& batch) {
        std::lock_guard lock(batch->error_mutex);
        if (!batch->error) batch->error = std::current_exception();
    }

    static void run_participant(const std::shared_ptr<DrawRealizationBatch>& batch,
                                size_t measurement_slot) {
        bool began = false;
        try {
            if (batch->begin) batch->begin(batch->context, measurement_slot);
            began = true;
            constexpr size_t kGrain = 4;
            for (;;) {
                const size_t first = batch->next.fetch_add(kGrain, std::memory_order_relaxed);
                if (first >= batch->count) break;
                const size_t last = std::min(first + kGrain, batch->count);
                for (size_t draw = first; draw < last; ++draw)
                    batch->draw(batch->context, draw, measurement_slot);
            }
        } catch (...) {
            remember_error(batch);
        }
        if (began && batch->end) {
            try { batch->end(batch->context, measurement_slot); }
            catch (...) { remember_error(batch); }
        }
    }

    void worker_main(size_t worker_index) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::shared_ptr<DrawRealizationBatch> batch;
            {
                std::unique_lock lock(mutex_);
                work_cv_.wait(lock, [&] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) return;
                seen_generation = generation_;
                batch = active_;
            }
            if (!batch || worker_index >= batch->worker_participants) continue;
            run_participant(batch, worker_index + 1);
            if (batch->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lock(batch->done_mutex);
                batch->done_cv.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::shared_ptr<DrawRealizationBatch> active_;
    uint64_t generation_ = 0;
    bool stopping_ = false;
};

size_t configured_draw_realization_threads() {
    static const size_t threads = [] {
        const unsigned hardware = std::thread::hardware_concurrency();
        size_t result = std::min<size_t>(hardware ? hardware : 4u, 8u);
        if (const char* value = std::getenv("PROSPER_DRAW_REALIZE_THREADS")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0') result = std::clamp<size_t>(parsed, 1u, 32u);
        }
        return std::max<size_t>(result, 1u);
    }();
    return threads;
}

size_t parallel_draw_minimum() {
    static const size_t minimum = [] {
        size_t result = 32;
        if (const char* value = std::getenv("PROSPER_DRAW_REALIZE_MIN_DRAWS")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0') result = std::clamp<size_t>(parsed, 2u, 4096u);
        }
        return result;
    }();
    return minimum;
}

bool parallel_draw_diagnostic_active(bool log) {
    if (log || std::getenv("PROSPER_SERIAL_DRAW_REALIZE")) return true;
    static constexpr const char* diagnostics[] = {
        "PROSPER_RTLOG", "PROSPER_INTERPLOG", "PROSPER_RESDUMP",
        "PROSPER_DYNTRACE_FAIL", "PROSPER_DRAWDIAG", "PROSPER_ONLY_ATLAS",
        "PROSPER_CAPTION_DIAG", "PROSPER_VS_DUMP", "PROSPER_SHADER_DUMP",
        "PROSPER_SHADER_DUMP_SUCCESS", "PROSPER_DRAWLOG", "PROSPER_DBG",
        "PROSPER_STAGE_FOLD_PROFILE", "PROSPER_DESCRIPTOR_VALIDATE",
        "PROSPER_NO_SHADER_CACHE",
        // PROSPER_TEXCONTENT queries is_live_render_target(), and that callback is NOT a pure read:
        // it begins with drain_guest_gpu_writes(), which steals the pending guest-write queue and
        // mutates an unsynchronised map. Calling it from parallel realization workers is undefined
        // behaviour AND perturbs the cache state the diagnostic exists to report. Serialising is the
        // whole reason this list exists.
        "PROSPER_TEXCONTENT",
    };
    for (const char* name : diagnostics)
        if (std::getenv(name)) return true;
    return false;
}

DrawRealizationPool& draw_realization_pool() {
    static DrawRealizationPool pool(configured_draw_realization_threads() - 1);
    return pool;
}

struct ParallelDrawSlot {
    DrawItem item;
    bool realized = false;
};

struct ParallelWorkerMeasurement {
    DrawRealizationPhaseStats draw_before;
    StageTablePhaseStats table_before;
    DrawRealizationPhaseStats draw_delta;
    StageTablePhaseStats table_delta;
    uint64_t readable_calls = 0;
    uint64_t readable_hits = 0;
    uint64_t readable_os_probes = 0;
    std::unique_ptr<GuestReadableSubmitScope> readable_scope;
};

struct ParallelDrawContext {
    const GpuState* state = nullptr;
    uint32_t max_shader_dwords = 0;
    bool retain_shared_shader_words = false;
    bool parent_readable_active = false;
    // PROSPER_DESCRIPTOR_VALIDATE, read ONCE for this submit and handed to every worker (#2287).
    // Previously each worker read it twice per draw, and on Windows getenv takes a process-wide
    // lock, so the cost was contention across workers rather than a constant per call.
    //
    // Read here rather than assumed unset. It is in parallel_draw_diagnostic_active's list, so this
    // path only runs when the variable is clear and passing nullptr would be correct TODAY -- but
    // that would make descriptor validation depend on a diagnostic list it has no visible
    // relationship to, and removing the entry from that list would silently stop validating instead
    // of failing. One getenv per submit costs nothing and owes nothing to that coupling.
    const char* descriptor_validate_mode = nullptr;
    std::vector<ParallelDrawSlot> slots;
    std::vector<ParallelWorkerMeasurement> measurements;
};

void parallel_draw_worker_begin(void* opaque, size_t worker_slot) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    auto& measurement = context.measurements[worker_slot];
    if (worker_slot != 0)
        measurement.readable_scope = std::make_unique<GuestReadableSubmitScope>();
    measurement.draw_before = draw_realization_phase_stats();
    measurement.table_before = stage_table_phase_stats();
}

void parallel_draw_worker_execute(void* opaque, size_t draw_index, size_t) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    const GpuState& state = *context.state;
    const GpuState::Draw& draw = state.draws[draw_index];
    ParallelDrawSlot& slot = context.slots[draw_index];
    slot.realized = realize_draw_item(
        state.state_at_draw(draw_index), &draw, draw.index_count,
        context.max_shader_dwords, false, slot.item, nullptr,
        context.retain_shared_shader_words, &context.descriptor_validate_mode);
    if (slot.realized) {
        slot.item.draw_index = draw_index;
        slot.item.command_order = draw.command_order;
    }
}

void parallel_draw_worker_end(void* opaque, size_t worker_slot) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    auto& measurement = context.measurements[worker_slot];
    const DrawRealizationPhaseStats draw_after = draw_realization_phase_stats();
    const StageTablePhaseStats table_after = stage_table_phase_stats();
    measurement.draw_delta = {
        draw_after.draws - measurement.draw_before.draws,
        draw_after.table_ms - measurement.draw_before.table_ms,
        draw_after.shader_ms - measurement.draw_before.shader_ms,
    };
    measurement.table_delta = {
        table_after.calls - measurement.table_before.calls,
        table_after.metadata_ms - measurement.table_before.metadata_ms,
        table_after.dynamic_fold_ms - measurement.table_before.dynamic_fold_ms,
        table_after.resources_ms - measurement.table_before.resources_ms,
    };
    if (worker_slot != 0) {
        measurement.readable_calls = g_guest_readable_cache.calls;
        measurement.readable_hits = g_guest_readable_cache.hits;
        measurement.readable_os_probes = g_guest_readable_cache.os_probes;
        measurement.readable_scope.reset();
    }
}

struct ParallelDrawStatsState {
    std::mutex mutex;
    ParallelDrawRealizationStats totals;
};

ParallelDrawStatsState& parallel_draw_stats_state() {
    static ParallelDrawStatsState state;
    return state;
}

} // namespace

ParallelDrawRealizationStats parallel_draw_realization_stats() {
    auto& state = parallel_draw_stats_state();
    std::lock_guard lock(state.mutex);
    return state.totals;
}

std::vector<DrawItem> realize_gpustate_draws_parallel(
        const GpuState& st, uint32_t max_shader_dwords, bool log,
        bool retain_shared_shader_words, bool* attempted) {
    if (attempted) *attempted = false;
    const size_t thread_count = configured_draw_realization_threads();
    if (thread_count < 2 || st.draws.size() < parallel_draw_minimum() ||
        parallel_draw_diagnostic_active(log))
        return {};
    if (attempted) *attempted = true;

    DrawRealizationPool& pool = draw_realization_pool();
    const size_t worker_count = std::min(pool.worker_count(), thread_count - 1);
    ParallelDrawContext context;
    context.state = &st;
    context.max_shader_dwords = max_shader_dwords;
    context.retain_shared_shader_words = retain_shared_shader_words;
    context.parent_readable_active = g_guest_readable_cache.active;
    context.descriptor_validate_mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");   // once, not per draw
    context.slots.resize(st.draws.size());
    context.measurements.resize(worker_count + 1);

    auto batch = std::make_shared<DrawRealizationBatch>();
    batch->context = &context;
    batch->begin = parallel_draw_worker_begin;
    batch->draw = parallel_draw_worker_execute;
    batch->end = parallel_draw_worker_end;
    batch->count = st.draws.size();
    const auto begin = std::chrono::steady_clock::now();
    pool.run(batch, worker_count);
    const auto end = std::chrono::steady_clock::now();

    // The parent thread's phase/readability counters feed the existing submit timing report. Worker
    // thread-local deltas must be merged explicitly so parallel work is not mislabeled as "other".
    for (size_t i = 1; i < context.measurements.size(); ++i) {
        const auto& measurement = context.measurements[i];
        g_draw_realization_phases.draws += measurement.draw_delta.draws;
        g_draw_realization_phases.table_ms += measurement.draw_delta.table_ms;
        g_draw_realization_phases.shader_ms += measurement.draw_delta.shader_ms;
        g_stage_table_phases.calls += measurement.table_delta.calls;
        g_stage_table_phases.metadata_ms += measurement.table_delta.metadata_ms;
        g_stage_table_phases.dynamic_fold_ms += measurement.table_delta.dynamic_fold_ms;
        g_stage_table_phases.resources_ms += measurement.table_delta.resources_ms;
        if (context.parent_readable_active) {
            g_guest_readable_cache.calls += measurement.readable_calls;
            g_guest_readable_cache.hits += measurement.readable_hits;
            g_guest_readable_cache.os_probes += measurement.readable_os_probes;
        }
    }

    {
        auto& stats = parallel_draw_stats_state();
        std::lock_guard lock(stats.mutex);
        ++stats.totals.batches;
        stats.totals.semantic_draws += st.draws.size();
        stats.totals.worker_threads += worker_count + 1;
        stats.totals.wall_ms +=
            std::chrono::duration<double, std::milli>(end - begin).count();
    }

    std::vector<DrawItem> items;
    items.reserve(st.draws.size());
    for (auto& slot : context.slots)
        if (slot.realized) items.push_back(std::move(slot.item));
    return items;
}

namespace {
enum class RetainedSubmitKind : uint8_t { Draw, Dispatch, DmaCopy, ParserStall, MemoryEffect };
struct RetainedSubmitOperation {
    RetainedSubmitKind kind;
    size_t index;
    uint64_t command_order;
};

bool use_per_draw_policy(const GpuState& st) {
    static const bool force_perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    static const bool force_folded = getenv("PROSPER_FOLDED") != nullptr;
    return force_perdraw || (!force_folded && (st.draws.size() > 1 || !st.dispatches.empty()));
}

bool retained_draw_selected(const GpuState& st, size_t index) {
    return use_per_draw_policy(st) || index + 1 == st.draws.size();
}

bool resolve_indirect_draw_arguments(const GpuState& submit, const GpuState::Draw& source,
                                     GpuState::Draw& resolved) {
    resolved = source;
    if (!source.indirect) return true;
    constexpr uint32_t kArgumentBytes = 5u * sizeof(uint32_t);
    if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
        !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr, "[agc] indexed indirect draw skipped: unreadable arguments at 0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr));
        return false;
    }
    uint32_t args[5] = {};
    std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
    const uint32_t index_count = args[0];
    const uint32_t instance_count = args[1];
    const uint32_t first_index = args[2];
    const int32_t vertex_offset = static_cast<int32_t>(args[3]);
    const uint32_t first_instance = args[4];
    constexpr uint32_t kMaxIndirectCount = 1u << 20;
    if (!index_count || !instance_count) return false;  // hardware no-op
    if (index_count > kMaxIndirectCount || instance_count > kMaxIndirectCount ||
        first_instance != 0 || !source.index_base) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indexed indirect draw skipped: count=%u instances=%u first=%u "
                         "vertex_offset=%d first_instance=%u index_base=0x%llx\n",
                         index_count, instance_count, first_index, vertex_offset, first_instance,
                         static_cast<unsigned long long>(source.index_base));
        return false;
    }
    const GpuState& draw_state = source.state ? *source.state : submit;
    const uint64_t element_bytes = index_elem_bytes(draw_state.index_type);
    if (!element_bytes) return false;
    if (first_index > (UINT64_MAX - source.index_base) / element_bytes) return false;
    resolved.index_count = index_count;
    resolved.instance_count = instance_count;
    resolved.indexed = true;
    resolved.index_offset = first_index;
    resolved.index_addr = source.index_base + static_cast<uint64_t>(first_index) * element_bytes;
    resolved.from_offset = true;
    resolved.indirect_vertex_offset = vertex_offset;
    resolved.has_vertex_offset_override = true;
    resolved.indirect = false;
    if (std::getenv("PROSPER_INDIRECTLOG")) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 256)
            std::fprintf(stderr,
                         "[agc-indirect] draw args=0x%llx count=%u instances=%u first=%u "
                         "vertex_offset=%d index_base=0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr), index_count,
                         instance_count, first_index, vertex_offset,
                         static_cast<unsigned long long>(source.index_base));
    }
    return true;
}

enum class DispatchArgumentResolution : uint8_t { Ready, Noop, Invalid };

// Indirect dispatch outcome census, reported once at teardown.
//
// Why a census and not more log lines: PROSPER_INDIRECTLOG is rate-limited to its first 64 packets,
// which on a GTA V route are all early-boot, and an early-boot sample of a GPU-driven pipeline says
// nothing about the steady state. Worse, the all-zero case below returns Noop SILENTLY -- so a route
// in which every indirect dispatch is dropped is externally indistinguishable from one in which none
// are, which is exactly the ambiguity that let this path go unexamined.
//
// The counts are what make the reading falsifiable: `ready` versus `zero_args` over a whole route
// answers "are the group counts ever actually produced", and a total that does not match the decoded
// indirect-dispatch count means the census itself is mis-placed rather than the subject misbehaving.
struct IndirectDispatchCensus {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> ready{0};
    std::atomic<uint64_t> zero_args{0};
    std::atomic<uint64_t> zero_groups{0};
    std::atomic<uint64_t> unreadable{0};
    std::atomic<uint64_t> direct_noop{0};
    ~IndirectDispatchCensus() {
        const uint64_t seen = total.load(std::memory_order_relaxed);
        if (!seen || !std::getenv("PROSPER_INDIRECTLOG")) return;
        std::fprintf(stderr,
                     "[agc-indirect-census] indirect dispatches=%llu ready=%llu zero-args=%llu "
                     "zero-groups=%llu unreadable=%llu (direct no-ops=%llu)\n",
                     (unsigned long long)seen, (unsigned long long)ready.load(),
                     (unsigned long long)zero_args.load(),
                     (unsigned long long)zero_groups.load(),
                     (unsigned long long)unreadable.load(),
                     (unsigned long long)direct_noop.load());
    }
};
IndirectDispatchCensus& indirect_dispatch_census() {
    static IndirectDispatchCensus census;
    return census;
}

DispatchArgumentResolution resolve_indirect_dispatch_arguments(
        const GpuState::Dispatch& source, GpuState::Dispatch& resolved) {
    resolved = source;
    if (!source.indirect) {
        const bool noop = direct_compute_dispatch_is_noop(source);
        if (noop) indirect_dispatch_census().direct_noop.fetch_add(1, std::memory_order_relaxed);
        return noop ? DispatchArgumentResolution::Noop : DispatchArgumentResolution::Ready;
    }
    auto& census = indirect_dispatch_census();
    census.total.fetch_add(1, std::memory_order_relaxed);
    constexpr uint32_t kArgumentBytes = 3u * sizeof(uint32_t);
    if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
        !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr, "[agc] indirect dispatch skipped: unreadable arguments at 0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr));
        census.unreadable.fetch_add(1, std::memory_order_relaxed);
        return DispatchArgumentResolution::Invalid;
    }
    uint32_t args[3] = {};
    std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
    const auto log_resolution = [&](const char* outcome, const ComputeLaunchDimensions* launch) {
        if (!std::getenv("PROSPER_INDIRECTLOG")) return;
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) >= 256) return;
        const uint64_t code_addr = source.state
            ? compute_dispatch_code_addr(*source.state, source) : 0;
        std::fprintf(stderr,
                     "[agc-indirect-resolve] outcome=%s args=0x%llx code=0x%llx "
                     "dims=%ux%ux%u groups=%ux%ux%u\n",
                     outcome,
                     static_cast<unsigned long long>(source.indirect_args_addr),
                     static_cast<unsigned long long>(code_addr),
                     args[0], args[1], args[2],
                     launch ? launch->groups_x : 0u,
                     launch ? launch->groups_y : 0u,
                     launch ? launch->groups_z : 0u);
    };
    if (!args[0] || !args[1] || !args[2]) {
        // Log at ordered realization, after every preceding compute fence and writeback. The
        // command-processor's similarly named packet trace runs while the stream is still being
        // folded and can therefore show the producer's previous-frame contents instead.
        log_resolution("zero-argument", nullptr);
        census.zero_args.fetch_add(1, std::memory_order_relaxed);
        return DispatchArgumentResolution::Noop;
    }
    resolved.threads_x = args[0];
    resolved.threads_y = args[1];
    resolved.threads_z = args[2];
    resolved.indirect = false;
    const ComputeLaunchDimensions launch = resolve_compute_launch(resolved);
    if (!launch.groups_x || !launch.groups_y || !launch.groups_z) {
        log_resolution("zero-groups", &launch);
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indirect dispatch skipped: dimensions=%ux%ux%u resolve to "
                         "%ux%ux%u workgroups\n",
                         args[0], args[1], args[2], launch.groups_x, launch.groups_y,
                         launch.groups_z);
        census.zero_groups.fetch_add(1, std::memory_order_relaxed);
        return DispatchArgumentResolution::Noop;
    }
    log_resolution("ready", &launch);
    census.ready.fetch_add(1, std::memory_order_relaxed);
    return DispatchArgumentResolution::Ready;
}

// `failure`, when supplied, is filled on every failure return from the draw path below.
// The two exits above realize_draw_item name themselves, because nothing was attempted there and a
// shader/pipeline diagnostic cannot exist; the third forwards whatever realize_draw_item determined
// (#1636). Callers that do not want a diagnostic keep passing nothing.
bool realize_retained_draw(const GpuState& st, size_t index, float scale_x, float scale_y,
                           DrawItem& item, OperationRealizationFailure* failure = nullptr) {
    if (dropped_draw_census_enabled())
        retained_draw_attempts().fetch_add(1, std::memory_order_relaxed);
    const auto note = [&](RealizationFailureReason reason) {
        // These two exits are the census's blind spot, and it is the blind spot that matters most
        // for a "written by nothing" surface: they drop an operation BEFORE any render state is
        // extracted, so the draw never reaches the sites that report a colour target and the
        // surface reads as one the guest never rendered to. Derive the target here -- only when the
        // census is on, because extract_render_state is not free -- so an indirect operation's
        // destination is nameable.
        if (index < st.draws.size() && dropped_draw_census_enabled()) {
            const RenderState rs = extract_render_state(
                use_per_draw_policy(st) ? st.state_at_draw(index) : st);
            report_dropped_draw_target(
                rs.color0_base,
                reason == RealizationFailureReason::RetainedDrawNotSelected ? "retained-not-selected"
                                                                            : "indirect-arguments",
                rs.cb_target_mask, rs.cb_shader_mask);
        }
        // An out-of-range index has no planned operation to attach to, and a record whose identity
        // matches nothing fails validate_failure_diagnostics for the WHOLE capture. Report nothing
        // rather than poison the capture; the caller still sees false.
        if (!failure || index >= st.draws.size()) return false;
        *failure = {};
        failure->kind = SubmitOperationKind::Draw;
        failure->index = index;
        failure->command_order = st.draws[index].command_order;
        failure->reason = reason;
        return false;
    };
    if (index >= st.draws.size() || !retained_draw_selected(st, index))
        return note(RealizationFailureReason::RetainedDrawNotSelected);
    const bool per_draw = use_per_draw_policy(st);
    const GpuState& draw_state = per_draw ? st.state_at_draw(index) : st;
    GpuState::Draw draw;
    if (!resolve_indirect_draw_arguments(st, st.draws[index], draw))
        return note(RealizationFailureReason::IndirectArguments);
    const bool log = getenv("PROSPER_GFXLOG") != nullptr || getenv("PROSPER_EXECLOG") != nullptr;
    if (!realize_draw_item(draw_state, &draw, draw.index_count, 0x10000, log, item,
                           failure, true)) {
        // realize_draw_item resets and fills the record, including pipeline/targets/extent, but has
        // no notion of which retained operation it belongs to.
        if (failure) {
            failure->kind = SubmitOperationKind::Draw;
            failure->index = index;
            failure->command_order = draw.command_order;
            if (failure->reason == RealizationFailureReason::None)
                failure->reason = RealizationFailureReason::Unknown;
        }
        return false;
    }
    item.draw_index = index;
    item.command_order = draw.command_order;
    if (scale_x != 1.0f || scale_y != 1.0f)
        scale_resolved_render_area(item.ps, scale_x, scale_y);
    return true;
}

enum class RetainedComputeRealization : uint8_t { Realized, Failed };

RetainedComputeRealization realize_retained_compute(
        const GpuState& st, size_t index, const GpuState::Dispatch& resolved_dispatch,
        uint64_t submit_no, ComputeItem& item,
        OperationRealizationFailure* failure = nullptr) {
    if (index >= st.dispatches.size()) return RetainedComputeRealization::Failed;
    // DMA-bearing submits are uncommon. A one-dispatch state keeps the mature realization path
    // intact while ensuring it runs only after every preceding ordered producer has landed.
    GpuState one = st.dispatches[index].state ? *st.dispatches[index].state : st;
    one.dispatches.clear();
    one.dispatches.push_back(resolved_dispatch);
    std::vector<OperationRealizationFailure> failures;
    std::vector<ComputeItem> realized = realize_compute_dispatches(
        one, submit_no, failure ? &failures : nullptr);
    if (realized.empty()) {
        if (failure && !failures.empty()) {
            *failure = std::move(failures.front());
            failure->index = index;
            failure->command_order = st.dispatches[index].command_order;
        }
        return RetainedComputeRealization::Failed;
    }
    item = std::move(realized.front());
    item.dispatch_index = index;
    return RetainedComputeRealization::Realized;
}

// ---------------------------------------------------------------------------------------------
// PROSPER_COMPUTE_TREE_WATCH=ADDR:RECORDS[:SHIFT:MASK]
//
// Address-keyed, evaluated around EVERY realized dispatch. The program-keyed parent walk above
// answers "was the table cyclic when its consumer started"; this answers "which dispatch made it
// cyclic", which is a different question and the one the consumer's hang depends on.
//
// Two comparisons per dispatch, and they are deliberately not the same comparison:
//   pre  -- current bytes against the LAST observation, whenever that was. A change here was made
//           by something that is not a dispatch (guest CPU, a copy, a graphics write).
//   post -- current bytes against this dispatch's own pre-image. A change here is attributable to
//           this dispatch, by construction.
// Reporting them separately is what keeps "some dispatch between these two wrote it" from being
// recorded as "this dispatch wrote it".
//
// The watch runs on every dispatch, not only on the programs already known to bind a range
// containing the address, and reports `toucher=0/1` for each change. A change from a non-toucher
// is a finding rather than noise: it means an out-of-bounds write, or a binding path the resource
// traversal does not model. An instrument restricted to the suspects it already has cannot
// discover that it is looking at the wrong set.
struct LiveComputeTreeWatchState {
    bool have_previous = false;
    std::vector<uint32_t> previous;
    ComputeParentWalkReport previous_report;
    uint64_t previous_submit = 0;
    uint64_t previous_order = 0;
    uint64_t previous_program = 0;
    uint64_t observations = 0;
    uint64_t reported_changes = 0;
};

const std::optional<ComputeTreeWatchSelector>& compute_tree_watch_selector() {
    static const auto selector = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_TREE_WATCH");
        auto parsed = parse_compute_tree_watch_selector(value);
        if (value && *value && !parsed)
            std::fprintf(stderr,
                         "[compute-tree-watch] invalid selector '%s'; expected "
                         "ADDR:RECORDS[:SHIFT:MASK]\n",
                         value);
        return parsed;
    }();
    return selector;
}

LiveComputeTreeWatchState& compute_tree_watch_state() {
    static LiveComputeTreeWatchState state;
    return state;
}

// The guest allocation is the authority for what a later CPU-side read of this table sees. Reading
// the running program's staged resource copy instead would make a pre/post pair compare two
// different sources, which manufactures changes that nobody wrote.
bool read_compute_tree_watch_words(const ComputeTreeWatchSelector& selector,
                                   std::vector<uint32_t>& out) {
    const uint64_t bytes = static_cast<uint64_t>(selector.records) * sizeof(uint32_t);
    if (!guest_readable(selector.addr, bytes)) return false;
    out.resize(selector.records);
    std::memcpy(out.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(selector.addr)),
                static_cast<size_t>(bytes));
    return true;
}

// Name the binding and fetch pc when the running program's own table reaches the watched address.
// A change reported with toucher=0 is the interesting case -- an out-of-bounds write, or a binding
// path the traversal does not model -- so it must stay distinguishable from a change made by a
// program that legitimately binds the range. Computed before the dispatch, because the ComputeItem
// carrying the resource table is moved into the backend.
std::string describe_compute_tree_watch_touch(const ShaderResourceTable* resources,
                                              uint64_t wanted, uint64_t bytes) {
    if (!resources) return "toucher=0";
    // WINDOW, not address. The watch detects changes anywhere in the window, so asking whether a
    // program binds only the window's first byte makes the two halves of the instrument answer
    // different questions -- which reported a program that binds window+128 as a non-toucher, i.e.
    // as an out-of-bounds write that was not one.
    const std::vector<ComputeAddressWatchHit> hits =
        compute_address_window_hits(*resources, wanted, bytes);
    if (hits.empty()) return "toucher=0";
    std::string touch = "toucher=1 at=";
    for (size_t i = 0; i < hits.size(); ++i) {
        char one[64];
        std::snprintf(one, sizeof(one), "%s%u:%u", i ? "," : "", hits[i].binding, hits[i].fetch_pc);
        touch += one;
    }
    return touch;
}

void report_compute_tree_watch(const char* phase, const ComputeTreeWatchSelector& selector,
                               uint64_t program, const std::string& touch, uint64_t submit_no,
                               size_t dispatch_index,
                               uint64_t order, const std::vector<uint32_t>& before,
                               const ComputeParentWalkReport& before_report,
                               const std::vector<uint32_t>& after, uint64_t previous_submit,
                               uint64_t previous_order, uint64_t previous_program,
                               ComputeParentWalkReport& after_report) {
    constexpr uint32_t kReportedSlots = 12;
    uint32_t changed = 0;
    const std::vector<ComputeTreeWatchDelta> deltas =
        compute_tree_watch_deltas(before, after, kReportedSlots, &changed);
    if (!changed) return;

    LiveComputeTreeWatchState& state = compute_tree_watch_state();
    ++state.reported_changes;

    after_report = analyze_compute_parent_walk(after, selector.records, selector.index_shift,
                                               selector.index_mask);
    const ComputeTreeSiblingReport before_siblings = analyze_compute_tree_siblings(before);
    const ComputeTreeSiblingReport after_siblings = analyze_compute_tree_siblings(after);
    const ComputeTreeWatchTransition transition =
        classify_compute_tree_watch_transition(before_report, after_report, true);

    std::fprintf(stderr,
                 "[compute-tree-watch] addr=0x%llx phase=%s transition=%s program=0x%llx "
                 "submit=%llu dispatch=%zu order=%llu %s changed=%u records=%u "
                 "pre{cycles=%u cyclic-roots=%u oob-roots=%u max-depth=%u pairs=%u unpaired=%u} "
                 "post{cycles=%u cyclic-roots=%u oob-roots=%u max-depth=%u pairs=%u unpaired=%u} "
                 "since{submit=%llu order=%llu program=0x%llx}\n",
                 static_cast<unsigned long long>(selector.addr), phase,
                 compute_tree_watch_transition_name(transition),
                 static_cast<unsigned long long>(program),
                 static_cast<unsigned long long>(submit_no), dispatch_index,
                 static_cast<unsigned long long>(order), touch.c_str(), changed, selector.records,
                 before_report.distinct_cycles, before_report.cyclic_roots,
                 before_report.oob_roots, before_report.max_depth, before_siblings.pairs,
                 before_siblings.unpaired_side, after_report.distinct_cycles,
                 after_report.cyclic_roots, after_report.oob_roots, after_report.max_depth,
                 after_siblings.pairs, after_siblings.unpaired_side,
                 static_cast<unsigned long long>(previous_submit),
                 static_cast<unsigned long long>(previous_order),
                 static_cast<unsigned long long>(previous_program));

    for (const ComputeTreeWatchDelta& delta : deltas)
        std::fprintf(stderr, "[compute-tree-watch]   slot %u 0x%08x -> 0x%08x\n", delta.index,
                     delta.before, delta.after);
    if (changed > kReportedSlots)
        std::fprintf(stderr, "[compute-tree-watch]   ... %u further changed slots not listed\n",
                     changed - kReportedSlots);

    // PROSPER_COMPUTE_TREE_WATCH_DUMP=<dir>: write both images of a transition that INTRODUCES
    // cycles. Twelve reported slots identify that something changed; they cannot answer what shape
    // the damage has -- whether the wrong records cluster by workgroup, by wave, at the tail of the
    // extent, or not at all. That question decides between a lane-activation defect and a
    // value-computation defect, and it needs the whole array on disk.
    //
    // Gated on the clean -> cyclic transition specifically. Dumping every change would write
    // hundreds of megabytes across a routed run and bury the two images that matter.
    // CleanToClean is dumped too when an aux range is configured: a cyclic sample with no clean
    // counterpart from the same run cannot show what CHANGED, only what is present.
    const bool want_clean_control = std::getenv("PROSPER_COMPUTE_TREE_WATCH_AUX") != nullptr &&
                                    transition == ComputeTreeWatchTransition::CleanToClean;
    if (transition != ComputeTreeWatchTransition::CleanToCyclic && !want_clean_control) return;
    const char* dump_root = std::getenv("PROSPER_COMPUTE_TREE_WATCH_DUMP");
    if (!dump_root || !*dump_root) return;
    auto write_image = [&](const char* which, const std::vector<uint32_t>& words) {
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/tree-%s-s%llu-d%zu-o%llu-p%llx.bin", dump_root, which,
                      static_cast<unsigned long long>(submit_no), dispatch_index,
                      static_cast<unsigned long long>(order),
                      static_cast<unsigned long long>(program));
        std::FILE* file = std::fopen(path, "wb");
        if (!file) {
            std::fprintf(stderr, "[compute-tree-watch] dump open failed: %s\n", path);
            return;
        }
        const size_t written = std::fwrite(words.data(), sizeof(uint32_t), words.size(), file);
        const int closed = std::fclose(file);
        if (written != words.size() || closed != 0)
            std::fprintf(stderr, "[compute-tree-watch] dump write failed: %s\n", path);
        else
            std::fprintf(stderr, "[compute-tree-watch]   dumped %s -> %s\n", which, path);
    };
    write_image("pre", before);
    write_image("post", after);

    // PROSPER_COMPUTE_TREE_WATCH_AUX=0xADDR:DWORDS — dump a SECOND guest range at the same moment.
    //
    // The watched table is the OUTPUT. Its input lives somewhere else, and the question that matters
    // is what differs about that input between a dispatch that produces a correct tree and one that
    // does not. GTA V's builder emits pairs=1030 unpaired=0 for eleven consecutive submits at an
    // identical launch geometry and then never again, which rules its lowering correct and makes the
    // input the only remaining variable -- but nothing was capturing the input.
    const char* aux = std::getenv("PROSPER_COMPUTE_TREE_WATCH_AUX");
    if (!aux || !*aux) return;
    char* aux_end = nullptr;
    const uint64_t aux_addr = std::strtoull(aux, &aux_end, 0);
    if (!aux_addr || !aux_end || *aux_end != ':') return;
    const uint64_t aux_dwords = std::strtoull(aux_end + 1, &aux_end, 0);
    if (!aux_dwords || aux_dwords > (1u << 22u) || (aux_end && *aux_end)) return;
    const uint64_t aux_bytes = aux_dwords * sizeof(uint32_t);
    if (!guest_readable(aux_addr, aux_bytes)) {
        std::fprintf(stderr, "[compute-tree-watch]   aux 0x%llx unreadable\n",
                     static_cast<unsigned long long>(aux_addr));
        return;
    }
    std::vector<uint32_t> aux_words(static_cast<size_t>(aux_dwords));
    std::memcpy(aux_words.data(),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(aux_addr)),
                static_cast<size_t>(aux_bytes));
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/aux-%llx-s%llu-d%zu-o%llu.bin", dump_root,
                  static_cast<unsigned long long>(aux_addr),
                  static_cast<unsigned long long>(submit_no), dispatch_index,
                  static_cast<unsigned long long>(order));
    std::FILE* file = std::fopen(path, "wb");
    if (!file) return;
    const size_t written = std::fwrite(aux_words.data(), sizeof(uint32_t), aux_words.size(), file);
    const int closed = std::fclose(file);
    std::fprintf(stderr, "[compute-tree-watch]   dumped aux -> %s (%s)\n", path,
                 (written == aux_words.size() && closed == 0) ? "ok" : "SHORT");
}

// Returns the pre-image so the caller can hand it back for the post comparison. Empty means the
// window was unreadable and no post comparison should be attempted -- comparing against an empty
// pre-image would report every slot as changed.
std::vector<uint32_t> observe_compute_tree_watch_pre(const ComputeItem& item, uint64_t submit_no,
                                                     size_t dispatch_index, uint64_t order,
                                                     const std::string& touch) {
    const auto& selected = compute_tree_watch_selector();
    if (!selected) return {};
    std::vector<uint32_t> current;
    if (!read_compute_tree_watch_words(*selected, current)) {
        // An unreadable watch window produced NO output at all -- indistinguishable from "nothing
        // writes this address", which is the conclusion it would have supported. Report it once.
        // Measured need: watching GTA V's 0x2052ac0000 returned complete silence, and the silence
        // meant the address is not guest-readable, not that no program writes it.
        static std::mutex mutex;
        static std::set<uint64_t> reported;
        bool first = false;
        {
            std::lock_guard lock(mutex);
            first = reported.insert(selected->addr).second;
        }
        if (first)
            std::fprintf(stderr,
                         "[compute-tree-watch] addr=0x%llx UNREADABLE (%u dwords) — this watch can "
                         "report nothing about it; its silence is not evidence of no writer\n",
                         static_cast<unsigned long long>(selected->addr), selected->records);
        return {};
    }

    LiveComputeTreeWatchState& state = compute_tree_watch_state();
    ++state.observations;
    ComputeParentWalkReport current_report;
    if (!state.have_previous) {
        current_report = analyze_compute_parent_walk(current, selected->records,
                                                     selected->index_shift, selected->index_mask);
        const ComputeTreeSiblingReport siblings = analyze_compute_tree_siblings(current);
        std::fprintf(stderr,
                     "[compute-tree-watch] addr=0x%llx phase=first submit=%llu dispatch=%zu "
                     "order=%llu records=%u cycles=%u cyclic-roots=%u oob-roots=%u max-depth=%u "
                     "pairs=%u unpaired=%u\n",
                     static_cast<unsigned long long>(selected->addr),
                     static_cast<unsigned long long>(submit_no), dispatch_index,
                     static_cast<unsigned long long>(order), selected->records,
                     current_report.distinct_cycles, current_report.cyclic_roots,
                     current_report.oob_roots, current_report.max_depth, siblings.pairs,
                     siblings.unpaired_side);
    } else {
        current_report = state.previous_report;
        report_compute_tree_watch("pre", *selected, item.code_addr, touch, submit_no,
                                  dispatch_index, order,
                                  state.previous, state.previous_report, current,
                                  state.previous_submit, state.previous_order,
                                  state.previous_program, current_report);
    }
    state.have_previous = true;
    state.previous = current;
    state.previous_report = current_report;
    state.previous_submit = submit_no;
    state.previous_order = order;
    state.previous_program = item.code_addr;
    return current;
}

void observe_compute_tree_watch_post(uint64_t program, const std::string& touch,
                                     uint64_t submit_no, size_t dispatch_index, uint64_t order,
                                     const std::vector<uint32_t>& pre_image) {
    const auto& selected = compute_tree_watch_selector();
    if (!selected || pre_image.empty()) return;
    std::vector<uint32_t> current;
    if (!read_compute_tree_watch_words(*selected, current)) return;

    LiveComputeTreeWatchState& state = compute_tree_watch_state();
    ComputeParentWalkReport after_report = state.previous_report;
    report_compute_tree_watch("post", *selected, program, touch, submit_no, dispatch_index, order,
                              pre_image, state.previous_report, current, submit_no, order, program,
                              after_report);
    state.previous = current;
    state.previous_report = after_report;
    state.previous_submit = submit_no;
    state.previous_order = order;
    state.previous_program = program;
}

struct LiveComputeParentWalkPreflight {
    bool selected = false;
    bool available = false;
    bool suspicious = false;
    uint64_t resource_addr = 0;
    uint32_t resource_bytes = 0;
    uint64_t resource_hash = 0;
    ComputeParentWalkReport report;
};

const std::optional<ComputeParentWalkSelector>& compute_parent_walk_selector() {
    static const auto selector = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_PARENT_WALK");
        auto parsed = parse_compute_parent_walk_selector(value);
        if (value && *value && !parsed)
            std::fprintf(stderr,
                         "[compute-parent-walk] invalid selector '%s'; expected "
                         "PROGRAM:FETCH_PC:SHIFT:MASK[:DEEP]\n",
                         value);
        return parsed;
    }();
    return selector;
}

void dump_suspicious_parent_walk(const ComputeItem& item, uint64_t hash,
                                 const std::vector<uint32_t>& words) {
    const char* root = std::getenv("PROSPER_COMPUTE_PARENT_WALK_DUMP");
    if (!root || !*root) return;
    std::error_code error;
    std::filesystem::create_directories(root, error);
    if (error) {
        std::fprintf(stderr, "[compute-parent-walk] dump directory failed: %s\n",
                     error.message().c_str());
        return;
    }
    char leaf[192] = {};
    std::snprintf(leaf, sizeof(leaf),
                  "parent-walk-s%llu-d%llu-o%llu-%016llx.bin",
                  static_cast<unsigned long long>(item.submit_no),
                  static_cast<unsigned long long>(item.dispatch_index),
                  static_cast<unsigned long long>(item.command_order),
                  static_cast<unsigned long long>(hash));
    const std::filesystem::path path = std::filesystem::path(root) / leaf;
    FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "[compute-parent-walk] dump open failed: %s\n",
                     path.string().c_str());
        return;
    }
    const size_t written = std::fwrite(
        words.data(), sizeof(uint32_t), words.size(), file);
    const int close_result = std::fclose(file);
    if (written != words.size() || close_result != 0) {
        std::fprintf(stderr, "[compute-parent-walk] dump write failed: %s\n",
                     path.string().c_str());
        return;
    }
    std::fprintf(stderr, "[compute-parent-walk] dumped %zu bytes to %s\n",
                 words.size() * sizeof(uint32_t), path.string().c_str());
}

LiveComputeParentWalkPreflight preflight_compute_parent_walk(
        const ComputeItem& item, uint64_t previous_code,
        bool previous_realized, bool previous_executed) {
    LiveComputeParentWalkPreflight result;
    const auto& selected = compute_parent_walk_selector();
    if (!selected || item.code_addr != selected->program_addr) return result;
    result.selected = true;

    const ShaderResource* resource = nullptr;
    if (item.resources) {
        for (const ShaderResource& candidate : item.resources->resources) {
            if (candidate.fetch_pc != selected->fetch_pc) continue;
            if (resource) {
                std::fprintf(stderr,
                             "[compute-parent-walk] code=0x%llx fetch-pc=%u "
                             "unavailable=ambiguous-resource\n",
                             static_cast<unsigned long long>(item.code_addr),
                             selected->fetch_pc);
                return result;
            }
            resource = &candidate;
        }
    }
    constexpr uint32_t kMaximumDiagnosticRecords = 1u << 20u;
    if (!resource || !resource->size || (resource->size % sizeof(uint32_t)) != 0u ||
        resource->stride != sizeof(uint32_t) ||
        resource->size / sizeof(uint32_t) > kMaximumDiagnosticRecords) {
        std::fprintf(stderr,
                     "[compute-parent-walk] code=0x%llx fetch-pc=%u "
                     "unavailable=resource-shape addr=0x%llx bytes=%u stride=%u\n",
                     static_cast<unsigned long long>(item.code_addr), selected->fetch_pc,
                     static_cast<unsigned long long>(resource ? resource->gpu_addr : 0u),
                     resource ? resource->size : 0u, resource ? resource->stride : 0u);
        return result;
    }

    const uint8_t* source = nullptr;
    if (resource->host_data && resource->host_data_size >= resource->size) {
        source = resource->host_data;
    } else if (!resource->host_data && resource->host_data_size == 0u &&
               guest_readable(resource->gpu_addr, resource->size)) {
        source = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(resource->gpu_addr));
    }
    if (!source) {
        std::fprintf(stderr,
                     "[compute-parent-walk] code=0x%llx fetch-pc=%u "
                     "unavailable=unreadable addr=0x%llx bytes=%u\n",
                     static_cast<unsigned long long>(item.code_addr), selected->fetch_pc,
                     static_cast<unsigned long long>(resource->gpu_addr), resource->size);
        return result;
    }

    std::vector<uint32_t> words(resource->size / sizeof(uint32_t));
    std::memcpy(words.data(), source, resource->size);
    const uint64_t launched = std::min<uint64_t>(
        static_cast<uint64_t>(item.launch.threads_x) * item.launch.threads_y *
            item.launch.threads_z,
        UINT32_MAX);
    result.available = true;
    result.resource_addr = resource->gpu_addr;
    result.resource_bytes = resource->size;
    result.resource_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(words.data()), resource->size);
    result.report = analyze_compute_parent_walk(
        words, static_cast<uint32_t>(launched), selected->index_shift,
        selected->index_mask);
    result.suspicious = compute_parent_walk_suspicious(
        result.report, selected->deep_threshold);
    std::fprintf(stderr,
                 "[compute-parent-walk] submit=%llu dispatch=%llu order=%llu "
                 "code=0x%llx fetch-pc=%u addr=0x%llx bytes=%u hash=%016llx "
                 "groups=%ux%ux%u local=%ux%ux%u records=%u roots=%u "
                 "cycles=%u cyclic-roots=%u oob-roots=%u max-depth=%u "
                 "max-root=%u deep-threshold=%u suspicious=%u "
                 "previous-code=0x%llx previous-realized=%u previous-executed=%u\n",
                 static_cast<unsigned long long>(item.submit_no),
                 static_cast<unsigned long long>(item.dispatch_index),
                 static_cast<unsigned long long>(item.command_order),
                 static_cast<unsigned long long>(item.code_addr), selected->fetch_pc,
                 static_cast<unsigned long long>(result.resource_addr),
                 result.resource_bytes,
                 static_cast<unsigned long long>(result.resource_hash),
                 item.launch.groups_x, item.launch.groups_y, item.launch.groups_z,
                 item.launch.local_x, item.launch.local_y, item.launch.local_z,
                 result.report.records, result.report.roots,
                 result.report.distinct_cycles, result.report.cyclic_roots,
                 result.report.oob_roots, result.report.max_depth,
                 result.report.max_depth_root, selected->deep_threshold,
                 static_cast<unsigned>(result.suspicious),
                 static_cast<unsigned long long>(previous_code),
                 static_cast<unsigned>(previous_realized),
                 static_cast<unsigned>(previous_executed));
    if (result.suspicious) dump_suspicious_parent_walk(item, result.resource_hash, words);
    return result;
}
} // namespace

std::vector<ComputeAuthorityBoundary> compute_authority_draw_resource_boundaries(
        const DrawItem& item, uint64_t submit_no) {
    std::vector<ComputeAuthorityBoundary> boundaries;
    const size_t vertex_resources = item.vrt ? item.vrt->resources.size() : 0;
    const size_t fragment_resources = item.prt ? item.prt->resources.size() : 0;
    boundaries.reserve((vertex_resources + fragment_resources) * 2 + 1);
    const auto append_table = [&](const ShaderResourceTable* table) {
        if (!table) return;
        for (const ShaderResource& resource : table->resources) {
            // Replay-owned bytes do not read the live guest address. Ordinary live resources have
            // no host_data and are reported with the same conservative footprint capture uses.
            if (resource.host_data || !resource.gpu_addr) continue;
            const uint64_t bytes = gpu_capture_resource_footprint(resource);
            const bool known = bytes != 0 &&
                resource.gpu_addr <= UINT64_MAX - (bytes - 1);
            boundaries.push_back({
                ComputeAuthorityBoundaryKind::DrawResource,
                submit_no,
                item.command_order,
                resource.gpu_addr,
                bytes,
                known,
                resource.binding,
                static_cast<uint32_t>(resource.cls),
                false,
            });
            const uint64_t metadata_bytes =
                gpu_capture_dcc_metadata_footprint(resource);
            if (resource.dcc_metadata_host_data || !resource.metadata_addr ||
                !metadata_bytes)
                continue;
            const bool metadata_known =
                resource.metadata_addr <= UINT64_MAX - (metadata_bytes - 1);
            boundaries.push_back({
                ComputeAuthorityBoundaryKind::DrawResource,
                submit_no,
                item.command_order,
                resource.metadata_addr,
                metadata_bytes,
                metadata_known,
                resource.binding,
                static_cast<uint32_t>(resource.cls),
                false,
            });
        }
    };
    append_table(item.vrt.get());
    append_table(item.prt.get());
    boundaries.push_back({
        ComputeAuthorityBoundaryKind::DrawResourceEnd,
        submit_no,
        item.command_order,
        0,
        0,
        false,
        UINT32_MAX,
        UINT32_MAX,
        true,
    });
    return boundaries;
}

static OrderedSubmitResult execute_ordered_gpustate(const GpuState& st, uint32_t width,
                                                     uint32_t height, uint64_t submit_no,
                                                     const LiveRenderFn& render,
                                                     const LiveComputeFn& compute,
                                                     OrderedGpustateCaptureTrace* capture_trace,
                                                     const std::vector<DrawItem>* eager_draws) {
    GuestGpuWriteSubmitScope guest_gpu_write_scope;
    if (st.dma_execution_rejected) {
        for (const GpuState::Dispatch& dispatch : st.dispatches)
            notify_compute_authority_unknown(
                ComputeAuthorityBoundaryKind::Compute,
                submit_no, dispatch.command_order);
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] ordered DMA submit not executed: unsupported eager/deferred "
                         "guest-memory dependency\n");
        return {};
    }
    std::vector<RetainedSubmitOperation> executable;
    executable.reserve(st.draws.size() + st.dispatches.size() + st.dma_copies.size() +
                       st.parser_stalls.size() + st.ordered_memory_effects.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        if (retained_draw_selected(st, i))
            executable.push_back({RetainedSubmitKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        if (!direct_compute_dispatch_is_noop(st.dispatches[i]))
            executable.push_back({RetainedSubmitKind::Dispatch, i,
                                  st.dispatches[i].command_order});
    for (size_t i = 0; i < st.dma_copies.size(); ++i)
        executable.push_back({RetainedSubmitKind::DmaCopy, i, st.dma_copies[i].command_order});
    for (size_t i = 0; i < st.parser_stalls.size(); ++i)
        executable.push_back({RetainedSubmitKind::ParserStall, i,
                              st.parser_stalls[i].command_order});
    for (size_t i = 0; i < st.ordered_memory_effects.size(); ++i)
        executable.push_back({RetainedSubmitKind::MemoryEffect, i,
                              st.ordered_memory_effects[i].cmd.stream_order});
    std::stable_sort(executable.begin(), executable.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });

    std::unordered_map<size_t, size_t> eager_draw_by_index;
    if (eager_draws)
        for (size_t i = 0; i < eager_draws->size(); ++i)
            eager_draw_by_index[static_cast<size_t>((*eager_draws)[i].draw_index)] = i;

    size_t total_spans = 0;
    bool in_draw_span = false;
    for (const auto& operation : executable) {
        if (render && operation.kind == RetainedSubmitKind::Draw) {
            if (!in_draw_span) ++total_spans;
            in_draw_span = true;
        } else {
            in_draw_span = false;
        }
    }

    uint32_t full_width = present_width(), full_height = present_height();
    const float scale_x = full_width ? static_cast<float>(width) / full_width : 1.0f;
    const float scale_y = full_height ? static_cast<float>(height) / full_height : 1.0f;
    OrderedSubmitResult result;
    bool producer_epoch_ok = true;
    bool indirect_dependencies_ok = true;
    bool final_callback_sent = false;
    uint64_t previous_compute_code = 0;
    bool previous_compute_realized = false;
    bool previous_compute_executed = false;
    std::vector<DrawItem> span;
    auto flush_span = [&](bool authoritative_readback = false) {
        if (span.empty() || !render) return;
        LiveRenderPhase saved = g_live_phase;
        const bool final_span = result.render_spans + 1 == total_spans;
        g_live_phase = {result.render_spans == 0, final_span, authoritative_readback};
        RenderedFrame rendered = render(span, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
        span.clear();
        ++result.render_spans;
        final_callback_sent |= final_span;
    };

    for (const auto& operation : executable) {
        switch (operation.kind) {
            case RetainedSubmitKind::Draw: {
                // PROSPER_DRAW_CENSUS=1 — the most basic number about a missing world, and nothing
                // reported it: how many draws does prosper actually execute, and how many of them
                // are indirect? An absent world with ten draws per frame and one with ten thousand
                // are different problems, and every investigation of GTA V's missing world so far
                // has proceeded without knowing which it is.
                //
                // Counted before the `render` early-out so a run without the live renderer still
                // reports what the command stream contained. Printed at powers of two.
                // PROSPER_TARGET_WATCH=0x…,0x… — EXACT and unsampled: is this address EVER a
                // colour target, in any of the eight MRT slots?
                //
                // Separate from the census below because the census samples 1 in 32 draws, and a
                // sampled zero cannot answer an "ever" question. A surface written by ten draws is
                // missed with probability (31/32)^10 ≈ 73%; by one draw, 97%. The census's zero for
                // GTA V's 0x2052ac0000 therefore only excludes a *frequent* writer, which is not
                // what was asked of it.
                //
                // Cost is eight register lookups per draw with no lock (per-address atomics), which
                // is what the 1-in-32 sampling was protecting against when it applied to a mutex and
                // a map insert. Watch list is bounded so the per-draw cost stays fixed.
                {
                    struct TargetWatch {
                        uint64_t addr;
                        std::atomic<uint64_t> hits[8];
                        std::atomic<uint64_t> total;
                    };
                    static constexpr size_t kMaxWatch = 8;
                    static std::array<TargetWatch, kMaxWatch> watch{};
                    static size_t watch_count = 0;
                    static std::atomic<uint64_t> watch_draws{0};
                    // Parsed by the shared STRICT parser, which is the whole reason it exists. This
                    // site was the original occurrence of the base-0 trap: `strtoull(..., 0)` accepts a
                    // bare decimal, so `PROSPER_TARGET_WATCH=2063380000` armed a watch on
                    // 2,063,380,000, printed "watching 1 address(es)", and produced a confident zero
                    // for an address nobody asked about. A watch whose null is meaningless is worse
                    // than no watch, and this one's nulls are quoted as evidence.
                    //
                    // A malformed list arms NOTHING. An oversized list is REJECTED rather than
                    // truncated: silently dropping the ninth address would answer "never a target" for
                    // a surface that was never examined.
                    static const bool watch_enabled = [] {
                        const char* spec = std::getenv("PROSPER_TARGET_WATCH");
                        if (!spec || !*spec) return false;
                        std::vector<uint64_t> addrs;
                        if (!prosper::gpu::parse_hex_watch_list(spec, addrs)) {
                            std::fprintf(stderr,
                                         "[target-watch] ignoring malformed PROSPER_TARGET_WATCH=\"%s\" "
                                         "(expected 0x-prefixed hex addresses, comma separated) "
                                         "-- NOT armed\n", spec);
                            return false;
                        }
                        if (addrs.size() > kMaxWatch) {
                            std::fprintf(stderr,
                                         "[target-watch] PROSPER_TARGET_WATCH lists %zu addresses, "
                                         "more than the %zu-address bound -- NOT armed (truncating "
                                         "would report 'never a target' for addresses never examined)\n",
                                         addrs.size(), kMaxWatch);
                            return false;
                        }
                        for (const uint64_t v : addrs) watch[watch_count++].addr = v;
                        std::fprintf(stderr, "[target-watch] watching %zu address(es)\n",
                                     watch_count);
                        return watch_count != 0;
                    }();
                    if (watch_enabled) {
                        namespace P = prosper::agc::Pm4;
                        constexpr uint32_t kColorRegisterStride = 0xf;
                        const uint64_t wn = watch_draws.fetch_add(1) + 1;
                        if (const GpuState* snap = st.draws[operation.index].state.get()) {
                            for (uint32_t slot = 0; slot < 8u; ++slot) {
                                const auto lo = snap->cx.find(
                                    P::CB_COLOR0_BASE + slot * kColorRegisterStride);
                                if (lo == snap->cx.end() || !lo->second) continue;
                                const auto hi = snap->cx.find(P::CB_COLOR0_BASE_EXT + slot);
                                uint64_t base = static_cast<uint64_t>(lo->second) << 8;
                                if (hi != snap->cx.end())
                                    base |= static_cast<uint64_t>(hi->second & 0xffu) << 40;
                                for (size_t w = 0; w < watch_count; ++w) {
                                    if (watch[w].addr != base) continue;
                                    watch[w].hits[slot].fetch_add(1, std::memory_order_relaxed);
                                    watch[w].total.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                        if ((wn & (wn - 1)) == 0 && wn >= 4096u) {
                            for (size_t w = 0; w < watch_count; ++w) {
                                std::string slots;
                                for (uint32_t s = 0; s < 8u; ++s) {
                                    const uint64_t h = watch[w].hits[s].load();
                                    if (!h) continue;
                                    slots += " slot" + std::to_string(s) + "=" + std::to_string(h);
                                }
                                std::fprintf(stderr,
                                             "[target-watch] addr=0x%llx draws=%llu of %llu%s\n",
                                             (unsigned long long)watch[w].addr,
                                             (unsigned long long)watch[w].total.load(),
                                             (unsigned long long)wn,
                                             slots.empty() ? "  NEVER A COLOUR TARGET" :
                                                 slots.c_str());
                            }
                        }
                    }
                }
                if (std::getenv("PROSPER_DRAW_CENSUS")) {
                    static std::atomic<uint64_t> seen{0}, indirect_seen{0};
                    const uint64_t n = seen.fetch_add(1) + 1;
                    if (st.draws[operation.index].indirect)
                        indirect_seen.fetch_add(1, std::memory_order_relaxed);
                    // WHERE the draws go, not just how many. 131,072 draws execute while the world
                    // is absent, so the question is which surfaces they write and whether the one
                    // that reaches the screen is among them. A count alone cannot answer that.
                    //
                    // Keyed on the colour-0 base and its extent, reported at teardown so the
                    // per-draw path stays a map insert.
                    // SAMPLED, 1 in 32. The first version took a mutex and inserted into a map on
                    // every one of 131,072 draws and reported twelve lines at each power of two; the
                    // routed run stalled at 1,024 draws and never reached gameplay. An instrument
                    // that changes the subject's behaviour measures the instrument.
                    static std::mutex target_mutex;
                    static std::map<std::tuple<uint64_t, uint32_t, uint32_t>, uint64_t> targets;
                    if ((n & 31u) == 0u) {
                        // Straight from the draw's own register snapshot, the same two registers
                        // render_state.cpp uses (CB_COLOR0_BASE + its _EXT high bits). Avoids
                        // building a RenderState per draw on a path that runs 131,072 times.
                        // ALL EIGHT MRT slots, not just slot 0. The first version read only
                        // CB_COLOR0_BASE, and its own control exposed that: two surfaces which
                        // demonstrably HIT the RTT cache (0x2063380000, 0x2085de0000) never appeared
                        // in the census at all. A deferred renderer writes a G-buffer across slots
                        // 0..7 in one draw, so a slot-0-only census cannot see most of what a frame
                        // renders into -- and "is address X ever a render target" was exactly the
                        // question being asked of it.
                        //
                        // Register layout from render_state.cpp: stride 0xf per slot for BASE, and
                        // CB_COLOR0_BASE_EXT + slot for the high bits.
                        namespace P = prosper::agc::Pm4;
                        constexpr uint32_t kColorRegisterStride = 0xf;
                        if (const GpuState* snap = st.draws[operation.index].state.get()) {
                            std::lock_guard lock(target_mutex);
                            for (uint32_t slot = 0; slot < 8u; ++slot) {
                                const auto lo = snap->cx.find(
                                    P::CB_COLOR0_BASE + slot * kColorRegisterStride);
                                if (lo == snap->cx.end() || !lo->second) continue;
                                const auto hi = snap->cx.find(P::CB_COLOR0_BASE_EXT + slot);
                                uint64_t base = static_cast<uint64_t>(lo->second) << 8;
                                if (hi != snap->cx.end())
                                    base |= static_cast<uint64_t>(hi->second & 0xffu) << 40;
                                ++targets[{base, slot, 0u}];
                            }
                        }
                    }
                    if ((n & (n - 1)) == 0 && n >= 4096u) {
                        std::fprintf(stderr,
                                     "[draw-census] draws=%llu indirect=%llu submit=%llu\n",
                                     (unsigned long long)n,
                                     (unsigned long long)indirect_seen.load(),
                                     (unsigned long long)submit_no);
                        std::lock_guard lock(target_mutex);
                        // Busiest first. The numerically-lowest twelve addresses said nothing: the
                        // question is whether one target dominates (a scene buffer) or the draws are
                        // spread over hundreds of small ones.
                        std::vector<std::pair<uint64_t, std::tuple<uint64_t, uint32_t, uint32_t>>>
                            ranked;
                        ranked.reserve(targets.size());
                        for (const auto& e : targets) ranked.push_back({e.second, e.first});
                        std::sort(ranked.begin(), ranked.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                        std::fprintf(stderr, "[draw-census]   distinct colour targets (sampled): %zu\n",
                                     targets.size());
                        size_t shown = 0;
                        for (const auto& r : ranked) {
                            const std::pair<const std::tuple<uint64_t, uint32_t, uint32_t>, uint64_t>
                                entry{r.second, r.first};
                            // All of them at the final report. Twelve was enough to see whether one
                            // target dominates; it is not enough to answer "is address X ever a
                            // render target", which is the question a specific suspect raises.
                            const size_t limit = n >= 65536u ? targets.size() : 12u;
                            if (shown++ >= limit) { std::fprintf(stderr,
                                "[draw-census]   ... %zu further targets\n",
                                targets.size() - limit); break; }
                            std::fprintf(stderr,
                                         "[draw-census]   target=0x%llx slot=%u draws=%llu\n",
                                         (unsigned long long)std::get<0>(entry.first),
                                         std::get<1>(entry.first),
                                         (unsigned long long)entry.second);
                        }
                    }
                }
                if (!render) break;
                if (st.draws[operation.index].indirect &&
                    (!indirect_dependencies_ok || !producer_epoch_ok)) {
                    // The indirect latch drops this draw SILENTLY on an ordinary run -- the failure
                    // is recorded only when a capture trace happens to be active. On a title whose
                    // world is GPU-driven that is the difference between "the world is missing" and
                    // "N indirect draws were dropped because a producer was declined", and nothing
                    // reported it.
                    //
                    // Counted always, printed at powers of two so the volume stays bounded on a
                    // title that does this every frame. The count is the point: an absent world with
                    // zero dropped indirect draws and one with thousands are different problems.
                    static std::atomic<uint64_t> latched_draws{0};
                    const uint64_t dropped = latched_draws.fetch_add(1) + 1;
                    if ((dropped & (dropped - 1)) == 0)
                        std::fprintf(stderr,
                                     "[agc] indirect DRAW dropped by the dependency latch "
                                     "(count=%llu, submit=%llu, deps-ok=%u producer-ok=%u)\n",
                                     (unsigned long long)dropped,
                                     (unsigned long long)submit_no,
                                     indirect_dependencies_ok ? 1u : 0u,
                                     producer_epoch_ok ? 1u : 0u);
                    if (capture_trace) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Draw, operation.index,
                            operation.command_order,
                            RealizationFailureReason::IndirectDependencies});
                    }
                    break;
                }
                // Resource realization can inspect guest descriptor/backing state.  No exact draw
                // read range has yet been proven at this seam, so an armed authority census must
                // fail closed before that first possible consumer rather than after rendering.
                notify_compute_authority_unknown(
                    ComputeAuthorityBoundaryKind::Draw,
                    submit_no, operation.command_order);
                DrawItem item;
                bool realized = false;
                OperationRealizationFailure failure;
                bool failure_known = false;
                if (eager_draws) {
                    const auto found = eager_draw_by_index.find(operation.index);
                    if (found != eager_draw_by_index.end()) {
                        item = (*eager_draws)[found->second];
                        realized = true;
                    }
                } else {
                    realized = realize_retained_draw(
                        st, operation.index, scale_x, scale_y, item,
                        capture_trace ? &failure : nullptr);
                    failure_known = capture_trace != nullptr;
                }
                if (realized) {
                    notify_compute_authority_draw_resources(item, submit_no);
                    if (capture_trace) {
                        snapshot_pending_gpu_capture_draw_resource(
                            capture_trace->pending_capture, item, {}, &st);
                        capture_trace->draws.push_back(item);
                    }
                    span.push_back(std::move(item));
                } else {
                    notify_compute_authority_draw_unrealized(
                        submit_no, operation.command_order);
                    if (!capture_trace) break;
                    // The reason used to die here: this path simply broke, and gpu_capture later
                    // synthesized an empty Unknown record for the unrealized operation (#1636).
                    if (failure_known) {
                        failure.command_order = operation.command_order;
                        capture_trace->failures.push_back(std::move(failure));
                    } else {
                        // Eager path: realize_gpustate_draws DOES have a failures out-parameter,
                        // but this submit's call site (:6023) passes nullptr, so the reason was
                        // dropped in that earlier pass and Unknown is the honest answer here rather
                        // than a guess. Not a simple plumb-through to fix: requesting failures also
                        // takes the serial path (gpu_execute.hpp:1650 gates parallel realization on
                        // `!failures`), so it trades draw-realization throughput for diagnostics.
                        // Tracked in #1643.
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Draw, operation.index,
                            operation.command_order, RealizationFailureReason::Unknown});
                    }
                }
                break;
            }
            case RetainedSubmitKind::Dispatch: {
                flush_span();
                const uint64_t current_compute_code = compute_dispatch_code_addr(
                    st, st.dispatches[operation.index]);
                const bool indirect = st.dispatches[operation.index].indirect;
                if (indirect && (!indirect_dependencies_ok || !producer_epoch_ok)) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    if (capture_trace) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order,
                            RealizationFailureReason::IndirectDependencies});
                    }
                    producer_epoch_ok = false;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = false;
                    previous_compute_executed = false;
                    break;
                }
                GpuState::Dispatch resolved_dispatch;
                const DispatchArgumentResolution argument_resolution =
                    resolve_indirect_dispatch_arguments(
                        st.dispatches[operation.index], resolved_dispatch);
                if (argument_resolution == DispatchArgumentResolution::Noop) {
                    // Argument memory is readable and proves that no wave launches. This is neutral
                    // even when no compute backend is installed: there is no producer to execute.
                    if (capture_trace)
                        capture_trace->noops.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order});
                    break;
                }
                if (argument_resolution != DispatchArgumentResolution::Ready) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    if (capture_trace) {
                        // A dispatch's indirect arguments fail exactly as a draw's do; recording it
                        // as Unknown was the documented gap in the reason enum, not a distinct case.
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order,
                            RealizationFailureReason::IndirectArguments});
                    }
                    producer_epoch_ok = false;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = false;
                    previous_compute_executed = false;
                    break;
                }
                if (!compute) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    if (capture_trace) {
                        // The dispatch was ready and nothing about it failed: there is simply no
                        // live compute backend. Recording that as Unknown makes a configuration
                        // fact indistinguishable from a defect in the dispatch itself.
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order,
                            RealizationFailureReason::ComputeBackendUnavailable});
                    }
                    producer_epoch_ok = false;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = false;
                    previous_compute_executed = false;
                    break;
                }
                ComputeItem item;
                OperationRealizationFailure failure;
                const RetainedComputeRealization realization = realize_retained_compute(
                    st, operation.index, resolved_dispatch, submit_no, item,
                    capture_trace ? &failure : nullptr);
                if (realization == RetainedComputeRealization::Realized) {
                    // PROSPER_COMPUTE_ADDRESS_WATCH=0xADDR: name every program whose realized
                    // resource table touches one guest allocation, with the binding and fetch pc
                    // that reach it. One table scan per dispatch, no per-op cost.
                    //
                    // Placed HERE, beside the parent walk, and that placement is the point: this
                    // watch first went into realize_compute_dispatches() and printed nothing for a
                    // whole route, because the live path is realize_retained_compute(). A silent
                    // zero from an instrument on an unused code path is indistinguishable from
                    // "nothing touches this buffer" -- which is the conclusion it would have
                    // supported, and it is false. Anchor a new watch to a line already proven to
                    // execute on the route being measured.
                    // PROSPER_COMPUTE_RESOURCE_MAP=0xADDR[,0xADDR...]: print each named program's
                    // realized resource table once, with binding, fetch pc, base and size. The
                    // question it exists to answer is "do these two programs bind the same buffer",
                    // which the address watch cannot express -- that one starts from an address and
                    // finds programs, and here the address is precisely what is unknown.
                    //
                    // Deduped per (program, table shape) so a program dispatched forty times per
                    // route prints once. Volume matters: the existing PROSPER_COMPUTELOG_RESOURCES
                    // needs PROSPER_COMPUTELOG, which slows the subject enough that a routed run
                    // stops reaching the phase being measured.
                    if (const char* map = std::getenv("PROSPER_COMPUTE_RESOURCE_MAP")) {
                        bool wanted_program = false;
                        for (const char* cursor = map; cursor && *cursor;) {
                            char* end = nullptr;
                            const uint64_t one = std::strtoull(cursor, &end, 0);
                            if (end == cursor) break;
                            if (one == item.code_addr) { wanted_program = true; break; }
                            cursor = (*end == ',') ? end + 1 : end;
                            if (!*end) break;
                        }
                        if (wanted_program && item.resources) {
                            uint64_t shape = item.resources->resources.size();
                            for (const ShaderResource& r : item.resources->resources)
                                shape = shape * 1000003ull + r.gpu_addr + r.size + r.binding;
                            static std::mutex map_mutex;
                            static std::set<std::pair<uint64_t, uint64_t>> map_logged;
                            bool first = false;
                            {
                                std::lock_guard lock(map_mutex);
                                first = map_logged.emplace(item.code_addr, shape).second;
                            }
                            if (first) {
                                std::fprintf(stderr,
                                             "[compute-resource-map] program=0x%llx submit=%llu "
                                             "dispatch=%zu resources=%zu\n",
                                             static_cast<unsigned long long>(item.code_addr),
                                             static_cast<unsigned long long>(submit_no),
                                             operation.index,
                                             item.resources->resources.size());
                                for (const ShaderResource& r : item.resources->resources) {
                                    std::fprintf(stderr,
                                                 "[compute-resource-map]   binding=%u fetch-pc=%u "
                                                 "base=0x%llx size=%u stride=%u entries=%u\n",
                                                 r.binding, r.fetch_pc,
                                                 static_cast<unsigned long long>(r.gpu_addr),
                                                 r.size, r.stride, r.table_index_count);
                                    for (size_t e = 0; e < r.table_entries.size(); ++e)
                                        std::fprintf(stderr,
                                                     "[compute-resource-map]     entry=%zu "
                                                     "base=0x%llx size=%u\n",
                                                     e,
                                                     static_cast<unsigned long long>(
                                                         r.table_entries[e].gpu_addr),
                                                     r.table_entries[e].size);
                                }
                            }
                        }
                    }
                    if (const char* watch = std::getenv("PROSPER_COMPUTE_ADDRESS_WATCH")) {
                        char* end = nullptr;
                        const uint64_t wanted = std::strtoull(watch, &end, 0);
                        if (end && !*end && wanted && item.resources) {
                            // Traversal lives in compute_address_watch_hits so the scalar/array
                            // branch is testable: a runtime-array parent's gpu_addr is the
                            // DESCRIPTOR TABLE address, not a backing range, and testing it here
                            // fabricated matches and double-reported entries.
                            for (const ComputeAddressWatchHit& hit :
                                     compute_address_watch_hits(*item.resources, wanted)) {
                                std::fprintf(stderr,
                                             "[compute-address-watch] addr=0x%llx %s program=0x%llx "
                                             "submit=%llu dispatch=%zu order=%llu binding=%u "
                                             "fetch-pc=%u entry=%u base=0x%llx size=%u\n",
                                             static_cast<unsigned long long>(wanted),
                                             hit.from_array_entry ? "array-entry" : "resource",
                                             static_cast<unsigned long long>(item.code_addr),
                                             static_cast<unsigned long long>(submit_no),
                                             operation.index,
                                             static_cast<unsigned long long>(
                                                 operation.command_order),
                                             hit.binding, hit.fetch_pc, hit.entry_index,
                                             static_cast<unsigned long long>(hit.base), hit.size);
                            }
                        }
                    }
                    const LiveComputeParentWalkPreflight preflight =
                        preflight_compute_parent_walk(
                            item, previous_compute_code, previous_compute_realized,
                            previous_compute_executed);
                    if (preflight.selected && preflight.available && preflight.suspicious) {
                        std::fprintf(stderr,
                                     "[compute-parent-walk] DIAGNOSTIC-ONLY skip suspicious "
                                     "dispatch submit=%llu dispatch=%zu order=%llu code=0x%llx\n",
                                     static_cast<unsigned long long>(submit_no), operation.index,
                                     static_cast<unsigned long long>(operation.command_order),
                                     static_cast<unsigned long long>(item.code_addr));
                        notify_compute_authority_unknown(
                            ComputeAuthorityBoundaryKind::Compute,
                            submit_no, operation.command_order);
                        if (capture_trace) {
                            // A deliberate diagnostic skip, not a failure of the dispatch. Naming it
                            // keeps the census honest about how much of a phase prosper chose not to
                            // run versus could not.
                            capture_trace->failures.push_back({
                                SubmitOperationKind::Dispatch, operation.index,
                                operation.command_order,
                                RealizationFailureReason::SuspiciousDispatchSkipped});
                        }
                        producer_epoch_ok = false;
                        previous_compute_code = current_compute_code;
                        previous_compute_realized = true;
                        previous_compute_executed = false;
                        break;
                    }
                    // Pre/post around the dispatch, so a change to the watched table is
                    // attributed to the dispatch that made it. Captured before the move: the
                    // post observation needs the program address, and `item` is consumed below.
                    // PROSPER_COMPUTE_ZERO_BEFORE=0xPROGRAM:0xADDR:DWORDS — zero a guest range
                    // immediately before the named program dispatches. DIAGNOSTIC ONLY: it fixes
                    // nothing, because on hardware something must be performing whatever
                    // initialisation this stands in for, and identifying that is the actual work.
                    //
                    // It exists to test one specific prediction that no amount of reading the ISA
                    // settles: GTA V's tree builder writes parent links only for children whose node
                    // type is in {2,5}, so the slots it skips must already hold a value that
                    // terminates the consumer's `while (i != 0) i = bfe(rec[i], 3, 27)` walk. Zero
                    // terminates; a leftover index from an earlier generation of the same array does
                    // not. If the array is simply never cleared, zeroing it here should turn a cyclic
                    // tree into a legitimately sparse one -- pairs below 1030 and cycles at zero.
                    // That outcome is not reachable by any other experiment available.
                    if (const char* zero_spec = std::getenv("PROSPER_COMPUTE_ZERO_BEFORE")) {
                        char* cursor = nullptr;
                        const uint64_t want_program = std::strtoull(zero_spec, &cursor, 0);
                        if (cursor && *cursor == ':' && want_program == item.code_addr) {
                            const uint64_t zero_addr = std::strtoull(cursor + 1, &cursor, 0);
                            uint64_t zero_dwords = 0;
                            if (cursor && *cursor == ':')
                                zero_dwords = std::strtoull(cursor + 1, &cursor, 0);
                            const uint64_t zero_bytes = zero_dwords * sizeof(uint32_t);
                            if (zero_addr && zero_dwords && zero_dwords <= (1u << 22u) &&
                                guest_readable(zero_addr, zero_bytes)) {
                                std::memset(
                                    reinterpret_cast<void*>(static_cast<uintptr_t>(zero_addr)), 0,
                                    static_cast<size_t>(zero_bytes));
                                static std::mutex zero_mutex;
                                static uint64_t zero_count = 0;
                                uint64_t count = 0;
                                {
                                    std::lock_guard lock(zero_mutex);
                                    count = ++zero_count;
                                }
                                if (count <= 3 || (count % 25) == 0)
                                    std::fprintf(stderr,
                                                 "[compute-zero-before] program=0x%llx addr=0x%llx "
                                                 "dwords=%llu submit=%llu dispatch=%zu count=%llu\n",
                                                 (unsigned long long)item.code_addr,
                                                 (unsigned long long)zero_addr,
                                                 (unsigned long long)zero_dwords,
                                                 (unsigned long long)submit_no, operation.index,
                                                 (unsigned long long)count);
                            }
                        }
                    }
                    const uint64_t tree_watch_program = item.code_addr;
                    const ComputeLaunchDimensions tree_watch_launch = item.launch;
                    const std::string tree_watch_touch =
                        compute_tree_watch_selector()
                            ? describe_compute_tree_watch_touch(
                                  item.resources.get(), compute_tree_watch_selector()->addr,
                                  static_cast<uint64_t>(compute_tree_watch_selector()->records) *
                                      sizeof(uint32_t))
                            : std::string();
                    const std::vector<uint32_t> tree_watch_pre =
                        observe_compute_tree_watch_pre(item, submit_no, operation.index,
                                                       operation.command_order, tree_watch_touch);
                    const bool executed = capture_trace
                        ? compute({item}) : compute({std::move(item)});
                    log_compute_dispatch(tree_watch_program, submit_no, operation.index,
                                         operation.command_order,
                                         executed ? "executed" : "backend-declined",
                                         &tree_watch_launch);
                    observe_compute_tree_watch_post(tree_watch_program, tree_watch_touch, submit_no,
                                                    operation.index, operation.command_order,
                                                    tree_watch_pre);
                    if (capture_trace && executed)
                        capture_trace->computes.push_back(std::move(item));
                    if (capture_trace && !executed) {
                        // Everything about the dispatch resolved; the live backend refused to run
                        // it. That is a different investigation from an unresolved argument or a
                        // missing producer, and recording all three as Unknown hid which one a
                        // phase was actually made of.
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order,
                            RealizationFailureReason::ComputeExecutionDeclined});
                    }
                    result.compute_executed |= executed;
                    producer_epoch_ok &= executed;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = true;
                    previous_compute_executed = executed;
                } else if (capture_trace && failure.reason != RealizationFailureReason::None) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    capture_trace->failures.push_back(std::move(failure));
                    producer_epoch_ok = false;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = false;
                    previous_compute_executed = false;
                } else {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    producer_epoch_ok = false;
                    previous_compute_code = current_compute_code;
                    previous_compute_realized = false;
                    previous_compute_executed = false;
                }
                break;
            }
            case RetainedSubmitKind::DmaCopy: {
                flush_span(true);
                const GpuState::DmaCopy& copy = st.dma_copies[operation.index];
                // Source and destination are distinct ordered consumers: a disjoint source must not
                // hide an overlapping destination (or vice versa).
                const bool source_gds = ((copy.sels >> 8u) & 0xffu) == 1u;
                const bool destination_gds = (copy.sels & 0xffu) == 1u;
                if (!source_gds)
                    notify_compute_authority_range(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order, copy.src, copy.bytes);
                // Selector byte 1 names a GDS offset, not guest memory. The source still consumes a
                // guest/render-target range, while the destination is ordered by this executor's
                // shared GDS backing and must not manufacture a guest range at (for example) 0x24.
                if (!destination_gds)
                    notify_compute_authority_range(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order, copy.dst, copy.bytes);
                std::vector<uint8_t> current_source;
                const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                    copy.src, copy.bytes, current_source);
                if (source_result == LiveTargetByteReadResult::InvalidRange) {
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        std::fprintf(stderr,
                                     "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                     static_cast<unsigned long long>(copy.src), copy.bytes);
                    producer_epoch_ok = false;
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order);
                    break;
                }
                const bool executed = execute_ordered_dma_copy(
                    copy, source_result == LiveTargetByteReadResult::Success
                              ? current_source.data() : nullptr);
                producer_epoch_ok &= executed;
                if (!executed)
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order);
                break;
            }
            case RetainedSubmitKind::ParserStall:
                flush_span();
                // Once an argument-producing epoch fails, a later empty/redundant stall cannot
                // make the stale bytes trustworthy again. Keep the submit poisoned until it ends.
                indirect_dependencies_ok &= producer_epoch_ok;
                producer_epoch_ok = true;
                break;
            case RetainedSubmitKind::MemoryEffect:
                flush_span();
                {
                    const GpuState::MemoryEffect& effect =
                        st.ordered_memory_effects[operation.index];
                    const Pm4Command& command = effect.cmd;
                    bool exact = false;
                    switch (command.kind) {
                        case Pm4Command::Kind::ReleaseMem:
                            notify_compute_authority_range(
                                ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                submit_no, operation.command_order, command.rel_addr,
                                command.rel_data_sel == 1u ? 4u : 8u);
                            exact = command.rel_addr != 0;
                            break;
                        case Pm4Command::Kind::EventWrite:
                            notify_compute_authority_range(
                                ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                submit_no, operation.command_order, command.event_addr, 8u);
                            exact = command.event_addr != 0;
                            break;
                        case Pm4Command::Kind::WriteData:
                            if (command.wd_valid) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                    submit_no, operation.command_order, command.wd_addr,
                                    static_cast<uint64_t>(command.wd_num) * 4u);
                                exact = command.wd_addr != 0 && command.wd_num != 0;
                            }
                            break;
                        case Pm4Command::Kind::DmaData: {
                            // Selector byte 1 names the 64 KiB GDS offset domain, not guest VA.
                            // Every other valid destination is an exact guest write range.  A
                            // An asserted address source (or a legacy >32-bit source) is observed
                            // separately; numeric width alone is insufficient for new HLE packets.
                            const bool source_gds = ((command.dd_sels >> 8u) & 0xffu) == 1u;
                            const bool destination_gds = (command.dd_sels & 0xffu) == 1u;
                            const bool address_source =
                                (command.dd_sels & kDmaDataAddressSource) != 0 ||
                                command.dd_src > UINT32_MAX;
                            if (!source_gds && address_source) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::Dma, submit_no,
                                    operation.command_order, command.dd_src,
                                    command.dd_bytes);
                                exact = command.dd_bytes != 0;
                            }
                            if (!destination_gds) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::Dma, submit_no,
                                    operation.command_order, command.dd_dst,
                                    command.dd_bytes);
                                exact = exact ||
                                    (command.dd_dst != 0 && command.dd_bytes != 0);
                            }
                            break;
                        }
                        default:
                            break;
                    }
                    if (!exact)
                        notify_compute_authority_unknown(
                            ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                            submit_no, operation.command_order);
                    execute_ordered_memory_effect(effect);
                }
                break;
        }
    }
    flush_span();
    // A semantic draw record can fail only when lazily realized at its ordered position. If that
    // record was counted as a later span, the last successful callback was intentionally marked
    // intermediate. Send an empty terminal callback so the frontend can recover cached scanout,
    // close timing state, and publish exactly once without re-rendering any draw.
    if (render && result.render_spans && !final_callback_sent) {
        LiveRenderPhase saved = g_live_phase;
        g_live_phase = {false, true, false};
        RenderedFrame rendered = render({}, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
    }
    return result;
}

void set_submit_renderer(LiveRenderFn fn) { g_live = std::move(fn); }
bool have_submit_renderer()               { return static_cast<bool>(g_live); }
uint8_t* compute_gds_backing()            { return g_compute_gds.data(); }
size_t   compute_gds_size()               { return g_compute_gds.size(); }
void set_submit_compute(LiveComputeFn fn) { g_compute = std::move(fn); }
bool have_submit_compute()                { return static_cast<bool>(g_compute); }
void notify_compute_authority_boundary(
        const ComputeAuthorityBoundary& boundary) {
    dispatch_compute_authority_boundary(boundary);
}
void set_compute_authority_boundary_observer(
        ComputeAuthorityBoundaryObserver observer) {
    std::lock_guard lock(g_compute_authority_boundary_mutex);
    g_compute_authority_boundary_observer = std::move(observer);
    g_compute_authority_boundary_enabled.store(
        static_cast<bool>(g_compute_authority_boundary_observer),
        std::memory_order_release);
}

static LiveTargetQueryFn g_live_target_query;   // registered by the live renderer (#590)
void set_live_target_query(LiveTargetQueryFn fn) { g_live_target_query = std::move(fn); }
bool is_live_render_target(uint64_t gpu_addr) {
    return g_live_target_query && g_live_target_query(gpu_addr);
}
static LiveTargetReaderFn g_live_target_reader;
void set_live_target_reader(LiveTargetReaderFn fn) { g_live_target_reader = std::move(fn); }

static LiveTargetImageImportFn g_live_target_image_import;
static LiveTargetImageReleaseFn g_live_target_image_release;
static LiveTargetImageWrittenFn g_live_target_image_written;
void set_live_target_image_importer(LiveTargetImageImportFn import_fn,
                                    LiveTargetImageReleaseFn release_fn) {
    g_live_target_image_import = std::move(import_fn);
    g_live_target_image_release = std::move(release_fn);
}
void set_live_target_image_written_notifier(LiveTargetImageWrittenFn written_fn) {
    g_live_target_image_written = std::move(written_fn);
}

static MetadataKindQueryFn g_metadata_kind_query;
void set_metadata_kind_query(MetadataKindQueryFn fn) { g_metadata_kind_query = std::move(fn); }

CompressionMetadataKind classify_compression_metadata_kind(const MetadataKindRequest& request) {
    // No renderer, no correlation, no kind. Unknown authorizes nothing downstream, so a backend running
    // without a registered renderer fails closed rather than inheriting a guess.
    if (!request.metadata_addr || !g_metadata_kind_query) return CompressionMetadataKind::Unknown;
    return g_metadata_kind_query(request);
}
bool import_live_render_target_image(uint64_t gpu_addr, const LiveTargetImageRequest& request,
                                     LiveTargetImageImport& import) {
    import = LiveTargetImageImport{};
    if (!g_live_target_image_import) return false;
    if (!g_live_target_image_import(gpu_addr, request, import)) {
        import = LiveTargetImageImport{};
        return false;
    }
    if (!import.valid()) {
        // The importer pinned the entry before returning true; drop that pin rather than leaking a
        // permanently un-evictable cache entry if a future importer breaks the contract.
        release_live_render_target_image(gpu_addr);
        import = LiveTargetImageImport{};
        return false;
    }
    return true;
}
void release_live_render_target_image(uint64_t gpu_addr) {
    if (g_live_target_image_release) g_live_target_image_release(gpu_addr);
}
void notify_live_render_target_image_written(const LiveTargetImageWrite& write) {
    if (g_live_target_image_written && write.valid()) g_live_target_image_written(write);
}
static SharedVulkanContext g_shared_vulkan;
void set_shared_vulkan_context(const SharedVulkanContext& context) { g_shared_vulkan = context; }
SharedVulkanContext shared_vulkan_context() { return g_shared_vulkan; }

uint32_t select_native_compute_subgroup_size(const SharedVulkanContext& context,
                                             const ComputeShaderConfig& config,
                                             bool allow_multiwave, bool disabled) {
    const bool adoptable = context.valid() && context.compute_queue_supported &&
        context.storage_image_read_without_format &&
        context.storage_image_write_without_format;
    if (disabled || !adoptable ||
        !context.compute_subgroup_size_control || !context.compute_full_subgroups ||
        !context.compute_subgroup_vote || !context.compute_subgroup_arithmetic ||
        !context.max_compute_workgroup_subgroups || !context.max_compute_workgroup_size_x ||
        !context.max_compute_workgroup_invocations || !config.local_x || !config.local_y ||
        !config.local_z || (config.wave_size != 32u && config.wave_size != 64u) ||
        config.wave_size < context.min_compute_subgroup_size ||
        config.wave_size > context.max_compute_subgroup_size)
        return 0;

    // The native shader declares a flattened LocalSize=(guest X*Y*Z,1,1), then reconstructs the
    // guest 3D local/global IDs from SubgroupId/SubgroupLocalInvocationId. Besides avoiding any
    // implementation-defined lane ordering, this makes Vulkan's REQUIRE_FULL_SUBGROUPS X-dimension
    // rule explicit. Keep the multiplication and maxComputeWorkgroupSubgroups bound overflow-safe.
    const uint64_t xy = static_cast<uint64_t>(config.local_x) * config.local_y;
    if (xy > UINT64_MAX / config.local_z) return 0;
    const uint64_t local_invocations = xy * config.local_z;
    const uint64_t subgroup_capacity = static_cast<uint64_t>(config.wave_size) *
        context.max_compute_workgroup_subgroups;
    // One guest wave removes the portable shell without adding inter-subgroup coordinate recovery.
    // Multi-wave kernels can be faster or slower depending on their LDS/barrier shape, so retain
    // the portable default until a diagnostic run opts into the exact experimental contract.
    if (local_invocations % config.wave_size != 0 ||
        (!allow_multiwave && local_invocations != config.wave_size) ||
        local_invocations > subgroup_capacity ||
        local_invocations > context.max_compute_workgroup_size_x ||
        local_invocations > context.max_compute_workgroup_invocations ||
        local_invocations > UINT32_MAX)
        return 0;
    return config.wave_size;
}

// Present unification (#1270): see gpu_execute.hpp. The atomic gates the lock so the common (headless /
// non-shared / app-not-yet-adopted) path pays only a single acquire load and takes no lock. Set true
// exactly once, by prosper-app, after it has adopted the shared queue for present and before its first
// present submit; never cleared mid-run (the app owns the shared device for the process lifetime once
// adopted). acquire/release ordering publishes the adoption's writes to the guest thread.
static std::atomic<bool> g_shared_present_active{false};
std::mutex& shared_present_submit_mutex() {
    static std::mutex m;
    return m;
}
void set_shared_present_active(bool active) {
    g_shared_present_active.store(active, std::memory_order_release);
}
bool shared_present_active() {
    return g_shared_present_active.load(std::memory_order_acquire);
}
static std::atomic<bool> g_gpu_present_active{false};
void set_gpu_present_active(bool active) {
    g_gpu_present_active.store(active, std::memory_order_release);
}
bool gpu_present_active() {
    return g_gpu_present_active.load(std::memory_order_acquire);
}
bool read_live_render_target(uint64_t gpu_addr, LiveTargetSnapshot& snapshot) {
    snapshot = {};
    return g_live_target_reader && g_live_target_reader(gpu_addr, snapshot);
}
static LiveTargetByteRangeReaderFn g_live_target_byte_range_reader;
void set_live_target_byte_range_reader(LiveTargetByteRangeReaderFn fn) {
    g_live_target_byte_range_reader = std::move(fn);
}
LiveTargetByteReadResult read_live_render_target_bytes(uint64_t gpu_addr, uint32_t bytes,
                                                       std::vector<uint8_t>& output) {
    output.clear();
    if (!g_live_target_byte_range_reader)
        return LiveTargetByteReadResult::NotFound;
    return g_live_target_byte_range_reader(gpu_addr, bytes, output);
}

void set_guest_gpu_write_observer(GuestGpuWriteObserver observer) {
    g_guest_gpu_write_observer = std::move(observer);
}
// PROSPER_GUEST_WRITE_WATCH=0xADDR[,0xADDR…] — report every guest-side GPU write that overlaps a
// named address, with the caller that produced it.
//
// A colour-target census answers "which DRAW wrote this surface", and a compute address watch
// answers "which DISPATCH bound it". Neither can see a DMA_DATA copy, a WRITE_DATA, a RELEASE_MEM or
// an EVENT_WRITE, so a surface filled by any of those reads as "written by nothing" in both -- which
// is exactly the state GTA V's 4K HDR scene colour is in, and the reason a third instrument is
// needed before concluding that nothing fills it.
void report_guest_write_watch(uint64_t addr, uint64_t size, const char* origin) {
    static const std::vector<uint64_t> watched = [] {
        std::vector<uint64_t> out;
        const char* spec = std::getenv("PROSPER_GUEST_WRITE_WATCH");
        for (const char* p = spec; p && *p;) {
            char* end = nullptr;
            const uint64_t v = std::strtoull(p, &end, 0);
            if (end == p) break;
            if (v) out.push_back(v);
            p = (*end == ',') ? end + 1 : end;
        }
        return out;
    }();
    if (watched.empty()) return;
    for (const uint64_t want : watched) {
        // Any overlap counts: a fill of a 4K surface is one large range, not a write at its base.
        if (!(addr <= want && want < addr + size)) continue;
        static std::mutex mutex;
        static std::map<std::pair<uint64_t, std::string>, uint64_t> seen;
        std::lock_guard lock(mutex);
        const uint64_t n = ++seen[{want, origin}];
        if (n <= 4 || (n & (n - 1)) == 0)
            std::fprintf(stderr,
                         "[guest-write-watch] 0x%llx covered by %s write addr=0x%llx size=%llu "
                         "(x%llu)\n",
                         (unsigned long long)want, origin, (unsigned long long)addr,
                         (unsigned long long)size, (unsigned long long)n);
    }
}

// Thread-local origin for the next guest write, so PROSPER_GUEST_WRITE_WATCH can name the PM4
// packet that produced it. "a guest write covers this surface" and "a DMA_DATA fill of 512 KiB
// covers this surface" are different facts, and only the second says whether the depth contents
// were actually replaced.
// "unknown", not "gpu". A default is not a measurement: naming it after a producer made an
// unattributed bucket read as "genuinely the guest", and a published census then reported
// prosper's own byte-preserving writebacks as guest HTILE writes.
thread_local const char* g_guest_write_origin = "unknown";
void set_guest_gpu_write_origin(const char* origin) {
    g_guest_write_origin = origin ? origin : "unknown";
}
const char* guest_gpu_write_origin() { return g_guest_write_origin; }

void notify_guest_gpu_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    report_guest_write_watch(addr, size, g_guest_write_origin);
    // Page-protection watches observe guest CPU stores, but device/DMA writes can mutate the same
    // direct-memory pages without a CPU protection fault. Mark the virtual range dirty as part of the
    // existing authoritative GPU-write notification so cross-submit texture/compute caches never trust
    // an Unchanged watch over bytes written by the GPU. This may run after a host-mirrored write and
    // is deliberately idempotent.
    prosper::host::guest_write_watch_notify_gpu_write(addr, size);
    if (g_guest_gpu_writes.active) {
        if (g_guest_gpu_writes.writes.size() < kGuestGpuWriteJournalCapacity)
            g_guest_gpu_writes.writes.push_back({addr, size});
        else
            g_guest_gpu_writes.overflowed = true;
    }
    if (g_guest_gpu_write_observer) g_guest_gpu_write_observer(addr, size, g_guest_write_origin);
}
void notify_guest_gpu_write_preserving_bytes(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    report_guest_write_watch(addr, size, "gpu-preserving");
    // The observer owns renderer-resident aliases (color/depth targets and their CPU snapshots),
    // which may differ from the exact guest bytes even when a compute result does not. Guest-memory
    // caches, page watches, and the submit journal remain valid because the caller proved that those
    // bytes were not modified.
    //
    // Forward the classification this function already knows. Dropping it here is what made every
    // byte-preserving compute writeback arrive at the queue as the unattributed default, so a census
    // could not separate prosper's own writes from the guest's -- the exact distinction the origin
    // field exists to draw. A caller-set origin wins, since it is more specific than this one.
    const char* preserving_origin =
        std::strcmp(g_guest_write_origin, "unknown") == 0 ? "gpu-preserving" : g_guest_write_origin;
    if (g_guest_gpu_write_observer) g_guest_gpu_write_observer(addr, size, preserving_origin);
}
GuestGpuWriteSnapshot guest_gpu_write_snapshot() {
    if (!g_guest_gpu_writes.active || g_guest_gpu_writes.overflowed) return {};
    return {g_guest_gpu_writes.submit_serial, g_guest_gpu_writes.writes.size()};
}
bool guest_gpu_write_tracking_active() { return g_guest_gpu_writes.active; }
GuestGpuWriteQuery guest_gpu_writes_since(const GuestGpuWriteSnapshot& snapshot,
                                           uint64_t addr, uint64_t size) {
    if (!snapshot.submit_serial || !g_guest_gpu_writes.active ||
        snapshot.submit_serial != g_guest_gpu_writes.submit_serial ||
        g_guest_gpu_writes.overflowed ||
        snapshot.write_count > g_guest_gpu_writes.writes.size())
        return GuestGpuWriteQuery::Unknown;
    for (size_t i = snapshot.write_count; i < g_guest_gpu_writes.writes.size(); ++i) {
        const auto& write = g_guest_gpu_writes.writes[i];
        if (ranges_overlap(addr, size, write.addr, write.size))
            return GuestGpuWriteQuery::Overlap;
    }
    return GuestGpuWriteQuery::Unchanged;
}
LiveRenderPhase live_render_phase()       { return g_live_phase; }

// See PresentSubmitScope in gpu_execute.hpp. Thread-local and counted: the renderer callback runs
// synchronously on the submitting thread, and a nested scope must not end its parent's.
namespace { thread_local unsigned g_present_submit_depth = 0; }
PresentSubmitScope::PresentSubmitScope()  { ++g_present_submit_depth; }
PresentSubmitScope::~PresentSubmitScope() { --g_present_submit_depth; }
bool present_submit_in_progress()         { return g_present_submit_depth != 0; }

std::vector<uint8_t> render_submit_items(const std::vector<DrawItem>& items,
                                         uint32_t width, uint32_t height) {
    if (!g_live) return {};
    RenderedFrame frame = g_live(items, width, height);
    return frame.storage ? *frame.storage : std::vector<uint8_t>{};
}
bool execute_compute_items(const std::vector<ComputeItem>& items) {
    // Realization-time half of the paired read, taken at command-ordered execution rather than at
    // fold. Deliberately before the dispatch runs, so the comparison isolates what happened BETWEEN
    // resolution and execution and not what this dispatch itself wrote.
    const bool ok = g_compute && !items.empty() && g_compute(items);
    compute_memprobe_flush("post-compute-submit");
    return ok;
}

bool execute_ordered_and_present(const GpuState& st, uint32_t width, uint32_t height,
                                 uint64_t submit_no, bool publish) {
    if ((!g_live && !g_compute && st.dma_copies.empty()) ||
        (st.draws.empty() && st.dispatches.empty() && st.dma_copies.empty())) return false;
    // Guest allocations referenced by a GPU submit must remain mapped until that submit completes.
    // Reuse positive page/VirtualQuery results only inside this synchronous execution window; the
    // scope is discarded before guest code can submit a later mapping generation.
    GuestReadableSubmitScope guest_readable_scope;
    // This submit's frame is headed for the publish gate below, so the renderer owes us a frame of
    // exactly width*height*4 bytes (#1986). Opened regardless of `publish`: the gate's extent test
    // is independent of it, and the frame also feeds the capture, so the contract must not differ
    // between a published and an unpublished submit.
    PresentSubmitScope present_submit_scope;
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitBegin, submit_no);
    using TimingClock = std::chrono::steady_clock;
    const bool timing_enabled = std::getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto timing_start = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const ShaderRecompileCacheStats shader_before = timing_enabled
        ? shader_recompile_cache_stats() : ShaderRecompileCacheStats{};
    const ShaderDecodeCacheStats decode_before = timing_enabled
        ? shader_decode_cache_stats() : ShaderDecodeCacheStats{};
    const DrawRealizationPhaseStats phases_before = timing_enabled
        ? draw_realization_phase_stats() : DrawRealizationPhaseStats{};
    const StageTablePhaseStats table_phases_before = timing_enabled
        ? stage_table_phase_stats() : StageTablePhaseStats{};
    const ParallelDrawRealizationStats parallel_before = timing_enabled
        ? parallel_draw_realization_stats() : ParallelDrawRealizationStats{};
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    const bool has_ordered_dma = !st.dma_copies.empty();
    const bool has_indirect = std::any_of(st.draws.begin(), st.draws.end(),
                                         [](const auto& draw) { return draw.indirect; }) ||
                              std::any_of(st.dispatches.begin(), st.dispatches.end(),
                                         [](const auto& dispatch) { return dispatch.indirect; });
    const bool needs_ordered_realization = has_ordered_dma || has_indirect || !st.dispatches.empty();
    // Draws without DMA/indirect arguments remain safe to prepare in parallel. Compute resources,
    // however, are always realized at their ordered position: a preceding dispatch in the same
    // submit can write a pointer or descriptor consumed by the next dispatch (Astro Bot's BVH root
    // is one such dependency). Pre-realizing every compute snapshots stale guest bytes.
    const bool can_eagerly_realize_draws = !has_ordered_dma && !has_indirect;
    // The normal AGC path is serialized, but execute_ordered_and_present is public and tests may
    // call it concurrently. The active collection generation is process-global because eager draw
    // workers need to see it, so serialize armed executions only; the default path never locks.
    std::unique_lock<std::mutex> null_image_source_probe_run_lock(
        g_null_image_source_probe_run_mutex, std::defer_lock);
    if (null_image_source_probe_program()) null_image_source_probe_run_lock.lock();
    std::vector<DrawItem> draws;
    if (can_eagerly_realize_draws && g_live && width && height) {
        g_collect_null_image_source_probe_submit.store(
            null_image_source_probe_program() ? submit_no : 0u, std::memory_order_relaxed);
        draws = realize_gpustate_draws(st, 0x10000, sx, sy, nullptr, true);
        g_collect_null_image_source_probe_submit.store(0, std::memory_order_relaxed);
    }
    std::vector<NullImageSourceProbe> null_image_source_probes =
        take_null_image_source_probes(draws, submit_no);
    const ShaderRecompileCacheStats shader_after = timing_enabled
        ? shader_recompile_cache_stats() : ShaderRecompileCacheStats{};
    const ShaderDecodeCacheStats decode_after = timing_enabled
        ? shader_decode_cache_stats() : ShaderDecodeCacheStats{};
    const DrawRealizationPhaseStats phases_after = timing_enabled
        ? draw_realization_phase_stats() : DrawRealizationPhaseStats{};
    const StageTablePhaseStats table_phases_after = timing_enabled
        ? stage_table_phase_stats() : StageTablePhaseStats{};
    const ParallelDrawRealizationStats parallel_after = timing_enabled
        ? parallel_draw_realization_stats() : ParallelDrawRealizationStats{};
    const auto timing_draws_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    std::vector<ComputeItem> computes = !needs_ordered_realization && g_compute
        ? realize_compute_dispatches(st, submit_no) : std::vector<ComputeItem>{};
    const auto timing_compute_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};

    auto operations = plan_submit_operations(st);
    const auto timing_plan_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    // DMA-bearing submits are captured from semantic state so v14 can retain both endpoint backings
    // even though live execution deliberately realizes their consumers only at ordered positions.
    auto pending_capture = begin_requested_gpu_capture(
        draws, computes, operations, width, height, &st, submit_no,
        static_cast<uint64_t>(st.draws.size()), nullptr,
        /*defer_materialization=*/needs_ordered_realization);
    snapshot_pending_gpu_capture_compute_gds(
        pending_capture.get(), g_compute_gds.data(), g_compute_gds.size());
    OrderedGpustateCaptureTrace capture_trace;
    capture_trace.pending_capture = pending_capture.get();
    OrderedSubmitResult result = needs_ordered_realization
        ? execute_ordered_gpustate(st, width, height, submit_no, g_live, g_compute,
                                   pending_capture ? &capture_trace : nullptr,
                                   can_eagerly_realize_draws ? &draws : nullptr)
        : execute_ordered_items(operations, draws, computes, g_live, g_compute, width, height);
    check_null_image_source_probes(null_image_source_probes, submit_no);
    const auto timing_backend_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const std::vector<uint8_t>& px = result.frame.bytes();

    if (pending_capture) {
        std::string error;
        notify_compute_authority_unknown(
            ComputeAuthorityBoundaryKind::Capture, submit_no);
        omit_noop_dispatches(operations, capture_trace.noops);
        const std::vector<DrawItem>* capture_draws = needs_ordered_realization
            ? &capture_trace.draws : &draws;
        const std::vector<ComputeItem>* capture_computes = needs_ordered_realization
            ? &capture_trace.computes : &computes;
        if (!finish_requested_gpu_capture(std::move(pending_capture), px, error,
                                          capture_draws, capture_computes, &operations, &st,
                                          needs_ordered_realization ? &capture_trace.failures : nullptr))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitEnd, submit_no);
    const bool frame_ready = px.size() == static_cast<size_t>(width) * height * 4;
    // A submit that RENDERED but whose frame does not match the requested extent is dropped here
    // with no trace anywhere: the frontend's own failure log prints only for an EMPTY frame
    // (instrument trap 87), so a non-empty wrong-extent frame is silently discarded and the
    // publish counter simply stops. Sonic Frontiers' publish freeze (#1968) lived in exactly that
    // blind spot through two investigations — the guest kept submitting, the renderer kept
    // producing pixels, and nothing said why nothing reached the screen. Reported unconditionally
    // and rate-limited, because an extent mismatch between the renderer and its caller is never
    // expected. Diagnostic only; the publish decision is unchanged.
    if (publish && !frame_ready && !px.empty()) {
        static std::atomic<int> logged{0};
        const int n = logged.fetch_add(1);
        if (n < 32)
            std::fprintf(stderr,
                         "[agc] PUBLISH DROPPED submit #%llu: the renderer returned %zu bytes for a "
                         "requested %ux%u frame (%zu expected) — nothing is published for this "
                         "submit%s\n",
                         (unsigned long long)submit_no, px.size(), width, height,
                         static_cast<size_t>(width) * height * 4,
                         n == 31 ? " [further reports suppressed]" : "");
    }
    const bool presented = frame_ready && publish;
    if (presented) present_write_frame(result.frame.storage, width, height, result.frame.origin);
    if (timing_enabled) {
        const auto timing_done = TimingClock::now();
        auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        struct TimingTotals {
            uint64_t submits = 0, draws = 0, dispatches = 0, render_spans = 0;
            uint64_t shader_hits = 0, shader_misses = 0, shader_bypasses = 0;
            uint64_t decode_hits = 0, decode_misses = 0, decode_invalidations = 0;
            uint64_t readable_calls = 0, readable_hits = 0, readable_os_probes = 0;
            uint64_t parallel_batches = 0, parallel_draws = 0, parallel_threads = 0;
            double realize_draws = 0, realize_compute = 0, plan = 0, backend = 0, publish = 0;
            double table_build = 0, shader_lookup = 0, shader_compile = 0;
            double table_metadata = 0, table_dynamic_fold = 0, table_resources = 0;
            double parallel_wall = 0;
        };
        static TimingTotals totals;
        static TimingTotals window;
        auto accumulate = [&](TimingTotals& timing) {
            timing.submits++;
            timing.draws += draws.size();
            timing.dispatches += computes.size();
            timing.render_spans += result.render_spans;
            timing.shader_hits += shader_after.hits - shader_before.hits;
            timing.shader_misses += shader_after.misses - shader_before.misses;
            timing.shader_bypasses += shader_after.bypasses - shader_before.bypasses;
            timing.decode_hits += decode_after.hits - decode_before.hits;
            timing.decode_misses += decode_after.misses - decode_before.misses;
            timing.decode_invalidations += decode_after.invalidations - decode_before.invalidations;
            timing.readable_calls += g_guest_readable_cache.calls;
            timing.readable_hits += g_guest_readable_cache.hits;
            timing.readable_os_probes += g_guest_readable_cache.os_probes;
            timing.parallel_batches += parallel_after.batches - parallel_before.batches;
            timing.parallel_draws += parallel_after.semantic_draws - parallel_before.semantic_draws;
            timing.parallel_threads += parallel_after.worker_threads - parallel_before.worker_threads;
            timing.parallel_wall += parallel_after.wall_ms - parallel_before.wall_ms;
            timing.realize_draws += ms(timing_start, timing_draws_ready);
            timing.realize_compute += ms(timing_draws_ready, timing_compute_ready);
            timing.plan += ms(timing_compute_ready, timing_plan_ready);
            timing.backend += ms(timing_plan_ready, timing_backend_done);
            timing.publish += ms(timing_backend_done, timing_done);
            timing.table_build += phases_after.table_ms - phases_before.table_ms;
            timing.table_metadata += table_phases_after.metadata_ms - table_phases_before.metadata_ms;
            timing.table_dynamic_fold += table_phases_after.dynamic_fold_ms -
                                         table_phases_before.dynamic_fold_ms;
            timing.table_resources += table_phases_after.resources_ms - table_phases_before.resources_ms;
            timing.shader_lookup += phases_after.shader_ms - phases_before.shader_ms;
            timing.shader_compile += shader_after.compile_ms - shader_before.compile_ms;
        };
        accumulate(totals);
        accumulate(window);
        if (totals.submits % 25 == 0) {
            const double n = static_cast<double>(totals.submits);
            const double total = totals.realize_draws + totals.realize_compute + totals.plan +
                                 totals.backend + totals.publish;
            std::fprintf(stderr,
                         "[render-timing] submits=%llu draws=%llu dispatches=%llu spans=%llu "
                         "avg_ms: total=%.2f "
                         "realize_draws=%.2f realize_compute=%.2f plan=%.2f backend=%.2f publish=%.2f\n",
                         (unsigned long long)totals.submits, (unsigned long long)totals.draws,
                         (unsigned long long)totals.dispatches,
                         (unsigned long long)totals.render_spans, total / n,
                         totals.realize_draws / n, totals.realize_compute / n, totals.plan / n,
                         totals.backend / n, totals.publish / n);
            const double wn = static_cast<double>(window.submits);
            const double window_total = window.realize_draws + window.realize_compute + window.plan +
                                        window.backend + window.publish;
            std::fprintf(stderr,
                         "[render-window] submits=%llu avg_items: draws=%.1f dispatches=%.1f "
                         "spans=%.1f shaders: hit=%.1f miss=%.1f bypass=%.1f "
                         "decode: hit=%.1f miss=%.1f invalid=%.1f "
                         "readable: hit=%.1f/%.1f os=%.1f "
                         // #2215 trap 143: the WALL fields are printed together and they compose to
                         // `total`; the thread-summed ones are fenced off behind an explicit label.
                         // They used to be interleaved, and three of them are individually LARGER
                         // than `total` (tables=918 against total=464 on one Blue Prince window),
                         // because realization runs on `threads` workers and these are sums across
                         // them. A reader who divided one by `total` got a share that is not a
                         // share -- six of those shipped from one lane in a day, and the other
                         // lane retracted one of the same shape.
                         "avg_ms[wall, sums to total]: total=%.2f realize_draws=%.2f "
                         "realize_compute=%.2f plan=%.2f backend=%.2f publish=%.2f "
                         "| [SUMMED OVER %.1f THREADS -- divide by threads before comparing to "
                         "total]: tables=%.2f shader_lookup=%.2f shader_compile=%.2f "
                         "metadata=%.2f fold=%.2f resources=%.2f "
                         "| parallel[wall]: batches=%.2f draws=%.1f wall=%.2f\n",
                         (unsigned long long)window.submits, window.draws / wn,
                         window.dispatches / wn, window.render_spans / wn,
                         window.shader_hits / wn, window.shader_misses / wn,
                         window.shader_bypasses / wn, window.decode_hits / wn,
                         window.decode_misses / wn, window.decode_invalidations / wn,
                         window.readable_hits / wn, window.readable_calls / wn,
                         window.readable_os_probes / wn, window_total / wn,
                         window.realize_draws / wn,
                         window.realize_compute / wn, window.plan / wn, window.backend / wn,
                         window.publish / wn,
                         // 1.0, not 0.0, when no parallel batch ran: realization was then SERIAL,
                         // so these sums already ARE wall and dividing by the printed figure is
                         // still the right instruction. Printing 0.0 told the reader to divide by
                         // zero, which is worse than the ambiguity this line was fixed to remove.
                         window.parallel_batches
                             ? static_cast<double>(window.parallel_threads) /
                                   static_cast<double>(window.parallel_batches)
                             : 1.0,
                         window.table_build / wn,
                         window.shader_lookup / wn, window.shader_compile / wn,
                         window.table_metadata / wn, window.table_dynamic_fold / wn,
                         window.table_resources / wn,
                         window.parallel_batches / wn, window.parallel_draws / wn,
                         window.parallel_wall / wn);
            window = {};
        }
    }
    return presented;
}

bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height, bool publish) {
    if (!g_live || st.draws.empty() || !width || !height) return false;
    // Same extent contract as the ordered path: this frame is checked against width*height*4 below
    // before it can be published (#1986).
    PresentSubmitScope present_submit_scope;
    // Bind the target dimensions and defer to the pure core, which recompiles the shaders from their
    // SHADER_PGM addresses and resolves fixed-function state before calling back into the live renderer.
    // Scale the guest viewport to our framebuffer: `width`/`height` are the render target (reduced by
    // PROSPER_RENDER_SCALE), while the guest programs its viewport in full present-resolution pixels — so
    // without this a 1/N render shows only the bottom-left 1/N of the frame.
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width  / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    std::vector<DrawItem> items = realize_gpustate_draws(
        st, 0x10000, sx, sy, nullptr, true);
    if (items.empty()) return false;
    std::vector<SubmitOperation> operations;
    operations.reserve(items.size());
    for (const auto& item : items)
        operations.push_back({SubmitOperationKind::Draw,
                              static_cast<size_t>(item.draw_index), item.command_order});
    auto pending = begin_requested_gpu_capture(items, {}, operations, width, height);
    RenderedFrame rendered = g_live(items, width, height);
    if (pending) {
        std::string error;
        if (!finish_requested_gpu_capture(std::move(pending), rendered.bytes(), error))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    if (rendered.size() != static_cast<size_t>(width) * height * 4) return false;
    if (!publish) return false;
    present_write_frame(rendered.storage, width, height, rendered.origin);
    return true;
}

} // namespace prosper::gpu
