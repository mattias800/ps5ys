// CPU-only equivalence and allocation regressions for the two per-draw shader property scans.
// Keep the vector references independent of the production streaming implementations. No cache,
// Vulkan device, guest mapping, diagnostic output, or elapsed-time threshold participates here.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

using namespace prosper::gpu;

namespace {
thread_local bool count_allocations = false;
thread_local size_t allocations = 0;
int failures = 0;
size_t checks = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { ++failures; std::printf("[FAIL] %s\n", label); }
}
} // namespace

// Rdna2Inst vectors use ordinary (not over-aligned) new. Count only while the single-threaded
// public property call is active; fixture/reference allocations and printf stay outside the scope.
static_assert(alignof(Rdna2Inst) <= alignof(std::max_align_t));
void* operator new(std::size_t size) {
    if (count_allocations) ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
uint32_t vector_export_mask(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    uint32_t packed = 0;
    std::array<bool, kFragmentColorOutputs> realized{};
    for (const auto& in : instructions) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP || in.exp_target >= realized.size() ||
            realized[in.exp_target] || in.exp_en == 0)
            continue;
        packed |= (in.exp_en & 0xfu) << (in.exp_target * 4u);
        realized[in.exp_target] = true;
    }
    return packed;
}

VertexPrologInfo vector_prolog(const uint32_t* code, size_t dwords) {
    VertexPrologInfo result;
    if (!code || !dwords) return result;
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    for (const auto& in : instructions) {
        if (in.is_end || in.fmt == Rdna2Format::EXP || in.fmt == Rdna2Format::Unknown)
            return {};
        if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x20) continue;
        if (in.n_src != 1 || in.src[0].kind != OperandKind::SGPR ||
            in.src[0].value != 6 || in.len_dwords != 1)
            return {};
        result.valid = in.pc != 0;
        result.setpc_pc = in.pc;
        result.prefix_dwords = in.pc;
        break;
    }
    if (!result.valid) return {};
    for (const auto& in : instructions) {
        if (in.pc >= result.setpc_pc) break;
        if (in.fmt != Rdna2Format::SOPP ||
            (in.opcode != 0x02 && (in.opcode < 0x04 || in.opcode > 0x09)))
            continue;
        const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
        if (target < 0 || target > result.setpc_pc) return {};
    }
    return result;
}

bool same_prolog(const VertexPrologInfo& a, const VertexPrologInfo& b) {
    return a.valid == b.valid && a.setpc_pc == b.setpc_pc &&
           a.prefix_dwords == b.prefix_dwords;
}

void compare_prefixes(const uint32_t* code, size_t dwords, const char* label) {
    for (size_t length = 0; length <= dwords; ++length) {
        const uint32_t expected_mask = vector_export_mask(code, length);
        const VertexPrologInfo expected_prolog = vector_prolog(code, length);
        allocations = 0;
        count_allocations = true;
        const uint32_t actual_mask = fragment_color_export_mask(code, length);
        count_allocations = false;
        const size_t mask_allocations = allocations;
        allocations = 0;
        count_allocations = true;
        const VertexPrologInfo actual_prolog = rdna2_vertex_prolog_info(code, length);
        count_allocations = false;
        const size_t prolog_allocations = allocations;
        const bool equivalent = actual_mask == expected_mask &&
                                same_prolog(actual_prolog, expected_prolog);
        const bool allocation_free = mask_allocations == 0 && prolog_allocations == 0;
        if (!equivalent || !allocation_free)
            std::printf("  case=%s length=%zu mask=%08x/%08x allocations=%zu/%zu\n",
                        label, length, actual_mask, expected_mask,
                        mask_allocations, prolog_allocations);
        check(equivalent, "both properties match vector reference at this supplied-length prefix");
        check(allocation_free, "normal property scans allocate no instruction vectors");
    }
}

constexpr uint32_t nop = 0xBF800000u;
constexpr uint32_t endpgm = 0xBF810000u;
constexpr uint32_t transfer = 0xBE802006u;
constexpr uint32_t unknown = 0xffffffffu;
constexpr uint32_t exp_word(uint32_t target, uint32_t en) {
    return 0xF8001000u | (target << 4u) | en;
}
constexpr uint32_t branch(uint32_t opcode, int displacement) {
    return 0xBF800000u | (opcode << 16u) | static_cast<uint16_t>(displacement);
}
} // namespace

int main() {
    // The allocation contract deliberately excludes the optional diagnostic's deduplicating set.
#ifdef _WIN32
    _putenv_s("PROSPER_PROLOGLOG", "");
#else
    unsetenv("PROSPER_PROLOGLOG");
#endif
    check(rdna2_decode_one(&unknown, 1).fmt == Rdna2Format::Unknown,
          "unknown terminator control really decodes as Unknown");
    const uint32_t allocation_control[] = {nop, endpgm};
    allocations = 0;
    count_allocations = true;
    const uint32_t control_mask = vector_export_mask(allocation_control, 2);
    count_allocations = false;
    check(allocations > 0 && control_mask == 0,
          "old vector reference positively exercises the allocation counter");

    compare_prefixes(nullptr, 0, "empty");
    check(!rdna2_vertex_prolog_info(nullptr, 7).valid, "null prolog remains invalid");
    const std::vector<std::vector<uint32_t>> corpus = {
        {nop}, {endpgm, transfer}, {unknown, transfer}, {transfer},
        {nop, transfer, unknown}, {nop, transfer, branch(2, 32767), endpgm},
        {nop, 0xBE802008u, transfer}, // wrong SGPR pair before a later otherwise-valid transfer
        {nop, 0xBE802080u, transfer}, // inline constant, not the reserved SGPR pair
        {nop, exp_word(0, 0), 0, transfer}, // even a zero-mask export rejects a prolog
        {exp_word(0, 3), 0, endpgm, exp_word(7, 15), 0},
        {exp_word(0, 3), 0, unknown, exp_word(7, 15), 0},
        {exp_word(8, 15), 0, exp_word(7, 10), 0, endpgm},
        {exp_word(0, 0), 0, exp_word(0, 3), 0, exp_word(0, 12), 0,
         exp_word(7, 0), 0, exp_word(7, 10), 0, endpgm},
        {0xBE8003FFu, 0x12345678u, transfer}, // literal scalar instruction, prefixes split it
        {0xE00C2000u, 0x80010000u, nop, transfer}, // two-word buffer instruction
        {branch(2, -2), branch(4, 1), nop, transfer}, // bad minimum followed by a valid target
        {branch(2, 4), branch(4, 1), nop, transfer}, // bad maximum followed by a valid target
        {branch(2, 2), branch(4, -2), nop, transfer}, // both boundary targets 3 and 0
        {branch(3, 32767), nop, transfer}, // opcode outside the exact direct-branch set
    };
    for (const auto& code : corpus) compare_prefixes(code.data(), code.size(), "fixed corpus");

    const auto& duplicate_exports = corpus[12];
    check(fragment_color_export_mask(duplicate_exports.data(), duplicate_exports.size()) ==
              0xa0000003u,
          "first nonzero export wins independently for MRT0 and MRT7");
    check(fragment_color_export_mask(corpus[10].data(), corpus[10].size()) == 3,
          "Unknown prevents a later MRT7 export from affecting the mask");
    check(!rdna2_vertex_prolog_info(corpus[15].data(), corpus[15].size()).valid &&
          !rdna2_vertex_prolog_info(corpus[16].data(), corpus[16].size()).valid &&
          rdna2_vertex_prolog_info(corpus[17].data(), corpus[17].size()).valid,
          "branch minimum and maximum each reject independently, both boundaries are accepted");

    for (uint32_t opcode : {2u, 4u, 5u, 6u, 7u, 8u, 9u}) {
        for (int displacement : {-32768, -3, -2, -1, 0, 1, 2, 32767}) {
            const uint32_t code[] = {branch(opcode, displacement), nop, transfer};
            compare_prefixes(code, 3, "signed branch bounds");
            const auto info = rdna2_vertex_prolog_info(code, 3);
            const bool valid = displacement >= -1 && displacement <= 1;
            check(info.valid == valid && (!valid ||
                      (info.setpc_pc == 2 && info.prefix_dwords == 2)),
                  "every recognized direct-branch opcode honors signed prefix bounds");
        }
    }

    // Reuse precisely the same address/length while changing both property results. No address-only
    // memoization can pass this arm. Fixture construction and mutations precede allocation scopes.
    uint32_t mutable_code[] = {nop, transfer, endpgm, endpgm};
    compare_prefixes(mutable_code, 4, "mutation before");
    check(rdna2_vertex_prolog_info(mutable_code, 4).valid, "initial mutable prolog is valid");
    mutable_code[1] = exp_word(7, 5);
    mutable_code[2] = 0;
    compare_prefixes(mutable_code, 4, "mutation after");
    check(!rdna2_vertex_prolog_info(mutable_code, 4).valid &&
          fragment_color_export_mask(mutable_code, 4) == 0x50000000u,
          "same-address mutation immediately changes prolog and MRT7 properties");
    mutable_code[1] = transfer;
    mutable_code[2] = endpgm;
    compare_prefixes(mutable_code, 4, "mutation restored");
    check(rdna2_vertex_prolog_info(mutable_code, 4).valid &&
          fragment_color_export_mask(mutable_code, 4) == 0,
          "restoring bytes restores both properties without stale state");

    std::printf("shader property scans: %zu checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
