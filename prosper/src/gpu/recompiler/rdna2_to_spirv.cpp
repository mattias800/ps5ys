// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {

namespace {


} // namespace

void record_recompile_reject_reason_for_test(const RecompileDiagnosticContext& diagnostic,
                                             const char* tag, const char* role,
                                             const char* payload) {
    log_recompile_diagnostic(diagnostic, tag, role, "%s", payload);
}

std::string last_terminal_reject_reason(uint64_t program_address) {
    std::lock_guard lock(terminal_reject_mutex());
    const auto& reasons = terminal_reject_reasons();
    const auto found = reasons.find(program_address);
    return found == reasons.end() ? std::string() : found->second;
}

void log_compute_recompile_skip_diagnostic(const RecompileDiagnosticContext& diagnostic) {
    log_recompile_diagnostic(diagnostic, "compute-recompile-reject", "consequent",
                             "reason=empty-result dispatch-skipped");
}


FragmentInterpolationLayout::FragmentInterpolationLayout() {
    for (auto& locations : parameter_locations) locations.fill(kUnusedLocation);
    system_locations.fill(kUnusedLocation);
}

namespace {


// The f16 bit pattern an inline float constant supplies in a 16-bit operand position (ISA Table 10
// lists per-width encodings: "0.5 ... half: 0x3800" etc.). Only 1/(2*pi) (code 248, 0x3118) differs
// from rounding the f32 value — the f32 table entry 0.15915494 would round to a different last bit
// than the documented operand, so 16-bit consumers must use these bits, not the f32 constant.

}  // namespace

// Extend the narrow compiler shape
//
//   s_cbranch_scc* ELSE
//   THEN...
//   s_endpgm
// ELSE:
//   ELSE...
//   s_endpgm
//
// into the existing structured-if/else input. The first s_endpgm is represented as a synthetic
// s_branch to the second end, which is semantically exact: it skips the other terminating arm, while
// the shared SPIR-V shell may still converge solely to publish outputs/return. Keep this deliberately
// conservative: one scalar conditional, adjacent straight-line else arm, and a real terminating end.
// Immediate-end/nop-end tails remain the established early-out shape and are not rewritten.
static bool extend_terminating_if_else(const uint32_t* code, size_t dwords,
                                       std::vector<Rdna2Inst>& instructions,
                                       size_t* required_dwords = nullptr,
                                       uint32_t* synthetic_branch_pc = nullptr) {
    if (synthetic_branch_pc) *synthetic_branch_pc = UINT32_MAX;
    if (!code || instructions.empty()) return false;
    auto first_end = std::find_if(instructions.begin(), instructions.end(),
                                  [](const Rdna2Inst& in) { return in.is_end; });
    if (first_end == instructions.end()) return false;

    const Rdna2Inst* branch = nullptr;
    for (auto it = instructions.begin(); it != first_end; ++it) {
        if (it->fmt != Rdna2Format::SOPP) continue;
        if (it->opcode == 0x04 || it->opcode == 0x05) {
            if (branch) return false;
            branch = &*it;
        } else if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) {
            return false;
        }
    }
    if (!branch || branch->simm16 <= 0) return false;
    const int64_t target64 = static_cast<int64_t>(branch->pc) + branch->len_dwords + branch->simm16;
    if (target64 < 0 || static_cast<uint64_t>(target64) >= dwords) return false;
    const uint32_t target = static_cast<uint32_t>(target64);
    if (target != first_end->pc + first_end->len_dwords) return false;

    std::vector<Rdna2Inst> tail;
    const size_t tail_dwords = rdna2_walk(code + target, dwords - target, tail);
    if (tail.empty() || !tail.back().is_end) return false;
    bool has_real_else = false;
    for (auto it = tail.begin(); it != tail.end() - 1; ++it) {
        if (it->fmt == Rdna2Format::SOPP) {
            if (it->opcode == 0x00) continue; // padding is harmless but not a real else arm
            if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) return false;
        }
        has_real_else = true;
    }
    if (!has_real_else) return false;

    for (auto& in : tail) in.pc += target;
    const uint32_t merge_pc = tail.back().pc;
    const uint32_t skip_dwords = merge_pc - (first_end->pc + first_end->len_dwords);
    if (!skip_dwords || skip_dwords > static_cast<uint32_t>(INT16_MAX)) return false;
    first_end->fmt = Rdna2Format::SOPP;
    first_end->opcode = 0x02; // s_branch merge_pc: terminates the lexical then arm
    first_end->simm16 = static_cast<int32_t>(skip_dwords);
    first_end->words[0] = 0xbf820000u | (skip_dwords & 0xffffu);
    first_end->is_end = false;
    if (synthetic_branch_pc) *synthetic_branch_pc = first_end->pc;
    instructions.insert(instructions.end(), tail.begin(), tail.end());
    if (required_dwords) *required_dwords = target + tail_dwords;
    return true;
}

FragmentInterpolationLayout fragment_interpolation_layout(
        const uint32_t* code, size_t dwords,
        const PixelSystemInputMapping* system_inputs,
        const PixelInputMapping* pixel_inputs) {
    FragmentInterpolationLayout layout;
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    std::array<uint8_t, 32> selectors{};
    uint32_t highest_attribute = 0;
    bool has_attribute = false;
    for (const auto& instruction : instructions) {
        if (instruction.is_end) break;
        if (instruction.fmt != Rdna2Format::VINTRP || instruction.vintrp_attr >= 32) continue;
        const uint32_t attr = instruction.vintrp_attr;
        layout.attribute_mask |= 1u << attr;
        highest_attribute = std::max(highest_attribute, attr);
        has_attribute = true;
        if (instruction.opcode == 0 || instruction.opcode == 1)
            layout.smooth_mask |= 1u << attr;
        else if (instruction.opcode == 2 && instruction.src[0].value < 3)
            selectors[attr] |= static_cast<uint8_t>(1u << instruction.src[0].value);
    }
    if (pixel_inputs) {
        layout.passthrough_mask =
            pixel_inputs->effective_passthrough_mask() & layout.attribute_mask;
        // #3051: the guest's own FLAT_SHADE control bit, independent of which VINTRP opcode the
        // shader used to read the attribute (see PixelInputMapping::effective_flat_mask()). Computed
        // here, before the requires_geometry early-out below, because it must apply to every
        // fragment program -- not only ones that also need the portable interpolation-geometry
        // fallback.
        layout.flat_mask = pixel_inputs->effective_flat_mask() & layout.attribute_mask;
    }

    // P10/P20 have no ordinary Vulkan varying equivalent. P0 can retain the cheap Flat-input path
    // when it is the attribute's only interpolation mode; mixed P0+smooth needs the geometry copy too.
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (selectors[attr] & 0x3u) layout.requires_geometry = true;
        if ((selectors[attr] & 0x4u) && (layout.smooth_mask & (1u << attr)))
            layout.requires_geometry = true;
    }
    if (!layout.requires_geometry) return layout;

    uint32_t location = has_attribute ? highest_attribute + 1 : 0;
    for (uint32_t attr = 0; attr < 32; ++attr) {
        for (uint32_t selector = 0; selector < 3; ++selector) {
            if (!(selectors[attr] & (1u << selector))) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.parameter_locations[attr][selector] = location++;
        }
    }
    if (system_inputs) {
        for (uint32_t field = 0; field < 7; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit) || !(system_inputs->ena & bit)) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.system_locations[field] = location++;
        }
    }
    return layout;
}

uint32_t fragment_consumed_attribute_mask(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    // Decoded instruction by instruction rather than through `rdna2_walk`, and this is the whole
    // point of the function. `rdna2_walk` stops at the first `is_end` AND at the first Unknown
    // encoding (rdna2_decode.cpp), which is exactly the prefix `fragment_interpolation_layout`
    // already sees -- so routing through it would make this mask EQUAL to `attribute_mask`, not a
    // superset, and the margin the caller's safety argument depends on would be zero. Stepping past
    // both terminators is what buys the margin.
    //
    // The bias is deliberate and one-directional. Decoding past a terminator can decode data as
    // instructions, so a spurious VINTRP hit is possible; that costs one dead output varying.
    // MISSING an attribute would hand the fragment stage an input the vertex stage no longer
    // exports, which is the regression this analysis must never cause. Over-report, never under.
    const size_t span = rdna2_recompile_code_span(code, dwords);
    uint32_t mask = 0;
    for (size_t pc = 0; pc < span;) {
        const Rdna2Inst instruction = rdna2_decode_one(code + pc, span - pc);
        if (instruction.fmt == Rdna2Format::VINTRP && instruction.vintrp_attr < 32)
            mask |= 1u << instruction.vintrp_attr;
        if (!instruction.len_dwords) break;   // safety: never advance 0
        pc += instruction.len_dwords;
    }
    return mask;
}

bool dead_varying_elimination_enabled() {
    static const bool disabled = getenv("PROSPER_NO_DEAD_VARYING_ELIM") != nullptr;
    return !disabled;
}

void apply_fragment_consumption(PixelInputMapping& mapping,
                                const uint32_t* fragment_code, size_t dwords) {
    if (!dead_varying_elimination_enabled() || !mapping.valid_mask || !fragment_code || !dwords)
        return;
    mapping.consumed_mask = fragment_consumed_attribute_mask(fragment_code, dwords);
    mapping.consumed_known = true;
}

std::vector<uint32_t> recompile_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_position,
        bool synthesize_rect) {
    SpirvCompute builder;
    return builder.build_interpolation_geometry(layout, capture_position, synthesize_rect);
}

namespace {
}

// #2418: does this shader read SCC anywhere? Used to decide whether the fragment stage should pay for
// an exact wave vote when a mask op writes SCC. Deliberately CONSERVATIVE and whole-shader: it answers
// "could SCC ever be consumed", not "is this particular write live". A false positive costs one shader a
// subgroup-size requirement it might not have needed; a false NEGATIVE would silently drop the vote a
// consumer depends on, so every reader is listed even where the recompiler handles it elsewhere.
//
// SCC readers on RDNA2 (doc 70648): SOP2 s_cselect_b32/b64 (0x0a/0x0b) and the carry-in forms
// s_addc_u32/s_subb_u32 (0x04/0x05); SOPP s_cbranch_scc0/scc1 (0x04/0x05); SOP1 s_cmov_b32/b64
// (0x02/0x03); SOPK s_cmovk_i32 (0x02). SOPC and s_cmp_*/s_cmpk_* WRITE SCC and are not readers.
//
// SOPK was omitted in the first version of this scan and that was a genuine false negative -- the
// direction this function's own contract calls the dangerous one, since a missed reader silently
// drops the vote its consumer depends on. Caught in review of #2416. The file already documents the
// hazard at the `sopk_writes_scalar_data` exclusion note: "several SOPK ops (s_addk/s_mulk/s_cmovk/
// s_cmpk) READ or read-modify-write their dst via the implicit SIMM16".
//
// Only s_cmovk_i32 READS SCC in this space, verified rather than taken from a table --
// `llvm-mc -mcpu=gfx1030 -disassemble` over the SOPK opcode range gives 0x00 s_movk_i32,
// 0x02 s_cmovk_i32, 0x03.. s_cmpk_* (which WRITE SCC), 0x0f s_addk_i32, 0x10 s_mulk_i32,
// 0x12/0x13 s_getreg/s_setreg. Including the whole SOPK format would be conservative in the safe
// direction but costs every shader containing an s_movk -- which is nearly all of them -- a
// subgroup-size requirement it does not need.
inline bool shader_reads_scc(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        switch (in.fmt) {
            case Rdna2Format::SOP2:
                if (in.opcode == 0x04 || in.opcode == 0x05 ||
                    in.opcode == 0x0a || in.opcode == 0x0b) return true;
                break;
            case Rdna2Format::SOPP:
                if (in.opcode == 0x04 || in.opcode == 0x05) return true;
                break;
            case Rdna2Format::SOP1:
                if (in.opcode == 0x02 || in.opcode == 0x03) return true;
                break;
            case Rdna2Format::SOPK:
                if (in.opcode == 0x02) return true;   // s_cmovk_i32: conditional move ON SCC
                break;
            default: break;
        }
    }
    return false;
}

std::unordered_set<uint32_t> safe_execz_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> safe;
    // pc of the terminating s_endpgm — a forward execz whose target is here skips straight to the end.
    uint32_t end_pc = 0; bool have_end = false;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; have_end = true; break; }
    for (const auto& br : ins) {
        if (br.fmt != Rdna2Format::SOPP || br.opcode != 0x08 || br.simm16 <= 0) continue;
        const uint32_t target = br.pc + br.len_dwords + (uint32_t)br.simm16;
        bool ok = target > br.pc;
        bool target_found = false;
        for (const auto& in : ins) {
            if (in.pc == target) { target_found = true; break; }
        }
        if (!target_found) ok = false;
        // LOOP EXIT (#1183): if a backward branch that jumps to at-or-before this execz sits inside the
        // block [br, target), then this execz is a data-dependent loop's EXIT, not an if/guard-to-end —
        // even when its target happens to be s_endpgm. Linearizing it here would strand the loop's
        // back-edge as a straight-line reject; instead leave it OUT of `safe` so detect_divergent_loops
        // claims it and emit_divloop reconstructs the structured loop. Back-edges are the shapes that
        // detector recognizes: unconditional s_branch (0x02) or s_cbranch_execnz (0x09).
        bool is_loop_exit = false;
        for (const auto& bb : ins) {
            if (bb.fmt != Rdna2Format::SOPP || (bb.opcode != 0x02 && bb.opcode != 0x09)) continue;
            if (bb.pc <= br.pc || bb.pc >= target) continue;                 // must sit inside this block
            const int64_t bt = static_cast<int64_t>(bb.pc) + bb.len_dwords + bb.simm16;   // signed target
            if (bt <= static_cast<int64_t>(br.pc)) { is_loop_exit = true; break; }
        }
        if (is_loop_exit) continue;
        // Two shapes are safe to linearize (drop the branch, run the block under per-lane EXEC):
        //  * IF/ENDIF rejoining live code (target < end): only EXEC-predicated VGPR writes are safe,
        //    since scalar/VCC/memory writes past the merge would be observed by later code.
        //  * GUARD-TO-END (target == s_endpgm): the block's scalar/SGPR/VCC writes are DEAD (nothing
        //    runs after s_endpgm) and wave-uniform, memory loads are fault-free (robustBufferAccess) into
        //    predicated VGPRs, and memory stores are EXEC-predicated (conditional store). So anything is
        //    safe EXCEPT an EXP export (not EXEC-predicated → would export from inactive lanes).
        const bool guard_to_end = have_end && target >= end_pc;
        for (const auto& in : ins) {
            if (in.pc <= br.pc || in.pc >= target || in.is_end) continue;
            if (in.fmt == Rdna2Format::EXP) { ok = false; break; }   // exports are never EXEC-masked
            if (guard_to_end) continue;                              // dead-at-end / predicated / fault-free
            // IF/ENDIF rejoining live code: safe to linearize iff every write is EXEC-predicated. VGPR ALU
            // (VOP1/2/3) predicates its writes; memory ops (MIMG/MUBUF/DS) predicate their VGPR loads (and
            // are fault-free under robust access) and EXEC-predicate their stores. SGPR/VCC writes (SOP*/
            // VOPC/SMEM) are NOT predicated and would be observed past the merge -> still unsafe.
            // CARVE-OUTS: three op groups inside the "safe" VALU formats have UNPREDICATED scalar side
            // effects (emit_alu writes rs.sreg/rs.vcc/rs.sreg_bool for them, and predicate_write only
            // covers VGPRs) — on hardware the skipped block would have preserved VCC/the SGPR:
            //   VOP1 0x02 v_readfirstlane_b32 (writes an SGPR), VOP2 0x28-0x2A carry ops (write VCC),
            //   VOP3B 0x128-0x12A and 0x30F/0x310/0x319 (write the carry-out SGPR pair/VCC), and
            //   VOP3B v_mad_u64_u32 0x176 (its 65th-bit carry mask also lands in VCC/an SGPR pair
            //   unpredicated).
            const bool scalar_side_effect =
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A) ||
                vop3b_fresh_carry_output(in) ||
                (in.fmt == Rdna2Format::VOP3 &&
                 ((in.opcode >= 0x128 && in.opcode <= 0x12A) || in.opcode == 0x176));
            if (!scalar_side_effect &&
                (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                 in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF ||
                 in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::DS ||
                 in.fmt == Rdna2Format::FLAT ||
                 sopp_is_noop(in))) {
                continue;
            }
            if (scalar_side_effect) { ok = false; break; }
            // A pure scalar move (s_mov_b32/b64: no SCC/VCC/memory side effect) whose destination SGPR is
            // DEAD at the merge is also safe to linearize: the unconditional write is overwritten before any
            // later read, so masked-off lanes never observe it. (The tonemap/sRGB divergent-ifs load a scalar
            // constant used only within the block and reset it right after — see shader 033.)
            if (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x03 || in.opcode == 0x04) &&
                in.dst.kind == OperandKind::SGPR) {
                const bool b64 = (in.opcode == 0x04);
                // dst must be a plain SGPR (s0..s105). SOP1 destinations also decode EXEC/VCC/M0
                // (106/107/124/126/127) as SGPR-kind, but a move into those has wave-wide side effects read
                // implicitly (not via an SGPR operand) — they can never be proven dead, so exclude them.
                const int hi = in.dst.value + (b64 ? 1 : 0);
                if (hi <= 105 && sgpr_dead_at_merge(ins, target, in.dst.value) &&
                    (!b64 || sgpr_dead_at_merge(ins, target, in.dst.value + 1))) continue;
            }
            // A scalar LOAD (s_load/s_buffer_load) inside the block is likewise safe iff every SGPR it
            // writes is DEAD at the merge: the load itself is wave-uniform and fault-free, and a result
            // overwritten before any read is never observed by post-merge code. This is the divergent
            // lighting/fog block's `s_buffer_load_dwordx2 vcc, …` scratch-load shape (DOLL VS, #273);
            // sgpr_dead_at_merge understands VCC's implicit readers, so vcc-targeted loads qualify.
            if (in.fmt == Rdna2Format::SMEM) {
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;   default: break;
                }
                if (n) {
                    bool all_dead = true;
                    for (uint32_t k = 0; k < n && all_dead; k++)
                        all_dead = sgpr_dead_at_merge(ins, target, in.dst.value + (int)k);
                    if (all_dead) continue;
                }
            }
            ok = false;
            break;
        }
        if (ok) safe.insert(br.pc);
    }
    return safe;
}

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
}

// FRAGMENT alpha-test / clip() discard via a SCALAR BRANCH. A per-lane condition (v_cmp -> VCC) is folded
// into a saved-EXEC survivor mask by a 64-bit wave-mask op (s_and/s_andn2_b64 sDST,sDST,vcc — which on HW
// sets SCC = "any lane survives"), then `s_cbranch_scc0 <fwd>` skips the shading when NO lane survives; the
// block then narrows EXEC (s_wqm exec, sDST) and shades, and the export lowers to an OpKill of the failed
// lanes. Per-invocation the wave early-out is a pure optimization — running the block for a lane that will
// be OpKill'd at export is harmless — so the branch is safe to LINEARIZE (drop it, run the block straight-
// line) exactly like a forward s_cbranch_execz. Recognize it by the mask op IMMEDIATELY preceding the
// branch (hints ignored). Returns the pc of each such branch. This is the shape of every Unity cutout /
// text draw; rejecting the branch dropped all of them (The Messenger's missing cutscene text, #102).
std::unordered_set<uint32_t> mask_test_branches(const std::vector<Rdna2Inst>& ins,
                                                bool allow_b32_masks = false) {
    std::unordered_set<uint32_t> out;
    std::unordered_set<uint32_t> b32_mask_writer_pcs;
    std::unordered_set<uint32_t> block_entries;
    size_t active_count = 0;
    while (active_count < ins.size() && !ins[active_count].is_end) ++active_count;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < active_count; ++i) index_by_pc.emplace(ins[i].pc, i);
    auto is_scalar_branch = [](const Rdna2Inst& candidate) {
        return candidate.fmt == Rdna2Format::SOPP &&
            (candidate.opcode == 0x02 ||
             (candidate.opcode >= 0x04 && candidate.opcode <= 0x09));
    };
    auto branch_target_pc = [](const Rdna2Inst& candidate) -> int64_t {
        return static_cast<int64_t>(candidate.pc) + candidate.len_dwords + candidate.simm16;
    };
    for (size_t i = 0; i < active_count; ++i) {
        if (!is_scalar_branch(ins[i])) continue;
        const int64_t target = branch_target_pc(ins[i]);
        if (target >= 0 && target <= UINT32_MAX)
            block_entries.insert(static_cast<uint32_t>(target));
    }

    // Transfer one instruction's one-word Wave32-mask provenance. The domain is a MUST fact: an
    // SGPR is present only when every path reaching this instruction proves that it contains a
    // complete Wave32 lane mask. This preserves masks through a conditional fall-through (the live
    // Astro alpha-test preamble) while intersecting away a compare performed on only one predecessor.
    auto transfer_b32_masks = [&](const Rdna2Inst& in,
                                  const std::unordered_set<int>& incoming,
                                  bool* mask_writer) {
        std::unordered_set<int> masks = incoming;
        auto tracked_mask_source = [&](const Operand& source) {
            return source.value == 126 ||
                   ((source.kind == OperandKind::SGPR ||
                     source.kind == OperandKind::Special) &&
                    incoming.contains(source.value));
        };
        bool writes_b32_mask = false;
        int mask_dst = -1;
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            mask_dst = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
            writes_b32_mask = true;
        } else if (in.fmt == Rdna2Format::SOP1 &&
                   (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                    sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
            writes_b32_mask = tracked_mask_source(in.src[0]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::SOP2 &&
                   (in.opcode == 0x0a ||
                    (in.opcode >= 0x0e && in.opcode <= 0x1c &&
                     (in.opcode & 1u) == 0))) {
            const bool real_mask_source = tracked_mask_source(in.src[0]) ||
                                          tracked_mask_source(in.src[1]);
            auto representable = [&](const Operand& source) {
                return tracked_mask_source(source) || source.kind == OperandKind::InlineInt;
            };
            writes_b32_mask = real_mask_source && representable(in.src[0]) &&
                              representable(in.src[1]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::VOP3 &&
                   in.opcode >= 0x128 && in.opcode <= 0x12a &&
                   in.sdst.kind == OperandKind::SGPR) {
            writes_b32_mask = tracked_mask_source(in.src[2]);
            mask_dst = in.sdst.value;
        } else if (vop3b_fresh_carry_output(in)) {
            writes_b32_mask = true;
            mask_dst = in.sdst.value;
        }

        auto erase_written_words = [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                masks.erase(base + static_cast<int>(word));
        };
        const uint32_t dst_width = scalar_write_width(in);
        if (dst_width) erase_written_words(in.dst.value, dst_width);
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
            erase_written_words(in.dst.value, 1);
        if (vop3_writes_mask_sdst(in))
            erase_written_words(in.sdst.value,
                ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
                 vop3b_fresh_carry_output(in)) ? 1u : 2u);
        if (writes_b32_mask && mask_dst >= 0 && mask_dst != 126)
            masks.insert(mask_dst);
        if (mask_writer) *mask_writer = writes_b32_mask && mask_dst >= 0 && mask_dst != 126;
        return masks;
    };

    if (allow_b32_masks && active_count) {
        std::vector<std::unordered_set<int>> incoming(active_count);
        std::vector<bool> reachable(active_count, false);
        std::vector<size_t> worklist{0};
        reachable[0] = true;
        auto merge_into = [&](size_t successor, const std::unordered_set<int>& masks) {
            if (!reachable[successor]) {
                incoming[successor] = masks;
                reachable[successor] = true;
                worklist.push_back(successor);
                return;
            }
            std::unordered_set<int> intersection;
            for (int reg : incoming[successor])
                if (masks.contains(reg)) intersection.insert(reg);
            if (intersection != incoming[successor]) {
                incoming[successor] = std::move(intersection);
                worklist.push_back(successor);
            }
        };
        for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
            const size_t i = worklist[cursor];
            const auto outgoing = transfer_b32_masks(ins[i], incoming[i], nullptr);
            auto merge_pc = [&](int64_t pc) {
                if (pc < 0 || pc > UINT32_MAX) return;
                const auto found = index_by_pc.find(static_cast<uint32_t>(pc));
                if (found != index_by_pc.end()) merge_into(found->second, outgoing);
            };
            if (is_scalar_branch(ins[i])) {
                merge_pc(branch_target_pc(ins[i]));
                if (ins[i].opcode != 0x02 && i + 1 < active_count)
                    merge_into(i + 1, outgoing);
            } else if (i + 1 < active_count) {
                merge_into(i + 1, outgoing);
            }
        }
        for (size_t i = 0; i < active_count; ++i) {
            if (!reachable[i]) continue;
            bool writes_mask = false;
            (void)transfer_b32_masks(ins[i], incoming[i], &writes_mask);
            if (writes_mask) b32_mask_writer_pcs.insert(ins[i].pc);
        }
    }

    const Rdna2Inst* prev = nullptr;
    for (const auto& in : ins) {
        if (block_entries.contains(in.pc)) prev = nullptr;
        if (in.is_end) break;
        if (sopp_is_noop(in)) continue;                         // hints don't break the mask->branch pairing
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05) && in.simm16 > 0) {
            // scc0/scc1 FORWARD branch whose SCC was set by a 64-bit wave-mask op: SOP2 s_and_b64(0x0f) /
            // s_or_b64(0x11) / s_xor_b64(0x13) / s_andn2_b64(0x15), or SOP1 s_and/or_saveexec_b64 (0x24/0x25),
            // writing a plain SGPR-pair kill mask. (A branch on a v_cmp/s_cmp SCC is a REAL uniform-if and is
            // NOT matched — prev would be a SOPC/ALU, not a mask op.)
            if (prev) {
                const bool b64_mask_sop2 = prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0f || prev->opcode == 0x11 ||
                     prev->opcode == 0x13 || prev->opcode == 0x15);
                // The same compiler alpha-test shape uses the one-word logical family when
                // SPI_PS_IN_CONTROL proves Wave32. The B32 result is then the complete wave mask,
                // and SCC is the same whole-wave early-out vote as for the B64 form.
                const bool b32_mask_sop2 = allow_b32_masks && prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0e || prev->opcode == 0x10 ||
                     prev->opcode == 0x12 || prev->opcode == 0x14) &&
                    b32_mask_writer_pcs.contains(prev->pc);
                const bool mask_sop2 = b64_mask_sop2 || b32_mask_sop2;
                bool mask_saveexec = prev->fmt == Rdna2Format::SOP1 &&
                                     (prev->opcode == 0x24 || prev->opcode == 0x25 ||
                                      prev->opcode == 0x37);
                // The kill mask may live in a plain SGPR pair (s0..s105) OR in VCC itself — DOLL's
                // alpha-cull PS does `s_andn2_b64 vcc, exec, vcc; s_cbranch_scc0 <null-export>` then
                // `s_mov_b64 exec, vcc; export`. The SOP2 dst field decodes VCC_LO as SGPR 106, and
                // emit_alu's mask ops route a 106/107 dst to rs.vcc, so the same linearization holds
                // (the branch is a whole-wave early-out; per-invocation the export's OpKill covers it).
                if ((mask_sop2 || mask_saveexec) && prev->dst.kind == OperandKind::SGPR && prev->dst.value <= 106)
                    out.insert(in.pc);
            }
        }

        prev = is_scalar_branch(in) ? nullptr : &in;
    }
    return out;
}
// allow_vcc: also accept a forward s_cbranch_vccz/vccnz. Fragment reduces the per-lane bool through an
// enforced native wave64 subgroup; vertex retains the guarded per-invocation representation. Compute
// routes accepted VCC/EXEC branches through the CFG dispatcher, whose workgroup scratch reduction spans
// the configured 32/64-lane guest wave independently of the implementation-defined host subgroup width.
// code/dwords: the raw stream, so a branch target past the first s_endpgm can be decoded and
// verified to be a genuine early-out (see below) instead of blanket-clamped.
ForwardIf detect_forward_if(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                            const uint32_t* code, size_t dwords,
                            const std::unordered_set<uint32_t>* skip = nullptr) {
    ForwardIf F; const Rdna2Inst* br = nullptr; int nbranch = 0; uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        switch (in.opcode) {
            case 0x02: case 0x08: case 0x09:                         // s_branch / execz / execnz -> reject
                return F;
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (!allow_vcc) return F;                            // compute: needs a wave-uniform VCC test
                br = &in; nbranch++; break;
            case 0x04: case 0x05:                                    // scc0 / scc1 (SCC is scalar/wave-uniform)
                if (skip && skip->count(in.pc)) break;               // alpha-test kill-mask branch: handled by
                                                                     // straight-line linearization, not a struct-if
                br = &in; nbranch++; break;
            default: break;                                          // hints (nop/waitcnt/…) are fine
        }
    }
    if (nbranch != 1 || !br) return F;
    uint32_t tgt = branch_target(*br);
    // Early-out: a forward branch PAST s_endpgm skips the rest of the shader — but only if execution
    // at the target immediately terminates. Decode from tgt and require the first non-s_nop
    // instruction to be s_endpgm (the compiled early-out shape `s_cbranch L; work; s_endpgm;
    // L: s_endpgm`). Anything else is REAL reachable code (an else-block) that the old blanket clamp
    // silently discarded — valid SPIR-V, wrong semantics (#129) — so reject instead (the caller's
    // straight-line fallback also rejects the branch, loudly). When verified, clamp the merge to
    // end_pc so the conditional block is [branch_pc+1, end_pc) and s_endpgm is emitted after the merge.
    bool early = false;
    if (tgt > end_pc && tgt != UINT32_MAX) {
        if (!code || tgt >= dwords) return F;    // target outside the decode window: can't verify
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + tgt, dwords - tgt, tail);
        bool ends_immediately = false;
        for (const auto& ti : tail) {
            if (ti.is_end) { ends_immediately = true; break; }
            if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;   // s_nop padding
            break;                               // real instruction at the target -> not an early-out
        }
        if (!ends_immediately) return F;
        tgt = end_pc; early = true;
    }
    if (tgt <= br->pc || tgt > end_pc) return F;                     // must be forward, within the stream
    F.found = true; F.branch_pc = br->pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
    F.on_scc0 = (br->opcode == 0x04 || br->opcode == 0x06);         // scc0/vccz: skip block when flag==0
    F.on_vcc  = (br->opcode == 0x06 || br->opcode == 0x07);
    return F;
}

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

// S_MOV_B64 from VCC publishes both an exact Bool-domain saved mask and its two ballot words under
// native Wave64. The compact structured emitter preserves both views through its SSA/PHI machinery,
// but unlike the dispatcher it previously had no lifetime tag telling S_FF1/S_BCNT which view owns
// the pair at an exact consumer. Compute a small forward MUST analysis over the decoded scalar CFG:
// a saved-mask fact is generated by an exact VCC copy reached from a proved mask-domain VCC or by a
// SAVEEXEC destination (which receives OLD_EXEC independently of its logical source). Every
// overlapping scalar write kills it, and joins retain it only when every reachable predecessor
// agrees. Indirect PC updates have no successor, so they cannot manufacture a dominance fact beyond
// an unknown transfer.

} // namespace

namespace {

const Rdna2Inst* last_scalar_writer(const std::vector<Rdna2Inst>& ins, uint32_t before_pc,
                                    int reg) {
    const Rdna2Inst* result = nullptr;
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= before_pc) break;
        bool writes_reg = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            writes_reg |= reg >= base && reg < base + static_cast<int>(width);
        });
        if (writes_reg) result = &in;
    }
    return result;
}

bool immediate_operand(const Operand& operand, const Rdna2Inst& in, uint32_t& value) {
    if (operand.kind == OperandKind::Literal) { value = in.literal; return true; }
    if (operand.kind == OperandKind::InlineInt) {
        value = static_cast<uint32_t>(operand.value);
        return true;
    }
    return false;
}

bool binary_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return (reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate)) ||
           (reg_operand(in.src[1], reg) && immediate_operand(in.src[0], in, immediate));
}

bool ordered_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate);
}

bool binary_regs(const Rdna2Inst& in, int a, int b) {
    return (reg_operand(in.src[0], a) && reg_operand(in.src[1], b)) ||
           (reg_operand(in.src[0], b) && reg_operand(in.src[1], a));
}

PcrelDispatchInfo detect_pcrel_dispatch(const std::vector<Rdna2Inst>& ins,
                                        const uint32_t* code, size_t dwords,
                                        size_t program_dwords) {
    PcrelDispatchInfo out;
    if (!code || ins.empty()) return out;

    std::unordered_set<uint32_t> instruction_pcs;
    std::vector<uint32_t> branch_targets;
    for (const auto& in : ins) {
        instruction_pcs.insert(in.pc);
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03)
            branch_targets.push_back(scalar_branch_target(in));
    }

    for (const auto& setpc : ins) {
        if (setpc.is_end || setpc.fmt != Rdna2Format::SOP1 || setpc.opcode != 0x20) continue;
        const int jump_lo = setpc.src[0].value;
        const Rdna2Inst* jump_add = last_scalar_writer(ins, setpc.pc, jump_lo);
        const Rdna2Inst* jump_addc = last_scalar_writer(ins, setpc.pc, jump_lo + 1);
        if (!jump_add || !jump_addc || jump_add->fmt != Rdna2Format::SOP2 ||
            jump_add->opcode != 0x00 || jump_addc->fmt != Rdna2Format::SOP2 ||
            jump_addc->opcode != 0x04 || jump_add->pc >= jump_addc->pc) continue;

        int table_lo = -1;
        for (const Operand& source : jump_add->src) {
            if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
                source.value != jump_lo) table_lo = source.value;
        }
        if (table_lo < 0 || !binary_regs(*jump_add, jump_lo, table_lo) ||
            !binary_regs(*jump_addc, jump_lo + 1, table_lo + 1)) continue;
        const Rdna2Inst* target_getpc = last_scalar_writer(ins, jump_add->pc, jump_lo);
        if (!target_getpc || target_getpc->fmt != Rdna2Format::SOP1 ||
            target_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* table_load = last_scalar_writer(ins, jump_add->pc, table_lo);
        if (!table_load || table_load != last_scalar_writer(ins, jump_add->pc, table_lo + 1) ||
            table_load->fmt != Rdna2Format::SMEM || table_load->opcode != 0x01 ||
            table_load->literal != 0 || table_load->src[0].kind != OperandKind::SGPR) continue;
        const int table_base_lo = table_load->src[0].value;
        const int selector = table_load->src[1].value;

        const Rdna2Inst* table_add = last_scalar_writer(ins, table_load->pc, table_base_lo);
        const Rdna2Inst* table_addc = last_scalar_writer(ins, table_load->pc, table_base_lo + 1);
        if (!table_add || !table_addc || table_add->fmt != Rdna2Format::SOP2 ||
            table_add->opcode != 0x00 || table_addc->fmt != Rdna2Format::SOP2 ||
            table_addc->opcode != 0x04 || table_add->pc >= table_addc->pc) continue;
        uint32_t table_delta = 0, high_zero = 1;
        if (!binary_reg_immediate(*table_add, table_base_lo, table_delta) ||
            !binary_reg_immediate(*table_addc, table_base_lo + 1, high_zero) || high_zero != 0)
            continue;
        const Rdna2Inst* table_getpc = last_scalar_writer(ins, table_add->pc, table_base_lo);
        if (!table_getpc || table_getpc->fmt != Rdna2Format::SOP1 ||
            table_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* shift = last_scalar_writer(ins, table_load->pc, selector);
        if (!shift || shift->fmt != Rdna2Format::SOP2 || shift->opcode != 0x1e) continue;
        uint32_t shift_amount = 0;
        // s_lshl_b32 is ordered: selector << 3 scales the qword index, while 3 << selector
        // is a different program and must not be admitted by the commutative matcher.
        if (!ordered_reg_immediate(*shift, selector, shift_amount) || shift_amount != 3) continue;
        const Rdna2Inst* clamp = last_scalar_writer(ins, shift->pc, selector);
        if (!clamp || clamp->fmt != Rdna2Format::SOP2 || clamp->opcode != 0x07) continue;
        uint32_t selector_max = 0;
        if (!binary_reg_immediate(*clamp, selector, selector_max) || selector_max > 63) continue;

        const Rdna2Inst* adjust = last_scalar_writer(ins, clamp->pc, selector);
        int32_t selector_addend = 0;
        if (adjust && adjust->fmt == Rdna2Format::SOP2 &&
            (adjust->opcode == 0x00 || adjust->opcode == 0x02)) {
            uint32_t addend = 0;
            if (!binary_reg_immediate(*adjust, selector, addend)) continue;
            selector_addend = static_cast<int32_t>(addend);
        } else {
            adjust = nullptr;
        }
        const Rdna2Inst* selector_load = last_scalar_writer(
            ins, adjust ? adjust->pc : clamp->pc, selector);
        if (!selector_load || selector_load->fmt != Rdna2Format::SMEM ||
            selector_load->opcode != 0x08 || selector_load->src[0].kind != OperandKind::SGPR ||
            selector_load->src[1].kind != OperandKind::Special ||
            selector_load->src[1].value != 125) continue;

        const uint64_t table_byte =
            static_cast<uint64_t>(table_getpc->pc + table_getpc->len_dwords) * 4u + table_delta;
        const size_t entry_count = static_cast<size_t>(selector_max) + 1;
        if ((table_byte & 7u) || table_byte / 4 < program_dwords ||
            table_byte / 4 + entry_count * 2 > dwords) continue;

        std::vector<uint32_t> targets;
        targets.reserve(entry_count);
        const int64_t target_pc_byte =
            static_cast<int64_t>(target_getpc->pc + target_getpc->len_dwords) * 4;
        bool table_ok = true;
        for (size_t index = 0; index < entry_count; ++index) {
            const size_t word = static_cast<size_t>(table_byte / 4) + index * 2;
            const int64_t relative = static_cast<int64_t>(
                (static_cast<uint64_t>(code[word + 1]) << 32) | code[word]);
            const int64_t target_byte = target_pc_byte + relative;
            if (target_byte < 0 || (target_byte & 3) ||
                target_byte / 4 > static_cast<int64_t>(UINT32_MAX) ||
                !instruction_pcs.contains(static_cast<uint32_t>(target_byte / 4))) {
                table_ok = false;
                break;
            }
            targets.push_back(static_cast<uint32_t>(target_byte / 4));
        }
        if (!table_ok || targets.empty()) continue;
        const uint32_t merge_pc = *std::max_element(targets.begin(), targets.end());
        if (merge_pc <= setpc.pc) continue;
        for (uint32_t target : targets) if (target < setpc.pc + setpc.len_dwords || target > merge_pc)
            table_ok = false;
        if (!table_ok) continue;

        const std::vector<uint32_t> setup = {
            selector_load->pc,
            adjust ? adjust->pc : UINT32_MAX,
            clamp->pc, shift->pc, table_getpc->pc, table_add->pc, table_addc->pc,
            table_load->pc, target_getpc->pc, jump_add->pc, jump_addc->pc, setpc.pc,
        };
        const uint32_t setup_first = selector_load->pc;
        for (uint32_t target : branch_targets) {
            if (target > setup_first && target <= setpc.pc) { table_ok = false; break; }
        }
        if (!table_ok) continue;

        out.valid = true;
        out.selector_sgpr_base = static_cast<uint32_t>(selector_load->src[0].value);
        out.selector_byte_offset = selector_load->literal;
        out.selector_addend = selector_addend;
        out.selector_max = selector_max;
        out.setpc_pc = setpc.pc;
        out.merge_pc = merge_pc;
        out.required_dwords = static_cast<size_t>(table_byte / 4) + entry_count * 2;
        out.target_pcs = std::move(targets);
        for (uint32_t pc : setup) if (pc != UINT32_MAX) out.setup_pcs.push_back(pc);
        return out;
    }
    return out;
}

bool specialize_pcrel_dispatch(std::vector<Rdna2Inst>& ins, const PcrelDispatchInfo& info,
                               uint32_t selected_target) {
    if (!info.valid || std::find(info.target_pcs.begin(), info.target_pcs.end(), selected_target) ==
                           info.target_pcs.end()) return false;
    std::unordered_set<uint32_t> remove(info.setup_pcs.begin(), info.setup_pcs.end());

    // A compiler may jump over an alternate entry prologue before it starts the dispatch setup. Fold
    // only forward unconditional branches wholly contained in that prelude; any external entry into a
    // skipped range makes the specialization unprovable.
    for (const auto& branch : ins) {
        if (branch.pc >= info.setpc_pc || branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x02)
            continue;
        const uint32_t target = scalar_branch_target(branch);
        if (target <= branch.pc || target > info.setpc_pc) return false;
        for (const auto& other : ins) {
            if (other.fmt != Rdna2Format::SOPP || other.pc == branch.pc || other.opcode < 0x02 ||
                other.opcode > 0x09 || other.opcode == 0x03) continue;
            const uint32_t entered = scalar_branch_target(other);
            if (entered > branch.pc + branch.len_dwords && entered < target) return false;
        }
        remove.insert(branch.pc);
        for (const auto& skipped : ins)
            if (skipped.pc >= branch.pc + branch.len_dwords && skipped.pc < target)
                remove.insert(skipped.pc);
    }

    uint32_t selected_end = info.merge_pc;
    for (const auto& in : ins) {
        if (in.pc < selected_target || in.pc >= info.merge_pc) continue;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        // Internal forward branches and loop back-edges remain in the selected routine and are
        // validated/structured by emit_body. Only the compiler's route terminator jumps to the
        // common merge and can be removed as a now-redundant branch.
        if (scalar_branch_target(in) != info.merge_pc) continue;
        selected_end = in.pc;
        remove.insert(in.pc);
        break;
    }

    std::vector<Rdna2Inst> specialized;
    specialized.reserve(ins.size());
    for (const auto& in : ins) {
        const bool prelude = in.pc < info.setpc_pc;
        const bool selected = in.pc >= selected_target && in.pc < selected_end;
        const bool merge = in.pc >= info.merge_pc;
        if ((prelude || selected || merge) && !remove.contains(in.pc)) specialized.push_back(in);
    }
    if (specialized.empty()) return false;

    // No surviving branch may enter an omitted alternative. The ordinary structurizer performs the
    // remaining detailed CFG checks after this coarse specialization boundary check.
    std::unordered_set<uint32_t> retained;
    uint32_t end_pc = 0;
    for (const auto& in : specialized) retained.insert(in.pc);
    for (const auto& in : specialized) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : specialized) {
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03 || in.simm16 < 0) continue;
        const uint32_t target = scalar_branch_target(in);
        // Existing forward-if validation accepts a compiler early-out just beyond the primary
        // S_ENDPGM only after proving that raw target terminates immediately. Preserve that case for
        // the detailed validator; targets into an omitted alternative still fail here.
        if (!retained.contains(target) && target <= end_pc) return false;
    }
    ins = std::move(specialized);
    return true;
}

struct ShaderConstantValue {
    bool known = false;
    uint32_t value = 0;
};

ShaderConstantValue shader_constant_operand(
        const std::vector<Rdna2Inst>& ins, size_t block_first, size_t use_index,
        const Rdna2Inst& use, const Operand& operand, uint32_t depth) {
    if (depth > 16) return {};
    if (operand.kind == OperandKind::InlineInt)
        return {true, static_cast<uint32_t>(operand.value)};
    if (operand.kind == OperandKind::Literal)
        return use.has_literal ? ShaderConstantValue{true, use.literal} : ShaderConstantValue{};
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special)
        return {};

    const int reg = operand.value;
    for (size_t i = use_index; i-- > block_first;) {
        const Rdna2Inst& writer = ins[i];
        // VCC_LO/HI can temporarily hold ordinary scalar data, as in Astro's exact branch setup,
        // but any intervening vector ALU may replace the architectural VCC pair implicitly. The
        // shared scalar-writer inventory intentionally models that in the Bool domain, so reject it
        // explicitly here instead of walking past a VOPC to an obsolete scalar definition.
        if ((reg == 106 || reg == 107) &&
            (writer.fmt == Rdna2Format::VOP1 || writer.fmt == Rdna2Format::VOP2 ||
             writer.fmt == Rdna2Format::VOPC || writer.fmt == Rdna2Format::VOP3 ||
             writer.fmt == Rdna2Format::VOP3P))
            return {};
        bool writes = false;
        uint32_t width = 0;
        int base = -1;
        for_each_scalar_write(writer, [&](int candidate_base, uint32_t candidate_width) {
            if (reg >= candidate_base &&
                reg < candidate_base + static_cast<int>(candidate_width)) {
                writes = true;
                base = candidate_base;
                width = candidate_width;
            }
        });
        if (!writes) continue;

        // Only one-dword pure scalar data writers participate. A pair write, memory result, lane
        // read, or wave-mask producer is intentionally not a shader-constant proof.
        if (base != reg || width != 1) return {};
        if (writer.fmt == Rdna2Format::SOP1 && writer.opcode == 0x03) { // s_mov_b32
            return shader_constant_operand(
                ins, block_first, i, writer, writer.src[0], depth + 1);
        }
        if (writer.fmt != Rdna2Format::SOP2) return {};

        const ShaderConstantValue lhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[0], depth + 1);
        const ShaderConstantValue rhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[1], depth + 1);
        if (!lhs.known || !rhs.known) return {};

        switch (writer.opcode) {
            case 0x00: return {true, lhs.value + rhs.value}; // s_add_u32
            case 0x01: return {true, lhs.value - rhs.value}; // s_sub_u32
            case 0x02: return {true, lhs.value + rhs.value}; // s_add_i32
            case 0x03: return {true, lhs.value - rhs.value}; // s_sub_i32
            case 0x0e: return {true, lhs.value & rhs.value}; // s_and_b32
            case 0x10: return {true, lhs.value | rhs.value}; // s_or_b32
            case 0x12: return {true, lhs.value ^ rhs.value}; // s_xor_b32
            // Shifts take the amount from S1[4:0] -- RDNA2 ISA -- so the `& 31` is required.
            //
            // It is deliberately UNTESTED, because on this host it is untestable: x86 `shl` already
            // masks its count to 5 bits, so an unmasked `lhs.value << rhs.value` yields the SAME
            // answer here (measured: naive `5u << 32` prints 5). The mask is therefore not fixing a
            // wrong result on x86; it is removing undefined behaviour that a compiler is entitled to
            // exploit, and that would diverge on a host whose shift does not wrap. Do not add a
            // regression arm claiming to prove it -- such an arm passes with the mask removed, which
            // makes it a control that cannot fail. The arms below cover only what is observable:
            // that these opcodes fold at all.
            //
            // Opcodes verified against llvm-mc (gfx1030), not against this file's own tables: the
            // decoder that produces the listing you would otherwise check them with is upstream of
            // them, so it cannot check them. #2481 records a mnemonic error that survived three
            // internally consistent anchors and inverted a frontier conclusion.
            case 0x1e: return {true, lhs.value << (rhs.value & 31u)};  // s_lshl_b32
            case 0x20: return {true, lhs.value >> (rhs.value & 31u)};  // s_lshr_b32
            case 0x22: return {true, static_cast<uint32_t>(            // s_ashr_i32
                std::bit_cast<int32_t>(lhs.value) >> (rhs.value & 31u))};
            case 0x26: return {true, lhs.value * rhs.value}; // s_mul_i32
            default: return {};
        }
    }
    // No in-block writer means an entry/user SGPR or other runtime state. Never specialize it.
    return {};
}

bool shader_constant_compare(const std::vector<Rdna2Inst>& ins, size_t block_first,
                             size_t compare_index, bool& result) {
    const Rdna2Inst& compare = ins[compare_index];
    if (compare.fmt != Rdna2Format::SOPC || compare.opcode > 0x0b) return false;
    const ShaderConstantValue lhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[0], 0);
    const ShaderConstantValue rhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[1], 0);
    if (!lhs.known || !rhs.known) return false;
    const int32_t signed_lhs = std::bit_cast<int32_t>(lhs.value);
    const int32_t signed_rhs = std::bit_cast<int32_t>(rhs.value);

    switch (compare.opcode) {
        case 0x00: result = signed_lhs == signed_rhs; break;
        case 0x01: result = signed_lhs != signed_rhs; break;
        case 0x02: result = signed_lhs >  signed_rhs; break;
        case 0x03: result = signed_lhs >= signed_rhs; break;
        case 0x04: result = signed_lhs <  signed_rhs; break;
        case 0x05: result = signed_lhs <= signed_rhs; break;
        case 0x06: result = lhs.value == rhs.value; break;
        case 0x07: result = lhs.value != rhs.value; break;
        case 0x08: result = lhs.value >  rhs.value; break;
        case 0x09: result = lhs.value >= rhs.value; break;
        case 0x0a: result = lhs.value <  rhs.value; break;
        case 0x0b: result = lhs.value <= rhs.value; break;
        default: return false;
    }
    return true;
}

bool scalar_cfg_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
        sopp_opcode_is_direct_branch(in.opcode);
}

void prune_scalar_cfg_reachability(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::vector<bool> reachable(ins.size(), false);
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t i = pending.back();
        pending.pop_back();
        if (i >= ins.size() || reachable[i]) continue;
        reachable[i] = true;
        const Rdna2Inst& in = ins[i];
        if (in.is_end || (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) continue;
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) pending.push_back(target->second);
            if (in.opcode == 0x02) continue;
        }
        if (i + 1 < ins.size()) pending.push_back(i + 1);
    }

    std::vector<Rdna2Inst> retained;
    retained.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
        if (reachable[i]) retained.push_back(ins[i]);
    // Once a taken branch's omitted arm is gone, its target is often the next retained instruction.
    // Turn that now-redundant edge into a no-op so the ordinary straight-line/structured paths do not
    // need to reconstruct an empty branch region.
    for (size_t i = 0; i + 1 < retained.size(); ++i) {
        Rdna2Inst& in = retained[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02 &&
            scalar_branch_target(in) == retained[i + 1].pc) {
            in.opcode = 0x00;
            in.simm16 = 0;
            in.words[0] = 0xbf800000u;
        }
    }
    ins = std::move(retained);
}

size_t specialize_shader_constant_branches(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return 0;
    // An indirect PC transfer can enter code outside the explicit SOPP graph. Keep the whole shader
    // unspecialized rather than treating its lexical successor as the only possible destination.
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::SOP1 &&
                   in.opcode >= 0x20 && in.opcode <= 0x22;
        })) return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::set<size_t> block_starts{0};
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) block_starts.insert(target->second);
        }
        if ((scalar_cfg_branch(in) || in.is_end ||
             (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) &&
            i + 1 < ins.size())
            block_starts.insert(i + 1);
    }

    size_t specialized_count = 0;
    for (size_t i = 1; i < ins.size(); ++i) {
        Rdna2Inst& branch = ins[i];
        if (branch.fmt != Rdna2Format::SOPP ||
            (branch.opcode != 0x04 && branch.opcode != 0x05) || branch.simm16 <= 0)
            continue; // shader-constant SCC only; VCCZ/EXECZ remain runtime wave conditions
        if (!index_by_pc.contains(scalar_branch_target(branch)))
            continue; // do not specialize an exit beyond the decoded instruction graph
        const Rdna2Inst& compare = ins[i - 1];
        if (compare.pc + compare.len_dwords != branch.pc ||
            compare.fmt != Rdna2Format::SOPC)
            continue;
        const auto block = block_starts.upper_bound(i - 1);
        const size_t block_first = block == block_starts.begin() ? 0 : *std::prev(block);
        if (block_first > i - 1) continue;

        bool scc = false;
        if (!shader_constant_compare(ins, block_first, i - 1, scc)) continue;
        const bool taken = branch.opcode == 0x05 ? scc : !scc;
        if (taken) {
            branch.opcode = 0x02; // s_branch: retain the exact immediate target
            branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
        } else {
            branch.opcode = 0x00; // s_nop 0: retain fallthrough
            branch.simm16 = 0;
            branch.words[0] = 0xbf800000u;
        }
        ++specialized_count;
    }
    if (!specialized_count) return 0;

    // Entry-rooted reachability after replacing proven conditions. Unknown conditional branches
    // retain both successors, so an unresolved resource remains in the instruction stream whenever
    // any runtime path can execute it.
    prune_scalar_cfg_reachability(ins);
    return specialized_count;
}

size_t specialize_proven_null_bvh_exits(std::vector<Rdna2Inst>& ins,
                                        const ShaderResourceTable* rt,
                                        uint32_t wave_size) {
    if (!rt || (wave_size != 32 && wave_size != 64) || ins.empty()) return 0;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    size_t specialized = 0;
    for (const ShaderResource& resource : rt->resources) {
        if (!is_proven_null_bvh(resource)) continue;
        const auto found = index_by_pc.find(resource.fetch_pc);
        if (found == index_by_pc.end()) continue;
        const size_t ray_index = found->second;
        if (ray_index + 5 >= ins.size()) continue;
        const Rdna2Inst& ray = ins[ray_index];
        const Rdna2Inst& wait = ins[ray_index + 1];
        const Rdna2Inst& compare = ins[ray_index + 2];
        const Rdna2Inst& scalar_compare = ins[ray_index + 3];
        const Rdna2Inst& exec_copy = ins[ray_index + 4];
        Rdna2Inst& exit = ins[ray_index + 5];

        // Exact Wave32 no-hit idiom:
        //   image_bvh_intersect_ray vN..vN+3, NULL_BVH
        //   s_waitcnt ...
        //   v_cmp_ne_u32 sM, -1, vN
        //   s_cmp_lg_u32 0, sM
        //   s_mov_b32 exec_lo, vcc_lo
        //   s_cbranch_scc0 EXIT
        // The Wave64 sibling uses the complete two-word mask instead:
        //   v_cmp_ne_u32 s[M:M+1], -1, vN
        //   s_cmp_lg_u64 s[M:M+1], 0
        //   s_mov_b64 exec, vcc
        // The null lowering writes -1 to every ray result for each active lane. The saved compare
        // mask is therefore exactly zero, making SCC zero and the exit unconditional. Requiring the
        // exact width-specific compare and EXEC copy avoids assuming anything about an unobserved
        // mask half. Adjacency and exact register links prevent a nearby unrelated compare from
        // proving the branch.
        const bool ray_shape = ray.fmt == Rdna2Format::MIMG && ray.opcode == 0xe6u &&
            ray.mimg_dmask == 0xfu && ray.dst.kind == OperandKind::VGPR;
        const bool wait_shape = wait.pc == ray.pc + ray.len_dwords &&
            wait.fmt == Rdna2Format::SOPP && wait.opcode == 0x0cu &&
            wait.words[0] == 0xbf8c3f70u; // s_waitcnt vmcnt(0)
        const bool compare_shape = compare.pc == wait.pc + wait.len_dwords &&
            compare.fmt == Rdna2Format::VOPC && compare.opcode == 0xc5u &&
            compare.dst.kind == OperandKind::SGPR &&
            compare.dst.value <= (wave_size == 32 ? 105 : 104) &&
            compare.src[0].kind == OperandKind::InlineInt && compare.src[0].value == -1 &&
            compare.src[1].kind == OperandKind::VGPR && compare.src[1].value == ray.dst.value;
        const bool scalar_shape = scalar_compare.pc == compare.pc + compare.len_dwords &&
            scalar_compare.fmt == Rdna2Format::SOPC &&
            (wave_size == 32
                ? scalar_compare.opcode == 0x07u &&
                  scalar_compare.src[0].kind == OperandKind::InlineInt &&
                  scalar_compare.src[0].value == 0 &&
                  scalar_compare.src[1].kind == OperandKind::SGPR &&
                  scalar_compare.src[1].value == compare.dst.value
                : scalar_compare.opcode == 0x13u &&
                  scalar_compare.src[0].kind == OperandKind::SGPR &&
                  scalar_compare.src[0].value == compare.dst.value &&
                  scalar_compare.src[1].kind == OperandKind::InlineInt &&
                  scalar_compare.src[1].value == 0);
        const bool copy_shape = exec_copy.pc == scalar_compare.pc + scalar_compare.len_dwords &&
            exec_copy.fmt == Rdna2Format::SOP1 &&
            exec_copy.opcode == (wave_size == 32 ? 0x03u : 0x04u) &&
            exec_copy.dst.kind == OperandKind::SGPR && exec_copy.dst.value == 126 &&
            exec_copy.src[0].kind == OperandKind::Special && exec_copy.src[0].value == 106;
        const bool exit_shape = exit.pc == exec_copy.pc + exec_copy.len_dwords &&
            exit.fmt == Rdna2Format::SOPP && exit.opcode == 0x04u && exit.simm16 > 0 &&
            index_by_pc.contains(scalar_branch_target(exit));
        if (!ray_shape || !wait_shape || !compare_shape || !scalar_shape || !copy_shape ||
            !exit_shape)
            continue;

        // Some compiler-generated traversal loops enter with an empty scalar stack, visit one
        // root ray, then pop work written by that ray.  When the dispatch-scoped root is proven
        // null, the exact no-hit branch above reaches the empty-stack test without writing the
        // depth.  That makes the pop/back-edge arm unreachable, which in turn proves that the
        // loop-selected ray sites cannot seed themselves.  Keep this deliberately narrower than
        // ordinary scalar constant propagation: every entry, register relationship, and branch on
        // the first path must match the observed stack idiom.
        auto specialize_empty_stack = [&]() {
            const uint32_t null_exit_pc = scalar_branch_target(exit);
            const auto tail_found = index_by_pc.find(null_exit_pc);
            if (tail_found == index_by_pc.end() || tail_found->second + 1 >= ins.size())
                return false;
            const size_t tail_index = tail_found->second;
            const Rdna2Inst& stack_compare = ins[tail_index];
            Rdna2Inst& stack_exit = ins[tail_index + 1];
            if (stack_compare.fmt != Rdna2Format::SOPC || stack_compare.opcode != 0x07u ||
                stack_compare.src[0].kind != OperandKind::InlineInt ||
                stack_compare.src[0].value != 0 ||
                stack_compare.src[1].kind != OperandKind::SGPR ||
                stack_compare.src[1].value < 0 || stack_compare.src[1].value > 105 ||
                stack_exit.pc != stack_compare.pc + stack_compare.len_dwords ||
                stack_exit.fmt != Rdna2Format::SOPP || stack_exit.opcode != 0x04u ||
                stack_exit.simm16 <= 0 ||
                !index_by_pc.contains(scalar_branch_target(stack_exit)))
                return false;
            // The compiler allocates this scalar stack depth opportunistically (the observed
            // traversal kernels use both s41 and s45). Derive the physical word from the exact
            // empty-stack comparison, then require the initializer and every write check below to
            // agree with it. This broadens only register allocation, not the proven control/data
            // relationship.
            const int stack_reg = stack_compare.src[1].value;
            const uint32_t stack_exit_pc = scalar_branch_target(stack_exit);

            // The first-iteration selector branch targets the block that initializes all four
            // ray results to invalid before the guarded root query.  Requiring this complete
            // eight-instruction prefix prevents a nearby null ray from being mistaken for the
            // traversal root that owns the scalar stack below.
            if (ray_index < 8) return false;
            const size_t root_block = ray_index - 8;
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const Rdna2Inst& init = ins[root_block + lane];
                if (init.fmt != Rdna2Format::VOP1 || init.opcode != 0x01u ||
                    init.dst.kind != OperandKind::VGPR ||
                    init.dst.value != ray.dst.value + static_cast<int>(lane) ||
                    init.src[0].kind != OperandKind::InlineInt || init.src[0].value != -1)
                    return false;
            }
            const Rdna2Inst& mask_copy = ins[root_block + 4];
            const Rdna2Inst& empty_guard = ins[root_block + 5];
            const Rdna2Inst& root_index = ins[root_block + 6];
            const Rdna2Inst& root_nop = ins[root_block + 7];
            const bool mask_copy_shape = wave_size == 32
                ? mask_copy.fmt == Rdna2Format::SOP1 &&
                  mask_copy.opcode == kSop1OpcodeAndSaveexecB32 &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::Special &&
                  mask_copy.src[0].value == 107
                : mask_copy.fmt == Rdna2Format::SOP1 && mask_copy.opcode == 0x24u &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::SGPR;
            if (!mask_copy_shape ||
                empty_guard.fmt != Rdna2Format::SOPP || empty_guard.opcode != 0x08u ||
                scalar_branch_target(empty_guard) != scalar_compare.pc ||
                root_index.fmt != Rdna2Format::VOP1 || root_index.opcode != 0x01u ||
                root_index.dst.kind != OperandKind::VGPR ||
                root_index.dst.value != ray.dst.value ||
                root_index.src[0].kind != OperandKind::SGPR ||
                root_nop.fmt != Rdna2Format::SOPP || root_nop.opcode != 0x00u)
                return false;

            size_t selector_branch_index = SIZE_MAX;
            size_t stack_init_index = SIZE_MAX;
            bool selector_scc = false;
            for (size_t i = root_block; i-- > 2;) {
                const Rdna2Inst& branch = ins[i];
                if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x05u ||
                    scalar_branch_target(branch) != ins[root_block].pc ||
                    ins[i - 1].pc + ins[i - 1].len_dwords != branch.pc ||
                    ins[i - 1].fmt != Rdna2Format::SOPC)
                    continue;
                for (size_t j = i - 1; j-- > 1;) {
                    const Rdna2Inst& selector_init = ins[j];
                    if (selector_init.fmt != Rdna2Format::SOP1 ||
                        selector_init.opcode != 0x03u ||
                        selector_init.dst.kind != OperandKind::SGPR ||
                        selector_init.src[0].kind != OperandKind::InlineInt ||
                        selector_init.src[0].value != 37 ||
                        root_index.src[0].value != selector_init.dst.value)
                        continue;
                    size_t candidate_stack_init = SIZE_MAX;
                    if (wave_size == 32) {
                        const Rdna2Inst& stack_init = ins[j - 1];
                        if (stack_init.pc + stack_init.len_dwords == selector_init.pc &&
                            stack_init.fmt == Rdna2Format::SOP1 &&
                            stack_init.opcode == 0x03u &&
                            stack_init.dst.kind == OperandKind::SGPR &&
                            stack_init.dst.value == stack_reg &&
                            stack_init.src[0].kind == OperandKind::InlineInt &&
                            stack_init.src[0].value == 0)
                            candidate_stack_init = j - 1;
                    } else {
                        // The Wave64 sibling schedules independent scalar address setup between
                        // `stack=0` and `selector=37`. Accept only the nearest in-block write to the
                        // derived stack register, and require it to be the exact zero initializer.
                        for (size_t k = j; k-- > 0;) {
                            bool writes_candidate = false;
                            for_each_scalar_write(ins[k], [&](int base, uint32_t width) {
                                writes_candidate |= base <= stack_reg &&
                                    stack_reg < base + static_cast<int>(width);
                            });
                            if (!writes_candidate) {
                                if (scalar_cfg_branch(ins[k]) || ins[k].is_end) break;
                                continue;
                            }
                            const Rdna2Inst& stack_init = ins[k];
                            if (stack_init.fmt == Rdna2Format::SOP1 &&
                                stack_init.opcode == 0x03u &&
                                stack_init.dst.kind == OperandKind::SGPR &&
                                stack_init.dst.value == stack_reg &&
                                stack_init.src[0].kind == OperandKind::InlineInt &&
                                stack_init.src[0].value == 0)
                                candidate_stack_init = k;
                            break;
                        }
                    }
                    if (candidate_stack_init == SIZE_MAX) continue;
                    bool has_branch = false;
                    for (size_t k = candidate_stack_init + 1; k < i; ++k)
                        has_branch |= scalar_cfg_branch(ins[k]);
                    if (has_branch ||
                        !shader_constant_compare(ins, j, i - 1, selector_scc) ||
                        !selector_scc)
                        continue;
                    selector_branch_index = i;
                    stack_init_index = candidate_stack_init;
                    break;
                }
                if (selector_branch_index != SIZE_MAX) break;
            }
            if (selector_branch_index == SIZE_MAX) return false;

            auto writes_stack = [&](const Rdna2Inst& instruction) {
                bool writes = false;
                for_each_scalar_write(instruction, [&](int base, uint32_t width) {
                    writes |= base <= stack_reg &&
                        stack_reg < base + static_cast<int>(width);
                }, /*wave32_one_word_masks=*/wave_size == 32);
                return writes;
            };
            // Only the selector setup, root/no-hit block, and empty-stack comparison are reachable
            // before the proven exits.  None may alter the initialized stack depth.
            for (size_t i = stack_init_index + 1; i <= selector_branch_index; ++i)
                if (writes_stack(ins[i])) return false;
            for (size_t i = root_block; i <= ray_index + 5; ++i)
                if (writes_stack(ins[i])) return false;
            if (writes_stack(stack_compare)) return false;

            const uint32_t proof_begin = ins[stack_init_index].pc;
            // An indirect transfer or an external edge into the middle of the proof could bypass
            // the zero initializer.  Re-entry at the initializer itself is harmless: it resets the
            // invariant before the selector is evaluated again.
            for (const Rdna2Inst& instruction : ins) {
                if (instruction.fmt == Rdna2Format::SOP1 &&
                    instruction.opcode >= 0x20u && instruction.opcode <= 0x22u)
                    return false;
                if (!scalar_cfg_branch(instruction)) continue;
                const uint32_t target = scalar_branch_target(instruction);
                const bool source_inside = instruction.pc >= proof_begin &&
                    instruction.pc < stack_exit_pc;
                if (!source_inside && target > proof_begin && target < stack_exit_pc)
                    return false;
            }

            Rdna2Inst& selector_branch = ins[selector_branch_index];
            selector_branch.opcode = 0x02u;
            selector_branch.words[0] = 0xbf820000u |
                static_cast<uint16_t>(selector_branch.simm16);
            stack_exit.opcode = 0x02u;
            stack_exit.words[0] = 0xbf820000u |
                static_cast<uint16_t>(stack_exit.simm16);
            return true;
        };

        (void)specialize_empty_stack();
        exit.opcode = 0x02;
        exit.words[0] = 0xbf820000u | static_cast<uint16_t>(exit.simm16);
        ++specialized;
    }
    if (specialized) prune_scalar_cfg_reachability(ins);
    return specialized;
}

bool valid_scalar_pair_base(int base) {
    return base >= 0 && !(base & 1) &&
        (base <= 104 || base == 106 ||
         (base >= 108 && base <= 122) || base == 126);
}

bool scalar_pair_operand(const Operand& operand, int& base) {
    if ((operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) ||
        !valid_scalar_pair_base(operand.value))
        return false;
    base = operand.value;
    return true;
}

bool valid_buffer_resource_base(const Operand& operand) {
    if (operand.kind != OperandKind::SGPR || operand.value < 0 ||
        (operand.value & 3))
        return false;
    return operand.value <= 100 ||
        (operand.value >= 108 && operand.value <= 120);
}

bool zero_record_load_shape(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::MUBUF &&
        in.opcode == kMubufOpcodeLoadDword && in.len_dwords == 2u &&
        in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 &&
        in.src[0].kind == OperandKind::VGPR &&
        valid_buffer_resource_base(in.src[1]) &&
        in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0 &&
        !in.mubuf_glc && !in.mubuf_dlc && !in.mubuf_lds && !in.mubuf_tfe;
}

bool scalar_instruction_writes_anything(const Rdna2Inst& in) {
    bool writes = false;
    for_each_scalar_write(in, [&](int, uint32_t) { writes = true; });
    return writes;
}

bool dynamic_vgpr_destination(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
}

bool writes_vgpr(const Rdna2Inst& in, int vgpr) {
    if (dynamic_vgpr_destination(in)) return true;
    const uint32_t words = rdna2_vgpr_write_count(in);
    if (words && in.dst.kind == OperandKind::VGPR && in.dst.value <= vgpr &&
        vgpr < in.dst.value + static_cast<int>(words))
        return true;
    return rdna2_tfe_status_vgpr(in) == vgpr;
}

// Prove that every lane active at `and_index` was also active when `load_index` wrote zero. The
// scalar compiler idioms in the live shader elect one lane and later restore a saved mask; tracking
// only subset lineage is enough and deliberately cannot prove an unrelated mask or an expanding
// EXEC write. Conditional skips inside the interval must contain no proof-relevant writes so the
// lexical transfer below represents both paths.
bool zero_load_reaches_and_under_exec_subset(const std::vector<Rdna2Inst>& ins,
                                             size_t load_index, size_t and_index,
                                             const std::unordered_map<uint32_t, size_t>& index_by_pc) {
    const int zero_vgpr = ins[load_index].dst.value;

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& branch = ins[i];
        if (!scalar_cfg_branch(branch)) continue;
        const auto target = index_by_pc.find(scalar_branch_target(branch));
        if (target == index_by_pc.end()) return false;

        const bool source_inside = i > load_index && i < and_index;
        // The compare and branch are part of the proof too: an external edge to either would
        // bypass the zero-reaching definition just as surely as one into the transfer interval.
        const bool target_enters_after_load = target->second > load_index &&
            target->second <= and_index + 2u;
        if (!source_inside && target_enters_after_load) return false;
        if (!source_inside) continue;
        if (branch.opcode == kSoppOpcodeBranch || target->second <= i ||
            target->second > and_index)
            return false;
        for (size_t skipped = i + 1; skipped < target->second; ++skipped) {
            const Rdna2Inst& candidate = ins[skipped];
            if (rdna2_instruction_may_change_exec(candidate) ||
                scalar_instruction_writes_anything(candidate) ||
                writes_vgpr(candidate, zero_vgpr))
                return false;
        }
    }

    std::array<bool, 128> mask_subset{};
    bool exec_subset = true; // the load's EXEC is the reference set

    auto pair_is_subset = [&](const Operand& operand) {
        if (operand.value == 126 &&
            (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special))
            return exec_subset;
        int base = -1;
        return scalar_pair_operand(operand, base) &&
            mask_subset[base] && mask_subset[base + 1];
    };

    for (size_t i = load_index + 1; i < and_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (writes_vgpr(in, zero_vgpr)) return false;

        bool derived_mask = false;
        int mask_dst = -1;
        bool next_exec_subset = exec_subset;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
            if (in.dst.value == 126) {
                next_exec_subset = pair_is_subset(in.src[0]);
                if (!next_exec_subset) return false;
            } else if (in.dst.value == 127) {
                // A B64 destination pair must be even-aligned. Reject the invalid EXEC_HI-rooted
                // packet here rather than letting it carry proof state through this exact analysis.
                return false;
            } else if (pair_is_subset(in.src[0])) {
                if (!valid_scalar_pair_base(in.dst.value))
                    return false;
                derived_mask = true;
                mask_dst = in.dst.value;
            }
        }
        if (in.fmt == Rdna2Format::SOP1 &&
            in.opcode == kSop1OpcodeAndSaveexecB64) {
            // The destination receives OLD_EXEC and the new EXEC is OLD_EXEC & src.
            if (!valid_scalar_pair_base(in.dst.value))
                return false;
            derived_mask = exec_subset;
            mask_dst = in.dst.value;
            next_exec_subset = exec_subset;
        } else if (rdna2_instruction_may_change_exec(in)) {
            // CMPX only removes lanes from the current mask. Every other unhandled EXEC writer may
            // activate a lane whose destination was preserved by the zero-record load.
            if (!(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) &&
                !(in.fmt == Rdna2Format::SOP1 &&
                  (in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeAndSaveexecB64)))
                return false;
        }

        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = base + static_cast<int>(word);
                if (reg < 0 || reg >= 128) continue;
                mask_subset[reg] = false;
            }
        });
        if (derived_mask && mask_dst >= 0 && mask_dst + 1 < 128) {
            mask_subset[mask_dst] = true;
            mask_subset[mask_dst + 1] = true;
        }
        exec_subset = next_exec_subset;
    }
    return exec_subset;
}

size_t specialize_zero_record_execz_exits(std::vector<Rdna2Inst>& ins,
                                          const ShaderResourceTable* rt,
                                          uint32_t wave_size) {
    if (!rt || wave_size != 64 || ins.empty()) return 0;
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return (in.fmt == Rdna2Format::SOP1 &&
                    in.opcode >= kSop1OpcodeSetpcB64 &&
                    in.opcode <= kSop1OpcodeRfeB64) ||
                   (in.fmt == Rdna2Format::SOPK &&
                    (in.opcode == kSopkOpcodeCallB64 ||
                     in.opcode == kSopkOpcodeSubvectorLoopBegin ||
                     in.opcode == kSopkOpcodeSubvectorLoopEnd));
        }))
        return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);
    std::set<uint32_t> specialized_branches;

    for (const ShaderResource& resource : rt->resources) {
        if (!is_zero_record_raw_buffer(resource)) continue;
        // Translation resolves an instruction-scoped buffer through the table's first matching
        // fetch PC. A later zero marker must not override an earlier ordinary resource here.
        if (rt->by_fetch_pc(resource.fetch_pc) != &resource) continue;
        const auto load_found = index_by_pc.find(resource.fetch_pc);
        if (load_found == index_by_pc.end()) continue;
        const size_t load_index = load_found->second;
        const Rdna2Inst& load = ins[load_index];
        if (!zero_record_load_shape(load)) continue;

        for (size_t i = load_index + 1; i + 2 < ins.size(); ++i) {
            const Rdna2Inst& bit_and = ins[i];
            const Rdna2Inst& compare = ins[i + 1];
            const Rdna2Inst& branch = ins[i + 2];
            if (writes_vgpr(bit_and, load.dst.value)) break;
            const bool and_shape = bit_and.fmt == Rdna2Format::VOP2 &&
                bit_and.opcode == kVop2OpcodeAndB32 &&
                bit_and.len_dwords == 1u && !bit_and.has_sdwa &&
                !bit_and.has_modifier && !bit_and.has_dpp &&
                bit_and.dst.kind == OperandKind::VGPR &&
                bit_and.src[0].kind == OperandKind::InlineInt &&
                bit_and.src[0].value == 7 &&
                bit_and.src[1].kind == OperandKind::VGPR &&
                bit_and.src[1].value == load.dst.value;
            if (!and_shape) continue;
            const bool compare_shape = compare.pc == bit_and.pc + bit_and.len_dwords &&
                compare.fmt == Rdna2Format::VOPC &&
                compare.opcode == kVopcOpcodeCmpxEqU32 &&
                compare.len_dwords == 1u && !compare.has_sdwa &&
                !compare.has_modifier && !compare.has_dpp &&
                compare.src[0].kind == OperandKind::InlineInt &&
                compare.src[0].value == 5 &&
                compare.src[1].kind == OperandKind::VGPR &&
                compare.src[1].value == bit_and.dst.value;
            const bool branch_shape = branch.pc == compare.pc + compare.len_dwords &&
                branch.fmt == Rdna2Format::SOPP &&
                branch.opcode == kSoppOpcodeCbranchExecz && branch.simm16 > 0 &&
                index_by_pc.contains(scalar_branch_target(branch));
            if (!compare_shape || !branch_shape ||
                !zero_load_reaches_and_under_exec_subset(
                    ins, load_index, i, index_by_pc))
                continue;
            specialized_branches.insert(branch.pc);
            break;
        }
    }

    for (uint32_t pc : specialized_branches) {
        Rdna2Inst& branch = ins[index_by_pc.at(pc)];
        branch.opcode = kSoppOpcodeBranch;
        branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
    }
    if (!specialized_branches.empty()) prune_scalar_cfg_reachability(ins);
    return specialized_branches.size();
}

} // namespace

bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target) {
    return specialize_pcrel_dispatch(instructions, info, selected_target);
}

size_t rdna2_specialize_shader_constant_branches(
        std::vector<Rdna2Inst>& instructions) {
    return specialize_shader_constant_branches(instructions);
}

size_t rdna2_specialize_proven_null_bvh_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_proven_null_bvh_exits(instructions, resources, wave_size);
}

size_t rdna2_specialize_zero_record_execz_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_zero_record_execz_exits(instructions, resources, wave_size);
}

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords) {
    PcrelDispatchInfo out;
    if (!code || !dwords) return out;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    return detect_pcrel_dispatch(ins, code, dwords, program_dwords);
}

size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    size_t required = program_dwords;
    std::vector<Rdna2Inst> terminating_cfg = ins;
    size_t terminating_span = program_dwords;
    if (extend_terminating_if_else(code, dwords, terminating_cfg, &terminating_span))
        required = std::max(required, terminating_span);
    // Detection both proves the compiler idiom and bounds every referenced table. Do not retain an
    // arbitrary post-ENDPGM trailer: only bytes that can affect the generated SPIR-V belong in the key.
    (void)detect_pcrel_tables(terminating_cfg, code, dwords, &required);
    const PcrelDispatchInfo dispatch = detect_pcrel_dispatch(ins, code, dwords, program_dwords);
    if (dispatch.valid) required = std::max(required, dispatch.required_dwords);
    return std::min(required, dwords);
}

bool compute_trip_witness_active(uint64_t program_address) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    // Mirrors every arming rule that does not need the program's bytes. The GDS-use refusal is the
    // one exception and lives with the caller that already decodes the program (see gpu_executor's
    // uses_gds), so this must not be treated as the complete predicate.
    if (!settings.bound) return false;
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) return false;
    if (!settings.only_program) return true;
    return program_address == settings.only_program;
}

// Does THIS module write the trip-bound witness?
//
// Derived from the compiled artifact, not from process history. An earlier revision kept a global
// set of program addresses that had ever emitted one, which cannot express the contract the host
// needs: the set was monotonic and keyed only by address, so once a program emitted under one phase,
// recompiling the SAME address under a phase it does not have still answered "instrumented" -- and
// the host would then read and clear guest-visible dwords no shader in the current module writes.
//
// Reading the module removes the whole class: the answer is a property of the bytes the backend is
// about to run, so it cannot be stale, cannot be defeated by a shader-cache hit, and needs no
// invalidation. The witness's first field is published by an atomic through an OpAccessChain onto the
// internal GDS binding at kComputeTripWitnessDword, which nothing else emits.
bool spirv_writes_trip_witness(const std::vector<uint32_t>& spirv) {
    // FAIL CLOSED, and "well formed" means the exact signature this function relies on -- not merely
    // that the words parse. The result authorizes the host to write guest-visible GDS, so every step
    // that could be true by accident has to be pinned:
    //
    //   * a decorated ID must actually name an OpVariable (a decoration can outlive its target);
    //   * a candidate must not carry conflicting DescriptorSet/Binding values;
    //   * the instructions consumed must have their EXACT operand counts -- a truncated OpAtomicUMax
    //     whose declared length happens to end at the module boundary passes any `word + len` check;
    //   * the access chain must have the shape the builder emits, not merely enough operands.
    //
    // This predicate consumes prosper's own generator output, so accepting exactly that canonical
    // form and refusing everything else is both sufficient and the conservative choice. It does not
    // attempt general SPIR-V validation, and it is not a substitute for spirv-val.
    constexpr uint32_t kSpirvMagic = 0x07230203u;
    if (spirv.size() < 5 || spirv[0] != kSpirvMagic) return false;

    constexpr uint32_t OpDecorate = 71, OpVariable = 59, OpConstant = 43, OpAccessChain = 65,
                       OpAtomicUMax = 239;
    constexpr uint32_t DecorationBinding = 33, DecorationDescriptorSet = 34;
    // Exact word counts for the forms consumed below (opcode word included).
    constexpr uint32_t kDecorateLiteralWords = 4;      // OpDecorate target decoration literal
    constexpr uint32_t kVariableWords = 4;             // result-type result storage-class, NO init
    constexpr uint32_t kAccessChainWords = 6;          // result-type result base member-0 slot
    constexpr uint32_t kAtomicUMaxWords = 7;           // result-type result pointer scope sem value

    std::set<uint32_t> variables;
    std::map<uint32_t, uint32_t> descriptor_set, binding;
    std::set<uint32_t> conflicting;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;   // truncated stream: fail closed
        // EXACT, like every other instruction consumed as proof. The builder emits the internal-GDS
        // variable as four words with no initializer; a longer OpVariable is a different declaration
        // and this predicate has no business reasoning about it.
        if (op == OpVariable && len == kVariableWords) {
            variables.insert(spirv[word + 2]);
        } else if (op == OpDecorate && len == kDecorateLiteralWords) {
            const uint32_t target = spirv[word + 1], value = spirv[word + 3];
            auto record = [&](std::map<uint32_t, uint32_t>& into) {
                const auto existing = into.find(target);
                if (existing == into.end()) into.emplace(target, value);
                else if (existing->second != value) conflicting.insert(target);
            };
            if (spirv[word + 2] == DecorationDescriptorSet) record(descriptor_set);
            else if (spirv[word + 2] == DecorationBinding) record(binding);
        }
        word += len;
    }

    std::set<uint32_t> witness_variables;
    for (uint32_t id : variables) {
        if (conflicting.count(id)) continue;
        const auto set_it = descriptor_set.find(id);
        const auto binding_it = binding.find(id);
        if (set_it != descriptor_set.end() && set_it->second == 0u &&
            binding_it != binding.end() && binding_it->second == kComputeInternalGdsBinding)
            witness_variables.insert(id);
    }
    if (witness_variables.empty()) return false;

    std::set<uint32_t> zero_constants, slot_constants, witness_pointers;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;
        if (op == OpConstant && len == 4) {
            if (spirv[word + 3] == 0u) zero_constants.insert(spirv[word + 2]);
            else if (spirv[word + 3] == kComputeTripWitnessDword)
                slot_constants.insert(spirv[word + 2]);
        } else if (op == OpAccessChain && len == kAccessChainWords &&
                   witness_variables.count(spirv[word + 3]) &&
                   zero_constants.count(spirv[word + 4]) &&
                   slot_constants.count(spirv[word + 5])) {
            witness_pointers.insert(spirv[word + 2]);
        } else if (op == OpAtomicUMax && len == kAtomicUMaxWords &&
                   witness_pointers.count(spirv[word + 3])) {
            return true;
        }
        word += len;
    }
    return false;
}

// RDNA waves order their own LDS instructions, but the portable compute shell represents guest
// wave lanes as independent Vulkan invocations. A plain OpStore performed by lane 0 therefore does
// not publish its value to a following cross-lane atomic merely because every invocation reaches
// the atomic later in program order. GTA V's BVH bounds kernel uses this exact wave-synchronous
// idiom: select EXEC=1, initialize six adjacent dwords with one B64 and one B128 write, restore
// EXEC=-1, set vaddr=0, then issue three DS_MIN_F32 and three DS_MAX_F32 operations.
//
// Preserve the original byte-exact adjacent packet as a fast path: insert emitter-only S_BARRIERs
// before and after its six atomics. For a separated packet in one guest wave, ask emit_body to route
// every float atomic through the dispatcher's synchronized common phase. A proved lane-zero writer can
// retain ordinary stores; otherwise each preceding store becomes an atomic exchange, matching RDNA's
// serialized indexed bank conflicts without introducing a Vulkan write/write data race. The first
// common-phase barrier publishes those writes; each trailing barrier completes that atomic before the
// next dispatcher iteration or later gather. AcquireRelease on an individual atomic orders memory but
// is not an arrival barrier, so neither edge can be omitted. Every multi-wave separated shape rejects
// visibly. A real guest barrier, or an atomic with no preceding ordinary store in its phase, remains
// architectural and needs no synthesized edge.
struct LdsFminmaxSynchronization {
    bool needs_dispatcher = false;
    std::unordered_set<uint32_t> atomicized_store_pcs;
};

bool prepare_lds_fminmax_synchronization(std::vector<Rdna2Inst>& ins,
                                         RecompileDiagnosticContext diagnostic,
                                         bool at_most_one_guest_wave,
                                         LdsFminmaxSynchronization* synchronization = nullptr) {
    if (synchronization) *synchronization = {};
    auto ordinary_lds_store = [](const Rdna2Inst& in) {
        if (in.fmt != Rdna2Format::DS || in.ds_gds) return false;
        return in.opcode == 0x0d || in.opcode == 0x0e || in.opcode == 0x4d ||
               in.opcode == 0x4e ||
               in.opcode == 0xb0 || in.opcode == 0xde || in.opcode == 0xdf;
    };
    auto float_lds_atomic = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && !in.ds_gds &&
               (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32);
    };
    auto store_data_registers = [](const Rdna2Inst& in) {
        std::vector<int> registers;
        auto append = [&](int first, uint32_t count) {
            for (uint32_t word = 0; word < count; ++word)
                registers.push_back(first + static_cast<int>(word));
        };
        switch (in.opcode) {
            case 0x0d: case 0xb0: append(in.src[1].value, 1); break;
            case 0x0e:
                append(in.src[1].value, 1);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 1);
                break;
            case 0x4d: append(in.src[1].value, 2); break;
            case 0x4e:
                append(in.src[1].value, 2);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 2);
                break;
            case 0xde: append(in.src[1].value, 3); break;
            case 0xdf: append(in.src[1].value, 4); break;
            default: break;
        }
        return registers;
    };
    // Atomic exchange is equivalent to RDNA's indexed-bank serialization only when colliding lanes
    // write the same bits. Prove the narrow but generic form this family needs: every stored dword's
    // last writer is a plain v_mov from a scalar/literal source, and no control/EXEC edge can let a
    // store lane bypass that writer. Different addresses remain independent; equal addresses then
    // have identical candidate values, so the exchange winner is immaterial.
    auto store_data_are_wave_uniform = [&](size_t store_index) {
        const Rdna2Inst& store = ins[store_index];
        for (int reg : store_data_registers(store)) {
            size_t writer_index = ins.size();
            for (size_t j = store_index; j-- > 0;) {
                if (writes_vgpr(ins[j], reg)) {
                    writer_index = j;
                    break;
                }
            }
            if (writer_index == ins.size()) return false;
            const Rdna2Inst& writer = ins[writer_index];
            if (writer.fmt != Rdna2Format::VOP1 || writer.opcode != 0x01 ||
                writer.has_modifier || writer.has_sdwa || writer.has_dpp ||
                writer.src[0].kind == OperandKind::VGPR)
                return false;
            for (size_t j = writer_index + 1; j < store_index; ++j) {
                const Rdna2Inst& between = ins[j];
                if (rdna2_instruction_may_change_exec(between) ||
                    (between.fmt == Rdna2Format::SOPP && between.opcode >= 0x02u &&
                     between.opcode <= 0x12u && between.opcode != 0x03u &&
                     between.opcode != 0x0cu))
                    return false;
            }
            for (const Rdna2Inst& edge : ins) {
                if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                    edge.opcode <= 0x22u)
                    return false;
                if (edge.fmt != Rdna2Format::SOPP || edge.opcode < 0x02u ||
                    edge.opcode > 0x09u || edge.opcode == 0x03u)
                    continue;
                const uint32_t target = branch_target(edge);
                const bool source_outside = edge.pc < writer.pc || edge.pc >= store.pc;
                if (source_outside && target > writer.pc && target <= store.pc)
                    return false;
            }
        }
        return true;
    };
    auto words_are = [](const Rdna2Inst& in, uint32_t word0, uint32_t word1) {
        return in.words[0] == word0 && in.words[1] == word1;
    };

    static constexpr uint32_t kAtomicWord0[6] = {
        0xd8480000u, 0xd8480004u, 0xd8480008u,
        0xd84c000cu, 0xd84c0010u, 0xd84c0014u,
    };
    static constexpr uint32_t kAtomicWord1[6] = {
        0x00000900u, 0x00000a00u, 0x00000b00u,
        0x00000600u, 0x00000700u, 0x00000800u,
    };

    std::vector<size_t> phase_stores;
    bool phase_stores_are_single_lane = true;
    bool exec_is_single_lane = false;
    bool dispatcher_initializer_exec = false;
    uint32_t dispatcher_initializer_pc = UINT32_MAX;
    uint32_t phase_dispatcher_initializer_pc = UINT32_MAX;
    std::vector<size_t> synth_before;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) {
            phase_stores.clear();
            phase_stores_are_single_lane = true;
            phase_dispatcher_initializer_pc = UINT32_MAX;
            continue;
        }
        // These byte-exact scalar mask operations are the two lane-zero forms used by the known
        // initializers. A control-flow edge or any other EXEC writer ends the straight-line proof;
        // the dispatcher supplies ordering, not ownership of racing ordinary stores.
        if (in.words[0] == 0xbeea2481u || in.words[0] == 0xbefe0481u) {
            exec_is_single_lane = true; // s_and_saveexec_b64 vcc,1 / s_mov_b64 exec,1
            dispatcher_initializer_exec = in.words[0] == 0xbeea2481u;
            dispatcher_initializer_pc = dispatcher_initializer_exec ? in.pc : UINT32_MAX;
        } else if (rdna2_instruction_may_change_exec(in) ||
                   (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09) ||
                     in.opcode == 0x12))) {
            exec_is_single_lane = false;
            dispatcher_initializer_exec = false;
            dispatcher_initializer_pc = UINT32_MAX;
        }
        if (ordinary_lds_store(in)) {
            phase_stores.push_back(i);
            phase_stores_are_single_lane &= exec_is_single_lane;
            if (!dispatcher_initializer_exec ||
                (phase_dispatcher_initializer_pc != UINT32_MAX &&
                 phase_dispatcher_initializer_pc != dispatcher_initializer_pc))
                phase_stores_are_single_lane = false;
            else
                phase_dispatcher_initializer_pc = dispatcher_initializer_pc;
            continue;
        }
        if (!float_lds_atomic(in) || phase_stores.empty()) continue;

        bool exact = i >= 4 && i + 9 < ins.size() && phase_stores.size() == 2 &&
            phase_stores[0] == i - 4 && phase_stores[1] == i - 3 &&
            words_are(ins[i - 4], 0xd9340010u, 0x0000040cu) &&
            words_are(ins[i - 3], 0xdb7c0000u, 0x0000000cu) &&
            ins[i - 2].words[0] == 0xbefe04c1u && // s_mov_b64 exec, -1
            ins[i - 1].words[0] == 0x7e000280u;   // v_mov_b32 v0, 0
        for (size_t atomic = 0; exact && atomic < 6; ++atomic)
            exact = words_are(ins[i + atomic], kAtomicWord0[atomic], kAtomicWord1[atomic]);
        exact = exact &&
            ins[i + 6].words[0] == 0xbefe0481u && // pc81 s_mov_b64 exec, 1
            ins[i + 7].words[0] == 0x7e080280u && // pc82 v_mov_b32 v4, 0
            words_are(ins[i + 8], 0xdbfc0000u, 0x00000004u) && // pc83 ds_read_b128 v[0:3],v4
            words_are(ins[i + 9], 0xd9d80010u, 0x04000004u);   // pc85 ds_read_b64 v[4:5],v4

        bool found_lane0 = false;
        uint32_t lane0_pc = UINT32_MAX;
        if (exact) {
            for (size_t j = i - 4; j-- > 0;) {
                const Rdna2Inst& prefix = ins[j];
                if (prefix.words[0] == 0xbefe0481u) { // s_mov_b64 exec, 1
                    found_lane0 = true;
                    lane0_pc = prefix.pc;
                    for (size_t k = j + 1; k < i - 2; ++k) {
                        const Rdna2Inst& between = ins[k];
                        if (between.fmt == Rdna2Format::SOPP ||
                            (rdna2_instruction_may_change_exec(between) &&
                             between.words[0] != 0xbefe04c1u)) {
                            found_lane0 = false;
                            break;
                        }
                    }
                    break;
                }
                if (prefix.fmt == Rdna2Format::SOPP ||
                    rdna2_instruction_may_change_exec(prefix))
                    break;
            }
        }
        if (found_lane0) {
            const uint32_t last_store_pc = ins[i - 3].pc;
            for (const Rdna2Inst& edge : ins) {
                if (edge.pc >= lane0_pc || edge.fmt != Rdna2Format::SOPP ||
                    edge.opcode < 0x02 || edge.opcode > 0x09 || edge.opcode == 0x03)
                    continue;
                const uint32_t target = branch_target(edge);
                // The exact EXEC=1 writer must dominate both initializer stores. An edge from its
                // prefix may target the writer itself, but entering after it can leave EXEC full
                // and turn the supposedly single-writer OpStores into same-address races.
                if (target > lane0_pc && target <= last_store_pc) {
                    found_lane0 = false;
                    break;
                }
            }
        }
        if (!exact || !found_lane0) {
            bool dispatcher_initializer_dominates =
                phase_dispatcher_initializer_pc != UINT32_MAX;
            if (dispatcher_initializer_dominates) {
                const uint32_t last_store_pc = ins[phase_stores.back()].pc;
                for (const Rdna2Inst& edge : ins) {
                    // The deferred proof deliberately owns only a linear initializer region. Any
                    // scalar edge in that region could skip one of its lane-zero operations, while
                    // a later edge back into it could revisit a store after EXEC was restored full.
                    // Reject both, independent of whether the branch condition happens to look
                    // constant in this shader. GTA's captured initializer has no such edges.
                    if (edge.fmt == Rdna2Format::SOPP && edge.opcode >= 0x02 &&
                        edge.opcode <= 0x09 && edge.opcode != 0x03) {
                        const uint32_t target = branch_target(edge);
                        if (edge.pc <= last_store_pc || target <= last_store_pc) {
                            dispatcher_initializer_dominates = false;
                            break;
                        }
                    }
                    // An indirect PC update has no statically bounded target, so it cannot prove
                    // that the single-writer initializer is never re-entered.
                    if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                        edge.opcode <= 0x22u) {
                        dispatcher_initializer_dominates = false;
                        break;
                    }
                }
            }
            if (synchronization) {
                if (!at_most_one_guest_wave) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-recompile-reject", "terminal",
                        "pc=%u reason=multiwave-lds-fminmax-dispatcher", in.pc);
                    return false;
                }
                synchronization->needs_dispatcher = true;
                // A proven lane-zero initializer has one writer and can retain ordinary OpStores.
                // For a general one-wave initializer whose data is identical across active lanes,
                // make each exact preceding DS write an atomic exchange. RDNA serializes indexed
                // bank conflicts; equal colliding values make Vulkan's exchange winner immaterial.
                if (!phase_stores_are_single_lane || !dispatcher_initializer_dominates) {
                    if (!std::all_of(phase_stores.begin(), phase_stores.end(),
                                     store_data_are_wave_uniform)) {
                        log_recompile_diagnostic(
                            diagnostic, "compute-recompile-reject", "terminal",
                            "pc=%u reason=nonuniform-lds-store-before-ds-fminmax", in.pc);
                        return false;
                    }
                    for (size_t store : phase_stores)
                        synchronization->atomicized_store_pcs.insert(ins[store].pc);
                }
                phase_stores.clear();
                phase_stores_are_single_lane = true;
                phase_dispatcher_initializer_pc = UINT32_MAX;
                continue;
            }
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=unsynchronized-lds-store-before-ds-fminmax", in.pc);
            return false;
        }
        // EXEC=1 selects lane zero independently in every guest wave. The exact initializer uses
        // ordinary same-address stores, so more than one wave would race even though each wave has
        // only one active lane. Keep the title-observed single-wave workgroup admissible and leave a
        // different launch shape fail-visible until its cross-wave ownership can be proved.
        if (!at_most_one_guest_wave) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=multiwave-lds-fminmax-initializer", in.pc);
            return false;
        }

        synth_before.push_back(i);
        synth_before.push_back(i + 6);
        phase_stores.clear();
        phase_stores_are_single_lane = true;
        phase_dispatcher_initializer_pc = UINT32_MAX;
        i += 9; // both synthesized boundaries belong to this complete live atomic/gather group
    }

    std::vector<uint32_t> synth_pcs;
    synth_pcs.reserve(synth_before.size());
    for (size_t index : synth_before) synth_pcs.push_back(ins[index].pc);
    for (auto it = synth_before.rbegin(); it != synth_before.rend(); ++it) {
        Rdna2Inst barrier;
        barrier.pc = ins[*it].pc; // boundary immediately before the atomic at this guest PC
        barrier.fmt = Rdna2Format::SOPP;
        barrier.opcode = 0x0a;
        barrier.words[0] = 0xbf8a0000u;
        barrier.len_dwords = 0;   // emitter-only marker; never part of the guest byte stream
        ins.insert(ins.begin() + static_cast<std::ptrdiff_t>(*it), barrier);
    }

    // Prove each synthesized boundary is top-level even when the compact structurizer (rather than
    // the phase dispatcher) owns the program. Branches and loops may finish before the boundary,
    // but no edge may skip it, enter it from the far side, or carry only part of a workgroup back
    // across it. Traps/indirect PC changes before it also fail closed. This is the same edge
    // invariant used by the barrier-phase route without its unrelated >2-branch selection policy.
    for (uint32_t barrier_pc : synth_pcs) {
        bool uniform = true;
        for (const Rdna2Inst& in : ins) {
            if (in.pc < barrier_pc &&
                (in.is_end ||
                 (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) ||
                 (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)))
                uniform = false;
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                uniform = false;
        }
        if (!uniform) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=lds-fminmax-publication-barrier-not-workgroup-uniform",
                barrier_pc);
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt, uint32_t lds_bytes,
                                     uint32_t compute_pgm_rsrc1,
                                     bool force_cfg_for_test,
                                     uint32_t local_x_for_test,
                                     uint32_t threads_x_for_test) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // This shell publishes register state after the structured region, so both terminal arms can
    // converge in the host SPIR-V without inventing further guest execution. Graphics exports keep
    // separate side-effect bookkeeping and remain on the conservative reject path for this shape.
    (void)extend_terminating_if_else(code, dwords, ins);
    // The synthetic test shell is one Wave64 workgroup, matching the live GTA dispatch.
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, {}, true, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.compute_pgm_rsrc1 = compute_pgm_rsrc1;
    // Size the LDS array from the shader's real allocation when known (#130): bytes -> dwords, at
    // least the ds ops need, clamped to the RDNA2 64 KB (16384-dword) max. 0 keeps the 16 KB default.
    if (lds_bytes) {
        uint32_t dw = (lds_bytes + 3) / 4;
        b.lds_dwords = dw > 16384u ? 16384u : (dw ? dw : 1u);
    }
    if (!local_x_for_test || local_x_for_test > 1024) return {};
    b.begin(num_inputs ? num_inputs : 1, rt, local_x_for_test, 1, 1, 64, 0);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    const auto no_export = [](RegState&, const Rdna2Inst&){ return false; };
    const uint32_t initial_active =
        force_cfg_for_test && threads_x_for_test &&
                threads_x_for_test % local_x_for_test != 0
            ? b.invocation_within_extent(threads_x_for_test, 1, 1)
            : 0;
    const bool emitted = force_cfg_for_test
        ? emit_cfg_state_machine(
              b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
              /*allow_smem*/true, no_export, code, dwords, initial_active, false)
        : emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                    /*allow_smem*/true, no_export, code, dwords,
                    nullptr, true, 0, false, lds_fminmax_synchronization.needs_dispatcher);
    if (!emitted) return {};
    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's prior
    // value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

bool fragment_vcc_branch_is_wave_uniform_for_test(
        const uint32_t* code, size_t dwords, uint32_t branch_pc) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    return vcc_exit_is_wave_uniform(instructions, branch_pc);
}

std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config,
                                        RecompileDiagnosticContext diagnostic) {
    const bool has_null_guarded_raw_store = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_proven_null_guarded_raw_store);
    const bool has_nullable_output_raw_buffer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_nullable_raw_buffer_marker_candidate);
    const bool has_selected_sbuffer_descriptor = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_selected_sbuffer_marker_candidate);
    const bool has_gta5_packed_pointer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_packed_pointer_marker_candidate);
    const bool has_indirect_pointer_relocation = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_indirect_pointer_relocation_marker_candidate);
    const bool has_gta5_cf9200_no_backing = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_cf9200_no_backing_marker_candidate);
    const ShaderResource* selected_sbuffer_descriptor =
        has_selected_sbuffer_descriptor ? rt->by_fetch_pc(153u) : nullptr;
    // A resource table is externally constructible and can outlive the shader bytes or dispatch
    // that produced it. Re-establish the complete static guard and dynamic null-entry contract at
    // the final translation boundary before any marker is permitted to erase a real store.
    if (has_null_guarded_raw_store &&
        !rdna2_gta5_null_guarded_raw_store_dispatch(
            code, dwords, config.user_sgprs.data(), config.user_sgprs.size()))
        return {};
    if (has_nullable_output_raw_buffer &&
        !rdna2_gta5_nullable_output_dispatch(code, dwords, config, *rt))
        return {};
    if (has_selected_sbuffer_descriptor &&
        !rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, *rt))
        return {};
    if (has_gta5_packed_pointer &&
        !rdna2_gta5_packed_pointer_dispatch(code, dwords, config, *rt))
        return {};
    IndirectPointerRelocationProof indirect_pointer_proof;
    IndirectBufferRelocationInfo indirect_pointer_info;
    if (has_indirect_pointer_relocation &&
        !validate_rdna2_indirect_pointer_relocations(
            code, dwords, config, *rt,
            &indirect_pointer_proof, &indirect_pointer_info))
        return {};
    if (has_gta5_cf9200_no_backing &&
        !rdna2_gta5_cf9200_no_backing_dispatch(code, dwords, config, *rt))
        return {};
    const uint32_t local_x = std::max(1u, config.local_x);
    const uint32_t local_y = std::max(1u, config.local_y);
    const uint32_t local_z = std::max(1u, config.local_z);
    const uint32_t wave_size = config.wave_size == 32 ? 32u : 64u;
    const uint64_t local_count = static_cast<uint64_t>(local_x) * local_y * local_z;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // A dispatch-scoped proven-null BVH can collapse an exact no-hit exit and fully matched empty-stack
    // traversal cycle before generic shader-byte constant folding. Resource identity (including null
    // marker + fetch PC) is already part of the compute module cache key, so a later non-null dispatch
    // receives a distinct module.
    (void)rdna2_specialize_proven_null_bvh_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_zero_record_execz_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_shader_constant_branches(ins);
    // See recompile_valu: compute has no branch-external EXP state, so the common host-shell merge is
    // only a place to finish the invocation after either guest arm has terminated.
    (void)extend_terminating_if_else(code, dwords, ins);
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, diagnostic, local_count <= wave_size, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.diagnostic = diagnostic;
    b.gta5_selected_sbuffer_dispatch_validated = has_selected_sbuffer_descriptor;
    if (selected_sbuffer_descriptor && has_selected_sbuffer_descriptor)
        b.gta5_selected_sbuffer_soffset = selected_sbuffer_descriptor->selected_sbuffer_soffset;
    b.gta5_cf9200_no_backing_dispatch_validated = has_gta5_cf9200_no_backing;
    b.indirect_buffer_dispatch_validated = has_gta5_packed_pointer;
    if (has_gta5_packed_pointer) {
        const ShaderResource* packed = rt->by_fetch_pc(kGta5PackedPointerSourcePc);
        if (!packed || !is_gta5_packed_pointer_resource(*packed)) return {};
        b.indirect_buffer_binding = packed->binding;
        b.indirect_buffer_source_bytes = static_cast<uint32_t>(packed->size);
        b.indirect_buffer_slot_count = packed->indirect_buffer_slot_count;
        b.indirect_buffer_contract_tag = packed->indirect_buffer_contract_tag;
        b.indirect_buffer_header_bytes = packed->indirect_buffer_header_bytes;
        b.indirect_buffer_slot_bytes = packed->indirect_buffer_slot_bytes;
        const ShaderResource* atomic = rt->by_fetch_pc(kGta5PackedPointerAtomicSourcePc);
        if (!atomic || atomic->size != kGta5PackedPointerAtomicBindingBytes ||
            (atomic->gpu_addr & 7u) != 0u)
            return {};
        b.indirect_buffer_atomic_binding = atomic->binding;
        b.indirect_buffer_atomic_byte_offset = kGta5PackedPointerAtomicByteOffset;
    }
    if (has_indirect_pointer_relocation) {
        const ShaderResource* relocated = rt->by_fetch_pc(
            indirect_pointer_proof.source_fetch_pc);
        const auto& marker = relocated
            ? relocated->indirect_pointer_relocation
            : IndirectPointerRelocationBinding{};
        const uint32_t expected_carrier_version =
            indirect_pointer_proof.bound_kind ==
                    IndirectPointerBoundKind::StaticFootprint
                ? kIndirectPointerStaticFootprintLayout.version
                : indirect_pointer_proof.bound_kind ==
                        IndirectPointerBoundKind::DescriptorRange
                    ? kIndirectPointerDescriptorRangeLayout.version
                    : 0u;
        const uint64_t record_directory_offset =
            static_cast<uint64_t>(indirect_pointer_info.source_bytes) +
            kIndirectBufferRelocationHeaderBytes;
        const uint64_t segment_directory_offset = record_directory_offset +
            static_cast<uint64_t>(indirect_pointer_info.records.size()) *
                kIndirectBufferRelocationRecordBytes;
        if (!relocated || !is_indirect_pointer_relocation_resource(*relocated) ||
            !expected_carrier_version || marker.carrier_version != expected_carrier_version ||
            marker.record_count != indirect_pointer_info.records.size() ||
            marker.record_count != indirect_pointer_proof.record_count ||
            marker.segment_count != indirect_pointer_info.segments.size() ||
            indirect_pointer_info.source_bytes != relocated->size ||
            record_directory_offset > UINT32_MAX ||
            segment_directory_offset > UINT32_MAX ||
            marker.segment_directory_byte_offset != segment_directory_offset)
            return {};
        b.indirect_pointer_proof = &indirect_pointer_proof;
        b.indirect_pointer_binding = relocated->binding;
        b.indirect_pointer_source_bytes = indirect_pointer_info.source_bytes;
        b.indirect_pointer_record_count = marker.record_count;
        b.indirect_pointer_record_directory_byte_offset =
            static_cast<uint32_t>(record_directory_offset);
        b.indirect_pointer_segment_count = marker.segment_count;
        b.indirect_pointer_segment_directory_byte_offset =
            marker.segment_directory_byte_offset;
        b.indirect_pointer_payload_byte_offset =
            indirect_pointer_info.payload_byte_offset;
        b.indirect_pointer_carrier_bytes = marker.binding_bytes;
        b.indirect_pointer_source_stride = indirect_pointer_proof.source_stride;
        b.indirect_pointer_source_pointer_byte_offset =
            indirect_pointer_proof.pointer_byte_offset;
    }
    if (config.lds_bytes) {
        uint32_t dw = (config.lds_bytes + 3) / 4;
        b.lds_dwords = std::min(16384u, std::max(1u, dw));
    }
    const bool has_partial_workgroup = config.threads_x % local_x != 0 ||
                                       config.threads_y % local_y != 0 ||
                                       config.threads_z % local_z != 0;
    const BarrierPhasedCompute barrier_phases = analyze_barrier_phased_compute(ins);
    const bool partial_barrier_phases = config.exact_thread_extent && has_partial_workgroup &&
        barrier_phases.found && !barrier_phases.guarded;
    const bool exact_partial_dispatcher = config.exact_thread_extent &&
        has_partial_workgroup && (b.gta5_selected_sbuffer_dispatch_validated ||
                                  b.indirect_buffer_dispatch_validated ||
                                  has_indirect_pointer_relocation);
    b.native_subgroup_size = config.native_subgroup_size == wave_size &&
        local_count <= UINT32_MAX && local_count % wave_size == 0 ? wave_size : 0u;
    // A partial guest wave needs the portable dispatcher's per-lane ACTIVE bit. Native subgroup
    // operations cannot be entered by only the real prefix of the final host subgroup.
    if (partial_barrier_phases || exact_partial_dispatcher)
        b.native_subgroup_size = 0;
    // PROSPER_DBG: report the inputs to that decision, not just its outcome (#2429).
    //
    // Every wave-width-dependent lowering in this file gates on `b.native_subgroup_size` -- the
    // VCC-as-scalar-data path at :5604 most consequentially, since when it is 0 the guest's
    // `s_add_u32 sN, sM, vcc_lo` never resolves, the descriptor never lands, and the draw is
    // skipped. Nothing printed any of this, under any variable, so "was that path active on this
    // device?" could only be inferred from the adapter's advertised width.
    //
    // `config.native_subgroup_size` is NOT the adapter's advertised width -- it is the OUTPUT of
    // select_native_compute_subgroup_size() (gpu_executor.cpp), an adoption decision with THREE
    // `return 0` sites comprising 22 clauses -- 25 if `adoptable`'s four ANDed device checks are
    // counted individually, which is defensible since each is independently sufficient. It spans
    // device features, queue support, workgroup limits and the dispatch's own dimensions, and
    // yields 0 when it declines. Zero therefore means "no native width was adopted", NOT "the
    // device is narrower than wave_size", and the three cases below are distinguished for that
    // reason.
    //
    // Counted rather than estimated, because two lanes published two different guesses at it on the
    // same day (#2483 "~14", #2484 "roughly eight") and neither had derived the number.
    //
    // This line dedupes on the exact input tuple, so it answers "which combinations exist" cheaply --
    // a couple of lines for a whole boot. For a per-dispatch CENSUS (how many dispatches fall in
    // each category, which this instrument's dedupe destroys by design) use PROSPER_SUBGROUP_LOG in
    // gpu_executor.cpp instead.
    //
    // That inference is WRONG, and reporting only the effective value would preserve the error:
    // the expression above is zero for THREE independent reasons -- the device width not matching
    // `wave_size`, an implausible `local_count`, or a workgroup that is not a whole number of waves
    // (`local_count % wave_size`). A dispatch with a partial final wave disables the path on an
    // adapter whose width matches perfectly. #2429 attributes it entirely to the first cause, and
    // that is checkable only if all three inputs are printed.
    //
    // Deduplicated on the exact tuple rather than rate-limited, because the interesting event is a
    // DISTINCT combination appearing, not the hundredth repeat of one -- and a kernel that disables
    // the path for a different reason than its predecessors is exactly what a rate limit would drop.
    if (getenv("PROSPER_DBG")) {
        static std::mutex mx;
        // Keyed on the EXACT inputs, local_count included. An earlier revision packed
        // `local_count % wave_size` instead, which collapsed dispatches that differ only in
        // workgroup shape -- and the line prints `local=`, so one row then named whichever
        // instance arrived first and stood silently for the rest. Measured on GTA V's six
        // native=0 shapes, that key produced three rows, one of which represented 1024, 256,
        // 256 and 256 while printing only 1024 -- and the 256-wide ones were the multi-wave
        // case that mattered. A diagnostic may aggregate, but it must not name one member of
        // a bucket as though it were the bucket.
        static std::set<std::tuple<uint32_t, uint32_t, uint64_t, uint32_t>> seen;
        std::lock_guard<std::mutex> lk(mx);
        if (seen.insert(std::make_tuple(config.native_subgroup_size, wave_size,
                                        local_count, b.native_subgroup_size)).second)
            std::fprintf(stderr,
                         "[subgroup-width] device=%u wave=%u local=%llu local%%wave=%llu -> "
                         "native_subgroup_size=%u (%s)\n",
                         config.native_subgroup_size, wave_size,
                         (unsigned long long)local_count,
                         (unsigned long long)(local_count % (wave_size ? wave_size : 1u)),
                         b.native_subgroup_size,
                         b.native_subgroup_size
                             ? "width-dependent lowerings ENABLED"
                             : (partial_barrier_phases
                                    ? "DISABLED: partial barrier phases require the portable "
                                      "dispatcher"
                                : (config.native_subgroup_size == 0
                                    ? "DISABLED: no native width adopted -- "
                                      "select_native_compute_subgroup_size() declined"
                                    : (config.native_subgroup_size != wave_size
                                           ? "DISABLED: adopted width != wave_size"
                                           : (local_count > UINT32_MAX
                                                  ? "DISABLED: local_count exceeds the plausibility "
                                                    "guard"
                                                  : "DISABLED: workgroup is not a whole number "
                                                    "of waves")))));
    }
    b.native_storage_format_support = config.native_storage_format_support;
    b.storage_buffer_int64_atomics = config.storage_buffer_int64_atomics;
    b.packed_r11_storage = config.packed_r11_storage;
    b.compute_pgm_rsrc1 = config.compute_pgm_rsrc1;
    b.begin(1, rt, local_x, local_y, local_z, wave_size,
            static_cast<uint32_t>(config.user_sgprs.size()));
    b.allow_b32_masks = wave_size == 32;
    if (has_indirect_pointer_relocation &&
        indirect_pointer_proof.bound_kind ==
            IndirectPointerBoundKind::DescriptorRange)
        b.declare_indirect_pointer_descriptor_capture();
    b.declare_guest_scratch(scratch);
    uint32_t initial_dispatch_active = 0;
    if (partial_barrier_phases || exact_partial_dispatcher)
        initial_dispatch_active = b.invocation_within_extent(
            config.threads_x, config.threads_y, config.threads_z);
    else if (config.exact_thread_extent && has_partial_workgroup)
        b.guard_invocation_extent(config.threads_x, config.threads_y, config.threads_z);

    RegState rs;
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.exec = b.btrue();
    // Inline descriptors are represented by the resource table, not scalar SSA values. Leaving
    // their SGPR range absent also preserves the existing direct-provenance rule: a format MUBUF may
    // fall back to by_sgpr_base only while its SRSRC has not been overwritten by shader code.
    std::set<uint32_t> descriptor_sgprs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            uint32_t words = (resource.cls == ResourceClass::Texture ||
                              resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; word++)
                descriptor_sgprs.insert(resource.sgpr_base + word);
        }
    }
    for (size_t i = 0; i < config.user_sgprs.size(); i++) {
        const uint32_t value = b.load_push_constant(static_cast<uint32_t>(i));
        if (descriptor_sgprs.count(static_cast<uint32_t>(i)))
            rs.sreg_input[static_cast<int>(i)] = value;
        else
            rs.sreg[static_cast<int>(i)] = value;
    }

    rs.vreg[0] = b.localid_comp[0];
    if (config.tidig_comp_cnt >= 1) rs.vreg[1] = b.localid_comp[1];
    if (config.tidig_comp_cnt >= 2) rs.vreg[2] = b.localid_comp[2];

    int system_sgpr = static_cast<int>(config.user_sgprs.size());
    if (config.tgid_x_en) rs.sreg[system_sgpr++] = b.groupid[0];
    if (config.tgid_y_en) rs.sreg[system_sgpr++] = b.groupid[1];
    if (config.tgid_z_en) rs.sreg[system_sgpr++] = b.groupid[2];
    if (config.tg_size_en)
        rs.sreg[system_sgpr] = b.uconst(local_x * local_y * local_z);

    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/true, [](RegState&, const Rdna2Inst&) { return false; },
                   code, dwords, nullptr, true, initial_dispatch_active, false,
                   lds_fminmax_synchronization.needs_dispatcher))
        return {};
    // Exact dispatch contracts execute their partial final wave through the CFG dispatcher's ACTIVE
    // bit. Padded Vulkan lanes stay in the dispatcher and its synthesized workgroup barriers, but
    // cannot execute guest memory effects. The full program, launch, and resource proof above is the
    // authority boundary for extending the selected-SBUFFER path to the packed-pointer program.
    if (exact_partial_dispatcher && b.uses_barrier)
        b.partial_barrier_phases_emitted = true;
    // The entry guard is intentionally divergent only in the final partial workgroup. Vulkan requires
    // every workgroup invocation to participate uniformly in OpControlBarrier, including barriers the
    // recompiler synthesizes for wave operations. Reject this uncommon combination instead of emitting
    // a module that could deadlock or observe undefined workgroup-memory behavior.
    if (has_partial_workgroup && b.uses_barrier && !b.partial_barrier_phases_emitted) {
        log_recompile_diagnostic(
            b.diagnostic, "compute-recompile-reject", "terminal",
            "reason=partial-workgroup-barrier threads=%ux%ux%u local=%ux%ux%u",
            config.threads_x, config.threads_y, config.threads_z,
            local_x, local_y, local_z);
        return {};
    }
    return b.finish();
}

bool compute_shader_prefers_native_multiwave(const std::vector<Rdna2Inst>& ins,
                                             const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    bool low = false;
    bool high = false;
    bool guest_barrier = false;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3) {
            low |= in.opcode == 0x365;
            high |= in.opcode == 0x366;
        }
        guest_barrier |= in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a;
        if (low && high) return true;
    }
    if (!guest_barrier || !code) return false;

    // The guarded phase splitter's shared proof is stronger than the whole-stream CFG shape:
    // branches and loops inside one barrier-free phase do not make the outer uniform guard or its
    // barriers divergent. Such kernels need an exact native subgroup for nested guest-wave votes.
    if (const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        phased.found && phased.guarded)
        return true;

    // Mirror the conservative, acyclic subset of emit_body's structured-compute admission. Counting
    // raw VCC/EXEC opcodes is insufficient: kill-mask branches may be safely linearized, loop exits
    // are owned by another emitter, and rejected CFGs never reach guest_wave_any. Requiring the same
    // accepted ForwardIf regions proves that portable lowering really emits two scratch barriers per
    // counted vote and that exact-subgroup lowering removes them. Loops deliberately stay behind the
    // explicit experiment until their additional compute guards are shared with this analysis.
    auto safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    const std::vector<DivLoop> loops =
        detect_divergent_loops(ins, safe, /*fragment*/false, diagnostic, "multiwave-probe");
    if (!loops.empty()) return false;

    bool rejected = false;
    const std::vector<ForwardIf> branches = detect_forward_ifs(
        ins, /*allow_vcc*/false, code, dwords, &safe, nullptr, &rejected,
        /*compute_wave_branches*/true, diagnostic);
    if (rejected) return false;

    auto top_level_pc = [&](uint32_t pc) {
        for (const ForwardIf& parent : branches) {
            const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
            if (parent.branch_pc < pc && pc < parent_end) return false;
        }
        return true;
    };
    const bool barriers_are_top_level = std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a || top_level_pc(in.pc);
    });
    if (!barriers_are_top_level) return false;

    size_t structured_wave_votes = 0;
    for (const ForwardIf& branch : branches) {
        if (!branch.on_exec && !branch.on_vcc) continue;
        if (!top_level_pc(branch.branch_pc)) return false;
        ++structured_wave_votes;
    }
    // Four proven scratch-emulated votes keep the default narrower than the all-multi-wave experiment.
    return structured_wave_votes >= 4;
}

bool compute_shader_prefers_native_multiwave(const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    return compute_shader_prefers_native_multiwave(ins, code, dwords, diagnostic);
}

// See FlatLoadInfo / analyze_flat_loads in the header (#1171). A `base + offset` flat address VGPR pair
// v[N:N+1] is computed as vN = add(base_lo_sgpr, offset_lo) [carry-out] and v(N+1) =
// add(base_hi_sgpr, offset_hi, carry) [carry-in], where s[base_lo:base_hi] are consecutive user SGPRs
// holding a 64-bit guest pointer (the low kernel-arg pointer dword feeds the low address dword). We
// identify the base by the NEAREST prior definition of each address dword being an integer add that
// reads a user-range SGPR, and require the low dword to add the SGPR one below the high dword's.
// Anything else (a non-add producer, a non-user SGPR, a store/atomic/LDS/global-with-saddr form) leaves
// the load unresolved so the caller keeps failing visibly. flat_access_info lives in this TU's
// anonymous namespace but is visible here (internal linkage is still TU-wide).
// Resolution is over LINEAR program order (not CFG-aware). That is exact for the target decode kernels
// (the address adds sit in the same block immediately before the load) and the base is a loop-invariant
// kernel-arg pointer, so the resolved base is stable across loop iterations; the executor's
// guest_readable_mapping_containing validation is the runtime backstop against a bogus base.
FlatLoadAnalysis analyze_flat_loads(const uint32_t* code, size_t dwords, uint32_t user_sgpr_count) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    FlatLoadAnalysis out;

    auto writes_vgpr = [](const Rdna2Inst& d, uint32_t reg) {
        return d.dst.kind == OperandKind::VGPR && static_cast<uint32_t>(d.dst.value) == reg;
    };
    // If `d` is an integer add that could produce a 64-bit-pointer dword, return the user-range SGPR it
    // adds (else -1). Recognizes the add_co / add_co_ci / add_nc forms a compiler emits for pointer
    // arithmetic: VOP2 0x25 (add_nc) / 0x28 (add_co_ci), VOP3 0x30F (add_co) / 0x125 (add_nc) / 0x128
    // (add_co_ci).
    auto add_user_sgpr = [&](const Rdna2Inst& d) -> int32_t {
        const bool is_add =
            (d.fmt == Rdna2Format::VOP2 && (d.opcode == 0x25 || d.opcode == 0x28)) ||
            (d.fmt == Rdna2Format::VOP3 &&
             (d.opcode == 0x30F || d.opcode == 0x125 || d.opcode == 0x128));
        if (!is_add) return -1;
        for (int s = 0; s < 2; ++s)
            if (d.src[s].kind == OperandKind::SGPR && d.src[s].value >= 0 &&
                static_cast<uint32_t>(d.src[s].value) < user_sgpr_count)
                return d.src[s].value;
        return -1;
    };
    // The user-range SGPR added by the nearest prior writer of `reg` (or -1 if that writer is not a
    // user-SGPR add). We trust only the immediate definition — a later non-add redefinition breaks the
    // pattern and must not be skipped over.
    auto prior_add_sgpr = [&](uint32_t reg, size_t before) -> int32_t {
        for (size_t j = before; j-- > 0;)
            if (!ins[j].is_end && writes_vgpr(ins[j], reg))
                return add_user_sgpr(ins[j]);
        return -1;
    };

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::FLAT) continue;
        if (in.flat_segment == 1u) continue;   // scratch spill: analyze_static_scratch owns it
        out.any = true;
        const FlatAccessInfo access = flat_access_info(in.opcode);
        FlatLoadInfo info;
        info.load_pc = in.pc;
        info.vaddr_lo_reg = in.src[0].kind == OperandKind::VGPR
                                ? static_cast<uint32_t>(in.src[0].value) : 0u;
        info.dst_reg = static_cast<uint32_t>(in.dst.value);
        info.bits = access.bits;
        info.components = access.components;
        info.sign_extend = access.sign_extend;
        // Resolvable only as a plain LOAD with a VGPR address pair and a null SADDR (true flat segment).
        const bool shape = access.valid && !access.store && !in.flat_lds &&
                           in.src[0].kind == OperandKind::VGPR &&
                           in.src[1].kind == OperandKind::Special && in.src[1].value == 125;
        if (shape) {
            const uint32_t lo = info.vaddr_lo_reg, hi = lo + 1;
            const int32_t hi_base = prior_add_sgpr(hi, i);
            const int32_t lo_base = prior_add_sgpr(lo, i);
            if (hi_base > 0 && lo_base == hi_base - 1)
                info.base_sgpr = lo_base;
        }
        if (info.base_sgpr < 0) out.all_resolved = false;
        out.loads.push_back(info);
    }
    return out;
}

std::vector<uint32_t> safe_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> s = safe_execz_branches(ins);
    return std::vector<uint32_t>(s.begin(), s.end());
}

std::vector<uint32_t> structured_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    bool rejected = false;
    const auto branches = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords, &safe,
                                             nullptr, &rejected,
                                             /*compute_wave_branches*/true);
    std::vector<uint32_t> pcs;
    if (!rejected) for (const auto& branch : branches) pcs.push_back(branch.branch_pc);
    return pcs;
}

std::vector<uint32_t> mask_test_branches_for_test(const uint32_t* code, size_t dwords,
                                                  bool wave32) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto branches = mask_test_branches(ins, wave32);
    return std::vector<uint32_t>(branches.begin(), branches.end());
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords,
                                     std::vector<RecompileUnsupportedSite>* sites) {
    if (sites) sites->clear();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t synthetic_branch_pc = UINT32_MAX;
    (void)extend_terminating_if_else(code, dwords, ins, nullptr, &synthetic_branch_pc);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
    b.cselect_b64_low_only_pcs = proven_cselect_b64_low_only_pcs(ins);
    b.cselect_b64_low_only_analysis_done = true;
    // Coverage is deliberately a context-free instruction census. Preserve its historical
    // CSELECT exception, but do not let a newly recognized whole-CFG logical lifetime poison the
    // scratch compute state and make later instructions look unsupported. Real compute emission
    // runs the complete proof in emit_body.
    b.vcc_b32_low_only_pcs =
        proven_wave64_vcc_b32_low_only_pcs(ins, /*include_logical*/false);
    b.vcc_b32_low_only_analysis_done = true;
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // emit_alu is a per-instruction check and rejects control-flow branches, but the whole-stream emit_body
    // RECONSTRUCTS a counted loop and a forward uniform-if. Credit the branches emit_body consumes (loop
    // back-edge + exit, and the forward-if branch) as handled, so coverage matches what actually recompiles
    // (previously the MSAA-resolve loop shaders 031-034 were mis-flagged "blocked" at their s_cbranch_scc0).
    const CountedLoop cL = detect_counted_loop(ins);
    std::vector<ForwardIf> cFs;
    if (cL.found) {
        // Match emit_body's counted-loop composition: inspect the truncated prefix independently
        // from the loop exit/back-edge, then inspect the recursively-emitted suffix. Feeding the
        // complete stream to detect_forward_ifs makes the canonical exit look like an IF whose
        // alleged else terminator is the backward loop branch, so otherwise-valid prefix/postfix
        // branches are incorrectly counted as unsupported.
        std::vector<Rdna2Inst> prefix;
        for (const auto& in : ins) {
            if (in.pc >= cL.header_pc) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= cL.header_pc) continue;
            prefix.push_back(in);
        }
        Rdna2Inst prefix_end;
        prefix_end.pc = cL.header_pc;
        prefix_end.is_end = true;
        prefix.push_back(prefix_end);
        bool prefix_rejected = false;
        std::vector<ForwardIf> prefix_ifs = detect_forward_ifs(
            prefix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr,
            &prefix_rejected, /*compute_wave_branches*/true);
        const bool prefix_crosses_loop = std::any_of(
            prefix_ifs.begin(), prefix_ifs.end(), [&](const ForwardIf& branch) {
                return branch.early_out ||
                    (branch.has_else ? branch.merge_pc : branch.target_pc) > cL.header_pc;
            });
        if (!prefix_rejected && !prefix_crosses_loop)
            cFs.insert(cFs.end(), prefix_ifs.begin(), prefix_ifs.end());

        std::vector<Rdna2Inst> suffix;
        for (const auto& in : ins) if (in.pc >= cL.exit_pc) suffix.push_back(in);
        std::vector<ForwardIf> suffix_ifs = detect_forward_ifs(
            suffix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr, nullptr,
            /*compute_wave_branches*/true);
        cFs.insert(cFs.end(), suffix_ifs.begin(), suffix_ifs.end());
    } else {
        cFs = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords,
                                 nullptr, nullptr, nullptr,
                                 /*compute_wave_branches*/true);   // matches the compute shell (#590)
    }
    auto cf_reconstructed = [&](const Rdna2Inst& i) {
        if (cL.found && (i.pc == cL.backedge_pc || i.pc == cL.exit_branch_pc)) return true;
        for (const auto& F : cFs)
            if (i.pc == F.branch_pc || (F.has_else && i.pc == F.sb_pc)) return true;
        return false;
    };

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        // The rewritten first s_endpgm is compiler-only control flow. It lets the structured emitter
        // reuse its ordinary if/else path, but coverage describes decoded guest instructions and has
        // always excluded s_endpgm, so do not inflate total/ALU with this synthetic arm skip.
        if (in.pc == synthetic_branch_pc) continue;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        const SavedB64MaskSnapshot saved_masks = snapshot_saved_b64_masks(rs, in);
        const bool emitted = emit_alu(
            b, rs, in, ok, /*allow_exec_update*/true, &safe_branches,
            /*allow_smem*/true, /*rt*/nullptr, /*allow_wave*/true);
        if (emitted && ok)
            record_scalar_write(
                rs, in,
                allows_compute_scalar_vcc_bridge(b), saved_masks);
        bool handled = cf_reconstructed(in) || (emitted && ok);
        // Shapes the recompiler handles only in context (a resource table for MIMG sample/load/LOD/store
        // and buffer_load/store_format; a fragment stage for VINTRP). This table-less compute-shell pass
        // rejects them, so count them apart from truly-unsupported (cross-lane, etc.). Instruction-aware
        // for MIMG so deferred variants (NSA multi-dword addr; arrayed/cube/MSAA dims) are NOT overcounted
        // as recompilable — they still land in `unsupported`, matching what the recompiler actually accepts.
        auto table_dependent = [](const Rdna2Inst& i) {
            switch (i.fmt) {
                case Rdna2Format::MIMG: {
                    // Storage load/store handle 1D/2D/3D + 1D/2D_ARRAY (dims 0,1,2,4,5) and NSA.
                    // Sampled 2D_MSAA IMAGE_LOAD is narrower: only the exact consecutive-address or
                    // one-extra NSA [x,y,sample] shapes accepted by emit_alu are table-dependent.
                    // Do not credit dim7 or unused NSA address bytes merely because a T# could exist.
                    const bool st_dim = i.mimg_dim <= 2u || i.mimg_dim == 4u || i.mimg_dim == 5u;
                    if (i.opcode == 0x00u) {
                        if (st_dim) return true;
                        const bool msaa_address_shape = i.len_dwords == 2u ||
                            (i.len_dwords == 3u && (i.words[2] & 0xffff0000u) == 0u);
                        return i.mimg_dim == 6u && !i.mimg_unorm && !i.has_modifier &&
                               msaa_address_shape;
                    }
                    if (i.opcode == 0x01u || i.opcode == 0x09u)
                        return rdna2_mimg_zero_mip_shape(i);
                    if (i.opcode == 0x08u) return st_dim;                       // image_store (no per-sample MSAA store)
                    if (i.opcode == 0x0fu || i.opcode == 0x11u)   // image_atomic_swap/add R32_UINT 2D / 2D_ARRAY
                        // #2265: 2D_ARRAY (dim 5) admitted alongside 2D. This is the COVERAGE
                        // predicate -- it decides whether the instruction counts as supported for
                        // the census, and it was the last of the four sites still reporting the
                        // arrayed form as unsupported after #2272 widened the lowering. A 2D_ARRAY
                        // arrayed atomic reaches its layer through the address VGPRs, not through a
                        // longer ENCODING: CrossWorlds' own instruction is `dim=5 ... len=2`, so
                        // `len_dwords` is the NSA-vs-packed encoding length and pinning it to 3 for
                        // dim 5 would reject exactly the instruction this admits. Matches the
                        // lowering gate, which rejects only `len_dwords < 2`.
                        return (i.mimg_dim == 1u || i.mimg_dim == 5u) && i.mimg_dmask == 1u &&
                               !i.mimg_unorm && i.len_dwords >= 2u;
                    // image_get_resinfo 1D/2D/3D, plus 2D_ARRAY (dim 5), which lowers as Dim_2D with
                    // Arrayed and reports the layer count as its third result (#2790). CUBE (dim 3) is
                    // still declined: its stacked-face lowering (#273) would have to promise face-count
                    // semantics for the third result that this has not established.
                    if (i.opcode == 0x0eu) return i.mimg_dim <= 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x60u)                                     // fragment image_get_lod 2D
                        return i.mimg_dim == 1u && i.len_dwords == 2u &&
                               !mimg_get_lod_has_unmodeled_controls(i) &&
                               (i.mimg_dmask & 0x3u) && !(i.mimg_dmask & ~0x3u);
                    // sample*: 2D (NSA ok); plus implicit-LOD image_sample (0x20) / LOD-0 image_sample_lz
                    // (0x27) from a 3D texture; sample_b (0x25) and gather4_lz (0x47) are 2D. 2D_ARRAY (dim 5)
                    // is accepted for all sample paths and handled as its base 2D slice (array index dropped,
                    // #325) — so array-sampling draws recompile+render instead of being skipped.
                    // 0xa0 is the high-bit sibling of image_sample (0x20), lowered identically (GTA V, #1145).
                    if (i.opcode == 0x20u || i.opcode == 0x27u || i.opcode == 0xa0u)
                        return i.mimg_dim == 1u || i.mimg_dim == 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x22u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    if (i.opcode == 0x2fu)
                        return i.mimg_dim == 1u || i.mimg_dim == 3u || i.mimg_dim == 5u;
                    if (i.opcode == 0x24u || i.opcode == 0x25u || i.opcode == 0x47u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u ||                    // load/store_format_*
                                                 (i.opcode >= 0x0Cu && i.opcode <= 0x0Fu);  // load_dword/x2/x4/x3 (need the V#)
                case Rdna2Format::MTBUF:  return i.opcode <= 0x07u && !i.mtbuf_tfe;
                // Wide scalar loads are descriptor-table fetches. A real resource table lets the
                // emitter preserve their provenance without reading a fallback buffer, including the
                // register-offset bindless form; the table-less coverage shell must not call that a
                // newly unsupported instruction.
                case Rdna2Format::SMEM:   return i.opcode == 0x02u || i.opcode == 0x03u;
                case Rdna2Format::VINTRP: return true;               // handled in the fragment shell
                default: return false;
            }
        };
        if (handled) { cov.alu++; }
        else if (table_dependent(in)) { cov.table_dependent++; }
        else {
            cov.unsupported++;
            if (cov.first_bad_fmt < 0) {
                cov.first_bad_fmt = (int)in.fmt;
                cov.first_bad_op = in.opcode;
                cov.first_bad_pc = in.pc;
            }
            // Recorded from the SAME branch that increments the counter, so the enumeration cannot
            // disagree with `unsupported` about which instructions are in the class.
            if (sites) sites->push_back({(int)in.fmt, in.opcode, in.pc});
        }
    }
    return cov;
}

std::vector<uint32_t> cselect_b64_low_only_pcs_for_test(
        const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto proven = proven_cselect_b64_low_only_pcs(ins);
    std::vector<uint32_t> result(proven.begin(), proven.end());
    std::sort(result.begin(), result.end());
    return result;
}

static uint64_t shader_program_hash(const uint32_t* code, size_t dwords) {
    uint64_t hash = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const uint8_t*>(code);
    for (size_t i = 0; i < dwords * sizeof(uint32_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint32_t effective_fragment_wave_size(uint32_t requested_wave_size,
                                             size_t program_dwords,
                                             uint64_t program_hash) {
    if (requested_wave_size != 32 && requested_wave_size != 64) return 0;
    // Compatibility for the one captured Astro fragment whose older producer omitted
    // SPI_PS_IN_CONTROL.PS_W32_EN. Its complete byte identity proves the same Wave32 contract; that
    // contract must select both one-word mask semantics and a 32-lane native subgroup.
    const bool legacy_wave32 = requested_wave_size == 64 && program_dwords == 3142 &&
        program_hash == 0x616dd4c0b241fbb1ull;
    return requested_wave_size == 32 || legacy_wave32 ? 32u : 64u;
}

uint32_t fragment_effective_wave_size_for_test(uint32_t requested_wave_size,
                                               size_t program_dwords,
                                               uint64_t program_hash) {
    return effective_fragment_wave_size(requested_wave_size, program_dwords, program_hash);
}

uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords) {
    uint32_t packed = 0;
    // One nibble per MRT, MRT0..MRT7. Sized 2 until 2026-08-15, which silently forced
    // `write_mask &= 0` for slots 2..7 at gpu_execute.hpp's EXP.EN gate -- so a shader exporting to
    // MRT2+ had those attachments dropped no matter what CB_TARGET_MASK and CB_SHADER_MASK said.
    std::array<bool, kFragmentColorOutputs> realized{};
    // This property needs no retained instructions. Match rdna2_walk's bounded decode and
    // termination rules without allocating/copying a complete instruction vector on every draw.
    for (size_t pc = 0; pc < dwords;) {
        const Rdna2Inst in = rdna2_decode_one(code + pc, dwords - pc);
        if (in.is_end || in.fmt == Rdna2Format::Unknown) break;
        if (in.fmt == Rdna2Format::EXP && in.exp_target < realized.size() &&
            !realized[in.exp_target] && in.exp_en != 0) {
            packed |= (in.exp_en & 0xFu) << (in.exp_target * 4u);
            realized[in.exp_target] = true;
        }
        if (in.len_dwords == 0) break;
        pc += in.len_dwords;
    }
    return packed;
}

static std::vector<uint32_t> recompile_fragment_impl(
        const uint32_t* code, size_t dwords,
        const ShaderResourceTable* rt,
        const PixelSystemInputMapping* system_inputs,
        uint32_t pcrel_dispatch_target,
        const FragmentInterpolationLayout* interpolation,
        uint32_t wave_size,
        RecompileDiagnosticContext diagnostic) {
    if (wave_size != 32 && wave_size != 64) return {};
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    if (pcrel_dispatch_target != UINT32_MAX) {
        const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        if (!specialize_pcrel_dispatch(ins, dispatch, pcrel_dispatch_target)) {
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "pcrel dispatch specialization target=%u",
                                     pcrel_dispatch_target);
            return {};
        }
    }
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // Preserve hardware target locations. MRT0 and MRT1 are backed by real Vulkan attachments; later
    // targets remain unsupported and must never be silently remapped to location 0 (#635).
    constexpr uint32_t kMrtzDepth = 1u << 0;
    constexpr uint32_t kMrtzStencil = 1u << 1;
    constexpr uint32_t kMrtzSampleMask = 1u << 2;
    constexpr uint32_t kMrtzAlpha = 1u << 3;
    constexpr uint32_t kUnsupportedMrtz = kMrtzStencil | kMrtzAlpha;
    uint32_t color_mask = 0;
    bool has_null_export = false;
    bool has_depth_export = false;
    bool has_sample_mask_export = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target < kFragmentColorOutputs) color_mask |= 1u << in.exp_target;
        else if (in.exp_target == 8 && !in.exp_compr) {
            has_depth_export |= (in.exp_en & kMrtzDepth) != 0;
            has_sample_mask_export |= (in.exp_en & kMrtzSampleMask) != 0;
        }
        else if (in.exp_target == 9) has_null_export = true;
    }
    // A NULL export is a real fragment-shader terminator. Depth/stencil-only draws use it after
    // narrowing EXEC to the surviving samples, so the module intentionally has no color outputs.
    // Keep other unsupported MRT-only programs fail-visible instead of accepting every no-color PS.
    if (!color_mask && !has_null_export && !has_depth_export && !has_sample_mask_export) {
        if (pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(
                diagnostic, "recompile-reject", "terminal",
                "fragment has no supported export dwords=%zu ins=%zu first=%08x last=%08x "
                "pcrel-target=%u",
                dwords, ins.size(), dwords ? code[0] : 0u,
                dwords ? code[dwords - 1] : 0u, pcrel_dispatch_target);
        else
            log_recompile_diagnostic(
                diagnostic, "recompile-reject", "terminal",
                "fragment has no supported export dwords=%zu ins=%zu first=%08x last=%08x",
                dwords, ins.size(), dwords ? code[0] : 0u,
                dwords ? code[dwords - 1] : 0u);
        return {};
    }

    const FragmentInterpolationLayout derived_interpolation = interpolation
        ? *interpolation : fragment_interpolation_layout(code, dwords, system_inputs);
    if (!derived_interpolation.valid) {
        log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                 "invalid fragment interpolation layout");
        return {};
    }
    const uint32_t effective_wave_size = effective_fragment_wave_size(
        wave_size, program_dwords, shader_program_hash(code, program_dwords));
    SpirvCompute b;
    b.diagnostic = diagnostic;
    b.wave_size = effective_wave_size;
    b.begin_fragment(rt, color_mask);
    // SPI_PS_IN_CONTROL.PS_W32_EN proves that EXEC_HI/VCC_HI are unused and the low-half mask
    // operations below represent the complete wave. Keep the older byte-exact captured exception
    // until every replay/capture producer carries the stage register into this entry point.
    b.allow_b32_masks = effective_wave_size == 32;
    // Fragment I/O value tap (PROSPER_FS_TAP=draw:pc): redirect the MRT0 colour export to the intermediate
    // VGPR produced at that PC so the rendered frame visualises the value. The `draw:` prefix is consumed by
    // gpu_replay (which re-recompiles only that draw's FS). Parse the same complete selector here so an
    // invalid or overflowing PC cannot silently become PC zero or truncate to 32 bits.
    if (const char* tap = getenv("PROSPER_FS_TAP")) {
        uint64_t draw = 0;
        uint32_t pc = 0;
        if (parse_fragment_tap_selector(tap, draw, pc)) b.tap_pc = pc;
    }
    b.declare_guest_scratch(scratch);
    b.fragment_interpolation = &derived_interpolation;
    // P0-only attributes retain the cheap Flat varying path. Mixed smooth/explicit-parameter reads
    // are legal with the portable geometry stage and use separate packed locations there.
    if (!derived_interpolation.requires_geometry) {
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::VINTRP && in.opcode == 2)
                b.flat_attrs.insert(in.vintrp_attr);
        }
    }
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    // #2418: a static property of the decoded stream, set once and never mutated during emission.
    // Gates the fragment SCC re-arm after mask ops so only shaders that actually consume SCC pay the
    // exact-wave-vote's subgroup-size requirement.
    rs.reads_scc = shader_reads_scc(ins);
    if (system_inputs) {
        // RDNA2 PS system values are packed in field order. ADDR reserves each field's documented
        // width even when ENA is clear, allowing a driver to keep later VGPR numbers stable. Vulkan
        // exposes the four floating-point position terms directly as FragCoord.xyzw.
        static constexpr uint8_t widths[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t vgpr = 0;
        for (uint32_t field = 0; field < 16; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit)) continue;
            if (system_inputs->ena & bit) {
                if (field <= 6 && derived_interpolation.requires_geometry) {
                    for (uint32_t component = 0; component < widths[field]; ++component) {
                        const uint32_t value = b.system_interpolation_component(field, component);
                        if (!value) {
                            log_recompile_diagnostic(
                                diagnostic, "recompile-reject", "terminal",
                                "missing fragment system interpolation field=%u component=%u",
                                field, component);
                            return {};
                        }
                        rs.vreg[(int)(vgpr + component)] = value;
                    }
                } else if (field >= 8 && field <= 11) {
                    rs.vreg[(int)vgpr] = b.fragcoord_component(field - 8);
                }
            }
            vgpr += widths[field];
        }
    }
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // FRAGMENT-only: also linearize alpha-test / clip() kill-mask s_cbranch_scc0/scc1 early-outs (Unity
    // cutout + text draws). Merged into the same set so emit_alu drops the branch and detect_forward_if
    // skips it; the block's EXEC narrow + the export's OpKill do the per-invocation discard (#102). This is
    // NOT added for the vertex/compute shells (their scc branches are real uniform-ifs / NGG culling).
    for (uint32_t pc : mask_test_branches(ins, b.allow_b32_masks)) safe_branches.insert(pc);
    std::array<bool, kFragmentColorOutputs> exported{};
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP MRT0/MRT1 -> matching output
        // An export while EXEC is narrowed (lanes killed by an alpha test / v_cmpx and not restored to
        // all-on) must not write the inactive lanes. Lower it to a real fragment discard: OpKill the lanes
        // whose EXEC bit is false, then export from the survivors under full EXEC. This is exactly the
        // alpha-tested-sprite shape (image_sample -> v_cmp alpha<ref -> s_andn2 saved,saved,vcc -> s_wqm
        // exec,saved -> shade -> export): the surviving lanes are the ones that passed the test. (When EXEC
        // was never narrowed this is a no-op — the common sRGB/tonemap restore-then-export path.)
        if (state.exec_narrowed) {
            b.discard_unless(state.exec);
            state.exec = b.btrue();
            state.exec_narrowed = false;
        }
        // MRTZ (target 8): EN bit 0 exports depth from VSRC0 and bit 2 exports sample coverage from
        // VSRC2. Vulkan represents those as FragDepth and SampleMask[0], respectively. Stencil
        // reference (bit 1) needs a separate extension-backed path and remains fail-visible; COMPR
        // MRTZ and the bit-3 alpha-to-coverage payload are likewise unmodeled.
        if (in.exp_target == 8) {
            if (in.exp_compr || !in.exp_en || (in.exp_en & kUnsupportedMrtz)) return false;
            const bool exports_depth = (in.exp_en & kMrtzDepth) != 0;
            const bool exports_sample_mask = (in.exp_en & kMrtzSampleMask) != 0;
            bool eok = true;
            const uint32_t z = exports_depth
                ? operand_bits(b, state, in, in.src[0], &eok) : 0;
            const uint32_t sample_mask = exports_sample_mask
                ? operand_bits(b, state, in, in.src[2], &eok) : 0;
            if (!eok) return false;
            if (exports_depth) b.export_depth(z);
            if (exports_sample_mask) b.export_sample_mask(sample_mask);
            return true;
        }
        if (in.exp_target < exported.size()) {
            // EN (Table 56) selects which VSRC channels the export sends; hardware does not update
            // disabled components. The executor maps this mask to Vulkan colorWriteMask, so the SPIR-V
            // value in a disabled channel is irrelevant and must not force a read of a stale VGPR.
            if (in.exp_en == 0) return true;
            bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
            if (in.exp_compr) {
                // COMPR: the 4 channels are two f16x2 pairs — src[0] holds (r,g), src[1] holds (b,a).
                // Unpack each half to a float and reassemble the vec4 (the pkrtz'd tonemap/sRGB output).
                const uint32_t p0 = (in.exp_en & 0x3u)
                    ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
                const uint32_t p1 = (in.exp_en & 0xCu)
                    ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? b.unpack_half(p0, 0) : b.uconst(0),
                               (in.exp_en & 0x2u) ? b.unpack_half(p0, 1) : b.uconst(0),
                               (in.exp_en & 0x4u) ? b.unpack_half(p1, 0) : b.uconst(0),
                               (in.exp_en & 0x8u) ? b.unpack_half(p1, 1) : b.uconst(0));
            } else {
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0),
                               (in.exp_en & 0x2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0),
                               (in.exp_en & 0x4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0),
                               (in.exp_en & 0x8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0));
            }
            if (!eok) return false;
            exported[in.exp_target] = true;
        }
        return true;   // NULL carries the EXEC/discard effect above; ignore additional exports for now
    };
    // cmpx is now ALLOWED (allow_exec_update=true): a fragment divergent-if (v_cmpx ... s_mov exec,saved)
    // is handled by EXEC predication like compute, and the export is guarded above. Memory ops need a
    // resource table. Loops (if any) are reconstructed by emit_body.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG") && pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(diagnostic, "recompile-reject", "consequent",
                                     "pcrel target=%u body failed", pcrel_dispatch_target);
        return {};
    }
    // ANY colour slot counts, not just the first two. This tested `exported[0] || exported[1]` while
    // the array was sized 2, which read as "did anything export"; once the shell carries eight
    // outputs it silently became "did MRT0 or MRT1 export", and a shader whose only colour output is
    // MRT2+ was rejected outright -- dropping its draws rather than its attachments. The guard's
    // purpose is unchanged: a fragment program that emits no colour, no NULL, no depth and no sample
    // mask is still fail-visible.
    const bool exported_any_color =
        std::any_of(exported.begin(), exported.end(), [](bool e) { return e; });
    if (!exported_any_color && !has_null_export && !has_depth_export &&
        !has_sample_mask_export) {
        if (pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "emitted no fragment color pcrel-target=%u",
                                     pcrel_dispatch_target);
        else
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "emitted no fragment color");
        return {};
    }
    return b.finish();
}

std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt,
                                         const PixelSystemInputMapping* system_inputs,
                                         uint32_t pcrel_dispatch_target,
                                         const FragmentInterpolationLayout* interpolation,
                                         bool wave32,
                                         RecompileDiagnosticContext diagnostic) {
    return recompile_fragment_impl(code, dwords, rt, system_inputs,
                                   pcrel_dispatch_target, interpolation,
                                   wave32 ? 32u : 64u, diagnostic);
}

std::vector<uint32_t> recompile_fragment_wave32_for_test(
        const uint32_t* code, size_t dwords) {
    return recompile_fragment_impl(code, dwords, nullptr, nullptr,
                                   UINT32_MAX, nullptr, 32,
                                   {RecompileDiagnosticStage::Fragment, 0});
}

uint32_t fragment_spirv_required_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.FragmentSubgroupSize=";
    bool legacy_arithmetic_marker = false;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2 &&
            spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
            legacy_arithmetic_marker = true;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(
                        text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length && (value == 32 || value == 64))
                        return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // Compatibility for cached/captured modules produced before explicit module metadata.
    return legacy_arithmetic_marker ? 64u : 0u;
}

uint32_t compute_spirv_min_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.ComputeSubgroupMin=";
    uint32_t required = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length &&
                        (value == 4 || value == 16 || value == 32 || value == 64))
                        required = std::max(required, static_cast<uint32_t>(value));
                }
            }
        }
        offset += words;
    }
    return required;
}

uint32_t fragment_spirv_required_subgroup_reasons(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return UINT32_MAX;
    constexpr char prefix[] = "Prosper.FragmentSubgroupWhy=";
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return UINT32_MAX;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value =
                        std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length) return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // No marker: the module predates #2147. UINT32_MAX rather than 0, so a caller cannot read
    // 'unknown' as 'nothing required it'.
    return UINT32_MAX;
}

uint32_t fragment_spirv_required_subgroup_features(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    uint32_t features = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2) {
            if (spirv[offset + 1] == Cap_GroupNonUniformVote)
                features |= kFragmentSubgroupVote;
            else if (spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
                features |= kFragmentSubgroupArithmetic;
            else if (spirv[offset + 1] == Cap_GroupNonUniformShuffle)
                features |= kFragmentSubgroupShuffle;
            else if (spirv[offset + 1] == Cap_GroupNonUniformBallot)
                features |= kFragmentSubgroupBallot;
        }
        offset += words;
    }
    return features;
}

bool fragment_spirv_uses_internal_gds(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;
    std::unordered_map<uint32_t, uint32_t> sets, bindings;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return false;
        if (opcode == Op_Decorate && words == 4) {
            if (spirv[offset + 2] == Dec_DescriptorSet)
                sets[spirv[offset + 1]] = spirv[offset + 3];
            else if (spirv[offset + 2] == Dec_Binding)
                bindings[spirv[offset + 1]] = spirv[offset + 3];
        }
        offset += words;
    }
    for (const auto& [variable, set] : sets) {
        auto binding = bindings.find(variable);
        if (set == 1 && binding != bindings.end() && binding->second == 0) return true;
    }
    return false;
}

// Astro Bot's observed NGG wrappers use wave-shared plumbing for vertex allocation/compaction. The
// current Vulkan vertex shell intentionally projects those complete wrappers to one private guest
// lane; applying the projection to arbitrary NGG programs would silently miscompile peer-lane state.
// Keep each exception byte-exact and fail closed for every other wrapper. The hashes are FNV-1a over
// the little-endian instruction bytes through S_ENDPGM, matching the raw hashes in capture
// diagnostics. A proven PC-relative constant-table tail is deliberately excluded from the wrapper
// identity while remaining available to the recompiler.
static bool is_astro_bot_ngg_one_lane_wrapper(const uint32_t* code, size_t dwords) {
    if (!code) return false;
    std::vector<Rdna2Inst> instructions;
    const size_t program_dwords = rdna2_walk(code, dwords, instructions);
    if (program_dwords != 54 && program_dwords != 734 && program_dwords != 749 &&
        program_dwords != 3124 && program_dwords != 3435 && program_dwords != 3455 &&
        program_dwords != 3917)
        return false;
    const uint64_t hash = shader_program_hash(code, program_dwords);
    return (program_dwords == 54 && hash == 0x9e9d8e37bcc70607ull) ||
           (program_dwords == 734 && hash == 0x79eb2b954b07dc8eull) ||
           // The same 734-word culling wrapper is live-linked after its exact 15-word fetch prolog.
           (program_dwords == 749 && hash == 0xb440349937df751eull) ||
           (program_dwords == 3124 && hash == 0x41e6ac616c18d295ull) ||
           (program_dwords == 3435 && hash == 0xfad7a9f486523cfcull) ||
           // The same 3435-word wrapper is live-linked after its exact 20-word fetch prolog.
           // Hashing the complete concatenated program keeps the one-lane projection byte-exact.
           (program_dwords == 3455 && hash == 0x562ce5ad01c4c6e3ull) ||
           (program_dwords == 3917 && hash == 0x7f5f2349e2816f5eull);
}

VertexPrologInfo rdna2_vertex_prolog_info(const uint32_t* code, size_t dwords) {
    VertexPrologInfo result;
    if (!code || !dwords) return result;

    const bool prologlog = getenv("PROSPER_PROLOGLOG") != nullptr;
    uint64_t phash = 0xcbf29ce484222325ull;
    if (prologlog)
        for (size_t i = 0; i < dwords && i < 4096; ++i) phash = (phash ^ code[i]) * 0x100000001b3ull;
    auto prolog_note = [&](const char* what, const Rdna2Inst* at) {
        if (!prologlog) return;
        static std::set<uint64_t> seen; static std::mutex mx;
        std::lock_guard<std::mutex> lk(mx);
        if (!seen.insert(phash).second) return;
        fprintf(stderr, "[prologlog] hash=%016llx dwords=%zu %s pc=%u fmt=%d\n",
                (unsigned long long)phash, dwords, what, at ? at->pc : 0u, at ? (int)at->fmt : -1);
    };
    // Only the prefix before the transfer can belong to a fetch prolog. Track its branch-target
    // bounds while decoding, so neither the prefix nor the unused shader tail needs a vector.
    int64_t min_branch_target = 0;
    int64_t max_branch_target = 0;
    for (size_t pc = 0; pc < dwords;) {
        Rdna2Inst instruction = rdna2_decode_one(code + pc, dwords - pc);
        instruction.pc = static_cast<uint32_t>(pc);
        // A fetch prolog has no architectural output or program termination of its own. Encountering
        // either before the transfer means this is a complete/different shader, not the split ABI.
        if (instruction.is_end || instruction.fmt == Rdna2Format::EXP ||
            instruction.fmt == Rdna2Format::Unknown) {
            prolog_note("BAIL", &instruction);
            return {};
        }
        if (instruction.fmt == Rdna2Format::SOP1 && instruction.opcode == 0x20) {
            // GFX9+ merged-stage fetch prologs receive the continuation PC in reserved s[6:7].
            // Keep this exact pair: an arbitrary indirect jump must not become host fallthrough.
            if (instruction.n_src != 1 || instruction.src[0].kind != OperandKind::SGPR ||
                instruction.src[0].value != 6 || instruction.len_dwords != 1)
                return {};
            prolog_note("TRANSFER", &instruction);
            // Every direct branch must remain inside the retained prefix or land exactly on the
            // transfer (main pc0 after linking), never in discarded padding/data. The linked body
            // recompiler performs the remaining structured-CFG validation.
            if (instruction.pc == 0 || min_branch_target < 0 ||
                max_branch_target > instruction.pc)
                return {};
            result.valid = true;
            result.setpc_pc = instruction.pc;
            result.prefix_dwords = instruction.pc;
            return result;
        }
        if (instruction.fmt == Rdna2Format::SOPP &&
            (instruction.opcode == 0x02 ||
             (instruction.opcode >= 0x04 && instruction.opcode <= 0x09))) {
            const int64_t target = static_cast<int64_t>(instruction.pc) + instruction.len_dwords +
                                   static_cast<int64_t>(instruction.simm16);
            min_branch_target = std::min(min_branch_target, target);
            max_branch_target = std::max(max_branch_target, target);
        }
        if (instruction.len_dwords == 0) break;
        pc += instruction.len_dwords;
    }
    return {};
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.
struct NggPassthroughLayout {
    bool valid = false;
    uint32_t producer_base_vgpr = 0;
    uint32_t producer_base_byte = 0;
    uint32_t record_stride_bytes = 0;
    std::array<int32_t, 4> position = {-1, -1, -1, -1};
    std::array<std::array<int32_t, 4>, 32> params{};
    uint32_t param_mask = 0;

    NggPassthroughLayout() {
        for (auto& param : params) param.fill(-1);
    }
};

struct NggLdsSource {
    bool valid = false;
    uint32_t byte_offset = 0;
    uint32_t stride_bytes = 0;
    uint32_t index_vgpr = 0;
};

// Resolve the terminal wrapper's canonical `stride * exporter + constant` LDS address.  Requiring
// the nearest writer to be this exact u24 MAD shape keeps the optimization fail-closed when a user
// GS or a different compiler layout performs real output computation.
NggLdsSource ngg_terminal_lds_source(const std::vector<Rdna2Inst>& ins, size_t load_index,
                                     uint32_t output_vgpr) {
    const Rdna2Inst& load = ins[load_index];
    uint32_t address_vgpr = 0, component_byte = 0;
    if (load.fmt != Rdna2Format::DS || load.ds_gds) return {};
    if (load.opcode == 0x36u) {                         // ds_read_b32
        if (output_vgpr != static_cast<uint32_t>(load.dst.value)) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal;
    } else if (load.opcode == 0x37u) {                  // ds_read2_b32
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr > first + 1u) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        const uint32_t component = output_vgpr - first;
        component_byte = ((load.literal >> (component * 8u)) & 0xffu) * 4u;
    } else if (load.opcode == 0x76u || load.opcode == 0xfeu || load.opcode == 0xffu) {
        const uint32_t count = load.opcode == 0x76u ? 2u : load.opcode == 0xfeu ? 3u : 4u;
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr >= first + count) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal + (output_vgpr - first) * 4u;
    } else {
        return {};
    }

    for (size_t j = load_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.dst.kind != OperandKind::VGPR ||
            static_cast<uint32_t>(writer.dst.value) != address_vgpr)
            continue;
        // The compacted record address has either of the two canonical compiler forms below:
        //
        //   v_mad_u32_u24 addr, stride, exporter, constant
        //   v_mul_u32_u24 addr, stride, exporter; ds_read ... offset:constant
        //
        // The latter avoids a MAD when the entire constant fits in the DS instruction's immediate.
        // Both prove the same `stride * exporter + constant` identity; accepting only these exact
        // integer-u24 forms keeps arbitrary wrapper address arithmetic fail-closed.
        const bool mad = writer.fmt == Rdna2Format::VOP3 && writer.opcode == 0x143u &&
                         !writer.has_modifier && writer.has_literal;
        const bool mul = writer.fmt == Rdna2Format::VOP2 && writer.opcode == 0x0bu &&
                         !writer.has_modifier && !writer.has_literal;
        if (!mad && !mul) return {};
        // The first two operands may be swapped.
        int stride_src = -1, index_src = -1;
        for (int k = 0; k < 2; ++k) {
            if (writer.src[k].kind == OperandKind::InlineInt && writer.src[k].value > 0)
                stride_src = k;
            else if (writer.src[k].kind == OperandKind::VGPR)
                index_src = k;
        }
        if (stride_src < 0 || index_src < 0 ||
            (mad && writer.src[2].kind != OperandKind::Literal))
            return {};
        const uint32_t stride = static_cast<uint32_t>(writer.src[stride_src].value);
        if ((stride & 3u) || stride < 16u || stride > 4096u ||
            (mad && writer.literal > UINT32_MAX - component_byte))
            return {};
        return {true, (mad ? writer.literal : 0u) + component_byte, stride,
                static_cast<uint32_t>(writer.src[index_src].value)};
    }
    return {};
}

NggLdsSource ngg_find_terminal_output(const std::vector<Rdna2Inst>& ins, size_t export_index,
                                      uint32_t output_vgpr) {
    // The direct output loads sit in the export block.  Stop at the nearest writer; skipping a
    // transform or phi would turn a user GS into a false passthrough.
    for (size_t j = export_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.fmt == Rdna2Format::DS) {
            const NggLdsSource source = ngg_terminal_lds_source(ins, j, output_vgpr);
            if (source.valid) return source;
            const uint32_t first = static_cast<uint32_t>(writer.dst.value);
            const uint32_t count = writer.opcode == 0x37u || writer.opcode == 0x76u ? 2u
                                 : writer.opcode == 0xfeu ? 3u
                                 : writer.opcode == 0xffu ? 4u : 1u;
            if (output_vgpr >= first && output_vgpr < first + count) return {};
        }
        if (writer.dst.kind == OperandKind::VGPR &&
            static_cast<uint32_t>(writer.dst.value) == output_vgpr)
            return {};
    }
    return {};
}

NggPassthroughLayout analyze_ngg_passthrough(const uint32_t* prolog, size_t prefix_dwords,
                                             const uint32_t* main, size_t main_dwords) {
    NggPassthroughLayout out;
    auto reject = [&](const char* reason) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-ngg-passthrough-reject] %s\n", reason);
        return NggPassthroughLayout{};
    };
    std::vector<Rdna2Inst> producer, wrapper;
    rdna2_walk(prolog, prefix_dwords, producer);
    rdna2_walk(main, main_dwords, wrapper);
    if (producer.empty() || wrapper.empty()) return reject("empty producer/wrapper");

    // Prove that every LDS operation in the producer is a non-atomic store into one record base.
    // The private-LDS execution is valid only before the cross-lane wrapper starts reading it.
    bool saw_store = false;
    uint32_t base_vgpr = UINT32_MAX;
    std::set<uint32_t> stored_bytes;
    for (const Rdna2Inst& in : producer) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) return reject("producer terminates or exports");
        if (in.fmt != Rdna2Format::DS) continue;
        if (in.ds_gds || in.src[0].kind != OperandKind::VGPR)
            return reject("producer uses non-LDS DS address");
        const uint32_t base = static_cast<uint32_t>(in.src[0].value);
        if (base_vgpr == UINT32_MAX) base_vgpr = base;
        if (base != base_vgpr) return reject("producer has multiple LDS record bases");
        auto store_byte = [&](uint32_t byte) {
            if ((byte & 3u) == 0u) stored_bytes.insert(byte);
        };
        if (in.opcode == 0x0du) {
            store_byte(in.literal);
        } else if (in.opcode == 0x0eu) {
            store_byte((in.literal & 0xffu) * 4u);
            store_byte(((in.literal >> 8) & 0xffu) * 4u);
        } else if (in.opcode == 0x4du || in.opcode == 0xdeu || in.opcode == 0xdfu) {
            const uint32_t count = in.opcode == 0x4du ? 2u : in.opcode == 0xdeu ? 3u : 4u;
            for (uint32_t k = 0; k < count; ++k) store_byte(in.literal + k * 4u);
        } else {
            return reject("producer has a non-store/cross-lane DS operation");
        }
        saw_store = true;
    }
    if (!saw_store || stored_bytes.empty()) return reject("producer writes no aligned LDS record");

    bool saw_alloc = false, saw_primitive_export = false, saw_position = false;
    uint32_t common_stride = 0, common_index = UINT32_MAX;
    std::vector<uint32_t> absolute_offsets;
    auto accept_source = [&](const NggLdsSource& source, int32_t& destination) -> bool {
        if (!source.valid || (common_stride && source.stride_bytes != common_stride) ||
            (common_index != UINT32_MAX && source.index_vgpr != common_index))
            return false;
        common_stride = source.stride_bytes;
        common_index = source.index_vgpr;
        destination = static_cast<int32_t>(source.byte_offset);
        absolute_offsets.push_back(source.byte_offset);
        return true;
    };
    for (size_t i = 0; i < wrapper.size(); ++i) {
        const Rdna2Inst& in = wrapper[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10u) saw_alloc = true;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target == 20u) { saw_primitive_export = true; continue; }
        if (in.exp_target != 12u && in.exp_target < 32u) continue;
        if (in.exp_compr) return reject("terminal output uses a compressed export");
        if (in.exp_target == 12u) {
            if (saw_position || in.exp_en != 0xfu)
                return reject("POS0 is duplicated or incomplete");
            saw_position = true;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.position[component]))
                    return reject("POS0 does not directly load the terminal LDS record");
            }
        } else {
            const uint32_t param = in.exp_target - 32u;
            if (param >= out.params.size() || (out.param_mask & (1u << param)))
                return reject("PARAM export is out of range or duplicated");
            out.param_mask |= 1u << param;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!(in.exp_en & (1u << component))) continue;
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.params[param][component]))
                    return reject("PARAM does not directly load the terminal LDS record");
            }
        }
    }
    if (!saw_alloc || !saw_primitive_export || !saw_position || absolute_offsets.empty() ||
        !common_stride || base_vgpr == UINT32_MAX)
        return reject("wrapper lacks the no-GS allocation/export shape");

    const uint32_t wrapper_base = *std::min_element(absolute_offsets.begin(), absolute_offsets.end());
    const uint32_t producer_base = *stored_bytes.begin();
    auto normalize = [&](int32_t& byte) -> bool {
        if (byte < 0) return true;
        const uint32_t absolute = static_cast<uint32_t>(byte);
        if (absolute < wrapper_base) return false;
        const uint32_t relative = absolute - wrapper_base;
        if ((relative & 3u) || relative >= common_stride ||
            producer_base > UINT32_MAX - relative ||
            !stored_bytes.count(producer_base + relative))
            return false;
        byte = static_cast<int32_t>((producer_base + relative) / 4u);
        return true;
    };
    for (int32_t& component : out.position)
        if (!normalize(component)) return reject("POS0 offset does not match the producer record");
    for (auto& param : out.params)
        for (int32_t& component : param)
            if (!normalize(component)) return reject("PARAM offset does not match the producer record");

    out.valid = true;
    out.producer_base_vgpr = base_vgpr;
    out.producer_base_byte = producer_base;
    out.record_stride_bytes = common_stride;
    return out;
}

} // namespace

static std::vector<uint32_t> recompile_vertex_impl(const uint32_t* code, size_t dwords,
                                                   const ShaderResourceTable* rt,
                                                   const PixelInputMapping* pixel_inputs,
                                                   bool capture_position,
                                                   uint32_t virtual_lds_dwords,
                                                   const NggPassthroughLayout* passthrough,
                                                   bool allow_test_ngg_output_gate,
                                                   bool allow_test_ngg_one_lane,
                                                   RecompileDiagnosticContext diagnostic) {
    const uint32_t passthrough_mask =
        pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    SpirvCompute b;
    // The vertex stage recorded its rejects at address 0, and record_terminal_reject_reason()
    // early-returns on a zero address -- so `last_terminal_reject_reason()` came back empty for
    // EVERY vertex program and the unconditional skip line could only print `reason=unrecorded`.
    // #3130 fixed exactly this for the fragment stage; the vertex half was left behind.
    b.diagnostic = diagnostic;
    b.capture_position = capture_position;   // geometry-probe: mark gl_Position for xfb capture (gated)
    b.vertex_lds_dwords = std::min(virtual_lds_dwords, 16384u);
    b.vertices_per_instance = rt ? rt->vertices_per_instance : 0u;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): redirect the position export to the intermediate VGPR
    // produced at that PC. Applies to the vertex stage only; captured via the geometry probe.
    if (const char* tap = getenv("PROSPER_SHADER_TAP")) b.tap_pc = static_cast<uint32_t>(strtoul(tap, nullptr, 0));
    b.begin_vertex(rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // NGG vertex shaders (the exact GS_ALLOC_REQ message present) carry the vertex index in v5, not v0, and wrap
    // the body in wave-packing plumbing (s_sendmsg / exp prim / s_lshr_b64 exec) that lowers to no-ops in
    // our per-invocation model. Detect NGG and bind the index to v5 as well.
    bool ngg = passthrough && passthrough->valid;
    for (const auto& in : ins) { if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10 &&
            in.words[0] == 0xBF900009u) { ngg = true; break; } }
    const bool exact_ngg_projection = ngg && is_astro_bot_ngg_one_lane_wrapper(code, dwords);
    // The generic split-stage path uses private LDS only after the producer/wrapper analyzer has
    // proven that all visible outputs come from one per-vertex record. Wave-sensitive instruction
    // approximations remain restricted to the byte-exact projection above.
    b.ngg_private_lds = exact_ngg_projection || (passthrough && passthrough->valid) ||
                        (ngg && allow_test_ngg_output_gate);
    // The legacy byte-exact NGG projection and its explicit unit-test hook predate the linked-stage
    // ABI's exact LDS allocation. Preserve their proven 16 KiB private scratch contract when no
    // allocation was supplied; generic linked producers still require their real plumbed size.
    if (!b.vertex_lds_dwords && (exact_ngg_projection || allow_test_ngg_output_gate))
        b.vertex_lds_dwords = 4096;
    // Every wave/peer approximation is an exception for the one captured Astro wrapper, not a
    // property of the GS_ALLOC_REQ opcode. Other NGG programs retain only the ordinary merged-stage
    // ABI setup below and fail closed if they reach a lane-sensitive operation.
    b.ngg_one_lane = exact_ngg_projection || (ngg && allow_test_ngg_one_lane);
    // A LO+HI all-ones pair is the compiler's explicit wave64 lane-index construction. Infer the
    // width from that machine-code proof instead of assuming every NGG program is wave64. A low-only
    // producer may be wave32 or may use only half of a wave64 mask, so it remains fail-closed until
    // the graphics wave-size contract is plumbed independently.
    bool logical_mbcnt_lo = false, logical_mbcnt_hi = false, logical_mbcnt_invalid = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::VOP3 || (in.opcode != 0x365 && in.opcode != 0x366))
            continue;
        const bool all_ones = in.src[0].kind == OperandKind::InlineInt &&
                              in.src[0].value == -1;
        // Which instruction disqualified the model, not just that something did. The all-ones pair
        // is lowerable on its own, so on a mixed program the reject surfaces at IT and names a pc
        // that is not the cause -- Stray's 0x300f190000 rejects at pc=4 while the general-mask
        // forms that disqualified it are at pc 277-286 (#3135).
        if (!all_ones && b.vertex_general_mask_mbcnt_pc == UINT32_MAX)
            b.vertex_general_mask_mbcnt_pc = in.pc;
        logical_mbcnt_invalid |= !all_ones;
        logical_mbcnt_lo |= all_ones && in.opcode == 0x365;
        logical_mbcnt_hi |= all_ones && in.opcode == 0x366;
    }
    b.ngg_logical_lane = passthrough && passthrough->valid && !logical_mbcnt_invalid &&
                         logical_mbcnt_lo && logical_mbcnt_hi;
    if (b.ngg_logical_lane) b.wave_size = 64;
    b.allow_b32_masks = b.ngg_one_lane;
    uint32_t ngg_output_gate_begin = UINT32_MAX;
    uint32_t ngg_output_gate_end = 0;
    if (ngg) {
        // A terminal compacted-output suffix may reconstruct its values from a shader-embedded
        // constant table. Detect those loads with the same bounded PC-relative proof used by the
        // emitter; an arbitrary external load or any buffer write must not broaden this gate.
        PcrelTables output_tables;
        if (b.ngg_private_lds)
            output_tables = detect_pcrel_tables(ins, code, dwords);
        auto scalar_output_setup = [](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::SOP1 &&
                candidate.fmt != Rdna2Format::SOP2 &&
                candidate.fmt != Rdna2Format::SOPK)
                return false;
            bool wrote_data = false;
            bool safe = !rdna2_instruction_may_change_exec(candidate);
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                wrote_data = true;
                safe &= base >= 0 && base + static_cast<int>(width) <= 106;
                safe &= !scalar_write_is_b64_mask(candidate, base);
            });
            return wrote_data && safe;
        };
        // MUBUF-only is DELIBERATE, not an oversight now that `output_tables.mtbuf` exists three
        // lines up (#2859). This gate is a narrowing allow-list for the byte-exact Astro Bot
        // compacted-output wrapper; admitting a typed consumer would widen an NGG output gate with no
        // title evidence behind it. A typed embedded table still FOLDS -- it just does not open this
        // gate, which is the conservative direction.
        auto embedded_output_load = [&](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::MUBUF || candidate.mubuf_lds)
                return false;
            const bool read_only = candidate.opcode <= 0x03u ||
                (candidate.opcode >= 0x0cu && candidate.opcode <= 0x0fu);
            return read_only && output_tables.mubuf.contains(candidate.pc);
        };
        uint32_t end_pc = UINT32_MAX;
        for (const auto& in : ins)
            if (in.is_end) { end_pc = in.pc; break; }
        // An NGG primitive shader finishes by compacting surviving vertices, CMPX-testing whether
        // this lane owns one of those vertices, then exporting POS/PARAM values before S_ENDPGM.
        // Vulkan's vertex shell already represents one surviving guest vertex per invocation, but
        // retaining the condition is still useful when the compacted count is zero. Permit exports
        // under narrowed EXEC only for this mechanically bounded terminal output gate; ordinary
        // vertex CMPX/export programs remain rejected below.
        for (size_t i = 1; i < ins.size(); ++i) {
            const Rdna2Inst& branch = ins[i];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                branch.simm16 <= 0 || branch_target(branch) != end_pc)
                continue;
            size_t previous = i;
            do { --previous; } while (previous > 0 && sopp_is_noop(ins[previous]));
            if (ins[previous].fmt != Rdna2Format::VOPC ||
                !vopc_is_cmpx(ins[previous].opcode))
                continue;
            // Production accepts data-dependent vertex suppression only for the byte-exact Astro
            // wrapper. The explicit test hook below exercises active/inactive export selection with
            // a tiny shader without turning that shader shape into a runtime allow-list exception.
            if (!b.ngg_private_lds && !allow_test_ngg_output_gate)
                continue;
            bool has_position = false;
            bool output_only = true;
            std::vector<uint32_t> trailing_vcc_branches;
            for (size_t j = i + 1; j < ins.size() && ins[j].pc < end_pc; ++j) {
                const Rdna2Inst& candidate = ins[j];
                if (candidate.fmt == Rdna2Format::EXP) {
                    has_position |= candidate.exp_target == 12;
                    continue;
                }
                // Astro's compacted-output suffix reconstructs the surviving vertex from private
                // LDS or a bounded shader-embedded table immediately before exporting it. Vector,
                // DS, and table-load destination writes are EXEC-predicated by emit_alu; scalar ALU
                // may only build ordinary data/descriptor registers. Admit them only for the
                // byte-exact wrapper (or the explicit test hook); arbitrary NGG shaders never reach
                // this exception, and buffer stores/external reads remain rejected.
                const bool output_rebuild = b.ngg_private_lds || allow_test_ngg_output_gate;
                if (output_rebuild &&
                    (candidate.fmt == Rdna2Format::VOP1 ||
                     candidate.fmt == Rdna2Format::VOP2 ||
                     candidate.fmt == Rdna2Format::VOP3 ||
                     candidate.fmt == Rdna2Format::VOP3P ||
                     (candidate.fmt == Rdna2Format::VOPC &&
                      !vopc_is_cmpx(candidate.opcode)) ||
                     candidate.fmt == Rdna2Format::DS ||
                     scalar_output_setup(candidate) ||
                     embedded_output_load(candidate)))
                    continue;
                if (candidate.fmt == Rdna2Format::SOPC) continue;
                if (sopp_is_noop(candidate)) continue;
                if (candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x04 || candidate.opcode == 0x05) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc)
                    continue;
                // The 7f5f wrapper exports POS, compares a per-vertex flag into VCC, then conditionally
                // skips only its trailing PARAM exports. Those exports cannot affect position/topology;
                // linearizing the branch merely supplies otherwise-undefined varyings for that path.
                if (has_position && candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x06 || candidate.opcode == 0x07) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc) {
                    trailing_vcc_branches.push_back(candidate.pc);
                    continue;
                }
                output_only = false;
                break;
            }
            if (has_position && output_only) {
                ngg_output_gate_begin = branch.pc;
                ngg_output_gate_end = end_pc;
                safe_branches.insert(trailing_vcc_branches.begin(), trailing_vcc_branches.end());
                break;
            }
        }
    }
    uint32_t vidx = b.load_vertex_index();
    uint32_t iidx = b.load_instance_index();
    rs.vreg[0] = vidx;                       // Legacy VS ABI: v0 = vertex index
    rs.vreg[3] = iidx;                       // Legacy VS ABI: v3 = instance index
    if (ngg) {
        // Locate the NGG prologue's LDS-to-ES vertex-index handoff without hard-coding a register or
        // program counter: the first MUBUF vaddr must be most recently defined by a scalar DS read.
        // The one-lane backend substitutes BuiltIn VertexIndex at that exact read (see DS lowering).
        for (size_t use = 0; use < ins.size(); ++use) {
            if (ins[use].is_end) break;
            if ((ins[use].fmt != Rdna2Format::MUBUF && ins[use].fmt != Rdna2Format::MTBUF) ||
                ins[use].src[0].kind != OperandKind::VGPR)
                continue;
            const int index_reg = ins[use].src[0].value;
            for (size_t def = use; def-- > 0;) {
                const Rdna2Inst& candidate = ins[def];
                if (candidate.dst.kind != OperandKind::VGPR || candidate.dst.value != index_reg)
                    continue;
                if (candidate.fmt == Rdna2Format::DS && candidate.opcode == 0x36) {
                    b.ngg_vertex_index_read_pc = candidate.pc;
                    b.ngg_vertex_index_value = vidx;
                }
                break;
            }
            break;
        }
        // GFX10's merged GS/ES ABI enters the ES prolog with vertex/instance indices in v5/v8 and
        // merged-wave info in s3: per-wave ES/GS counts [7:0]/[15:8], wave-in-TG [27:24], and
        // TG wave count [31:28]. The host draw has already omitted padding invocations, so expose a
        // full logical wave while deriving the architectural wave ID from the flattened invocation.
        rs.vreg[5] = vidx;
        rs.vreg[8] = iidx;
        if (exact_ngg_projection) {
            rs.sreg[3] = b.uconst(1);

        // The merged NGG wrapper guards its whole counted ES loop with EXECZ. In the single-lane
        // model above that one ES lane is active by construction, so the wave-empty shortcut cannot
        // be taken. Let the ordinary counted-loop lowering consume the loop instead of rejecting the
        // redundant outer guard merely because the original hardware mask was vector-shaped.
        const CountedLoop loop = detect_counted_loop(ins);
        if (loop.found) {
            // NGG culling unrolls several EXEC-predicated LDS blocks inside its counted loop. A
            // larger block may contain a smaller already-safe EXECZ block plus CMPX comparisons,
            // and ends by restoring a VCC-saved mask to EXEC. The generic safe-execz pass deliberately
            // rejects CMPX writes. Here they are exact: the branch immediately follows a CMPX that
            // narrowed EXEC, all effects in the skipped block are EXEC-predicated (further CMPX can
            // only narrow it again), and the common target performs the same EXEC=VCC restore on both
            // paths. Scan inside-out so nested blocks are proven before their parents.
            for (size_t branch_index = ins.size(); branch_index-- > 0;) {
                const Rdna2Inst& branch = ins[branch_index];
                if (branch.pc < loop.header_pc || branch.pc > loop.backedge_pc ||
                    branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                    branch.simm16 <= 0)
                    continue;
                size_t previous = branch_index;
                while (previous > 0) {
                    --previous;
                    if (!sopp_is_noop(ins[previous])) break;
                }
                if (previous >= branch_index || ins[previous].fmt != Rdna2Format::VOPC ||
                    !vopc_is_cmpx(ins[previous].opcode))
                    continue;
                const uint32_t target_pc = branch_target(branch);
                size_t target_index = ins.size();
                for (size_t i = 0; i < ins.size(); ++i)
                    if (ins[i].pc == target_pc) { target_index = i; break; }
                bool restores_saved_exec = false;
                for (size_t i = target_index; i < ins.size() && i < target_index + 3; ++i) {
                    const Rdna2Inst& candidate = ins[i];
                    if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == 0x04 &&
                        candidate.dst.value >= 126 &&
                        (candidate.src[0].value == 106 || candidate.src[0].value == 107)) {
                        restores_saved_exec = true;
                        break;
                    }
                    bool clobbers_saved_mask = rdna2_instruction_may_change_exec(candidate) ||
                        candidate.fmt == Rdna2Format::VOPC;
                    for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                        clobbers_saved_mask |= base < 108 && 106 < base + static_cast<int>(width);
                    });
                    if (clobbers_saved_mask ||
                        (candidate.fmt == Rdna2Format::SOPP && candidate.opcode >= 0x02 &&
                         candidate.opcode <= 0x09 && candidate.opcode != 0x03))
                        break;
                }
                if (!restores_saved_exec)
                    continue;
                bool safe_block = true;
                for (const auto& candidate : ins) {
                    if (candidate.pc <= branch.pc || candidate.pc >= target_pc) continue;
                    const bool predicated_or_masked =
                        candidate.fmt == Rdna2Format::VOP1 ||
                        candidate.fmt == Rdna2Format::VOP2 ||
                        candidate.fmt == Rdna2Format::VOP3 ||
                        candidate.fmt == Rdna2Format::VOP3P ||
                        (candidate.fmt == Rdna2Format::VOPC &&
                         vopc_is_cmpx(candidate.opcode)) ||
                        candidate.fmt == Rdna2Format::MIMG ||
                        candidate.fmt == Rdna2Format::MUBUF ||
                        candidate.fmt == Rdna2Format::MTBUF ||
                        candidate.fmt == Rdna2Format::DS ||
                        candidate.fmt == Rdna2Format::FLAT;
                    const bool nested_safe = candidate.fmt == Rdna2Format::SOPP &&
                        candidate.opcode == 0x08 && safe_branches.count(candidate.pc);
                    if (!predicated_or_masked && !nested_safe && !sopp_is_noop(candidate)) {
                        safe_block = false;
                        break;
                    }
                }
                if (safe_block) safe_branches.insert(branch.pc);
            }
            for (const auto& in : ins) {
                if (in.pc >= loop.header_pc) break;
                if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                    branch_target(in) == loop.exit_pc)
                    safe_branches.insert(in.pc);
            }
        }
        } else {
            const uint32_t wave = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, b.vertex_invocation_id(), b.uconst(6)),
                b.uconst(0xFu));
            rs.sreg[3] = b.ibin(Op_BitwiseOr, b.uconst(0x40004040u),
                                b.ibin(Op_ShiftLeftLogical, wave, b.uconst(24)));
        }
    }
    bool exported = false;
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP POS0..3 -> gl_Position; PARAM -> varyings
        bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
        // COMPR pos/param exports carry two packed f16x2 pairs, not four f32 fields — reading the
        // VSRCs as full floats would pass packed-half bit patterns as x/y and stale registers as
        // z/w. Never observed in a vertex stage (compilers export positions/params at 32 bits);
        // reject fail-visibly until a live title exercises one.
        if (in.exp_compr) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex compressed export pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        // v_cmpx is now allowed in the vertex shell (allow_exec_update=true below): a divergent block
        // (v_cmpx … s_mov_b64 exec, saved — DOLL's per-vertex lighting/fog attenuation) predicates its
        // VGPR writes like compute. A vertex MUST still export from full EXEC — the compiled shape
        // always restores EXEC before its pos/param exports; if one ever arrives narrowed, reject
        // (fail-visibly) rather than export possibly-inactive-lane values.
        const bool terminal_ngg_output = ngg_output_gate_begin != UINT32_MAX &&
            in.pc > ngg_output_gate_begin && in.pc < ngg_output_gate_end;
        if (state.exec_narrowed && !terminal_ngg_output && (in.exp_target >= 32 ||
            (in.exp_target >= 12 && in.exp_target <= 16))) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex export under narrowed exec pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        if (in.exp_target == 12) {
            // POS0 is the mandatory x/y/z/w position vector. POS1..POS4 carry ancillary position
            // data (clip/cull distances, point size, viewport/layer selection according to the
            // programmed position format) and must never be mistaken for gl_Position merely because
            // an NGG shader emits one before POS0. Until those built-ins are modeled, retain the
            // existing deliberate behavior of ignoring them.
            // A position export must supply all four components (EN=0xF); a partial POS0 is not
            // meaningfully completable in the current model, so reject rather than invent components.
            if (in.exp_en != 0xFu) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "partial vertex position export pc=%u en=0x%x",
                                         in.pc, in.exp_en);
                return false;
            }
            uint32_t x = operand_bits(b, state, in, in.src[0], &eok);
            uint32_t y = operand_bits(b, state, in, in.src[1], &eok);
            uint32_t z = operand_bits(b, state, in, in.src[2], &eok);
            uint32_t w = operand_bits(b, state, in, in.src[3], &eok);
            if (terminal_ngg_output && state.exec_narrowed) {
                // The exact compacted-output wrapper emits a contiguous prefix of complete
                // primitives. Map every inactive suffix invocation to one identical clip point;
                // primitives assembled solely from that suffix are therefore degenerate instead of
                // accidentally reusing the last active vertex. Keep the real values on the true path.
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero);
                y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero);
                w = b.sel(state.exec, w, b.uconst(fbits(1.0f)));
            }
            b.export_position(x, y, z, w);
            exported = true;
        } else if (in.exp_target >= 32) {                    // PARAM0.. -> remapped PS input varying
            const uint32_t source = in.exp_target - 32;
            // EN gates which channels the export sends (vec2/vec3 varyings use EN=0x3/0x7):
            // hardware leaves disabled channels unwritten (undefined for the PS). Substitute a
            // deterministic 0.0 for them instead of exporting stale VGPR data.
            uint32_t x = (in.exp_en & 1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
            uint32_t y = (in.exp_en & 2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
            uint32_t z = (in.exp_en & 4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0);
            uint32_t w = (in.exp_en & 8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0);
            if (terminal_ngg_output && state.exec_narrowed) {
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero); y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero); w = b.sel(state.exec, w, zero);
            }
            const uint32_t gx = x, gy = y, gz = z, gw = w;
            // #2945: skip a slot the fragment program never reads. See PixelInputMapping::consumes
            // -- SPI_PS_INPUT_CNTL is sticky, so `valid_mask` alone fans one export out to 32
            // locations and overruns maxVertexOutputComponents.
            if (!pixel_inputs || source >= 32 || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))
                    b.export_param(source, gx, gy, gz, gw);   // absent control retains identity wiring
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;
                    const uint32_t raw_offset = pixel_inputs->controls[ps_input] & 0x3Fu;
                    const uint32_t offset = (passthrough_mask & (1u << ps_input))
                        ? (raw_offset & 0x1fu) : raw_offset;
                    if (offset == source) b.export_param(ps_input, gx, gy, gz, gw);
                }
            }
        }
        if (!eok && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-export-reject] pc=%u target=%u unresolved export operand\n",
                         in.pc, in.exp_target);
        return eok;
    };
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] body or export lowering failed\n");
        return {};
    }
    if (passthrough && passthrough->valid) {
        const auto base_it = rs.vreg.find(static_cast<int>(passthrough->producer_base_vgpr));
        if (base_it == rs.vreg.end() || !b.vertex_lds_dwords) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[vertex-ngg-passthrough-reject] missing producer LDS base/allocation\n");
            return {};
        }
        b.declare_lds();
        const uint32_t base_dword = b.ibin(
            Op_ShiftRightLogical, base_it->second, b.uconst(2u));
        auto record_load = [&](int32_t dword) -> uint32_t {
            return b.lds_load(dword == 0
                ? base_dword
                : b.ibin(Op_IAdd, base_dword, b.uconst(static_cast<uint32_t>(dword))));
        };
        b.export_position(record_load(passthrough->position[0]),
                          record_load(passthrough->position[1]),
                          record_load(passthrough->position[2]),
                          record_load(passthrough->position[3]));
        exported = true;

        for (uint32_t source = 0; source < passthrough->params.size(); ++source) {
            if (!(passthrough->param_mask & (1u << source))) continue;
            std::array<uint32_t, 4> value{};
            for (uint32_t component = 0; component < 4; ++component) {
                const int32_t dword = passthrough->params[source][component];
                value[component] = dword >= 0 ? record_load(dword) : b.uconst(0u);
            }
            if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))   // #2945
                    b.export_param(source, value[0], value[1], value[2], value[3]);
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;    // #2945
                    if ((pixel_inputs->controls[ps_input] & 0x3fu) == source)
                        b.export_param(ps_input, value[0], value[1], value[2], value[3]);
                }
            }
        }
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-ngg-passthrough] base=v%u stride=%u params=%08x\n",
                         passthrough->producer_base_vgpr,
                         passthrough->record_stride_bytes, passthrough->param_mask);
    }
    if (!exported) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] shader emitted no POS0\n");
        return {};
    }
    // OFFSET=0x20 asks the interpolator to synthesize a constant instead of consuming a PARAM
    // export. GFX10 DEFAULT_VAL encodes 0000, 0001, 1110, and 1111. Materialize those outputs in
    // the Vulkan vertex stage, whose fixed-function interface has no equivalent default source.
    if (pixel_inputs) {
        for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
            if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
            if (!pixel_inputs->consumes(ps_input)) continue;            // #2945
            const uint32_t control = pixel_inputs->controls[ps_input];
            if ((passthrough_mask & (1u << ps_input)) ||
                (control & 0x3Fu) != 0x20u) continue;
            const uint32_t one = b.uconst(0x3F800000u), zero = b.uconst(0u);
            const uint32_t default_val = (control >> 8) & 0x3u;
            const bool xyz_one = (default_val & 0x2u) != 0;
            const bool w_one = (default_val & 0x1u) != 0;
            b.export_param(ps_input, xyz_one ? one : zero, xyz_one ? one : zero,
                           xyz_one ? one : zero, w_one ? one : zero);
        }
    }
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt,
                                       const PixelInputMapping* pixel_inputs,
                                       bool capture_position,
                                       uint32_t virtual_lds_dwords,
                                       RecompileDiagnosticContext diagnostic) {
    return recompile_vertex_impl(code, dwords, rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, true, false,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, false, true,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt,
                                             const PixelInputMapping* pixel_inputs,
                                             bool capture_position,
                                             uint32_t virtual_lds_dwords,
                                             RecompileDiagnosticContext diagnostic) {
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog, prolog_dwords);
    if (!info.valid || !main || !main_dwords) return {};

    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span || info.prefix_dwords > SIZE_MAX - main_span) return {};
    const NggPassthroughLayout passthrough =
        analyze_ngg_passthrough(prolog, info.prefix_dwords, main, main_span);
    if (passthrough.valid) {
        return recompile_vertex_impl(prolog, info.prefix_dwords, rt, pixel_inputs,
                                     capture_position, virtual_lds_dwords, &passthrough, false,
                                     false, diagnostic);
    }
    std::vector<uint32_t> linked;
    linked.reserve(info.prefix_dwords + main_span);
    linked.insert(linked.end(), prolog, prolog + info.prefix_dwords);
    linked.insert(linked.end(), main, main + main_span);
    return recompile_vertex_impl(linked.data(), linked.size(), rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

} // namespace prosper::gpu
