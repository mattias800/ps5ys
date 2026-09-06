// test_vertex_output_budget — a recompiled VERTEX shader must not declare an output interface the
// device cannot express, and the bound that keeps it inside the limit is what the FRAGMENT program
// actually reads (#2945).
//
// The defect this pins:
//
//   SPI_PS_INPUT_CNTL_0..31 are CONTEXT registers, and context registers are STICKY. RenderState
//   marks a slot valid when the register is merely PRESENT in the context map, so a pixel shader
//   with ONE interpolant inherits 31 stale slots from whatever ran before it and reports
//   `valid_mask == 0xffffffff`. The vertex emitter fans a single `EXP PARAM0` out to every valid
//   slot, so that stale mask turns one export into 32 output varyings = 128 components; with
//   gl_Position's 4 that is 132 components. The layer rejects it. Its verbatim message, and the
//   caveat about reading the arithmetic beside it, are recorded once in `docs/GRAPHICS.md`; this
//   test pins the LOCATION COUNT, which is the thing the emitter controls.
//
// The mask used here — every slot valid, every control selecting PARAM0 — is exactly the shape the
// BALAN capture reports (`ps-inputs valid=ffffffff ... 0=param0 1=param0 ... 31=param0`).
//
// The test carries its own MUTATION ARM. `consumed_known == false` is precisely the pre-fix
// behaviour, so the second half recompiles the same program with the bound switched off and asserts
// the interface really does blow the limit. Without that arm "1 location" would be consistent with a
// program that never had 32 in the first place, and the guard would be untested.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpVariable = 59;
constexpr uint32_t kDecorationLocation = 30;
constexpr uint32_t kStorageClassInput = 1;
constexpr uint32_t kStorageClassOutput = 3;

// Locations decorated on variables of `storage_class`. Walking the module rather than trusting a
// counter in the emitter: the property under test is what the DRIVER will see.
std::set<uint32_t> interface_locations(const std::vector<uint32_t>& spirv, uint32_t storage_class) {
    std::set<uint32_t> result;
    if (spirv.size() < 5) return result;
    std::vector<std::pair<uint32_t, uint32_t>> decorated;   // id -> location
    std::set<uint32_t> ids;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        const uint32_t opcode = spirv[word] & 0xffffu;
        if (count == 0 || word + count > spirv.size()) break;
        if (opcode == kOpDecorate && count >= 4 && spirv[word + 2] == kDecorationLocation)
            decorated.push_back({spirv[word + 1], spirv[word + 3]});
        if (opcode == kOpVariable && count >= 4 && spirv[word + 3] == storage_class)
            ids.insert(spirv[word + 2]);
        word += count;
    }
    for (const auto& [id, location] : decorated)
        if (ids.count(id)) result.insert(location);
    return result;
}

// The pixel-input wiring a sticky SPI_PS_INPUT_CNTL block produces: every slot valid, every slot
// selecting PARAM0.
PixelInputMapping sticky_all_slots_param0() {
    PixelInputMapping mapping;
    mapping.valid_mask = 0xffffffffu;
    for (uint32_t input = 0; input < mapping.controls.size(); ++input) mapping.controls[input] = 0;
    return mapping;
}

size_t stores_to_location(const std::vector<uint32_t>& spirv, uint32_t location) {
    uint32_t variable = 0;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        if (!count || word + count > spirv.size()) return SIZE_MAX;
        if ((spirv[word] & 0xffffu) == kOpDecorate && count == 4 &&
            spirv[word + 2] == kDecorationLocation && spirv[word + 3] == location)
            variable = spirv[word + 1];
        word += count;
    }
    size_t stores = 0;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        if (!count || word + count > spirv.size()) return SIZE_MAX;
        if ((spirv[word] & 0xffffu) == 62 && count >= 3 && spirv[word + 1] == variable)
            ++stores;
        word += count;
    }
    return stores;
}

}  // namespace

int main() {
    printf("== test_vertex_output_budget ==\n");

    // VS: fullscreen triangle from gl_VertexIndex, plus EXP PARAM0 = (float(vid), 0, 0, 1).
    // Same program as test_interp_render, which is the one that proves the export is real.
    const uint32_t vs[] = {
        0x7e140d00u, 0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100202f6u, 0x100404f6u,
        0x060202f3u, 0x060404f3u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xf800020fu,
        0x0403030au, 0xbf810000u,
    };
    // PS: v_interp attr0.x -> v0; exp mrt0. It reads attribute 0 and nothing else.
    const uint32_t ps[] = {
        0xc8000000u, 0xc8010001u, 0x7e020280u, 0x7e0402f2u, 0xf800080fu, 0x02010100u, 0xbf810000u,
    };
    const size_t vs_dwords = sizeof(vs) / sizeof(vs[0]);
    const size_t ps_dwords = sizeof(ps) / sizeof(ps[0]);

    // --- The analysis itself -------------------------------------------------------------------
    CHECK(fragment_consumed_attribute_mask(ps, ps_dwords) == 1u,
          "a PS whose only VINTRP reads attribute 0 consumes exactly {0}");
    CHECK(fragment_consumed_attribute_mask(nullptr, 0) == 0u,
          "no program to analyse consumes nothing (no read through a null pointer)");

    // THE CONTAINMENT ARM. The bound is safe only if what the vertex stage keeps covers every
    // Location the fragment stage declares, and the fragment emitter's declarations come from
    // FragmentInterpolationLayout::attribute_mask. An earlier revision of this analysis routed
    // through `rdna2_walk` and was therefore EQUAL to that mask rather than a superset -- correct,
    // but with no margin, and its comment claimed a margin it did not have. This pins the real
    // property instead of the claim.
    const uint32_t consumed = fragment_consumed_attribute_mask(ps, ps_dwords);
    const uint32_t declared = fragment_interpolation_layout(ps, ps_dwords).attribute_mask;
    printf("  consumed=%08x declared=%08x\n", consumed, declared);
    CHECK((declared & ~consumed) == 0,
          "the consumed mask CONTAINS every attribute the interpolation layout declares");

    // Containment is satisfied by EQUALITY, and equality is the normal case -- say so here rather
    // than let a reader infer a margin from the assertion. `fragment_consumed_attribute_mask` walks
    // `rdna2_recompile_code_span`, and for an ordinary program that span IS the prefix
    // `fragment_interpolation_layout`'s own `rdna2_walk` stops at, so the two masks come out equal.
    //
    // Where the margin exists is the case that matters: the span is extended by
    // `extend_terminating_if_else` and by the pcrel table/dispatch detectors, and this function does
    // not stop at `is_end` inside it. So if the fragment path ever gains the tail extension
    // `recompile_valu` and `recompile_compute` already have -- the hazard that makes an equal mask
    // unsafe -- the span grows and this walk follows it, while a plain `rdna2_walk` would not.
    // The line below is the property that holds unconditionally; the margin above is conditional,
    // and neither is stronger than it is.

    PixelInputMapping bounded = sticky_all_slots_param0();
    apply_fragment_consumption(bounded, ps, ps_dwords);
    // apply_fragment_consumption is a no-op under PROSPER_NO_DEAD_VARYING_ELIM, which would make
    // every assertion below vacuous rather than failing. Say so instead of measuring nothing.
    if (!bounded.consumed_known) {
        printf("  [SKIP] PROSPER_NO_DEAD_VARYING_ELIM is set; this test measures the bound it "
               "disables\n== SKIP ==\n");
        return 0;
    }
    CHECK(bounded.consumed_mask == 1u, "the stamped mapping carries the fragment program's {0}");
    CHECK(bounded.valid_mask == 0xffffffffu,
          "valid_mask is left alone -- the bound is a separate fact, not a rewrite of the registers");
    CHECK(bounded.consumes(0) && !bounded.consumes(1) && !bounded.consumes(31),
          "consumes() answers per slot");

    // --- The fix -------------------------------------------------------------------------------
    std::vector<uint32_t> bounded_vs =
        recompile_vertex(vs, vs_dwords, nullptr, &bounded);
    CHECK(!bounded_vs.empty() && bounded_vs[0] == 0x07230203u, "bounded VS recompiles to SPIR-V");
    const std::set<uint32_t> bounded_out = interface_locations(bounded_vs, kStorageClassOutput);
    printf("  bounded vertex output locations: %zu\n", bounded_out.size());
    CHECK(bounded_out.size() == 1 && bounded_out.count(0) == 1,
          "one PARAM0 export against a sticky all-slots mask emits ONE output varying");
    // 4 components per location, plus gl_Position's 4. 128 is maxVertexOutputComponents on every
    // AMD device prosper runs on. (The Vulkan minimum is 64; 128 is the value that matters here
    // because it is what the device reporting the error actually advertises.)
    const size_t bounded_components = bounded_out.size() * 4 + 4;
    CHECK(bounded_components <= 128,
          "the bounded vertex interface fits maxVertexOutputComponents (128)");

    // The fragment stage must still find every input it declares. This is what makes dropping the
    // rest safe rather than merely smaller.
    std::vector<uint32_t> frag = recompile_fragment(ps, ps_dwords);
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "PS recompiles to SPIR-V");
    const std::set<uint32_t> frag_in = interface_locations(frag, kStorageClassInput);
    bool covered = true;
    for (uint32_t location : frag_in) if (!bounded_out.count(location)) covered = false;
    CHECK(covered, "every fragment input location is still produced by the vertex stage");

    // #3416: remove the synthetic PARAM export while retaining the fullscreen position export.
    // The paired PS still reads location 0. Its interface must exist without manufacturing a value
    // that the guest VS never wrote, and without adding the other 31 sticky controls.
    std::vector<uint32_t> position_only(vs, vs + vs_dwords - 3);
    position_only.push_back(0xbf810000u);
    const auto missing_param = recompile_vertex(
        position_only.data(), position_only.size(), nullptr, &bounded);
    CHECK(!missing_param.empty(), "position-only VS with a consumed PS input recompiles");
    const auto missing_out = interface_locations(missing_param, kStorageClassOutput);
    CHECK(missing_out == frag_in && missing_out.size() == 1,
          "an unexported but consumed parameter still has the required Vulkan interface");
    CHECK(stores_to_location(missing_param, 0) == 0,
          "an absent guest PARAM export does not acquire an invented value");
    CHECK(stores_to_location(bounded_vs, 0) > 0,
          "a real guest PARAM export still writes the consumed location");

    // --- The mutation arm: the pre-fix behaviour, and it must blow the limit -------------------
    PixelInputMapping unbounded = sticky_all_slots_param0();   // consumed_known stays false
    std::vector<uint32_t> unbounded_vs =
        recompile_vertex(vs, vs_dwords, nullptr, &unbounded);
    const std::set<uint32_t> unbounded_out = interface_locations(unbounded_vs, kStorageClassOutput);
    const size_t unbounded_components = unbounded_out.size() * 4 + 4;
    printf("  unbounded vertex output locations: %zu (%zu components)\n",
           unbounded_out.size(), unbounded_components);
    CHECK(unbounded_out.size() == 32,
          "MUTATION ARM: without the bound, one export fans out to all 32 sticky slots");
    CHECK(unbounded_components > 128,
          "MUTATION ARM: that interface exceeds maxVertexOutputComponents, so the assertions above "
          "are testing a real guard");

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
