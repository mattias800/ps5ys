// test_multidraw_render — the multi-draw backend (render_runner.h render_draws_rgba) records N draws
// into ONE framebuffer (cleared once), each with its own pipeline + blend state. Proof: a red OPAQUE
// fullscreen draw followed by a green ADDITIVE fullscreen draw composites to YELLOW at the center — a
// color that is impossible unless both draws hit the same accumulating target (a fresh clear per draw
// would leave only the last draw's green). This exercises the multi-draw spine independently of the
// game's per-draw register-snapshot resolution.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/render_state.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    printf("== test_multidraw_render ==\n");
    CHECK(prosper::test::persistent_pipeline_cache_limit_value(nullptr) == 4096,
          "pipeline cache default covers modern deferred-renderer working sets");
    CHECK(prosper::test::persistent_pipeline_cache_limit_value("17") == 17,
          "pipeline cache entry override remains exact");
    set_env("PROSPER_RENDER_TIMING", "1");
    set_env("PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES", "2");
    const uint32_t W = 64, H = 64;

    // Known-good fullscreen-triangle vertex shader (SPIR-V), shared by both draws.
    #include "../tools/boot_trace/refvs.inc"
    std::vector<uint32_t> vs(kRefVs, kRefVs + sizeof(kRefVs) / 4);
    auto ref_vs_with_depth = [](std::vector<uint32_t> words, uint32_t depth_bits) {
        // kRefVs builds gl_Position.z from float OpConstant %37. Patch only that literal so tests can
        // model the tiny cross-shader position drift seen between UE4's depth and base passes.
        for (size_t i = 5; i < words.size();) {
            const uint32_t word_count = words[i] >> 16;
            const uint32_t opcode = words[i] & 0xffffu;
            if (word_count == 4 && opcode == 43 /* OpConstant */ &&
                words[i + 1] == 6 /* float type */ && words[i + 2] == 0x25) {
                words[i + 3] = depth_bits;
                return words;
            }
            if (!word_count || i + word_count > words.size()) break;
            i += word_count;
        }
        return std::vector<uint32_t>{};
    };

    // Two solid-color pixel shaders, recompiled from tiny RDNA2 EXP blobs (v_mov the 4 color VGPRs, then
    // EXP mrt0). Inline consts: 0xF2 = 1.0f, 0x80 = 0.0f.  RED = (1,0,0,1)  GREEN = (0,1,0,1).
    static const uint32_t kRedPs[]   = {0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    static const uint32_t kGreenPs[] = {0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    static const uint32_t kQuarterRedPs[] = {
        0x7E0002FFu, 0x3E800000u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    std::vector<uint32_t> red   = recompile_fragment(kRedPs,   sizeof(kRedPs)   / 4, nullptr);
    std::vector<uint32_t> green = recompile_fragment(kGreenPs, sizeof(kGreenPs) / 4, nullptr);
    std::vector<uint32_t> quarter_red = recompile_fragment(
        kQuarterRedPs, sizeof(kQuarterRedPs) / 4, nullptr);
    CHECK(!vs.empty() && !red.empty() && !green.empty() && !quarter_red.empty(),
          "fullscreen VS + red/green/quarter-red PS available");

    ResolvedPipelineState opaque{};
    opaque.topology = 3 /*VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST*/; opaque.color_write_mask = 0xF;
    ResolvedPipelineState additive = opaque;              // green = src*ONE + dst*ONE (accumulate onto red)
    additive.blend_enable = true;
    additive.src_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    additive.dst_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    additive.color_blend_op         = 0 /*VK_BLEND_OP_ADD*/;

    auto center = [&](const std::vector<uint8_t>& px) -> const uint8_t* {
        return px.empty() ? nullptr : &px[((size_t)(H / 2) * W + W / 2) * 4];
    };

    // Single opaque red draw -> red center (baseline; no accumulation).
    {
        prosper::test::BackendDraw d; d.vs = vs; d.fs = red; d.ps = &opaque; d.vcount = 3;
        std::vector<uint8_t> px = prosper::test::render_draws_rgba({d}, W, H);
        CHECK(px.size() == (size_t)W * H * 4, "single-draw path rendered a frame");
        const uint8_t* c = center(px);
        if (c) CHECK(c[0] > 0xC0 && c[1] < 0x40 && c[2] < 0x40, "one opaque red draw -> RED center");
    }

    // #919's contract, re-based on the right register by #1724: a colour-disabled draw reaches
    // Vulkan with zero write masks and can still populate stencil for a later colour pass. The
    // disabling now comes from the guest's explicit CB_TARGET_MASK, never from
    // CB_COLOR_CONTROL.MODE — see the block below, which asserts MODE=DISABLE alone does NOT
    // suppress, and render_state.cpp for the Astro Bot measurement behind that change.
    {
        RenderState raw{};
        raw.prim_type = 4; // triangle list
        raw.cb_target_mask = 0x0u;          // the real colour-disable signal
        raw.cb_shader_mask = 0xFu;
        raw.has_cb_color_control = true;
        raw.cb_color_control = 0x00CC0010u; // MODE=NORMAL — irrelevant to the mask
        ResolvedPipelineState disabled = resolve_pipeline_state(raw);
        CHECK(disabled.color_write_mask == 0,
              "a zero CB_TARGET_MASK reaches Vulkan with colour writes disabled");

        ResolvedPipelineState writer = disabled;
        writer.stencil_enable = true;
        writer.stencil_compare_op[0] = writer.stencil_compare_op[1] = 7; // ALWAYS
        writer.stencil_pass_op[0] = writer.stencil_pass_op[1] = 2;       // REPLACE
        writer.stencil_op_val[0] = writer.stencil_op_val[1] = 2;
        writer.stencil_write_mask[0] = writer.stencil_write_mask[1] = 0xff;

        prosper::test::BackendDraw w; w.vs = vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        const std::vector<uint8_t> suppressed = prosper::test::render_draws_rgba({w}, W, H);
        const uint8_t* sc = center(suppressed);
        CHECK(sc && sc[2] > 0xC0 && sc[0] < 0x40 && sc[1] < 0x40,
              "zero-target-mask red fragment preserves the blue colour attachment");

        ResolvedPipelineState reader = opaque;
        reader.stencil_enable = true;
        reader.stencil_compare_op[0] = reader.stencil_compare_op[1] = 2; // EQUAL
        reader.stencil_ref[0] = reader.stencil_ref[1] = 2;
        reader.stencil_compare_mask[0] = reader.stencil_compare_mask[1] = 0xff;
        prosper::test::BackendDraw r; r.vs = vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        const std::vector<uint8_t> consumed = prosper::test::render_draws_rgba({w, r}, W, H);
        const uint8_t* cc = center(consumed);
        CHECK(cc && cc[1] > 0xC0 && cc[0] < 0x40 && cc[2] < 0x40,
              "a colour-disabled draw still writes stencil consumed by a later colour pass");

        // #1724, at pixel level: MODE=DISABLE with an explicit write-all mask must RENDER.
        // The control above is what makes this non-vacuous: it renders the SAME red fragment through
        // the SAME harness with CB_TARGET_MASK=0 and gets blue, proving the harness honours a zero
        // write mask. So a red centre here can only come from the mask being kept.
        RenderState mode_disabled{};
        mode_disabled.prim_type = 4;
        mode_disabled.cb_target_mask = mode_disabled.cb_shader_mask = 0xFu;
        mode_disabled.has_cb_color_control = true;
        mode_disabled.cb_color_control = 0x00CC0000u;   // MODE=DISABLE, write-all masks
        ResolvedPipelineState mode_disabled_ps = resolve_pipeline_state(mode_disabled);
        CHECK(mode_disabled_ps.color_write_mask == 0xFu,
              "MODE=DISABLE with a write-all CB_TARGET_MASK resolves to an RGBA write mask");
        prosper::test::BackendDraw m; m.vs = vs; m.fs = red; m.ps = &mode_disabled_ps; m.vcount = 3;
        const std::vector<uint8_t> rendered = prosper::test::render_draws_rgba({m}, W, H);
        const uint8_t* mc = center(rendered);
        CHECK(mc && mc[0] > 0xC0 && mc[1] < 0x40 && mc[2] < 0x40,
              "MODE=DISABLE draw with an explicit mask reaches the colour attachment");
    }

    // Hardware instance count reaches both Vulkan submission paths. Additive quarter-red makes the
    // number of identical instances directly observable: one instance contributes ~64 red, while
    // three contribute ~192. The indexed sibling must produce the same pixels.
    {
        prosper::test::BackendDraw one;
        one.vs = vs; one.fs = quarter_red; one.ps = &additive; one.vcount = 3;
        prosper::test::BackendDraw zero = one; zero.instance_count = 0;
        prosper::test::BackendDraw zero_indexed = zero; zero_indexed.indices = {0, 1, 2};
        prosper::test::BackendDraw three = one; three.instance_count = 3;
        prosper::test::BackendDraw three_indexed = three; three_indexed.indices = {0, 1, 2};
        const std::vector<uint8_t> px_zero = prosper::test::render_draws_rgba({zero}, W, H);
        const std::vector<uint8_t> px_zero_indexed =
            prosper::test::render_draws_rgba({zero_indexed}, W, H);
        const std::vector<uint8_t> px_one = prosper::test::render_draws_rgba({one}, W, H);
        const std::vector<uint8_t> px_three = prosper::test::render_draws_rgba({three}, W, H);
        const std::vector<uint8_t> px_three_indexed =
            prosper::test::render_draws_rgba({three_indexed}, W, H);
        // A NON-ZERO index-buffer arena offset, which nothing else in the suite reaches (#2253).
        //
        // Every other indexed draw here and in test_indexed_render is alone in its
        // render_draws_rgba() call, so each gets a fresh arena and binds at offset 0 — the one
        // value that is still correct if `v.ioffset` were dropped from the bind sites entirely.
        // Those arms therefore cannot fail for the reason they appear to cover.
        //
        // Two indexed draws in ONE pass puts the second at align_storage_offset(12), non-zero by
        // construction. The index DATA must differ for the offset to matter: identical payloads
        // would render the same picture from either slice and the arm would be void. So the second
        // draw is degenerate ({0,0,0} collapses the triangle to a point, contributing nothing) and
        // the pass must match a single ordinary draw. Bind the second at offset 0 and it reads the
        // FIRST draw's {0,1,2}, drawing a second additive triangle — visible immediately.
        //
        // Verified by mutation, not assumed: hardcoding 0 at both vkCmdBindIndexBuffer sites turns
        // this arm red and leaves the rest of the file green.
        prosper::test::BackendDraw first_indexed = one;  first_indexed.indices = {0, 1, 2};
        prosper::test::BackendDraw second_degenerate = one; second_degenerate.indices = {0, 0, 0};
        const std::vector<uint8_t> px_pair =
            prosper::test::render_draws_rgba({first_indexed, second_degenerate}, W, H);
        CHECK(px_pair == px_one,
              "a second indexed draw in one pass reads ITS OWN slice at a non-zero arena offset "
              "(degenerate indices contribute nothing; binding at 0 would re-draw the first)");

        const uint8_t* zero_center = center(px_zero);
        const uint8_t* one_center = center(px_one);
        const uint8_t* three_center = center(px_three);
        CHECK(zero_center && zero_center[0] < 0x40 && zero_center[1] < 0x40 &&
                  zero_center[2] > 0xC0,
              "zero non-indexed instances leave the blue clear untouched");
        CHECK(!px_zero.empty() && px_zero_indexed == px_zero,
              "zero indexed instances are the same no-op");
        CHECK(one_center && one_center[0] >= 48 && one_center[0] <= 80,
              "one additive quarter-red instance contributes one layer");
        CHECK(three_center && one_center && three_center[0] >= one_center[0] + 96 &&
                  three_center[0] >= 160 && three_center[0] <= 224,
              "three non-indexed instances contribute three additive layers");
        CHECK(!px_three.empty() && px_three_indexed == px_three,
              "indexed and non-indexed Vulkan draws consume the same instance count");
    }

    // Red opaque THEN green additive, in one submit -> yellow center (both composited into one target).
    {
        prosper::test::BackendDraw d0; d0.vs = vs; d0.fs = red;   d0.ps = &opaque;   d0.vcount = 3;
        prosper::test::BackendDraw d1; d1.vs = vs; d1.fs = green; d1.ps = &additive; d1.vcount = 3;
        std::vector<uint8_t> px = prosper::test::render_draws_rgba({d0, d1}, W, H);
        CHECK(px.size() == (size_t)W * H * 4, "two-draw submit rendered a frame");
        const uint8_t* c = center(px);
        if (c) {
            CHECK(c[0] > 0xC0 && c[1] > 0xC0 && c[2] < 0x40,
                  "two draws composite into ONE cleared-once framebuffer -> YELLOW center (red+green)");
        }
        const prosper::test::BackendRenderTimingStats timing =
            prosper::test::backend_render_timing_stats();
        CHECK(timing.calls == 1 && timing.draws == 2,
              "backend timing publishes the completed call and draw count");
        CHECK(timing.total_ms() > 0 && timing.draw_setup_ms > 0,
              "backend timing publishes a non-empty phase breakdown");
        CHECK(timing.draw_setup_ms + 0.001 >= timing.setup_shader_ms + timing.setup_fixed_ms +
                  timing.setup_resources_ms + timing.setup_pipeline_ms,
              "draw-setup subphases fit inside the backend draw-setup phase");
        const prosper::test::BackendPipelineCacheStats first_cache =
            prosper::test::backend_pipeline_cache_stats();
        CHECK(first_cache.references == 2 && first_cache.hits == 1 && first_cache.misses == 1,
              "pipeline cache reuses the prior opaque pipeline but separates blend state");

        std::vector<uint8_t> cached = prosper::test::render_draws_rgba({d0, d1}, W, H);
        const prosper::test::BackendPipelineCacheStats cached_stats =
            prosper::test::backend_pipeline_cache_stats();
        CHECK(cached == px, "persistent pipeline hits preserve multi-draw pixels byte-for-byte");
        CHECK(cached_stats.references == 2 && cached_stats.hits == 2 && cached_stats.misses == 0,
              "repeated draw contracts hit the persistent pipeline cache");

        set_env("PROSPER_NO_BACKEND_PIPELINE_CACHE", "1");
        std::vector<uint8_t> bypassed = prosper::test::render_draws_rgba({d0, d1}, W, H);
        const prosper::test::BackendPipelineCacheStats bypassed_stats =
            prosper::test::backend_pipeline_cache_stats();
        set_env("PROSPER_NO_BACKEND_PIPELINE_CACHE", nullptr);
        CHECK(bypassed == px, "pipeline-cache disable A/B preserves output byte-for-byte");
        CHECK(bypassed_stats.references == 2 && bypassed_stats.bypasses == 2 &&
                  bypassed_stats.hits == 0,
              "pipeline-cache disable A/B bypasses every lookup");
    }

    // Core dynamic state changes must reuse ONE pipeline and still change coverage. The priming
    // draw is culled completely, so only the second draw can colour the center. It also retains
    // the depth attachment when the second draw disables depth, testing the mixed-enable pass.
    {
        ResolvedPipelineState prime = opaque;
        prime.depth_test_enable = true;
        prime.depth_compare_op = VK_COMPARE_OP_ALWAYS;
        // Keep the attachment format D32+S8 in every arm; switching D32 to D32+S8 is a valid
        // pipeline compatibility difference even when stencil operations themselves are dynamic.
        prime.stencil_enable = true;
        prime.stencil_compare_op[0] = prime.stencil_compare_op[1] = VK_COMPARE_OP_ALWAYS;
        prime.cull_mode = VK_CULL_MODE_FRONT_AND_BACK;
        prosper::test::BackendDraw d;
        d.vs = vs; d.fs = red; d.ps = &prime; d.vcount = 3;
        CHECK(!prosper::test::render_draws_rgba({d}, W, H).empty(),
              "warm the shared depth/stencil pipeline");
        for (unsigned axis = 0; axis < 10; ++axis) {
            ResolvedPipelineState state = prime;
            state.cull_mode = VK_CULL_MODE_NONE;
            bool visible = true;
            switch (axis) {
            case 0: state.depth_test_enable = false; state.stencil_enable = false; break;
            case 1: state.depth_write_enable = true; break;
            case 2: state.depth_compare_op = VK_COMPARE_OP_NEVER; visible = false; break;
            case 3:
                state.stencil_enable = true;
                state.stencil_compare_op[0] = state.stencil_compare_op[1] = VK_COMPARE_OP_NEVER;
                visible = false;
                break;
            case 4:
                state.stencil_enable = true;
                for (unsigned face = 0; face < 2; ++face) {
                    state.stencil_compare_op[face] = VK_COMPARE_OP_ALWAYS;
                    state.stencil_fail_op[face] = VK_STENCIL_OP_REPLACE;
                    state.stencil_pass_op[face] = VK_STENCIL_OP_INCREMENT_AND_CLAMP;
                    state.stencil_depth_fail_op[face] = VK_STENCIL_OP_DECREMENT_AND_CLAMP;
                }
                break;
            case 5: state.cull_mode = VK_CULL_MODE_FRONT_AND_BACK; visible = false; break;
            case 6: state.front_face = VK_FRONT_FACE_CLOCKWISE; break;
            case 7: state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
            case 8: state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
            default: break; // restore all defaults after the preceding changes
            }
            auto changed = d; changed.ps = &state;
            const auto pixels = prosper::test::render_draws_rgba({d, changed}, W, H);
            const auto cache = prosper::test::backend_pipeline_cache_stats();
            printf("  dynamic-state axis %u\n", axis);
            CHECK(cache.references == 2 && cache.hits == 2 && cache.misses == 0,
                  "depth/stencil/cull/topology values reuse the warmed pipeline");
            CHECK(pixels.size() == static_cast<size_t>(W) * H * 4,
                  "dynamic state render produced a complete frame");
            if (const auto* c = center(pixels))
                CHECK((c[0] > 0xC0) == visible,
                      "cached pipeline honours the current draw's coverage state");
        }
        CHECK(prosper::test::pipeline_topology_class(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) !=
                  prosper::test::pipeline_topology_class(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP),
              "line and triangle classes cannot alias on restricted dynamic-topology devices");
    }

    // Hundreds of Evergate draws repeat identical constant/vertex payloads and descriptor contracts.
    // A synchronous backend call may share those immutable objects after exact comparison, while the
    // disable switch provides a byte-identical one-object-per-reference control.
    {
        const uint32_t fetch_vs[] = {
            0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u,
            0xf80008cfu, 0x04030201u, 0xbf810000u,
        };
        ShaderResourceTable fetch_resources;
        ShaderResource vertex_resource{};
        vertex_resource.cls = ResourceClass::VertexBuffer;
        vertex_resource.format = DataFormat::Float32;
        vertex_resource.num_components = 2;
        vertex_resource.binding = 3;
        vertex_resource.stride = 8;
        vertex_resource.sgpr_base = 8;
        fetch_resources.resources.push_back(vertex_resource);
        const std::vector<uint32_t> buffer_vs = recompile_vertex(
            fetch_vs, sizeof(fetch_vs) / sizeof(fetch_vs[0]), &fetch_resources);
        CHECK(!buffer_vs.empty(), "storage-buffer-reading vertex shader available");
        auto f = [](float value) {
            union { float f; uint32_t u; } bits{value};
            return bits.u;
        };
        prosper::test::FrameResource filler;
        filler.binding = 2;
        filler.set = 0;
        filler.buffer_identity = 0x7020000000000002ull;
        filler.dwords.assign(17, 0x40000000u);
        prosper::test::FrameResource buffer;
        buffer.binding = 3;
        buffer.set = 0;
        buffer.buffer_identity = 0x7020000000000003ull;
        buffer.dwords.assign(63, 0u);
        const uint32_t fullscreen_triangle[] = {
            f(-1.0f), f(-1.0f), f(3.0f), f(-1.0f), f(-1.0f), f(3.0f),
        };
        std::copy(std::begin(fullscreen_triangle), std::end(fullscreen_triangle),
                  buffer.dwords.begin());
        prosper::test::BackendDraw d0;
        d0.vs = buffer_vs; d0.fs = red; d0.ps = &opaque; d0.vcount = 3;
        d0.R = {filler, buffer};
        prosper::test::BackendDraw d1;
        d1.vs = buffer_vs; d1.fs = green; d1.ps = &additive; d1.vcount = 3;
        d1.R = {filler, buffer};

        const auto pool_before = prosper::test::render_host_buffer_pool_stats();
        const std::vector<uint8_t> shared =
            prosper::test::render_draws_rgba({d0, d1}, W, H);
        const auto shared_stats = prosper::test::backend_resource_reuse_stats();
        const auto pool_after_first = prosper::test::render_host_buffer_pool_stats();
        CHECK(shared_stats.buffer_references == 4 && shared_stats.unique_buffers == 2,
              "identical guest-buffer payloads share two arena slices across draws");
        // AT MOST one new arena buffer, not EXACTLY one. The property this asserts is that two
        // logical uploads share a single arena rather than taking a buffer each; the old `== +1`
        // additionally required the arena to be EMPTY on entry, which is a precondition of the
        // fixture rather than a property of the renderer. Indexed draws take arena slices since
        // #2253, so the indexed draws at :165/:167 leave a partially-used arena and these two
        // uploads now fit inside it — one arena, zero new buffers, the property satisfied and the
        // old proxy violated.
        //
        // The inequality keeps the discriminating power that matters: the regression this guards
        // against is each logical upload allocating its own host buffer, which is `+2` and still
        // fails. Verified by construction rather than assumed — `PROSPER_NO_INDEX_ARENA=1` restores
        // the empty-arena precondition and the original `== +1` holds, which is what identified the
        // dependency in the first place.
        CHECK(pool_after_first.misses <= pool_before.misses + 1 &&
                  pool_after_first.cached_buffers <= pool_before.cached_buffers + 1,
              "two logical uploads share one persistent host-buffer arena (at most one new buffer)");
        const uint8_t* shared_center = center(shared);
        CHECK(shared_center && shared_center[0] > 0xC0 && shared_center[1] > 0xC0 &&
                  shared_center[2] < 0x40,
              "shader reads the fullscreen vertex payload from a nonzero arena offset");
        CHECK(shared_stats.descriptor_set_layout_references == 2 &&
                  shared_stats.unique_descriptor_set_layouts == 1 &&
                  shared_stats.pipeline_layout_references == 2 &&
                  shared_stats.unique_pipeline_layouts == 1 &&
                  shared_stats.descriptor_pools == 1,
              "identical draw contracts share layouts and one call-wide descriptor pool");

        // The live frontend can hand the synchronous backend a proven-readable immutable guest view
        // instead of copying it into each FrameResource. Exercise that representation independently of
        // guest memory and require it to preserve rendering plus exact-content upload reuse.
        prosper::test::BackendDraw shared_contents0 = d0;
        prosper::test::BackendDraw shared_contents1 = d1;
        for (prosper::test::BackendDraw* draw : {&shared_contents0, &shared_contents1}) {
            draw->R[0].dwords.clear();
            draw->R[0].dwords_view = filler.dwords.data();
            draw->R[0].dwords_view_count = filler.dwords.size();
            draw->R[1].dwords.clear();
            draw->R[1].dwords_view = buffer.dwords.data();
            draw->R[1].dwords_view_count = buffer.dwords.size();
        }
        const std::vector<uint8_t> shared_contents = prosper::test::render_draws_rgba(
            {shared_contents0, shared_contents1}, W, H);
        const auto shared_contents_stats = prosper::test::backend_resource_reuse_stats();
        const auto pool_after_shared_contents = prosper::test::render_host_buffer_pool_stats();
        CHECK(shared_contents == shared,
              "immutable buffer views render byte-identically to owned vectors");
        CHECK(shared_contents_stats.buffer_references == 4 &&
                  shared_contents_stats.unique_buffers == 2,
              "immutable buffer views retain exact backend upload reuse");
        // #1268: the second draw's references carry the SAME view pointer + identity, so the
        // per-call memo must resolve them without re-hashing; only the first draw's two
        // references pay a content hash. (The owned-vector block above copies the dwords into
        // each BackendDraw, so its four references have distinct pointers and all hash —
        // asserted below — which is exactly the hash+memcmp dedup path the memo falls back to.)
        CHECK(shared_contents_stats.buffer_ref_memo_hits == 2 &&
                  shared_contents_stats.buffer_hash_calls == 2 &&
                  shared_contents_stats.buffer_hash_skipped_unique == 0,
              "repeat view references resolve via the per-call memo without re-hashing");
        CHECK(shared_stats.buffer_ref_memo_hits == 0 &&
                  shared_stats.buffer_hash_calls == 4,
              "owned-vector references (distinct pointers) keep full hash+memcmp dedup");

        prosper::test::BackendDraw capacity_peer0 = d0;
        prosper::test::BackendDraw capacity_peer1 = d1;
        capacity_peer0.R[1].dwords.resize(61);
        capacity_peer1.R[1].dwords.resize(61);
        const std::vector<uint8_t> pooled_again = prosper::test::render_draws_rgba(
            {capacity_peer0, capacity_peer1}, W, H);
        const auto pool_after_second = prosper::test::render_host_buffer_pool_stats();
        const auto layout_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(pool_after_second.hits == pool_after_shared_contents.hits + 1 &&
                  pool_after_second.misses == pool_after_shared_contents.misses &&
                  pooled_again == shared,
              "next logical size reuses the mapped arena byte-identically");
        CHECK(layout_stats.persistent_pipeline_layout_hits >= 1 &&
                  layout_stats.persistent_pipeline_layout_entries >= 1,
              "equal pipeline-layout contracts reuse the bounded cross-call cache");

        set_env("PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE", "1");
        const std::vector<uint8_t> uncached_layouts =
            prosper::test::render_draws_rgba({d0, d1}, W, H);
        const auto uncached_layout_stats = prosper::test::backend_resource_reuse_stats();
        set_env("PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE", nullptr);
        CHECK(uncached_layout_stats.persistent_pipeline_layout_hits == 0 &&
                  uncached_layouts == shared,
              "layout-cache disable switch restores byte-identical call-local layouts");

        const auto pool_before_bypass = prosper::test::render_host_buffer_pool_stats();
        set_env("PROSPER_NO_BACKEND_BUFFER_POOL", "1");
        const std::vector<uint8_t> pool_bypassed =
            prosper::test::render_draws_rgba({d0, d1}, W, H);
        const auto pool_after_bypass = prosper::test::render_host_buffer_pool_stats();
        set_env("PROSPER_NO_BACKEND_BUFFER_POOL", nullptr);
        CHECK(pool_after_bypass.hits == pool_before_bypass.hits &&
                  pool_after_bypass.misses == pool_before_bypass.misses,
              "buffer-pool disable switch restores transient Vulkan uploads");
        CHECK(pool_bypassed == shared,
              "pooled and forced-transient storage buffers render byte-identically");

        // #1432: every failure in the fresh Vulkan upload path must tear down any partial objects,
        // avoid copying through a null mapping, and bind one zero word instead. A buffer-reading VS
        // makes that fallback observable: the valid payload covers the target red, while zeros
        // collapse all three vertices to one point and preserve the blue clear.
        prosper::test::BackendDraw failure_draw = d0;
        failure_draw.R = {buffer};
        set_env("PROSPER_NO_BACKEND_BUFFER_POOL", "1");
        const prosper::test::RenderBufferUploadFailureStep failure_steps[] = {
            prosper::test::RenderBufferUploadFailureStep::CreateBuffer,
            prosper::test::RenderBufferUploadFailureStep::AllocateMemory,
            prosper::test::RenderBufferUploadFailureStep::BindMemory,
            prosper::test::RenderBufferUploadFailureStep::MapMemory,
        };
        for (const auto step : failure_steps) {
            prosper::test::inject_render_buffer_upload_failure_once(step);
            const std::vector<uint8_t> fallback =
                prosper::test::render_draws_rgba({failure_draw}, W, H);
            const uint8_t* fallback_center = center(fallback);
            const auto fallback_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(fallback_center && fallback_center[0] < 0x40 &&
                      fallback_center[1] < 0x40 && fallback_center[2] > 0xC0 &&
                      fallback_stats.buffer_upload_fallbacks == 1 &&
                      fallback_stats.unique_buffers == 1,
                  "fresh buffer-upload failure binds a one-word zero fallback without crashing");
        }
        set_env("PROSPER_NO_BACKEND_BUFFER_POOL", nullptr);

        // #1268: buffers above the 4 KiB hash-dedup bound skip content hashing entirely (live Blue
        // Prince data: zero dedup hits across 556K hashed references at ~1.8 GiB/s of hashing).
        // Repeat references still resolve through the per-call memo, and rendering is unchanged.
        // (Placed after the pool-delta assertions above: the 8 KiB upload adds its own pool miss.)
        {
            prosper::test::FrameResource large = buffer;
            large.dwords.assign(2048, 0u);
            std::copy(std::begin(fullscreen_triangle), std::end(fullscreen_triangle),
                      large.dwords.begin());
            large.buffer_identity = 0x7020000000000004ull;
            prosper::test::BackendDraw large0 = d0;
            prosper::test::BackendDraw large1 = d1;
            for (prosper::test::BackendDraw* draw : {&large0, &large1}) {
                draw->R[1].dwords.clear();
                draw->R[1].dwords_view = large.dwords.data();
                draw->R[1].dwords_view_count = large.dwords.size();
                draw->R[1].buffer_identity = large.buffer_identity;
            }
            const std::vector<uint8_t> large_frame =
                prosper::test::render_draws_rgba({large0, large1}, W, H);
            const auto large_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(large_frame == shared,
                  "large-buffer views render byte-identically (hash-dedup bound is inert visually)");
            CHECK(large_stats.buffer_hash_skipped_large == 1 &&
                      large_stats.buffer_ref_memo_hits == 1 &&
                      large_stats.buffer_hash_calls == 2 &&
                      large_stats.unique_buffers == 2,
              "large buffer skips content hashing; its repeat reference resolves via the memo "
              "(the two owned filler copies keep small-buffer hash dedup)");
        }

        // #1268 review follow-up: the memo key includes buffer_identity — the SAME view pointer
        // under two DIFFERENT identities must not memo-merge (equal bytes at distinct identities
        // remain distinct uploads, as the identity-keyed SharedBufferKey already guarantees).
        {
            prosper::test::BackendDraw ident0 = d0;
            prosper::test::BackendDraw ident1 = d1;
            for (prosper::test::BackendDraw* draw : {&ident0, &ident1}) {
                draw->R[1].dwords.clear();
                draw->R[1].dwords_view = buffer.dwords.data();
                draw->R[1].dwords_view_count = buffer.dwords.size();
            }
            ident1.R[1].buffer_identity++;
            const std::vector<uint8_t> ident_frame =
                prosper::test::render_draws_rgba({ident0, ident1}, W, H);
            const auto ident_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(ident_frame == shared,
                  "distinct-identity views render byte-identically");
            CHECK(ident_stats.buffer_ref_memo_hits == 0 && ident_stats.unique_buffers == 3,
                  "the memo keys on identity: one view pointer under two identities stays distinct");
        }

        prosper::test::BackendDraw distinct = d1;
        distinct.R[1].buffer_identity++;
        const std::vector<uint8_t> distinct_output =
            prosper::test::render_draws_rgba({d0, distinct}, W, H);
        const auto distinct_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(distinct_stats.buffer_references == 4 && distinct_stats.unique_buffers == 3 &&
                  distinct_output == shared,
              "equal bytes at different guest buffer identities remain distinct");

        set_env("PROSPER_NO_BACKEND_RESOURCE_SHARE", "1");
        const std::vector<uint8_t> unshared =
            prosper::test::render_draws_rgba({d0, d1}, W, H);
        const auto unshared_stats = prosper::test::backend_resource_reuse_stats();
        set_env("PROSPER_NO_BACKEND_RESOURCE_SHARE", nullptr);
        CHECK(unshared_stats.buffer_references == 4 && unshared_stats.unique_buffers == 4 &&
                  unshared_stats.unique_descriptor_set_layouts == 2 &&
                  unshared_stats.unique_pipeline_layouts == 2,
              "resource-share disable switch restores distinct per-draw immutable objects");
        CHECK(!shared.empty() && shared == unshared,
              "shared and unshared per-draw resources render byte-identically");

        const std::vector<uint8_t> original_layout_output =
            prosper::test::render_draws_rgba({d0}, W, H);
        prosper::test::BackendDraw alternate_layout = d0;
        alternate_layout.R[0].binding = 1;
        const std::vector<uint8_t> alternate_output =
            prosper::test::render_draws_rgba({alternate_layout}, W, H);
        const auto bounded_layout_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(alternate_output == original_layout_output &&
                  bounded_layout_stats.persistent_pipeline_layout_entries == 2 &&
                  bounded_layout_stats.persistent_pipeline_layout_evictions >= 1,
              "pipeline-layout cache evicts an older contract at its configured bound");

        // Warm both bounded entries, then require three distinct contracts in one call. The third
        // layout must remain call-local: evicting either of the first two would destroy an object
        // already referenced by a pipeline recorded for this submission.
        prosper::test::render_draws_rgba({d0}, W, H);
        prosper::test::render_draws_rgba({alternate_layout}, W, H);
        prosper::test::BackendDraw overflow_layout = d0;
        overflow_layout.R[0].binding = 0;
        const std::vector<uint8_t> protected_output = prosper::test::render_draws_rgba(
            {d0, alternate_layout, overflow_layout}, W, H);
        const auto protected_layout_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(protected_output == original_layout_output &&
                  protected_layout_stats.persistent_pipeline_layout_hits >= 2 &&
                  protected_layout_stats.persistent_pipeline_layout_misses >= 1 &&
                  protected_layout_stats.persistent_pipeline_layout_entries == 2 &&
                  protected_layout_stats.persistent_pipeline_layout_evictions == 0,
              "pipeline-layout eviction preserves every contract used by the current call");
    }

    // #531: a guest scissor must constrain a color-disabled stencil writer. The following full-target
    // reader shades only where stencil==2, proving both the dynamic draw scissor and its exclusive
    // right/bottom edges affect the depth/stencil attachment rather than only color output.
    {
        ResolvedPipelineState writer = opaque;
        writer.color_write_mask = 0;
        writer.has_scissor = true;
        writer.scissor_left = 16; writer.scissor_top = 12;
        writer.scissor_right = 48; writer.scissor_bottom = 44;
        writer.stencil_enable = true;
        writer.stencil_compare_op[0] = writer.stencil_compare_op[1] = 7; // ALWAYS
        writer.stencil_pass_op[0] = writer.stencil_pass_op[1] = 2;       // REPLACE
        writer.stencil_op_val[0] = writer.stencil_op_val[1] = 2;
        writer.stencil_write_mask[0] = writer.stencil_write_mask[1] = 0xff;

        ResolvedPipelineState reader = opaque;
        reader.stencil_enable = true;
        reader.stencil_compare_op[0] = reader.stencil_compare_op[1] = 2; // EQUAL
        reader.stencil_ref[0] = reader.stencil_ref[1] = 2;
        reader.stencil_compare_mask[0] = reader.stencil_compare_mask[1] = 0xff;

        ResolvedPipelineState off_target_clear = opaque;
        off_target_clear.color_write_mask = 0;
        off_target_clear.stencil_clear_enable = true;
        off_target_clear.stencil_clear_value = 0;
        // #1361: keep this an EFFECTIVE clear shape (enable + write mask). With the #1355
        // write-path gate, a bare clear bit is inert and this check would pass vacuously without
        // exercising the off-target scissor clamp it was written for (#531).
        off_target_clear.stencil_enable = true;
        off_target_clear.stencil_compare_op[0] = off_target_clear.stencil_compare_op[1] = 7;
        off_target_clear.stencil_write_mask[0] = off_target_clear.stencil_write_mask[1] = 0xff;
        off_target_clear.has_scissor = true;
        off_target_clear.scissor_left = 100; off_target_clear.scissor_top = 100;
        off_target_clear.scissor_right = 120; off_target_clear.scissor_bottom = 120;

        prosper::test::BackendDraw w; w.vs = vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        prosper::test::BackendDraw c;
        c.vs = vs; c.fs = red; c.ps = &off_target_clear; c.vcount = 3;
        prosper::test::BackendDraw r; r.vs = vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        const std::vector<uint8_t> px = prosper::test::render_draws_rgba({w, c, r}, W, H);
        auto green_at = [&](uint32_t x, uint32_t y) {
            const uint8_t* p = px.empty() ? nullptr : &px[((size_t)y * W + x) * 4];
            return p && p[1] > 0xC0 && p[0] < 0x40 && p[2] < 0x40;
        };
        CHECK(green_at(16, 12) && green_at(47, 43),
              "scissored stencil writer includes its left/top and last interior samples");
        CHECK(!green_at(15, 12) && !green_at(48, 12) && !green_at(16, 44),
              "scissored stencil writer excludes outside and right/bottom boundary samples");
        CHECK(green_at(32, 32),
              "fully off-target explicit clear is skipped without erasing the scissored stencil");
    }

    // A guest depth/stencil surface survives renderer calls. The first call writes stencil=2 with no
    // color; the second call LOADs the same guest-identified attachment and an EQUAL-2 draw turns green.
    // A different identity starts cleared and fails, while an explicit clear on the persisted identity
    // also rejects the reader. This is the cross-submit contract used by Messenger's level masks (#518).
    {
        ResolvedPipelineState writer = opaque;
        writer.color_write_mask = 0;
        writer.stencil_enable = true;
        writer.stencil_compare_op[0] = writer.stencil_compare_op[1] = 7; // ALWAYS
        writer.stencil_pass_op[0] = writer.stencil_pass_op[1] = 2;       // REPLACE
        writer.stencil_op_val[0] = writer.stencil_op_val[1] = 2;
        writer.stencil_write_mask[0] = writer.stencil_write_mask[1] = 0xff;
        writer.stencil_read_base = writer.stencil_write_base = 0x11110000;

        ResolvedPipelineState reader = opaque;
        reader.stencil_enable = true;
        reader.stencil_compare_op[0] = reader.stencil_compare_op[1] = 2; // EQUAL
        reader.stencil_ref[0] = reader.stencil_ref[1] = 2;
        reader.stencil_compare_mask[0] = reader.stencil_compare_mask[1] = 0xff;
        reader.stencil_read_base = reader.stencil_write_base = 0x11110000;

        prosper::test::BackendDraw w; w.vs = vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        prosper::test::BackendDraw r; r.vs = vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        (void)prosper::test::render_draws_rgba({w}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> loaded = prosper::test::render_draws_rgba({r}, W, H, nullptr, nullptr, true);
        const uint8_t* lc = center(loaded);
        CHECK(lc && lc[1] > 0xC0 && lc[0] < 0x40,
              "persistent guest DS identity LOADs stencil written by an earlier renderer call");

        // The same dependency must survive when both calls are recorded before one queue submit.
        // Give the calls GPU-only color targets so the stencil producer does not force an early CPU
        // readback; the consumer's requested readback flushes both command buffers behind one fence.
        ResolvedPipelineState batched_writer = writer;
        batched_writer.stencil_read_base = batched_writer.stencil_write_base = 0x99410000;
        ResolvedPipelineState batched_reader = reader;
        batched_reader.stencil_read_base = batched_reader.stencil_write_base = 0x99410000;
        prosper::test::BackendDraw bw = w; bw.ps = &batched_writer;
        prosper::test::BackendDraw br = r; br.ps = &batched_reader;
        prosper::test::BackendColorTarget batch_write_target{
            0x9941000000000001ull, false, false};
        prosper::test::BackendColorTarget batch_read_target{
            0x9941000000000002ull, false, true};
        prosper::test::BackendSubmissionBatch ds_batch;
        const std::vector<uint8_t> pending = prosper::test::render_draws_rgba(
            {bw}, W, H, nullptr, nullptr, true, &batch_write_target,
            nullptr, nullptr, nullptr, &ds_batch, false);
        const std::vector<uint8_t> batched_loaded = prosper::test::render_draws_rgba(
            {br}, W, H, nullptr, nullptr, true, &batch_read_target,
            nullptr, nullptr, nullptr, &ds_batch, true);
        const uint8_t* blc = center(batched_loaded);
        CHECK(pending.empty() && blc && blc[1] > 0xC0 && blc[0] < 0x40,
              "batched renderer calls publish persistent stencil before a later LOAD");

        ResolvedPipelineState fresh = reader;
        fresh.stencil_read_base = fresh.stencil_write_base = 0x22220000;
        prosper::test::BackendDraw f = r; f.ps = &fresh;
        std::vector<uint8_t> rejected = prosper::test::render_draws_rgba({f}, W, H, nullptr, nullptr, true);
        const uint8_t* fc = center(rejected);
        CHECK(fc && fc[1] < 0x80, "new guest DS identity starts cleared and rejects EQUAL-2");

        ResolvedPipelineState cleared = reader;
        cleared.stencil_clear_enable = true;
        cleared.stencil_clear_value = 0;
        prosper::test::BackendDraw c = r; c.ps = &cleared;
        std::vector<uint8_t> after_clear = prosper::test::render_draws_rgba({c}, W, H, nullptr, nullptr, true);
        const uint8_t* cc = center(after_clear);
        CHECK(cc && cc[1] < 0x80, "explicit guest stencil clear executes in order before the draw");
    }

    // Initializing one aspect of a combined D32S8 image must not make the other aspect valid. Unity
    // first uses this surface for stencil while depth is ALWAYS and read-only, then changes to GEQUAL.
    // The later draw must initialize the still-unused depth plane from its reverse-Z clear value (0),
    // rather than LOADing the depth fallback (1) used while only stencil contents mattered (#540).
    {
        ResolvedPipelineState stencil_first = opaque;
        stencil_first.color_write_mask = 0;
        stencil_first.depth_test_enable = true;
        stencil_first.depth_compare_op = 7; // ALWAYS
        stencil_first.depth_clear_value = 1.0f;
        stencil_first.depth_read_base = stencil_first.depth_write_base = 0x33330000;
        stencil_first.stencil_enable = true;
        stencil_first.stencil_compare_op[0] = stencil_first.stencil_compare_op[1] = 7; // ALWAYS
        stencil_first.stencil_pass_op[0] = stencil_first.stencil_pass_op[1] = 2;       // REPLACE
        stencil_first.stencil_op_val[0] = stencil_first.stencil_op_val[1] = 1;
        stencil_first.stencil_read_base = stencil_first.stencil_write_base = 0x44440000;

        ResolvedPipelineState depth_later = opaque;
        depth_later.depth_test_enable = true;
        depth_later.depth_compare_op = 6; // GEQUAL
        depth_later.depth_clear_value = 0.0f;
        depth_later.depth_read_base = depth_later.depth_write_base = 0x33330000;
        depth_later.stencil_read_base = depth_later.stencil_write_base = 0x44440000;

        prosper::test::BackendDraw s; s.vs = vs; s.fs = red; s.ps = &stencil_first; s.vcount = 3;
        prosper::test::BackendDraw d; d.vs = vs; d.fs = green; d.ps = &depth_later; d.vcount = 3;
        (void)prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> initialized =
            prosper::test::render_draws_rgba({d}, W, H, nullptr, nullptr, true);
        const uint8_t* c = center(initialized);
        CHECK(c && c[1] > 0xC0 && c[0] < 0x40,
              "stencil-only initialization does not poison a later reverse-Z depth plane");
    }

    // Programming DB_DEPTH_CLEAR only updates clear state; it does not clear a depth allocation until
    // DB_RENDER_CONTROL enables the clear. Treating a stale programmed word as the implicit contents of
    // a newly-created host image can reject every fragment. Astro Bot exposes this with 0x0437077f --
    // its packed 1919x1079 surface maxima, interpreted as the tiny float 2.15e-36 -- under LEQUAL.
    {
        std::vector<uint32_t> mid_depth_vs = ref_vs_with_depth(vs, 0x3f000000u); // 0.5f
        CHECK(!mid_depth_vs.empty(), "fullscreen VS can expose stale depth-clear initialization");

        ResolvedPipelineState stale = opaque;
        stale.depth_test_enable = true;
        stale.depth_compare_op = 3; // LEQUAL
        stale.has_depth_clear = true;
        uint32_t packed_extent = 0x0437077fu;
        std::memcpy(&stale.depth_clear_value, &packed_extent, sizeof(packed_extent));
        stale.depth_read_base = stale.depth_write_base = 0x77770000;

        prosper::test::BackendDraw s;
        s.vs = mid_depth_vs; s.fs = green; s.ps = &stale; s.vcount = 3;
        std::vector<uint8_t> implicit =
            prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        const uint8_t* ic = center(implicit);
        CHECK(ic && ic[1] > 0xC0 && ic[0] < 0x40,
              "stale DB_DEPTH_CLEAR does not initialize an uncleared LEQUAL surface");

        ResolvedPipelineState explicit_clear = stale;
        explicit_clear.depth_clear_enable = true;
        explicit_clear.depth_read_base = explicit_clear.depth_write_base = 0x88880000;
        prosper::test::BackendDraw e = s; e.ps = &explicit_clear;
        std::vector<uint8_t> explicitly_rejected =
            prosper::test::render_draws_rgba({e}, W, H, nullptr, nullptr, true);
        const uint8_t* ec = center(explicitly_rejected);
        CHECK(ec && ec[1] < 0x80,
              "explicit DB_RENDER_CONTROL clear still consumes the programmed depth value");
    }

    // UE4 can emit its reverse-Z depth prepass and EQUAL base pass from different shaders. Their
    // translated clip-space positions may differ by one float ULP. On the guest this pair shades;
    // strict Vulkan EQUAL rejects it. Relax only a read-only EQUAL on an already-valid, explicitly
    // reverse-Z guest surface to GEQUAL, which continues rejecting fragments behind the prepass.
    {
        std::vector<uint32_t> depth_vs = ref_vs_with_depth(vs, 0x3f000000u); // 0.5f
        std::vector<uint32_t> drift_vs = ref_vs_with_depth(vs, 0x3f000001u); // next float above 0.5f
        CHECK(!depth_vs.empty() && !drift_vs.empty(), "fullscreen VS depth literal is patchable");

        ResolvedPipelineState writer = opaque;
        writer.color_write_mask = 0;
        writer.depth_test_enable = true;
        writer.depth_write_enable = true;
        writer.depth_compare_op = 7; // ALWAYS
        writer.has_depth_clear = true;
        writer.depth_clear_value = 0.0f;
        writer.depth_read_base = writer.depth_write_base = 0x55550000;

        ResolvedPipelineState reader = opaque;
        reader.depth_test_enable = true;
        reader.depth_compare_op = 2; // EQUAL
        reader.has_depth_clear = true;
        reader.depth_clear_value = 0.0f;
        reader.depth_read_base = reader.depth_write_base = 0x55550000;

        prosper::test::BackendDraw w; w.vs = depth_vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        prosper::test::BackendDraw r; r.vs = drift_vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        (void)prosper::test::render_draws_rgba({w}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> compatible =
            prosper::test::render_draws_rgba({r}, W, H, nullptr, nullptr, true);
        const uint8_t* rc = center(compatible);
        CHECK(rc && rc[1] > 0xC0 && rc[0] < 0x40,
              "persistent reverse-Z EQUAL tolerates one-ULP translated shader drift");

        ResolvedPipelineState fresh = reader;
        fresh.depth_read_base = fresh.depth_write_base = 0x66660000;
        prosper::test::BackendDraw f = r; f.ps = &fresh;
        std::vector<uint8_t> uninitialized =
            prosper::test::render_draws_rgba({f}, W, H, nullptr, nullptr, true);
        const uint8_t* fc = center(uninitialized);
        CHECK(fc && fc[1] < 0x80,
              "reverse-Z EQUAL stays strict before the guest depth surface is populated");

        ResolvedPipelineState standard_z = reader;
        standard_z.depth_clear_value = 1.0f;
        prosper::test::BackendDraw s = r; s.ps = &standard_z;
        std::vector<uint8_t> strict =
            prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        const uint8_t* sc = center(strict);
        CHECK(sc && sc[1] < 0x80, "standard-Z EQUAL remains exact");
    }

    // #1351: the backend must APPLY the resolved depth bias, not merely decode it. Precedent for
    // requiring execution coverage: #456's cull/front-face landed decode-only and its reversed
    // VkFrontFace survived until #534. Contract under test: a LESS draw at exactly the stored
    // depth is rejected without bias, and a negative constant bias (Vulkan adds
    // constant * 2^(e-23) on D32F) nudges its fragments nearer so the same draw passes.
    {
        std::vector<uint32_t> z_vs = ref_vs_with_depth(vs, 0x3f000000u); // 0.5f
        CHECK(!z_vs.empty(), "bias test fullscreen VS at z=0.5 available");

        ResolvedPipelineState bias_writer = opaque;
        bias_writer.color_write_mask = 0;
        bias_writer.depth_test_enable = true;
        bias_writer.depth_write_enable = true;
        bias_writer.depth_compare_op = 7; // ALWAYS
        bias_writer.depth_read_base = bias_writer.depth_write_base = 0x77660000;

        ResolvedPipelineState less_reader = opaque;
        less_reader.depth_test_enable = true;
        less_reader.depth_compare_op = 1; // LESS
        less_reader.depth_read_base = less_reader.depth_write_base = 0x77660000;

        prosper::test::BackendDraw bw; bw.vs = z_vs; bw.fs = red; bw.ps = &bias_writer; bw.vcount = 3;
        prosper::test::BackendDraw br; br.vs = z_vs; br.fs = green; br.ps = &less_reader; br.vcount = 3;
        (void)prosper::test::render_draws_rgba({bw}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> unbiased =
            prosper::test::render_draws_rgba({br}, W, H, nullptr, nullptr, true);
        const uint8_t* uc = center(unbiased);
        CHECK(uc && uc[1] < 0x80, "LESS at exactly the stored depth is rejected without bias");

        ResolvedPipelineState biased_reader = less_reader;
        biased_reader.depth_bias_enable = 1u;
        biased_reader.depth_bias_constant = -64.0f;   // ~-3.8e-6 at e(0.5) on D32F — decisive, tiny
        biased_reader.depth_bias_slope = 0.0f;
        biased_reader.depth_bias_clamp = 0.0f;
        prosper::test::BackendDraw bb = br; bb.ps = &biased_reader;
        std::vector<uint8_t> biased =
            prosper::test::render_draws_rgba({bb}, W, H, nullptr, nullptr, true);
        const uint8_t* bc = center(biased);
        CHECK(bc && bc[1] > 0xC0 && bc[0] < 0x40,
              "a negative constant depth bias flips the same LESS draw to passing (#1351)");
    }

    // #1355: STENCIL_CLEAR_ENABLE is the stencil twin of the #1352 depth rule — the bit
    // substitutes the VALUE of stencil writes and only acts through the enabled stencil write
    // path. A writes-disabled clear rect between a stencil writer and an EQUAL reader must not
    // destroy the written plane.
    {
        CHECK(!prosper::test::stencil_clear_effective(true, false, 0xff, 0xff),
              "a stencil clear with STENCIL_ENABLE off is inert (#1355)");
        CHECK(!prosper::test::stencil_clear_effective(true, true, 0, 0),
              "a stencil clear with a zero write mask is inert");
        CHECK(prosper::test::stencil_clear_effective(true, true, 0xff, 0),
              "a real stencil clear shape (enable + mask) stays effective");

        ResolvedPipelineState swriter = opaque;
        swriter.color_write_mask = 0;
        swriter.stencil_enable = true;
        swriter.stencil_compare_op[0] = swriter.stencil_compare_op[1] = 7; // ALWAYS
        swriter.stencil_pass_op[0] = swriter.stencil_pass_op[1] = 2;       // REPLACE
        swriter.stencil_op_val[0] = swriter.stencil_op_val[1] = 2;
        swriter.stencil_write_mask[0] = swriter.stencil_write_mask[1] = 0xff;
        swriter.stencil_read_base = swriter.stencil_write_base = 0x88770000;

        ResolvedPipelineState inert_sclear{};
        inert_sclear.topology = 3; inert_sclear.color_write_mask = 0;
        inert_sclear.stencil_clear_enable = true;
        inert_sclear.stencil_clear_value = 7;
        inert_sclear.stencil_enable = false;   // write path fully disabled
        inert_sclear.stencil_read_base = inert_sclear.stencil_write_base = 0x88770000;

        ResolvedPipelineState sreader = opaque;
        sreader.stencil_enable = true;
        sreader.stencil_compare_op[0] = sreader.stencil_compare_op[1] = 2; // EQUAL
        sreader.stencil_ref[0] = sreader.stencil_ref[1] = 2;
        sreader.stencil_compare_mask[0] = sreader.stencil_compare_mask[1] = 0xff;
        sreader.stencil_write_mask[0] = sreader.stencil_write_mask[1] = 0;
        sreader.stencil_read_base = sreader.stencil_write_base = 0x88770000;

        prosper::test::BackendDraw sw; sw.vs = vs; sw.fs = red; sw.ps = &swriter; sw.vcount = 3;
        prosper::test::BackendDraw scl; scl.vs = vs; scl.fs = red; scl.ps = &inert_sclear; scl.vcount = 3;
        prosper::test::BackendDraw sr; sr.vs = vs; sr.fs = green; sr.ps = &sreader; sr.vcount = 3;
        (void)prosper::test::render_draws_rgba({sw}, W, H, nullptr, nullptr, true);
        (void)prosper::test::render_draws_rgba({scl}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> preserved =
            prosper::test::render_draws_rgba({sr}, W, H, nullptr, nullptr, true);
        const uint8_t* pc = center(preserved);
        CHECK(pc && pc[1] > 0xC0 && pc[0] < 0x40,
              "EQUAL reader still sees the written stencil after an inert clear rect (#1355)");

        // #1361: an inert clear against a NEVER-written identity must not phantom-initialize the
        // plane to its clear value. A partial gate (clear-site only) would let the forced stencil
        // pass create the image, seed it from the latched clear value, and mark it valid — a later
        // EQUAL-7 reader would then wrongly pass. With the full gate the reader's fresh plane
        // defaults to 0 and EQUAL-7 rejects everywhere.
        ResolvedPipelineState inert_fresh = inert_sclear;
        inert_fresh.stencil_read_base = inert_fresh.stencil_write_base = 0x88780000;
        ResolvedPipelineState seven_reader = sreader;
        seven_reader.stencil_ref[0] = seven_reader.stencil_ref[1] = 7;
        seven_reader.stencil_read_base = seven_reader.stencil_write_base = 0x88780000;
        prosper::test::BackendDraw fcl; fcl.vs = vs; fcl.fs = red; fcl.ps = &inert_fresh; fcl.vcount = 3;
        prosper::test::BackendDraw fr7; fr7.vs = vs; fr7.fs = green; fr7.ps = &seven_reader; fr7.vcount = 3;
        (void)prosper::test::render_draws_rgba({fcl}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> untouched =
            prosper::test::render_draws_rgba({fr7}, W, H, nullptr, nullptr, true);
        const uint8_t* uc7 = center(untouched);
        CHECK(uc7 && uc7[1] < 0x80,
              "an inert clear cannot phantom-initialize a never-written plane (#1361)");
    }

    // #1334: copy_persistent_color_target — the MSAA-resolve GPU copy. Render red into
    // persistent identity A (GPU-resident, no readback), copy A -> B on the device, and read B
    // back: the destination-keyed image must hold the source pixels (previously the resolve only
    // shared CPU pixels while the destination image stayed stale-but-valid; after a #780 CPU-copy
    // discard, consumers imported black — Blue Prince's compute tonemap, #1287/#1381).
    {
        prosper::test::BackendDraw rd; rd.vs = vs; rd.fs = red; rd.ps = &opaque; rd.vcount = 3;
        prosper::test::BackendColorTarget src_target{0x1334000000000001ull, false, false};
        const std::vector<uint8_t> gpu_only = prosper::test::render_draws_rgba(
            {rd}, W, H, nullptr, nullptr, false, &src_target);
        CHECK(gpu_only.empty(), "resolve-copy source rendered GPU-resident (no readback)");
        std::string copy_err;
        CHECK(prosper::test::copy_persistent_color_target(
                  0x1334000000000001ull, 0x1334000000000002ull, W, H,
                  VK_FORMAT_R8G8B8A8_UNORM, copy_err),
              "GPU-side persistent color copy succeeds");
        std::vector<uint8_t> copied; std::string read_err;
        CHECK(prosper::test::readback_persistent_color_target(
                  0x1334000000000002ull, W, H, VK_FORMAT_R8G8B8A8_UNORM, copied, read_err) &&
                  copied.size() == (size_t)W * H * 4,
              "copy destination reads back");
        const uint8_t* cc = copied.size() == (size_t)W * H * 4
            ? &copied[((size_t)(H / 2) * W + W / 2) * 4] : nullptr;
        CHECK(cc && cc[0] > 0xC0 && cc[1] < 0x40,
              "destination holds the source pixels (RED) after the copy (#1334)");
        std::string missing_err;
        CHECK(!prosper::test::copy_persistent_color_target(
                  0x1334000000000003ull, 0x1334000000000004ull, W, H,
                  VK_FORMAT_R8G8B8A8_UNORM, missing_err),
              "a missing source declines the copy");

        // Verify that copy_persistent_color_target evicts older entries when the target cache
        // reaches capacity, rather than failing and dropping the destination to CPU-only pixels.
        const size_t count_limit = prosper::test::persistent_color_target_count_limit();
        const uint64_t evict_dummy_base = 0x1334000000001000ull;
        size_t dummy_count = 0;
        while (prosper::test::persistent_color_target_cache().size() < count_limit) {
            const uint64_t id = evict_dummy_base + dummy_count++;
            prosper::test::PersistentColorTargetKey key{id, W, H, VK_FORMAT_R8G8B8A8_UNORM};
            auto& entry = prosper::test::persistent_color_target_cache()[key];
            entry.last_use = 1; // old generation eligible for eviction
            entry.valid = false;
        }
        std::string evict_copy_err;
        CHECK(prosper::test::copy_persistent_color_target(
                  0x1334000000000001ull, 0x1334000000000005ull, W, H,
                  VK_FORMAT_R8G8B8A8_UNORM, evict_copy_err),
              "GPU-side persistent color copy succeeds at capacity limit by evicting older target");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
