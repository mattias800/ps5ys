// test_pipeline_render — the Vulkan-gated half of pipeline realization: build a real VkPipeline whose
// fixed-function state comes from resolve_pipeline_state(RenderState) and prove the resolved state
// actually drives rendering. The chain under test is end-to-end: RDNA2 registers (a GpuState) ->
// extract_render_state -> resolve_pipeline_state -> VkGraphicsPipelineCreateInfo -> pixels.
//
// Deterministic proof via the color write mask: the embedded shader draws a solid-RED triangle over a
// BLUE clear. With CB_TARGET_MASK=0xF the triangle is visible (center pixel red); with CB_TARGET_MASK=0
// no channels are written, so the triangle is invisible (center stays blue). Both masks are produced
// by resolve_pipeline_state from a real register value — not hand-set — so this exercises the whole path.
#include "gpu/state/render_state.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <vulkan/vulkan.h>

// Pixels cannot reveal whether the driver received reusable compilation data. Observe the real
// API boundary while still executing every pipeline creation on the device.
static unsigned driver_cache_calls = 0;
static bool driver_cache_missing = false;
static VkResult observe_graphics_pipeline_cache(
    VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkGraphicsPipelineCreateInfo* info, const VkAllocationCallbacks* allocator,
    VkPipeline* pipelines) {
    ++driver_cache_calls;
    driver_cache_missing |= cache == VK_NULL_HANDLE;
    return vkCreateGraphicsPipelines(device, cache, count, info, allocator, pipelines);
}
#define vkCreateGraphicsPipelines observe_graphics_pipeline_cache
#include "fixtures/render_runner.h"
#undef vkCreateGraphicsPipelines
#include "fixtures/spirv_triangle.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
using prosper::test::render_triangle_rgba;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Build a minimal GpuState whose context registers describe a triangle-list draw with the given
// per-MRT0 color write mask, then run it through the real extract -> resolve path.
static ResolvedPipelineState resolve_for(uint32_t target_mask) {
    namespace P = prosper::agc::Pm4;
    GpuState st;
    // VGT_PRIMITIVE_TYPE.PRIM_TYPE = 4 (triangle list). Field is at bit 0 for PRIM_TYPE.
    st.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    st.cx[P::CB_TARGET_MASK]     = target_mask;
    return resolve_pipeline_state(extract_render_state(st));
}

int main() {
    printf("== test_pipeline_render ==\n");
    const uint32_t W = 64, H = 64;
    const size_t center = ((size_t)(H/2) * W + (W/2)) * 4;

    std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv)/4);
    std::vector<uint32_t> frag(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv)/4);

    // Full write mask (CB_TARGET_MASK=0xF): the red triangle must be visible at the center.
    ResolvedPipelineState full = resolve_for(0xF);
    CHECK(full.topology == 3, "prim_type 4 resolved to VK topology TRIANGLE_LIST (3)");
    CHECK(full.color_write_mask == 0xF, "CB_TARGET_MASK 0xF resolved to RGBA write mask");
    std::vector<uint8_t> img = render_triangle_rgba(vert, frag, W, H, &full);
    CHECK(img.size() == (size_t)W*H*4, "rendered with resolved pipeline state (full mask)");
    if (img.size() == (size_t)W*H*4) {
        uint8_t r = img[center], g = img[center+1], b = img[center+2];
        printf("  center (full mask)   = (%u,%u,%u)\n", r, g, b);
        CHECK(r > 128 && b < 128, "full mask: center pixel is the red triangle (write mask honored)");
    }

    // Zero write mask (CB_TARGET_MASK=0): no channels written, so the triangle is invisible — the
    // center keeps the blue clear color. This is the same pipeline except for the resolved write mask.
    ResolvedPipelineState none = resolve_for(0x0);
    CHECK(none.color_write_mask == 0x0, "CB_TARGET_MASK 0 resolved to empty write mask");
    std::vector<uint8_t> img0 = render_triangle_rgba(vert, frag, W, H, &none);
    CHECK(img0.size() == (size_t)W*H*4, "rendered with resolved pipeline state (zero mask)");
    if (img0.size() == (size_t)W*H*4) {
        uint8_t r = img0[center], g = img0[center+1], b = img0[center+2];
        printf("  center (zero mask)   = (%u,%u,%u)\n", r, g, b);
        CHECK(b > 128 && r < 128, "zero mask: center pixel stays blue clear (write mask honored)");
    }

    // Blend honored: additive blend (src=One, dst=One, Add) of the red triangle over the blue clear
    // must produce magenta at the center. The blend fields come from resolve_pipeline_state (RenderState
    // blend enums -> Vk blend enums), so this proves the resolved blend state drives real output.
    RenderState rsb;
    rsb.prim_type      = 4;       // triangle list
    rsb.cb_target_mask = 0xF;     // write RGBA
    rsb.blend_enable   = true;
    rsb.color_src_blend = 0x01;   // One  -> VK ONE (1)
    rsb.color_dst_blend = 0x01;   // One  -> VK ONE (1)
    rsb.color_comb_fcn  = 0;      // Add  -> VK_BLEND_OP_ADD (0)
    ResolvedPipelineState blend = resolve_pipeline_state(rsb);
    CHECK(blend.blend_enable && blend.src_color_blend_factor == 1 && blend.dst_color_blend_factor == 1 &&
          blend.color_blend_op == 0, "resolved additive blend state (One/One/Add)");
    std::vector<uint8_t> imgb = render_triangle_rgba(vert, frag, W, H, &blend);
    if (imgb.size() == (size_t)W*H*4) {
        uint8_t r = imgb[center], g = imgb[center+1], b = imgb[center+2];
        printf("  center (additive)    = (%u,%u,%u)\n", r, g, b);
        CHECK(r > 128 && g < 128 && b > 128, "additive blend: red over blue -> magenta (blend state honored)");
    } else { CHECK(false, "additive blend render produced a frame"); }

    // AMD's obsolete BOTH_* source factors program a complementary source/destination pair and
    // override the destination field. Patch the fragment shader's sole 1.0 alpha constant to 0.25;
    // red-over-blue then gives (0.25, 0, 0.75) for BOTH_SRC_ALPHA and the inverse for BOTH_INV.
    {
        std::vector<uint32_t> quarter_alpha_frag = frag;
        bool patched_alpha = false;
        for (uint32_t& word : quarter_alpha_frag) {
            if (word == 0x3f800000u) {
                word = 0x3e800000u;
                patched_alpha = true;
                break;
            }
        }
        CHECK(patched_alpha, "patched triangle fragment alpha from 1.0 to 0.25");

        auto resolve_both = [](uint32_t source_factor) {
            namespace Pm4 = prosper::agc::Pm4;
            GpuState st;
            st.uc[Pm4::VGT_PRIMITIVE_TYPE] = 4;
            st.cx[Pm4::CB_TARGET_MASK] = 0xFu;
            st.cx[Pm4::CB_BLEND0_CONTROL] =
                (source_factor << Pm4::CB_BLEND0_CONTROL_COLOR_SRCBLEND_SHIFT) |
                (0u << Pm4::CB_BLEND0_CONTROL_COLOR_DESTBLEND_SHIFT) |
                (1u << Pm4::CB_BLEND0_CONTROL_ENABLE_SHIFT);
            return resolve_pipeline_state(extract_render_state(st));
        };

        ResolvedPipelineState both_src = resolve_both(0x0bu);
        CHECK(both_src.blend_enable && both_src.src_color_blend_factor == 6u &&
                  both_src.dst_color_blend_factor == 7u,
              "raw BOTH_SRC_ALPHA overrides ZERO dst with SRC_ALPHA/ONE_MINUS_SRC_ALPHA");
        std::vector<uint8_t> img_both_src =
            render_triangle_rgba(vert, quarter_alpha_frag, W, H, &both_src);
        if (img_both_src.size() == (size_t)W * H * 4) {
            uint8_t r = img_both_src[center], g = img_both_src[center + 1], b = img_both_src[center + 2];
            printf("  center (both src)    = (%u,%u,%u)\n", r, g, b);
            CHECK(r > 40 && r < 90 && g < 16 && b > 165 && b < 215,
                  "BOTH_SRC_ALPHA: quarter-alpha red over blue -> quarter red, three-quarter blue");
        } else { CHECK(false, "BOTH_SRC_ALPHA render produced a frame"); }

        ResolvedPipelineState both_inv = resolve_both(0x0cu);
        CHECK(both_inv.blend_enable && both_inv.src_color_blend_factor == 7u &&
                  both_inv.dst_color_blend_factor == 6u,
              "raw BOTH_INV_SRC_ALPHA overrides ZERO dst with inverse paired factors");
        std::vector<uint8_t> img_both_inv =
            render_triangle_rgba(vert, quarter_alpha_frag, W, H, &both_inv);
        if (img_both_inv.size() == (size_t)W * H * 4) {
            uint8_t r = img_both_inv[center], g = img_both_inv[center + 1], b = img_both_inv[center + 2];
            printf("  center (both inv)    = (%u,%u,%u)\n", r, g, b);
            CHECK(r > 165 && r < 215 && g < 16 && b > 40 && b < 90,
                  "BOTH_INV_SRC_ALPHA: quarter-alpha red over blue -> three-quarter red, quarter blue");
        } else { CHECK(false, "BOTH_INV_SRC_ALPHA render produced a frame"); }
    }

    // Logic op honored: AMD ROP3 XOR (0x66) combines the red fragment with the blue destination,
    // producing magenta. The operation is decoded from real CB_COLOR_CONTROL state and reaches the
    // VkPipelineColorBlendStateCreateInfo, proving the full register-to-pixel path.
    {
        namespace P = prosper::agc::Pm4;
        GpuState st;
        st.uc[P::VGT_PRIMITIVE_TYPE] = 4;
        st.cx[P::CB_TARGET_MASK] = 0xF;
        st.cx[P::CB_COLOR_CONTROL] =
            (P::CB_COLOR_CONTROL_MODE_NORMAL << P::CB_COLOR_CONTROL_MODE_SHIFT) |
            (0x66u << P::CB_COLOR_CONTROL_ROP3_SHIFT);
        ResolvedPipelineState logic = resolve_pipeline_state(extract_render_state(st));
        CHECK(logic.logic_op_enable && logic.logic_op == VK_LOGIC_OP_XOR,
              "CB_COLOR_CONTROL ROP3 XOR resolves to VK_LOGIC_OP_XOR");
        st.cx[P::CB_BLEND0_CONTROL] =
            (1u << P::CB_BLEND0_CONTROL_DISABLE_ROP3_SHIFT);
        CHECK(!resolve_pipeline_state(extract_render_state(st)).logic_op_enable,
              "CB_BLEND0_CONTROL.DISABLE_ROP3 suppresses XOR through the raw register path");
        st.cx.erase(P::CB_BLEND0_CONTROL);
        if (prosper::test::render_vk_ctx().logic_op_enabled) {
            std::vector<uint8_t> imgl = render_triangle_rgba(vert, frag, W, H, &logic);
            if (imgl.size() == (size_t)W*H*4) {
                uint8_t r = imgl[center], g = imgl[center+1], b = imgl[center+2];
                printf("  center (logic XOR)   = (%u,%u,%u)\n", r, g, b);
                CHECK(r > 128 && g < 128 && b > 128,
                      "logic XOR: red source XOR blue destination -> magenta");
            } else { CHECK(false, "logic XOR render produced a frame"); }
        } else {
            printf("  [skip] Vulkan device does not advertise logicOp\n");
        }
    }

    // Depth honored: the harness clears depth to 0.5 and the triangle's fragments are at z=0.0.
    // ZFUNC=LESS (1): 0.0 < 0.5 -> passes -> red visible. ZFUNC=GREATER (4): 0.0 > 0.5 -> fails ->
    // triangle rejected, center stays blue. Both compare ops come from resolve_pipeline_state.
    RenderState rsz; rsz.prim_type = 4; rsz.cb_target_mask = 0xF;
    rsz.z_enable = true; rsz.z_write_enable = true;
    rsz.zfunc = 1;   // LESS
    ResolvedPipelineState zpass = resolve_pipeline_state(rsz);
    CHECK(zpass.depth_test_enable && zpass.depth_compare_op == 1, "resolved depth test LESS");
    std::vector<uint8_t> imgzp = render_triangle_rgba(vert, frag, W, H, &zpass);
    if (imgzp.size() == (size_t)W*H*4) {
        uint8_t r = imgzp[center], b = imgzp[center+2];
        printf("  center (z LESS)      = (%u,%u,%u)\n", r, imgzp[center+1], b);
        CHECK(r > 128 && b < 128, "depth LESS: fragment z=0 < clear 0.5 passes -> red (depth honored)");
    } else { CHECK(false, "depth LESS render produced a frame"); }

    rsz.zfunc = 4;   // GREATER
    ResolvedPipelineState zfail = resolve_pipeline_state(rsz);
    CHECK(zfail.depth_compare_op == 4, "resolved depth test GREATER");
    std::vector<uint8_t> imgzf = render_triangle_rgba(vert, frag, W, H, &zfail);
    if (imgzf.size() == (size_t)W*H*4) {
        uint8_t r = imgzf[center], b = imgzf[center+2];
        printf("  center (z GREATER)   = (%u,%u,%u)\n", r, imgzf[center+1], b);
        CHECK(b > 128 && r < 128, "depth GREATER: fragment z=0 > clear 0.5 fails -> stays blue (depth honored)");
    } else { CHECK(false, "depth GREATER render produced a frame"); }

    // Viewport honored: a guest PA_CL_VPORT with NEGATIVE yscale (GNM's +Y-up NDC) resolves to a
    // negative-height Vulkan viewport, which must render the scene vertically MIRRORED relative to
    // the default full-target viewport. Proof: the flipped render matches the default render mirrored
    // row-by-row (small tolerance for edge fill-rule differences), and the triangle is vertically
    // asymmetric so the comparison cannot pass vacuously.
    {
        namespace P = prosper::agc::Pm4;
        auto f2u = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        GpuState st;
        st.uc[P::VGT_PRIMITIVE_TYPE]   = 4;                    // triangle list
        st.cx[P::CB_TARGET_MASK]       = 0xF;
        st.cx[P::PA_CL_VPORT_XSCALE]   = f2u((float)W / 2);    // full-target X
        st.cx[P::PA_CL_VPORT_XOFFSET]  = f2u((float)W / 2);
        st.cx[P::PA_CL_VPORT_YSCALE]   = f2u(-(float)H / 2);   // +Y-up NDC -> Vulkan flip
        st.cx[P::PA_CL_VPORT_YOFFSET]  = f2u((float)H / 2);
        st.cx[P::PA_CL_VPORT_ZSCALE]   = f2u(1.0f);
        st.cx[P::PA_CL_VPORT_ZOFFSET]  = f2u(0.0f);
        ResolvedPipelineState flip = resolve_pipeline_state(extract_render_state(st));
        CHECK(flip.has_viewport, "PA_CL_VPORT registers resolved to a guest viewport");
        CHECK(flip.viewport_h == -(float)H && flip.viewport_y == (float)H,
              "negative yscale resolved to a negative-height (flipped) Vulkan viewport");
        std::vector<uint8_t> imgf = render_triangle_rgba(vert, frag, W, H, &flip);
        CHECK(imgf.size() == (size_t)W*H*4, "rendered with flipped guest viewport");
        if (imgf.size() == (size_t)W*H*4) {
            auto redat = [&](const std::vector<uint8_t>& im, uint32_t x, uint32_t y) {
                const uint8_t* p = &im[((size_t)y * W + x) * 4]; return p[0] > 128 && p[2] < 128; };
            // The default render (`img`, full mask, from above) mirrored must match the flipped render.
            size_t mismatch = 0, self_mismatch = 0;
            for (uint32_t y = 0; y < H; y++) for (uint32_t x = 0; x < W; x++) {
                if (redat(imgf, x, y) != redat(img, x, H - 1 - y)) mismatch++;
                if (redat(img,  x, y) != redat(img, x, H - 1 - y)) self_mismatch++;
            }
            printf("  flip mismatch=%zu asymmetry=%zu (of %u px)\n", mismatch, self_mismatch, W * H);
            CHECK(mismatch < (size_t)W * H / 50, "flipped viewport == default render mirrored vertically");
            CHECK(self_mismatch > (size_t)W * H / 50, "triangle is vertically asymmetric (test not vacuous)");
        }

        // AMD FACE and VkFrontFace use the same numeric convention (0=CCW, 1=CW). The negative-height
        // viewport mirrors the image without requiring a second semantic inversion here: this clip-CCW
        // triangle remains front-facing for FACE=0. This catches the reversed VkFrontFace constants that
        // hid Messenger's foreground while exercising the exact flipped-viewport pipeline combination.
        st.cx[P::PA_SU_SC_MODE_CNTL] = (1u << 1);   // CULL_BACK, FACE=0 (CCW front)
        ResolvedPipelineState ccw_front = resolve_pipeline_state(extract_render_state(st));
        CHECK(ccw_front.cull_mode == 2u && ccw_front.front_face == 0u,
              "CULL_BACK + FACE=0 resolves to Vulkan BACK + CCW");
        std::vector<uint8_t> img_ccw = render_triangle_rgba(vert, frag, W, H, &ccw_front);
        if (img_ccw.size() == (size_t)W*H*4) {
            const uint8_t* p = &img_ccw[center];
            CHECK(p[0] > 128 && p[2] < 128,
                  "negative viewport + FACE=0 retains the CCW triangle");
        } else { CHECK(false, "FACE=0 cull render produced a frame"); }

        st.cx[P::PA_SU_SC_MODE_CNTL] = (1u << 1) | (1u << 2); // CULL_BACK, FACE=1 (CW front)
        ResolvedPipelineState cw_front = resolve_pipeline_state(extract_render_state(st));
        CHECK(cw_front.cull_mode == 2u && cw_front.front_face == 1u,
              "CULL_BACK + FACE=1 resolves to Vulkan BACK + CW");
        std::vector<uint8_t> img_cw = render_triangle_rgba(vert, frag, W, H, &cw_front);
        if (img_cw.size() == (size_t)W*H*4) {
            const uint8_t* p = &img_cw[center];
            CHECK(p[2] > 128 && p[0] < 128,
                  "negative viewport + FACE=1 culls the CCW triangle");
        } else { CHECK(false, "FACE=1 cull render produced a frame"); }
    }

    CHECK(driver_cache_calls > 1 && !driver_cache_missing,
          "distinct graphics pipelines receive the device's driver compilation cache");
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
