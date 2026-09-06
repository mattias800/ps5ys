// test_texture_sample_render — a recompiled PIXEL shader that does MIMG image_sample reads a real
// bound texture and the sampled texel reaches the framebuffer.
//
// This is stage 4 of the resource-binding plan (textures). The pixel shader samples a 2x2 RGBA texture
// with four distinct texels; a NEAREST/normalized sampler maps coord 0.25 -> texel column/row 0 and
// coord 0.75 -> column/row 1. We render the fullscreen triangle twice with two coords and assert the
// framebuffer shows the corresponding texel color — proving MIMG decode, OpImageSampleImplicitLod,
// coordinate assembly (VADDR VGPRs -> u,v), the combined-image-sampler binding, and dmask->VDATA all
// work end to end. A broken path would leave the frame blue (clear) or the wrong texel.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/render_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_texture_sample_render ==\n");
    if (std::getenv("PROSPER_BACKEND_TEXTURE_CACHE_MB")) {
        std::fprintf(stderr, "texture sample test requires PROSPER_BACKEND_TEXTURE_CACHE_MB absent\n");
        return 2;
    }
    const uint32_t W = 64, H = 64;
    const auto center_red_at = [](const std::vector<uint8_t>& pixels,
                                  uint32_t width, uint32_t height) -> uint8_t {
        return pixels.size() == static_cast<size_t>(width) * height * 4
            ? pixels[(static_cast<size_t>(height / 2) * width + width / 2) * 4] : 0;
    };
    const auto center_red = [=](const std::vector<uint8_t>& pixels) -> uint8_t {
        return center_red_at(pixels, W, H);
    };

    using prosper::test::BackendSubmissionState;
    using prosper::test::backend_submission_state;
    CHECK(backend_submission_state(false, false) == BackendSubmissionState::NotSubmitted &&
              backend_submission_state(true, true) == BackendSubmissionState::Complete &&
              backend_submission_state(true, false) == BackendSubmissionState::Pending,
          "submission cleanup distinguishes never-submitted, completed, and pending work");
    CHECK(prosper::test::backend_timestamp_delta(0xfdu, 0x03u, 8) == 6u &&
              prosper::test::backend_timestamp_delta(11u, 29u, 64) == 18u &&
              prosper::test::backend_timestamp_delta(11u, 29u, 0) == 0u,
          "device timestamp deltas handle valid-bit wrap and unsupported queues");

    {
        prosper::test::FrameResource bgra_target;
        bgra_target.swizzle[0] = 6u;
        bgra_target.swizzle[1] = 5u;
        bgra_target.swizzle[2] = 4u;
        bgra_target.swizzle[3] = 7u;
        bgra_target.render_target_guest_format = VK_FORMAT_B8G8R8A8_UNORM;
        const auto canonical =
            prosper::test::backend_sampled_component_swizzle(bgra_target);
        CHECK((canonical == std::array<uint32_t, 4>{4u, 5u, 6u, 7u}),
              "BGRA renderer target composes DST_SEL into canonical RGBA component order");

        bgra_target.render_target_guest_format = VK_FORMAT_R8G8B8A8_UNORM;
        const auto semantic =
            prosper::test::backend_sampled_component_swizzle(bgra_target);
        CHECK((semantic == std::array<uint32_t, 4>{6u, 5u, 4u, 7u}),
              "RGBA renderer target retains a semantic BGR component permutation");

        bgra_target.render_target_guest_format = VK_FORMAT_B8G8R8A8_UNORM;
        bgra_target.swizzle[0] = 4u;
        bgra_target.swizzle[1] = 4u;
        bgra_target.swizzle[2] = 4u;
        bgra_target.swizzle[3] = 1u;
        const auto replicated =
            prosper::test::backend_sampled_component_swizzle(bgra_target);
        CHECK((replicated == std::array<uint32_t, 4>{6u, 6u, 6u, 1u}),
              "BGRA target composition preserves replication and constant selectors");
    }

    {
        bool speculative_state_valid = true;
        bool retained_resources_released = false;
        prosper::test::BackendSubmissionBatch discarded_batch;
        discarded_batch.enqueue(VK_NULL_HANDLE);
        discarded_batch.add_failure_cleanup([&]() { speculative_state_valid = false; });
        discarded_batch.add_cleanup([&]() { retained_resources_released = true; });
        discarded_batch.discard();
        discarded_batch.complete();
        CHECK(!discarded_batch.pending() && !speculative_state_valid && retained_resources_released,
              "discarded submission batch invalidates speculative state before cleanup");
    }

    // Fullscreen-triangle VS (from gl_VertexIndex; no resource table needed).
    const uint32_t vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled fullscreen-triangle VS -> SPIR-V");

    // Pixel shader: u,v = literal coords; image_sample v[0:3], v[0:1], s[8:15], s[16:19] dim:2D dmask:0xf;
    // exp mrt0 v0..v3. The two literal dwords (indices 1 and 3) are the u,v coords — patched per render.
    const uint32_t ps_template[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    // The T# is placed directly in user-data SGPRs (SRSRC base = s8) -> resolved via sgpr_base.
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1; /*2D*/
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt.resources.push_back(t); }

    // 2x2 RGBA8: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
    const uint8_t texels[2*2*4] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    prosper::test::TexDesc td{ /*binding*/4, /*w*/2, /*h*/2, texels };

    auto sample_center = [&](uint32_t u_bits, uint32_t v_bits, uint8_t out_rgb[3]) -> bool {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = u_bits; ps[3] = v_bits;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        out_rgb[0] = c[0]; out_rgb[1] = c[1]; out_rgb[2] = c[2];
        return true;
    };

    // coord 0.25 -> texel index floor(0.25*2)=0; coord 0.75 -> floor(0.75*2)=1.
    const uint32_t C025 = 0x3e800000u /*0.25f*/, C075 = 0x3f400000u /*0.75f*/;
    uint8_t rgb[3];

    CHECK(recompile_fragment(ps_template, sizeof(ps_template)/sizeof(ps_template[0]), nullptr).empty(),
          "image_sample PS is rejected without a resource table");

    bool ok0 = sample_center(C025, C025, rgb);
    printf("  (0.25,0.25) center=(%u,%u,%u)\n", ok0?rgb[0]:0, ok0?rgb[1]:0, ok0?rgb[2]:0);
    CHECK(ok0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "sampling texel (0,0) yields RED");

    bool ok1 = sample_center(C075, C025, rgb);
    printf("  (0.75,0.25) center=(%u,%u,%u)\n", ok1?rgb[0]:0, ok1?rgb[1]:0, ok1?rgb[2]:0);
    CHECK(ok1 && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "sampling texel (1,0) yields GREEN (proves u routing)");

    // #3045: vkCreateImage's result used to be discarded in the texture-upload path and fed
    // straight into vkGetImageMemoryRequirements, so any real allocation failure handed the
    // Vulkan loader a VK_NULL_HANDLE. inject_render_texture_create_failure_once() simulates that
    // exact failure deterministically for the next texture upload -- a real over-large request
    // can't be relied on to fail here, since this box's RADV reports maxImageArrayLayers=8192,
    // well above the backend's own 2048 ceiling (#3043), so it never actually reaches the driver.
    // A fixed backend must skip the draw carrying the unmaterializable texture rather than use
    // the null handle: the pass still renders (blue clear, per this file's header), but the texel
    // that WOULD have been GREEN never gets drawn.
    {
        prosper::test::inject_render_texture_create_failure_once();
        uint8_t frgb[3];
        bool okFail = sample_center(C075, C025, frgb);
        printf("  vkCreateImage-failure (0.75,0.25) center=(%u,%u,%u)\n",
               okFail?frgb[0]:0, okFail?frgb[1]:0, okFail?frgb[2]:0);
        CHECK(okFail && frgb[2] > 0x80 && frgb[0] < 0x40 && frgb[1] < 0x40,
              "vkCreateImage failure skips the draw (blue clear survives) instead of using the "
              "null image handle (#3045)");
    }

    // Dynamic single-channel textures (notably Astro Bot's FMV planes) stay R8 through upload. The
    // component mapping reproduces the live frontend's historical coverage broadcast without a 4x
    // CPU expansion to RGBA8.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C025; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        const uint8_t r8_texels[4] = {32, 96, 160, 224};
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = r8_texels; resource.tw = 2; resource.th = 2;
        resource.texture_format = VK_FORMAT_R8_UNORM;
        resource.swizzle[0] = resource.swizzle[1] =
            resource.swizzle[2] = resource.swizzle[3] = 4;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        const std::vector<uint8_t> pixels =
            prosper::test::render_draws_rgba({draw}, W, H);
        const uint8_t* center = pixels.size() == static_cast<size_t>(W) * H * 4
            ? &pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4] : nullptr;
        CHECK(center && center[0] >= 31 && center[0] <= 33 &&
                         center[1] == center[0] && center[2] == center[0],
              "native R8 sampled upload broadcasts its channel without RGBA expansion");
        CHECK(prosper::test::backend_color_bytes_per_pixel(VK_FORMAT_R8_UNORM) == 1,
              "native R8 upload accounts one byte per texel");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R8G8_UNORM) ==
                  VK_FORMAT_R8G8_UNORM &&
                  prosper::test::backend_color_bytes_per_pixel(VK_FORMAT_R8G8_UNORM) == 2,
              "native RG8 upload retains two channels and accounts two bytes per texel");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R16G16_SFLOAT) ==
                  VK_FORMAT_R16G16_SFLOAT &&
                  prosper::test::backend_color_bytes_per_pixel(
                      VK_FORMAT_R16G16_SFLOAT) == 4 &&
                  prosper::test::backend_image_numeric_class(
                      VK_FORMAT_R16G16_SFLOAT) == SpirvImageNumericClass::Float,
              "native RG16F targets retain two half-float channels and their sampled type");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R16_SFLOAT) ==
                  VK_FORMAT_R16_SFLOAT &&
                  prosper::test::backend_color_bytes_per_pixel(VK_FORMAT_R16_SFLOAT) == 2 &&
                  prosper::test::backend_image_numeric_class(VK_FORMAT_R16_SFLOAT) ==
                      SpirvImageNumericClass::Float,
              "native R16F targets retain one half-float channel and their sampled type");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R32_UINT) == VK_FORMAT_R32_UINT &&
                  prosper::test::backend_color_bytes_per_pixel(VK_FORMAT_R32_UINT) == 4,
              "R32_UINT storage images retain their typed four-byte Vulkan format");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R32G32B32A32_UINT) ==
                  VK_FORMAT_R32G32B32A32_UINT &&
                  prosper::test::backend_color_bytes_per_pixel(
                      VK_FORMAT_R32G32B32A32_UINT) == 16,
              "portable raw-uvec4 storage images retain all four dword channels");
        CHECK(prosper::test::backend_color_format(VK_FORMAT_R32G32B32A32_SFLOAT) ==
                  VK_FORMAT_R32G32B32A32_SFLOAT &&
                  prosper::test::backend_color_bytes_per_pixel(
                      VK_FORMAT_R32G32B32A32_SFLOAT) == 16,
              "RGBA32F color targets retain all four float channels");
    }

    // A uniform DCC clear reaches the sampled image without a full CPU-sized pixel allocation or
    // staging upload. Keep the declared extent larger than one texel so this also covers arbitrary
    // normalized coordinates over the full image.
    {
        std::vector<uint32_t> ps(ps_template,
                                 ps_template + sizeof(ps_template) / sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C075;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.has_uniform_color = true;
        resource.uniform_color = {0.25f, 0.5f, 0.75f, 1.0f};
        resource.tw = 64; resource.th = 32;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        const std::vector<uint8_t> pixels =
            prosper::test::render_draws_rgba({draw}, W, H);
        const uint8_t* center = pixels.size() == static_cast<size_t>(W) * H * 4
            ? &pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4] : nullptr;
        CHECK(center && center[0] >= 63 && center[0] <= 65 &&
                         center[1] >= 127 && center[1] <= 129 &&
                         center[2] >= 190 && center[2] <= 192,
              "uniform texture is initialized by a GPU clear and samples the requested color");
        const auto stats = prosper::test::backend_texture_upload_stats();
        CHECK(stats.unique_uploads == 1 && stats.upload_bytes == 64u * 32u * 4u,
              "uniform GPU clear is accounted without CPU pixel backing");
    }

    // The live renderer commonly submits hundreds of draws that reference the same few decoded
    // textures. The backend must upload identical pixels once per call while preserving the legacy
    // rendered bytes and separate per-descriptor views/samplers.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        prosper::test::FrameResource swizzled_resource = resource;
        swizzled_resource.swizzle[0] = 5;
        swizzled_resource.swizzle[1] = 4;
        prosper::test::BackendDraw swizzled_draw = draw;
        swizzled_draw.R = {swizzled_resource};

        std::vector<uint8_t> same_binding =
            prosper::test::render_draws_rgba({draw, draw}, W, H);
        const auto same_binding_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(!same_binding.empty() &&
                  same_binding_stats.texture_binding_references == 2 &&
                  same_binding_stats.unique_texture_bindings == 1,
              "identical texture view and sampler contracts share one Vulkan binding pair");

        std::vector<uint8_t> shared =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto shared_stats = prosper::test::backend_texture_upload_stats();
        const auto shared_resource_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(shared_stats.references == 2 && shared_stats.unique_uploads == 1,
              "draws with separate views over shared pixels produce one backend texture upload");
        CHECK(shared_resource_stats.texture_binding_references == 2 &&
                  shared_resource_stats.unique_texture_bindings == 2,
              "different component swizzles retain distinct Vulkan image views");

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1");
#else
        setenv("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1", 1);
#endif
        std::vector<uint8_t> legacy =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto legacy_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "");
#else
        unsetenv("PROSPER_NO_BACKEND_TEXTURE_SHARE");
#endif
        CHECK(legacy_stats.references == 2 && legacy_stats.unique_uploads == 2,
              "disable switch restores one backend upload per texture reference");
        CHECK(!shared.empty() && shared == legacy,
              "shared and legacy uploads with distinct view swizzles render byte-identically");
    }

    // A non-zero content ID is an exact-validation proof supplied by the frontend. The backend may
    // retain that uploaded image across render callbacks, but the diagnostic switch must preserve
    // rendered output while forcing the old upload path for controlled A/B runs.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        resource.persistent_texture_id = 0x7020000000000001ull;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;

        // A first upload is published to the persistent map while its command buffer is merely
        // queued, so a discarded batch must invalidate that speculative image. Retrying the same
        // version must upload again rather than sampling undefined contents as an exact-version hit.
        prosper::test::FrameResource failed_resource = resource;
        failed_resource.persistent_texture_id = 0x70200000000000f1ull;
        failed_resource.persistent_texture_version = 1;
        prosper::test::BackendDraw failed_draw = draw;
        failed_draw.R = {failed_resource};
        constexpr uint64_t failed_target_id = 0x75900000000000f1ull;
        prosper::test::BackendColorTarget failed_target{failed_target_id, false, false};
        prosper::test::BackendSubmissionBatch failed_batch;
        const std::vector<uint8_t> failed_pending = prosper::test::render_draws_rgba(
            {failed_draw}, W, H, nullptr, nullptr, false, &failed_target,
            nullptr, nullptr, nullptr, &failed_batch, false);
        failed_batch.discard();
        failed_batch.complete();
        const std::vector<uint8_t> failed_retry = prosper::test::render_draws_rgba(
            {failed_draw}, W, H);
        const auto failed_retry_stats = prosper::test::backend_texture_upload_stats();
        CHECK(failed_pending.empty() && !failed_batch.pending() &&
                  failed_retry_stats.persistent_misses == 1 &&
                  failed_retry_stats.unique_uploads == 1 && !failed_retry.empty(),
              "a discarded first versioned upload is rebuilt before reuse");
        prosper::test::invalidate_persistent_color_target(failed_target_id);

        std::vector<uint8_t> first = prosper::test::render_draws_rgba({draw}, W, H);
        const auto first_stats = prosper::test::backend_texture_upload_stats();
        CHECK(prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 0,
              "publishing a new image does not count its transient first-use binding");
        std::vector<uint8_t> reused = prosper::test::render_draws_rgba({draw}, W, H);
        const auto reused_stats = prosper::test::backend_texture_upload_stats();
        const auto first_binding_stats = prosper::test::backend_resource_reuse_stats();
        std::vector<uint8_t> reused_again = prosper::test::render_draws_rgba({draw}, W, H);
        const auto reused_binding_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(first_stats.persistent_misses == 1 && first_stats.unique_uploads == 1,
              "first exact-validated texture version is uploaded into the persistent cache");
        CHECK(reused_stats.persistent_hits == 1 && reused_stats.unique_uploads == 0 &&
                   reused_stats.upload_bytes == 0,
              "same exact-validated texture version skips its next callback upload");
        CHECK(first_binding_stats.persistent_texture_binding_misses == 1 &&
                  first_binding_stats.persistent_texture_binding_hits == 0 &&
                  first_binding_stats.persistent_texture_binding_entries == 1,
              "first persistent-image reuse retains its exact image-view/sampler binding");
        CHECK(reused_binding_stats.persistent_texture_binding_hits == 1 &&
                  reused_binding_stats.persistent_texture_binding_misses == 0 &&
                  reused_binding_stats.persistent_texture_binding_entries == 1,
              "next callback reuses the retained exact texture binding");
        CHECK(!first.empty() && first == reused && reused == reused_again,
              "persistent texture reuse renders byte-identically to its initial upload");

        // Bound sampler/view contracts per persistent texture. Each call uses one distinct, valid
        // single-mip LOD-bias contract; the 33rd retained contract evicts an idle older one without
        // touching the binding used by the current call.
        prosper::test::BackendResourceReuseStats bounded_binding_stats;
        for (uint32_t i = 1; i <= 32; ++i) {
            resource.lod_bias = static_cast<float>(i) / 64.0f;
            draw.R = {resource};
            std::vector<uint8_t> bounded = prosper::test::render_draws_rgba({draw}, W, H);
            CHECK(!bounded.empty(), "bounded persistent texture binding renders");
            bounded_binding_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(bounded_binding_stats.persistent_texture_binding_entries ==
                      (i < 32 ? i + 1 : 32),
                  "every new binding increments the exact census, eviction keeps it bounded");
        }
        CHECK(bounded_binding_stats.persistent_texture_binding_evictions == 1 &&
                  bounded_binding_stats.persistent_texture_binding_entries == 32,
              "persistent texture bindings evict an idle contract at the per-texture bound");
        resource.lod_bias = 0.0f;
        draw.R = {resource};

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1");
#else
        setenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1", 1);
#endif
        std::vector<uint8_t> bypassed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto bypassed_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "");
#else
        unsetenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES");
#endif
        CHECK(bypassed_stats.persistent_hits == 0 && bypassed_stats.unique_uploads == 1,
              "persistent texture disable switch restores an upload on every callback");
        CHECK(reused == bypassed,
              "persistent cache and forced-upload paths render byte-identically");

        uint8_t changed_texels[sizeof(texels)];
        std::memcpy(changed_texels, texels, sizeof(texels));
        changed_texels[4] = 255;  // sampled top-right texel: green -> yellow
        resource.tex_rgba = changed_texels;
        resource.persistent_texture_version = 1;
        draw.R = {resource};
        std::vector<uint8_t> refreshed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto refreshed_stats = prosper::test::backend_texture_upload_stats();
        std::vector<uint8_t> refreshed_reuse = prosper::test::render_draws_rgba({draw}, W, H);
        const auto refreshed_reuse_stats = prosper::test::backend_texture_upload_stats();
        CHECK(refreshed_stats.persistent_misses == 1 &&
                  refreshed_stats.unique_uploads == 1 &&
                  refreshed_stats.persistent_cached_bytes == reused_stats.persistent_cached_bytes,
              "a newer validated version refreshes its existing persistent image allocation");
        CHECK(refreshed_reuse_stats.persistent_hits == 1 &&
                  refreshed_reuse_stats.unique_uploads == 0 &&
                  refreshed_reuse_stats.persistent_cached_bytes ==
                      reused_stats.persistent_cached_bytes,
              "the refreshed version skips its next callback upload without growing the cache");
        CHECK(!refreshed.empty() && refreshed != reused && refreshed == refreshed_reuse,
              "versioned persistent refresh renders changed pixels and then reuses them exactly");

        resource.tex_rgba = changed_texels;
        resource.persistent_texture_id = 0x7020000000000002ull;
        resource.persistent_texture_version = 0;
        draw.R = {resource};
        std::vector<uint8_t> changed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto changed_stats = prosper::test::backend_texture_upload_stats();
        CHECK(changed_stats.persistent_misses == 1 && changed_stats.unique_uploads == 1,
              "a new exact-validated content version cannot hit the prior image");
        CHECK(!changed.empty() && changed != reused,
              "a new content version uploads and renders its changed pixels");

        CHECK(prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 32,
              "publishing a second image preserves all bindings of the first image");
        const std::vector<uint8_t> second_image_reused =
            prosper::test::render_draws_rgba({draw}, W, H);
        const auto second_image_bindings = prosper::test::backend_resource_reuse_stats();
        const std::vector<uint8_t> second_image_hit =
            prosper::test::render_draws_rgba({draw}, W, H);
        CHECK(second_image_reused == changed && second_image_hit == changed &&
                  second_image_bindings.persistent_texture_binding_entries == 33 &&
                  second_image_bindings.persistent_texture_binding_misses == 1 &&
                  prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 33 &&
                  prosper::test::backend_resource_reuse_stats().persistent_texture_binding_hits == 1,
              "aggregate counts 32 bindings plus a second image's binding, unchanged on a hit");
        const auto second_image_draw = draw;

        // Retire an INVALID image which already owns 32 bindings, not just the zero-binding cold
        // upload above. A discarded version refresh invalidates it; the retry must subtract all
        // those bindings while preserving the second image's one binding.
        resource.persistent_texture_id = 0x7020000000000001ull;
        resource.persistent_texture_version = 2;
        resource.tex_rgba = texels;
        draw.R = {resource};
        constexpr uint64_t refresh_target_id = 0x75900000000000f2ull;
        prosper::test::BackendColorTarget refresh_target{refresh_target_id, false, false};
        prosper::test::BackendSubmissionBatch refresh_batch;
        const std::vector<uint8_t> refresh_pending = prosper::test::render_draws_rgba(
            {draw}, W, H, nullptr, nullptr, false, &refresh_target,
            nullptr, nullptr, nullptr, &refresh_batch, false);
        const auto pending_binding_stats = prosper::test::backend_resource_reuse_stats();
        refresh_batch.discard();
        refresh_batch.complete();
        const std::vector<uint8_t> while_first_invalid =
            prosper::test::render_draws_rgba({second_image_draw}, W, H);
        const auto resident_invalid_bindings = prosper::test::backend_resource_reuse_stats();
        CHECK(while_first_invalid == changed &&
                  resident_invalid_bindings.persistent_texture_binding_entries == 33 &&
                  resident_invalid_bindings.persistent_texture_binding_hits == 1 &&
                  prosper::test::backend_texture_upload_stats().unique_uploads == 0,
              "invalid but resident image keeps its 32 bindings counted until actual retirement");
        const std::vector<uint8_t> refresh_retry = prosper::test::render_draws_rgba({draw}, W, H);
        const auto retry_bindings = prosper::test::backend_resource_reuse_stats();
        const auto retry_uploads = prosper::test::backend_texture_upload_stats();
        CHECK(refresh_pending.empty() && !refresh_batch.pending() &&
                  pending_binding_stats.persistent_texture_binding_entries == 33 &&
                  retry_bindings.persistent_texture_binding_entries == 1 &&
                  retry_uploads.persistent_misses == 1 && retry_uploads.unique_uploads == 1 &&
                  refresh_retry == reused,
              "invalid-image replacement subtracts all 32 owned bindings, retains other image, restores pixels");
        const std::vector<uint8_t> refresh_retry_hit = prosper::test::render_draws_rgba({draw}, W, H);
        CHECK(refresh_retry_hit == refresh_retry &&
                  prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 2,
              "rebuilt image contributes exactly one new persistent binding on its next use");
        prosper::test::invalidate_persistent_color_target(refresh_target_id);

        // The existing device-budget seam avoids driver-specific allocation sizes or 1025 calls.
        // One byte admits no RGBA texture. An unrelated current image leaves both old images idle,
        // so end-of-pass LRU eviction must remove their bindings and restore the census to zero.
        const VkDeviceSize saved_budget = prosper::test::persistent_texture_cache_device_budget();
        {
            struct ScopedTextureBudget {
                VkDeviceSize saved;
                ScopedTextureBudget() {
                    prosper::test::BackendPersistentResourceGuard guard;
                    saved = prosper::test::persistent_texture_cache_device_budget();
                    prosper::test::persistent_texture_cache_device_budget() = 1;
                }
                ~ScopedTextureBudget() {
                    prosper::test::BackendPersistentResourceGuard guard;
                    prosper::test::persistent_texture_cache_device_budget() = saved;
                }
            } budget_guard;
            CHECK(prosper::test::persistent_texture_cache_limit() == 1,
                  "one-byte device-budget lever is effective, not shadowed by an environment override");
            auto eviction_draw = draw;
            eviction_draw.R[0].persistent_texture_id = 0x70200000000000f3ull;
            const std::vector<uint8_t> evicting_pixels =
                prosper::test::render_draws_rgba({eviction_draw}, W, H);
            CHECK(evicting_pixels == reused &&
                      prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 0 &&
                      prosper::test::backend_texture_upload_stats().persistent_cached_bytes == 0,
                  "whole-image LRU eviction removes every retained binding without changing rendered pixels");
        }
        CHECK(prosper::test::persistent_texture_cache_device_budget() == saved_budget,
              "scoped eviction pressure restores the previous device budget");
        const std::vector<uint8_t> after_eviction = prosper::test::render_draws_rgba({draw}, W, H);
        const auto after_eviction_uploads = prosper::test::backend_texture_upload_stats();
        const std::vector<uint8_t> after_eviction_hit = prosper::test::render_draws_rgba({draw}, W, H);
        CHECK(after_eviction == reused && after_eviction_hit == reused &&
                  after_eviction_uploads.persistent_misses == 1 &&
                  prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries == 1,
              "post-eviction image and binding are rebuilt from zero with exact output");
    }

    // A guest color target can remain on the backend GPU between render calls and be sampled by
    // exact identity. Compare that path with the established CPU readback+upload route, then prove
    // a LOADing second pass can skip readback without changing the eventual sampled pixels.
    {
        static const uint32_t kRedPs[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
            0xF800180Fu, 0x03020100u, 0xBF810000u};
        static const uint32_t kGreenPs[] = {
            0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
            0xF800180Fu, 0x03020100u, 0xBF810000u};
        std::vector<uint32_t> red = recompile_fragment(
            kRedPs, sizeof(kRedPs) / sizeof(kRedPs[0]), nullptr);
        std::vector<uint32_t> green = recompile_fragment(
            kGreenPs, sizeof(kGreenPs) / sizeof(kGreenPs[0]), nullptr);
        std::vector<uint32_t> sample_ps(
            ps_template, ps_template + sizeof(ps_template) / sizeof(ps_template[0]));
        sample_ps[1] = C025; sample_ps[3] = C025;
        std::vector<uint32_t> sample = recompile_fragment(
            sample_ps.data(), sample_ps.size(), &rt);

        ResolvedPipelineState opaque{};
        opaque.topology = 3; opaque.color_write_mask = 0xF;
        ResolvedPipelineState additive = opaque;
        additive.blend_enable = true;
        additive.src_color_blend_factor = 1; // ONE
        additive.dst_color_blend_factor = 1; // ONE
        additive.color_blend_op = 0;         // ADD

        prosper::test::BackendDraw producer;
        producer.vs = vert; producer.fs = red; producer.ps = &opaque; producer.vcount = 3;
        constexpr uint64_t target_id = 0x7590000000000001ull;
        prosper::test::BackendColorTarget first_target{target_id, false, true};
        std::vector<uint8_t> first = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &first_target);
        const auto first_target_stats = prosper::test::backend_color_target_stats();
        CHECK(first.size() == (size_t)W * H * 4 && first_target_stats.writes == 1 &&
                  first_target_stats.write_hits == 0 && first_target_stats.readbacks == 1,
              "first persistent color-target write materializes its requested CPU result");

        prosper::test::FrameResource cpu_resource;
        cpu_resource.binding = 4; cpu_resource.set = 1;
        cpu_resource.tex_rgba = first.data(); cpu_resource.tw = W; cpu_resource.th = H;
        prosper::test::BackendDraw cpu_sample;
        cpu_sample.vs = vert; cpu_sample.fs = sample; cpu_sample.R = {cpu_resource};
        cpu_sample.vcount = 3;
        std::vector<uint8_t> cpu_roundtrip = prosper::test::render_draws_rgba(
            {cpu_sample}, W, H);

        prosper::test::FrameResource gpu_resource = cpu_resource;
        gpu_resource.tex_rgba = nullptr;
        gpu_resource.persistent_render_target_id = target_id;
        prosper::test::BackendDraw gpu_sample = cpu_sample;
        gpu_sample.R = {gpu_resource};
        std::vector<uint8_t> gpu_resident = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        const auto sampled_stats = prosper::test::backend_color_target_stats();
        CHECK(!gpu_resident.empty() && gpu_resident == cpu_roundtrip &&
                  sampled_stats.sampled_hits == 1,
              "GPU-resident target sampling matches CPU readback+upload byte-for-byte");

        prosper::test::FrameResource bgra_gpu_resource = gpu_resource;
        bgra_gpu_resource.render_target_guest_format = VK_FORMAT_B8G8R8A8_UNORM;
        bgra_gpu_resource.swizzle[0] = 6u;
        bgra_gpu_resource.swizzle[1] = 5u;
        bgra_gpu_resource.swizzle[2] = 4u;
        bgra_gpu_resource.swizzle[3] = 7u;
        prosper::test::BackendDraw bgra_gpu_sample = gpu_sample;
        bgra_gpu_sample.R = {bgra_gpu_resource};
        const std::vector<uint8_t> bgra_canonical_sample =
            prosper::test::render_draws_rgba({bgra_gpu_sample}, W, H);
        CHECK(bgra_canonical_sample == cpu_roundtrip && center_red(bgra_canonical_sample) > 0x80,
              "BGRA target plus BGR DST_SEL samples the canonical renderer red channel");

        // A packed mip tail may use the same guest ID for two independently rendered extents. The
        // 32x16 destination is not feedback against the retained 64x32 source: declining the borrow
        // on address alone leaves this resource with neither GPU pixels nor a CPU fallback and
        // samples transparent black.
        prosper::test::BackendColorTarget same_id_smaller_target{
            target_id, false, true};
        const std::vector<uint8_t> same_id_different_extent =
            prosper::test::render_draws_rgba(
                {gpu_sample}, W / 2u, H / 2u, nullptr, nullptr, false,
                &same_id_smaller_target);
        const auto same_id_stats = prosper::test::backend_color_target_stats();
        CHECK(!same_id_different_extent.empty() &&
                  center_red_at(same_id_different_extent, W / 2u, H / 2u) ==
                      center_red(cpu_roundtrip) &&
                  same_id_stats.sampled_hits == 1,
              "same-address different-extent target borrows the retained source view");

        // Exercise the generic compute-image borrow carried by FrameResource. Reuse this exact
        // backend-owned image as a convenient fixture, but address it only through the opaque
        // borrowed-image fields: the backend must build a view/sampler, preserve the owner's layout,
        // retain the lease through completion, and never destroy the borrowed VkImage.
        auto* borrowed_fixture = prosper::test::find_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        prosper::test::FrameResource borrowed_resource = cpu_resource;
        borrowed_resource.tex_rgba = nullptr;
        borrowed_resource.persistent_render_target_id = 0;
        borrowed_resource.borrowed_compute_image =
            borrowed_fixture ? static_cast<void*>(borrowed_fixture->image) : nullptr;
        borrowed_resource.borrowed_compute_device =
            static_cast<void*>(prosper::test::render_vk_ctx().dev);
        borrowed_resource.borrowed_compute_image_layout = borrowed_fixture
            ? static_cast<uint32_t>(borrowed_fixture->layout) : 0u;
        borrowed_resource.borrowed_compute_image_lease = std::make_shared<int>(1);
        prosper::test::BackendDraw borrowed_sample = cpu_sample;
        borrowed_sample.R = {borrowed_resource};
        std::vector<uint8_t> borrowed_resident = prosper::test::render_draws_rgba(
            {borrowed_sample}, W, H);
        std::vector<uint8_t> borrowed_followup = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        CHECK(borrowed_fixture && borrowed_resident == cpu_roundtrip &&
                  borrowed_followup == cpu_roundtrip,
              "borrowed compute-image sampling preserves pixels, owner layout, and image lifetime");

        // A live renderer span can retain resources while later target passes are still being
        // recorded into the same queue batch. Drop every caller-owned lease reference before the
        // fence: only the backend cleanup scope may keep the borrowed VkImage pinned at that point.
        bool pending_lease_released = false;
        prosper::test::FrameResource pending_borrow = borrowed_resource;
        pending_borrow.borrowed_compute_image_lease = std::shared_ptr<void>(
            new int(1), [&](void* allocation) {
                pending_lease_released = true;
                delete static_cast<int*>(allocation);
            });
        prosper::test::BackendDraw pending_sample = cpu_sample;
        pending_sample.R = {pending_borrow};
        constexpr uint64_t pending_target_id = 0x7590000000000002ull;
        prosper::test::BackendColorTarget pending_target{
            pending_target_id, false, false};
        prosper::test::BackendSubmissionBatch pending_batch;
        const std::vector<uint8_t> pending_result = prosper::test::render_draws_rgba(
            {pending_sample}, W, H, nullptr, nullptr, false, &pending_target,
            nullptr, nullptr, nullptr, &pending_batch, false);
        pending_borrow.borrowed_compute_image_lease.reset();
        pending_sample.R.clear();
        CHECK(pending_result.empty() && pending_batch.pending() && !pending_lease_released,
              "pending graphics submission retains the borrowed compute-image lease");
        const prosper::test::BackendSubmissionBatchResult pending_submit =
            pending_batch.submit_and_wait(
                prosper::test::render_vk_ctx().dev,
                prosper::test::render_vk_ctx().queue, false);
        pending_batch.complete();
        CHECK(pending_submit.submit_result == VK_SUCCESS &&
                  pending_submit.wait_result == VK_SUCCESS && pending_lease_released,
              "borrowed compute-image lease releases only after graphics completion");
        prosper::test::invalidate_persistent_color_target(pending_target_id);

        // Multiple ordered target calls may record into one queue submission. The producer returns no
        // CPU pixels; the consumer samples that not-yet-submitted target, then its requested readback
        // flushes both command buffers behind one fence.
#ifdef _WIN32
        _putenv_s("PROSPER_RENDER_TIMING", "1");
#else
        setenv("PROSPER_RENDER_TIMING", "1", 1);
#endif
        constexpr uint64_t batched_target_id = 0x9940000000000001ull;
        prosper::test::BackendColorTarget batched_target{batched_target_id, false, false};
        prosper::test::BackendSubmissionBatch submission_batch;
        const std::vector<uint8_t> batched_producer = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &batched_target,
            nullptr, nullptr, nullptr, &submission_batch, false);
        const auto producer_timing = prosper::test::backend_render_timing_stats();
        prosper::test::FrameResource batched_resource = gpu_resource;
        batched_resource.persistent_render_target_id = batched_target_id;
        prosper::test::BackendDraw batched_sample = gpu_sample;
        batched_sample.R = {batched_resource};
        const std::vector<uint8_t> batched_roundtrip = prosper::test::render_draws_rgba(
            {batched_sample}, W, H, nullptr, nullptr, false, nullptr,
            nullptr, nullptr, nullptr, &submission_batch, true);
        const auto consumer_timing = prosper::test::backend_render_timing_stats();
        constexpr uint64_t forced_sync_target_id = 0x9940000000000002ull;
        prosper::test::BackendColorTarget forced_sync_target{
            forced_sync_target_id, false, true};
        prosper::test::BackendSubmissionBatch forced_sync_batch;
        const std::vector<uint8_t> forced_sync_pixels = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &forced_sync_target,
            nullptr, nullptr, nullptr, &forced_sync_batch, false);
        const auto forced_sync_timing = prosper::test::backend_render_timing_stats();

        // An unbound color target (persistent_id == 0) must not request or execute readback even
        // when persistence is inactive, while setting persistent_id non-zero requests it.
        prosper::test::BackendColorTarget unbound_target{0, false, false};
        prosper::test::BackendSubmissionBatch unbound_batch;
        const std::vector<uint8_t> unbound_pixels = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &unbound_target,
            nullptr, nullptr, nullptr, &unbound_batch, false);
        const auto unbound_timing = prosper::test::backend_render_timing_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_RENDER_TIMING", "");
#else
        unsetenv("PROSPER_RENDER_TIMING");
#endif
        CHECK(batched_producer.empty() && producer_timing.queue_submits == 0 &&
                  producer_timing.fence_waits == 0,
              "intermediate batched target records without submitting or waiting");
        CHECK(batched_roundtrip == cpu_roundtrip &&
                  consumer_timing.command_buffers == 2 &&
                  consumer_timing.queue_submits == 1 && consumer_timing.fence_waits == 1,
              "batched producer-to-sampler output matches synchronous output with one submit/wait");
        const auto& timing_ctx = prosper::test::render_vk_ctx();
        const bool gpu_timestamps_supported = timing_ctx.timestamp_valid_bits != 0 &&
                                              timing_ctx.timestamp_period_ns > 0.0;
        CHECK(!gpu_timestamps_supported ||
                  (consumer_timing.gpu_timestamp_samples == 1 &&
                   consumer_timing.gpu_device_ms > 0.0 &&
                   consumer_timing.gpu_device_ms <= consumer_timing.gpu_wait_ms),
              "batched timing reports one device envelope across all command buffers");
        CHECK(forced_sync_pixels == first && forced_sync_timing.queue_submits == 1 &&
                  (!gpu_timestamps_supported ||
                   (forced_sync_timing.gpu_timestamp_samples == 1 &&
                    forced_sync_timing.gpu_device_ms > 0.0 &&
                    forced_sync_timing.gpu_device_ms <= forced_sync_timing.gpu_wait_ms)),
              "synchronous readback closes its device envelope despite a deferred-flush hint");
        CHECK(unbound_pixels.empty() && unbound_timing.queue_submits == 0,
              "unbound target with persistent_id=0 does not trigger readback or flush");
        prosper::test::invalidate_persistent_color_target(forced_sync_target_id);

        prosper::test::BackendDraw add_green;
        add_green.vs = vert; add_green.fs = green; add_green.ps = &additive;
        add_green.vcount = 3;
        std::vector<uint8_t> cpu_accumulated = prosper::test::render_draws_rgba(
            {add_green}, W, H, first.data());
        cpu_resource.tex_rgba = cpu_accumulated.data();
        cpu_sample.R = {cpu_resource};
        std::vector<uint8_t> cpu_accumulated_sample = prosper::test::render_draws_rgba(
            {cpu_sample}, W, H);

        prosper::test::BackendColorTarget deferred_target{target_id, true, false};
        std::vector<uint8_t> deferred = prosper::test::render_draws_rgba(
            {add_green}, W, H, nullptr, nullptr, false, &deferred_target);
        const auto deferred_stats = prosper::test::backend_color_target_stats();
        const bool pinned = prosper::test::pin_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        // Exceed the target-cache entry cap after the deferred target becomes the oldest live entry, so
        // LRU eviction actually fires. Derive the bound from the configured cap (+margin) rather than a
        // magic number: the default was raised 64 -> 256 (#1177), and a fixed 70 no longer triggers any
        // eviction, which would let this pin-vs-eviction guard pass trivially. Without the submit-lifetime
        // pin, normal LRU pressure discards the deferred target's only current pixels before the
        // frontend's final callback can materialize them.
        const uint64_t pressure_count = prosper::test::persistent_color_target_count_limit() + 40;
        for (uint64_t pressure = 0; pressure < pressure_count; ++pressure) {
            prosper::test::BackendColorTarget pressure_target{
                target_id + 0x1000 + pressure, false, false};
            prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &pressure_target);
        }
        std::vector<uint8_t> materialized;
        std::string materialize_error;
        const bool materialized_ok = prosper::test::readback_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM, materialized, materialize_error);
        const bool unpinned = prosper::test::unpin_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        std::vector<uint8_t> deferred_sample = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        CHECK(deferred.empty() && deferred_stats.write_hits == 1 &&
                  deferred_stats.readbacks == 0,
              "persistent LOAD pass can complete without allocating a CPU readback");
        CHECK(pinned && materialized_ok && materialized == cpu_accumulated && unpinned,
              "pinned GPU-only target survives eviction pressure until ordered materialization");
        CHECK(!deferred_sample.empty() && deferred_sample == cpu_accumulated_sample,
              "deferred-readback accumulation matches the CPU reference after sampling");

        prosper::test::invalidate_persistent_color_target(target_id);
        gpu_resource.tex_rgba = cpu_accumulated.data();
        gpu_sample.R = {gpu_resource};
        std::vector<uint8_t> invalidated_fallback = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        const auto fallback_stats = prosper::test::backend_color_target_stats();
        CHECK(fallback_stats.sampled_hits == 0 &&
                  invalidated_fallback == cpu_accumulated_sample,
              "invalidated GPU target uses the supplied CPU fallback instead of stale pixels");

        // #1284: with deferred RTT readback the persistent image can hold a target's ONLY pixels.
        // Evicting an unpinned valid target must hand those exact pixels to the registered sink
        // before the image is destroyed; without the sink the content would be silently lost.
        constexpr uint64_t sink_target_id = 0x6210000000000001ull;
        prosper::test::BackendColorTarget sink_target{sink_target_id, false, false};
        const std::vector<uint8_t> sink_producer_output = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &sink_target);
        struct SinkCapture {
            uint32_t w = 0, h = 0;
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::vector<uint8_t> pixels;
            int hits = 0;
        } sink_capture;
        prosper::test::persistent_color_target_evict_sink() =
            [&sink_capture](uint64_t id, uint32_t w, uint32_t h, VkFormat format,
                            std::vector<uint8_t>&& pixels) {
                if (id != sink_target_id) return;
                ++sink_capture.hits;
                sink_capture.w = w; sink_capture.h = h; sink_capture.format = format;
                sink_capture.pixels = std::move(pixels);
            };
        const uint64_t sink_pressure_count =
            prosper::test::persistent_color_target_count_limit() + 40;
        for (uint64_t pressure = 0; pressure < sink_pressure_count; ++pressure) {
            prosper::test::BackendColorTarget sink_pressure_target{
                sink_target_id + 0x2000 + pressure, false, false};
            prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &sink_pressure_target);
        }
        prosper::test::persistent_color_target_evict_sink() = nullptr;
        CHECK(sink_producer_output.empty() && sink_capture.hits == 1 &&
                  sink_capture.w == W && sink_capture.h == H &&
                  sink_capture.format == VK_FORMAT_R8G8B8A8_UNORM &&
                  sink_capture.pixels == first,
              "evicting a GPU-only color target hands its exact pixels to the registered sink");
    }

    // Renderer-owned FP16 targets must retain HDR values through both CPU RTT readback/upload and
    // direct GPU-resident sampling. A producer value of 2.0 multiplied by 0.25 in the consumer is
    // 0.5; the historical RGBA8 conversion clamped it to 1.0 first and therefore produced 0.25.
    {
        static const uint32_t kHdrPs[] = {
            0x7E0002FFu, 0x40000000u,  // v0 = 2.0
            0x7E0202FFu, 0x3E800000u,  // v1 = 0.25
            0x7E040280u,               // v2 = 0.0
            0x7E0602F2u,               // v3 = 1.0
            0xF800180Fu, 0x03020100u, 0xBF810000u,
        };
        const uint32_t kHdrSamplePs[] = {
            0x7E0002FFu, C025, 0x7E0202FFu, C025,
            0xF0800F08u, 0x00820000u,  // image_sample v[0:3]
            0x100000FFu, 0x3E800000u,  // v0 *= 0.25
            0xF800000Fu, 0x03020100u, 0xBF810000u,
        };
        std::vector<uint32_t> hdr = recompile_fragment(
            kHdrPs, sizeof(kHdrPs) / sizeof(kHdrPs[0]), nullptr);
        std::vector<uint32_t> hdr_sample = recompile_fragment(
            kHdrSamplePs, sizeof(kHdrSamplePs) / sizeof(kHdrSamplePs[0]), &rt);
        CHECK(!hdr.empty() && !hdr_sample.empty(),
              "recompiled native FP16 producer and sampled consumer shaders");
        if (!hdr.empty() && !hdr_sample.empty()) {
            ResolvedPipelineState fp16_state{};
            fp16_state.topology = 3;
            fp16_state.color_write_mask = 0xF;
            fp16_state.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            prosper::test::BackendDraw producer;
            producer.vs = vert; producer.fs = hdr; producer.ps = &fp16_state;
            producer.vcount = 3;

            constexpr uint64_t fp16_target_id = 0x7590000000000016ull;
            prosper::test::BackendColorTarget fp16_target{
                fp16_target_id, false, true, VK_FORMAT_R16G16B16A16_SFLOAT};
            std::vector<uint8_t> native = prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &fp16_target);
            bool native_ok = native.size() == static_cast<size_t>(W) * H * 8;
            float native_red = 0.0f;
            if (native_ok) {
                uint16_t half = 0;
                std::memcpy(&half, native.data() +
                    ((static_cast<size_t>(H / 2) * W + W / 2) * 8), sizeof(half));
                native_red = half_to_float(half);
                native_ok = native_red > 1.99f && native_red < 2.01f;
            }
            CHECK(native_ok, "native FP16 target readback preserves HDR value 2.0");

            prosper::test::FrameResource fp16_resource;
            fp16_resource.binding = 4; fp16_resource.set = 1;
            fp16_resource.tex_rgba = native.data(); fp16_resource.tw = W; fp16_resource.th = H;
            fp16_resource.texture_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            prosper::test::BackendDraw consumer;
            consumer.vs = vert; consumer.fs = hdr_sample; consumer.R = {fp16_resource};
            consumer.vcount = 3;
            std::vector<uint8_t> cpu_sampled = prosper::test::render_draws_rgba(
                {consumer}, W, H);
            const uint8_t cpu_red = center_red(cpu_sampled);
            CHECK(cpu_red >= 126 && cpu_red <= 129,
                  "FP16 readback/upload consumer observes 2.0 before scaling to 0.5");

            prosper::test::FrameResource gpu_fp16_resource = fp16_resource;
            gpu_fp16_resource.tex_rgba = nullptr;
            gpu_fp16_resource.persistent_render_target_id = fp16_target_id;
            consumer.R = {gpu_fp16_resource};
            std::vector<uint8_t> gpu_sampled = prosper::test::render_draws_rgba(
                {consumer}, W, H);
            const auto gpu_sample_stats = prosper::test::backend_color_target_stats();
            CHECK(gpu_sample_stats.sampled_hits == 1 && gpu_sampled == cpu_sampled,
                  "GPU-resident FP16 target sampling matches native CPU RTT round-trip");

            auto* borrowed_fp16_fixture = prosper::test::find_persistent_color_target(
                fp16_target_id, W, H, VK_FORMAT_R16G16B16A16_SFLOAT);
            prosper::test::FrameResource borrowed_fp16_resource = fp16_resource;
            borrowed_fp16_resource.tex_rgba = nullptr;
            borrowed_fp16_resource.borrowed_compute_image = borrowed_fp16_fixture
                ? static_cast<void*>(borrowed_fp16_fixture->image) : nullptr;
            borrowed_fp16_resource.borrowed_compute_device =
                static_cast<void*>(prosper::test::render_vk_ctx().dev);
            borrowed_fp16_resource.borrowed_compute_image_layout = borrowed_fp16_fixture
                ? static_cast<uint32_t>(borrowed_fp16_fixture->layout) : 0u;
            borrowed_fp16_resource.borrowed_compute_image_lease = std::make_shared<int>(1);
            consumer.R = {borrowed_fp16_resource};
            const std::vector<uint8_t> borrowed_fp16_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            consumer.R = {gpu_fp16_resource};
            const std::vector<uint8_t> borrowed_fp16_followup =
                prosper::test::render_draws_rgba({consumer}, W, H);
            CHECK(borrowed_fp16_fixture && borrowed_fp16_sampled == cpu_sampled &&
                      borrowed_fp16_followup == cpu_sampled,
                  "borrowed native FP16 image preserves HDR sampling, layout, and lifetime");
            prosper::test::invalidate_persistent_color_target(fp16_target_id);

            // GTA V's normal/material G-buffer is a two-channel FP16 attachment. Treating its four
            // bytes as RGBA8 leaves allocation sizes deceptively valid while changing both numeric
            // values. Prove the exact native producer, readback/upload, and resident-image routes.
            ResolvedPipelineState rg16_state = fp16_state;
            rg16_state.color0_format = VK_FORMAT_R16G16_SFLOAT;
            producer.ps = &rg16_state;
            constexpr uint64_t rg16_target_id = 0x7590000000000012ull;
            prosper::test::BackendColorTarget rg16_target{
                rg16_target_id, false, true, VK_FORMAT_R16G16_SFLOAT};
            const std::vector<uint8_t> rg16_native = prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &rg16_target);
            bool rg16_native_ok = rg16_native.size() == static_cast<size_t>(W) * H * 4;
            if (rg16_native_ok) {
                uint16_t halves[2]{};
                std::memcpy(halves, rg16_native.data() +
                    ((static_cast<size_t>(H / 2) * W + W / 2) * 4), sizeof(halves));
                const float red_value = half_to_float(halves[0]);
                const float green_value = half_to_float(halves[1]);
                rg16_native_ok = red_value > 1.99f && red_value < 2.01f &&
                                 green_value > 0.249f && green_value < 0.251f;
            }
            CHECK(rg16_native_ok,
                  "native RG16F G-buffer readback preserves two half-float values");

            prosper::test::FrameResource rg16_resource;
            rg16_resource.binding = 4; rg16_resource.set = 1;
            rg16_resource.tex_rgba = rg16_native.data();
            rg16_resource.tw = W; rg16_resource.th = H;
            rg16_resource.texture_format = VK_FORMAT_R16G16_SFLOAT;
            consumer.R = {rg16_resource};
            const std::vector<uint8_t> rg16_cpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            const size_t rg16_center_offset =
                (static_cast<size_t>(H / 2) * W + W / 2) * 4;
            const bool rg16_cpu_ok = rg16_cpu_sampled.size() ==
                    static_cast<size_t>(W) * H * 4 &&
                rg16_cpu_sampled[rg16_center_offset] >= 126 &&
                rg16_cpu_sampled[rg16_center_offset] <= 129 &&
                rg16_cpu_sampled[rg16_center_offset + 1] >= 63 &&
                rg16_cpu_sampled[rg16_center_offset + 1] <= 65;
            CHECK(rg16_cpu_ok,
                  "RG16F readback/upload consumer observes native R and G values");

            prosper::test::FrameResource gpu_rg16_resource = rg16_resource;
            gpu_rg16_resource.tex_rgba = nullptr;
            gpu_rg16_resource.persistent_render_target_id = rg16_target_id;
            consumer.R = {gpu_rg16_resource};
            const std::vector<uint8_t> rg16_gpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            const auto rg16_sample_stats = prosper::test::backend_color_target_stats();
            CHECK(rg16_sample_stats.sampled_hits == 1 &&
                      rg16_gpu_sampled == rg16_cpu_sampled,
                  "GPU-resident RG16F G-buffer sampling matches its native CPU round-trip");
            prosper::test::invalidate_persistent_color_target(rg16_target_id);

            // The adjacent one-channel material target must retain the same FP16 numeric contract;
    // widening it to RGBA8 changed both texel width and numeric values.
            ResolvedPipelineState r16_state = fp16_state;
            r16_state.color0_format = VK_FORMAT_R16_SFLOAT;
            producer.ps = &r16_state;
            constexpr uint64_t r16_target_id = 0x7590000000000013ull;
            prosper::test::BackendColorTarget r16_target{
                r16_target_id, false, true, VK_FORMAT_R16_SFLOAT};
            const std::vector<uint8_t> r16_native = prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &r16_target);
            bool r16_native_ok = r16_native.size() == static_cast<size_t>(W) * H * 2;
            if (r16_native_ok) {
                uint16_t half = 0;
                std::memcpy(&half, r16_native.data() +
                    ((static_cast<size_t>(H / 2) * W + W / 2) * 2), sizeof(half));
                const float red_value = half_to_float(half);
                r16_native_ok = red_value > 1.99f && red_value < 2.01f;
            }
            CHECK(r16_native_ok,
                  "native R16F material target readback preserves its half-float value");

            prosper::test::FrameResource r16_resource;
            r16_resource.binding = 4; r16_resource.set = 1;
            r16_resource.tex_rgba = r16_native.data();
            r16_resource.tw = W; r16_resource.th = H;
            r16_resource.texture_format = VK_FORMAT_R16_SFLOAT;
            consumer.R = {r16_resource};
            const std::vector<uint8_t> r16_cpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            prosper::test::FrameResource gpu_r16_resource = r16_resource;
            gpu_r16_resource.tex_rgba = nullptr;
            gpu_r16_resource.persistent_render_target_id = r16_target_id;
            consumer.R = {gpu_r16_resource};
            const std::vector<uint8_t> r16_gpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            const auto r16_sample_stats = prosper::test::backend_color_target_stats();
            CHECK(r16_sample_stats.sampled_hits == 1 &&
                      center_red(r16_cpu_sampled) >= 126 &&
                      center_red(r16_cpu_sampled) <= 129 &&
                      r16_gpu_sampled == r16_cpu_sampled,
                  "GPU-resident R16F material sampling matches its native CPU round-trip");
            prosper::test::invalidate_persistent_color_target(r16_target_id);

            ResolvedPipelineState r11_state = fp16_state;
            r11_state.color0_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            producer.ps = &r11_state;
            constexpr uint64_t r11_target_id = 0x7590000000000011ull;
            prosper::test::BackendColorTarget r11_target{
                r11_target_id, false, true, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
            const std::vector<uint8_t> r11_native = prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &r11_target);
            prosper::test::FrameResource r11_resource;
            r11_resource.binding = 4; r11_resource.set = 1;
            r11_resource.tex_rgba = r11_native.data();
            r11_resource.tw = W; r11_resource.th = H;
            r11_resource.texture_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            consumer.R = {r11_resource};
            const std::vector<uint8_t> r11_cpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);

            prosper::test::FrameResource gpu_r11_resource = r11_resource;
            gpu_r11_resource.tex_rgba = nullptr;
            gpu_r11_resource.persistent_render_target_id = r11_target_id;
            consumer.R = {gpu_r11_resource};
            const std::vector<uint8_t> r11_gpu_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);

            auto* borrowed_r11_fixture = prosper::test::find_persistent_color_target(
                r11_target_id, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
            prosper::test::FrameResource borrowed_r11_resource = r11_resource;
            borrowed_r11_resource.tex_rgba = nullptr;
            borrowed_r11_resource.borrowed_compute_image = borrowed_r11_fixture
                ? static_cast<void*>(borrowed_r11_fixture->image) : nullptr;
            borrowed_r11_resource.borrowed_compute_device =
                static_cast<void*>(prosper::test::render_vk_ctx().dev);
            borrowed_r11_resource.borrowed_compute_image_layout = borrowed_r11_fixture
                ? static_cast<uint32_t>(borrowed_r11_fixture->layout) : 0u;
            borrowed_r11_resource.borrowed_compute_image_lease = std::make_shared<int>(1);
            consumer.R = {borrowed_r11_resource};
            const std::vector<uint8_t> r11_borrowed_sampled =
                prosper::test::render_draws_rgba({consumer}, W, H);
            consumer.R = {gpu_r11_resource};
            const std::vector<uint8_t> r11_borrowed_followup =
                prosper::test::render_draws_rgba({consumer}, W, H);
            CHECK(r11_native.size() == static_cast<size_t>(W) * H * 4 &&
                      center_red(r11_cpu_sampled) >= 126 &&
                      center_red(r11_cpu_sampled) <= 129 &&
                      r11_gpu_sampled == r11_cpu_sampled && borrowed_r11_fixture &&
                      r11_borrowed_sampled == r11_cpu_sampled &&
                      r11_borrowed_followup == r11_cpu_sampled,
                  "borrowed native R11G11B10 image preserves HDR sampling, layout, and lifetime");
            prosper::test::invalidate_persistent_color_target(r11_target_id);
        }
    }

    // The backend upload key includes depth and image dimensionality. Exercise a real 3D image here
    // so depth-1 3D resources cannot accidentally regress to a 2D Vulkan image/view during sharing.
    {
        const uint32_t ps_3d[] = {
            0x7e0002ffu, C025, 0x7e0202ffu, C025, 0x7e0402ffu, C075,
            0xf0800f10u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable rt_3d;
        { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 2;
          t.width = 2; t.height = 2; t.depth = 2; t.sgpr_base = 8;
          rt_3d.resources.push_back(t); }
        std::vector<uint32_t> frag_3d =
            recompile_fragment(ps_3d, sizeof(ps_3d) / sizeof(ps_3d[0]), &rt_3d);
        CHECK(!frag_3d.empty(), "recompiled a 3D image_sample fragment shader");
        if (!frag_3d.empty()) {
            const uint8_t volume[2*2*2*4] = {
                255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
                0,255,0,255, 0,255,0,255, 0,255,0,255, 0,255,0,255,
            };
            prosper::test::FrameResource resource_3d;
            resource_3d.binding = 4; resource_3d.set = 1;
            resource_3d.tex_rgba = volume;
            resource_3d.tw = 2; resource_3d.th = 2; resource_3d.td = 2;
            resource_3d.img_dim = 2;
            prosper::test::BackendDraw draw_3d;
            draw_3d.vs = vert; draw_3d.fs = frag_3d; draw_3d.R = {resource_3d};
            draw_3d.vcount = 3;
            std::vector<uint8_t> volume_px =
                prosper::test::render_draws_rgba({draw_3d, draw_3d}, W, H);
            const auto volume_stats = prosper::test::backend_texture_upload_stats();
            bool volume_ok = volume_px.size() == static_cast<size_t>(W) * H * 4;
            if (volume_ok) {
                const uint8_t* c = &volume_px[((size_t)(H/2) * W + W/2) * 4];
                volume_ok = c[1] > 0x80 && c[0] < 0x40 && c[2] < 0x40;
            }
            CHECK(volume_ok, "shared 3D upload samples the selected depth slice");
            CHECK(volume_stats.references == 2 && volume_stats.unique_uploads == 1 &&
                      volume_stats.upload_bytes == sizeof(volume),
                  "3D texture sharing accounts for depth and uploads the volume once");
        }
    }

    bool ok2 = sample_center(C025, C075, rgb);
    printf("  (0.25,0.75) center=(%u,%u,%u)\n", ok2?rgb[0]:0, ok2?rgb[1]:0, ok2?rgb[2]:0);
    CHECK(ok2 && rgb[2] > 0x80 && rgb[0] < 0x40 && rgb[1] < 0x40, "sampling texel (0,1) yields BLUE (proves v routing)");

    // #275: anisotropy applied. A sampler built with max_aniso_ratio > 0 (here 4 -> 16x) must create
    // validly and still sample correctly — the fullscreen quad isn't minified, so aniso changes nothing
    // about the result, but a broken apply (invalid usage / wrong sampler) would blank the frame or move
    // the texel. Same (0.75,0.25) -> texel (1,0) = GREEN check, now through the anisotropic sampler.
    {
        prosper::test::TexDesc td_a{ /*binding*/4, /*w*/2, /*h*/2, texels, /*max_aniso_ratio*/4u };
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        bool okA = !frag.empty() && frag[0] == 0x07230203u;
        if (okA) {
            std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td_a);
            okA = px.size() == (size_t)W*H*4;
            if (okA) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        }
        printf("  aniso(16x) (0.75,0.25) center=(%u,%u,%u)\n", okA?rgb[0]:0, okA?rgb[1]:0, okA?rgb[2]:0);
        CHECK(okA && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40,
              "anisotropic sampler (ratio 4/16x) still samples texel (1,0) = GREEN (valid apply)");
    }

    // image_load (integer texel fetch, no sampler): x,y = inline-int coords into v0,v1; image_load
    // v[0:3], v[0:1], s[8:15]; exp mrt0. Coord (x,y) directly indexes the texel — no filtering.
    const uint32_t il_template[] = {
        0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    auto fetch_center = [&](uint32_t x, uint32_t y, uint8_t o[3]) -> bool {
        std::vector<uint32_t> ps(il_template, il_template + sizeof(il_template)/sizeof(il_template[0]));
        ps[0] = 0x7e000200u | (128u + x);   // v_mov_b32 v0, x   (inline int)
        ps[1] = 0x7e020200u | (128u + y);   // v_mov_b32 v1, y
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        o[0]=c[0]; o[1]=c[1]; o[2]=c[2]; return true;
    };
    bool okL0 = fetch_center(0, 0, rgb);
    printf("  image_load(0,0) center=(%u,%u,%u)\n", okL0?rgb[0]:0, okL0?rgb[1]:0, okL0?rgb[2]:0);
    CHECK(okL0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "image_load texel (0,0) yields RED");
    bool okL1 = fetch_center(1, 1, rgb);
    printf("  image_load(1,1) center=(%u,%u,%u)\n", okL1?rgb[0]:0, okL1?rgb[1]:0, okL1?rgb[2]:0);
    CHECK(okL1 && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80, "image_load texel (1,1) yields WHITE (proves integer coords)");

    // Guest 2D_MSAA IMAGE_LOAD uses an explicit integer sample coordinate. Prosper materializes the
    // four exact guest sample planes as a single-sample 2D-array and lowers that coordinate to the
    // host layer. Four distinct R32F values prove upload order, view shape, and shader routing together.
    {
        const uint32_t msaa_ps_template[] = {
            0x7e0a0280u,                       // v5 = x = 0
            0x7e0c0280u,                       // v6 = y = 0
            0x7e0e0280u,                       // v7 = sample (patched below)
            0xf0000130u, 0x00000305u,         // exact image_load v3,v[5:7],s[0:7] dmask:x 2D_MSAA
            0x7e000303u,                       // v0 = loaded R32F bits
            0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable msaa_rt;
        { ShaderResource image{}; image.cls = ResourceClass::Texture;
          image.format = DataFormat::Float32; image.num_components = 1;
          image.binding = 4; image.sgpr_base = 0; image.img_dim = 6;
          image.width = 1; image.height = 1; image.depth = 1;
          image.sample_count = 4; image.declared_mip_levels = 1;
          msaa_rt.resources.push_back(image); }
        const float sample_planes[4] = {0.125f, 0.375f, 0.625f, 0.875f};
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = reinterpret_cast<const uint8_t*>(sample_planes);
        resource.tex_byte_size = sizeof(sample_planes);
        resource.tw = 1; resource.th = 1; resource.td = 1;
        resource.img_dim = 6; resource.sample_count = 4;
        resource.texture_format = VK_FORMAT_R32_SFLOAT;
        resource.mag_filter = resource.min_filter = 0;
        bool distinct_samples = true;
        const uint8_t expected_red[4] = {32, 96, 159, 223};
        for (uint32_t sample = 0; sample < 4; ++sample) {
            std::vector<uint32_t> ps(std::begin(msaa_ps_template),
                                     std::end(msaa_ps_template));
            ps[2] = 0x7e0e0200u | (128u + sample);
            const std::vector<uint32_t> frag = recompile_fragment(
                ps.data(), ps.size(), &msaa_rt);
            prosper::test::BackendDraw draw;
            draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
            const std::vector<uint8_t> pixels = prosper::test::render_draws_rgba(
                {draw}, W, H);
            const uint8_t* center = pixels.size() == static_cast<size_t>(W) * H * 4
                ? &pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4] : nullptr;
            distinct_samples &= center &&
                std::abs(static_cast<int>(center[0]) - expected_red[sample]) <= 2 &&
                center[1] < 4 && center[2] < 4;
        }
        CHECK(distinct_samples,
              "2D_MSAA IMAGE_LOAD selects four distinct host array layers by guest sample index");
        prosper::test::FrameResource short_resource = resource;
        short_resource.tex_byte_size = sizeof(sample_planes) - sizeof(float);
        prosper::test::BackendDraw short_draw;
        short_draw.vs = vert;
        short_draw.fs = recompile_fragment(
            msaa_ps_template, std::size(msaa_ps_template), &msaa_rt);
        short_draw.R = {short_resource}; short_draw.vcount = 3;
        CHECK(!prosper::test::backend_texture_plane_span_valid(short_resource) &&
                  prosper::test::render_draws_rgba({short_draw}, W, H).empty(),
              "2D_MSAA upload rejects a short sample-plane span before Vulkan access");
        prosper::test::FrameResource wrong_count = resource;
        wrong_count.sample_count = 2;
        CHECK(!prosper::test::backend_texture_plane_span_valid(wrong_count),
              "unsupported host sample-plane counts remain fail-visible at the backend boundary");
    }

    // #325: a guest 2D_ARRAY rides the same `sample_count` layer channel as the MSAA plane array.
    // The span predicate admitted sample_count > 1 ONLY for the four-plane R32F MSAA shape, so it
    // is what a frontend publishing a layer count would run into first. It was NOT what made
    // Tomb Raider I-III Remastered render flat: every graphics array arrives with sample_count == 1
    // today and passes at the `== 1u` fast path, so nothing is being rejected here yet. The arms
    // below are chosen to fail if the bounds arithmetic is dropped rather than merely to agree with
    // the implementation.
    {
        const uint32_t LW = 4, LH = 4, LAYERS = 8;
        std::vector<uint8_t> layer_bytes(static_cast<size_t>(LW) * LH * 4u * LAYERS, 0x40);
        prosper::test::FrameResource arr{};
        arr.tex_rgba = layer_bytes.data();
        arr.tex_byte_size = layer_bytes.size();
        arr.tw = LW; arr.th = LH; arr.td = 1;
        arr.img_dim = 5;
        arr.sample_count = LAYERS;
        arr.texture_format = VK_FORMAT_R8G8B8A8_UNORM;
        CHECK(prosper::test::backend_texture_plane_span_valid(arr),
              "a complete 8-layer 2D_ARRAY span is accepted (#325)");

        prosper::test::FrameResource short_arr = arr;
        short_arr.tex_byte_size = layer_bytes.size() - 1;
        CHECK(!prosper::test::backend_texture_plane_span_valid(short_arr),
              "a 2D_ARRAY one byte short of its last layer is rejected before Vulkan");

        // The bound must scale with the LAYER COUNT, not merely with one layer. A predicate that
        // checked a single layer's worth of bytes would pass this, and would then memcpy 8 layers
        // out of a 1-layer allocation.
        prosper::test::FrameResource one_layer_span = arr;
        one_layer_span.tex_byte_size = static_cast<size_t>(LW) * LH * 4u;
        CHECK(!prosper::test::backend_texture_plane_span_valid(one_layer_span),
              "a 2D_ARRAY span covering only one layer is rejected for an 8-layer resource");

        // Bytes-per-texel must come from the format. Declaring a 16-byte format over the same
        // buffer needs four times the span, so this arm fails if bpp is assumed to be 4.
        prosper::test::FrameResource wide_format = arr;
        wide_format.texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;
        CHECK(!prosper::test::backend_texture_plane_span_valid(wide_format),
              "2D_ARRAY span scales with the format's bytes per texel, not a fixed 4");

        // An unbounded layer count reaches vkCreateImage, whose result the upload path does not
        // check -- so an absurd arrayLayers produces VK_NULL_HANDLE and is handed to
        // vkGetImageMemoryRequirements (#3045). A 4x4 RGBA8 array of 8192 layers is only 512 KiB,
        // so the span check alone does NOT bound this in practice -- which is why the arm exists.
        // kBackendMaxArrayLayers is a pragmatic ceiling, not a portability guarantee: Vulkan Core
        // requires only 256 and this backend requests 1.1.
        {
            const uint32_t huge = prosper::test::kBackendMaxArrayLayers + 1u;
            std::vector<uint8_t> huge_bytes(static_cast<size_t>(LW) * LH * 4u * huge, 0x40);
            prosper::test::FrameResource too_many = arr;
            too_many.tex_rgba = huge_bytes.data();
            too_many.tex_byte_size = huge_bytes.size();
            too_many.sample_count = huge;
            CHECK(!prosper::test::backend_texture_plane_span_valid(too_many),
                  "a layer count above the backend's array-layer ceiling is rejected even when its "
                  "span is complete");
            prosper::test::FrameResource at_limit = arr;
            at_limit.sample_count = prosper::test::kBackendMaxArrayLayers;
            at_limit.tex_byte_size =
                static_cast<size_t>(LW) * LH * 4u * prosper::test::kBackendMaxArrayLayers;
            std::vector<uint8_t> limit_bytes(at_limit.tex_byte_size, 0x40);
            at_limit.tex_rgba = limit_bytes.data();
            CHECK(prosper::test::backend_texture_plane_span_valid(at_limit),
                  "exactly the backend's array-layer ceiling is still accepted");
        }

        prosper::test::FrameResource volume = arr;
        volume.td = 2;
        CHECK(!prosper::test::backend_texture_plane_span_valid(volume),
              "a layered resource that is also a volume is not an array upload");

        prosper::test::FrameResource storage = arr;
        storage.is_storage_image = true;
        CHECK(!prosper::test::backend_texture_plane_span_valid(storage),
              "a layered storage image is not admitted by the array arm");

        prosper::test::FrameResource no_pixels = arr;
        no_pixels.tex_rgba = nullptr;
        CHECK(!prosper::test::backend_texture_plane_span_valid(no_pixels),
              "a layered resource with no pixels is rejected");

        // The historical single-layer contract is untouched: tex_byte_size == 0 still means "no
        // explicit span" and must keep passing, or every ordinary texture in the project regresses.
        prosper::test::FrameResource single = arr;
        single.sample_count = 1;
        single.tex_byte_size = 0;
        CHECK(prosper::test::backend_texture_plane_span_valid(single),
              "single-layer resources keep the historical implicit-span contract");

        // And the MSAA arm still routes by img_dim, so admitting arrays cannot widen it.
        prosper::test::FrameResource msaa_two = arr;
        msaa_two.img_dim = 6; msaa_two.sample_count = 2;
        CHECK(!prosper::test::backend_texture_plane_span_valid(msaa_two),
              "the MSAA plane-array arm is unchanged by the 2D_ARRAY addition");
    }

    // The same image_load instruction becomes OpImageRead when its resource class is StorageImage.
    // That SPIR-V interface must be backed by a STORAGE_IMAGE descriptor over an image with STORAGE
    // usage in GENERAL layout, not the sampled texture's combined-image-sampler contract (#374).
    {
        ShaderResourceTable storage_rt;
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.format = DataFormat::Unorm8; image.num_components = 4;
          image.binding = 4; image.img_dim = 1; image.width = 2; image.height = 2;
          image.sgpr_base = 8; storage_rt.resources.push_back(image); }
        std::vector<uint32_t> storage_frag = recompile_fragment(
            il_template, sizeof(il_template) / sizeof(il_template[0]), &storage_rt);
        CHECK(!storage_frag.empty() && storage_frag[0] == 0x07230203u,
              "recompiled graphics image_load as a storage-image OpImageRead");
        if (!storage_frag.empty()) {
            const DescriptorValidationReport storage_interface =
                validate_spirv_descriptor_interface(
                    storage_frag, &storage_rt, 1, SpirvShaderStage::Fragment, false);
            const SpirvDescriptorBinding* storage_binding =
                find_spirv_descriptor_binding(storage_interface, 1, 4);
            CHECK(storage_binding && storage_binding->image_numeric_class ==
                      SpirvImageNumericClass::Uint &&
                      storage_binding->storage_image_format == 0u,
                  "graphics storage image reflects the portable unsigned-integer ABI");
            uint32_t storage_texels[2 * 2 * 4];
            for (size_t i = 0; i < std::size(storage_texels); ++i) {
                const float normalized = texels[i] / 255.0f;
                std::memcpy(&storage_texels[i], &normalized, sizeof(normalized));
            }
            prosper::test::FrameResource storage_resource;
            storage_resource.binding = 4; storage_resource.set = 1;
            storage_resource.tex_rgba = reinterpret_cast<const uint8_t*>(storage_texels);
            storage_resource.tw = 2; storage_resource.th = 2;
            storage_resource.texture_format = VK_FORMAT_R32G32B32A32_UINT;
            storage_resource.is_storage_image = true;
            storage_resource.storage_image_numeric_class =
                storage_binding ? storage_binding->image_numeric_class
                                : SpirvImageNumericClass::Unknown;
            prosper::test::BackendDraw storage_draw;
            storage_draw.vs = vert; storage_draw.fs = storage_frag;
            storage_draw.R = {storage_resource}; storage_draw.vcount = 3;
            prosper::test::FrameResource mismatched_storage = storage_resource;
            mismatched_storage.tex_rgba = texels;
            mismatched_storage.texture_format = VK_FORMAT_R8G8B8A8_UNORM;
            prosper::test::BackendDraw mismatched_draw = storage_draw;
            mismatched_draw.R = {mismatched_storage};
            CHECK(!prosper::test::backend_storage_image_numeric_contract_valid(
                      mismatched_storage) &&
                      prosper::test::render_draws_rgba({mismatched_draw}, W, H).empty(),
                  "UINT storage shader plus UNORM view is rejected before Vulkan (#1713)");
            prosper::test::FrameResource sampled_resource = storage_resource;
            sampled_resource.is_storage_image = false;
            sampled_resource.tex_rgba = texels;
            sampled_resource.texture_format = VK_FORMAT_R8G8B8A8_UNORM;
            prosper::test::BackendDraw sampled_draw = storage_draw;
            sampled_draw.fs = recompile_fragment(
                il_template, sizeof(il_template) / sizeof(il_template[0]), &rt);
            sampled_draw.R = {sampled_resource};
            std::vector<uint8_t> storage_px = prosper::test::render_draws_rgba(
                {storage_draw}, W, H);
            bool storage_ok = storage_px.size() == static_cast<size_t>(W) * H * 4;
            if (storage_ok) {
                const uint8_t* c = &storage_px[((size_t)(H / 2) * W + W / 2) * 4];
                storage_ok = c[0] > 0x80 && c[1] < 0x40 && c[2] < 0x40;
            }
            CHECK(storage_ok,
                  "graphics storage-image descriptor reads texel (0,0) into the framebuffer");
            // Separately exercise the alias case. Both descriptors reference the same decoded bytes
            // in one backend call but need distinct VkImages because their usage and layouts differ.
            prosper::test::render_draws_rgba({sampled_draw, storage_draw}, W, H);
            const auto storage_stats = prosper::test::backend_texture_upload_stats();
            CHECK(storage_stats.references == 2 && storage_stats.unique_uploads == 2,
                  "sampled and storage descriptors never share an incompatible image upload");
        }
    }

    // Astro Bot's exact IMAGE_ATOMIC_SWAP form. Use a one-pixel target so exactly one fragment
    // exchanges the R32_UINT word. The callback is the graphics->guest synchronization boundary:
    // call one returns 0.5 and writes 1.0 to guest-visible storage; call two must then return that 1.0.
    // This proves both real Vulkan execution and visibility across separate backend calls/CPU memory.
    {
        const uint32_t atomic_ps[] = {
            0x7e000280u,                         // v0 = x = 0
            0x7e020280u,                         // v1 = y = 0
            0x7e1202ffu, 0x3f800000u,           // v9 = exchange value (1.0f bits)
            0xf03c2108u, 0x00000900u,           // image_atomic_swap v9,[v0,v1],s[0:7] glc dmask:x 2D
            0x7e000280u, 0x7e020309u,           // R=0, G=returned v9
            0x7e040280u, 0x7e0602f2u,           // B=0, A=1
            0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable atomic_rt;
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.format = DataFormat::Uint32; image.num_components = 1;
          image.binding = 4; image.img_dim = 1; image.width = 1; image.height = 1;
          image.depth = 1; image.sgpr_base = 0; atomic_rt.resources.push_back(image); }
        const std::vector<uint32_t> atomic_frag = recompile_fragment(
            atomic_ps, std::size(atomic_ps), &atomic_rt);
        CHECK(!atomic_frag.empty() && atomic_frag[0] == 0x07230203u,
              "recompiled Astro image_atomic_swap fragment shader");
        if (!atomic_frag.empty()) {
            uint32_t return_word = 0x3f000000u;   // initial 0.5f bits
            prosper::test::FrameResource resource;
            resource.binding = 4; resource.set = 1;
            resource.tex_rgba = reinterpret_cast<const uint8_t*>(&return_word);
            resource.tw = 1; resource.th = 1;
            resource.texture_format = VK_FORMAT_R32_UINT;
            resource.is_storage_image = true;
            resource.storage_image_numeric_class = SpirvImageNumericClass::Uint;
            prosper::test::BackendDraw draw;
            draw.vs = vert; draw.fs = atomic_frag; draw.R = {resource}; draw.vcount = 3;
            const std::vector<uint8_t> returned =
                prosper::test::render_draws_rgba({draw}, 1, 1);
            const uint8_t* returned_pixel = returned.size() == 4 ? returned.data() : nullptr;
            CHECK(returned_pixel && returned_pixel[1] > 0x60 &&
                      returned_pixel[1] < 0xa0 && returned_pixel[0] < 0x40 &&
                      returned_pixel[2] < 0x40,
                  "R32_UINT image atomic returns the pre-operation 0.5 value to VDATA");

            // Keep the upload source distinct from the callback destination. This prevents the test
            // from passing merely because a mutable source variable was updated in place.
            const uint32_t upload_word = 0x3f000000u;
            uint32_t guest_word = upload_word;
            resource.tex_rgba = reinterpret_cast<const uint8_t*>(&upload_word);
            resource.storage_image_writeback =
                [&](const uint8_t* pixels, size_t bytes) {
                    if (bytes == sizeof(guest_word))
                        std::memcpy(&guest_word, pixels, sizeof(guest_word));
                };
            draw.vs = vert; draw.fs = atomic_frag; draw.R = {resource}; draw.vcount = 3;
            constexpr uint64_t storage_target_id = 0x41544f4d49430001ull;
            prosper::test::BackendColorTarget storage_target{
                storage_target_id, false, false};
            prosper::test::BackendSubmissionBatch storage_batch;
            const std::vector<uint8_t> storage_only =
                prosper::test::render_draws_rgba(
                    {draw}, 1, 1, nullptr, nullptr, false, &storage_target,
                    nullptr, nullptr, nullptr, &storage_batch, false);
            const auto storage_stats = prosper::test::backend_color_target_stats();
            printf("  image_atomic_swap storage-only guest=%08x source=%08x\n",
                   guest_word, upload_word);
            CHECK(storage_only.empty() && storage_stats.readbacks == 0,
                  "storage-image synchronization does not rely on a color readback");
            CHECK(upload_word == 0x3f000000u && guest_word == 0x3f800000u,
                  "graphics storage-image atomic writes back exact texels to CPU/guest memory");
            CHECK(!storage_batch.pending(),
                  "storage-image writeback flushes an otherwise deferred live-render batch");
            draw.R[0].tex_rgba = reinterpret_cast<const uint8_t*>(&guest_word);
            const std::vector<uint8_t> second =
                prosper::test::render_draws_rgba({draw}, 1, 1);
            const uint8_t* second_pixel = second.size() == 4 ? second.data() : nullptr;
            CHECK(second_pixel && second_pixel[1] > 0xc0 &&
                      second_pixel[0] < 0x40 && second_pixel[2] < 0x40,
                  "a later graphics call observes the prior storage-image writeback");
        }
    }

    // image_sample_lz (explicit LOD 0): coords (0.75,0.25) -> texel (1,0) = green, same as image_sample.
    const uint32_t lz[] = {
        0x7e0002ffu, 0x3f400000u, 0x7e0202ffu, 0x3e800000u, 0xf09c0f08u, 0x00820000u,
        0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzf = recompile_fragment(lz, sizeof(lz)/sizeof(lz[0]), &rt);
    bool okLz = !lzf.empty() && lzf[0] == 0x07230203u;
    if (okLz) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzf, W, H, nullptr, nullptr, nullptr, &td);
        okLz = px.size() == (size_t)W*H*4;
        if (okLz) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
    }
    printf("  image_sample_lz(0.75,0.25) center=(%u,%u,%u)\n", okLz?rgb[0]:0, okLz?rgb[1]:0, okLz?rgb[2]:0);
    CHECK(okLz && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "image_sample_lz (explicit LOD 0) samples texel (1,0) = GREEN");

    // image_sample_lz_o (#273 — DOLL FXAA): LOD-0 sample with a packed TEXEL offset in the first
    // vaddr (x=+1 in bits[5:0], y=+1 in bits[13:8]). Sampling (0.25,0.25) with offset (+1,+1) must
    // land on texel (1,1) = WHITE (the offset folds into the normalized coords via the level-0 size).
    const uint32_t lzo[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf0dc0f08u, 0x00820000u, 0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzof = recompile_fragment(lzo, sizeof(lzo)/sizeof(lzo[0]), &rt);
    bool okLzo = !lzof.empty() && lzof[0] == 0x07230203u;
    CHECK(okLzo, "recompiled image_sample_lz_o PS -> SPIR-V");
    if (okLzo) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzof, W, H, nullptr, nullptr, nullptr, &td);
        okLzo = px.size() == (size_t)W*H*4;
        if (okLzo) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_sample_lz_o(0.25,0.25,+1,+1) center=(%u,%u,%u)\n", okLzo?rgb[0]:0, okLzo?rgb[1]:0, okLzo?rgb[2]:0);
        CHECK(okLzo && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_sample_lz_o offset (+1,+1) from texel (0,0) samples texel (1,1) = WHITE");
    }

    // image_gather4_lz_o (locks the #296 helper's operand-ID fix): gather the ALPHA channel (dmask
    // 0x8) with a packed (+1,+1) offset — every texel's alpha is 255, so all four gathered values
    // are 1.0 and the export is WHITE. (Before the fix the emitted OpBitFieldSExtract used raw
    // integers 0/6/8 as operand IDs — an invalid module the driver rejects -> nothing renders.)
    const uint32_t g4o[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf15c0808u, 0x00820400u, 0xf800000fu, 0x07060504u, 0xbf810000u,
    };
    std::vector<uint32_t> g4of = recompile_fragment(g4o, sizeof(g4o)/sizeof(g4o[0]), &rt);
    bool okG4o = !g4of.empty() && g4of[0] == 0x07230203u;
    CHECK(okG4o, "recompiled image_gather4_lz_o PS -> SPIR-V");
    if (okG4o) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, g4of, W, H, nullptr, nullptr, nullptr, &td);
        okG4o = px.size() == (size_t)W*H*4;
        if (okG4o) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_gather4_lz_o alpha center=(%u,%u,%u)\n", okG4o?rgb[0]:0, okG4o?rgb[1]:0, okG4o?rgb[2]:0);
        CHECK(okG4o && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_gather4_lz_o gathers alpha=1.0 x4 -> WHITE (valid module, offset decoded)");
    }

    // Run this last: abandoning unproven GPU work intentionally poisons the renderer for the rest
    // of the device lifetime, so no subsequent render/eviction test should share this process.
    {
        bool speculative_state_valid = true;
        bool prior_cleanup_ran = false;
        bool later_cleanup_ran = false;
        bool borrowed_lease_released = false;
        bool late_borrowed_lease_released = false;
        auto borrowed_lease = std::shared_ptr<void>(
            new int(1),
            [&](void* ptr) {
                borrowed_lease_released = true;
                delete static_cast<int*>(ptr);
            });
        CHECK(!prosper::test::backend_has_unproven_submission(),
              "completed test submissions leave the global lifetime guard clear");
        {
            prosper::test::BackendSubmissionBatch abandoned_batch;
            abandoned_batch.enqueue(VK_NULL_HANDLE);
            abandoned_batch.add_failure_cleanup([&]() { speculative_state_valid = false; });
            abandoned_batch.add_cleanup([&, lease = borrowed_lease]() {
                (void)lease;
                prior_cleanup_ran = true;
            });
            borrowed_lease.reset();
            abandoned_batch.abandon_pending_resources();
            auto late_borrowed_lease = std::shared_ptr<void>(
                new int(2),
                [&](void* ptr) {
                    late_borrowed_lease_released = true;
                    delete static_cast<int*>(ptr);
                });
            abandoned_batch.add_cleanup([&, lease = late_borrowed_lease]() {
                (void)lease;
                later_cleanup_ran = true;
            });
            late_borrowed_lease.reset();
            abandoned_batch.complete();
            CHECK(!abandoned_batch.pending() && abandoned_batch.retains_pending_resources() &&
                      prosper::test::backend_has_unproven_submission() &&
                      !speculative_state_valid && !prior_cleanup_ran && !later_cleanup_ran,
                  "pending submission globally retains prior and late resources");
        }
        CHECK(!borrowed_lease_released && !late_borrowed_lease_released,
              "borrowed leases survive destruction of an abandoned submission batch");
        const std::vector<uint8_t> blocked = prosper::test::render_triangle_rgba(
            vert, lzf, W, H, nullptr, nullptr, nullptr, &td);
        CHECK(blocked.empty(),
              "global lifetime guard rejects renderer work from a fresh batch");

        bool rejected_state_valid = true;
        bool rejected_cleanup_ran = false;
        prosper::test::BackendSubmissionBatch rejected_batch;
        rejected_batch.enqueue(VK_NULL_HANDLE);
        rejected_batch.add_failure_cleanup([&]() { rejected_state_valid = false; });
        rejected_batch.add_cleanup([&]() { rejected_cleanup_ran = true; });
        const auto rejected = rejected_batch.submit_and_wait(
            VK_NULL_HANDLE, VK_NULL_HANDLE, false);
        rejected_batch.complete();
        CHECK(rejected.submit_result == VK_ERROR_DEVICE_LOST &&
                  rejected.wait_result == VK_ERROR_DEVICE_LOST &&
                  !rejected_batch.pending() && !rejected_state_valid && rejected_cleanup_ran,
              "global lifetime guard discards and cleans never-submitted queued work");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
