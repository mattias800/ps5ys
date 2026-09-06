// test_render_object_create_failure -- #3210: the NON-image object creates in `render_draw_pass_rgba`
// must check their result, and the two that were UNINITIALIZED must never be read on failure.
//
// Three severities, one mechanism (see render_runner.h's RenderVkObjectCreateSite comment):
//
//   1. `rp`/`fb` were declared without an initializer and used with no guard -- so on a failed
//      create an arbitrary stack word reached vkCreateFramebuffer, every pipeline's `renderPass`,
//      vkCmdBeginRenderPass and vkDestroyRenderPass. Those sites had NO failure path, so this
//      test's "the pass is dropped" assertions for them are genuinely new behaviour, not a
//      restatement of the status quo.
//   2. the depth/stencil view and the four sampled-texture view/sampler creates were checked
//      nowhere either, but their handles were value-initialized, so the failure was at least
//      deterministic. Two of the four also CACHED the result into the persistent binding map.
//   3. vkCreateShaderModule's result was discarded while `m` was initialized and the caller
//      already skipped the draw. Only the report is new there.
//
// WHICH ASSERTIONS ARE LOAD-BEARING. The log assertions are the contract: unfixed code prints
// nothing at all, so every one of them goes red. The "pass is dropped" / "draw is skipped" checks
// are load-bearing ONLY for cases 1 and 2 (which had no failure path at all); the shader-module
// arm's skip assertion would pass on unfixed code and is stated so a later change that reported by
// CONTINUING past the failure would still be caught.
//
// WHAT THIS TEST CANNOT DISTINGUISH. For case 1 it asserts that the pass is dropped and that the
// failure is named -- it CANNOT show that the pre-fix handle was indeterminate rather than null.
// Reading an uninitialized automatic is undefined behaviour with no observable value a test may
// assert on; in practice this box's RADV does leave the handle null, so an "unfixed code crashed"
// arm would be unreproducible and would prove nothing about a driver that behaves differently.
// The initialization is established by reading the code (`*out = VK_NULL_HANDLE` before the call),
// not by this test.
//
// The failures are injected, for the reason #3045 and #3180 both recorded: these are ordinary
// render passes, framebuffers, 2D views, trilinear samplers and small SPIR-V modules at the test's
// own 64x64 extent, so no real device can be made to refuse one on demand.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/state/render_state.hpp"
#include "fixtures/render_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::gpu;
using prosper::test::RenderVkObjectCreateSite;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

// Run `body` with stderr redirected to a scratch file and return everything it printed. stderr is
// not restored; every assertion in this test reports on stdout.
template <typename Body>
static std::string capture_stderr(const char* scratch, Body&& body) {
    std::fflush(stderr);
    if (!std::freopen(scratch, "w+", stderr)) {
        printf("  [FAIL] cannot redirect stderr\n"); fails++; return {};
    }
    body();
    std::fflush(stderr);
    std::string text;
    if (FILE* f = std::fopen(scratch, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        std::fclose(f);
    }
    std::remove(scratch);
    return text;
}

static const uint32_t W = 64, H = 64;

static uint8_t center_red(const std::vector<uint8_t>& px) {
    return px.size() == static_cast<size_t>(W) * H * 4
        ? px[(static_cast<size_t>(H / 2) * W + W / 2) * 4] : 0;
}

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    printf("== test_render_object_create_failure (#3210) ==\n");

    // The persistent GRAPHICS PIPELINE cache is process-wide and outlives a render, so a second
    // render of the same shaders + state never calls vkCreateShaderModule at all. That makes the
    // shader-module arm non-deterministic in the worst way: its control run populates the cache,
    // its injected run then skips the create, and the arm stays pending until it fires inside some
    // LATER arm's control run. (Observed exactly that before this line existed -- the transient
    // texture control run took the shader-module arm's injection.) Disable the cache so every
    // render in this test really does create its modules. Nothing else here depends on it; the
    // persistent texture-image and binding caches are separate and stay at their defaults.
    set_env("PROSPER_NO_BACKEND_PIPELINE_CACHE", "1");

    // Fullscreen-triangle VS from gl_VertexIndex; no resource table. Same words the texture-sample
    // render test uses, so the sampled texel really does land on the centre pixel.
    const uint32_t vs_words[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u,
        0xBF810000u,
    };
    const std::vector<uint32_t> vert = recompile_vertex(vs_words, std::size(vs_words));

    // Solid red into MRT0. Inline consts: 0xF2 = 1.0f, 0x80 = 0.0f.
    static const uint32_t kRedPs[] = {0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
                                      0xF800180Fu, 0x03020100u, 0xBF810000u};
    const std::vector<uint32_t> red_fs = recompile_fragment(kRedPs, std::size(kRedPs), nullptr);

    // image_sample at (0.25, 0.25) -> texel (0,0) of the 2x2 texture below, which is red.
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt.resources.push_back(t); }
    const uint32_t sample_ps[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    const std::vector<uint32_t> tex_fs = recompile_fragment(sample_ps, std::size(sample_ps), &rt);
    static const uint8_t texels[2 * 2 * 4] = {
        255, 0, 0, 255,    0, 255, 0, 255,
        0, 0, 255, 255,    255, 255, 255, 255,
    };

    CHECK(!vert.empty() && !red_fs.empty() && !tex_fs.empty(),
          "fullscreen VS, solid-red PS and sampling PS all recompiled");
    if (vert.empty() || red_fs.empty() || tex_fs.empty()) { printf("FAILED (%d)\n", fails); return 1; }

    ResolvedPipelineState opaque{};
    opaque.topology = 3 /*VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST*/;
    opaque.color_write_mask = 0xF;

    // Depth-enabled state: `use_depth` keys on depth_test_enable, and `use_ds` is what creates the
    // depth/stencil image and its view at all. persist_depth_stencil stays false, so the target is
    // transient and the ds cache is not involved.
    ResolvedPipelineState with_depth = opaque;
    with_depth.depth_test_enable = true;
    with_depth.depth_write_enable = true;
    with_depth.depth_compare_op = 7;   // ALWAYS

    const float black[4] = {0, 0, 0, 1};

    // ---- Case 1 and the mild siblings that need no persistent state -------------------------
    struct Arm {
        RenderVkObjectCreateSite site;
        const char* name;            // as it must appear in the log
        const char* api;             // the Vulkan entry point the report must name
        const ResolvedPipelineState* ps;
        bool drops_pass;             // true -> no frame at all; false -> a frame with no draw in it
        const char* extra;           // an extra substring the report must carry ("" -> none)
    };
    const Arm arms[] = {
        {RenderVkObjectCreateSite::RenderPass,  "render-pass",  "vkCreateRenderPass",
         &opaque,     true,  "subpasses=1"},
        {RenderVkObjectCreateSite::Framebuffer, "framebuffer",  "vkCreateFramebuffer",
         &opaque,     true,  "extent=64x64"},
        {RenderVkObjectCreateSite::DepthStencilView, "ds-view", "vkCreateImageView",
         &with_depth, true,  ""},
        {RenderVkObjectCreateSite::ShaderModule, "shader-module", "vkCreateShaderModule",
         &opaque,     false, ""},
    };

    auto run_solid = [&](const Arm& arm, bool inject, std::vector<uint8_t>* px) -> std::string {
        char scratch[160];
        std::snprintf(scratch, sizeof scratch, "test_render_object_create_failure_%s_%d.log",
                      arm.name, inject ? 1 : 0);
        return capture_stderr(scratch, [&] {
            prosper::test::BackendDraw d;
            d.vs = vert; d.fs = red_fs; d.ps = arm.ps; d.vcount = 3;
            if (inject) prosper::test::inject_render_vk_object_create_failure_once(arm.site);
            *px = prosper::test::render_draws_rgba({d}, W, H, nullptr, black);
        });
    };

    for (const Arm& arm : arms) {
        printf("  -- site %s --\n", arm.name);
        char msg[224];

        // Control FIRST: the same configuration with nothing armed must render red and stay silent.
        // Without it, an arm that reached its site by accident -- or never reached it and printed
        // from some other site -- would be indistinguishable from one that worked.
        std::vector<uint8_t> control_px;
        const std::string control_log = run_solid(arm, false, &control_px);
        std::snprintf(msg, sizeof msg, "%s: an uninjected run renders the red draw", arm.name);
        CHECK(center_red(control_px) > 0xC0, msg);
        std::snprintf(msg, sizeof msg, "%s: an uninjected run logs no create failure", arm.name);
        CHECK(!has(control_log, "[render-object-create-failed]"), msg);

        std::vector<uint8_t> failed_px;
        const std::string failure_log = run_solid(arm, true, &failed_px);

        // THE CONTRACT. Unfixed code prints nothing at all, so all of these go red.
        std::string expect_site =
            std::string("[render-object-create-failed] site=") + arm.name + " " + arm.api + " ";
        std::snprintf(msg, sizeof msg, "%s: the failure names ITS OWN site and API", arm.name);
        CHECK(has(failure_log, expect_site.c_str()), msg);
        std::snprintf(msg, sizeof msg,
                      "%s: the failure reports the real VkResult, not just 'a null handle'",
                      arm.name);
        // VK_ERROR_OUT_OF_DEVICE_MEMORY == -2, what the injector simulates.
        CHECK(has(failure_log, "result=-2"), msg);
        if (arm.extra[0]) {
            std::snprintf(msg, sizeof msg, "%s: the failure describes the request (%s)",
                          arm.name, arm.extra);
            CHECK(has(failure_log, arm.extra), msg);
        }

        if (arm.drops_pass) {
            // NEW behaviour for all three: these sites had no failure path whatsoever before
            // #3210, so an unfixed build reaches vkCmdBeginRenderPass with an indeterminate
            // handle instead of returning here.
            std::snprintf(msg, sizeof msg, "%s: the pass is dropped, not begun with a bad handle",
                          arm.name);
            CHECK(failed_px.empty(), msg);
        } else {
            // Pre-existing control flow, stated so a later change that reported by CONTINUING past
            // the failure would be caught. This one WOULD pass on unfixed code.
            std::snprintf(msg, sizeof msg,
                          "%s: the frame still renders and only the draw is skipped", arm.name);
            CHECK(failed_px.size() == static_cast<size_t>(W) * H * 4 &&
                  center_red(failed_px) < 0x20, msg);
        }
    }

    // ---- Case 2: the sampled-texture view and sampler, transient branch ----------------------
    {
        printf("  -- sites texture-view / texture-sampler (transient) --\n");
        auto run_tex = [&](RenderVkObjectCreateSite site, const char* tag,
                           std::vector<uint8_t>* px) -> std::string {
            char scratch[160];
            std::snprintf(scratch, sizeof scratch,
                          "test_render_object_create_failure_tex_%s.log", tag);
            return capture_stderr(scratch, [&] {
                prosper::test::FrameResource r;
                r.binding = 4; r.set = 1; r.tex_rgba = texels; r.tw = 2; r.th = 2;
                prosper::test::BackendDraw d;
                d.vs = vert; d.fs = tex_fs; d.ps = &opaque; d.R = {r}; d.vcount = 3;
                if (site != RenderVkObjectCreateSite::None)
                    prosper::test::inject_render_vk_object_create_failure_once(site);
                *px = prosper::test::render_draws_rgba({d}, W, H, nullptr, black);
            });
        };

        std::vector<uint8_t> control_px;
        const std::string control_log =
            run_tex(RenderVkObjectCreateSite::None, "control", &control_px);
        CHECK(center_red(control_px) > 0xC0,
              "transient texture: an uninjected run samples texel (0,0) -> red centre");
        CHECK(!has(control_log, "[render-object-create-failed]"),
              "transient texture: an uninjected run logs no create failure");

        std::vector<uint8_t> view_px;
        const std::string view_log =
            run_tex(RenderVkObjectCreateSite::TextureView, "view", &view_px);
        CHECK(has(view_log,
                  "[render-object-create-failed] site=texture-view vkCreateImageView result=-2"),
              "texture-view: the failure names its own site, API and VkResult");
        CHECK(view_px.size() == static_cast<size_t>(W) * H * 4 && center_red(view_px) < 0x20,
              "texture-view: the draw is skipped instead of binding a null view (new behaviour)");

        std::vector<uint8_t> sampler_px;
        const std::string sampler_log =
            run_tex(RenderVkObjectCreateSite::TextureSampler, "sampler", &sampler_px);
        CHECK(has(sampler_log,
                  "[render-object-create-failed] site=texture-sampler vkCreateSampler result=-2"),
              "texture-sampler: the failure names its own site, API and VkResult");
        CHECK(sampler_px.size() == static_cast<size_t>(W) * H * 4 &&
                  center_red(sampler_px) < 0x20,
              "texture-sampler: the draw is skipped instead of binding a null sampler");
        CHECK(!has(sampler_log, "site=texture-view "),
              "texture-sampler: the VIEW create at the same site pair succeeded and stayed silent");
    }

    // ---- Case 2, persistent branch: the failure must not be CACHED and re-served -------------
    {
        printf("  -- sites texture-view-persistent / texture-sampler-persistent --\n");
        constexpr uint64_t kTexId = 0x7210000000000001ull;
        auto make_draw = [&](float lod_bias) {
            prosper::test::FrameResource r;
            r.binding = 4; r.set = 1; r.tex_rgba = texels; r.tw = 2; r.th = 2;
            r.persistent_texture_id = kTexId;
            r.lod_bias = lod_bias;
            prosper::test::BackendDraw d;
            d.vs = vert; d.fs = tex_fs; d.ps = &opaque; d.R = {r}; d.vcount = 3;
            return d;
        };
        auto render = [&](const prosper::test::BackendDraw& d, RenderVkObjectCreateSite site,
                          const char* tag, std::vector<uint8_t>* px) -> std::string {
            char scratch[160];
            std::snprintf(scratch, sizeof scratch,
                          "test_render_object_create_failure_persist_%s.log", tag);
            return capture_stderr(scratch, [&] {
                if (site != RenderVkObjectCreateSite::None)
                    prosper::test::inject_render_vk_object_create_failure_once(site);
                *px = prosper::test::render_draws_rgba({d}, W, H, nullptr, black);
            });
        };

        // Warm the persistent IMAGE. This render takes the transient binding branch (the image is
        // not in the persistent map yet), so the persistent sites are unreachable here.
        std::vector<uint8_t> warm_px;
        const std::string warm_log = render(make_draw(0.0f), RenderVkObjectCreateSite::None,
                                            "warm", &warm_px);
        const size_t warm_binding_entries =
            prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries;
        CHECK(center_red(warm_px) > 0xC0 && !has(warm_log, "[render-object-create-failed]"),
              "persistent texture: the warming render succeeds and is silent");

        // Second render of the SAME contract: persistent image hit, binding-cache miss -> the
        // persistent view/sampler creates. Fail the view.
        std::vector<uint8_t> pv_px;
        const std::string pv_log = render(make_draw(0.0f),
                                          RenderVkObjectCreateSite::TextureViewPersistent,
                                          "view", &pv_px);
        CHECK(has(pv_log, "[render-object-create-failed] site=texture-view-persistent "
                          "vkCreateImageView result=-2"),
              "texture-view-persistent: the failure names its own site, API and VkResult");
        CHECK(pv_px.size() == static_cast<size_t>(W) * H * 4 && center_red(pv_px) < 0x20,
              "texture-view-persistent: the draw is skipped rather than caching a null view");
        CHECK(prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries ==
                  warm_binding_entries,
              "failed persistent view creation adds no binding to the exact aggregate");

        // A DIFFERENT sampler contract is a different binding key, so this misses the binding
        // cache too and reaches the same pair. Fail the sampler this time.
        std::vector<uint8_t> psm_px;
        const std::string psm_log = render(make_draw(1.0f / 64.0f),
                                           RenderVkObjectCreateSite::TextureSamplerPersistent,
                                           "sampler", &psm_px);
        CHECK(has(psm_log, "[render-object-create-failed] site=texture-sampler-persistent "
                           "vkCreateSampler result=-2"),
              "texture-sampler-persistent: the failure names its own site, API and VkResult");
        CHECK(psm_px.size() == static_cast<size_t>(W) * H * 4 && center_red(psm_px) < 0x20,
              "texture-sampler-persistent: the draw is skipped rather than caching a null sampler");
        CHECK(prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries ==
                  warm_binding_entries,
              "failed persistent sampler creation adds no binding despite successful view creation");

        // THE POINT OF THIS BLOCK. Before #3210 the two persistent sites emplaced whatever the
        // failed create left behind into PersistentTextureImage::bindings, so ONE failure was
        // retained under that binding key and re-served to every later draw that matched it. Both
        // keys above are now re-created cleanly, and the second of these renders is a binding-cache
        // HIT on the entry the first one stored -- so nothing null survived either failure.
        std::vector<uint8_t> recovered_a, recovered_b;
        const std::string recovered_log_a = render(make_draw(0.0f),
                                                   RenderVkObjectCreateSite::None,
                                                   "recovered_a", &recovered_a);
        const auto after_a = prosper::test::backend_resource_reuse_stats();
        const std::string recovered_log_b = render(make_draw(0.0f),
                                                   RenderVkObjectCreateSite::None,
                                                   "recovered_b", &recovered_b);
        const auto after_b = prosper::test::backend_resource_reuse_stats();
        CHECK(center_red(recovered_a) > 0xC0 &&
                  !has(recovered_log_a, "[render-object-create-failed]"),
              "a later draw with the FAILED binding key renders -- no null was cached");
        CHECK(after_a.persistent_texture_binding_misses == 1 &&
                  after_a.persistent_texture_binding_hits == 0 &&
                  after_a.persistent_texture_binding_entries == warm_binding_entries + 1,
              "...and it had to re-create the binding, proving the failure left no entry behind");
        CHECK(after_b.persistent_texture_binding_hits == 1 &&
                  after_b.persistent_texture_binding_misses == 0 && recovered_a == recovered_b &&
                  after_b.persistent_texture_binding_entries == warm_binding_entries + 1,
              "the re-created binding is then cached and re-served byte-identically");

        std::vector<uint8_t> psm_recovered;
        const std::string psm_recovered_log = render(make_draw(1.0f / 64.0f),
                                                     RenderVkObjectCreateSite::None,
                                                     "recovered_sampler", &psm_recovered);
        CHECK(center_red(psm_recovered) > 0xC0 &&
                  !has(psm_recovered_log, "[render-object-create-failed]"),
              "the sampler-failure binding key likewise re-creates cleanly on the next draw");
        CHECK(prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries ==
                  warm_binding_entries + 2,
              "recovering the second failed key contributes exactly one additional binding");

        // Failure AFTER eviction is a different transaction: the removed old binding stays gone
        // even when its replacement cannot be created. Fill the remaining 30 contracts first.
        for (uint32_t i = 2; i < 32; ++i) {
            std::vector<uint8_t> fill_px;
            const std::string fill_log = render(make_draw(static_cast<float>(i) / 64.0f),
                RenderVkObjectCreateSite::None, "fill_binding_limit", &fill_px);
            CHECK(center_red(fill_px) > 0xC0 && !has(fill_log, "[render-object-create-failed]") &&
                      prosper::test::backend_resource_reuse_stats().persistent_texture_binding_entries ==
                          warm_binding_entries + i + 1,
                  "fill to 32 valid contracts increments the aggregate one binding at a time");
        }
        for (uint32_t i = 32; i < 34; ++i) {
            const auto site = i == 32 ? RenderVkObjectCreateSite::TextureViewPersistent
                                      : RenderVkObjectCreateSite::TextureSamplerPersistent;
            const char* site_log = i == 32 ? "site=texture-view-persistent"
                                           : "site=texture-sampler-persistent";
            const auto full_cache_draw = make_draw(static_cast<float>(i) / 64.0f);
            std::vector<uint8_t> failed_px;
            const std::string failed_log = render(full_cache_draw, site, "failure_after_eviction", &failed_px);
            const auto failed_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(has(failed_log, site_log) && center_red(failed_px) < 0x20 &&
                      failed_px.size() == static_cast<size_t>(W) * H * 4 &&
                      failed_stats.persistent_texture_binding_entries == warm_binding_entries + 31 &&
                      failed_stats.persistent_texture_binding_evictions == 1,
                  "evict then fail view/sampler creation leaves exactly 31 owned bindings");
            std::vector<uint8_t> repaired_px;
            const std::string repaired_log = render(full_cache_draw, RenderVkObjectCreateSite::None,
                "repair_after_eviction", &repaired_px);
            const auto repaired_stats = prosper::test::backend_resource_reuse_stats();
            CHECK(repaired_px == recovered_a && !has(repaired_log, "[render-object-create-failed]") &&
                      repaired_stats.persistent_texture_binding_entries == warm_binding_entries + 32 &&
                      repaired_stats.persistent_texture_binding_misses == 1 &&
                      repaired_stats.persistent_texture_binding_evictions == 0,
                  "retry inserts one binding into the vacant slot without a second eviction");
        }
    }

    // ---- Deliberately unaffected: a STORAGE image has no sampler, and must not be dropped ----
    //
    // The sampler create is skipped entirely for a storage image (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    // ignores it), so `binding.sampler` stays VK_NULL_HANDLE by design. Treating that as a failure
    // would drop every storage-image draw in the project -- exactly the false trigger this guard
    // has to avoid. The armed sampler site must therefore still be pending after this render.
    {
        printf("  -- deliberately unaffected: storage image, no sampler --\n");
        ShaderResourceTable storage_rt;
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.format = DataFormat::Unorm8; image.num_components = 4;
          image.binding = 4; image.img_dim = 1; image.width = 2; image.height = 2;
          image.sgpr_base = 8; storage_rt.resources.push_back(image); }
        const uint32_t il_template[] = {
            0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u, 0xf800000fu, 0x03020100u,
            0xbf810000u,
        };
        const std::vector<uint32_t> storage_fs =
            recompile_fragment(il_template, std::size(il_template), &storage_rt);
        CHECK(!storage_fs.empty(), "recompiled a storage-image OpImageRead fragment shader");

        uint32_t storage_texels[2 * 2 * 4];
        for (size_t i = 0; i < std::size(storage_texels); ++i) {
            const float normalized = texels[i] / 255.0f;
            std::memcpy(&storage_texels[i], &normalized, sizeof(normalized));
        }
        prosper::test::FrameResource sr;
        sr.binding = 4; sr.set = 1;
        sr.tex_rgba = reinterpret_cast<const uint8_t*>(storage_texels);
        sr.tw = 2; sr.th = 2;
        sr.texture_format = VK_FORMAT_R32G32B32A32_UINT;
        sr.is_storage_image = true;
        sr.storage_image_numeric_class = SpirvImageNumericClass::Uint;
        prosper::test::BackendDraw sd;
        sd.vs = vert; sd.fs = storage_fs; sd.ps = &opaque; sd.R = {sr}; sd.vcount = 3;

        std::vector<uint8_t> storage_px;
        prosper::test::inject_render_vk_object_create_failure_once(
            RenderVkObjectCreateSite::TextureSampler);
        const std::string storage_log = capture_stderr(
            "test_render_object_create_failure_storage.log", [&] {
                storage_px = prosper::test::render_draws_rgba({sd}, W, H, nullptr, black);
            });
        CHECK(center_red(storage_px) > 0x80,
              "a storage-image draw still renders while the sampler site is armed");
        CHECK(!has(storage_log, "[render-object-create-failed]"),
              "...and reports nothing, because no sampler is created for it");
        CHECK(prosper::test::consume_render_vk_object_create_failure(
                  RenderVkObjectCreateSite::TextureSampler),
              "...and the arm is still pending, so the site was genuinely never reached");
    }

    // ---- Site selectivity and one-shot semantics ---------------------------------------------
    //
    // Without this, an arm above could pass because SOME site printed rather than because its own
    // one did -- the same reason #3180's test carries a selectivity block.
    {
        printf("  -- selectivity and one-shot semantics --\n");
        std::vector<uint8_t> px;
        prosper::test::inject_render_vk_object_create_failure_once(
            RenderVkObjectCreateSite::DepthStencilView);
        const std::string log = capture_stderr(
            "test_render_object_create_failure_selectivity.log", [&] {
                prosper::test::BackendDraw d;
                d.vs = vert; d.fs = red_fs; d.ps = &opaque; d.vcount = 3;   // no depth -> no ds view
                px = prosper::test::render_draws_rgba({d}, W, H, nullptr, black);
            });
        CHECK(center_red(px) > 0xC0,
              "a ds-view arm does not fire at the render-pass or framebuffer site");
        CHECK(!has(log, "[render-object-create-failed]"),
              "...and prints nothing while it stays armed");
        CHECK(prosper::test::consume_render_vk_object_create_failure(
                  RenderVkObjectCreateSite::DepthStencilView),
              "the unfired arm is still pending and consumes exactly once");
        CHECK(!prosper::test::consume_render_vk_object_create_failure(
                  RenderVkObjectCreateSite::DepthStencilView),
              "...and is spent after one consume");
        CHECK(!prosper::test::consume_render_vk_object_create_failure(
                  RenderVkObjectCreateSite::None),
              "the None site never fires");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
