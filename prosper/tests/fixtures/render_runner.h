// render_runner.h — inline helper to render a single triangle (3 vertices, no vertex input) with a
// given vertex + fragment SPIR-V pair into a WxH RGBA8 image, clearing to blue first, and return the
// pixels. Used to verify recompiled shaders end-to-end (render -> readback -> pixel asserts). The
// including test links Vulkan::Vulkan.
#pragma once
#include "host/memory/guest_write_watch.hpp"   // VA->phys for the #2932 target census
#include "shared/rtt/mrt_extent.hpp"
#include <vulkan/vulkan.h>
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/host_read_barrier.hpp"   // the availability half of a readback (#2944/#3249)
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/diagnostics/geometry_probe_arming.hpp"
#include "diagnostics/env_cache.hpp"       // PROSPER_ENV_ON / _VALUE: cached reads on per-draw paths
#include "diagnostics/env_numeric.hpp"     // #3267: a typo must not silently re-size a cache
#include "gpu/state/render_state.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "host/platform/gpu_submit_gate.hpp"   // refuse submits once the frontend shuts down (#3225)
#include "shared/rtt/rtt_scale.hpp"
#include "shared/device/vulkan_device_select.hpp"
#include "shared/perf/performance_timing_gate.hpp"
#include "shared/perf/performance_timing_policy.hpp"
#include "shared/present/readback_policy.hpp"
#include "gpu/present/videoout_present.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// For the memory-aware host-buffer pool budget (render_host_physical_memory_bytes).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace prosper::test {

// Write an RGBA8 framebuffer to a binary PPM (P6) so a rendered frame is viewable as an image. PPM is
// trivially convertible to PNG (e.g. `magick in.ppm out.png`). Used by the render demos to leave
// screenshots on disk. Returns true on success.
inline bool dump_ppm(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (size_t i = 0; i < (size_t)W * H; i++) fwrite(&px[i * 4], 1, 3, f);   // RGB (drop alpha)
    fclose(f); return true;
}

// Write an RGBA8 framebuffer to a 24-bit BMP — natively viewable on Windows (double-click). BMP rows
// are bottom-up and BGR, padded to a 4-byte boundary.
inline bool dump_bmp(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    const uint32_t rowpad = (4 - (W * 3) % 4) % 4;
    const uint32_t dataSize = (W * 3 + rowpad) * H;
    const uint32_t fileSize = 54 + dataSize;
    auto u16 = [&](uint32_t v){ uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; fwrite(b, 1, 2, f); };
    auto u32 = [&](uint32_t v){ uint8_t b[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)}; fwrite(b, 1, 4, f); };
    fputc('B', f); fputc('M', f); u32(fileSize); u32(0); u32(54);                 // file header
    u32(40); u32(W); u32(H); u16(1); u16(24); u32(0); u32(dataSize); u32(2835); u32(2835); u32(0); u32(0);  // info header
    for (int y = (int)H - 1; y >= 0; y--) {                                       // bottom-up rows
        for (uint32_t x = 0; x < W; x++) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4];
            fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);                        // BGR
        }
        for (uint32_t k = 0; k < rowpad; k++) fputc(0, f);
    }
    fclose(f); return true;
}

// A texture to bind for a recompiled shader's image_sample: `rgba` points to w*h*4 RGBA8 bytes,
// bound as a COMBINED_IMAGE_SAMPLER (nearest filter, clamp) at descriptor-set 0, `binding`.
struct TexDesc { uint32_t binding; uint32_t w; uint32_t h; const uint8_t* rgba;
                 uint32_t max_aniso_ratio = 0; };   // #275: S# anisotropy ratio (0 = isotropic)

inline uint64_t hash_buffer_words(const uint32_t* words, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t word = 0; word < count; ++word) {
        hash ^= words[word];
        hash *= 1099511628211ull;
    }
    return hash;
}

// One resource for the general N-binding path (render_triangle_rgba's `gres`): a storage buffer
// (dwords non-empty, tex_rgba null), combined image sampler, or storage image at `binding`. Lets a
// real game shader that declares several buffers and images have each bound distinctly.
struct FrameResource {
    uint32_t binding = 0;
    uint32_t set = 0;               // descriptor set: VS resources -> 0, PS resources -> 1 (they must not
                                    // share a set — both stages number bindings from 2, so one set would
                                    // collide binding 2/3 between stages and make the layout invalid).
    std::vector<uint32_t> dwords;   // owned storage-buffer contents (empty -> a 1-dword zero buffer)
    // A production callback may point directly at a fully readable immutable guest/capture range.
    // The backend consumes and uploads this view synchronously; an explicit submission batch retains
    // only the completed Vulkan upload, never this pointer. Tests/replays may use the same contract
    // when their backing outlives render_draws_rgba().
    const uint32_t* dwords_view = nullptr;
    size_t dwords_view_count = 0;
    // Exact guest resource identity for a storage buffer. The backend may share an immutable upload
    // within one synchronous call only when both this identity and the complete captured bytes match;
    // zero keeps synthetic/replay resources conservatively distinct.
    uint64_t buffer_identity = 0;
    // Backend-owned PS5 GDS storage. Unlike guest/capture buffers this is one persistent,
    // zero-initialized 64 KiB allocation shared across ordered render calls.
    bool is_internal_gds = false;
    const uint8_t* tex_rgba = nullptr;   // non-null => a texture; then tw/th are its dimensions
    // Readable byte span behind tex_rgba. Zero preserves the historical single-sample fixture
    // contract; multisample plane uploads require an explicit full span and reject before Vulkan.
    size_t tex_byte_size = 0;
    // Optional immutable owner for borrowed frontend pixels. Keeping this beside the raw pointer
    // makes FrameResource copies/moves retain the source through synchronous backend upload setup.
    std::shared_ptr<const std::vector<uint8_t>> tex_rgba_owner;
    // A DCC fast-clear is already a complete image description. Keep it compact until command
    // recording and initialize the full-size Vulkan image with vkCmdClearColorImage instead of
    // allocating, filling, and uploading tens of MiB of identical CPU pixels.
    bool has_uniform_color = false;
    std::array<float, 4> uniform_color{};
    uint32_t tw = 0, th = 0, td = 1;
    // Guest sample count. 2D_MSAA is uploaded as a single-sample 2D-array with one plane/layer per
    // sample; the Vulkan image itself always keeps VK_SAMPLE_COUNT_1_BIT.
    uint32_t sample_count = 1;
    // T#-declared mip-chain length (#1272). 1 (default) = single-level upload, the historical
    // behavior. >1 lets the backend generate a box-filtered chain — bounded by this declared count —
    // for plain-2D RGBA8 sampled textures, so minification stops point-sampling through dense art.
    uint32_t declared_mip_levels = 1;
    uint32_t img_dim = 1;             // ShaderResource/MIMG dim (1=2D, 2=3D); depth-1 3D stays 3D
    // #325: the guest T# is one prosper treats as a layered array, as decided by
    // guest_texture_is_uploaded_array(). The view type keys on this IN ADDITION to
    // `sample_count > 1`, which stays for the MSAA plane array -- that shape has no guest_array flag
    // and must keep its own arm. What this adds is the case the count cannot express: the
    // recompiler declares OpTypeImage Arrayed from the same predicate and cannot see how many layers
    // the uploader actually decoded -- a footprint cap, an RTT hit or a DCC fast-clear can all leave
    // the count at 1 -- so a view keyed on the count ALONE would silently become 2D under an Arrayed
    // declaration. A one-layer 2D_ARRAY view is legal and samples layer 0.
    bool guest_array = false;
    // Renderer-owned RTTs keep their native format between producer and consumer. Guest-backed
    // textures still arrive through the existing RGBA8 decoder unless explicitly tagged otherwise.
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    // StorageImage is a distinct Vulkan descriptor contract: no sampler, STORAGE usage, and GENERAL
    // layout. Keep the flag independent of tex_rgba because both sampled and storage images upload
    // decoded pixels through that pointer.
    bool is_storage_image = false;
    // Reflected OpTypeImage Sampled Type versus the materialized Vulkan view. Storage descriptors
    // without a proven class, or whose frontend could not produce the requested representation,
    // fail before Vulkan command recording instead of relying on undefined format compatibility.
    prosper::gpu::SpirvImageNumericClass storage_image_numeric_class =
        prosper::gpu::SpirvImageNumericClass::Unknown;
    bool storage_image_contract_valid = true;
    // Writable graphics storage images must publish their final linear texels before this backend call
    // returns. The live frontend uses this to restore the guest's linear/tiled layout and notify the
    // normal guest-write invalidation path, so later graphics, sampled aliases, compute, DMA, and CPU
    // consumers all observe the image side effect. An empty callback keeps read-only/test fixtures on
    // the ordinary asynchronous upload path.
    std::function<void(const uint8_t*, size_t)> storage_image_writeback;
    // Sampler state (Texture only). Defaults = LINEAR + clamp-to-edge — the harness's prior fixed
    // sampler — so render tests that build FrameResources directly stay byte-identical. The live path
    // fills these from the decoded S# (shader_resources.hpp). filter: 0=nearest, 1=linear; addr = Gen5
    // SQ_TEX CLAMP enum (0=wrap, 1=mirror, 2=clamp-last-texel, 6/7=border).
    uint32_t mag_filter = 1, min_filter = 1, mip_filter = 0;
    uint32_t addr_uvw[3] = {2, 2, 2};
    // Remaining S# sampler fields (#262). Defaults reproduce the current Vulkan sampler exactly (border
    // transparent-black, LOD 0..0, no bias), so FrameResources built directly by tests are byte-identical.
    uint32_t border_color_type = 0;
    float    min_lod = 0.0f, max_lod = 0.0f, lod_bias = 0.0f;
    // Anisotropy ratio (S# WORD0[11:9]; maxAnisotropy = 1<<ratio). 0 = isotropic (the default) -> the
    // sampler is byte-identical to before, so tests building FrameResources directly are unaffected (#275).
    uint32_t max_aniso_ratio = 0;
    // T# DST_SEL channel swizzle (SQ_SEL per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Default = identity
    // (R,G,B,A) == a no-op VkComponentMapping, so tests that build FrameResources directly are unchanged.
    uint32_t swizzle[4] = {4, 5, 6, 7};
    // Original guest CB_COLOR format when these pixels came from a renderer-owned target. The
    // backend stores color attachments in `backend_color_format()`'s canonical component order;
    // a later T# still describes the guest allocation's component order. Keeping the producer
    // format lets the sampled view compose those two mappings instead of applying the storage-order
    // swap a second time. Undefined means ordinary guest texture / no such canonicalization.
    VkFormat render_target_guest_format = VK_FORMAT_UNDEFINED;
    // Non-zero only when the frontend has revalidated the complete guest source backing and can
    // prove these decoded pixels are the same content version as a prior callback. The Vulkan
    // backend may retain an uploaded image under this ID; zero remains callback-local/transient.
    uint64_t persistent_texture_id = 0;
    // A non-zero version lets mutable, exactly-validated guest textures retain the allocation named
    // above while replacing its contents. Equal versions skip upload; a newer version records one
    // ordered refresh into the existing image. Zero preserves the historical immutable-ID contract.
    uint64_t persistent_texture_version = 0;
    // Non-zero identifies a color target retained by an earlier backend call. When that exact
    // target is available, bind its GPU image directly instead of uploading `tex_rgba` again.
    // `tex_rgba` remains an optional conservative fallback for an invalidated/missing target.
    uint64_t persistent_render_target_id = 0;
    // A guest mip chain is rendered as one independent persistent color target per level because
    // CB_COLOR names a single mip view at a time. A later T# names the complete allocation as one
    // sampled image. The frontend records the renderer-owned level identities here so the backend
    // can assemble them into one Vulkan mip image instead of binding level zero and silently
    // discarding every smaller level. Zero entries are missing levels; the backend derives those
    // from the previous available destination level. The count is the requested chain length.
    std::array<uint64_t, 16> persistent_render_target_mip_ids{};
    uint32_t persistent_render_target_mip_count = 0;
    // Non-zero identifies a persistent depth/stencil surface whose DEPTH plane this resource
    // samples (a shadow map / depth pyramid tap, #1275). The backend binds the retained Vulkan
    // depth image directly — prosper never writes rendered depth back to guest memory, so there
    // is no CPU fallback; an invalidated/missing surface falls through to the guest-byte decode.
    uint64_t persistent_depth_target_id = 0;
    // Exact typed-storage result borrowed from the live compute cache. The image and device handles
    // are opaque here only at the frontend boundary; this Vulkan backend verifies the device,
    // transitions from `borrowed_image_layout` for sampling, restores that layout afterward, and
    // retains the lease until the submission has completed.
    void* borrowed_compute_image = nullptr;
    void* borrowed_compute_device = nullptr;
    uint32_t borrowed_compute_image_layout = 0;
    // A 2D-array compute result cannot be bound directly to the vertically-stacked 2D image expected
    // by the cube lowering. Non-zero asks the backend to copy these source layers into consecutive
    // vertical regions of a sampled image without visiting host/guest memory.
    uint32_t borrowed_compute_vertical_stack_layers = 0;
    std::shared_ptr<void> borrowed_compute_image_lease;
    const uint32_t* buffer_words_data() const {
        return dwords_view && dwords_view_count
            ? dwords_view : (dwords.empty() ? nullptr : dwords.data());
    }
    size_t buffer_word_count() const {
        return dwords_view && dwords_view_count ? dwords_view_count : dwords.size();
    }
    bool is_texture() const {
        return tex_rgba != nullptr || has_uniform_color || persistent_render_target_id != 0 ||
               persistent_depth_target_id != 0 || borrowed_compute_image != nullptr;
    }
    // Per-entry payloads for a RUNTIME-SELECTED descriptor array (#2412 stage 5). Empty -- the only
    // state any producer creates today -- means an ordinary single descriptor and every path below
    // behaves exactly as before.
    //
    // Arity is DERIVED from this vector rather than carried in a separate count field, deliberately:
    // two sources of truth for "how many descriptors" is precisely the disagreement that makes a
    // layout declare N while a write supplies M, which is a validation error at bind time and a wrong
    // shader read if validation is off. One vector, one length, no way to disagree.
    std::vector<std::vector<uint32_t>> table_entries;
    // The number of descriptors this binding DECLARES: 1 ordinary, N for a table-indexed array.
    // This is the guest's intent, not what any Vulkan object ends up carrying -- for that, use
    // `written_descriptor_count()` below.
    //
    // This comment used to say the pool, the layout and the write "all three must agree, so all
    // three call this". They must agree; they did not all call this. The pool counts a texture as
    // ONE (`is_texture()` branches to `++storage_images` / `++sampled_images` regardless of arity)
    // and the texture and internal-GDS writes each supply ONE, while the layout took this value for
    // every class -- so the invariant the comment asserted was the one thing not being maintained.
    // #2477.
    uint32_t descriptor_arity() const {
        return table_entries.empty() ? 1u : static_cast<uint32_t>(table_entries.size());
    }
    // The number of descriptors the WRITE path actually supplies, and therefore what the POOL and the
    // LAYOUT must both declare (#2477). Only the storage-buffer path loops over `table_entries`; the
    // texture / storage-image write and the internal-GDS write each supply exactly one descriptor.
    //
    // Taking `descriptor_arity()` in the layout while the write supplied one produced precisely the
    // layout-declares-N / write-supplies-M disagreement the `table_entries` comment above exists to
    // prevent: the layout would declare N, elements 1..N-1 would never be written, and binding the
    // set would read undefined descriptors. The two sites are ~500 lines apart, so nothing local
    // showed it.
    //
    // This is a fail-visible GUARD, not array support for those classes: it keeps the three sites in
    // agreement and says so loudly if a producer ever populates `table_entries` on a class whose
    // write cannot honour it. Unreachable today -- nothing sets `table_entries` on a texture or the
    // internal GDS buffer -- which is why it is a guard rather than a fix, and why it needs no
    // producer to be worth having. `CONFIDENCE: HIGH` that silent truncation is the wrong direction
    // here: a layout that quietly declares fewer descriptors than the guest asked for is the
    // under-reporting failure, and it is the one nobody files.
    uint32_t written_descriptor_count() const {
        return is_texture() || is_internal_gds ? 1u : descriptor_arity();
    }
};

// Optional color-target contract for the live backend. A non-zero ID gives the target a stable
// guest identity across calls. Existing contents are loaded only when both `load_existing` and a
// valid matching image are present. `readback=false` leaves the completed result GPU-resident and
// deliberately returns an empty CPU vector.
struct BackendColorTarget {
    uint64_t persistent_id = 0;
    bool load_existing = true;
    bool readback = true;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // MRT1 has an independent guest identity and lifetime. Keeping its contract beside color0 lets
    // paired passes retain both attachments without changing the established single-target callers.
    uint64_t persistent_id1 = 0;
    bool load_existing1 = true;
    bool readback1 = true;
    VkFormat format1 = VK_FORMAT_UNDEFINED;
    // Slots 2..7. The named fields above predate the complete array and remain the contract for
    // slots 0 and 1; these carry the same two properties for the rest.
    //
    // Without them every slot above 1 was a TRANSIENT image created per backend call and cleared
    // with LOAD_OP_CLEAR, so returning to the same allocation in a later render group erased what an
    // earlier group had drawn there. That was unreachable while the fragment recompiler could only
    // export MRT0/MRT1; the moment it could export five, GTA V's G-buffer -- which accumulates
    // across roughly sixteen pass groups per frame -- kept only the last group's slots 2..4.
    std::array<uint64_t, prosper::gpu::kColorTargetCount> persistent_id_slots{};
    std::array<bool, prosper::gpu::kColorTargetCount> load_existing_slots{};
    // Per-slot twin of `readback`/`readback1`. Defaults to true so every existing caller keeps its
    // pixels; the depth-feedback splitter turns it off for non-final segments, which otherwise pay a
    // full-extent copy per slot per segment purely because `color_count > 2` forces a readback.
    std::array<bool, prosper::gpu::kColorTargetCount> readback_slots{
        true, true, true, true, true, true, true, true};
    // Initial contents for slots 2..7, the per-slot twin of `seed_rgba`/`seed_rgba1`. A depth
    // feedback split renders a logical pass in several physical passes; when the slots are not
    // persistent (the caller passed no identities, e.g. every path that runs with live GPU targets
    // disabled) the ONLY way a later segment sees an earlier one's pixels is to be seeded with them.
    // Without it the final segment created fresh transient attachments and cleared away everything
    // the earlier segments drew into MRT2+.
    std::array<const uint8_t*, prosper::gpu::kColorTargetCount> seed_slots{};
    // Slot 1's seed when the caller receives it through BackendMrtOutputs instead of `out_rgba1`.
    // The live renderer uses that API, so on a split its slot-1 readback landed in
    // `mrt_outputs.colors[1]` while the wrapper only carried a seed forward inside `if (out_rgba1)`
    // -- losing MRT1 on every non-final segment. The claim that "slots 0 and 1 already carry" was
    // true only of the legacy explicit-out_rgba1 API.
    const uint8_t* seed_rgba1_slot = nullptr;
};

// Optional complete-MRT readback contract. `color_count` is the active Vulkan attachment prefix
// (1..8); slots 1..count-1 receive their native-format bytes after rendering. Slot 0 remains the
// function's return value for compatibility with every existing caller.
struct BackendMrtOutputs {
    uint32_t color_count = 1;
    std::array<std::vector<uint8_t>, prosper::gpu::kColorTargetCount> colors;
};

inline VkFormat backend_color_format(VkFormat format) {
    if (format == VK_FORMAT_R16_SFLOAT ||
        format == VK_FORMAT_R16G16_SFLOAT ||
        format == VK_FORMAT_R16G16B16A16_SFLOAT ||
        format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        return format;
    if (format == VK_FORMAT_R8_UNORM)
        return VK_FORMAT_R8_UNORM;
    if (format == VK_FORMAT_R8_UINT)
        return VK_FORMAT_R8_UINT;
    if (format == VK_FORMAT_R8G8B8A8_UINT)
        return VK_FORMAT_R8G8B8A8_UINT;
    if (format == VK_FORMAT_R8G8_UNORM)
        return VK_FORMAT_R8G8_UNORM;
    if (format == VK_FORMAT_R32_UINT)
        return VK_FORMAT_R32_UINT;
    if (format == VK_FORMAT_R32G32B32A32_UINT)
        return VK_FORMAT_R32G32B32A32_UINT;
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (format == VK_FORMAT_R32_SFLOAT)
        return VK_FORMAT_R32_SFLOAT;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

inline std::array<uint32_t, 4> backend_sampled_component_swizzle(
    const FrameResource& resource) {
    std::array<uint32_t, 4> result{
        resource.swizzle[0], resource.swizzle[1],
        resource.swizzle[2], resource.swizzle[3]};
    // CB_COLOR ALT stores COLOR_8_8_8_8 as BGRA, while prosper deliberately materializes that
    // attachment as canonical RGBA8. SQ_IMG_RSRC DST_SEL is expressed against the guest's format
    // components, so translate R<->B selectors into the canonical host image's component space.
    // Constants, G/A, ordinary textures, and targets whose storage order was already RGBA remain
    // unchanged. This is composition, not suppression: semantic permutations still survive.
    if ((resource.render_target_guest_format == VK_FORMAT_B8G8R8A8_UNORM ||
         resource.render_target_guest_format == VK_FORMAT_B8G8R8A8_SRGB) &&
        backend_color_format(resource.render_target_guest_format) ==
            VK_FORMAT_R8G8B8A8_UNORM) {
        for (uint32_t& selector : result) {
            if (selector == 4u) selector = 6u;
            else if (selector == 6u) selector = 4u;
        }
    }
    return result;
}

inline uint32_t backend_color_bytes_per_pixel(VkFormat format) {
    format = backend_color_format(format);
    if (format == VK_FORMAT_R32G32B32A32_UINT ||
        format == VK_FORMAT_R32G32B32A32_SFLOAT) return 16u;
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT) return 8u;
    if (format == VK_FORMAT_R16G16_SFLOAT) return 4u;
    if (format == VK_FORMAT_R16_SFLOAT) return 2u;
    if (format == VK_FORMAT_R8_UNORM || format == VK_FORMAT_R8_UINT) return 1u;
    if (format == VK_FORMAT_R8G8_UNORM) return 2u;
    return 4u;
}

inline prosper::gpu::SpirvImageNumericClass backend_image_numeric_class(VkFormat format) {
    using NumericClass = prosper::gpu::SpirvImageNumericClass;
    switch (backend_color_format(format)) {
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT:
            return NumericClass::Uint;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            return NumericClass::Float;
        default:
            return NumericClass::Unknown;
    }
}

inline bool backend_storage_image_numeric_contract_valid(const FrameResource& resource) {
    if (!resource.is_storage_image) return true;
    return resource.storage_image_contract_valid &&
           resource.storage_image_numeric_class !=
               prosper::gpu::SpirvImageNumericClass::Unknown &&
           resource.storage_image_numeric_class ==
               backend_image_numeric_class(resource.texture_format);
}

// The first exact guest-MSAA contract is intentionally narrow. Ordinary resources retain their
// historical implicit byte-span behavior; a 2D_MSAA plane array must prove all four complete R32F
// planes are readable before any Vulkan object or memcpy is attempted.
// A pragmatic ceiling, NOT a portability guarantee -- an earlier revision of this comment claimed
// 2048 was "Vulkan's guaranteed minimum for maxImageArrayLayers" and that is wrong. The Core
// Required Limits table gives **256**; 2048 is the Roadmap 2022 / Vulkan 1.4 figure, and this
// backend requests VK_API_VERSION_1_1 (see vkCreateInstance below), so the guarantee that actually
// applies here is 256. This box's RADV reports 8192.
//
// So what this bound does is keep an absurd layer count away from vkCreateImage, whose result the
// upload path discards (#3045) -- an over-large arrayLayers yields VK_NULL_HANDLE and is passed
// straight to vkGetImageMemoryRequirements. It does NOT prove creatability on an arbitrary device;
// a device reporting the Core minimum of 256 can still reject a 512-layer image, and it will do so
// through that same unchecked path until #3045 lands. Querying the real limit is the correct fix
// and belongs with #3045, since this predicate has no device handle.
inline constexpr uint32_t kBackendMaxArrayLayers = 2048u;

// A guest 2D_ARRAY is the same shape as the MSAA plane array with a different provenance: N
// ordinary color layers laid out one after another, carried through the SAME `sample_count`
// channel, and owed the same proof that every layer is readable before any Vulkan object or memcpy
// exists. Everything downstream is already layer-generic -- the staging buffer sizes itself
// `tw*th*td*sample_count*bpp`, the image takes `arrayLayers = sample_count`, and the view selects
// VK_IMAGE_VIEW_TYPE_2D_ARRAY above 1 -- so once a frontend publishes a layer count, this predicate
// is what stands between it and the upload (#325). Mip generation is already excluded for
// `sample_count > 1`, which is what stops a generated chain bleeding across layer boundaries.
inline bool backend_texture_array_span_valid(const FrameResource& resource) {
    if (resource.img_dim != 5u || resource.td != 1u || resource.is_storage_image ||
        !resource.tex_rgba || !resource.tw || !resource.th || resource.sample_count < 2u ||
        resource.sample_count > kBackendMaxArrayLayers)
        return false;
    const uint32_t bpp = backend_color_bytes_per_pixel(resource.texture_format);
    if (!bpp) return false;
    const size_t row_bytes = static_cast<size_t>(resource.tw) * bpp;
    if (row_bytes / bpp != resource.tw) return false;
    if (resource.th > SIZE_MAX / row_bytes) return false;
    const size_t layer_bytes = row_bytes * resource.th;
    if (resource.sample_count > SIZE_MAX / layer_bytes) return false;
    return resource.tex_byte_size >= layer_bytes * resource.sample_count;
}

inline bool backend_texture_plane_span_valid(const FrameResource& resource) {
    if (resource.sample_count == 1u) return true;
    if (resource.img_dim == 5u) return backend_texture_array_span_valid(resource);
    if (resource.sample_count != 4u || resource.img_dim != 6u || resource.td != 1u ||
        resource.declared_mip_levels != 1u || resource.is_storage_image ||
        backend_color_format(resource.texture_format) != VK_FORMAT_R32_SFLOAT ||
        !resource.tex_rgba || !resource.tw || !resource.th)
        return false;
    const size_t row_bytes = static_cast<size_t>(resource.tw) * sizeof(float);
    if (resource.tw && row_bytes / resource.tw != sizeof(float) ||
        resource.th > SIZE_MAX / row_bytes)
        return false;
    const size_t plane_bytes = row_bytes * resource.th;
    if (resource.sample_count > SIZE_MAX / plane_bytes) return false;
    const size_t required = plane_bytes * resource.sample_count;
    return resource.tex_byte_size >= required;
}

struct BackendColorTargetStats {
    uint64_t writes = 0;
    uint64_t write_hits = 0;
    uint64_t sampled_hits = 0;
    uint64_t readbacks = 0;
    uint64_t cached_bytes = 0;
    uint64_t cached_entries = 0;
};

inline BackendColorTargetStats& backend_color_target_stats_storage() {
    static thread_local BackendColorTargetStats stats;
    return stats;
}

inline BackendColorTargetStats backend_color_target_stats() {
    return backend_color_target_stats_storage();
}

// Returns native target bytes (RGBA8 by default, RGBA16F when requested), or {} on failure. When `ps`
// is non-null, the pipeline's fixed-function state (topology, blend, color write mask) is taken from
// the resolved RDNA2 render-state — this is how the back-half realizes a GpuState as a real VkPipeline.
//
// When `vbuf` and/or `cbuf` are non-null, a descriptor set is bound with the constant buffer at
// binding 2 and the vertex buffer at binding 3 (as storage buffers), matching declare_cbufs() in the
// recompiler — this is how a table-recompiled vertex shader fetches real vertex/constant data. Each is
// the raw dword contents of the buffer to bind. When both are null, the pipeline layout is empty and
// the color-only path above is taken unchanged.
//
// When `tex` is non-null, its RGBA8 texels are uploaded to a sampled VkImage and bound as a combined
// image sampler at tex->binding — how a recompiled pixel shader's image_sample reaches a real texture.
// One draw for the multi-draw backend: recompiled VS+PS SPIR-V, its resolved fixed-function state, its
// set-tagged resources, vertex count, and instance count. render_draws_rgba preserves every draw in
// order, normally in one render pass; an attached-depth write/sample transition is the narrow case
// that requires an ordered pass boundary. render_triangle_rgba is a thin single-draw wrapper (below).
struct BackendDraw {
    std::vector<uint32_t> vs, gs, fs;
    prosper::gpu::SharedShaderWords vs_shared, fs_shared;
    uint64_t vs_identity = 0, fs_identity = 0;
    // Live-title authority for GTA V's reviewed WaveAny-only fragment route. Tests, replay and all
    // other titles leave this false, preserving the strict exact-wave admission contract.
    bool allow_native_fragment_vote_width = false;
    // Stable semantic draw ID from DrawItem::draw_index. Diagnostics must not use this backend
    // vector's pass-local offset: target/compute splitting can make that offset differ per pass.
    uint64_t draw_index = UINT64_MAX;
    // Global PM4 ordinal. Unlike draw_index this is comparable with interleaved compute operations
    // and therefore identifies which retained attachment layer is newer than a compute image.
    uint64_t command_order = 0;
    const prosper::gpu::ResolvedPipelineState* ps = nullptr;   // null -> triangle-list, write RGBA, no depth
    std::vector<FrameResource> R;                              // set-tagged resources (empty -> no descriptors)
    uint32_t vcount = 3;
    uint32_t instance_count = 1;
    int32_t vertex_offset = 0;
    // Indexed draw: 32-bit index data (the executor widens guest 16-bit indices). Non-empty -> the draw
    // is recorded as vkCmdBindIndexBuffer + vkCmdDrawIndexed(indices.size()), so gl_VertexIndex is the
    // fetched index — exactly what the recompiled VS's storage-buffer vertex fetch expects. Empty ->
    // plain vkCmdDraw(vcount). Both paths preserve instance_count.
    std::vector<uint32_t> indices;

    // Same precedence — and same trap — as DrawItem (#1434): a shared value WINS, so assigning
    // `vs`/`fs` on a draw that already carries one silently keeps the ORIGINAL shader. Substitute
    // through set_vs()/set_fs(), which also clear the cache identity so the persistent pipeline
    // cache compares the new words instead of hitting the stale entry. `gs` has no shared form; if
    // one is ever added it needs the same accessor/setter pair.
    const std::vector<uint32_t>& vs_words() const { return vs_shared ? *vs_shared : vs; }
    const std::vector<uint32_t>& gs_words() const { return gs; }
    const std::vector<uint32_t>& fs_words() const { return fs_shared ? *fs_shared : fs; }

    void set_vs(std::vector<uint32_t> words) {
        vs = std::move(words); vs_shared.reset(); vs_identity = 0;
    }
    void set_fs(std::vector<uint32_t> words) {
        fs = std::move(words); fs_shared.reset(); fs_identity = 0;
    }
};

struct BackendTextureUploadStats {
    size_t references = 0;
    size_t unique_uploads = 0;
    uint64_t upload_bytes = 0;
    size_t persistent_hits = 0;
    size_t persistent_misses = 0;
    uint64_t persistent_cached_bytes = 0;
};

// thread_local like every other per-call stats storage in this file (#2953). It was the one
// exception: a plain static written inside `render_draw_pass_rgba` and read by the caller after the
// call returns, so a second rendering thread could clobber it between the write and the read -- a
// data race on a non-atomic object, and silently wrong numbers even where it did not tear. Every
// reader in the tree reads it on the thread that just rendered, so per-thread storage is what the
// contract already was.
inline BackendTextureUploadStats& backend_texture_upload_stats_storage() {
    static thread_local BackendTextureUploadStats stats;
    return stats;
}

inline BackendTextureUploadStats backend_texture_upload_stats() {
    return backend_texture_upload_stats_storage();
}

// Per-call Vulkan objects that can be shared only when their complete immutable contracts match.
// These counters make the optimization auditable without exposing backend implementation details to
// the live renderer.
struct BackendResourceReuseStats {
    size_t buffer_references = 0;
    size_t unique_buffers = 0;
    size_t buffer_upload_fallbacks = 0;
    size_t texture_binding_references = 0;
    size_t unique_texture_bindings = 0;
    size_t descriptor_set_layout_references = 0;
    size_t unique_descriptor_set_layouts = 0;
    size_t pipeline_layout_references = 0;
    size_t unique_pipeline_layouts = 0;
    size_t descriptor_pools = 0;
    size_t persistent_pipeline_layout_hits = 0;
    size_t persistent_pipeline_layout_misses = 0;
    size_t persistent_pipeline_layout_entries = 0;
    size_t persistent_pipeline_layout_evictions = 0;
    size_t persistent_texture_binding_hits = 0;
    size_t persistent_texture_binding_misses = 0;
    size_t persistent_texture_binding_entries = 0;
    size_t persistent_texture_binding_evictions = 0;
    // Buffer content-hash economics (#1268): the shared-buffer dedup key embeds a full content
    // hash, which profiled as the dominant CPU term on Blue Prince's ~4,000-draw submits. These
    // count, per call like the rest of this struct, how much hashing actually ran and how much
    // was avoided (repeat references resolved by the per-call memo; unique-tag keys that cannot
    // match anything skip hashing entirely).
    uint64_t buffer_hash_calls = 0;
    uint64_t buffer_hash_dwords = 0;
    uint64_t buffer_hash_skipped_unique = 0;
    uint64_t buffer_hash_skipped_large = 0;
    uint64_t buffer_ref_memo_hits = 0;
    // Reference share is not byte share, and the cost here is BYTES. skipped_large at 9.3% of
    // references says nothing on its own about how much of the copy volume those buffers are --
    // they are large by definition, so their byte share is necessarily higher, but "higher" is not
    // a number. These two make the conclusion readable instead of derivable (#2245 review).
    uint64_t buffer_skipped_large_dwords = 0;   // payload that bypasses content dedup by size
    uint64_t buffer_upload_bytes = 0;           // bytes actually memcpy'd into mapped staging
};

inline BackendResourceReuseStats& backend_resource_reuse_stats_storage() {
    static thread_local BackendResourceReuseStats stats;
    return stats;
}

inline BackendResourceReuseStats backend_resource_reuse_stats() {
    return backend_resource_reuse_stats_storage();
}

// Cumulative across calls (the per-call struct above is reset at every render_draws_rgba entry, which
// tests rely on). PROSPER_HASH_STATS=1 prints these every few seconds from the render path so a live
// route shows the hashing economics without a debugger (#1268).
struct BackendHashStatsTotals {
    uint64_t references = 0;
    uint64_t memo_hits = 0;
    uint64_t hash_calls = 0;
    uint64_t hash_dwords = 0;
    uint64_t skipped_unique = 0;
    uint64_t skipped_large = 0;
    uint64_t unique_buffers = 0;
};
inline BackendHashStatsTotals& backend_hash_stats_totals() {
    static thread_local BackendHashStatsTotals totals;
    return totals;
}
inline void maybe_report_hash_stats() {
    static const bool on = getenv("PROSPER_HASH_STATS") != nullptr;
    if (!on) return;
    static thread_local auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(5)) return;
    last = now;
    const BackendHashStatsTotals& t = backend_hash_stats_totals();
    fprintf(stderr,
            "[hash-stats] refs=%llu memo_hits=%llu hash_calls=%llu hashed_MiB=%.1f "
            "skipped_unique=%llu skipped_large=%llu unique_buffers=%llu\n",
            (unsigned long long)t.references, (unsigned long long)t.memo_hits,
            (unsigned long long)t.hash_calls, (double)t.hash_dwords * 4.0 / (1024.0 * 1024.0),
            (unsigned long long)t.skipped_unique, (unsigned long long)t.skipped_large,
            (unsigned long long)t.unique_buffers);
}

struct BackendPipelineCacheStats {
    uint64_t references = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;
    uint64_t entries = 0;
    uint64_t evictions = 0;
};

inline BackendPipelineCacheStats& backend_pipeline_cache_stats_storage() {
    static thread_local BackendPipelineCacheStats stats;
    return stats;
}

inline BackendPipelineCacheStats backend_pipeline_cache_stats() {
    return backend_pipeline_cache_stats_storage();
}

struct PersistentPipelineKey {
    static constexpr size_t kInlineWords = 64;
    std::array<uint32_t, kInlineWords> inline_words{};
    std::vector<uint32_t> overflow_words;
    uint32_t word_count = 0;
    uint64_t hash = 1469598103934665603ull;

    bool operator==(const PersistentPipelineKey& other) const {
        if (hash != other.hash || word_count != other.word_count) return false;
        const size_t inline_count = std::min<size_t>(word_count, kInlineWords);
        return std::equal(inline_words.begin(), inline_words.begin() + inline_count,
                          other.inline_words.begin()) &&
               overflow_words == other.overflow_words;
    }
};

struct PersistentPipelineKeyHash {
    size_t operator()(const PersistentPipelineKey& key) const {
        return static_cast<size_t>(key.hash);
    }
};

struct PersistentPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint64_t last_use = 0;
};

// --- The backend's persistent-resource domain lock (#2953) --------------------------------------
//
// The process-lifetime caches this backend owns -- the persistent pipeline, pipeline-layout,
// colour-target, depth/stencil and texture caches, their byte totals and their generation counters
// -- are ONE domain with ONE owner at a time, and `render_draw_pass_rgba` is the critical section.
// There is no finer granularity that would be honest: that function looks an entry up, keeps the
// iterator or the reference across dozens of Vulkan calls, mutates through it, evicts through it,
// and updates a byte total by read-modify-write. A per-container lock taken and released around
// each individual access would describe none of that while reading as if it did.
//
// WHY IT EXISTS. #2953 recorded a host SIGSEGV on a guest job thread inside
// `persistent_texture_images.find()` -- libstdc++'s `_M_equals`, reached through
// `_M_find_before_node`, dereferencing a null node. A `find()` racing another thread's rehash reads
// torn bucket pointers, which is that crash shape; #278 already fixed the same shape once, one
// layer up, on the folded GpuState.
//
// WHY IT IS NOT SOMEBODY ELSE'S JOB. On the live route this backend is reached only from
// `agc_driver_submit_dcb` / `submit_dcb_stream`, which hold `g_agc_state_mu` (#278), so that route
// is already serialised -- by an invariant that lives in `src/hle/graphics/hle_agc.cpp`, a
// different layer, in a library this header is not compiled against. `gpu_replay`, `boot_trace` and
// every Vulkan test include `render_runner.h` and link no HLE at all. Nothing here stated the
// invariant, nothing enforced it, and nothing could detect its violation. State that owns
// process-lifetime Vulkan objects carries its own contract.
//
// COST. One uncontended mutex acquire/release plus four relaxed atomics per `render_draw_pass_rgba`
// call. That is a per-SUBMIT cost, not a per-draw one, against a call that records a command
// buffer, submits it and waits on a fence -- tens of nanoseconds against tens of microseconds at
// the very least. On the live route it is uncontended by construction, for the reason above.
//
// SCOPE. The guard makes `render_draw_pass_rgba` mutually exclusive with itself, which covers every
// static that function owns. It does NOT cover the colour-target and depth/stencil caches' OTHER
// entry points -- `invalidate_persistent_color_target*`, `readback_persistent_color_target`,
// `snapshot_persistent_ds_images` and the frontend's direct iteration of both caches -- which are
// reachable without this lock and are tracked in #3240. Nor is the multi-segment loop in
// `render_draws_rgba` atomic: the guard is released between segments, which is the same granularity
// two consecutive submits already have.
//
// IT MEASURES ITS OWN PREMISE. `in_flight` is incremented BEFORE the mutex acquire, so a blocked
// thread is counted: `backend_persistent_resource_peak_in_flight()` reports how many threads have
// ever wanted this domain at once, which is precisely the question #2953 left open and could not
// settle by reading code. A run ending at 1 has shown the single-thread assumption held for that
// run; above 1 has shown it false, and says so once on stderr.
// `backend_persistent_resource_overlaps()` counts threads observed inside the critical section
// simultaneously -- always 0 while this guard is taken, and non-zero the moment it is not, which is
// what the regression test asserts on.
inline std::mutex& backend_persistent_resource_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::atomic<int>& backend_persistent_resource_in_flight() {
    static std::atomic<int> in_flight{0};
    return in_flight;
}

inline std::atomic<int>& backend_persistent_resource_peak_in_flight() {
    static std::atomic<int> peak{0};
    return peak;
}

inline std::atomic<int>& backend_persistent_resource_depth() {
    static std::atomic<int> depth{0};
    return depth;
}

inline std::atomic<uint64_t>& backend_persistent_resource_overlaps() {
    static std::atomic<uint64_t> overlaps{0};
    return overlaps;
}

// One line, once per process, the first time two threads want this domain at once. Deliberately not
// behind an environment variable: it is the discovery that a design assumption is false, and the run
// that most needs to report it is the one nobody thought to arm a diagnostic for.
inline void backend_report_persistent_resource_contention(int in_flight) {
    static std::atomic<bool> reported{false};
    bool expected = false;
    if (!reported.compare_exchange_strong(expected, true)) return;
    std::fprintf(stderr,
                 "[render] %d threads are inside the backend's persistent-resource domain at once; "
                 "they are being serialised by its own lock (#2953)\n",
                 in_flight);
}

class BackendPersistentResourceGuard {
public:
    BackendPersistentResourceGuard() {
        const int in_flight =
            backend_persistent_resource_in_flight().fetch_add(1, std::memory_order_acq_rel) + 1;
        int peak = backend_persistent_resource_peak_in_flight().load(std::memory_order_relaxed);
        while (peak < in_flight &&
               !backend_persistent_resource_peak_in_flight().compare_exchange_weak(
                   peak, in_flight, std::memory_order_relaxed)) {
        }
        if (in_flight > 1) backend_report_persistent_resource_contention(in_flight);
        backend_persistent_resource_mutex().lock();
        if (backend_persistent_resource_depth().fetch_add(1, std::memory_order_acq_rel) != 0)
            backend_persistent_resource_overlaps().fetch_add(1, std::memory_order_relaxed);
    }

    ~BackendPersistentResourceGuard() {
        backend_persistent_resource_depth().fetch_sub(1, std::memory_order_acq_rel);
        backend_persistent_resource_mutex().unlock();
        backend_persistent_resource_in_flight().fetch_sub(1, std::memory_order_acq_rel);
    }

    BackendPersistentResourceGuard(const BackendPersistentResourceGuard&) = delete;
    BackendPersistentResourceGuard& operator=(const BackendPersistentResourceGuard&) = delete;
};

inline std::unordered_map<PersistentPipelineKey, PersistentPipeline, PersistentPipelineKeyHash>&
persistent_pipeline_cache() {
    static std::unordered_map<PersistentPipelineKey, PersistentPipeline, PersistentPipelineKeyHash> cache;
    return cache;
}

inline uint64_t& persistent_pipeline_generation() {
    static uint64_t generation = 0;
    return generation;
}

inline bool persistent_pipeline_cache_enabled() {
    return getenv("PROSPER_NO_BACKEND_PIPELINE_CACHE") == nullptr;
}

inline size_t persistent_pipeline_cache_limit_value(const char* value) {
    return value ? static_cast<size_t>(strtoull(value, nullptr, 10)) : size_t{4096};
}

inline size_t persistent_pipeline_cache_limit() {
    static const size_t limit = [] {
        return persistent_pipeline_cache_limit_value(
            getenv("PROSPER_PIPELINE_CACHE_ENTRIES"));
    }();
    return limit;
}

struct BackendWordVectorHash {
    size_t operator()(const std::vector<uint64_t>& words) const {
        uint64_t hash = 1469598103934665603ull;
        for (uint64_t word : words) {
            hash ^= word;
            hash *= 1099511628211ull;
        }
        return static_cast<size_t>(hash);
    }
};

struct PersistentBackendPipelineLayout {
    VkPipelineLayout handle = VK_NULL_HANDLE;
    uint64_t last_use = 0;
};

inline std::unordered_map<std::vector<uint64_t>, PersistentBackendPipelineLayout,
                          BackendWordVectorHash>&
persistent_backend_pipeline_layout_cache() {
    static std::unordered_map<std::vector<uint64_t>, PersistentBackendPipelineLayout,
                              BackendWordVectorHash> cache;
    return cache;
}

inline uint64_t& persistent_pipeline_layout_generation() {
    static uint64_t generation = 0;
    return generation;
}

inline bool persistent_pipeline_layout_cache_enabled() {
    return getenv("PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE") == nullptr;
}

inline size_t persistent_pipeline_layout_cache_limit() {
    static const size_t limit = [] {
        // 0 entries is not "off" -- it evicts on every miss, so a typo turns the cache into a
        // per-draw create/destroy loop that looks like a renderer regression (#3267).
        const char* value = getenv("PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES");
        return static_cast<size_t>(prosper::diag::env_u64_or_default_capped(
            "PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES", value, 256ull, SIZE_MAX, "entries"));
    }();
    return limit;
}

struct BackendRenderTimingStats {
    uint64_t calls = 0;
    uint64_t draws = 0;
    uint64_t command_buffers = 0;
    uint64_t queue_submits = 0;
    uint64_t fence_waits = 0;
    // WHY each flush happened (#2276). A Blue Prince gameplay submit performs 15.68 queue submits
    // and 15.68 fence waits -- sixteen CPU<->GPU round trips per rendered frame -- and nothing said
    // which of the three conditions in `flush_now` was responsible. They have different fixes: a
    // readback flush is a consumer wanting pixels, a storage-writeback flush is a compute result
    // going back to guest memory, and an explicit flush is the caller's own sequencing. Attributed
    // in the order the condition evaluates, so the four are exclusive and sum to the flush count.
    uint64_t flush_no_batch = 0;
    uint64_t flush_readback = 0;
    uint64_t flush_storage_writeback = 0;
    uint64_t flush_explicit = 0;
    uint64_t gpu_timestamp_samples = 0;
    double target_ms = 0;
    double draw_setup_ms = 0;
    double record_upload_ms = 0;
    double gpu_wait_ms = 0;
    // One device timestamp envelope covers the batch from the first command buffer starting through
    // the final command buffer finishing. It is a subset of the submit/fence wall-clock interval.
    double gpu_device_ms = 0;
    double readback_ms = 0;
    double cleanup_ms = 0;
    double setup_shader_ms = 0;
    double setup_fixed_ms = 0;
    // Leaves of setup_fixed_ms, which had none. `fixed` is 15.9% of a Blue Prince gameplay frame
    // (29.11 ms/submit, 13.85 us/draw) and until now was a single number, so anything inside it was
    // indistinguishable from anything else inside it -- the condition that let a diagnostic bill
    // 305 ms/submit to the renderer inside res_buffer_ms (instrument trap 130). index_upload is the
    // per-indexed-draw vkCreateBuffer + vkAllocateMemory + bind + map + copy at render_runner.h's
    // index block; the rest name the fixed-function state translation. setup_fixed_other_ms() is
    // printed SIGNED so the partition reports its own completeness (#2246).
    double res_fixed_index_upload_ms = 0;
    double res_fixed_blend_ms = 0;
    double res_fixed_depth_stencil_ms = 0;
    double res_fixed_viewport_ms = 0;
    double res_fixed_stages_ms = 0;
    // Printed as `pre_index_unsplit`, NOT `prologue`, and the name is the point. This leaf is
    // defined by where the timing region STARTS rather than by what it contains, so it is a residual
    // with a location rather than a component. `other=+0.27` reads as unattributed and invites
    // investigation; a component name reads as attributed and closes it, which would make the
    // partition look complete while the blind spot had only been renamed. Keeping the position in
    // the printed token turns "what is the prologue doing" into "what is IN that span", which is the
    // question #2254 exists to answer. Do not rename it to something that sounds like a phase until
    // it has been split into ones.
    double res_fixed_prologue_ms = 0;
    // Leaves of res_fixed_prologue_ms (#2254). That span was 12.60 ms/submit -- 39% of `fixed`,
    // ~6% of the frame -- and #2252 named it honestly as a positional residual with no identified
    // cause. `subgroup_scan` is the pair of fragment_spirv_required_subgroup_{size,features} calls,
    // each of which walks the ENTIRE SPIR-V module, per draw, unconditionally.
    double res_prologue_subgroup_scan_ms = 0;

    double setup_fixed_other_ms() const {
        return setup_fixed_ms - res_fixed_index_upload_ms - res_fixed_blend_ms -
               res_fixed_depth_stencil_ms - res_fixed_viewport_ms - res_fixed_stages_ms -
               res_fixed_prologue_ms;
    }
    double setup_resources_ms = 0;
    double setup_pipeline_ms = 0;
    // Sub-attribution of setup_resources_ms (#1284). `resources` is not descriptor bookkeeping alone:
    // the same interval also builds every texture upload (image/memory creation plus the staging
    // memcpy) and every storage-buffer upload, so a term that reads as "descriptor setup" can be
    // dominated by pixel and vertex bytes. These split it so the dominant sub-term is named rather
    // than assumed. res_texture_ms and res_buffer_ms cover the whole per-resource branch;
    // res_texture_upload_ms and res_texture_bind_ms are nested INSIDE res_texture_ms and cover only
    // the cache-miss work (image+staging build, view/sampler creation), so
    // res_texture_ms - res_texture_upload_ms - res_texture_bind_ms is the per-reference key/lookup
    // cost that a cache HIT still pays. res_descriptor_ms is the per-draw set layout/alloc/update.
    double res_texture_ms = 0;
    double res_texture_upload_ms = 0;
    double res_texture_bind_ms = 0;
    double res_buffer_ms = 0;
    // Nested INSIDE res_buffer_ms. acquire+copy were added as two CANDIDATE mechanisms, never as a
    // partition, and reading them as one hid the dominant term for months: measured on Blue Prince
    // gameplay, buffer=332.08 ms/submit against acquire=1.50 + copy=25.82, so 91.8% of the buffer
    // branch — 60.9% of the whole frame — was attributed to nothing. The enclosing `resources` line
    // self-checks to +0.00, which is exactly why nobody looked inside it: a partition that balances
    // at one scale says nothing about the scale below it. These four cover the remaining branches,
    // and res_buffer_other_ms() is printed SIGNED so an incomplete partition reports itself instead
    // of looking finished (#2245).
    double res_buffer_acquire_ms = 0;
    double res_buffer_copy_ms = 0;
    // vkCreateBuffer + vkAllocateMemory + the copy performed inside
    // create_transient_storage_buffer_upload — the fallback taken when a payload gets neither an
    // arena slice nor a pooled buffer. Timed by neither acquire nor copy before this.
    double res_buffer_create_ms = 0;
    // Split rather than one `index` bucket, because the two run over populations that differ by
    // ~5x and imply opposite fixes. `find` and the memo emplace are per REFERENCE (8.0M on the
    // #2245 run); the insert is per UNIQUE BUFFER (1.56M). If lookup dominates, the fix is the key
    // or the container; if insert dominates, there are simply many unique buffers and the fix is
    // upstream of the index entirely — the same distinction #2245 exists to draw between
    // skipped_unique and skipped_large. Summing them would be one more conflated bucket read as an
    // exhaustive one, which is the failure this whole change is about.
    double res_buffer_index_find_ms = 0;
    double res_buffer_index_insert_ms = 0;
    // hash_buffer_words over payloads <= 4 KiB (large ones take a unique tag and skip it).
    double res_buffer_hash_ms = 0;
    double res_descriptor_ms = 0;

    // Signed by construction: a negative value means the nested timers over-attribute (double
    // counting or a re-entered scope) and a large positive one means a branch is still unmeasured.
    // Clamping this to zero would make a broken instrument print identically to a correct one.
    double res_buffer_other_ms() const {
        return res_buffer_ms - res_buffer_acquire_ms - res_buffer_copy_ms - res_buffer_create_ms -
               res_buffer_index_find_ms - res_buffer_index_insert_ms - res_buffer_hash_ms;
    }

    double total_ms() const {
        return target_ms + draw_setup_ms + record_upload_ms + gpu_wait_ms + readback_ms + cleanup_ms;
    }
};

inline BackendRenderTimingStats& backend_render_timing_stats_storage() {
    static thread_local BackendRenderTimingStats stats;
    return stats;
}

inline BackendRenderTimingStats backend_render_timing_stats() {
    return backend_render_timing_stats_storage();
}

// `seed_rgba` (optional): native-format pixels to PRELOAD the color attachment with before the draws
// run (loadOp LOAD instead of the blue clear). This is real render-target memory semantics: a game
// pass that draws into a target it (or an earlier submit) already rendered composites OVER that
// content — without it every pass starts from the diagnostic blue clear, so cross-submit
// accumulation (UE4's UI-onto-backbuffer after a separate composite submit) is lost. Null (the
// default) keeps the blue-clear behavior byte-identical for every existing caller.
// `clear_rgba` (optional): 4 floats (RGBA, Vulkan order) to clear the color attachment to when no
// seed is supplied. Null keeps the legacy diagnostic blue — every test harness caller passes null,
// so their behavior is byte-identical. The live renderer passes the game's decoded fast-clear color
// (or opaque black when none), so real frames no longer start from blue (#309). PROSPER_CLEAR_DEBUG
// forces the blue back on regardless, so unrendered areas can still be spotted during development.
// Persistent Vulkan context. Creating a fresh instance+device PER render_draws_rgba call dominated
// wall-clock — every submit paid full device init — which made a many-draw frame (real gameplay is
// hundreds of draws/submit) impossibly slow and blocked headless scene investigation (#320). Create the
// instance/physical-device/device/queue ONCE (lazy, thread-safe static init) and reuse it across every
// call. Per-call Vulkan resources are created independently and are retained until their direct call
// or explicit ordered submission batch completes. The context intentionally leaks at process exit.
// LIFETIME INVARIANT: this context is intentionally never destroyed (no destructor; the device and
// instance leak at process exit). The compute backend BORROWS this device (#1091) and calls
// vkDestroyPipeline/vkFreeMemory on it at exit. Adding a destructor here that destroys the device
// would therefore create an immediate use-after-free in ~VulkanComputeContext. Do not add one
// without first giving compute an explicit release-before-teardown handshake.
//
// The mechanism on the compute side changed in #1704: that teardown now runs from a std::atexit
// handler registered after vkCreateInstance, not from a function-local static's destructor, so it is
// sequenced before any enabled Vulkan layer's own statics. That also makes the ordering against THIS
// context defined rather than unspecified — borrowing this device requires this context to already
// exist, so compute's handler is always registered later and therefore always runs first.
//
// Which means the specific use-after-free warned about above is now ordered away: a destructor added
// here would be registered earlier and would run after compute has released its objects. Do not read
// that as permission. The reasons not to add one are now different, not gone: guest threads can still
// be dispatching when exit() begins (execute_live_compute_items declines once the handler has run,
// but the window is not closed), and BorrowedComputeImageLease holds a raw VulkanComputeContext*.
// Give compute an explicit release-before-teardown handshake before adding a destructor here.
struct RenderVkCtx {
    VkInstance inst = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE;
    // Non-null only under PROSPER_VK_VALIDATION; without it the layer has no output sink.
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE; VkQueue queue = VK_NULL_HANDLE; uint32_t qfi = UINT32_MAX;
    VkDeviceSize storage_buffer_alignment = 1;
    double timestamp_period_ns = 0.0;
    uint32_t timestamp_valid_bits = 0;
    bool aniso_enabled = false; float max_aniso_limit = 1.0f;
    bool depth_bias_clamp_enabled = false;   // VkPhysicalDeviceFeatures::depthBiasClamp (#1349)
    bool logic_op_enabled = false; bool ok = false;
    bool fragment_stores_atomics = false;
    // Per-draw "fragment funnel" diagnostic (PROSPER_DRAW_STATS): pipeline-statistics + precise
    // occlusion queries. Enabled at device creation only when advertised; inert otherwise.
    bool pipeline_stats_enabled = false;
    bool occlusion_precise = false;
    // Geometry-probe (PROSPER_GEOM_PROBE): transform feedback for final clip-space positions.
    bool transform_feedback_enabled = false;
    bool subgroup_size_control = false;
    // Runtime-selected storage buffers (#2412). Successful contracts are bounded fixed arrays, so the
    // only descriptor-indexing feature they require is non-uniform storage-buffer array indexing.
    //
    // Measured available on both lanes' hardware: RADV STRIX_HALO and RTX 4090 report all of them true.
    // The AMD device additionally reports `…NonUniformIndexingNative = false`, which is a performance
    // note and not a correctness one here: our index is computed in scalar registers so it is
    // wave-uniform, and a driver waterfall over the distinct values present converges in one iteration.
    bool descriptor_indexing = false;
    bool storage_buffer_int64_atomics = false;
    bool compute_full_subgroups = false;
    uint32_t min_subgroup_size = 0, max_subgroup_size = 0;
    uint32_t max_compute_workgroup_subgroups = 0;
    uint32_t max_compute_workgroup_size_x = 0;
    uint32_t max_compute_workgroup_invocations = 0;
    VkShaderStageFlags required_subgroup_size_stages = 0;
    VkShaderStageFlags subgroup_stages = 0;
    VkSubgroupFeatureFlags subgroup_operations = 0;
    // Present unification (#1270): so prosper-app can adopt THIS device for its swapchain and blit the
    // renderer's front-buffer image straight to the screen (no 4K CPU round-trip). All additive and
    // only when advertised, so the headless test/screenshot path is byte-for-byte unchanged: on a
    // display-less target the surface instance-extensions and VK_KHR_swapchain are simply absent, these
    // stay false, and prosper-app falls back to its own separate present device + CPU pixels.
    bool present_surface_capable = false;   // instance enabled VK_KHR_surface (+ a platform surface ext)
    bool present_swapchain_capable = false; // device enabled VK_KHR_swapchain
    VkQueue present_queue = VK_NULL_HANDLE; // dedicated 2nd queue when the family has >=2, else == queue
    bool present_queue_shared = false;      // present_queue aliases the render queue -> submits need a mutex
};
inline const RenderVkCtx& render_vk_ctx() {
    static RenderVkCtx c = [] {
        RenderVkCtx r;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        // macOS: MoltenVK is linked directly, so VK_KHR_portability_enumeration is neither needed nor
        // accepted here (see prosper-app main.cpp). Only the device-level portability_subset matters.
        //
        // Present unification (#1270): enable the WSI surface instance-extensions when the loader
        // advertises them, so prosper-app can create its window surface on THIS instance and present on
        // this device (see present_surface_capable). "Add-if-available" only: a headless target (llvmpipe
        // CI, no display) advertises none, so the enabled list stays empty and instance creation is
        // exactly as before. This is created before SDL_Init in the app, so we can't consult
        // SDL_Vulkan_GetInstanceExtensions; instead enable every platform surface extension present and
        // let the app fall back to its own device if SDL still needs one we didn't get (#1270 R4).
        std::vector<const char*> inst_exts;
        // Hoisted so the validation block below can consult it: pushing an unadvertised extension
        // fails the WHOLE vkCreateInstance, which would also drop every WSI extension and turn off
        // present_surface_capable (#1270) -- a debugging switch must not degrade the present path.
        std::vector<VkExtensionProperties> avail;
        {
            static const char* const kWsiExts[] = {
                "VK_KHR_surface",
#if defined(_WIN32)
                "VK_KHR_win32_surface",
#elif defined(__APPLE__)
                "VK_EXT_metal_surface", "VK_MVK_macos_surface",
#else
                "VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface",
#endif
            };
            uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
            avail.resize(nie);
            if (nie) vkEnumerateInstanceExtensionProperties(nullptr, &nie, avail.data());
            auto has = [&](const char* name) {
                for (const auto& e : avail) if (!strcmp(e.extensionName, name)) return true;
                return false;
            };
            bool have_surface = has("VK_KHR_surface");
            if (have_surface) {
                bool have_platform = false;
                for (const char* e : kWsiExts) {
                    if (has(e)) { inst_exts.push_back(e); if (strcmp(e, "VK_KHR_surface")) have_platform = true; }
                }
                // A bare VK_KHR_surface with no platform surface is useless for a window; require both.
                r.present_surface_capable = have_platform;
                if (!have_platform) inst_exts.clear();
            }
            if (!inst_exts.empty()) {
                ici.enabledExtensionCount = (uint32_t)inst_exts.size();
                ici.ppEnabledExtensionNames = inst_exts.data();
            }
        }
        // PROSPER_VK_VALIDATION=1: enable the Khronos validation layer and register a messenger.
        //
        // This does NOT make validation newly possible -- `tools/vkval/vk_validation_scan.py`
        // (#1704, ctest `vkval_scan_logic`) has been scanning the suite for a long time and carries
        // its own baseline, because VVL's default debug_action writes to stdout/stderr by itself.
        // Four VUIDs were fixed off the back of it (#1713, #1714, #1717, #1726). Counts are
        // deliberately not quoted here, and should not be quoted from allowlist.txt's header
        // either -- that header is amended only SOMETIMES (#1714 deleted its entry without
        // amending it, leaving the stated ledger one id high). Run the scan and read the count it
        // computes. An earlier revision of this comment claimed validation "had never once been a
        // measurement in this project", which was simply false and would have retired a working
        // guard as unmeasured.
        //
        // What a messenger adds over the default action: in-process capture, so output can be
        // rate-limited (one violated VUID in a per-draw path otherwise fills the disk) and tagged;
        // a run that says whether it is armed; and coverage when the default action is off, which
        // is how some SDK builds and any VK_LAYER_* settings file can configure it.
        // Off by default; the layer costs real time per draw.
        const bool want_validation = [] {
            const char* e = getenv("PROSPER_VK_VALIDATION");
            return e && *e && strcmp(e, "0");
        }();
        static const char* const kValidationLayer = "VK_LAYER_KHRONOS_validation";
        constexpr uint32_t kMessengerPerIdLimit = 8;
        bool validation_enabled = false;
        bool messenger_wanted = true;
        std::vector<const char*> inst_layers;
        if (want_validation) {
            uint32_t nl = 0; vkEnumerateInstanceLayerProperties(&nl, nullptr);
            std::vector<VkLayerProperties> layers(nl);
            if (nl) vkEnumerateInstanceLayerProperties(&nl, layers.data());
            for (const auto& l : layers)
                if (!strcmp(l.layerName, kValidationLayer)) validation_enabled = true;
            if (validation_enabled) {
                inst_layers.push_back(kValidationLayer);
                ici.enabledLayerCount = (uint32_t)inst_layers.size();
                ici.ppEnabledLayerNames = inst_layers.data();
                bool have_debug_utils = false;
                for (const auto& e : avail)
                    if (!strcmp(e.extensionName, "VK_EXT_debug_utils")) have_debug_utils = true;
                if (have_debug_utils) {
                    inst_exts.push_back("VK_EXT_debug_utils");
                    ici.enabledExtensionCount = (uint32_t)inst_exts.size();
                    ici.ppEnabledExtensionNames = inst_exts.data();
                } else {
                    fprintf(stderr, "[vk-validation] VK_EXT_debug_utils is not advertised; the layer "
                                    "will run with its DEFAULT stdout/stderr action and this process "
                                    "will not rate-limit or tag its output\n");
                    fflush(stderr);
                    messenger_wanted = false;
                }
            } else {
                // Say so. A requested-but-absent layer is the silent-instrument case again.
                fprintf(stderr, "[vk-validation] PROSPER_VK_VALIDATION set but %s is not installed; "
                                "NO validation is running\n", kValidationLayer);
                fflush(stderr);
            }
        }

        // If the driver rejects the surface set (should not happen since each was advertised), retry with
        // no instance extensions so the headless render path never regresses on an unexpected loader.
        if (vkCreateInstance(&ici, nullptr, &r.inst) != VK_SUCCESS || !r.inst) {
            if (!inst_exts.empty()) {
                r.present_surface_capable = false;
                VkInstanceCreateInfo bare{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; bare.pApplicationInfo = &app;
                if (vkCreateInstance(&bare, nullptr, &r.inst) != VK_SUCCESS || !r.inst) return r;
                // The retry carries no layers and no extensions, so validation is GONE on this
                // instance. Announcing "active" here would be the exact laundered null this switch
                // exists to prevent -- a fourth silent state, and the worst of them.
                if (validation_enabled) {
                    fprintf(stderr, "[vk-validation] instance creation fell back to a bare create "
                                    "info; the validation layer was DROPPED and NO validation is "
                                    "running -- treat any clean result from this run as void\n");
                    fflush(stderr);
                    validation_enabled = false;
                }
            } else {
                return r;
            }
        }
        if (validation_enabled && messenger_wanted) {
            auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                r.inst, "vkCreateDebugUtilsMessengerEXT");
            VkDebugUtilsMessengerCreateInfoEXT dci{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT sev,
                                     VkDebugUtilsMessageTypeFlagsEXT,
                                     const VkDebugUtilsMessengerCallbackDataEXT* d,
                                     void*) -> VkBool32 {
                // Rate-limited per message id: one violated VUID in a per-draw path otherwise
                // produces gigabytes, and this project's guidance is explicit that a run log on the
                // tmpfs takes the machine down with it.
                // The spec requires this callback to be thread-safe, and prosper submits from more
                // than one thread (see present_queue_shared), so the rate-limit map needs a lock.
                static std::mutex seen_mu;
                static std::unordered_map<int32_t, uint32_t> seen;
                constexpr uint32_t kPerIdLimit = kMessengerPerIdLimit;
                uint32_t n;
                {
                    std::lock_guard<std::mutex> lk(seen_mu);
                    n = ++seen[d ? d->messageIdNumber : 0];
                }
                if (n > kPerIdLimit) return VK_FALSE;
                fprintf(stderr, "[vk-validation] %s %s%s\n",
                        sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "WARN",
                        d && d->pMessage ? d->pMessage : "(no message)",
                        n == kPerIdLimit ? "  [further reports of this id suppressed]" : "");
                fflush(stderr);
                return VK_FALSE;
            };
            if (!create || create(r.inst, &dci, nullptr, &r.debug_messenger) != VK_SUCCESS) {
                fprintf(stderr, "[vk-validation] layer loaded but the debug messenger could NOT be "
                                "registered; its output would be discarded, so treat any clean "
                                "result from this run as void\n");
                fflush(stderr);
            } else {
                fprintf(stderr, "[vk-validation] active (warnings and errors, %u per message id)\n",
                        kMessengerPerIdLimit);
                fflush(stderr);
            }
        }

        uint32_t nd = 0; vkEnumeratePhysicalDevices(r.inst, &nd, nullptr);
        if (!nd) return r;
        std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(r.inst, &nd, devs.data());
        const auto selection = prosper::frontend::select_vulkan_device(devs, VK_QUEUE_GRAPHICS_BIT);
        r.phys = selection.device;
        r.qfi = selection.queue_family;
        if (!r.phys || r.qfi == UINT32_MAX) return r;
        std::fprintf(stderr, "[render] Vulkan device: %s (%s)\n",
                     selection.properties.deviceName,
                     prosper::frontend::vulkan_device_type_name(selection.properties.deviceType));
        // Present unification (#1270): if the graphics family exposes a second queue, dedicate index 1
        // to prosper-app's present so the app's blit/present submits never contend the render queue's
        // external-synchronization. On a single-queue family (RADV STRIX_HALO is queueCount==1) the app
        // instead shares queue index 0 under a submit mutex. Requesting queueCount is harmless to the
        // headless path: the extra queue is simply never fetched by tests/screenshot.
        uint32_t family_queue_count = 1;
        {
            uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, nullptr);
            if (nq) {
                std::vector<VkQueueFamilyProperties> qfp(nq);
                vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, qfp.data());
                if (r.qfi < nq) {
                    family_queue_count = qfp[r.qfi].queueCount;
                    r.timestamp_valid_bits = qfp[r.qfi].timestampValidBits;
                }
            }
        }
        float prio[2] = {1.0f, 1.0f};
        const uint32_t want_queues = family_queue_count >= 2 ? 2u : 1u;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = r.qfi; qci.queueCount = want_queues; qci.pQueuePriorities = prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        // robustBufferAccess: OOB storage-buffer accesses are well-defined (predicated ops on
        // narrowed-EXEC lanes can't fault).
        VkPhysicalDeviceFeatures feats{}; feats.robustBufferAccess = VK_TRUE; dci.pEnabledFeatures = &feats;
        // samplerAnisotropy (#275): enable only if advertised; maxSamplerAnisotropy is the clamp ceiling.
        VkPhysicalDeviceFeatures supported{}; vkGetPhysicalDeviceFeatures(r.phys, &supported);
        // #2396: the recompiler emits SPIR-V declaring the Int64 capability (24 modules per GTA V run)
        // and we build PER-ATTACHMENT blend states for MRT -- both REQUIRE a device feature we never
        // requested. VUID-...pCode-08740 (24) and VUID-...pAttachments-00605 (20), both -> 0 when
        // enabled. Running against an unrequested capability is undefined behaviour, not a warning.
        // Enabled only when advertised, matching samplerAnisotropy (#275): on a device without them the
        // honest outcome is the layer error, not a silently miscompiled shader.
        if (supported.shaderInt64)      feats.shaderInt64 = VK_TRUE;
        if (supported.independentBlend) feats.independentBlend = VK_TRUE;
        VkPhysicalDeviceProperties phys_props{}; vkGetPhysicalDeviceProperties(r.phys, &phys_props);
        r.max_compute_workgroup_size_x = phys_props.limits.maxComputeWorkGroupSize[0];
        r.max_compute_workgroup_invocations = phys_props.limits.maxComputeWorkGroupInvocations;
        r.storage_buffer_alignment = std::max<VkDeviceSize>(
            1, phys_props.limits.minStorageBufferOffsetAlignment);
        r.timestamp_period_ns = phys_props.limits.timestampPeriod;
        r.aniso_enabled = supported.samplerAnisotropy;
        r.depth_bias_clamp_enabled = supported.depthBiasClamp;
        if (r.depth_bias_clamp_enabled) feats.depthBiasClamp = VK_TRUE;
        r.logic_op_enabled = supported.logicOp;
        r.max_aniso_limit = phys_props.limits.maxSamplerAnisotropy;
        if (r.aniso_enabled) feats.samplerAnisotropy = VK_TRUE;
        if (r.logic_op_enabled) feats.logicOp = VK_TRUE;
        // Portable AMD P0/P10/P20 lowering inserts a descriptor-free geometry pass on devices that
        // lack explicit vertex-parameter fragment extensions. Core geometryShader is available on
        // the Linux llvmpipe headless target and the desktop Vulkan drivers we support.
        feats.geometryShader = supported.geometryShader;
        // Storage-image shaders declare the format-free read/write capabilities. Enable every
        // corresponding core feature advertised by the device; fragment/vertex stores additionally
        // need their pipeline-stage store features.
        feats.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
        feats.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
        feats.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
        feats.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
        r.fragment_stores_atomics = supported.fragmentStoresAndAtomics;
        // Per-draw fragment-funnel diagnostic (PROSPER_DRAW_STATS): pipeline statistics + precise
        // occlusion. Enable only when advertised; costs nothing unless the diagnostic is used.
        feats.pipelineStatisticsQuery = supported.pipelineStatisticsQuery;
        r.pipeline_stats_enabled = supported.pipelineStatisticsQuery;
        if (supported.occlusionQueryPrecise) { feats.occlusionQueryPrecise = VK_TRUE; r.occlusion_precise = true; }
        // robustImageAccess (VK_EXT_image_robustness): OpImageRead OOB must return zero (#131). Guarded.
        // Device extensions accumulate into a vector so the (optional) image-robustness and (macOS)
        // portability-subset extensions coexist.
        std::vector<const char*> dev_exts;
        VkPhysicalDeviceImageRobustnessFeaturesEXT irf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT};
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup_properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup_core_properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceTransformFeedbackFeaturesEXT tf_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT};
        VkPhysicalDeviceDescriptorIndexingFeaturesEXT di_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
        VkPhysicalDeviceShaderAtomicInt64Features atomic_int64_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
        { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, nullptr);
          std::vector<VkExtensionProperties> de(ne);
          vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, de.data());
          for (uint32_t i = 0; i < ne; i++) {
              if (!strcmp(de[i].extensionName, "VK_EXT_image_robustness")) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &irf; vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (irf.robustImageAccess) {
                      irf.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &irf;
                      dev_exts.push_back("VK_EXT_image_robustness");
                  }
              }
              if (!strcmp(de[i].extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
                  // Runtime-selected storage buffers use bounded fixed arrays. Request only the one
                  // descriptor-indexing feature their non-uniform access chains require.
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &di_features;
                  vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  const bool have_ssbo_arrays =
                      di_features.shaderStorageBufferArrayNonUniformIndexing;
                  if (have_ssbo_arrays) {
                      // Request exactly the feature in use, not the whole struct as queried. Enabling a feature
                      // the design does not use widens the driver contract for no benefit, and a
                      // later reader cannot tell which ones are load-bearing.
                      VkPhysicalDeviceDescriptorIndexingFeaturesEXT want{
                          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
                      want.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
                      di_features = want;
                      di_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &di_features;
                      dev_exts.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
                      r.descriptor_indexing = true;
                      // Log the SUCCESS too, not only the shortfall. A diagnostic that fires only on
                      // failure makes the working case silent and therefore indistinguishable from code
                      // that never ran -- which is the ambiguity that has cost this project several false
                      // zeros today alone. This line is the lever check for stage 1: it is the only way
                      // to confirm the capability was actually acquired rather than merely compiled.
                      if (getenv("PROSPER_GFXLOG"))
                          fprintf(stderr, "[vk] descriptor indexing ENABLED "
                                          "(nonUniform ssbo)\n");
                  } else if (getenv("PROSPER_GFXLOG")) {
                      // Report the SHORTFALL, not merely "unavailable": which feature is missing decides
                      // whether a fallback is possible at all, and a bare "not supported" would send the
                      // next reader to the extension list when the extension is present.
                      fprintf(stderr,
                              "[vk] VK_EXT_descriptor_indexing lacks "
                              "shaderStorageBufferArrayNonUniformIndexing\n");
                  }
              }
              if (!strcmp(de[i].extensionName, VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &atomic_int64_features;
                  vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (supported.shaderInt64 &&
                      atomic_int64_features.shaderBufferInt64Atomics) {
                      VkPhysicalDeviceShaderAtomicInt64Features want{
                          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
                      want.shaderBufferInt64Atomics = VK_TRUE;
                      atomic_int64_features = want;
                      atomic_int64_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &atomic_int64_features;
                      dev_exts.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
                      r.storage_buffer_int64_atomics = true;
                  }
              }
              if (!strcmp(de[i].extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &subgroup_features;
                  vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (subgroup_features.subgroupSizeControl) {
                      VkPhysicalDeviceProperties2 p2{
                          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                      p2.pNext = &subgroup_core_properties;
                      subgroup_core_properties.pNext = &subgroup_properties;
                      vkGetPhysicalDeviceProperties2(r.phys, &p2);
                      subgroup_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &subgroup_features;
                      dev_exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
                      r.subgroup_size_control = true;
                      r.compute_full_subgroups = subgroup_features.computeFullSubgroups;
                      r.min_subgroup_size = subgroup_properties.minSubgroupSize;
                      r.max_subgroup_size = subgroup_properties.maxSubgroupSize;
                      r.max_compute_workgroup_subgroups =
                          subgroup_properties.maxComputeWorkgroupSubgroups;
                      r.required_subgroup_size_stages =
                          subgroup_properties.requiredSubgroupSizeStages;
                      r.subgroup_stages = subgroup_core_properties.supportedStages;
                      r.subgroup_operations = subgroup_core_properties.supportedOperations;
                  }
              }
              // Geometry-probe (PROSPER_GEOM_PROBE): capture the last pre-rasterization stage. Enable
              // only the base transformFeedback feature; separate geometry streams are not needed.
              if (!strcmp(de[i].extensionName, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME)) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &tf_features; vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (tf_features.transformFeedback) {
                      tf_features.geometryStreams = VK_FALSE;
                      tf_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &tf_features;
                      dev_exts.push_back(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
                      r.transform_feedback_enabled = true;
                  }
              }
              // Present unification (#1270): the swapchain device-extension, only when the instance is
              // surface-capable. Enabling it on the headless render device is harmless (no swapchain is
              // ever created there by tests/screenshot); prosper-app needs it to create its swapchain on
              // this shared device. Gated on present_surface_capable so a display-less build never
              // requests it.
              if (r.present_surface_capable &&
                  !strcmp(de[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                  dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
                  r.present_swapchain_capable = true;
              }
#ifdef __APPLE__
              // Spec-mandated: must be enabled when advertised (MoltenVK always advertises it).
              if (!strcmp(de[i].extensionName, "VK_KHR_portability_subset")) dev_exts.push_back("VK_KHR_portability_subset");
#endif
          } }
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
        if (vkCreateDevice(r.phys, &dci, nullptr, &r.dev) != VK_SUCCESS || !r.dev) return r;
        vkGetDeviceQueue(r.dev, r.qfi, 0, &r.queue);
        // Present unification (#1270): resolve the queue prosper-app will use to present. A dedicated
        // second queue (when the family has >=2) avoids contending the render queue; otherwise the app
        // shares queue 0 under a submit mutex (present_queue_shared). present capability requires the
        // swapchain extension to have been enabled above.
        if (r.present_swapchain_capable) {
            if (want_queues >= 2) {
                vkGetDeviceQueue(r.dev, r.qfi, 1, &r.present_queue);
                r.present_queue_shared = false;
            } else {
                r.present_queue = r.queue;
                r.present_queue_shared = true;
            }
        }
        r.ok = true;
        // Publish this device so the compute backend can adopt it instead of creating a second one
        // (#1091). Only the features compute needs are advertised; it declines the shared context if
        // they are missing and creates its own device exactly as before. Graphics and compute run
        // strictly sequentially on one thread, so sharing the queue needs no extra synchronization.
        if (!std::getenv("PROSPER_NO_SHARED_VULKAN_DEVICE")) {
            prosper::gpu::SharedVulkanContext shared;
            shared.instance = r.inst;
            shared.physical = r.phys;
            shared.device = r.dev;
            shared.queue = r.queue;
            shared.queue_family = r.qfi;
            // Publish what the device was actually CREATED with (feats), not what the physical
            // device merely supports. They are equal today because feats.x = supported.x
            // unconditionally, but the consumer's contract is "enabled here" -- if these ever became
            // gated like aniso/logicOp are, publishing `supported` would let compute emit shaders
            // declaring a capability the device never enabled.
            shared.storage_image_read_without_format = feats.shaderStorageImageReadWithoutFormat;
            shared.storage_image_write_without_format = feats.shaderStorageImageWriteWithoutFormat;
            shared.compute_subgroup_size_control = r.subgroup_size_control &&
                (r.required_subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                (r.subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT);
            shared.compute_full_subgroups = r.compute_full_subgroups;
            shared.compute_subgroup_vote =
                (r.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0;
            shared.compute_subgroup_arithmetic =
                (r.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
            // Publish the descriptor-indexing capability enabled on this device: the flag is set only
            // where non-uniform storage-buffer indexing was actually requested at device creation,
            // never from `supported`.
            shared.descriptor_indexing = r.descriptor_indexing;
            shared.storage_buffer_int64_atomics =
                r.storage_buffer_int64_atomics && feats.shaderInt64;
            shared.min_compute_subgroup_size = r.min_subgroup_size;
            shared.max_compute_subgroup_size = r.max_subgroup_size;
            shared.max_compute_workgroup_subgroups = r.max_compute_workgroup_subgroups;
            shared.max_compute_workgroup_size_x = r.max_compute_workgroup_size_x;
            shared.max_compute_workgroup_invocations =
                r.max_compute_workgroup_invocations;
            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &queue_family_count, nullptr);
            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(
                r.phys, &queue_family_count, queue_families.data());
            shared.compute_queue_supported = r.qfi < queue_families.size() &&
                (queue_families[r.qfi].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            const auto add_native_storage_format = [&](prosper::gpu::DataFormat format,
                                                       uint32_t components,
                                                       VkFormat vk_format) {
                VkImageFormatProperties properties{};
                constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                if (vkGetPhysicalDeviceImageFormatProperties(
                        r.phys, vk_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                        usage, 0, &properties) == VK_SUCCESS)
                    shared.native_storage_format_support |=
                        prosper::gpu::native_storage_format_support_bit(format, components);
                if (vkGetPhysicalDeviceImageFormatProperties(
                        r.phys, vk_format, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
                        usage, 0, &properties) == VK_SUCCESS)
                    shared.native_storage_format_support |=
                        prosper::gpu::native_storage_3d_format_support_bit(format, components);
            };
            add_native_storage_format(prosper::gpu::DataFormat::Unorm8, 1, VK_FORMAT_R8_UNORM);
            add_native_storage_format(prosper::gpu::DataFormat::Unorm8, 2, VK_FORMAT_R8G8_UNORM);
            add_native_storage_format(prosper::gpu::DataFormat::Unorm8, 4, VK_FORMAT_R8G8B8A8_UNORM);
            add_native_storage_format(prosper::gpu::DataFormat::Float16, 1, VK_FORMAT_R16_SFLOAT);
            add_native_storage_format(prosper::gpu::DataFormat::Float16, 2, VK_FORMAT_R16G16_SFLOAT);
            add_native_storage_format(prosper::gpu::DataFormat::Float16, 4, VK_FORMAT_R16G16B16A16_SFLOAT);
            add_native_storage_format(prosper::gpu::DataFormat::Float32, 1, VK_FORMAT_R32_SFLOAT);
            add_native_storage_format(prosper::gpu::DataFormat::Float32, 2, VK_FORMAT_R32G32_SFLOAT);
            add_native_storage_format(prosper::gpu::DataFormat::Float32, 4, VK_FORMAT_R32G32B32A32_SFLOAT);
            add_native_storage_format(
                prosper::gpu::DataFormat::Float10_11_11, 3,
                VK_FORMAT_B10G11R11_UFLOAT_PACK32);
            add_native_storage_format(
                prosper::gpu::DataFormat::Uint32, 1, VK_FORMAT_R32_UINT);
            add_native_storage_format(
                prosper::gpu::DataFormat::Uint8, 1, VK_FORMAT_R8_UINT);
            add_native_storage_format(
                prosper::gpu::DataFormat::Uint8, 4, VK_FORMAT_R8G8B8A8_UINT);
            add_native_storage_format(
                prosper::gpu::DataFormat::Uint16, 1, VK_FORMAT_R16_UINT);
            add_native_storage_format(
                prosper::gpu::DataFormat::Unorm2_10_10_10, 4,
                VK_FORMAT_A2B10G10R10_UNORM_PACK32);
            // Present unification (#1270): advertise present adoption only when the instance is
            // surface-capable AND the device enabled VK_KHR_swapchain AND a present queue was resolved.
            shared.present_capable = r.present_surface_capable && r.present_swapchain_capable &&
                                     r.present_queue != VK_NULL_HANDLE;
            shared.present_queue = r.present_queue;
            shared.present_queue_shared = r.present_queue_shared;
            prosper::gpu::set_shared_vulkan_context(shared);
        }
        return r;
    }();
    return c;
}

// Present unification (#1270): serialize a single queue CALL against prosper-app's present submits when
// they share one VkQueue, and only once the app has adopted the shared queue (shared_present_active());
// otherwise it is a plain call after an acquire atomic load, so the headless/test/screenshot path and
// every non-shared device are unaffected. vkQueueSubmit returns without waiting for GPU work; the
// wait-idle wrapper (used only on the batch fence-timeout and compute-drain ERROR paths) does drain the
// queue under the lock, which briefly blocks the peer thread's submits -- acceptable on those rare paths.
//
// Both wrappers also take a GPU submit-gate region (#3225). Until the frontend begins shutting
// down the gate always admits, so this is one uncontended CAS on a path that is about to enter the
// driver; after shutdown begins it refuses, and the call returns VK_ERROR_DEVICE_LOST WITHOUT
// entering the driver. That refusal is what lets prosper-app's drain reach zero before it _Exit()s
// a process whose guest thread it cannot join — a thread caught inside an amdgpu submission at
// exit_group() parks in __drm_exec_lock_obj and takes the host compositor down with it.
// VK_ERROR_DEVICE_LOST is the honest result: the device really is going away, and every caller
// here already treats a failed submit as "not submitted" and cleans up accordingly.
inline VkResult render_locked_queue_submit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
    prosper::GpuSubmitRegion gate;
    if (!gate.admitted()) return VK_ERROR_DEVICE_LOST;
    if (prosper::gpu::shared_present_active()) {
        std::lock_guard<std::mutex> lk(prosper::gpu::shared_present_submit_mutex());
        return vkQueueSubmit(q, n, s, f);
    }
    return vkQueueSubmit(q, n, s, f);
}
inline VkResult render_locked_queue_wait_idle(VkQueue q) {
    prosper::GpuSubmitRegion gate;
    if (!gate.admitted()) return VK_ERROR_DEVICE_LOST;
    if (prosper::gpu::shared_present_active()) {
        std::lock_guard<std::mutex> lk(prosper::gpu::shared_present_submit_mutex());
        return vkQueueWaitIdle(q);
    }
    return vkQueueWaitIdle(q);
}

struct BackendSubmissionBatchResult {
    VkResult submit_result = VK_SUCCESS;
    VkResult wait_result = VK_SUCCESS;
    uint64_t command_buffers = 0;
    uint64_t queue_submits = 0;
    uint64_t fence_waits = 0;
    uint64_t gpu_timestamp_samples = 0;
    double gpu_device_ms = 0.0;
};

inline constexpr uint64_t backend_timestamp_delta(uint64_t begin, uint64_t end,
                                                   uint32_t valid_bits) {
    if (!valid_bits || valid_bits > 64) return 0;
    if (valid_bits == 64) return end - begin;
    const uint64_t mask = (uint64_t{1} << valid_bits) - 1u;
    return (end - begin) & mask;
}

// --- per-pass GPU device time (#2333) -------------------------------------------------------
//
// The backend already measures ONE device-time envelope per submission batch. That answers "how
// long did the GPU take" and cannot answer the question #2276 is actually stuck on: whether the
// cost is rasterisation or the ~16 pass BOUNDARIES the work is split across. A single envelope is
// the sum, and the sum is the same either way.
//
// So this records a timestamp pair around each render pass. Gated: the pool creation and the two
// writes are real work, and this must not be paid on an ordinary run.
//
// It is deliberately one query pool PER PASS rather than one sized pool per command buffer. The
// pass count is not known when the command buffer starts -- the collapsed Blue Prince state records
// over a thousand passes into a single command buffer (#2333) -- so a pre-sized pool would either
// cap the measurement silently or guess large. A diagnostic that silently stops recording after N
// passes would report the surviving passes as though they were all of them, which is the exact
// failure mode this instrument exists to avoid.
struct BackendPassTiming {
    VkDevice dev = VK_NULL_HANDLE;
    VkQueryPool pool = VK_NULL_HANDLE;
    double period_ns = 0.0;
    uint32_t valid_bits = 0;
    uint32_t width = 0, height = 0;
    uint64_t target = 0;
    size_t draws = 0;
    // Draws in this pass whose colour write mask is non-zero. #2283 turns on whether a
    // no-colour-base pass is depth-only: extents that look like shadow maps are circumstantial, a
    // mask of 0 on every draw is the direct check. Counted rather than inferred so a single
    // colour-writing draw in such a pass is visible instead of averaged away.
    size_t colour_writing_draws = 0;
    bool ended = false;
};

inline bool backend_pass_timing_enabled() {
    // Live getenv rather than a cached value: gpu_replay arms this per invocation, and a
    // process-lifetime cache would make a second run in the same process silently unmeasured.
    static const bool on = getenv("PROSPER_PASS_TIMING") != nullptr;
    return on;
}

inline std::vector<BackendPassTiming>& backend_pass_timings() {
    static std::vector<BackendPassTiming> records;
    return records;
}

inline void backend_pass_timing_begin(VkDevice dev, VkCommandBuffer cmd, double period_ns,
                                      uint32_t valid_bits, uint32_t width, uint32_t height,
                                      uint64_t target, size_t draws, size_t colour_writing_draws) {
    if (!backend_pass_timing_enabled() || !dev || !cmd || period_ns <= 0.0 || !valid_bits) return;
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 2;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (vkCreateQueryPool(dev, &info, nullptr, &pool) != VK_SUCCESS || !pool) return;
    vkCmdResetQueryPool(cmd, pool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    backend_pass_timings().push_back(
        BackendPassTiming{dev, pool, period_ns, valid_bits, width, height, target, draws,
                          colour_writing_draws, false});
}

inline void backend_pass_timing_end(VkCommandBuffer cmd) {
    if (!backend_pass_timing_enabled() || !cmd) return;
    auto& records = backend_pass_timings();
    if (records.empty() || records.back().ended || !records.back().pool) return;
    // Same reason as the batch envelope: make the closing timestamp depend on every command in the
    // pass rather than on pipeline position alone, so this is one interval and not a lower bound.
    //
    // This barrier also ORDERS pass completions, so it can only ever produce non-overlapping
    // measured intervals -- which means "coverage never exceeds 100%" is a statement about the
    // measurement unless the barrier is removed and the result survives. It was, and it does:
    // without this barrier, 10,715 batches still show 0 over 100% and a median coverage of 92.1%
    // against 92.9% with it. The two agree within ~1 point, so the barrier is neither manufacturing
    // the non-overlap nor materially inflating per-pass times.
    //
    // That is corroboration rather than proof: removing the barrier makes each interval a lower
    // bound, which lowers sums and would MASK overlap rather than reveal it. So report per-pass
    // non-overlap as a property of the measurement, not of the workload.
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, records.back().pool, 1);
    records.back().ended = true;
}

// Called only after the batch's fence has proved completion -- reading a timestamp query whose
// commands may still be executing returns NOT_READY, and treating that as 0 ms would understate
// exactly the passes that took longest.
inline void backend_pass_timing_report(bool completed, double envelope_ms = -1.0) {
    if (!backend_pass_timing_enabled()) return;
    auto& records = backend_pass_timings();
    if (records.empty()) return;
    if (completed) {
        double total = 0.0;
        size_t reported = 0, unreadable = 0;
        for (const BackendPassTiming& record : records) {
            if (!record.ended) { ++unreadable; continue; }
            uint64_t values[2] = {0, 0};
            if (vkGetQueryPoolResults(record.dev, record.pool, 0, 2, sizeof(values), values,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
                ++unreadable;
                continue;
            }
            const uint64_t ticks =
                backend_timestamp_delta(values[0], values[1], record.valid_bits);
            const double ms = static_cast<double>(ticks) * record.period_ns / 1'000'000.0;
            total += ms;
            ++reported;
            fprintf(stderr,
                    "[pass-timing] pass=%zu %ux%u target=0x%llx draws=%zu cwm_draws=%zu device=%.4f ms\n",
                    reported - 1, record.width, record.height,
                    (unsigned long long)record.target, record.draws,
                    record.colour_writing_draws, ms);
        }
        // The count of passes that could NOT be read is printed even when it is zero. A per-pass
        // report whose passes silently go missing looks like a frame with fewer, cheaper passes --
        // which is a conclusion, not an absence of data.
        // The reconciliation, printed per BATCH rather than left to cross-run arithmetic.
        // Summing per-pass times over a whole run and comparing against a cumulative average
        // compares two different populations and cannot distinguish "this instrument covers a
        // subset of device work" from "the envelopes overlap so their sum was never a total".
        // Against the SAME batch's own envelope the question is local and has one answer:
        // coverage below 100% is work inside this command buffer that is not in a render pass
        // (compute, blits, readback copies), and coverage above 100% means the per-pass
        // intervals overlap each other.
        if (envelope_ms >= 0.0) {
            const double coverage = envelope_ms > 0.0 ? total / envelope_ms * 100.0 : 0.0;
            fprintf(stderr,
                    "[pass-timing] passes=%zu unreadable=%zu summed_device=%.3f ms envelope=%.3f ms coverage=%.1f%%\n",
                    reported, unreadable, total, envelope_ms, coverage);
        } else {
            fprintf(stderr, "[pass-timing] passes=%zu unreadable=%zu summed_device=%.3f ms\n",
                    reported, unreadable, total);
        }
    }
    for (const BackendPassTiming& record : records)
        if (record.pool) vkDestroyQueryPool(record.dev, record.pool, nullptr);
    records.clear();
}


// A successful queue submit followed by failed fence and queue-idle waits is not an ordinary
// failure: Vulkan still owns every referenced resource, so those objects must be retained.
enum class BackendSubmissionState { NotSubmitted, Complete, Pending };

inline constexpr BackendSubmissionState backend_submission_state(bool submitted, bool finished) {
    return !submitted ? BackendSubmissionState::NotSubmitted
                      : finished ? BackendSubmissionState::Complete
                                 : BackendSubmissionState::Pending;
}

// Once completion of submitted work cannot be proven, cached Vulkan objects may remain referenced
// beyond the local batch lifetime. Keep the renderer poisoned for the rest of the device lifetime:
// later callbacks must not submit work or evict any persistent object potentially owned by the GPU.
inline std::atomic<bool>& backend_unproven_submission_storage() {
    static std::atomic<bool> pending{false};
    return pending;
}

inline bool backend_has_unproven_submission() {
    return backend_unproven_submission_storage().load(std::memory_order_acquire);
}

inline void backend_mark_unproven_submission() {
    backend_unproven_submission_storage().store(true, std::memory_order_release);
}

// Collect command buffers that belong to one ordered renderer callback. Vulkan queue order preserves
// target producer/consumer dependencies; one fence on the final submission is enough to retain every
// referenced object until the complete callback has finished. Direct test callers keep the established
// synchronous behavior by omitting this object.
class BackendSubmissionBatch {
public:
    BackendSubmissionBatch() = default;
    BackendSubmissionBatch(const BackendSubmissionBatch&) = delete;
    BackendSubmissionBatch& operator=(const BackendSubmissionBatch&) = delete;

    ~BackendSubmissionBatch() {
        if (!commands_.empty()) {
            const RenderVkCtx& ctx = render_vk_ctx();
            if (ctx.ok) (void)submit_and_wait(ctx.dev, ctx.queue, false);
            else discard();
        }
        if (commands_.empty()) complete();
    }

    bool pending() const { return !commands_.empty(); }
    bool retains_pending_resources() const { return pending_resources_abandoned_; }

    void enqueue(VkCommandBuffer command) {
        if (!pending_resources_abandoned_)
            commands_.push_back(command);
    }

    void begin_gpu_timestamp(VkDevice dev, VkCommandBuffer command,
                             double period_ns, uint32_t valid_bits) {
        if (pending_resources_abandoned_ || !commands_.empty() || gpu_timestamp_.pool ||
            !dev || !command || period_ns <= 0.0 || !valid_bits)
            return;
        VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (vkCreateQueryPool(dev, &query_info, nullptr, &pool) != VK_SUCCESS || !pool)
            return;
        gpu_timestamp_ = {dev, pool, period_ns, valid_bits, false};
        vkCmdResetQueryPool(command, pool, 0, 2);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    }

    void end_gpu_timestamp(VkCommandBuffer command) {
        if (pending_resources_abandoned_ || !gpu_timestamp_.pool || gpu_timestamp_.ended || !command)
            return;
        // Command buffers in one queue submission may overlap. Make the final timestamp depend on
        // every earlier command in the batch, so this is one envelope rather than a sum of overlapping
        // per-command intervals.
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                             0, nullptr, 0, nullptr, 0, nullptr);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            gpu_timestamp_.pool, 1);
        gpu_timestamp_.ended = true;
    }

    void add_cleanup(std::function<void()> cleanup) {
        if (!pending_resources_abandoned_) {
            cleanups_.push_back(std::move(cleanup));
            return;
        }
        // Some resources are bundled only after submission diagnostics finish. If completion was
        // already found indeterminate, destroying this late closure would release those resources
        // (including borrowed-image leases) while the submitted command may still use them.
        (void)new std::function<void()>(std::move(cleanup));
    }

    // Persistent attachment state is updated speculatively so later command buffers in the same
    // ordered batch can LOAD/sample earlier results. If the batch cannot be submitted and completed,
    // every touched entry must be invalidated before its retained resources are released.
    void add_failure_cleanup(std::function<void()> cleanup) {
        if (!pending_resources_abandoned_)
            failure_cleanups_.push_back(std::move(cleanup));
    }

    void discard() {
        commands_.clear();
        release_gpu_timestamp();
        finish_persistent_state(false);
    }

    // Completion could not be proven after a successful submit. Invalidate speculative state, but
    // never invoke or destroy cleanup callbacks for work Vulkan may still own. Besides Vulkan
    // handles, their captures can contain ownership tokens for resources borrowed from another
    // cache. Keep the complete closure set alive until process teardown; a normal static owner is
    // unsuitable because its destructor could run while a lost device still owns the submission.
    // The sticky flag also rejects callbacks registered later in the current renderer call
    // (registration follows diagnostics).
    void abandon_pending_resources() {
        commands_.clear();
        // Vulkan may still own this query pool. Deliberately leak it with the other submitted
        // resources; destroying it after unproven completion would violate object lifetime.
        gpu_timestamp_ = {};
        finish_persistent_state(false);
        pending_resources_abandoned_ = true;
        backend_mark_unproven_submission();
        if (!cleanups_.empty()) {
            (void)new std::vector<std::function<void()>>(std::move(cleanups_));
        }
    }

    BackendSubmissionBatchResult submit_and_wait(VkDevice dev, VkQueue queue,
                                                  bool backend_trace) {
        BackendSubmissionBatchResult result;
        result.command_buffers = commands_.size();
        if (pending_resources_abandoned_) {
            result.submit_result = VK_ERROR_DEVICE_LOST;
            result.wait_result = VK_ERROR_DEVICE_LOST;
            return result;
        }
        if (backend_has_unproven_submission()) {
            // Another batch/helper has unproven submitted work. These commands never reached the
            // queue, so discard them, invalidate their speculative state, and let complete() safely
            // release their resources without making another Vulkan call on the poisoned device.
            result.submit_result = VK_ERROR_DEVICE_LOST;
            result.wait_result = VK_ERROR_DEVICE_LOST;
            discard();
            return result;
        }
        if (commands_.empty()) return result;

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = static_cast<uint32_t>(commands_.size());
        submit.pCommandBuffers = commands_.data();
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(dev, &fence_info, nullptr, &fence) != VK_SUCCESS || !fence) {
            result.submit_result = VK_ERROR_INITIALIZATION_FAILED;
            discard();
            return result;
        }
        if (backend_trace) {
            std::fprintf(stderr, "[backend-trace] queue-submit begin command_buffers=%zu\n",
                         commands_.size());
            std::fflush(stderr);
        }
        result.submit_result = render_locked_queue_submit(queue, 1, &submit, fence);
        result.queue_submits = 1;
        if (backend_trace) {
            std::fprintf(stderr,
                         "[backend-trace] queue-submit end result=%d; fence-wait begin\n",
                         static_cast<int>(result.submit_result));
            std::fflush(stderr);
        }
        if (result.submit_result == VK_SUCCESS) {
            result.wait_result = vkWaitForFences(
                dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
            result.fence_waits = 1;
            // Preserve lifetime safety even if the bounded diagnostic wait expires.
            if (result.wait_result != VK_SUCCESS)
                result.wait_result = render_locked_queue_wait_idle(queue);
        }
        if (backend_trace) {
            std::fprintf(stderr, "[backend-trace] fence-wait end result=%d\n",
                         static_cast<int>(result.wait_result));
            std::fflush(stderr);
        }
        // A GRAPHICS submission that fails to submit or fails to complete is reported
        // unconditionally, not only under the backend trace.
        //
        // Until now this was silent: the 5-second fence wait falls through to a blocking
        // queue-wait-idle and says nothing, so a hung or lost graphics submission left no record at
        // all. The compute backend, by contrast, reports its refusals loudly — with the result that
        // a device loss discovered at compute's NEXT submit was read for days as "a compute dispatch
        // hung the GPU" purely because compute was the only side that spoke. See instrument trap 170:
        // a loss at submit time names the submission that OBSERVES it, not the one that caused it.
        //
        // The sentence that used to end this paragraph -- "the hang was never compute's", argued from
        // 705 of 705 fence waits completing with zero timeouts -- is WITHDRAWN, and trap 171 records
        // why: the timeout is 30 seconds and the event is 2. Recording the wait DURATION found a
        // single 2,045 ms compute dispatch in a whole route, immediately before the loss. A zero
        // timeout count never meant a zero latency count. This reporting stands on its own merits:
        // a graphics submission that fails or hangs must not be silent either.
        if (result.submit_result != VK_SUCCESS || result.wait_result != VK_SUCCESS) {
            static std::atomic<int> reported{0};
            const int n = reported.fetch_add(1);
            if (n < 16 || (n & 255) == 0)
                std::fprintf(stderr,
                             "[backend] GRAPHICS submission failed: submit=%d wait=%d "
                             "command-buffers=%zu occurrence=%d\n",
                             static_cast<int>(result.submit_result),
                             static_cast<int>(result.wait_result),
                             result.command_buffers, n + 1);
        }
        const BackendSubmissionState state = backend_submission_state(
            result.submit_result == VK_SUCCESS,
            result.submit_result == VK_SUCCESS && result.wait_result == VK_SUCCESS);
        if (state != BackendSubmissionState::Pending) {
            if (state == BackendSubmissionState::Complete && gpu_timestamp_.ended) {
                uint64_t values[2]{};
                if (vkGetQueryPoolResults(
                        dev, gpu_timestamp_.pool, 0, 2, sizeof(values), values,
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                    const uint64_t ticks = backend_timestamp_delta(
                        values[0], values[1], gpu_timestamp_.valid_bits);
                    result.gpu_device_ms =
                        static_cast<double>(ticks) * gpu_timestamp_.period_ns / 1'000'000.0;
                    result.gpu_timestamp_samples = 1;
                }
            }
            release_gpu_timestamp();
            // Passed the envelope this batch just measured, so the two instruments are compared
            // against each other on identical work rather than across a run (#2333).
            backend_pass_timing_report(state == BackendSubmissionState::Complete,
                                       result.gpu_timestamp_samples ? result.gpu_device_ms : -1.0);
            vkDestroyFence(dev, fence, nullptr);
            commands_.clear();
            finish_persistent_state(state == BackendSubmissionState::Complete);
        } else {
            // Neither wait proved completion, so every command buffer and object captured by its
            // cleanup may still be in use. Retain the callbacks without invoking or destroying
            // them; leaking these one-shot resources and any borrowed ownership tokens they hold
            // is the only valid last-resort action on an effectively lost device.
            abandon_pending_resources();
        }
        return result;
    }

    void complete() {
        if (pending_resources_abandoned_) return;
        if (commands_.empty()) release_gpu_timestamp();
        for (auto& cleanup : cleanups_) cleanup();
        cleanups_.clear();
    }

private:
    struct GpuTimestamp {
        VkDevice dev = VK_NULL_HANDLE;
        VkQueryPool pool = VK_NULL_HANDLE;
        double period_ns = 0.0;
        uint32_t valid_bits = 0;
        bool ended = false;
    };

    void release_gpu_timestamp() {
        if (gpu_timestamp_.pool)
            vkDestroyQueryPool(gpu_timestamp_.dev, gpu_timestamp_.pool, nullptr);
        gpu_timestamp_ = {};
    }

    void finish_persistent_state(bool completed) {
        if (!completed)
            for (auto cleanup = failure_cleanups_.rbegin();
                 cleanup != failure_cleanups_.rend(); ++cleanup)
                (*cleanup)();
        failure_cleanups_.clear();
    }

    std::vector<VkCommandBuffer> commands_;
    GpuTimestamp gpu_timestamp_;
    std::vector<std::function<void()>> cleanups_;
    std::vector<std::function<void()>> failure_cleanups_;
    bool pending_resources_abandoned_ = false;
};

struct PersistentColorTargetKey {
    uint64_t id = 0;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool operator==(const PersistentColorTargetKey&) const = default;
};

struct PersistentColorTargetKeyHash {
    size_t operator()(const PersistentColorTargetKey& key) const {
        size_t h = std::hash<uint64_t>{}(key.id);
        auto mix = [&](uint32_t value) {
            h ^= static_cast<size_t>(value) + 0x9e3779b9u + (h << 6) + (h >> 2);
        };
        mix(key.width); mix(key.height); mix(static_cast<uint32_t>(key.format));
        return h;
    }
};

struct PersistentColorTargetImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint64_t last_use = 0;
    uint32_t pin_count = 0;
    bool valid = false;
};

inline std::unordered_map<PersistentColorTargetKey, PersistentColorTargetImage,
                          PersistentColorTargetKeyHash>& persistent_color_target_cache() {
    static std::unordered_map<PersistentColorTargetKey, PersistentColorTargetImage,
                              PersistentColorTargetKeyHash> cache;
    return cache;
}

inline VkDeviceSize& persistent_color_target_bytes() {
    static VkDeviceSize bytes = 0;
    return bytes;
}

inline uint64_t& persistent_color_target_generation() {
    static uint64_t generation = 0;
    return generation;
}

// Device-derived residency budget, initialized once from the physical device's largest DEVICE_LOCAL
// heap (see init_persistent_color_target_device_budget). 0 = not yet initialized -> the 256 MiB legacy
// fallback is used until the first render call has a VkPhysicalDevice in hand.
inline VkDeviceSize& persistent_color_target_device_budget() {
    static VkDeviceSize budget = 0;
    return budget;
}

// The historical fixed 256 MiB budget holds only ~4-8 targets at 4K (33 MiB RGBA8 / 66 MiB RGBA16F),
// so a deferred renderer's targets overflow it, are recreated non-resident, and every later sample of
// them falls to CPU detile (the #1177 bottleneck). Since a GPU-resident sample and a CPU detile produce
// identical pixels, growing this budget is correctness-preserving and only trades a bounded amount of
// device memory for far less per-frame CPU work. Precedence: explicit PROSPER_BACKEND_TARGET_CACHE_MB,
// then the device-derived budget, then the 256 MiB fallback.
inline VkDeviceSize persistent_color_target_limit() {
    static const bool have_env = getenv("PROSPER_BACKEND_TARGET_CACHE_MB") != nullptr;
    static const VkDeviceSize env_limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_TARGET_CACHE_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_BACKEND_TARGET_CACHE_MB", value, 256ull,
            UINT64_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    if (have_env) return env_limit;
    const VkDeviceSize dev = persistent_color_target_device_budget();
    return dev ? dev : static_cast<VkDeviceSize>(256ull * 1024ull * 1024ull);
}

// Size the default residency budget to a quarter of the largest device-local heap, clamped to
// [256 MiB, 4 GiB]. On an integrated GPU (shared system RAM, a very large heap) this lands at the 4 GiB
// ceiling; on a small discrete GPU it stays proportional so we never claim more than ~25% of VRAM.
// Idempotent: only the first caller (with a valid device) sets it; PROSPER_BACKEND_TARGET_CACHE_MB
// overrides it entirely (handled in persistent_color_target_limit).
inline void init_persistent_color_target_device_budget(
    const VkPhysicalDeviceMemoryProperties& memp) {
    if (persistent_color_target_device_budget() != 0) return;
    VkDeviceSize heap = 0;
    for (uint32_t i = 0; i < memp.memoryHeapCount; i++)
        if (memp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heap = std::max(heap, memp.memoryHeaps[i].size);
    VkDeviceSize budget = heap / 4;
    budget = std::max<VkDeviceSize>(budget, 256ull * 1024ull * 1024ull);
    budget = std::min<VkDeviceSize>(budget, 4096ull * 1024ull * 1024ull);
    persistent_color_target_device_budget() = budget;
    fprintf(stderr, "[render] persistent color-target residency budget = %llu MiB "
            "(device-local heap %llu MiB)\n",
            (unsigned long long)(budget / (1024ull * 1024ull)),
            (unsigned long long)(heap / (1024ull * 1024ull)));
}

// Sampled images and color targets are independent residency pools. A large immutable source can
// legitimately approach one GiB by itself (for example a maximum-size block-compressed atlas after
// expansion), so the historical fixed 1 GiB image budget can continuously evict the rest of a hot
// set. Keep small GPUs at that old bound while allowing capable devices up to 4 GiB. Unlike an
// allocation, this is only a ceiling; memory is committed on exact-version cache insertion.
inline VkDeviceSize persistent_texture_cache_budget_for_heap(VkDeviceSize heap) {
    constexpr VkDeviceSize min_bytes = 1024ull * 1024ull * 1024ull;
    constexpr VkDeviceSize max_bytes = 4096ull * 1024ull * 1024ull;
    return std::clamp<VkDeviceSize>(heap / 8u, min_bytes, max_bytes);
}

inline VkDeviceSize& persistent_texture_cache_device_budget() {
    static VkDeviceSize budget = 0;
    return budget;
}

inline void init_persistent_texture_cache_device_budget(
    const VkPhysicalDeviceMemoryProperties& memp) {
    if (persistent_texture_cache_device_budget() != 0) return;
    VkDeviceSize heap = 0;
    for (uint32_t i = 0; i < memp.memoryHeapCount; ++i)
        if (memp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heap = std::max(heap, memp.memoryHeaps[i].size);
    persistent_texture_cache_device_budget() =
        persistent_texture_cache_budget_for_heap(heap);
    fprintf(stderr, "[render] persistent texture-image residency budget = %llu MiB "
            "(device-local heap %llu MiB)\n",
            (unsigned long long)(persistent_texture_cache_device_budget() /
                                 (1024ull * 1024ull)),
            (unsigned long long)(heap / (1024ull * 1024ull)));
}

inline VkDeviceSize persistent_texture_cache_limit() {
    static const bool have_env = getenv("PROSPER_BACKEND_TEXTURE_CACHE_MB") != nullptr;
    static const VkDeviceSize env_limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_TEXTURE_CACHE_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_BACKEND_TEXTURE_CACHE_MB", value, 1024ull,
            UINT64_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    if (have_env) return env_limit;
    const VkDeviceSize device_budget = persistent_texture_cache_device_budget();
    return device_budget ? device_budget : 1024ull * 1024ull * 1024ull;
}

// Max number of distinct render targets kept GPU-resident (sampleable) at once. A target that would
// exceed this count is created non-resident (no SAMPLED_BIT), forcing every later sample of it onto the
// CPU detile path. At 4K a deferred renderer (G-buffer MRTs + lighting + post chain) easily needs more
// than the historical 64, so this is configurable via PROSPER_BACKEND_TARGET_CACHE_COUNT (see #1177).
inline size_t persistent_color_target_count_limit() {
    static const size_t count = []() -> size_t {
        const char* value = getenv("PROSPER_BACKEND_TARGET_CACHE_COUNT");
        // Raised from the historical 64 so a 4K deferred renderer's targets are bounded by the memory
        // budget (persistent_color_target_limit) rather than an arbitrarily small count (#1177).
        // `n ? n : 256` refuses a value that parses to 0 and NOTHING else: this is an unsigned
        // parse, so `=-1` saturated to UINT64_MAX, stayed non-zero, and removed the count bound
        // entirely (SIZE_MAX resident targets). The byte twin of this knob two functions up was
        // converted while this one was published as "already refuses" -- it was not (#3267 B1).
        const uint64_t n = prosper::diag::env_u64_or_default_capped(
            "PROSPER_BACKEND_TARGET_CACHE_COUNT", value, 256ull, SIZE_MAX, "targets");
        return n ? static_cast<size_t>(n) : size_t{256};
    }();
    return count;
}

// One ordered frontend callback may record several render groups into one Vulkan submission batch.
// While that batch is open, none of the existing targets can be evicted because an earlier command
// buffer may still reference it. Refusing every new persistent target at the nominal count limit is
// nevertheless incorrect: the new target becomes transient before the frontend can pin it, so a
// producer followed by a later group in the same batch loses the only GPU-authoritative copy. Keep a
// small, bounded admission window while eviction is deferred. Once the batch completes, the normal
// cleanup path prunes unpinned entries back to the configured limit; targets claimed by the frontend
// remain protected until their consumer releases them.
inline size_t persistent_color_target_count_ceiling(bool eviction_deferred) {
    const size_t limit = persistent_color_target_count_limit();
    constexpr size_t kDeferredEvictionHeadroom = 64;
    if (!eviction_deferred) return limit;
    return limit > SIZE_MAX - kDeferredEvictionHeadroom
        ? SIZE_MAX : limit + kDeferredEvictionHeadroom;
}

inline PersistentColorTargetImage* find_persistent_color_target(
    uint64_t id, uint32_t width, uint32_t height, VkFormat format, bool require_valid = true) {
    if (!id) return nullptr;
    format = backend_color_format(format);
    auto& cache = persistent_color_target_cache();
    auto found = cache.find({id, width, height, format});
    if (found == cache.end() || (require_valid && !found->second.valid)) return nullptr;
    return &found->second;
}

// Ordered frontend spans may need a GPU-only target to survive later backend calls until the submit's
// final callback materializes or presents it. Pins are explicit and short-lived; invalidation still
// makes the pixels unusable, while eviction waits until every owner releases the allocation.
inline bool pin_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                        VkFormat format) {
    PersistentColorTargetImage* target = find_persistent_color_target(
        id, width, height, backend_color_format(format));
    if (!target || target->pin_count == UINT32_MAX) return false;
    ++target->pin_count;
    return true;
}

inline bool unpin_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                          VkFormat format) {
    auto& cache = persistent_color_target_cache();
    auto found = cache.find({id, width, height, backend_color_format(format)});
    if (found == cache.end() || !found->second.pin_count) return false;
    --found->second.pin_count;
    return true;
}

// Guest/compute writes invalidate the GPU-produced version without immediately freeing the image.
// The next render may reuse the allocation, but it cannot LOAD or sample stale pixels.
inline void invalidate_persistent_color_target(uint64_t id) {
    if (!id) return;
    for (auto& [key, target] : persistent_color_target_cache())
        if (key.id == id) target.valid = false;
}

inline void invalidate_persistent_color_target_guest_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    const uint64_t end = size > UINT64_MAX - addr ? UINT64_MAX : addr + size;
    for (auto& [key, target] : persistent_color_target_cache()) {
        const uint64_t bytes = static_cast<uint64_t>(key.width) * key.height *
                               backend_color_bytes_per_pixel(key.format);
        const uint64_t target_end = bytes > UINT64_MAX - key.id ? UINT64_MAX : key.id + bytes;
        if (addr < target_end && key.id < end) target.valid = false;
    }
}

// A compute dispatch can publish both exact guest bytes and an equivalent device-local result into
// one borrowed renderer target. Normal guest-write invalidation runs first so every overlapping alias
// becomes stale; only the image proven to have received that result may then regain authority.
inline bool restore_persistent_color_target_after_mirrored_write(
    uint64_t id, uint32_t width, uint32_t height, VkFormat format) {
    PersistentColorTargetImage* target = find_persistent_color_target(
        id, width, height, backend_color_format(format), false);
    if (!target || !target->image || target->layout == VK_IMAGE_LAYOUT_UNDEFINED) return false;
    target->valid = true;
    target->last_use = ++persistent_color_target_generation();
    return true;
}

inline void destroy_persistent_color_target(const RenderVkCtx& ctx,
                                            PersistentColorTargetImage& target) {
    if (target.view) vkDestroyImageView(ctx.dev, target.view, nullptr);
    if (target.image) vkDestroyImage(ctx.dev, target.image, nullptr);
    if (target.memory) vkFreeMemory(ctx.dev, target.memory, nullptr);
    target = {};
}

inline bool readback_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                             VkFormat format, std::vector<uint8_t>& output,
                                             std::string& error);

// With deferred RTT readback (#1284) a valid persistent target can hold the ONLY copy of its
// rendered pixels. Eviction must hand those pixels back to the frontend's CPU cache before the
// image is destroyed, or the content is silently lost. Without a registered sink, eviction keeps
// the historical destroy-only behavior (every pass then still has an authoritative CPU copy).
inline std::function<void(uint64_t, uint32_t, uint32_t, VkFormat, std::vector<uint8_t>&&)>&
persistent_color_target_evict_sink() {
    static std::function<void(uint64_t, uint32_t, uint32_t, VkFormat, std::vector<uint8_t>&&)>
        sink;
    return sink;
}

inline bool evict_persistent_color_target(const RenderVkCtx& ctx, uint64_t current_generation) {
    if (backend_has_unproven_submission()) return false;
    auto& cache = persistent_color_target_cache();
    auto victim = cache.end();
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->second.last_use == current_generation || it->second.pin_count) continue;
        if (victim == cache.end() || it->second.last_use < victim->second.last_use)
            victim = it;
    }
    if (victim == cache.end()) return false;
    if (victim->second.valid) {
        auto& sink = persistent_color_target_evict_sink();
        if (sink) {
            std::vector<uint8_t> pixels;
            std::string error;
            if (readback_persistent_color_target(victim->first.id, victim->first.width,
                                                 victim->first.height, victim->first.format,
                                                 pixels, error))
                sink(victim->first.id, victim->first.width, victim->first.height,
                     victim->first.format, std::move(pixels));
        }
    }
    persistent_color_target_bytes() -= victim->second.bytes;
    destroy_persistent_color_target(ctx, victim->second);
    cache.erase(victim);
    return true;
}

// Transient allocations return to this pool only after their call or explicit submission batch has
// completed, so they can be safely recycled by exact Vulkan memory requirements. Keeping only the
// memory object avoids changing image layouts or descriptor lifetimes while removing the driver's
// expensive allocate/free churn between batches.
struct RenderMemoryKey {
    VkDeviceSize bytes = 0;
    uint32_t memory_type = UINT32_MAX;
    bool operator==(const RenderMemoryKey& other) const {
        return bytes == other.bytes && memory_type == other.memory_type;
    }
};

struct RenderMemoryKeyHash {
    size_t operator()(const RenderMemoryKey& key) const {
        return std::hash<uint64_t>{}(static_cast<uint64_t>(key.bytes)) ^
               (std::hash<uint32_t>{}(key.memory_type) << 1);
    }
};

struct RenderMemoryPool {
    std::mutex mutex;
    std::unordered_map<RenderMemoryKey, std::vector<VkDeviceMemory>, RenderMemoryKeyHash> available;
    std::unordered_map<VkDeviceMemory, RenderMemoryKey> active;
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

struct RenderMemoryPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

inline RenderMemoryPool& render_memory_pool() {
    static RenderMemoryPool pool;
    return pool;
}

inline bool render_memory_pool_enabled() {
    static const bool enabled = getenv("PROSPER_NO_MEMORY_POOL") == nullptr;
    return enabled;
}

inline VkDeviceSize render_memory_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_MEMORY_POOL_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_MEMORY_POOL_MB", value, 512ull,
            UINT64_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

inline VkDeviceMemory allocate_transient_render_memory(VkDevice device, VkDeviceSize bytes,
                                                       uint32_t memory_type) {
    if (memory_type == UINT32_MAX) return VK_NULL_HANDLE;
    RenderMemoryKey key{bytes, memory_type};
    if (render_memory_pool_enabled()) {
        RenderMemoryPool& pool = render_memory_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        auto found = pool.available.find(key);
        if (found != pool.available.end() && !found->second.empty()) {
            VkDeviceMemory memory = found->second.back();
            found->second.pop_back();
            if (found->second.empty()) pool.available.erase(found);
            pool.cached_bytes -= bytes;
            --pool.cached_allocations;
            ++pool.hits;
            pool.active.emplace(memory, key);
            return memory;
        }
        ++pool.misses;
    }

    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = bytes;
    allocation.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (render_memory_pool_enabled()) {
        RenderMemoryPool& pool = render_memory_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.active.emplace(memory, key);
    }
    return memory;
}

inline void release_transient_render_memory(VkDevice device, VkDeviceMemory memory) {
    if (!memory) return;
    if (!render_memory_pool_enabled()) {
        vkFreeMemory(device, memory, nullptr);
        return;
    }

    RenderMemoryPool& pool = render_memory_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto found = pool.active.find(memory);
    if (found == pool.active.end()) {
        vkFreeMemory(device, memory, nullptr);
        return;
    }
    const RenderMemoryKey key = found->second;
    pool.active.erase(found);
    constexpr size_t max_cached_allocations = 4096;
    const VkDeviceSize limit = render_memory_pool_limit();
    const VkDeviceSize remaining = pool.cached_bytes < limit ? limit - pool.cached_bytes : 0;
    if (pool.cached_allocations >= max_cached_allocations ||
        key.bytes > remaining) {
        ++pool.discarded;
        vkFreeMemory(device, memory, nullptr);
        return;
    }
    pool.available[key].push_back(memory);
    pool.cached_bytes += key.bytes;
    ++pool.cached_allocations;
}

inline RenderMemoryPoolStats render_memory_pool_stats() {
    RenderMemoryPool& pool = render_memory_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    return {pool.cached_bytes, pool.cached_allocations, pool.hits, pool.misses, pool.discarded};
}

inline uint32_t render_memory_type(VkPhysicalDevice phys, uint32_t bits,
                                   VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(phys, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return UINT32_MAX;
}

// Storage-buffer contents are rewritten for every synchronous render call, but their Vulkan object
// shapes repeat heavily. Keep capacity-class host-coherent buffers mapped between calls so the hot path
// only copies bytes. The backend normally packs call-local logical uploads into aligned slices of a few
// pooled arenas; the same pool also backs the per-upload fallback. A call or explicit submission batch
// completes before returning buffers, so no in-flight GPU work can observe a later upload. Descriptors
// retain exact logical offsets and ranges, so capacity padding and neighboring arena slices remain
// shader-inaccessible.
struct RenderHostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize bytes = 0;
    VkDeviceSize allocation_bytes = 0;
    // Release order, stamped when the buffer enters the cache. Only meaningful for cached entries;
    // it is what makes eviction least-recently-used rather than arbitrary (#1284).
    uint64_t last_use = 0;
};

struct RenderHostBufferPool {
    // A deque per capacity class, ordered oldest-release at the front. Acquire takes the BACK (the
    // most recently released buffer of that class, so the hottest pages come back first) and
    // eviction takes the FRONT (the least recently released). A vector cannot do both in O(1).
    std::unordered_map<VkDeviceSize, std::deque<RenderHostBuffer>> available;
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    // Monotonic release counter; see RenderHostBuffer::last_use.
    uint64_t release_clock = 0;
};

struct RenderHostBufferPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

inline RenderHostBufferPool& render_host_buffer_pool() {
    static thread_local RenderHostBufferPool pool;
    return pool;
}

inline bool render_host_buffer_pool_enabled() {
    return getenv("PROSPER_NO_BACKEND_BUFFER_POOL") == nullptr;
}

// Which capacity class holds the least-recently-released cached buffer.
//
// Eviction used to take `pool.available.begin()` — an arbitrary `unordered_map` bucket — so under
// pressure the pool discarded whichever class the hash happened to order first, which is very often
// the class about to be needed again. That is the failure mode that survives any budget smaller than
// the working set, so it is fixed independently of the budget (#1284).
//
// Each deque is ordered oldest-release at the front, so only the fronts can be the global oldest and
// the scan is over the number of capacity classes (~20-30 power-of-two sizes), not cached entries.
// Pure over pool state so the policy is unit-testable without a Vulkan device.
inline bool render_host_buffer_pool_lru_key(const RenderHostBufferPool& pool,
                                            VkDeviceSize& key_out) {
    bool found = false;
    uint64_t oldest = 0;
    for (const auto& [capacity, entries] : pool.available) {
        if (entries.empty()) continue;
        const uint64_t stamp = entries.front().last_use;
        if (!found || stamp < oldest) {
            found = true;
            oldest = stamp;
            key_out = capacity;
        }
    }
    return found;
}

// Host physical memory, for the memory-aware pool budget below. Duplicated rather than shared with
// the frontend's identical helper because this header is included BY the frontend, so taking the
// dependency the other way would invert the include order.
inline uint64_t render_host_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

// Budget for retained host-visible staging buffers.
//
// This was a flat 256 MiB, which is not a cache for a 3D title: Blue Prince's per-submit staging
// working set measures 974 MiB across 503 buffers, so the pool ran permanently at its ceiling with
// evictions EXACTLY equal to misses (~216k of each) — one buffer destroyed for every one created.
// Raising it to 2 GiB on that title took the backend submit from 203.06 to 125.98 ms, -34.8 %
// normalised per draw, and dropped evictions to zero (#1284).
//
// Sized as a fraction of host RAM rather than a bigger constant, mirroring
// `texture_decode_cache_limit_bytes`. The floor is the historical 256 MiB, so no host is given LESS
// than before; the ceiling bounds the worst case. An explicit `PROSPER_BACKEND_BUFFER_POOL_MB` wins
// outright, including values below the floor, because it is also the A/B lever and a constrained-host
// escape hatch. Pure and separated from `getenv` so it can be unit-tested across host sizes.
inline VkDeviceSize render_host_buffer_pool_limit_bytes(const char* override_mib,
                                                        uint64_t physical_memory_bytes) {
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    constexpr uint64_t kMinBytes = 256ull * kMiB;
    constexpr uint64_t kMaxBytes = 2048ull * kMiB;
    if (override_mib) {
        const uint64_t mib = strtoull(override_mib, nullptr, 10);
        if (mib > UINT64_MAX / kMiB) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib * kMiB);
    }
    if (!physical_memory_bytes) return static_cast<VkDeviceSize>(kMinBytes);
    uint64_t bytes = std::clamp(physical_memory_bytes / 8u, kMinBytes, kMaxBytes);
    bytes -= bytes % kMiB;
    return static_cast<VkDeviceSize>(bytes);
}

inline VkDeviceSize render_host_buffer_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const uint64_t physical = render_host_physical_memory_bytes();
        const VkDeviceSize bytes = render_host_buffer_pool_limit_bytes(
            getenv("PROSPER_BACKEND_BUFFER_POOL_MB"), physical);
        fprintf(stderr,
                "[render] backend host-buffer pool budget = %.1f MiB (host physical %.1f GiB)\n",
                bytes / (1024.0 * 1024.0), physical / (1024.0 * 1024.0 * 1024.0));
        return bytes;
    }();
    return limit;
}

inline VkDeviceSize render_host_buffer_arena_size() {
    static const VkDeviceSize bytes = []() -> VkDeviceSize {
        // The max(4, ...) floor means a typo does not crash -- it silently builds a FOUR-BYTE
        // arena, which is the worst kind of wrong setting: plausible, survivable, and slow (#3267).
        const char* value = getenv("PROSPER_BACKEND_BUFFER_ARENA_KB");
        const uint64_t kib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_BACKEND_BUFFER_ARENA_KB", value, 1024ull, UINT64_MAX / 1024ull, "KiB");
        return std::max<VkDeviceSize>(4, static_cast<VkDeviceSize>(kib) * 1024ull);
    }();
    return bytes;
}

inline void destroy_render_host_buffer(VkDevice device, RenderHostBuffer& buffer) {
    if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
}

inline RenderHostBuffer acquire_render_host_buffer(const RenderVkCtx& ctx,
                                                   VkDeviceSize bytes) {
    if (!bytes) return {};
    VkDeviceSize capacity = 4;
    while (capacity < bytes && capacity <= UINT64_MAX / 2) capacity *= 2;
    if (capacity < bytes) capacity = bytes;
    RenderHostBufferPool& pool = render_host_buffer_pool();
    auto found = pool.available.find(capacity);
    if (found != pool.available.end() && !found->second.empty()) {
        RenderHostBuffer buffer = found->second.back();
        found->second.pop_back();
        if (found->second.empty()) pool.available.erase(found);
        pool.cached_bytes -= buffer.allocation_bytes;
        --pool.cached_buffers;
        ++pool.hits;
        return buffer;
    }
    ++pool.misses;

    RenderHostBuffer buffer;
    buffer.bytes = capacity;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = capacity;
    // INDEX as well as STORAGE, so one pool serves both. Index data used to take a dedicated
    // vkCreateBuffer + vkAllocateMemory per indexed draw -- 55% of setup_fixed_ms and ~9% of a Blue
    // Prince gameplay frame (#2253, measured by #2252's partition).
    //
    // Widening the usage does not change how existing storage users BIND or WRITE these buffers; it
    // only makes the same memory legal for vkCmdBindIndexBuffer. It is not entirely free of the
    // storage path, though, and the honest statement is about ALLOCATION rather than binding: an
    // added usage bit can only narrow memoryRequirements.memoryTypeBits, so on a hypothetical device
    // with no HOST_VISIBLE type accepting INDEX usage the arena would fail to allocate and the
    // STORAGE path would fall back with it. That is a performance regression rather than a
    // correctness one -- the dedicated-buffer fallback covers both -- and no such device is known.
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    // TRANSFER_SRC only when PROSPER_BUFFER_ECHO is armed. The echo reads these slices back with
    // vkCmdCopyBuffer, and VUID-vkCmdCopyBuffer-srcBuffer-00118 requires the source to carry the
    // bit -- without it the diagnostic is itself invalid Vulkan, which validation reports 10 times
    // on one replay and which no amount of plausible-looking output would have revealed. Gated
    // rather than unconditional for the reason the comment above gives about INDEX: an added usage
    // bit can only narrow memoryRequirements.memoryTypeBits, and the default allocation path must
    // stay exactly as it is when the diagnostic is off.
    static const bool echo_usage = getenv("PROSPER_BUFFER_ECHO") != nullptr;
    if (echo_usage) info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &buffer.buffer) != VK_SUCCESS)
        return {};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer.buffer, &requirements);
    buffer.allocation_bytes = requirements.size;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = render_memory_type(
        ctx.phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocation.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(ctx.dev, &allocation, nullptr, &buffer.memory) != VK_SUCCESS ||
        vkBindBufferMemory(ctx.dev, buffer.buffer, buffer.memory, 0) != VK_SUCCESS ||
        vkMapMemory(ctx.dev, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS) {
        destroy_render_host_buffer(ctx.dev, buffer);
        return {};
    }
    return buffer;
}

inline void release_render_host_buffer(VkDevice device, RenderHostBuffer buffer) {
    if (!buffer.buffer || !buffer.memory || !buffer.mapped) {
        destroy_render_host_buffer(device, buffer);
        return;
    }
    constexpr size_t max_cached_buffers = 4096;
    const VkDeviceSize limit = render_host_buffer_pool_limit();
    if (!limit || buffer.allocation_bytes > limit) {
        destroy_render_host_buffer(device, buffer);
        return;
    }

    std::vector<RenderHostBuffer> evicted;
    RenderHostBufferPool& pool = render_host_buffer_pool();
    while ((pool.cached_buffers >= max_cached_buffers ||
            pool.cached_bytes > limit - buffer.allocation_bytes) &&
           !pool.available.empty()) {
        VkDeviceSize victim_key = 0;
        if (!render_host_buffer_pool_lru_key(pool, victim_key)) break;
        auto victim = pool.available.find(victim_key);
        if (victim == pool.available.end() || victim->second.empty()) break;
        RenderHostBuffer old = victim->second.front();
        victim->second.pop_front();
        if (victim->second.empty()) pool.available.erase(victim);
        pool.cached_bytes -= old.allocation_bytes;
        --pool.cached_buffers;
        ++pool.evictions;
        evicted.push_back(old);
    }
    if (pool.cached_buffers < max_cached_buffers &&
        pool.cached_bytes <= limit - buffer.allocation_bytes) {
        buffer.last_use = ++pool.release_clock;
        pool.available[buffer.bytes].push_back(buffer);
        pool.cached_bytes += buffer.allocation_bytes;
        ++pool.cached_buffers;
        buffer = {};
    }
    for (RenderHostBuffer& old : evicted) destroy_render_host_buffer(device, old);
    if (buffer.buffer) destroy_render_host_buffer(device, buffer);
}

// Deterministic one-shot injection for the fresh storage-buffer upload regression checks. The
// production path never arms this state; keeping it programmatic (rather than environment-driven)
// prevents an accidental live-run setting from manufacturing allocation failures.
enum class RenderBufferUploadFailureStep {
    None,
    CreateBuffer,
    AllocateMemory,
    BindMemory,
    MapMemory,
};

inline RenderBufferUploadFailureStep& render_buffer_upload_failure_once_storage() {
    static thread_local RenderBufferUploadFailureStep step =
        RenderBufferUploadFailureStep::None;
    return step;
}

inline void inject_render_buffer_upload_failure_once(RenderBufferUploadFailureStep step) {
    render_buffer_upload_failure_once_storage() = step;
}

inline bool consume_render_buffer_upload_failure(RenderBufferUploadFailureStep step) {
    RenderBufferUploadFailureStep& armed = render_buffer_upload_failure_once_storage();
    if (armed != step) return false;
    armed = RenderBufferUploadFailureStep::None;
    return true;
}

inline const char* render_buffer_upload_failure_name(RenderBufferUploadFailureStep step) {
    switch (step) {
        case RenderBufferUploadFailureStep::CreateBuffer: return "vkCreateBuffer";
        case RenderBufferUploadFailureStep::AllocateMemory: return "vkAllocateMemory";
        case RenderBufferUploadFailureStep::BindMemory: return "vkBindBufferMemory";
        case RenderBufferUploadFailureStep::MapMemory: return "vkMapMemory";
        default: return "none";
    }
}

// Deterministic one-shot injection for the #3045 regression check: simulate the NEXT sampled-
// texture vkCreateImage call failing, without depending on any real device's limits. A real
// over-large arrayLayers request cannot be relied on to fail here -- this box's RADV reports
// maxImageArrayLayers=8192, well above the backend's own kBackendMaxArrayLayers ceiling of 2048,
// so the array-layer path never actually reaches the driver's own rejection. The production path
// never arms this state.
inline bool& render_texture_create_failure_once_storage() {
    static thread_local bool armed = false;
    return armed;
}

inline void inject_render_texture_create_failure_once() {
    render_texture_create_failure_once_storage() = true;
}

inline bool consume_render_texture_create_failure_once() {
    bool& armed = render_texture_create_failure_once_storage();
    if (!armed) return false;
    armed = false;
    return true;
}

// #3180: the COLOR-TARGET vkCreateImage sites, the milder siblings of #3045's two.
//
// Each of the six already guarded the HANDLE before touching it further (`if (!img) return out;`),
// so unlike #3045 none of them fed a definitely-null image into a Vulkan call. What they did do is
// discard the VkResult, which left two smaller problems:
//
//   * the guard rested on a driver CONVENTION, not on the spec. Every implementation leaves
//     `pImage` at VK_NULL_HANDLE when vkCreateImage fails, but Vulkan does not require it, so a
//     conforming driver that left the slot untouched would walk straight past `if (!img)`. Testing
//     the result FIRST and the handle second removes that dependency at no cost.
//   * a failure was silent. The pass returned an empty frame with nothing in the run log, so
//     "the device refused a 4K attachment" and "the draw list was empty" produced the same
//     evidence, and the real VkResult -- the one fact that separates them -- was thrown away.
//
// The control flow is deliberately unchanged: every site still returns from the pass exactly where
// it did. This adds a report, and one rate-limited counter shared by all six so a device failing
// every target cannot flood a run log.
enum class RenderColorTargetCreateSite : uint32_t {
    None = 0,
    Slot0,              // primary slot-0 (the pass's own color target)
    Slot0Fallback,      // slot-0 retry as a transient image after a persistent-cache budget miss
    Slot1,              // primary slot-1 (MRT1's independent guest identity)
    Slot1Fallback,      // slot-1's budget-miss retry
    SlotExtra,          // primary MRT slots 2..7
    SlotExtraFallback,  // an MRT slot's budget-miss retry
};

inline const char* render_color_target_site_name(RenderColorTargetCreateSite site) {
    switch (site) {
        case RenderColorTargetCreateSite::Slot0:             return "slot0";
        case RenderColorTargetCreateSite::Slot0Fallback:     return "slot0-fallback";
        case RenderColorTargetCreateSite::Slot1:             return "slot1";
        case RenderColorTargetCreateSite::Slot1Fallback:     return "slot1-fallback";
        case RenderColorTargetCreateSite::SlotExtra:         return "slot-extra";
        case RenderColorTargetCreateSite::SlotExtraFallback: return "slot-extra-fallback";
        default:                                             return "none";
    }
}

// Deterministic one-shot injection, the site-selective twin of the sampled-texture hook above and
// for the same reason: these requests are ordinary 2D color attachments at the caller's own extent,
// so no real device can be made to refuse one on demand. Arming a site makes the NEXT create at
// exactly that site report VK_ERROR_OUT_OF_DEVICE_MEMORY without calling the driver. The production
// path never arms this state.
inline RenderColorTargetCreateSite& render_color_target_create_failure_storage() {
    static thread_local RenderColorTargetCreateSite armed = RenderColorTargetCreateSite::None;
    return armed;
}

inline void inject_render_color_target_create_failure_once(RenderColorTargetCreateSite site) {
    render_color_target_create_failure_storage() = site;
}

inline bool consume_render_color_target_create_failure(RenderColorTargetCreateSite site) {
    RenderColorTargetCreateSite& armed = render_color_target_create_failure_storage();
    if (armed != site || site == RenderColorTargetCreateSite::None) return false;
    armed = RenderColorTargetCreateSite::None;
    return true;
}

// Create a color target, reporting the driver's own verdict. Returns VK_SUCCESS with a non-null
// `out_image`, or a failure code with `out_image` forced to VK_NULL_HANDLE so the caller's existing
// handle guard stays correct whatever the driver left behind.
inline VkResult create_color_target_image(VkDevice dev, const VkImageCreateInfo& ci,
                                          RenderColorTargetCreateSite site, VkImage* out_image) {
    const VkResult result = consume_render_color_target_create_failure(site)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateImage(dev, &ci, nullptr, out_image);
    if (result == VK_SUCCESS && *out_image) return VK_SUCCESS;
    *out_image = VK_NULL_HANDLE;
    static std::atomic<uint32_t> color_target_create_failure_logs{0};
    if (color_target_create_failure_logs.fetch_add(1, std::memory_order_relaxed) < 32)
        std::fprintf(stderr,
                     "[color-target-create-failed] site=%s vkCreateImage result=%d extent=%ux%u "
                     "fmt=%d usage=0x%x -- dropping the pass\n",
                     render_color_target_site_name(site), (int)result,
                     ci.extent.width, ci.extent.height, (int)ci.format, (unsigned)ci.usage);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

// #3210: the NON-image object creates in this pass -- render pass, framebuffer, image view,
// sampler, shader module. Three severities share one mechanism here, and the split matters:
//
//   * `rp` and `fb` were declared WITHOUT an initializer and used with no guard whatsoever. Vulkan
//     writes an output handle only on success, so a failed create left them holding an
//     INDETERMINATE value -- not the null the #3180/#3045 sites conventionally get, an arbitrary
//     stack word. That word reached vkCreateFramebuffer, every pipeline's `renderPass`,
//     vkCmdBeginRenderPass and, worst, vkDestroyRenderPass. Destroying a non-handle is undefined.
//     Forcing the output null BEFORE the call (below) removes the indeterminate read at its source,
//     whatever the driver leaves behind; the caller then drops the pass.
//   * the image views and samplers behind a descriptor set ARE value-initialized (dview's
//     declaration, SharedTextureBinding's members), so those were deterministic -- but unchecked,
//     and a null view still reached VkDescriptorImageInfo and vkUpdateDescriptorSets. Two of them
//     also CACHED the result, so one transient failure was retained in
//     PersistentTextureImage::bindings and re-served to every later draw with the same key. The
//     callers now break before the cache insert, which is the sole writer of that map, so no null
//     can enter it and the hit path needs no guard of its own.
//   * vkCreateShaderModule's result was discarded, but `m` is initialized and `!v.vs || !v.fs`
//     already skips the draw. Folded in for the report only; the control flow is untouched.
//
// The report format matches #3180's: site name, the API, the real VkResult, and enough of the
// request to tell an out-of-memory apart from a rejected description.
enum class RenderVkObjectCreateSite : uint32_t {
    None = 0,
    RenderPass,               // the pass's own vkCreateRenderPass
    Framebuffer,              // vkCreateFramebuffer over the attachment views
    DepthStencilView,         // the depth/stencil attachment's view
    TextureViewPersistent,    // a sampled-texture view created for the persistent binding cache
    TextureSamplerPersistent, // ...and its sampler
    TextureView,              // a sampled/storage view created for a transient binding
    TextureSampler,           // ...and its sampler (never created for a storage image)
    ShaderModule,             // any of a draw's VS/GS/FS modules
};

inline const char* render_vk_object_site_name(RenderVkObjectCreateSite site) {
    switch (site) {
        case RenderVkObjectCreateSite::RenderPass:               return "render-pass";
        case RenderVkObjectCreateSite::Framebuffer:              return "framebuffer";
        case RenderVkObjectCreateSite::DepthStencilView:         return "ds-view";
        case RenderVkObjectCreateSite::TextureViewPersistent:    return "texture-view-persistent";
        case RenderVkObjectCreateSite::TextureSamplerPersistent: return "texture-sampler-persistent";
        case RenderVkObjectCreateSite::TextureView:              return "texture-view";
        case RenderVkObjectCreateSite::TextureSampler:           return "texture-sampler";
        case RenderVkObjectCreateSite::ShaderModule:             return "shader-module";
        default:                                                 return "none";
    }
}

// Deterministic one-shot injection, the same shape and the same reason as the color-target hook
// above: these are ordinary render passes, framebuffers, 2D views, trilinear samplers and small
// SPIR-V modules, so no real device can be made to refuse one on demand. Arming a site makes the
// NEXT create at exactly that site report VK_ERROR_OUT_OF_DEVICE_MEMORY without calling the driver.
// The production path never arms this state.
inline RenderVkObjectCreateSite& render_vk_object_create_failure_storage() {
    static thread_local RenderVkObjectCreateSite armed = RenderVkObjectCreateSite::None;
    return armed;
}

inline void inject_render_vk_object_create_failure_once(RenderVkObjectCreateSite site) {
    render_vk_object_create_failure_storage() = site;
}

inline bool consume_render_vk_object_create_failure(RenderVkObjectCreateSite site) {
    RenderVkObjectCreateSite& armed = render_vk_object_create_failure_storage();
    if (armed != site || site == RenderVkObjectCreateSite::None) return false;
    armed = RenderVkObjectCreateSite::None;
    return true;
}

// One counter for every site in this family, so a device refusing everything cannot flood a run
// log. Deliberately separate from the color-target counter: exhausting one must not silence the
// other, since they answer different questions about the same frame.
inline bool render_vk_object_create_failure_should_log() {
    static std::atomic<uint32_t> logs{0};
    return logs.fetch_add(1, std::memory_order_relaxed) < 32;
}

// Every helper below forces its output handle to VK_NULL_HANDLE on entry and again on failure, and
// returns a non-VK_SUCCESS code whenever the handle is not usable. VK_ERROR_INITIALIZATION_FAILED
// stands in for the "driver returned success and left the handle null" case that no spec-conforming
// implementation should produce but that the caller must not walk past either.
inline VkResult create_render_pass_checked(VkDevice dev, const VkRenderPassCreateInfo& ci,
                                           VkRenderPass* out_pass) {
    *out_pass = VK_NULL_HANDLE;
    const VkResult result = consume_render_vk_object_create_failure(
                                RenderVkObjectCreateSite::RenderPass)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateRenderPass(dev, &ci, nullptr, out_pass);
    if (result == VK_SUCCESS && *out_pass) return VK_SUCCESS;
    *out_pass = VK_NULL_HANDLE;
    if (render_vk_object_create_failure_should_log())
        std::fprintf(stderr,
                     "[render-object-create-failed] site=%s vkCreateRenderPass result=%d "
                     "attachments=%u subpasses=%u deps=%u -- dropping the pass\n",
                     render_vk_object_site_name(RenderVkObjectCreateSite::RenderPass), (int)result,
                     ci.attachmentCount, ci.subpassCount, ci.dependencyCount);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

inline VkResult create_framebuffer_checked(VkDevice dev, const VkFramebufferCreateInfo& ci,
                                           VkFramebuffer* out_fb) {
    *out_fb = VK_NULL_HANDLE;
    const VkResult result = consume_render_vk_object_create_failure(
                                RenderVkObjectCreateSite::Framebuffer)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateFramebuffer(dev, &ci, nullptr, out_fb);
    if (result == VK_SUCCESS && *out_fb) return VK_SUCCESS;
    *out_fb = VK_NULL_HANDLE;
    if (render_vk_object_create_failure_should_log())
        std::fprintf(stderr,
                     "[render-object-create-failed] site=%s vkCreateFramebuffer result=%d "
                     "extent=%ux%u attachments=%u -- dropping the pass\n",
                     render_vk_object_site_name(RenderVkObjectCreateSite::Framebuffer), (int)result,
                     ci.width, ci.height, ci.attachmentCount);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

inline VkResult create_render_image_view_checked(VkDevice dev, const VkImageViewCreateInfo& ci,
                                                 RenderVkObjectCreateSite site,
                                                 VkImageView* out_view) {
    *out_view = VK_NULL_HANDLE;
    const VkResult result = consume_render_vk_object_create_failure(site)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateImageView(dev, &ci, nullptr, out_view);
    if (result == VK_SUCCESS && *out_view) return VK_SUCCESS;
    *out_view = VK_NULL_HANDLE;
    if (render_vk_object_create_failure_should_log())
        std::fprintf(stderr,
                     "[render-object-create-failed] site=%s vkCreateImageView result=%d "
                     "fmt=%d viewType=%d mips=%u layers=%u\n",
                     render_vk_object_site_name(site), (int)result, (int)ci.format,
                     (int)ci.viewType, ci.subresourceRange.levelCount,
                     ci.subresourceRange.layerCount);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

inline VkResult create_render_sampler_checked(VkDevice dev, const VkSamplerCreateInfo& ci,
                                              RenderVkObjectCreateSite site,
                                              VkSampler* out_sampler) {
    *out_sampler = VK_NULL_HANDLE;
    const VkResult result = consume_render_vk_object_create_failure(site)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateSampler(dev, &ci, nullptr, out_sampler);
    if (result == VK_SUCCESS && *out_sampler) return VK_SUCCESS;
    *out_sampler = VK_NULL_HANDLE;
    if (render_vk_object_create_failure_should_log())
        std::fprintf(stderr,
                     "[render-object-create-failed] site=%s vkCreateSampler result=%d "
                     "mag=%d min=%d aniso=%d\n",
                     render_vk_object_site_name(site), (int)result, (int)ci.magFilter,
                     (int)ci.minFilter, (int)ci.anisotropyEnable);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

inline VkResult create_render_shader_module_checked(VkDevice dev,
                                                    const VkShaderModuleCreateInfo& ci,
                                                    VkShaderModule* out_module) {
    *out_module = VK_NULL_HANDLE;
    const VkResult result = consume_render_vk_object_create_failure(
                                RenderVkObjectCreateSite::ShaderModule)
        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : vkCreateShaderModule(dev, &ci, nullptr, out_module);
    if (result == VK_SUCCESS && *out_module) return VK_SUCCESS;
    *out_module = VK_NULL_HANDLE;
    if (render_vk_object_create_failure_should_log())
        std::fprintf(stderr,
                     "[render-object-create-failed] site=%s vkCreateShaderModule result=%d "
                     "bytes=%zu -- skipping draw\n",
                     render_vk_object_site_name(RenderVkObjectCreateSite::ShaderModule),
                     (int)result, (size_t)ci.codeSize);
    return result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result;
}

// Create, populate, and unmap one transient storage buffer. Every partial failure tears down in
// Vulkan lifetime order (buffer before any bound memory) and leaves both outputs null. Returning the
// exact failed step lets the caller report the binding before substituting a safe zero-word buffer.
inline RenderBufferUploadFailureStep create_transient_storage_buffer_upload(
        const RenderVkCtx& ctx, const void* source, VkDeviceSize bytes,
        VkBuffer& buffer, VkDeviceMemory& memory) {
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    auto fail = [&](RenderBufferUploadFailureStep step) {
        if (buffer) vkDestroyBuffer(ctx.dev, buffer, nullptr);
        if (memory) release_transient_render_memory(ctx.dev, memory);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return step;
    };

    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // See the pooled path: TRANSFER_SRC only while PROSPER_BUFFER_ECHO is armed, so the echo's
    // vkCmdCopyBuffer is legal and the default allocation is untouched.
    static const bool echo_usage = getenv("PROSPER_BUFFER_ECHO") != nullptr;
    if (echo_usage) info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkResult create_result =
        consume_render_buffer_upload_failure(RenderBufferUploadFailureStep::CreateBuffer)
            ? VK_ERROR_OUT_OF_DEVICE_MEMORY
            : vkCreateBuffer(ctx.dev, &info, nullptr, &buffer);
    if (create_result != VK_SUCCESS) {
        // A failed create has no live object; the output value is not a destroyable handle.
        buffer = VK_NULL_HANDLE;
        return RenderBufferUploadFailureStep::CreateBuffer;
    }
    if (!buffer) return fail(RenderBufferUploadFailureStep::CreateBuffer);

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer, &requirements);
    const uint32_t memory_type = render_memory_type(
        ctx.phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const bool inject_allocation_failure =
        consume_render_buffer_upload_failure(RenderBufferUploadFailureStep::AllocateMemory);
    if (memory_type != UINT32_MAX && !inject_allocation_failure)
        memory = allocate_transient_render_memory(ctx.dev, requirements.size, memory_type);
    if (!memory) return fail(RenderBufferUploadFailureStep::AllocateMemory);

    const VkResult bind_result =
        consume_render_buffer_upload_failure(RenderBufferUploadFailureStep::BindMemory)
            ? VK_ERROR_OUT_OF_DEVICE_MEMORY
            : vkBindBufferMemory(ctx.dev, buffer, memory, 0);
    if (bind_result != VK_SUCCESS) return fail(RenderBufferUploadFailureStep::BindMemory);

    void* mapped = nullptr;
    const VkResult map_result =
        consume_render_buffer_upload_failure(RenderBufferUploadFailureStep::MapMemory)
            ? VK_ERROR_MEMORY_MAP_FAILED
            : vkMapMemory(ctx.dev, memory, 0, bytes, 0, &mapped);
    if (map_result != VK_SUCCESS || !mapped)
        return fail(RenderBufferUploadFailureStep::MapMemory);
    std::memcpy(mapped, source, static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.dev, memory);
    return RenderBufferUploadFailureStep::None;
}

inline const RenderHostBuffer& render_internal_gds_buffer() {
    static const RenderHostBuffer buffer = [] {
        constexpr VkDeviceSize kGdsBytes = 64u * 1024u;
        RenderHostBuffer result = acquire_render_host_buffer(render_vk_ctx(), kGdsBytes);
        if (result.mapped) std::memset(result.mapped, 0, static_cast<size_t>(kGdsBytes));
        return result;
    }();
    return buffer;
}

inline void reset_internal_gds_for_test() {
    const RenderHostBuffer& buffer = render_internal_gds_buffer();
    if (buffer.mapped) std::memset(buffer.mapped, 0, 64u * 1024u);
}

inline uint32_t read_internal_gds_for_test(uint32_t byte_offset) {
    const RenderHostBuffer& buffer = render_internal_gds_buffer();
    if (!buffer.mapped || byte_offset > 64u * 1024u - sizeof(uint32_t)) return 0;
    uint32_t value = 0;
    std::memcpy(&value, static_cast<const uint8_t*>(buffer.mapped) + byte_offset, sizeof(value));
    return value;
}

inline RenderHostBufferPoolStats render_host_buffer_pool_stats() {
    RenderHostBufferPool& pool = render_host_buffer_pool();
    return {pool.cached_bytes, pool.cached_buffers, pool.hits, pool.misses, pool.evictions};
}

// A depth/stencil surface's identity includes WHICH ARRAY SLICE the attachment view selects, not
// only its allocation bases. A cube shadow map is ONE six-layer guest allocation: all six faces share
// a base and the guest picks the face with DB_DEPTH_VIEW.SLICE_START/MAX, programming
// 0x02000000, 0x02002001, 0x02004002 … 0x0200a005 for slices 0..5 against the same base (GTA V,
// verified per-cube on a live route).
//
// Omitting the slice made every face of a cube collide on one key, so each face render REUSED and
// overwrote the previous face's host image and only the last face survived. That is a corruption of
// retained depth, independent of whether anything samples it: no consumer could recover a valid cube
// from the cache however the sampling gate were relaxed.
//
// Non-layered surfaces program DB_DEPTH_VIEW=0 and therefore key at slice 0 exactly as before, so
// this widens identity only where the guest actually used layers.
struct PersistentDsKey {
    uint64_t dr = 0, dw = 0, sr = 0, sw = 0, htile = 0;
    uint32_t w = 0, h = 0, fmt = 0, slice = 0;
    // NOTE: the layer stride is deliberately NOT here. It used to be, as a `slice_bytes` member
    // excluded from operator== and from the hash -- which made this struct two things at once: an
    // identity, and a carrier for a value that identity ignored. That is not a style objection, it
    // is where a bug lived. The cache is keyed by this struct, so the key STORED in the map is the
    // one from whichever attachment created the entry, and the invalidation loop iterates those
    // stored keys. A producer that programmed DB_DEPTH_SLICE only after the entry existed therefore
    // never reached the invalidation at all, and one that reprogrammed it was ignored in favour of
    // the frozen first value -- silently, because an excluded field cannot cause a lookup to miss.
    //
    // The stride now lives in `ds_layer_stride_registry()`, keyed by the surface identity, so the
    // invalidation reads the latest observation rather than one frozen on the key.
    bool operator==(const PersistentDsKey& o) const {
        return dr == o.dr && dw == o.dw && sr == o.sr && sw == o.sw && htile == o.htile &&
               w == o.w && h == o.h && fmt == o.fmt && slice == o.slice;
    }
};

// Layer strides learned from a CONSUMER descriptor (a cube/array T#'s own layer_stride), which is
// the ONLY authority this file trusts for a layer stride.
//
// WHY THERE IS NO PRODUCER AUTHORITY HERE. An earlier revision of this branch also derived a stride
// from DB_DEPTH_SLICE.SLICE_TILE_MAX and gave it PRECEDENCE over the consumer's exact value, at
// CONFIDENCE: HIGH. That was wrong three times over and is removed:
//   * `DB_DEPTH_SLICE` at dword 0x17 with a SLICE_TILE_MAX field is the gfx6-8 register. The PS5 is
//     RDNA2 (gfx10), where the depth extent moved to DB_DEPTH_SIZE_XY at 0x7 -- which prosper itself
//     decodes, in `render_state.cpp`. Nothing in the repo consumed 0x16/0x17 before that revision.
//   * even on gfx6-8, (SLICE_TILE_MAX + 1) * 64 is a count of PIXELS, not of bytes.
//   * GTA V reads 0x00000000 on every depth surface in a routed boot, cubes included, which is
//     equally consistent with "the register is not programmed" and with "the register is not there".
//     A zero cannot distinguish those, so it is not evidence for the decode.
// A consumer layer_stride arrives with a SAMPLE, i.e. after the invalidation that would have wanted
// it, so the first cube sample is what makes invalidation slice-exact; before that it falls back to
// whole-allocation behaviour. That fallback is the safe direction (over-invalidation costs a
// re-render; under-invalidation shows stale pixels), which is exactly why an unvalidated decode that
// OUTRANKS an exact value is the wrong trade. Re-adding a producer authority needs the gfx10
// register, its units, and a title that actually programs it. See #2669.
//
// The IDENTITY of a layer-stride observation: (base, width, height), never `base` alone.
//
// A guest base is not a stable identity. The guest frees an allocation and maps another at the same
// address, and an entry keyed by base alone then answers for a surface that no longer exists. That
// is not bookkeeping pedantry: the stride sizes the range an invalidation covers, so a stride
// carried across a reuse either leaves stale pixels resident (under-invalidation) or evicts a
// neighbouring slice that was still good (over-invalidation). The previous version handled reuse by
// taking whichever observation came LAST -- a heuristic in the place an identity belongs, which
// cannot tell "the allocation was recycled" from "two live views disagree", and so reported the
// first as loudly as the second while silently absorbing neither correctly.
//
// Both sides derive width/height from the same numbers. The consumer publishes the dimensions it
// reads the surface with -- exactly the `tw`/`th` it passes to `read_persistent_ds_cube_depth` --
// and the lookup asks with the attachment's own `key.w`/`key.h`. Format is deliberately NOT part of
// the identity: the producer's DS format enum and a consumer T#'s format are different value
// spaces, so including it would make every lookup miss and silently disable per-slice invalidation
// rather than sharpen it. Width and height are enough to separate a recycled allocation from a
// genuine disagreement, which is the whole job here.
struct DsLayerStrideKey {
    uint64_t base = 0;
    uint32_t width = 0, height = 0;
    bool operator==(const DsLayerStrideKey& o) const {
        return base == o.base && width == o.width && height == o.height;
    }
};
struct DsLayerStrideKeyHash {
    size_t operator()(const DsLayerStrideKey& k) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { hash ^= static_cast<size_t>(v); hash *= 1099511628211ull; };
        mix(k.base); mix(k.width); mix(k.height);
        return hash;
    }
};

// One entry per identity. A single authority, so there is no precedence to get wrong.
struct DsLayerStrideEntry {
    uint64_t consumer_stride = 0;
    bool consumer_conflict_reported = false;
};

inline std::unordered_map<DsLayerStrideKey, DsLayerStrideEntry, DsLayerStrideKeyHash>&
ds_layer_stride_registry() {
    static std::unordered_map<DsLayerStrideKey, DsLayerStrideEntry, DsLayerStrideKeyHash> strides;
    return strides;
}
inline std::mutex& ds_layer_stride_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Records a stride learned from a CONSUMER descriptor. With a complete identity, allocation reuse no
// longer collides here at all -- a new surface at a recycled base has its own entry. A disagreement
// that survives the identity is therefore a genuine decode error, not reuse, and is reported once.
// The later value is taken, but that choice is arbitrary between two claims we cannot adjudicate:
// the REPORT is the mitigation, not the pick.
inline void note_ds_layer_stride(uint64_t base, uint32_t width, uint32_t height, uint64_t stride) {
    if (!base || !stride) return;
    const DsLayerStrideKey key{base, width, height};
    std::lock_guard<std::mutex> lock(ds_layer_stride_mutex());
    DsLayerStrideEntry& entry = ds_layer_stride_registry()[key];
    if (entry.consumer_stride && entry.consumer_stride != stride) {
        if (!entry.consumer_conflict_reported) {
            entry.consumer_conflict_reported = true;
            std::fprintf(stderr,
                         "[ds-stride] consumer disagreement at base=0x%llx %ux%u: had %llu, now told "
                         "%llu; taking the later value (identity already separates allocation reuse, "
                         "so this is a decode error)\n",
                         (unsigned long long)base, width, height,
                         (unsigned long long)entry.consumer_stride, (unsigned long long)stride);
        }
    }
    entry.consumer_stride = stride;
}

// Which of a DS surface's two depth bases identifies it in the stride registry.
//
// This binds every site that HAS both bases -- today that is the invalidation lookup, which reads
// them off `PersistentDsKey`. A surface written but never read has depth_read_base == 0, so a site
// picking the read base and a site picking the write base would never meet, and per-slice
// invalidation would silently never engage for exactly those surfaces. That divergence cannot be
// spotted by reading either site alone, which is why this is a function and not a convention.
//
// The consumer recorder in the live renderer is NOT such a site and does not call this: it holds one
// address, the guest address it sampled, and publishes under that. An earlier version of this
// comment claimed "both the recorder and the lookup MUST use this one expression", which was never
// true of the recorder -- an invariant asserted over a site that could not obey it. If a recorder
// ever does gain both bases, route it through here.
inline uint64_t ds_stride_identity_base(uint64_t depth_read_base, uint64_t depth_write_base) {
    return depth_read_base ? depth_read_base : depth_write_base;
}

// The layer stride for a surface identity, or 0 when no consumer descriptor has published one -- which
// callers must read as "unknown" and fall back to whole-allocation behaviour, never as a stride of
// zero. A guessed stride would silently retain faces a real write had invalidated, which is the one
// error direction that shows as stale pixels rather than as a missing surface.
//
// There is exactly one authority: a consumer descriptor's layer_stride. See the note above the
// registry for why the DB_DEPTH_SLICE-derived producer stride was removed rather than demoted.
inline uint64_t ds_layer_stride_for(uint64_t base, uint32_t width, uint32_t height) {
    if (!base) return 0;
    const DsLayerStrideKey key{base, width, height};
    std::lock_guard<std::mutex> lock(ds_layer_stride_mutex());
    const auto found = ds_layer_stride_registry().find(key);
    return found == ds_layer_stride_registry().end() ? 0 : found->second.consumer_stride;
}

// DB_DEPTH_VIEW.SLICE_START, including its two high bits (SLICE_START occupies bits 0..10 with
// SLICE_START_HI at 11..12; SLICE_MAX is bits 13..23). A single-face attachment programs
// START == MAX, which is what a cube face render does.
inline uint32_t ds_depth_view_slice_start(uint32_t db_depth_view) {
    return (db_depth_view & prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_START_MASK) |
           (((db_depth_view >> prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_START_HI_SHIFT) &
             prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_START_HI_MASK)
            << prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_START_HI_SHIFT);
}
// The production construction seam for a depth/stencil identity. Extracted so a test can exercise
// the SAME derivation the backend uses: a regression that builds PersistentDsKey by hand asserts
// only that the struct has a slice field, and stays green if the decode below is reverted.
inline PersistentDsKey persistent_ds_key_for(const prosper::gpu::ResolvedPipelineState& ps,
                                             uint64_t htile, uint32_t width, uint32_t height,
                                             uint32_t format);

inline uint32_t ds_depth_view_slice_max(uint32_t db_depth_view) {
    return (db_depth_view >> prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_MAX_SHIFT) &
           prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_MAX_MASK;
}

struct PersistentDsKeyHash {
    size_t operator()(const PersistentDsKey& k) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { hash ^= static_cast<size_t>(v); hash *= 1099511628211ull; };
        mix(k.dr); mix(k.dw); mix(k.sr); mix(k.sw); mix(k.htile);
        mix(k.w); mix(k.h); mix(k.fmt); mix(k.slice);
        return hash;
    }
};

struct PersistentDsImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool layout_initialized = false;
    bool depth_valid = false;
    bool stencil_valid = false;
    uint64_t last_depth_write = 0;   // sampled-bridge recency (#1275)
    uint64_t last_depth_command_order = 0;
    uint64_t last_depth_present = UINT64_MAX;
};

inline uint64_t& persistent_ds_write_generation() {
    static uint64_t generation = 0;
    return generation;
}

// DB_RENDER_CONTROL.DEPTH_CLEAR_ENABLE substitutes the VALUE of depth writes (DB_DEPTH_CLEAR); it
// does not create writes on its own — the write path still requires Z_ENABLE + Z_WRITE_ENABLE and a
// compare that can pass. Real guest clear draws program exactly that shape (test+write+ALWAYS);
// Blue Prince's per-light shadow loop instead issues a fullscreen rect with DEPTH_CLEAR_ENABLE set
// and DB_DEPTH_CONTROL fully disabled immediately BEFORE sampling the plane its shadow casters
// rendered — treating that as a clear/write destroyed the shadow map it is about to consume and
// banded every light (#1287). The guest's own draw order proves the writes-disabled shape must be
// depth-inert. CONFIDENCE: MED (write-path gating; the exercised title shapes are covered).
constexpr bool depth_clear_effective(bool clear_enabled, bool test_enabled, bool write_enabled,
                                     uint32_t compare_op) {
    return clear_enabled && test_enabled && write_enabled && compare_op != VK_COMPARE_OP_NEVER;
}

constexpr bool persistent_ds_pass_may_write_depth(bool clear_enabled, bool test_enabled,
                                                  bool write_enabled, uint32_t compare_op) {
    (void)clear_enabled;   // an effective clear already satisfies the write-path clause below
    return test_enabled && write_enabled && compare_op != VK_COMPARE_OP_NEVER;
}

// STENCIL_CLEAR_ENABLE is the stencil twin of the rule above (#1355): the bit substitutes the
// VALUE of stencil writes and only acts through the enabled stencil write path — STENCIL_ENABLE
// plus a nonzero write mask (driver stencil-clear draws program enable+ALWAYS+REPLACE+full mask).
// Op-level analysis (a KEEP-everywhere draw writes nothing) is deliberately omitted: requiring
// only enable+mask errs toward honoring clears, the safe direction for real guest clear shapes.
// CONFIDENCE: MED (write-path gating by analogy with the #1352 title evidence; no title observed
// exercising the writes-disabled stencil shape).
constexpr bool stencil_clear_effective(bool clear_enabled, bool stencil_enabled,
                                       uint32_t write_mask_front, uint32_t write_mask_back) {
    return clear_enabled && stencil_enabled && ((write_mask_front | write_mask_back) != 0u);
}

inline void note_persistent_ds_depth_write(PersistentDsImage& image, bool use_depth,
                                           bool depth_may_be_written,
                                           uint64_t command_order = 0) {
    if (use_depth && depth_may_be_written) {
        image.last_depth_write = ++persistent_ds_write_generation();
        image.last_depth_command_order = command_order;
        image.last_depth_present = prosper::gpu::present_count();
    }
}

inline PersistentDsKey persistent_ds_key_for(const prosper::gpu::ResolvedPipelineState& ps,
                                             uint64_t htile, uint32_t width, uint32_t height,
                                             uint32_t format) {
    // A census of the RAW dword only. It deliberately DERIVES NOTHING: the question this instrument
    // exists to answer is whether any title programs 0x17 at all on a gfx10 part, and a decode
    // printed beside the raw value would pre-judge exactly that. Observation is evidence; a decode
    // is a claim. (#2669)
    if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS")) {
        static std::mutex mutex;
        // (base, width, HEIGHT, raw) -- height was missing, so two surfaces sharing a base and width
        // but differing in height collapsed to one line and the second never printed. A census that
        // silently under-reports is the instrument lying about the subject.
        static std::set<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> seen;
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            first = seen.emplace(ps.depth_read_base, width, height, ps.db_depth_slice).second;
        }
        if (first)
            fprintf(stderr, "[ds-slice-census] base=0x%llx %ux%u db_depth_slice(0x17)=0x%08x\n",
                    (unsigned long long)ps.depth_read_base, width, height, ps.db_depth_slice);
    }
    return {ps.depth_read_base, ps.depth_write_base, ps.stencil_read_base, ps.stencil_write_base,
            htile, width, height, format, ds_depth_view_slice_start(ps.db_depth_view)};
}

inline std::unordered_map<PersistentDsKey, PersistentDsImage, PersistentDsKeyHash>&
persistent_ds_cache() {
    static std::unordered_map<PersistentDsKey, PersistentDsImage, PersistentDsKeyHash> cache;
    return cache;
}

// Sampled depth-plane lookup (#1275): a shadow-map / depth-pyramid T# addresses the DEPTH plane of
// a surface prosper rendered into a persistent Vulkan DS image and never wrote back to guest
// memory. Resolve that address (read or write base — the guest aliases both at the same plane) to
// the retained image so the consumer can bind it directly. Extent must match the T# exactly; only
// a valid, initialized depth plane may be sampled.
// The STENCIL plane is bridged on the same terms (2026-08-15). A combined depth/stencil surface has
// two guest bases: the key's dr/dw name the depth plane and its sr/sw name the stencil plane, and a
// guest T# addresses whichever plane it samples. Matching only dr/dw left every stencil sample
// falling through to a guest-byte decode of memory the renderer never writes — zeros — which is
// #1275's exact failure mode on the other plane. Grand Theft Auto V samples both planes of one
// 3840x2160 D32_SFLOAT_S8_UINT surface (dr=dw=0x2052ac0000, sr=sw=0x2054aa0000), declaring the
// depth plane Float32 and the stencil plane Uint8, which is what a deferred renderer's material and
// light-volume classification reads.
struct PersistentDsSampled {
    PersistentDsImage* image = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    // Exactly one of DEPTH/STENCIL: a sampled-image view of a combined format may expose only one
    // plane (VUID-VkDescriptorImageInfo-imageView-01976).
    VkImageAspectFlagBits aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
};
// Addresses that matched a live DS surface and were still declined, with why. Separate from the
// counters because an aggregate names a volume, never a fix.
inline std::unordered_map<uint64_t, std::pair<std::string, uint64_t>>& per_address_misses() {
    static std::unordered_map<uint64_t, std::pair<std::string, uint64_t>> misses;
    return misses;
}
// Is this address a depth or stencil plane of ANY retained DS surface, at any extent? Used to tell
// "we hold nothing here" apart from "we hold it and something upstream never asked", which the
// bridge's own counters cannot distinguish: a binding rejected by the caller's gate never reaches
// the lookup and so is invisible to every statistic the lookup keeps.
inline bool is_retained_ds_plane(uint64_t addr, uint32_t* w = nullptr, uint32_t* h = nullptr) {
    if (!addr) return false;
    for (const auto& [key, image] : persistent_ds_cache()) {
        (void)image;
        if (key.dr != addr && key.dw != addr && key.sr != addr && key.sw != addr) continue;
        if (w) *w = key.w;
        if (h) *h = key.h;
        return true;
    }
    return false;
}
// Why a sampled-depth lookup missed, classified against the retained-DS cache.
//
// Pure and environment-free on purpose. The logging path that consumes it is gated on
// PROSPER_DSBRIDGE_LOG, which is read into a function-local static — so a test that armed that
// variable would both trip the cached-env-arming guard and depend on nothing having called the
// bridge earlier in the process. Keeping the classification separate makes the reasoning testable
// without either hazard, and the log becomes a thin consumer.
struct DsBridgeMiss {
    // An enum rather than a string, so a consumer cannot compare against a literal that no longer
    // exists and cannot construct a string_view from a null reason. `None` is a real state: the
    // address matched a plane and nothing disqualified it, which is what a SERVABLE address
    // classifies as.
    enum class Reason { None, NoEntry, Extent, Uninitialized, StencilInvalid, DepthInvalid };
    Reason reason = Reason::NoEntry;
    bool matched_plane = false;      // the address is a plane of SOME retained surface
    bool extent_is_reason = false;   // ...and nothing at that address was the right size
    uint32_t cached_w = 0, cached_h = 0;  // a mismatching sibling's extent, for the detail string
};
// The reason word the bridge log prints; empty for NoEntry (that bucket is unbounded and genuinely
// "not ours") and for None.
inline const char* ds_bridge_miss_reason_name(DsBridgeMiss::Reason reason) {
    switch (reason) {
        case DsBridgeMiss::Reason::Extent:         return "extent";
        case DsBridgeMiss::Reason::Uninitialized:  return "uninitialized";
        case DsBridgeMiss::Reason::StencilInvalid: return "stencil-invalid";
        case DsBridgeMiss::Reason::DepthInvalid:   return "depth-invalid";
        case DsBridgeMiss::Reason::NoEntry:
        case DsBridgeMiss::Reason::None:           break;
    }
    return "";
}
inline DsBridgeMiss classify_ds_bridge_miss(uint64_t addr, uint32_t width, uint32_t height,
                                            uint32_t render_scale = 1,
                                            bool normalized_sampling = true) {
    DsBridgeMiss out;
    bool extent_bad = false, uninit = false, depth_req = false, stencil_req = false;
    // Did ANY entry at this address have a usable extent? One address can hold several cache
    // entries — a live surface plus a stale sibling retained at a different size — and only this
    // flag separates "nothing here is the right size" from "the right size is here and failed for
    // another reason". Without it a stale sibling sets extent_bad and, being tested first, masks
    // the real reason permanently.
    bool extent_ok_exists = false;
    for (const auto& [key, image] : persistent_ds_cache()) {
        const bool d = key.dr == addr || key.dw == addr;
        const bool st = key.sr == addr || key.sw == addr;
        if (!d && !st) continue;
        out.matched_plane = true;
        if (!prosper::frontend::rtt_sampled_extent_compatible(
                width, height, key.w, key.h, render_scale, normalized_sampling)) {
            extent_bad = true;
            out.cached_w = key.w;
            out.cached_h = key.h;
            continue;
        }
        extent_ok_exists = true;
        if (!image.layout_initialized || !image.image) { uninit = true; continue; }
        if (d && !image.depth_valid) depth_req = true;
        if (st && !image.stencil_valid) stencil_req = true;
    }
    // Extent is the reason ONLY when nothing at this address was the right size. Previously any
    // single wrong-sized sibling won this branch outright, so a surface prosper held at exactly the
    // requested size and declined for depth-invalid was reported as an extent mismatch against the
    // sibling's dimensions — naming the wrong entry, the wrong problem, and a fix that does not
    // exist. Measured on Grand Theft Auto V (PPSA04263): 0x20945c0000 reported
    // "extent (T# wants 1024x1536, retained image is 3840x2160)" 1403 times on one route while the
    // cache simultaneously held a 1024x1536 D32_SFLOAT entry at that very address.
    out.extent_is_reason = extent_bad && !extent_ok_exists;
    if (!out.matched_plane)          out.reason = DsBridgeMiss::Reason::NoEntry;
    else if (out.extent_is_reason)   out.reason = DsBridgeMiss::Reason::Extent;
    else if (uninit)                 out.reason = DsBridgeMiss::Reason::Uninitialized;
    else if (stencil_req)            out.reason = DsBridgeMiss::Reason::StencilInvalid;
    else if (depth_req)              out.reason = DsBridgeMiss::Reason::DepthInvalid;
    else                             out.reason = DsBridgeMiss::Reason::None;
    return out;
}
// The reason text the bridge log prints. Keyed off the CLASSIFIED reason, never off "some sibling
// mismatched": those had drifted apart, so a surface declined for depth-invalid still printed the
// extent sentence describing a stale sibling — text that contradicted the counter it was filed
// under and described a mismatch that was not the reason for anything.
inline std::string ds_bridge_miss_detail(const DsBridgeMiss& miss, uint32_t width, uint32_t height) {
    if (miss.reason == DsBridgeMiss::Reason::None ||
        miss.reason == DsBridgeMiss::Reason::NoEntry) return {};
    char detail[96];
    if (miss.extent_is_reason)
        std::snprintf(detail, sizeof detail, "extent (T# wants %ux%u, retained image is %ux%u)",
                      width, height, miss.cached_w, miss.cached_h);
    else
        std::snprintf(detail, sizeof detail, "%s", ds_bridge_miss_reason_name(miss.reason));
    return detail;
}
inline PersistentDsSampled find_persistent_ds_sampled(uint64_t addr, uint32_t width,
                                                      uint32_t height,
                                                      uint32_t render_scale = 1,
                                                      bool normalized_sampling = true) {
    if (!addr) return {};
    static const bool bridge_log = getenv("PROSPER_DSBRIDGE_LOG") != nullptr;
    // Prefer the most recently DEPTH-WRITTEN match: a surface re-keyed (D32 -> D32S8) keeps its
    // stale sibling entry, and unordered_map iteration order must not pick the winner.
    PersistentDsSampled best{};
    uint64_t best_write = 0;
    for (auto& [key, image] : persistent_ds_cache()) {
        if (!prosper::frontend::rtt_sampled_extent_compatible(
                width, height, key.w, key.h, render_scale, normalized_sampling))
            continue;
        const bool is_depth_plane = key.dr == addr || key.dw == addr;
        const bool is_stencil_plane = key.sr == addr || key.sw == addr;
        // A surface whose depth and stencil bases coincide cannot be disambiguated by address, so
        // resolve it as depth — the historical behaviour, and the only plane #1275 ever bridged.
        if (!is_depth_plane && !is_stencil_plane) continue;
        if (!image.layout_initialized || !image.image) continue;
        // PROSPER_DS_BRIDGE_IGNORE_VALID=1 -- DIAGNOSTIC ONLY, never a shipped default.
        //
        // GTA V's deferred lighting pass samples the main depth 0x2052ac0000, and the bridge
        // declines it here with dvalid=0 while svalid=1 on the SAME surface -- prosper holds the
        // image, the extent matches, and only the depth aspect is marked invalid. The sample then
        // falls back to guest memory, which is black for a surface prosper rendered into.
        //
        // Serving a stale depth is wrong as output. It is decisive as a DISCRIMINATOR: if the world
        // appears, the aspect-validity state is what withholds it and the fix is upstream in what
        // clears depth_valid; if nothing changes, the bridge is not the gate and this hypothesis
        // dies with it. Reports whether it actually overrode anything, so a null cannot be read as a
        // negative when the lever never fired.
        static const bool ignore_valid = getenv("PROSPER_DS_BRIDGE_IGNORE_VALID") != nullptr;
        const bool aspect_valid = is_depth_plane ? image.depth_valid : image.stencil_valid;
        if (!aspect_valid && !ignore_valid) continue;
        if (!aspect_valid && ignore_valid) {
            static std::mutex override_mutex;
            static std::map<uint64_t, uint64_t> overrides;
            uint64_t n = 0;
            {
                std::lock_guard lock(override_mutex);
                n = ++overrides[addr];
            }
            if ((n & (n - 1)) == 0)
                std::fprintf(stderr,
                             "[dsbridge] OVERRIDE addr=0x%llx %ux%u aspect=%s served despite "
                             "invalid (count=%llu)\n",
                             (unsigned long long)addr, key.w, key.h,
                             is_depth_plane ? "depth" : "stencil", (unsigned long long)n);
        }
        if (!best.image || image.last_depth_write > best_write) {
            best = {&image, static_cast<VkFormat>(key.fmt), key.w, key.h,
                    is_depth_plane ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT};
            best_write = image.last_depth_write;
        }
    }
    // Counters, not first-N samples. The old logging printed the first eight hits and the first
    // eight misses, which on a routed boot are all consumed by one plane during startup -- so it
    // could not answer "how often does this miss, and why", which is the only question worth asking
    // of a bridge. Reported at powers of two so a long run stays readable.
    if (bridge_log) {
        struct DsBridgeStats {
            std::atomic<uint64_t> calls{0};
            std::atomic<uint64_t> hit_depth{0}, hit_stencil{0};
            std::atomic<uint64_t> miss_no_entry{0};       // address is not a plane of any live DS
            std::atomic<uint64_t> miss_depth_invalid{0};  // plane matched, aspect not valid
            std::atomic<uint64_t> miss_stencil_invalid{0};
            std::atomic<uint64_t> miss_extent{0};         // plane matched, extent incompatible
            std::atomic<uint64_t> miss_uninitialized{0};
        };
        static DsBridgeStats stats;
        if (best.image) {
            (best.aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? stats.hit_stencil : stats.hit_depth)
                .fetch_add(1, std::memory_order_relaxed);
        } else {
            // Classify the miss against the cache, so "no such surface" is never confused with
            // "the surface is here and we declined it" -- they call for opposite fixes.
            const DsBridgeMiss miss = classify_ds_bridge_miss(
                addr, width, height, render_scale, normalized_sampling);
            switch (miss.reason) {
                case DsBridgeMiss::Reason::NoEntry:
                    stats.miss_no_entry.fetch_add(1, std::memory_order_relaxed); break;
                case DsBridgeMiss::Reason::Extent:
                    stats.miss_extent.fetch_add(1, std::memory_order_relaxed); break;
                case DsBridgeMiss::Reason::Uninitialized:
                    stats.miss_uninitialized.fetch_add(1, std::memory_order_relaxed); break;
                case DsBridgeMiss::Reason::StencilInvalid:
                    stats.miss_stencil_invalid.fetch_add(1, std::memory_order_relaxed); break;
                case DsBridgeMiss::Reason::DepthInvalid:
                    stats.miss_depth_invalid.fetch_add(1, std::memory_order_relaxed); break;
                case DsBridgeMiss::Reason::None: break;  // servable; the selector, not this, decides
            }
            // Per-ADDRESS, because an aggregate cannot name the fix. A run reporting
            // "depth-invalid=1089" says a thousand bindings declined; it does not say whether that
            // is one surface declining a thousand times or a thousand surfaces declining once, and
            // those are different defects. Only addresses that DID match a live surface are kept --
            // the no-entry bucket is unbounded and is genuinely just "not ours".
            if (!ds_bridge_miss_detail(miss, width, height).empty()) {
                static std::mutex reason_mutex;
                std::lock_guard lock(reason_mutex);
                auto& per = per_address_misses();
                if (per.size() < 256 || per.count(addr)) {
                    auto& row = per[addr];
                    row.first = ds_bridge_miss_detail(miss, width, height);
                    ++row.second;
                }
            }
        }
        const uint64_t n = stats.calls.fetch_add(1) + 1;
        if ((n & (n - 1)) == 0 && n >= 1024) {
            fprintf(stderr,
                    "[dsbridge] calls=%llu hit(depth=%llu stencil=%llu) "
                    "miss(no-entry=%llu depth-invalid=%llu stencil-invalid=%llu "
                    "extent=%llu uninit=%llu)\n",
                    (unsigned long long)n,
                    (unsigned long long)stats.hit_depth.load(),
                    (unsigned long long)stats.hit_stencil.load(),
                    (unsigned long long)stats.miss_no_entry.load(),
                    (unsigned long long)stats.miss_depth_invalid.load(),
                    (unsigned long long)stats.miss_stencil_invalid.load(),
                    (unsigned long long)stats.miss_extent.load(),
                    (unsigned long long)stats.miss_uninitialized.load());
            std::vector<std::pair<uint64_t, std::pair<std::string, uint64_t>>> ranked;
            {
                static std::mutex reason_mutex_read;
                std::lock_guard lock(reason_mutex_read);
                for (const auto& e : per_address_misses()) ranked.push_back(e);
            }
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
            for (size_t i = 0; i < ranked.size() && i < 12; ++i)
                fprintf(stderr, "[dsbridge]   declined addr=0x%llx reason=%s x%llu\n",
                        (unsigned long long)ranked[i].first, ranked[i].second.first.c_str(),
                        (unsigned long long)ranked[i].second.second);
            // The whole retained set, because "no-entry" is the largest miss bucket and it is the
            // one an address list cannot describe: it says an address is not a plane of anything we
            // hold, and the useful question is then WHAT we hold. A surface the guest samples as
            // depth that never appears here was never rendered into a retained DS image at all,
            // which is a different defect from one we hold and decline.
            fprintf(stderr, "[dsbridge]   retained DS surfaces: %zu\n",
                    persistent_ds_cache().size());
            for (const auto& [key, image] : persistent_ds_cache())
                fprintf(stderr,
                        "[dsbridge]     dr=0x%llx dw=0x%llx sr=0x%llx sw=0x%llx slice=%u "
                        "%ux%u fmt=%u dvalid=%d svalid=%d\n",
                        (unsigned long long)key.dr, (unsigned long long)key.dw,
                        (unsigned long long)key.sr, (unsigned long long)key.sw,
                        key.slice, key.w, key.h, key.fmt,
                        (int)image.depth_valid, (int)image.stencil_valid);
        }
    }
    if (best.image) return best;
    return {};
}

inline bool guest_ranges_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
    if (!a || !a_size || !b || !b_size) return false;
    return a < b + b_size && b < a + a_size;
}

// The host image is deliberately detached from guest tiled depth/stencil memory. A guest-side GPU
// write therefore makes the corresponding cached Vulkan aspect stale. Keep the other aspect valid:
// depth and stencil have independent guest allocations and Vulkan load/store operations, and a
// stencil-only compute update must not discard depth that a later pass samples. HTILE fast clears are
// the exception because the metadata can describe both aspects; conservatively invalidate both.
inline size_t invalidate_persistent_ds_guest_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return 0;
    size_t invalidated = 0;
    for (auto& [key, image] : persistent_ds_cache()) {
        const uint64_t depth_size = static_cast<uint64_t>(key.w) * key.h * 4;
        const uint64_t stencil_size = static_cast<uint64_t>(key.w) * key.h;
        const uint64_t htile_blocks = static_cast<uint64_t>((key.w + 7) / 8) *
                                      ((key.h + 7) / 8);
        const uint64_t htile_size = (htile_blocks * 4 + 0x7fff) & ~0x7fffull;
        // Each slice owns its OWN bytes. Testing every face against the allocation's FIRST slice
        // meant one write evicted an entire cube -- and, symmetrically, a write to face 5's bytes
        // was attributed to face 0. GTA V's cube shadows lost faces this way faster than they were
        // re-rendered, leaving one or two of six resident whenever the cube was sampled.
        //
        // The offset needs the layer stride, which the registry answers for this surface identity --
        // from a consumer descriptor, or 0 for "unknown" and the old whole-allocation behaviour.
        // Reading it here rather than off `key` is what keeps it fresh:
        // the keys iterated by this loop are the ones stored at entry creation, so a stride carried
        // on the key was whatever the FIRST attachment saw, forever.
        const uint64_t learned =
            ds_layer_stride_for(ds_stride_identity_base(key.dr, key.dw), key.w, key.h);
        const uint64_t slice_offset = learned ? static_cast<uint64_t>(key.slice) * learned : 0;
        const uint64_t slice_depth_bytes = learned ? learned : depth_size;
        // Offset a base only when there IS one. `guest_ranges_overlap` refuses a zero base, but
        // `0 + slice_offset` is not zero for any slice > 0, so adding first would slip a bogus low
        // address past that guard and test it against a real write.
        auto plane = [](uint64_t base, uint64_t offset) { return base ? base + offset : 0; };
        const bool depth_overlap =
            guest_ranges_overlap(addr, size, plane(key.dr, slice_offset), slice_depth_bytes) ||
            guest_ranges_overlap(addr, size, plane(key.dw, slice_offset), slice_depth_bytes);
        // STENCIL IS NOT OFFSET, and that is deliberate. `learned` is the DEPTH plane's layer
        // stride; stencil is a separate allocation with its own layout (1 byte/pixel against
        // depth's 4), so striding into it by depth's value lands roughly 4x too far out. That is the
        // UNSAFE error direction twice over: a real write to stencil slice N is missed, leaving
        // stale stencil resident, and a neighbouring slice's write can be misattributed to this one.
        // No authority for a stencil-plane layer stride exists here -- the consumer T# that supplies
        // the depth one describes the depth aspect -- so stencil keeps exactly the behaviour it has
        // on master rather than adopting a stride known to be wrong.
        //
        // Be precise about what that behaviour is, because "whole-allocation" would flatter it:
        // `stencil_size` is w*h, ONE slice's worth, tested at the plane base. For a layered stencil
        // surface a write to slice N > 0 is therefore still missed. That is a real gap and it is the
        // unsafe direction -- it is simply not a gap this change introduces or widens, and adopting
        // the depth stride would have made it worse rather than better. #2670 carries it.
        const bool stencil_overlap =
            guest_ranges_overlap(addr, size, key.sr, stencil_size) ||
            guest_ranges_overlap(addr, size, key.sw, stencil_size);
        const bool htile_overlap = guest_ranges_overlap(addr, size, key.htile, htile_size);
        if (!depth_overlap && !stencil_overlap && !htile_overlap) continue;
        // Which aspect a write actually hit, and the byte count that decided it. The count is the
        // interesting half: `slice_depth_bytes` comes from a LEARNED stride, so an over-large stride
        // silently widens the depth range past its own allocation and attributes a neighbouring
        // plane's write to depth. That misattribution is invisible in the aggregate line below --
        // it reports how many entries were touched, never which aspect or why.
        static const bool log_detail = getenv("PROSPER_DSLOG") != nullptr;
        if (log_detail)
            fprintf(stderr,
                    "[ds] invalidate dr=0x%llx sr=0x%llx htile=0x%llx slice=%u depth=%d stencil=%d "
                    "htile_hit=%d origin=%s addr=0x%llx size=%llu learned=%llu depth_bytes=%llu "
                    "stencil_bytes=%llu htile_bytes=%llu gap=%lld\n",
                    (unsigned long long)key.dr, (unsigned long long)key.sr,
                    (unsigned long long)key.htile, key.slice,
                    (int)depth_overlap, (int)stencil_overlap, (int)htile_overlap,
                    prosper::gpu::guest_gpu_write_origin(),
                    (unsigned long long)addr, (unsigned long long)size,
                    (unsigned long long)learned, (unsigned long long)slice_depth_bytes,
                    (unsigned long long)stencil_size, (unsigned long long)htile_size,
                    (long long)((key.sr ? key.sr : key.sw) - (key.dr ? key.dr : key.dw)));
        // PROSPER_DS_HTILE_INVALIDATE=0 -- experiment arm, default ON (historical behaviour).
        //
        // The retained depth image is a Vulkan image; guest memory does not back it. So a guest
        // write can only invalidate it by making guest memory authoritative for content the image
        // claims to hold. For the DEPTH and STENCIL planes that is exactly what a write means. For
        // HTILE it is true of a fast CLEAR (the metadata says "this tile is value X" and the plane
        // bytes are then ignored) and false of every other HTILE update -- a compute HiZ refresh
        // rewrites the metadata while the depth values it describes are unchanged.
        //
        // GTA V takes the second path 730 times per 200 s route against 135 real stencil writes,
        // and each one discards a fully rendered 4K depth buffer that the deferred lighting pass
        // samples on the very next pass. The arm exists to measure that split before any behaviour
        // is changed; decoding HTILE to tell a clear from a refresh is the actual fix.
        static const bool htile_invalidates = []() {
            const char* v = getenv("PROSPER_DS_HTILE_INVALIDATE");
            return !(v && v[0] == '0');
        }();
        // A "gpu-preserving" origin must NOT suppress this invalidation (#3089).
        //
        // The rule this replaces read "a byte-preserving HTILE write cannot change the logical
        // depth/stencil surface". That premise is false here, and the reason is structural rather
        // than a mistuned threshold: prosper never writes rendered HiZ back into the guest HTILE
        // plane, so the guest copy is a constant that the guest's own writes keep reproducing. The
        // comparator therefore reports changed=0 for a fast CLEAR exactly as readily as for the
        // decompress it was meant to recognise, and byte equality cannot separate the two.
        //
        // Measured on Blue Prince (PPSA25009): the compute kernel rewrites each of the three
        // 196,608-byte HTILE planes with all-zero words -- 49,152 of 49,152 words equal, zero
        // transitions -- so every write after the first compares equal and every DS invalidation is
        // suppressed (tally agree=1, suppressed=59,999). The retained depth is then never
        // discarded, stale depth rejects the scene, and the title renders a pure black frame: 0.00%
        // non-black across 16,500+ colour readbacks, against a fade-in to ~21% before the change.
        //
        // `notify_guest_gpu_write_preserving_bytes` already draws this line, and its contract is
        // the authority for it: byte-preservation licenses skipping guest-memory watches and the
        // submit journal, because those bytes provably did not move. It does not license skipping
        // the alias invalidation -- notifying the owners of renderer-resident aliases is the one
        // thing that path exists to still do, "which may differ from the exact guest bytes even
        // when a compute result does not". live_compute.cpp states the same invariant at the site
        // that produces the classification: "Renderer-alias invalidation and writer provenance
        // remain unconditional."
        //
        // GTA V keeps its picture without the shortcut, so this is not a trade of one title for
        // another: on the 540 s reach-story-mode route the suppression fires 3,177 times, and
        // disabling it changes neither the peak colour coverage (99.78% both arms) nor the targets
        // reaching it. Recovering that pass's retained depth needs the fix the comment above still
        // names -- decoding HTILE to tell a clear from a refresh -- not an equality test on a plane
        // prosper does not maintain.
        // PROSPER_HTILE_UNIFORMLOG (#3121) -- MEASUREMENT ONLY, no behaviour change.
        //
        // The comment above says the actual fix is "decoding HTILE to tell a clear from a refresh",
        // and #3093 removed the byte-preserving exception without one: every HTILE write now
        // invalidates. That fixed Blue Prince and regressed GTA V's deferred lighting, because
        // GTA's HiZ refresh loses the 4K depth its next pass samples.
        //
        // The candidate discriminator is UNIFORMITY of the written range, not byte equality:
        // a fast CLEAR states "every tile is value X" and writes one value everywhere, while a HiZ
        // refresh writes per-tile min/max Z and therefore varies. Blue Prince's measured clear is
        // uniform (49,152 of 49,152 words equal, ZERO transitions). Whether GTA's refresh is
        // non-uniform is the open question this logs, and nothing depends on the answer yet.
        if (htile_overlap) {
            static const bool uniformlog = getenv("PROSPER_HTILE_UNIFORMLOG") != nullptr;
            if (uniformlog && size >= 4) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(addr));
                const size_t n = static_cast<size_t>(size / 4);
                const uint32_t first = w[0];
                size_t equal = 0, transitions = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (w[i] == first) ++equal;
                    if (i && w[i] != w[i - 1]) ++transitions;
                }
                // Tally rather than print per write: this fires thousands of times per route, and a
                // per-write line would itself perturb the run it is measuring.
                static size_t calls = 0, uniform_calls = 0;
                ++calls;
                const bool uniform = (equal == n);
                if (uniform) ++uniform_calls;
                if (calls <= 8 || (calls % 500) == 0)
                    fprintf(stderr,
                            "[htile-uniform] call=%zu origin=%s words=%zu equal_to_first=%zu "
                            "transitions=%zu first=0x%08x uniform=%d  tally uniform=%zu/%zu\n",
                            calls, prosper::gpu::guest_gpu_write_origin(), n, equal, transitions,
                            first, (int)uniform, uniform_calls, calls);
            }
        }
        // A byte-preserving HTILE write does NOT discard retained depth (#3121, restoring the
        // exception that #3093 removed).
        //
        // #3093 removed this to fix Blue Prince's black frame, and stated that GTA V was unharmed
        // because "peak colour coverage (99.78% both arms)" was unchanged. That check could not see
        // the defect it caused: GTA's world still covered the frame, drawn with no lighting and a
        // grid artifact where depth-dependent sampling had gone wrong. Bisected to that commit from
        // a user report, 2026-08-29.
        //
        // MEASURED ON CURRENT MASTER, one binary, one environment variable, both titles:
        //
        //   suppression OFF (as #3093 left it)  GTA broken      Blue Prince max-nonblack 0.2085
        //   suppression ON  (this line)         GTA CORRECT     Blue Prince max-nonblack 0.2085
        //
        // So there is no trade here. Blue Prince is byte-for-byte as healthy either way -- 0.2085 is
        // the very "~21%" #3093 called its restored value -- while GTA V's deferred lighting only
        // survives with the exception. Whatever now carries Blue Prince, it is not this.
        //
        // WHAT IS STILL NOT KNOWN, stated so nobody mistakes this for an explanation. The two titles
        // are INDISTINGUISHABLE at this site. Both arrive as origin=gpu-preserving writing a fully
        // uniform all-zero plane with zero transitions -- GTA 6,500/6,500 uniform, Blue Prince
        // 62,000/62,000, first word 0x00000000 in both, differing only in plane size (73,728 vs
        // 49,152 words, i.e. resolution). A "uniform plane means a fast clear" discriminator was
        // hypothesised and MEASURED FALSE before it was written; PROSPER_HTILE_UNIFORMLOG below is
        // the instrument that killed it.
        // #3264: the `!byte_preserving` term is REMOVED, not weakened. It suppressed the
        // invalidation whenever the write carried the `gpu-preserving` origin, and that is unsound
        // for the reason GTA5_STATUS.md's own Ruled out section recorded on 2026-08-28 (#3089), one
        // day before it was reinstated: prosper never writes rendered HiZ back into the guest HTILE
        // plane, so the guest copy is a constant the guest's own writes keep reproducing, and the
        // byte comparator reports `changed=0` for a fast CLEAR exactly as readily as for the
        // decompress the rule was written for.
        //
        // There was never a trade-off between the two titles. Suppressing the invalidation leaves
        // STALE DEPTH, and the two titles present the same defect differently -- Blue Prince fails
        // the depth test against it and draws nothing, GTA V samples it and hangs the GPU. Measured
        // on one binary pair, same session, same routes:
        //
        //   Blue Prince title    4/5 frames render (256 distinct)  vs  0/5 (2 distinct)
        //   GTA V routed world   8/8 render, 0 device losses       vs  3/8 then black, 122 losses
        //
        // Dead Cells #611 is the prior counterexample where sparing the invalidation makes gameplay
        // geometry disappear, so this rule has now cost three titles. Restoring it again needs the
        // discriminator its own author named and did not build: decode HTILE to tell a clear from a
        // refresh. A byte comparison cannot, and the `gpu-preserving` origin is a process-global set
        // by whichever write ran last (gpu_executor.cpp, `g_guest_write_origin`), so keying on it is
        // order-dependent by construction.
        const bool byte_preserving =
            std::strcmp(prosper::gpu::guest_gpu_write_origin(), "gpu-preserving") == 0;
        const uint64_t current_present = prosper::gpu::present_count();
        const bool current_frame_depth =
            image.depth_valid && image.last_depth_write > 0 &&
            image.last_depth_present != UINT64_MAX &&
            image.last_depth_present == current_present;
        const bool htile_kill = htile_overlap && htile_invalidates &&
                                (!byte_preserving || !current_frame_depth);
        if (!depth_overlap && !stencil_overlap && !htile_kill) continue;
        if (depth_overlap || htile_kill) image.depth_valid = false;
        if (stencil_overlap || htile_kill) image.stencil_valid = false;
        ++invalidated;
    }
    if (invalidated && PROSPER_ENV_ON("PROSPER_DSLOG"))
        fprintf(stderr, "[ds] guest-write addr=%llx size=%llu invalidated=%zu\n",
                (unsigned long long)addr, (unsigned long long)size, invalidated);
    return invalidated;
}

struct MappedReadbackPlan {
    VkDeviceSize map_size = 0;
    VkDeviceSize invalidate_size = 0;
};

// VK_WHOLE_SIZE invalidates through the end of the CURRENT MAPPING, not unconditionally through the
// allocation. That mapping end must be atom-aligned or equal the allocation end (VUID 01389). Map the
// whole allocation as well, so the pair remains valid when a logical readback is shorter than its
// VkMemoryRequirements allocation (the observed shape is 60 logical bytes in a 64-byte allocation).
constexpr MappedReadbackPlan mapped_readback_plan(VkDeviceSize logical_bytes) {
    return logical_bytes ? MappedReadbackPlan{VK_WHOLE_SIZE, VK_WHOLE_SIZE}
                         : MappedReadbackPlan{};
}

// Make device writes visible before the CPU reads a mapped readback buffer. A HOST_CACHED memory
// type is not required to also be HOST_COHERENT, so every mapped READ of a TRANSFER_DST buffer must
// invalidate first. Specified as valid on coherent memory too, so callers can invoke it
// unconditionally rather than tracking the selected memory type.
// NOTE: this header contains a second readback allocator that requires HOST_COHERENT in every tier,
// so its mapped reads need no invalidate. Keep that requirement, or route its reads through this
// helper too -- relaxing it without adding invalidates would silently return stale bytes.
inline bool invalidate_mapped_readback(const RenderVkCtx& ctx, VkDeviceMemory memory,
                                       const MappedReadbackPlan& plan) {
    if (!memory || !plan.invalidate_size) return false;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = 0;
    range.size = plan.invalidate_size;
    return vkInvalidateMappedMemoryRanges(ctx.dev, 1, &range) == VK_SUCCESS;
}

// The OTHER half of a readback, and a separate operation from the invalidate above: the
// availability dependency into the HOST domain that a fence wait does not perform (#2944). It lives
// in src/gpu/execute/host_read_barrier.hpp because the live COMPUTE backend needs exactly the same
// rule for its dispatch results (#3249), and two spellings of one rule is how the next site gets
// missed. Re-exported here so this header's call sites and the `host_read_barrier` ctest case keep
// naming it through `prosper::test`.
using prosper::gpu::HostReadBarrier;
using prosper::gpu::host_read_barrier_for;
using prosper::gpu::backend_host_read_barrier_count;
using prosper::gpu::record_host_read_barrier;

// Two transfer WRITES to the same image subresource need a MEMORY dependency, not just an execution
// one (#3248).
//
// An execution dependency -- which any vkCmdPipelineBarrier between them provides, whatever image
// its VkImageMemoryBarrier names -- orders the two commands. It does not make the first write
// AVAILABLE, so the second write is not ordered against it in the memory-access sense and the final
// contents of the overlapping region are undefined. Synchronization validation calls this
// SYNC-HAZARD-WRITE-AFTER-WRITE and it is the shape of the mip-assembly defect in #3248: a full-image
// vkCmdClearColorImage followed by per-level vkCmdCopyImage, with barriers only on the copy SOURCES.
//
// The failure mode is why this is worth a helper rather than an inline barrier. Assembly deliberately
// clears to BLACK so a level the guest never rendered stays unavailable instead of being invented from
// a neighbour. If the clear lands after a copy, the level it eats reads as "missing" -- exactly what a
// correct run produces on purpose -- so the defect cannot be seen in the output it corrupts.
struct TransferWriteAfterWriteBarrier {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dst_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
};

// Pure, so the masks and the subresource scope can be asserted with no device
// (`mip_assembly_barrier` ctest case). No layout transition: both writes happen in
// TRANSFER_DST_OPTIMAL, and oldLayout == newLayout keeps the contents defined.
inline TransferWriteAfterWriteBarrier transfer_write_after_write_barrier_for(
        VkImage image, const VkImageSubresourceRange& range,
        VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    TransferWriteAfterWriteBarrier waw{};
    waw.barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    waw.barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    waw.barrier.oldLayout = layout;
    waw.barrier.newLayout = layout;
    waw.barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    waw.barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    waw.barrier.image = image;
    waw.barrier.subresourceRange = range;
    waw.src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    waw.dst_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    return waw;
}

// How many transfer write-after-write barriers this process has recorded. Same role as
// backend_host_read_barrier_count(): on every driver this project runs on the UNFIXED code returns
// correct pixels, so a structural counter is the only thing a test can falsify. Relaxed and
// process-wide -- an instrument, never a control input.
inline std::atomic<uint64_t>& backend_transfer_waw_barrier_count() {
    static std::atomic<uint64_t> count{0};
    return count;
}

inline void record_transfer_write_after_write_barrier(
        VkCommandBuffer command, VkImage image, const VkImageSubresourceRange& range,
        VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    if (!command || !image) return;
    const TransferWriteAfterWriteBarrier waw =
        transfer_write_after_write_barrier_for(image, range, layout);
    vkCmdPipelineBarrier(command, waw.src_stages, waw.dst_stages, 0,
                         0, nullptr, 0, nullptr, 1, &waw.barrier);
    backend_transfer_waw_barrier_count().fetch_add(1, std::memory_order_relaxed);
}

// Diagnostic-arming instruments (#3248). Both are read by the `render_diagnostic_paths` ctest case,
// which exists so the env-gated render diagnostics execute at least once under the validation layer:
// they were invisible to tools/vkval for a second reason on top of syncval being off -- no test ran
// them at all, so the layer never saw the state they record.
inline std::atomic<uint64_t>& backend_geom_probe_armed_count() {
    static std::atomic<uint64_t> count{0};
    return count;
}

// Incremented when the probe was ASKED for a draw whose last pre-rasterization stage does not
// declare the transform-feedback capture -- the case that used to arm anyway and report the
// resulting silence as "the draw produced no primitives".
inline std::atomic<uint64_t>& backend_geom_probe_undeclared_count() {
    static std::atomic<uint64_t> count{0};
    return count;
}

// One per PROSPER_DRAW_ISO re-render pass (the kill=-1 baseline plus one per killed draw).
inline std::atomic<uint64_t>& backend_draw_iso_pass_count() {
    static std::atomic<uint64_t> count{0};
    return count;
}

// CONTRACT: a pure TRANSFER_DST (readback) buffer may be backed by HOST_CACHED memory that is NOT
// HOST_COHERENT. Call invalidate_mapped_readback() after mapping and before reading one, and do not
// host-WRITE through such a mapping without a flush. Any usage including TRANSFER_SRC is always
// backed by HOST_COHERENT memory, so host writes to it need no flush.
inline bool persistent_ds_transfer_buffer(const RenderVkCtx& ctx, VkDeviceSize bytes,
                                          VkBufferUsageFlags usage, VkBuffer& buffer,
                                          VkDeviceMemory& memory, std::string& error) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes; info.usage = usage;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &buffer) != VK_SUCCESS) {
        error = "cannot create persistent DS transfer buffer"; return false;
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    // A pure TRANSFER_DST buffer is a GPU->CPU readback: the CPU READS every byte of it. Plain
    // HOST_VISIBLE|HOST_COHERENT memory is typically write-combined and uncached, which streams CPU
    // writes well but reads back at only a few hundred MB/s -- measured here at ~236 MB/s, making the
    // map+copy 99% of a color-target readback. Such a buffer therefore prefers HOST_CACHED so the
    // copy runs at cache speed. A buffer that is ALSO TRANSFER_SRC is host-written (an upload, or a
    // round trip), so it keeps the write-combined coherent selection that is right for writes and
    // needs no host flush.
    const bool readback = (usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0 &&
                          (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0;
    const uint32_t coherent_type = render_memory_type(
        ctx.phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t preferred = UINT32_MAX;
    if (readback) {
        preferred = render_memory_type(
            ctx.phys, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (preferred == UINT32_MAX)
            preferred = render_memory_type(
                ctx.phys, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    // Try the preferred (cached) type first, then the previously used coherent type. The retry
    // matters because a cached type can live in a small heap (BAR-style) that the coherent type does
    // not: without it, a cached-heap exhaustion would fail a readback that previously succeeded.
    auto try_allocate = [&](uint32_t type) {
        if (type == UINT32_MAX) return false;
        allocation.memoryTypeIndex = type;
        if (vkAllocateMemory(ctx.dev, &allocation, nullptr, &memory) != VK_SUCCESS) {
            memory = VK_NULL_HANDLE;
            return false;
        }
        if (vkBindBufferMemory(ctx.dev, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(ctx.dev, memory, nullptr); memory = VK_NULL_HANDLE;
            return false;
        }
        return true;
    };
    if (!try_allocate(preferred) && !(preferred != coherent_type && try_allocate(coherent_type))) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr); buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;
        error = "cannot allocate persistent DS transfer buffer"; return false;
    }
    return true;
}

inline BackendSubmissionState submit_persistent_ds_transfer(
        const RenderVkCtx& ctx, VkImage image, VkImageAspectFlags aspects,
        VkImageLayout old_layout, VkImageLayout transfer_layout, VkBuffer buffer,
        uint32_t width, uint32_t height, bool copy_depth, bool copy_stencil,
        bool upload, std::string& error) {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx.qfi;
    if (vkCreateCommandPool(ctx.dev, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        error = "cannot create persistent DS transfer command pool";
        return BackendSubmissionState::NotSubmitted;
    }
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool; command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(ctx.dev, &command_info, &command) != VK_SUCCESS ||
        vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        vkDestroyCommandPool(ctx.dev, pool, nullptr);
        error = "cannot begin persistent DS transfer command";
        return BackendSubmissionState::NotSubmitted;
    }
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = old_layout; barrier.newLayout = transfer_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspects, 0, 1, 0, 1};
    barrier.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
        (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    barrier.dstAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    const VkPipelineStageFlags source_stage = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    vkCmdPipelineBarrier(command, source_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    VkDeviceSize offset = 0;
    auto copy_plane = [&](VkImageAspectFlagBits aspect, VkDeviceSize bytes) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {static_cast<VkImageAspectFlags>(aspect), 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        if (upload)
            vkCmdCopyBufferToImage(command, buffer, image, transfer_layout, 1, &copy);
        else
            vkCmdCopyImageToBuffer(command, image, transfer_layout, buffer, 1, &copy);
        offset += bytes;
    };
    if (copy_depth) copy_plane(VK_IMAGE_ASPECT_DEPTH_BIT,
                               static_cast<VkDeviceSize>(width) * height * 4);
    if (copy_stencil) copy_plane(VK_IMAGE_ASPECT_STENCIL_BIT,
                                 static_cast<VkDeviceSize>(width) * height);
    // #2944: on the READBACK direction the caller maps `buffer` and reads it after the fence below.
    // The fence orders execution; this makes the copy available to the host. The UPLOAD direction is
    // the mirror image -- the host wrote and the device reads -- so it needs no host-read dependency.
    if (!upload) record_host_read_barrier(command, buffer);

    barrier.oldLayout = transfer_layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const bool recorded = vkEndCommandBuffer(command) == VK_SUCCESS;
    const bool fenced = recorded &&
        vkCreateFence(ctx.dev, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    const bool submitted = fenced && render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS;
    bool finished = submitted &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (submitted && !finished) finished = render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS;
    const BackendSubmissionState state = backend_submission_state(submitted, finished);
    if (state == BackendSubmissionState::Pending)
        backend_mark_unproven_submission();
    if (state != BackendSubmissionState::Pending) {
        if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
        vkDestroyCommandPool(ctx.dev, pool, nullptr);
    }
    if (state != BackendSubmissionState::Complete)
        error = "persistent DS transfer did not complete";
    return state;
}

// Materialize a valid GPU-only color target on demand. Ordered DMA may consume a target in a later
// submit, which the producing render callback cannot predict; keeping the fast no-readback path and
// synchronizing only at that consumer preserves both the persistent-target contract and DMA versioning.
inline bool readback_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                             VkFormat format, std::vector<uint8_t>& output,
                                             std::string& error) {
    output.clear(); error.clear();
    if (backend_has_unproven_submission()) {
        error = "Vulkan submission completion is unproven";
        return false;
    }
    format = backend_color_format(format);
    PersistentColorTargetImage* target = find_persistent_color_target(
        id, width, height, format);
    if (!target || !target->image || target->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        error = "persistent color target is unavailable";
        return false;
    }
    target->last_use = ++persistent_color_target_generation();
    const uint64_t texels = static_cast<uint64_t>(width) * height;
    const uint64_t bpp = backend_color_bytes_per_pixel(format);
    if (!width || !height || !bpp || texels > UINT64_MAX / bpp || texels * bpp > SIZE_MAX) {
        error = "persistent color target byte size is invalid";
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(texels * bpp);
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!persistent_ds_transfer_buffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       buffer, memory, error)) {
        error = "cannot allocate persistent color target readback buffer";
        return false;
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup = [&] {
        if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
        if (pool) vkDestroyCommandPool(ctx.dev, pool, nullptr);
        if (buffer) vkDestroyBuffer(ctx.dev, buffer, nullptr);
        if (memory) vkFreeMemory(ctx.dev, memory, nullptr);
    };
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx.qfi;
    if (vkCreateCommandPool(ctx.dev, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        cleanup(); error = "cannot create persistent color target readback command pool"; return false;
    }
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(ctx.dev, &command_info, &command) != VK_SUCCESS ||
        vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        cleanup(); error = "cannot begin persistent color target readback command"; return false;
    }

    const VkImageLayout saved_layout = target->layout;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = saved_layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = target->image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command, target->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &copy);
    record_host_read_barrier(command, buffer);   // #2944
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = saved_layout;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const bool recorded = vkEndCommandBuffer(command) == VK_SUCCESS;
    const bool fenced = recorded &&
        vkCreateFence(ctx.dev, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    const bool submitted = fenced && render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS;
    bool finished = submitted &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (submitted && !finished) finished = render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS;
    if (!finished) {
        // Submitted-but-unfinished: the command buffer (and the readback buffer it references)
        // may still be pending on a wedged queue, so destroying the pool, fence, or buffer is
        // invalid use. Deliberately leak the one-shot objects — the device is effectively lost
        // (#1383). The never-submitted failure modes still clean up normally.
        if (!submitted) cleanup();
        else backend_mark_unproven_submission();
        error = "persistent color target readback did not complete"; return false;
    }
    const MappedReadbackPlan mapping = mapped_readback_plan(bytes);
    void* mapped = nullptr;
    if (vkMapMemory(ctx.dev, memory, 0, mapping.map_size, 0, &mapped) != VK_SUCCESS || !mapped) {
        cleanup(); error = "cannot map persistent color target readback"; return false;
    }
    if (!invalidate_mapped_readback(ctx, memory, mapping)) {
        vkUnmapMemory(ctx.dev, memory);
        cleanup(); error = "cannot invalidate persistent color target readback"; return false;
    }
    const auto* first = static_cast<const uint8_t*>(mapped);
    output.assign(first, first + static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.dev, memory);
    ++backend_color_target_stats_storage().readbacks;
    cleanup();
    return true;
}

// #1334: GPU-side copy of one persistent color target into another identity. The MSAA resolve
// (CB MODE=RESOLVE) copies color0 -> color1 on hardware; sharing only the CPU pixels left the
// destination-keyed persistent image stale while its RttSurf inherited gpu_valid=true from the
// source — after a #780 guest-write CPU-copy discard, consumers (Blue Prince's compute tonemap)
// imported the stale image and read black, cascading the #1287 display-chain collapse. Copy the
// device-local pixels so the destination image is genuinely valid. Creates the destination if
// absent (same shape/usage as the render path; budget-checked with LRU eviction of unpinned older
// targets — an allocation failure returns false and the caller keeps the CPU pixels as the only truth,
// fail-visibly).
inline bool copy_persistent_color_target(uint64_t src_id, uint64_t dst_id, uint32_t width,
                                         uint32_t height, VkFormat format, std::string& error) {
    error.clear();
    if (backend_has_unproven_submission()) {
        error = "Vulkan submission completion is unproven";
        return false;
    }
    format = backend_color_format(format);
    if (!width || !height || !dst_id || src_id == dst_id) {
        error = "invalid copy identity"; return false;
    }
    PersistentColorTargetImage* src = find_persistent_color_target(src_id, width, height, format);
    if (!src || !src->image || !src->valid || src->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        error = "source persistent color target is unavailable";
        return false;
    }
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    src->last_use = ++persistent_color_target_generation();

    PersistentColorTargetKey dst_key{dst_id, width, height, format};
    auto [dst_it, dst_inserted] = persistent_color_target_cache().try_emplace(dst_key);
    PersistentColorTargetImage* dst = &dst_it->second;
    dst->last_use = ++persistent_color_target_generation();
    if (!dst->image) {
        VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = format;
        imgci.extent = {width, height, 1};
        imgci.mipLevels = 1; imgci.arrayLayers = 1; imgci.samples = VK_SAMPLE_COUNT_1_BIT;
        imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImage img = VK_NULL_HANDLE;
        if (vkCreateImage(ctx.dev, &imgci, nullptr, &img) != VK_SUCCESS || !img) {
            persistent_color_target_cache().erase(dst_key);
            error = "cannot create resolve destination image";
            return false;
        }
        VkMemoryRequirements ir{}; vkGetImageMemoryRequirements(ctx.dev, img, &ir);
        const VkDeviceSize limit = persistent_color_target_limit();
        const size_t count_limit = persistent_color_target_count_limit();
        const size_t count_ceiling = persistent_color_target_count_ceiling(true);
        if (ir.size <= limit) {
            ++src->pin_count;
            while ((persistent_color_target_cache().size() > count_limit ||
                    persistent_color_target_bytes() > limit - ir.size) &&
                   evict_persistent_color_target(ctx, persistent_color_target_generation())) {}
            --src->pin_count;
        }
        VkDeviceMemory imem = VK_NULL_HANDLE;
        VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        iai.allocationSize = ir.size;
        iai.memoryTypeIndex = render_memory_type(ctx.phys, ir.memoryTypeBits, 0);
        if (ir.size > limit || persistent_color_target_bytes() > limit - ir.size ||
            persistent_color_target_cache().size() > count_ceiling ||
            vkAllocateMemory(ctx.dev, &iai, nullptr, &imem) != VK_SUCCESS || !imem) {
            vkDestroyImage(ctx.dev, img, nullptr);
            persistent_color_target_cache().erase(dst_key);
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "resolve destination exceeds budget (ir=%llu limit=%llu cur_bytes=%llu "
                     "cache_size=%zu ceiling=%zu mem_type=%u)",
                     (unsigned long long)ir.size, (unsigned long long)limit,
                     (unsigned long long)persistent_color_target_bytes(),
                     persistent_color_target_cache().size(), count_ceiling,
                     iai.memoryTypeIndex);
            error = buf;
            return false;
        }
        if (vkBindImageMemory(ctx.dev, img, imem, 0) != VK_SUCCESS) {
            vkDestroyImage(ctx.dev, img, nullptr);
            vkFreeMemory(ctx.dev, imem, nullptr);
            persistent_color_target_cache().erase(dst_key);
            error = "cannot bind resolve destination memory";
            return false;
        }
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = format;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(ctx.dev, &ivci, nullptr, &view) != VK_SUCCESS || !view) {
            vkDestroyImage(ctx.dev, img, nullptr);
            vkFreeMemory(ctx.dev, imem, nullptr);
            persistent_color_target_cache().erase(dst_key);
            error = "cannot create resolve destination view";
            return false;
        }
        // Element pointers survive unordered_map rehash; this re-fetch is belt-and-braces only.
        dst = &persistent_color_target_cache()[dst_key];
        dst->image = img; dst->memory = imem; dst->view = view;
        dst->bytes = ir.size;
        persistent_color_target_bytes() += ir.size;
        dst->layout = VK_IMAGE_LAYOUT_UNDEFINED;
        src = find_persistent_color_target(src_id, width, height, format);
        if (!src) { error = "source evicted during destination creation"; return false; }
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup = [&] {
        if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
        if (pool) vkDestroyCommandPool(ctx.dev, pool, nullptr);
    };
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx.qfi;
    // A pre-existing valid destination image is exactly the stale object this helper exists to
    // replace — on ANY failure past this point it must not stay valid (#1382 review finding 2).
    auto fail = [&](const char* message) {
        dst->valid = false;
        error = message;
        return false;
    };
    if (vkCreateCommandPool(ctx.dev, &pool_info, nullptr, &pool) != VK_SUCCESS)
        return fail("cannot create resolve copy command pool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(ctx.dev, &command_info, &command) != VK_SUCCESS ||
        vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        cleanup(); return fail("cannot begin resolve copy command");
    }
    const VkImageLayout src_saved = src->layout;
    VkImageMemoryBarrier barriers[2]{};
    for (auto& barrier : barriers) {
        barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    barriers[0].image = src->image;
    barriers[0].oldLayout = src_saved;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[1].image = dst->image;
    barriers[1].oldLayout = dst->layout;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {width, height, 1};
    vkCmdCopyImage(command, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].newLayout = src_saved;
    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) {
        cleanup(); return fail("cannot record resolve copy command");
    }
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(ctx.dev, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        cleanup(); return fail("cannot create resolve copy fence");
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    const bool submitted = render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS;
    bool finished = submitted &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (submitted && !finished) finished = render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS;
    if (!finished) {
        // Submitted-but-unfinished means the command buffer may still be pending on a wedged
        // queue; destroying its pool or fence then is invalid use. Deliberately leak the
        // one-shot objects instead — the device is effectively lost (#1383).
        if (!submitted) cleanup();
        else backend_mark_unproven_submission();
        return fail("resolve copy did not complete");
    }
    cleanup();
    dst->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dst->valid = true;
    return true;
}

// Read one retained DS image's DEPTH plane back to the CPU as float32.
//
// The cube-shadow bridge needs it: a guest depth cube is one six-layer allocation whose faces are
// separate retained images (each keyed by its DB_DEPTH_VIEW slice), while the sampled representation
// the recompiler lowers a cube sample onto is a single vertically stacked w x 6h image. Nothing can
// bind six images as one cube, so the faces are gathered on the CPU and restacked.
//
// Costly by construction -- a full-extent copy per face -- so callers must gate it on a cube sample
// that actually resolves to retained faces, never run it speculatively.
inline bool read_persistent_ds_depth(PersistentDsImage& image, uint32_t width, uint32_t height,
                                     std::vector<float>& out, std::string& error) {
    error.clear();
    out.clear();
    if (!image.image || !image.layout_initialized || !image.depth_valid) {
        error = "retained DS image has no readable depth plane";
        return false;
    }
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    const size_t depth_bytes = static_cast<size_t>(width) * height * 4;
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!persistent_ds_transfer_buffer(ctx, depth_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       buffer, memory, error))
        return false;
    const BackendSubmissionState transfer = submit_persistent_ds_transfer(
        ctx, image.image, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, width, height,
        /*depth=*/true, /*stencil=*/false, false, error);
    if (transfer != BackendSubmissionState::Complete) {
        if (transfer != BackendSubmissionState::Pending) {
            vkDestroyBuffer(ctx.dev, buffer, nullptr);
            vkFreeMemory(ctx.dev, memory, nullptr);
        }
        if (error.empty()) error = "retained DS depth transfer did not complete";
        return false;
    }
    // This is a pure TRANSFER_DST readback, so its memory may be HOST_CACHED and NOT HOST_COHERENT
    // -- the allocator above actively prefers HOST_CACHED for this usage. Reading the mapping
    // without invalidating first returns whatever the CPU cache happens to hold, which for a
    // freshly-allocated buffer is plausible-looking garbage rather than an obvious failure. Both
    // sibling readbacks in this file do this; this one did not.
    const MappedReadbackPlan mapping = mapped_readback_plan(depth_bytes);
    void* mapped = nullptr;
    if (vkMapMemory(ctx.dev, memory, 0, mapping.map_size, 0, &mapped) != VK_SUCCESS || !mapped) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr);
        vkFreeMemory(ctx.dev, memory, nullptr);
        error = "cannot map retained DS depth readback";
        return false;
    }
    if (!invalidate_mapped_readback(ctx, memory, mapping)) {
        vkUnmapMemory(ctx.dev, memory);
        vkDestroyBuffer(ctx.dev, buffer, nullptr);
        vkFreeMemory(ctx.dev, memory, nullptr);
        error = "cannot invalidate retained DS depth readback";
        return false;
    }
    out.resize(static_cast<size_t>(width) * height);
    std::memcpy(out.data(), mapped, depth_bytes);
    vkUnmapMemory(ctx.dev, memory);
    vkDestroyBuffer(ctx.dev, buffer, nullptr);
    vkFreeMemory(ctx.dev, memory, nullptr);
    return true;
}

// Metadata-only selection for a retained depth cube. `overlay_version` is the newest selected
// depth-write generation, so a CPU-decoded six-face stack can be cached against renderer authority
// without consulting stale guest bytes. Any new face write advances that generation; an invalid or
// missing face makes the selection incomplete and therefore ineligible for that cache.
struct PersistentDsCubeSelection {
    std::array<PersistentDsImage*, 6> faces{};
    uint32_t present_mask = 0;
    uint32_t known_mask = 0;
    uint64_t overlay_version = 0;
};

inline PersistentDsCubeSelection select_persistent_ds_cube_depth(
    uint64_t base, uint32_t width, uint32_t height) {
    PersistentDsCubeSelection selected;
    std::array<uint64_t, 6> recency{};
    if (!base || !width || !height) return selected;
    for (auto& [key, image] : persistent_ds_cache()) {
        if (key.w != width || key.h != height ||
            (key.dr != base && key.dw != base) || key.slice >= 6u)
            continue;
        selected.known_mask |= 1u << key.slice;
        if (!image.depth_valid || !image.layout_initialized || !image.image)
            continue;
        if (!selected.faces[key.slice] || image.last_depth_write > recency[key.slice]) {
            selected.faces[key.slice] = &image;
            recency[key.slice] = image.last_depth_write;
            selected.present_mask |= 1u << key.slice;
        }
    }
    for (uint32_t slice = 0; slice < 6u; ++slice)
        if (selected.faces[slice])
            selected.overlay_version = std::max(
                selected.overlay_version, selected.faces[slice]->last_depth_write);
    return selected;
}

// All six faces of a depth cube whose faces are retained separately, or false when any is missing.
// `slices_found` reports how many were present, so a partial cube is a stated fact rather than a
// silent half-result.
inline bool read_persistent_ds_cube_depth(uint64_t base, uint32_t width, uint32_t height,
                                          std::array<std::vector<float>, 6>& faces,
                                          uint32_t& slices_found, std::string& error,
                                          uint32_t* present_mask = nullptr,
                                          uint32_t* known_mask = nullptr) {
    error.clear();
    slices_found = 0;
    const PersistentDsCubeSelection selected =
        select_persistent_ds_cube_depth(base, width, height);
    if (present_mask) *present_mask = selected.present_mask;
    if (known_mask) *known_mask = selected.known_mask;
    for (uint32_t slice = 0; slice < 6u; ++slice) {
        PersistentDsImage* found = selected.faces[slice];
        if (!found) continue;
        if (!read_persistent_ds_depth(*found, width, height, faces[slice], error)) return false;
        ++slices_found;
    }
    return slices_found != 0;
}

// Retained depth faces produced after a compute image. GTA V builds each shadow cube by copying
// five unchanged R16 layers in compute and then rendering the sixth face into a D32 attachment.
// Comparing the global PM4 ordinals makes that split explicit: only the renderer-newer face may
// replace the exact compute result. `last_depth_write` breaks ties between re-keyed cache entries;
// it is deliberately not compared with a command ordinal because those counters have different
// domains.
inline PersistentDsCubeSelection select_persistent_ds_cube_depth_after(
    uint64_t base, uint32_t width, uint32_t height, uint64_t producer_command_order) {
    PersistentDsCubeSelection selected;
    std::array<uint64_t, 6> recency{};
    if (!base || !producer_command_order) return selected;
    for (auto& [key, image] : persistent_ds_cache()) {
        if (key.w != width || key.h != height ||
            (key.dr != base && key.dw != base) || key.slice >= 6u)
            continue;
        selected.known_mask |= 1u << key.slice;
        if (!image.depth_valid || !image.layout_initialized || !image.image ||
            image.last_depth_command_order <= producer_command_order)
            continue;
        if (!selected.faces[key.slice] || image.last_depth_write > recency[key.slice]) {
            selected.faces[key.slice] = &image;
            recency[key.slice] = image.last_depth_write;
            selected.present_mask |= 1u << key.slice;
            selected.overlay_version = std::max(
                selected.overlay_version, image.last_depth_write);
        }
    }
    return selected;
}

inline bool read_persistent_ds_cube_depth_after(
    uint64_t base, uint32_t width, uint32_t height, uint64_t producer_command_order,
    std::array<std::vector<float>, 6>& faces, uint32_t& present_mask,
    uint32_t& known_mask, std::string& error) {
    error.clear();
    const PersistentDsCubeSelection selected = select_persistent_ds_cube_depth_after(
        base, width, height, producer_command_order);
    present_mask = selected.present_mask;
    known_mask = selected.known_mask;
    for (uint32_t slice = 0; slice < 6u; ++slice) {
        if (!(present_mask & (1u << slice))) continue;
        if (!read_persistent_ds_depth(*selected.faces[slice], width, height,
                                      faces[slice], error))
            return false;
    }
    return true;
}

inline bool snapshot_persistent_ds_images(std::vector<prosper::gpu::GpuCaptureDsSeed>& seeds,
                                          std::string& error) {
    error.clear(); seeds.clear();
    if (backend_has_unproven_submission()) {
        error = "Vulkan submission completion is unproven";
        return false;
    }
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    for (const auto& [key, image] : persistent_ds_cache()) {
        if (!image.depth_valid && !image.stencil_valid) continue;
        if (!image.image || !image.layout_initialized) {
            error = "valid persistent DS cache entry has no initialized image"; return false;
        }
        prosper::gpu::GpuCaptureDsSeed seed;
        seed.depth_read_base = key.dr; seed.depth_write_base = key.dw;
        seed.stencil_read_base = key.sr; seed.stencil_write_base = key.sw;
        seed.htile_data_base = key.htile; seed.width = key.w; seed.height = key.h;
        seed.slice = key.slice;   // a cube's faces differ only here
        if (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT))
            seed.format = prosper::gpu::GpuCaptureDsFormat::D32Float;
        else if (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT))
            seed.format = prosper::gpu::GpuCaptureDsFormat::D32FloatS8;
        else {
            error = "persistent DS cache uses an unsupported capture format"; return false;
        }
        seed.depth_valid = image.depth_valid;
        seed.stencil_valid = image.stencil_valid;
        const size_t depth_bytes = seed.depth_valid ? static_cast<size_t>(key.w) * key.h * 4 : 0;
        const size_t stencil_bytes = seed.stencil_valid ? static_cast<size_t>(key.w) * key.h : 0;
        VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
        if (!persistent_ds_transfer_buffer(ctx, depth_bytes + stencil_bytes,
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           buffer, memory, error)) return false;
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
            (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT)
                 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        const BackendSubmissionState transfer = submit_persistent_ds_transfer(
            ctx, image.image, aspects, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, key.w, key.h,
            seed.depth_valid, seed.stencil_valid, false, error);
        if (transfer != BackendSubmissionState::Complete) {
            // Pending work still owns the transfer buffer. Never-submitted failures are safe to
            // release because the helper has already released its command pool and fence.
            if (transfer != BackendSubmissionState::Pending) {
                vkDestroyBuffer(ctx.dev, buffer, nullptr);
                vkFreeMemory(ctx.dev, memory, nullptr);
            }
            return false;
        }
        const MappedReadbackPlan mapping =
            mapped_readback_plan(depth_bytes + stencil_bytes);
        void* mapped = nullptr;
        const bool mapped_ok =
            vkMapMemory(ctx.dev, memory, 0, mapping.map_size, 0, &mapped) == VK_SUCCESS;
        const bool invalidated = mapped_ok && invalidate_mapped_readback(ctx, memory, mapping);
        if (invalidated) {
            const auto* bytes = static_cast<const uint8_t*>(mapped);
            seed.depth.assign(bytes, bytes + depth_bytes);
            seed.stencil.assign(bytes + depth_bytes, bytes + depth_bytes + stencil_bytes);
            vkUnmapMemory(ctx.dev, memory);
        }
        if (mapped_ok && !invalidated) vkUnmapMemory(ctx.dev, memory);
        vkDestroyBuffer(ctx.dev, buffer, nullptr); vkFreeMemory(ctx.dev, memory, nullptr);
        if (!invalidated) {
            if (error.empty())
                error = mapped_ok ? "cannot invalidate persistent DS readback"
                                  : "cannot map persistent DS readback";
            return false;
        }
        seeds.push_back(std::move(seed));
    }
    return true;
}

inline bool restore_persistent_ds_image(const prosper::gpu::GpuCaptureDsSeed& seed,
                                        std::string& error) {
    error.clear();
    if (backend_has_unproven_submission()) {
        error = "Vulkan submission completion is unproven";
        return false;
    }
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    const VkFormat format = seed.format == prosper::gpu::GpuCaptureDsFormat::D32Float
        ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D32_SFLOAT_S8_UINT;
    const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
        (format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    // The seed carries its own slice (capture v55). Restoring every seed at slice 0 would collapse
    // a cube's six faces onto one identity on replay, which is the same defect the live cache had.
    PersistentDsKey key{seed.depth_read_base, seed.depth_write_base,
                        seed.stencil_read_base, seed.stencil_write_base,
                        seed.htile_data_base, seed.width, seed.height,
                        static_cast<uint32_t>(format), seed.slice};
    PersistentDsImage& image = persistent_ds_cache()[key];
    if (!image.image) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = format;
        info.extent = {seed.width, seed.height, 1};
        info.mipLevels = 1; info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;   // sampled depth bridge (#1275)
        if (vkCreateImage(ctx.dev, &info, nullptr, &image.image) != VK_SUCCESS) {
            error = "cannot create restored persistent DS image"; return false;
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(ctx.dev, image.image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = render_memory_type(ctx.phys, requirements.memoryTypeBits, 0);
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image.image; view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format; view_info.subresourceRange = {aspects, 0, 1, 0, 1};
        if (allocation.memoryTypeIndex == UINT32_MAX ||
            vkAllocateMemory(ctx.dev, &allocation, nullptr, &image.memory) != VK_SUCCESS ||
            vkBindImageMemory(ctx.dev, image.image, image.memory, 0) != VK_SUCCESS ||
            vkCreateImageView(ctx.dev, &view_info, nullptr, &image.view) != VK_SUCCESS) {
            if (image.view) vkDestroyImageView(ctx.dev, image.view, nullptr);
            vkDestroyImage(ctx.dev, image.image, nullptr);
            if (image.memory) vkFreeMemory(ctx.dev, image.memory, nullptr);
            image = {};
            error = "cannot allocate restored persistent DS image"; return false;
        }
    }
    const size_t transfer_bytes = seed.depth.size() + seed.stencil.size();
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!persistent_ds_transfer_buffer(ctx, transfer_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       buffer, memory, error)) return false;
    void* mapped = nullptr;
    if (vkMapMemory(ctx.dev, memory, 0, transfer_bytes, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr); vkFreeMemory(ctx.dev, memory, nullptr);
        error = "cannot map persistent DS upload"; return false;
    }
    if (!seed.depth.empty()) std::memcpy(mapped, seed.depth.data(), seed.depth.size());
    if (!seed.stencil.empty())
        std::memcpy(static_cast<uint8_t*>(mapped) + seed.depth.size(),
                    seed.stencil.data(), seed.stencil.size());
    vkUnmapMemory(ctx.dev, memory);
    const VkImageLayout old_layout = image.layout_initialized
        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    const BackendSubmissionState transfer = submit_persistent_ds_transfer(
        ctx, image.image, aspects, old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        buffer, seed.width, seed.height, seed.depth_valid, seed.stencil_valid, true, error);
    if (transfer != BackendSubmissionState::Pending) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr);
        vkFreeMemory(ctx.dev, memory, nullptr);
    }
    if (transfer != BackendSubmissionState::Complete) return false;
    image.layout_initialized = true;
    image.depth_valid = seed.depth_valid; image.stencil_valid = seed.stencil_valid;
    return true;
}

// fragment_spirv_uses_internal_gds walks the ENTIRE SPIR-V module and builds two unordered_maps,
// and both call sites below invoke it once per DRAW on a module that changes only when the shader
// does. On Blue Prince's collapsed Day One state (~4,048 draws/submit) it measured 5.64% of the
// saturated render thread, plus the allocator churn of two map constructions per draw (#2334).
//
// Memoized on fs_identity -- the same key, and for the same reason, as the subgroup-scan memo
// (#2259) further down: DrawItem identities come from the shader-recompile cache as
// `cache.next_identity++` (src/gpu/execute/gpu_executor.cpp), a monotonic counter that is never reset,
// never decremented and untouched by eviction. So an identity denotes one module for the life of
// the process -- an evicted shader recompiles to a NEW identity, which misses here and refills,
// and the stale entry is simply never looked up again. The counter starts at 1, so identity 0 is
// never minted and the identity-0 fallthrough below cannot collide with a real module.
//
// The memoized value is a pure function of the module words alone -- it reads no device state --
// so a hit is exactly what the call would have returned.
//
// PROSPER_NO_GDS_SCAN_MEMO: opt out, so the A/B for this change is single-variable on ONE binary
// rather than a comparison of two builds, matching PROSPER_NO_SUBGROUP_SCAN_MEMO (#2259),
// PROSPER_NO_INDEX_ARENA (#2258) and the PROSPER_NO_BACKEND_* family.
inline bool fragment_uses_internal_gds_memoized(uint64_t fs_identity,
                                                const std::vector<uint32_t>& fs_words) {
    static const bool no_gds_scan_memo = getenv("PROSPER_NO_GDS_SCAN_MEMO") != nullptr;
    if (!fs_identity || no_gds_scan_memo)
        return prosper::gpu::fragment_spirv_uses_internal_gds(fs_words);
    static thread_local std::unordered_map<uint64_t, bool> gds_scan_memo;
    constexpr size_t kGdsScanMemoMaxEntries = 4096;
    const auto found = gds_scan_memo.find(fs_identity);
    if (found != gds_scan_memo.end()) return found->second;
    const bool uses = prosper::gpu::fragment_spirv_uses_internal_gds(fs_words);
    // Cleared wholesale rather than aged, for the same reason as the subgroup-scan memo: there is no
    // LRU bookkeeping worth paying for on a path this hot, and a clear means this memo has STOPPED
    // WORKING (the scans are paid again in full, plus the churn) rather than that a threshold was
    // brushed. 523 distinct shaders per submit leaves ample headroom against 4,096.
    if (gds_scan_memo.size() >= kGdsScanMemoMaxEntries) gds_scan_memo.clear();
    gds_scan_memo.emplace(fs_identity, uses);
    return uses;
}

// `submission_batch` is an explicit live-renderer ownership scope. Calls with no requested CPU
// readback may return after recording; `flush_submission_batch` submits every accumulated command
// buffer in order, waits once, and releases all retained resources. Omitting the batch preserves the
// synchronous test/replay contract.
inline std::vector<uint8_t> render_draw_pass_rgba(std::span<const BackendDraw> draws,
                                                  uint32_t W, uint32_t H,
                                                  const uint8_t* seed_rgba = nullptr,
                                                  const float* clear_rgba = nullptr,
                                                  bool persist_depth_stencil = false,
                                                  const BackendColorTarget* color_target = nullptr,
                                                  const uint8_t* seed_rgba1 = nullptr,
                                                  const float* clear_rgba1 = nullptr,
                                                  std::vector<uint8_t>* out_rgba1 = nullptr,
                                                  BackendSubmissionBatch* submission_batch = nullptr,
                                                  bool flush_submission_batch = true,
                                                  std::span<const BackendDraw> logical_ds_draws = {},
                                                  BackendMrtOutputs* mrt_outputs = nullptr,
                                                  // #2283: does the CALLER want colour pixels back?
                                                  // Not "is there a colour target" -- absent target
                                                  // already means readback, because then the returned
                                                  // vector is the only way to get results, which is
                                                  // how every offscreen test asserts. Defaults true so
                                                  // every existing caller is bit-identical.
                                                  bool want_color_readback = true) {
    using TimingClock = std::chrono::steady_clock;
    const bool timing_log_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const prosper::frontend::PerformanceTimingMode timing_mode =
        prosper::frontend::performance_timing_mode(
            timing_log_enabled, prosper::frontend::interactive_performance_timing());
    const bool timing_enabled = timing_mode.measure;
    const auto timing_start = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    if (timing_enabled) backend_render_timing_stats_storage() = {};
    BackendColorTargetStats& color_target_stats = backend_color_target_stats_storage();
    color_target_stats = {};
    BackendResourceReuseStats& resource_reuse_stats = backend_resource_reuse_stats_storage();
    resource_reuse_stats = {};
    maybe_report_hash_stats();   // gated cumulative hashing economics (#1268)
    std::vector<uint8_t> out;
    if (out_rgba1) out_rgba1->clear();
    if (mrt_outputs)
        for (auto& color : mrt_outputs->colors) color.clear();
    if (draws.empty()) return out;
    // PROSPER_RENDER_DROP_UNPROVEN_DRAW=1 — DIAGNOSTIC. Discard only the draw carrying an
    // unproven resource instead of the whole batch.
    //
    // The default below returns an EMPTY batch: one resource failing its contract drops every draw
    // submitted alongside it. That is deliberately conservative, and its blast radius has never been
    // measured. On GTA V exactly one binding fails -- set=1 binding=46, a WRITABLE portable-uvec4
    // storage image, with compression, arraying, multisampling and dimensions all fine -- and the
    // question of whether that single unsupported resource is what removes the 3D world cannot be
    // answered without separating "this resource is unusable" from "this frame renders nothing".
    //
    // Diagnostic, not a fix: dropping the draw still loses whatever it would have written, and the
    // real work is supporting writable portable-uvec4 storage images. It exists to make the blast
    // radius measurable.
    const bool drop_only_unproven_draw =
        PROSPER_ENV_ON("PROSPER_RENDER_DROP_UNPROVEN_DRAW");
    std::vector<const BackendDraw*> proven_draws;
    proven_draws.reserve(draws.size());
    for (const BackendDraw& draw : draws) {
        bool draw_proven = true;
        for (const FrameResource& resource : draw.R)
            if (!backend_texture_plane_span_valid(resource) ||
                !backend_storage_image_numeric_contract_valid(resource)) {
                if (resource.is_storage_image) {
                    static uint32_t storage_contract_warnings = 0;
                    if (storage_contract_warnings++ < 32u)
                        std::fprintf(
                            stderr,
                            "[render] storage image set=%u binding=%u sampled-type=%u "
                            "view-format=%d materialized=%u rejected before Vulkan\n",
                            resource.set, resource.binding,
                            static_cast<unsigned>(resource.storage_image_numeric_class),
                            static_cast<int>(backend_color_format(resource.texture_format)),
                            resource.storage_image_contract_valid ? 1u : 0u);
                }
                draw_proven = false;
                break;
            }
        if (draw_proven) proven_draws.push_back(&draw);
        else if (!drop_only_unproven_draw) return out;
    }
    std::vector<BackendDraw> proven_storage;
    if (drop_only_unproven_draw && proven_draws.size() != draws.size()) {
        static uint32_t dropped_reports = 0;
        if (dropped_reports++ < 16u)
            std::fprintf(stderr,
                         "[render] DROP_UNPROVEN_DRAW: kept %zu of %zu draws in this batch\n",
                         proven_draws.size(), draws.size());
        if (proven_draws.empty()) return out;
        proven_storage.reserve(proven_draws.size());
        for (const BackendDraw* draw : proven_draws) proven_storage.push_back(*draw);
        // `draws` is a by-value span; rebinding it keeps every downstream use unchanged.
        // proven_storage outlives that use because it is declared in this scope.
        draws = std::span<const BackendDraw>(proven_storage);
    }
    if (backend_has_unproven_submission()) return out;
    // #2953. Everything from here on reads and writes the backend's persistent-resource domain --
    // the pipeline, pipeline-layout, colour-target, depth/stencil and texture caches, their byte
    // totals and their generation counters. Taken BEFORE `direct_submission` is constructed, so the
    // batch destructor (which submits, and whose failure cleanups touch the texture cache) still
    // runs inside the critical section; a guard declared after it would be destroyed first and
    // leave exactly those accesses outside. See the block comment on the guard for why the lock
    // lives here rather than being inherited from `g_agc_state_mu`.
    const BackendPersistentResourceGuard persistent_resource_guard;
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) return out;
    BackendSubmissionBatch direct_submission;
    BackendSubmissionBatch& active_submission = submission_batch
        ? *submission_batch : direct_submission;
    const bool avoid_cache_eviction = active_submission.pending() ||
                                      backend_has_unproven_submission();
    const bool persistent_color_targets_enabled =
        getenv("PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS") == nullptr;
    const bool persistent_color_enabled = persistent_color_targets_enabled && color_target &&
                                          color_target->persistent_id;
    const uint64_t color_target_generation = ++persistent_color_target_generation();
    VkInstance inst = ctx.inst; (void)inst; VkPhysicalDevice phys = ctx.phys;
    VkDevice dev = ctx.dev; VkQueue queue = ctx.queue; uint32_t qfi = ctx.qfi;
    const bool aniso_enabled = ctx.aniso_enabled; const float max_aniso_limit = ctx.max_aniso_limit;
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    init_persistent_color_target_device_budget(memp);   // size the residency budget once (#1177)
    init_persistent_texture_cache_device_budget(memp);
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const bool use_color1 = out_rgba1 != nullptr ||
        (mrt_outputs && mrt_outputs->color_count > 1);
    const uint32_t color_count = mrt_outputs
        ? std::clamp(mrt_outputs->color_count, 1u, prosper::gpu::kColorTargetCount)
        : (use_color1 ? 2u : 1u);
    const auto first_pipeline_format = [&](uint32_t slot) {
        for (const auto& draw : draws) {
            if (!draw.ps) continue;
            uint32_t raw = draw.ps->color_targets[slot].format;
            if (slot == 0 && !raw) raw = draw.ps->color0_format;
            if (slot == 1 && !raw) raw = draw.ps->color1_format;
            const VkFormat format = static_cast<VkFormat>(raw);
            if (format != VK_FORMAT_UNDEFINED) return backend_color_format(format);
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    };
    const VkFormat FMT = color_target && color_target->format != VK_FORMAT_UNDEFINED
        ? backend_color_format(color_target->format)
        : first_pipeline_format(0);
    std::array<VkFormat, prosper::gpu::kColorTargetCount> color_formats{};
    std::array<VkDeviceSize, prosper::gpu::kColorTargetCount> color_bytes{};
    std::array<VkDeviceSize, prosper::gpu::kColorTargetCount> color_offsets{};
    color_formats[0] = FMT;
    VkDeviceSize readback_bytes = 0;
    for (uint32_t slot = 0; slot < color_count; ++slot) {
        if (slot) color_formats[slot] = first_pipeline_format(slot);
        if (slot == 1 && color_target && color_target->format1 != VK_FORMAT_UNDEFINED)
            color_formats[slot] = backend_color_format(color_target->format1);
        color_offsets[slot] = readback_bytes;
        color_bytes[slot] = static_cast<VkDeviceSize>(W) * H *
            backend_color_bytes_per_pixel(color_formats[slot]);
        readback_bytes += color_bytes[slot];
    }
    // A slot-1 seed may arrive by either route; the explicit parameter wins when both are present,
    // since an argument is the caller being specific.
    const uint8_t* const effective_seed1 =
        seed_rgba1 ? seed_rgba1
                   : (color_target ? color_target->seed_rgba1_slot : nullptr);
    const VkFormat FMT1 = use_color1 ? color_formats[1] : VK_FORMAT_UNDEFINED;
    const bool persistent_color1_enabled = persistent_color_targets_enabled && use_color1 &&
                                           color_target && color_target->persistent_id1 &&
                                           color_target->persistent_id1 != color_target->persistent_id;
    const VkDeviceSize bytes = color_bytes[0];
    const VkDeviceSize bytes1 = use_color1 ? color_bytes[1] : 0;
    const uint32_t ds_attachment = color_count;
    // A feedback split is a physical Vulkan detail, not a guest draw-pass boundary. Attachment
    // identity/format and fresh-image fallback values therefore come from the original logical
    // span. Segment-local use/write/layout decisions and explicit clears remain local below.
    const std::span<const BackendDraw> logical_draws = logical_ds_draws.empty()
        ? draws : logical_ds_draws;
    // Depth attachment is created if ANY draw enables the depth test (the shared render pass has one
    // fixed attachment set); each draw's pipeline sets its own depthTest/Write/CompareOp. A frame with
    // no depth-using draw takes the color-only path unchanged.
    bool use_depth = false, use_stencil = false;
    // Initial values for a newly-created depth/stencil attachment (#371). Existing guest-identified
    // surfaces LOAD their contents below, and explicit DB_RENDER_CONTROL clears execute in draw order.
    // Latch depth and stencil independently: coupling them let a stencil-only first draw force the
    // wrong reverse-Z depth initial value for a later draw (#457).
    float    depth_clear   = 1.0f;
    uint32_t stencil_clear = 0;
    bool     got_depth_clear = false, got_stencil_clear = false;
    // A DEPTH_CLEAR_ENABLE bit only acts through the enabled depth-write path (see
    // depth_clear_effective) — the writes-disabled Blue Prince light-loop shape is depth-inert and
    // must not force a depth attachment, pick the clear value, or clear in-pass (#1287).
    const auto effective_depth_clear = [](const prosper::gpu::ResolvedPipelineState* ps) {
        return depth_clear_effective(ps->depth_clear_enable, ps->depth_test_enable,
                                     ps->depth_write_enable, ps->depth_compare_op);
    };
    for (const auto& d : draws) {
        if (!d.ps) continue;
        if (d.ps->depth_test_enable || effective_depth_clear(d.ps)) use_depth = true;
        if (d.ps->stencil_enable ||
            stencil_clear_effective(d.ps->stencil_clear_enable, d.ps->stencil_enable,
                                    d.ps->stencil_write_mask[0], d.ps->stencil_write_mask[1]))
            use_stencil = true;
    }
    for (const auto& d : logical_draws) {
        if (!d.ps) continue;
        if (d.ps->depth_test_enable || effective_depth_clear(d.ps)) {
            // DB_DEPTH_CLEAR is only consumed when DB_RENDER_CONTROL requests an actual clear. Merely
            // programming the register does not initialize a newly-created host depth image: it is
            // commonly stale state, and Astro Bot even leaves the packed 1920x1080 max coordinates
            // (0x0437077f) there. Interpreting those bits as a float initialized every LEQUAL surface
            // to 2.15e-36 and rejected the entire scene. Without an explicit clear, approximate the
            // unavailable guest depth contents with the compare-appropriate far value (#371).
            //
            // An explicit clear wins even on ALWAYS/NEVER. Otherwise those compare modes do not depend
            // on the initial value and must not poison the first later meaningful draw (#508).
            // Initial-contents approximation for a NEWLY-CREATED host image: an explicitly
            // programmed clear expresses the guest's intended surface value even when this pass's
            // write path cannot apply it (#508 keeps this for ALWAYS/NEVER too). The write-path
            // gate below governs what the pass DOES (in-pass clears, validity, recency), not what
            // a fresh image starts as. Note this latch sits inside the depth-using block above, so
            // a fully write-path-disabled clear draw (the #1287 inert shape) never reaches it —
            // only a test-enabled clear can supply fresh-image contents.
            if (!got_depth_clear && d.ps->depth_clear_enable) {
                depth_clear = d.ps->depth_clear_value;
                got_depth_clear = true;
            } else if (!got_depth_clear && d.ps->depth_compare_op != VK_COMPARE_OP_ALWAYS
                                            && d.ps->depth_compare_op != VK_COMPARE_OP_NEVER) {
                depth_clear = (d.ps->depth_compare_op == VK_COMPARE_OP_GREATER ||
                               d.ps->depth_compare_op == VK_COMPARE_OP_GREATER_OR_EQUAL)
                    ? 0.0f : 1.0f;
                got_depth_clear = true;
            }
        }
        // #1355: like the depth gate above, a STENCIL_CLEAR_ENABLE bit with the stencil write
        // path disabled is inert and must not force a stencil attachment or clear in-pass. The
        // initial-value latch stays reachable through stencil_enable only, mirroring the depth
        // fresh-image approximation.
        if (d.ps->stencil_enable ||
            stencil_clear_effective(d.ps->stencil_clear_enable, d.ps->stencil_enable,
                                    d.ps->stencil_write_mask[0], d.ps->stencil_write_mask[1])) {
            // Mirror the depth contract (#371/#508 via #1361): DB_STENCIL_CLEAR is consumed as a
            // fresh image's initial contents only when the guest explicitly requests a clear.
            // Latching it from every stencil-using draw let a STALE register word poison a new
            // plane (the stencil analog of #371's Astro Bot depth case); without a clear the
            // unknown guest contents are approximated as 0.
            if (!got_stencil_clear && d.ps->stencil_clear_enable) {
                stencil_clear = d.ps->stencil_clear_value; got_stencil_clear = true;
            }
        }
    }
    if (const char* v = getenv("PROSPER_STENCIL_CLEAR"))
        stencil_clear = static_cast<uint32_t>(strtoul(v, nullptr, 0)) & 0xFFu;
    // Diagnostic A/B twin of PROSPER_STENCIL_CLEAR: override the derived initial depth value.
    // The #371 approximation picks the compare-appropriate always-pass value; sweeping this
    // instead reveals whether a pass's draws sit at DIFFERENT depths (a mid value culls some
    // draws but not others), i.e. whether real guest depth contents would gate them.
    // PROSPER_DEPTH_CLEAR_WHY=1 — how a pass's fresh-image depth value was derived, next to what its
    // draws actually compare with.
    //
    // The latch above is FIRST-DRAW-WINS, so one draw fixes the initial value for every later draw in
    // the pass. #457 already fixed one instance of that class (a stencil-only first draw forcing the
    // wrong reverse-Z depth value); this reports whether it is happening again on the depth side,
    // which the derived float alone cannot show. A pass whose draws are predominantly GREATER but
    // whose value came out 1.0 is rejecting its own geometry.
    if (PROSPER_ENV_ON("PROSPER_DEPTH_CLEAR_WHY") && use_depth) {
        uint32_t greater = 0, less = 0, other = 0, explicit_clear = 0;
        uint32_t greater_colour = 0, less_colour = 0;
        uint32_t reversed_viewports = 0, forward_viewports = 0;
        float vp_min = -1.0f, vp_max = -1.0f;
        int first_op = -1;
        // What DB_DEPTH_CLEAR actually holds, whether or not a draw enables it. #371 forbids
        // CONSUMING it without an explicit enable -- Astro Bot leaves packed 1920x1080 max
        // coordinates in the register, which read as 2.15e-36 and rejected whole scenes -- but
        // reading it here is free and says whether a plausible clear value is even present on the
        // passes that need one. A surface fast-cleared through HTILE metadata sets no draw's
        // depth_clear_enable, so the value it was cleared to is invisible to the latch above.
        float first_clear_value = -1.0f;
        for (const auto& d : logical_draws) {
            if (!d.ps) continue;
            if (!(d.ps->depth_test_enable || effective_depth_clear(d.ps))) continue;
            if (first_op < 0) {
                first_op = static_cast<int>(d.ps->depth_compare_op);
                first_clear_value = d.ps->depth_clear_value;
            }
            // The viewport depth range, which is how a reverse-Z surface declares itself: a guest
            // using near=1/far=0 programs min_depth=1, max_depth=0. If the range is the ordinary
            // 0..1 while the compares are GREATER, the reversal lives in the projection matrix
            // instead and the stored values are what the shader emits.
            if (d.ps->has_viewport) {
                if (d.ps->min_depth > d.ps->max_depth) ++reversed_viewports;
                else ++forward_viewports;
                vp_min = d.ps->min_depth; vp_max = d.ps->max_depth;
            }
            if (d.ps->depth_clear_enable) ++explicit_clear;
            // Split by whether the draw WRITES COLOUR. The initial value should serve the draws
            // whose output is lost when they fail, and a depth-only or mask draw loses nothing
            // visible. Plain majority ignored this and reproduced the old answer on exactly the
            // passes that matter (7 vs 7, and 6 vs 7 the wrong way).
            const bool writes_colour = std::any_of(
                d.ps->color_targets.begin(), d.ps->color_targets.end(),
                [](const auto& t) { return t.write_mask != 0; });
            switch (d.ps->depth_compare_op) {
                case VK_COMPARE_OP_GREATER: case VK_COMPARE_OP_GREATER_OR_EQUAL:
                    ++greater; if (writes_colour) ++greater_colour; break;
                case VK_COMPARE_OP_LESS: case VK_COMPARE_OP_LESS_OR_EQUAL:
                    ++less; if (writes_colour) ++less_colour; break;
                default: ++other; break;
            }
        }
        static std::mutex why_mutex;
        static std::map<std::tuple<uint64_t, float, uint32_t, uint32_t, uint32_t>, uint64_t> seen;
        std::lock_guard lock(why_mutex);
        const uint64_t n = ++seen[{color_target ? color_target->persistent_id : 0ull,
                                   depth_clear, greater, less, other}];
        if ((n & (n - 1)) == 0)
            fprintf(stderr,
                    "[depth-clear-why] target=0x%llx derived=%.3f reg_clear=%.6g first_op=%d "
                    "explicit=%u compares{greater=%u less=%u other=%u} "
                    "colour{greater=%u less=%u} vp{min=%g max=%g rev=%u fwd=%u} "
                    "draws=%zu (x%llu)\n",
                    (unsigned long long)(color_target ? color_target->persistent_id : 0ull),
                    depth_clear, first_clear_value, first_op, explicit_clear, greater, less, other,
                    greater_colour, less_colour, vp_min, vp_max, reversed_viewports,
                    forward_viewports, logical_draws.size(), (unsigned long long)n);
    }
    if (const char* v = PROSPER_ENV_VALUE("PROSPER_DEPTH_CLEAR"))
        depth_clear = strtof(v, nullptr);
    if (getenv("PROSPER_NO_DEPTH"))   use_depth = false;     // diag: isolate depth-test rejection
    if (getenv("PROSPER_NO_STENCIL")) use_stencil = false;   // diag: isolate stencil masking
    const bool use_ds = use_depth || use_stencil;
    // Use a stencil-capable depth format ONLY when a draw actually uses stencil (a UI mask). The
    // depth-only path keeps the original D32 depth-only format + aspect, so existing render tests are
    // byte-identical (#264).
    // Physical passes created for one logical draw batch must retain the exact DS attachment that
    // the original unsplit pass would have selected. Segment-local first states can carry aliased
    // or partly stale DB bases (for example dr=0,dw=P followed by dr=P,dw=P); re-keying at the pass
    // boundary would bind a newly-cleared attachment even though sampled-depth lookup still finds
    // the producer image. The logical span is supplied only by the split wrapper below.
    const prosper::gpu::ResolvedPipelineState* identity = nullptr;
    bool logical_use_stencil = false;
    for (const auto& d : logical_draws) {
        if (d.ps && (d.ps->stencil_enable ||
                     stencil_clear_effective(d.ps->stencil_clear_enable,
                                             d.ps->stencil_enable,
                                             d.ps->stencil_write_mask[0],
                                             d.ps->stencil_write_mask[1])))
            logical_use_stencil = true;
        if (d.ps && (d.ps->depth_test_enable || d.ps->stencil_enable ||
                     effective_depth_clear(d.ps) ||
                     stencil_clear_effective(d.ps->stencil_clear_enable, d.ps->stencil_enable,
                                             d.ps->stencil_write_mask[0],
                                             d.ps->stencil_write_mask[1])) && !identity)
            identity = d.ps;
    }
    if (getenv("PROSPER_NO_STENCIL")) logical_use_stencil = false;
    const bool has_ds_identity = identity && (identity->depth_read_base || identity->depth_write_base ||
                                               identity->stencil_read_base || identity->stencil_write_base);
    const bool persistent_ds = persist_depth_stencil && use_ds && has_ds_identity;
    // A persistent attachment must keep a stable format even across a depth-only call between stencil
    // users. Nonzero stencil identity means this guest surface owns a stencil plane.
    const bool format_has_stencil = logical_use_stencil || (persistent_ds &&
        (identity->stencil_read_base || identity->stencil_write_base));
    const VkFormat DFMT = format_has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    const VkImageAspectFlags DASPECT = VK_IMAGE_ASPECT_DEPTH_BIT |
                                       (format_has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    VkImage dimg = VK_NULL_HANDLE; VkDeviceMemory dmem = VK_NULL_HANDLE; VkImageView dview = VK_NULL_HANDLE;
    bool ds_layout_initialized = false;
    bool depth_was_valid = false, stencil_was_valid = false;
    bool depth_used_meaningfully = false;
    bool depth_may_be_written = false;
    uint64_t depth_write_command_order = 0;
    bool stencil_may_be_written = false;
    for (const auto& d : draws) {
        if (!d.ps) continue;
        depth_used_meaningfully |= effective_depth_clear(d.ps) || d.ps->depth_write_enable ||
            (d.ps->depth_test_enable && d.ps->depth_compare_op != VK_COMPARE_OP_ALWAYS &&
                                        d.ps->depth_compare_op != VK_COMPARE_OP_NEVER);
        const bool draw_may_write_depth = persistent_ds_pass_may_write_depth(
            d.ps->depth_clear_enable, d.ps->depth_test_enable, d.ps->depth_write_enable,
            d.ps->depth_compare_op);
        depth_may_be_written |= draw_may_write_depth;
        if (draw_may_write_depth)
            depth_write_command_order = std::max(depth_write_command_order, d.command_order);
        stencil_may_be_written |= stencil_clear_effective(
            d.ps->stencil_clear_enable, d.ps->stencil_enable,
            d.ps->stencil_write_mask[0], d.ps->stencil_write_mask[1]);
        // Keep the layout decision conservative: Vulkan validates attachment writability from
        // pipeline state, and a nonzero guest write mask leaves the aspect writable even when this
        // draw happens to program KEEP for every outcome.
        stencil_may_be_written |= d.ps->stencil_enable &&
            (d.ps->stencil_write_mask[0] != 0u || d.ps->stencil_write_mask[1] != 0u);
    }
    PersistentDsImage* cached_ds = nullptr;
    PersistentDsKey ds_key{};
    if (persistent_ds) {
        uint64_t htile_identity = identity->htile_data_base;
        // Captures through v5 did not serialize DB_HTILE_DATA_BASE. The explicit replay migration
        // switch recovers one preserved artifact whose manifest proves Unity allocated HTILE in the
        // 64 KiB block immediately before stencil. Never infer this layout for live/current state.
        if (!htile_identity && getenv("PROSPER_GPU_REPLAY_LEGACY_HTILE_BEFORE_STENCIL") &&
            identity->stencil_read_base >= 0x10000)
            htile_identity = identity->stencil_read_base - 0x10000;
        ds_key = persistent_ds_key_for(*identity, htile_identity, W, H, (uint32_t)DFMT);
        cached_ds = &persistent_ds_cache()[ds_key];
        // The key carries the PASS extent, so one guest depth surface reached through two passes of
        // different extent becomes two independent entries -- and the larger one's guest range is
        // sized from that extent, which is how a 512x512 shadow cascade acquires a 33 MB depth range
        // that swallows its neighbours. Log the first sighting of each identity so the producing
        // pass is nameable rather than inferred.
        static const bool log_new_ds = getenv("PROSPER_DSLOG") != nullptr;
        if (log_new_ds && !cached_ds->image)
            fprintf(stderr, "[ds] new-entry dr=0x%llx dw=0x%llx sr=0x%llx slice=%u extent=%ux%u\n",
                    (unsigned long long)ds_key.dr, (unsigned long long)ds_key.dw,
                    (unsigned long long)ds_key.sr, ds_key.slice, ds_key.w, ds_key.h);
        dimg = cached_ds->image; dmem = cached_ds->memory; dview = cached_ds->view;
        ds_layout_initialized = cached_ds->layout_initialized;
        depth_was_valid = cached_ds->depth_valid;
        stencil_was_valid = cached_ds->stencil_valid;
    }
    // A guest may sample the depth plane while that SAME surface remains attached for depth/stencil
    // tests. Bendy's deferred-lighting and glow passes use read-only depth plus writable stencil in
    // exactly this shape (#1186). The sampled-depth bridge used to reject it as feedback and upload
    // zeros. Vulkan supports the hardware contract directly: keep depth read-only for both the
    // attachment and shader view, while the stencil aspect may remain writable. Never enable the
    // path when any draw in this render pass can write depth.
    bool self_sampled_depth = false;
    if (cached_ds && depth_was_valid && !depth_may_be_written) {
        for (const auto& draw : draws) {
            for (const auto& resource : draw.R) {
                if (!resource.persistent_depth_target_id || resource.img_dim != 1u ||
                    resource.tw != W || resource.th != H)
                    continue;
                if (find_persistent_ds_sampled(resource.persistent_depth_target_id,
                                               resource.tw, resource.th).image == cached_ds) {
                    self_sampled_depth = true;
                    break;
                }
            }
            if (self_sampled_depth) break;
        }
    }
    const VkImageLayout self_depth_layout = format_has_stencil && stencil_may_be_written
        ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    if (PROSPER_ENV_ON("PROSPER_DSLOG")) {
        static uint64_t call_id = 0;
        const uint64_t id = ++call_id;
        fprintf(stderr,
                "[ds] call=%llu size=%ux%u draws=%zu use=%d/%d persistent=%d valid=%d/%d/%d "
                "key=%llx/%llx/%llx/%llx htile=%llx fmt=%u initial=%g/%u\n",
                (unsigned long long)id, W, H, draws.size(), (int)use_depth, (int)use_stencil,
                (int)persistent_ds, (int)ds_layout_initialized,
                (int)depth_was_valid, (int)stencil_was_valid,
                (unsigned long long)ds_key.dr, (unsigned long long)ds_key.dw,
                (unsigned long long)ds_key.sr, (unsigned long long)ds_key.sw,
                (unsigned long long)ds_key.htile, ds_key.fmt, depth_clear, stencil_clear);
        for (size_t i = 0; i < draws.size(); ++i) {
            const auto* ps = draws[i].ps;
            if (!ps) {
                fprintf(stderr, "[ds] call=%llu draw=%zu state=none\n",
                        (unsigned long long)id, i);
                continue;
            }
            fprintf(stderr,
                    "[ds] call=%llu draw=%zu bases=%llx/%llx/%llx/%llx "
                    "depth=%d/%d/op%u clear=%d/%g stencil=%d clear=%d/%u "
                    "view=%08x htile=%llx hsurf=%08x info=%08x/%08x/%08x "
                    "size=%08x/%08x/%08x\n",
                    (unsigned long long)id, i,
                    (unsigned long long)ps->depth_read_base,
                    (unsigned long long)ps->depth_write_base,
                    (unsigned long long)ps->stencil_read_base,
                    (unsigned long long)ps->stencil_write_base,
                    (int)ps->depth_test_enable, (int)ps->depth_write_enable,
                    ps->depth_compare_op, (int)ps->depth_clear_enable, ps->depth_clear_value,
                    (int)ps->stencil_enable, (int)ps->stencil_clear_enable,
                    ps->stencil_clear_value, ps->db_depth_view,
                    (unsigned long long)ps->htile_data_base, ps->db_htile_surface,
                    ps->db_depth_info, ps->db_z_info, ps->db_stencil_info,
                    ps->db_depth_size_xy, ps->db_depth_size, ps->db_depth_slice);
        }
    }

    // Protect every GPU target sampled by this call from LRU eviction while its descriptors are built.
    for (const auto& draw : draws)
        for (const auto& resource : draw.R)
            if (persistent_color_targets_enabled && resource.persistent_render_target_id) {
                if (auto* sampled = find_persistent_color_target(
                        resource.persistent_render_target_id, resource.tw, resource.th,
                        backend_color_format(resource.texture_format)))
                    sampled->last_use = color_target_generation;
                const uint32_t mip_count = std::min<uint32_t>(
                    resource.persistent_render_target_mip_count,
                    static_cast<uint32_t>(resource.persistent_render_target_mip_ids.size()));
                for (uint32_t level = 1; level < mip_count; ++level) {
                    const uint64_t id = resource.persistent_render_target_mip_ids[level];
                    if (!id) continue;
                    if (auto* sampled = find_persistent_color_target(
                            id, std::max(resource.tw >> level, 1u),
                            std::max(resource.th >> level, 1u),
                            backend_color_format(resource.texture_format)))
                        sampled->last_use = color_target_generation;
                }
            }

    bool persistent_color = persistent_color_enabled;
    PersistentColorTargetKey color_key{};
    PersistentColorTargetImage* cached_color = nullptr;
    if (persistent_color) {
        color_key = {color_target->persistent_id, W, H, FMT};
        // PROSPER_TARGET_PHYS (#2932): print each colour target's guest VA and the physical address
        // it maps to, to test whether any target ALIASES the page a title flips to the display. On
        // Stray the answer is no -- and the same census answered a question it was not built for,
        // which is why it is still here: the flipped VAs 0x9fc0000000 / 0x9fc2000000 turn up in this
        // very census as 4K colour attachments, so those buffers are render targets rather than a
        // separate region the renderer never touches (`docs/STRAY_STATUS.md`, § Ruled out).
        //
        // What a line here does and does not license: it is printed after the empty-draw early-out,
        // so the target was BOUND as an attachment of a pass carrying at least one draw. It does not
        // say a draw's output reached memory -- masks, discards and store behaviour all sit
        // downstream -- so do not read a line here as "the picture was written".
        //
        // EVERY colour slot is censused, not just slot 0, and that scope is the whole point rather
        // than thoroughness for its own sake. `is_live_render_target` consults `g_rtt`, and `g_rtt`
        // is populated for every slot (`live_renderer.cpp` keys it by `pass_bases[slot]`), so a
        // slot-1..7 target aliasing the scanout page would be a live render target the present path
        // ALREADY knows by VA -- exactly the link the alias hypothesis proposes. A slot-0-only
        // census would leave that case unobserved while reading like a general answer, which is the
        // charter's "positive control tests the discriminator, never the domain" in miniature.
        static const bool target_phys_census = std::getenv("PROSPER_TARGET_PHYS") != nullptr;
        if (target_phys_census) {
            const auto census_slot = [&](unsigned slot, uint64_t va) {
                if (!va) return;
                uint64_t tphys = 0;
                size_t alias_n = 0;
                if (prosper::host::guest_write_watch_va_to_phys(va, tphys, &alias_n))
                    std::fprintf(stderr, "[target-phys] slot%u va=0x%llx %ux%u -> phys=0x%llx\n",
                                 slot, (unsigned long long)va, W, H, (unsigned long long)tphys);
                else
                    std::fprintf(stderr, "[target-phys] slot%u va=0x%llx %ux%u -> UNRESOLVED "
                                         "(aliases=%zu)\n",
                                 slot, (unsigned long long)va, W, H, alias_n);
            };
            census_slot(0, color_target->persistent_id);
            // Slot 1 lives in its own field; only report it when it is a DISTINCT surface, matching
            // the test the MRT path itself applies, so a single-target pass does not print twice.
            if (color_target->persistent_id1 &&
                color_target->persistent_id1 != color_target->persistent_id)
                census_slot(1, color_target->persistent_id1);
            for (size_t sl = 2; sl < color_target->persistent_id_slots.size(); ++sl)
                census_slot((unsigned)sl, color_target->persistent_id_slots[sl]);
        }
        auto [found, inserted] = persistent_color_target_cache().try_emplace(color_key);
        cached_color = &found->second;
        cached_color->last_use = color_target_generation;
    }

    VkImage img = cached_color ? cached_color->image : VK_NULL_HANDLE;
    VkDeviceMemory imem = cached_color ? cached_color->memory : VK_NULL_HANDLE;
    VkImageView view = cached_color ? cached_color->view : VK_NULL_HANDLE;
    if (!img) {
        VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = FMT; imgci.extent = {W, H, 1};
        imgci.mipLevels = 1; imgci.arrayLayers = 1; imgci.samples = VK_SAMPLE_COUNT_1_BIT;
        imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      (persistent_color ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u) |
                      ((seed_rgba || persistent_color) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
        // #3180: the result is the primary test, the handle the secondary one.
        if (create_color_target_image(dev, imgci, RenderColorTargetCreateSite::Slot0,
                                      &img) != VK_SUCCESS) {
            if (cached_color) persistent_color_target_cache().erase(color_key);
            return out;
        }
        VkMemoryRequirements ir{}; vkGetImageMemoryRequirements(dev, img, &ir);
        VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        iai.allocationSize = ir.size; iai.memoryTypeIndex = pick(ir.memoryTypeBits, 0);
        if (persistent_color) {
            const VkDeviceSize limit = persistent_color_target_limit();
            // NOTE: the unsigned `limit - ir.size` below is guarded by short-circuit ordering, not by the
            // budget floor: `ir.size > limit` (here) and `ir.size <= limit` (retention check) are evaluated
            // first, so the subtraction only runs when `ir.size <= limit`. Keep those operands ahead of it —
            // reordering would let a single over-budget target underflow the subtraction to a huge value.
            while (!avoid_cache_eviction &&
                   (persistent_color_target_cache().size() > persistent_color_target_count_limit() || ir.size > limit ||
                    persistent_color_target_bytes() > limit - ir.size) &&
                   evict_persistent_color_target(ctx, color_target_generation)) {}
            if (ir.size <= limit && persistent_color_target_cache().size() <=
                    persistent_color_target_count_ceiling(avoid_cache_eviction) &&
                persistent_color_target_bytes() <= limit - ir.size &&
                vkAllocateMemory(dev, &iai, nullptr, &imem) == VK_SUCCESS) {
                cached_color->bytes = ir.size;
                persistent_color_target_bytes() += ir.size;
            } else {
                vkDestroyImage(dev, img, nullptr);
                img = VK_NULL_HANDLE;
                persistent_color_target_cache().erase(color_key);
                cached_color = nullptr;
                persistent_color = false;
                imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              (seed_rgba ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
                if (create_color_target_image(dev, imgci,
                                              RenderColorTargetCreateSite::Slot0Fallback,
                                              &img) != VK_SUCCESS)
                    return out;
                vkGetImageMemoryRequirements(dev, img, &ir);
                iai.allocationSize = ir.size; iai.memoryTypeIndex = pick(ir.memoryTypeBits, 0);
            }
        }
        if (!imem)
            imem = allocate_transient_render_memory(dev, iai.allocationSize, iai.memoryTypeIndex);
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = FMT;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const bool target_ready = imem &&
            vkBindImageMemory(dev, img, imem, 0) == VK_SUCCESS &&
            vkCreateImageView(dev, &ivci, nullptr, &view) == VK_SUCCESS && view;
        if (!target_ready) {
            // Caching a half-created target (null view / unbound memory) would break every
            // later LOAD of this identity (#1383): drop it fail-visibly instead.
            if (view) vkDestroyImageView(dev, view, nullptr);
            vkDestroyImage(dev, img, nullptr);
            if (cached_color) {
                if (cached_color->bytes) persistent_color_target_bytes() -= cached_color->bytes;
                if (imem) vkFreeMemory(dev, imem, nullptr);
                persistent_color_target_cache().erase(color_key);
            } else if (imem) {
                release_transient_render_memory(dev, imem);
            }
            return out;
        }
        if (cached_color) {
            cached_color->image = img;
            cached_color->memory = imem;
            cached_color->view = view;
        }
    }
    const bool load_cached_color = cached_color && cached_color->valid &&
                                   color_target->load_existing && !seed_rgba;
    if (cached_color) {
        ++color_target_stats.writes;
        color_target_stats.write_hits = load_cached_color ? 1 : 0;
    }
    if (color_target && color_target->persistent_id && PROSPER_ENV_ON("PROSPER_BACKEND_LOAD_LOG"))
        fprintf(stderr,
                "[backend-load] id=0x%llx %ux%u fmt=%d cached=%d valid=%d load_existing=%d "
                "seed=%d readback=%d -> load=%d\n",
                (unsigned long long)color_target->persistent_id, W, H, (int)FMT,
                cached_color != nullptr, cached_color ? (int)cached_color->valid : -1,
                (int)color_target->load_existing, seed_rgba != nullptr,
                (int)color_target->readback, (int)load_cached_color);

    std::array<VkImage, prosper::gpu::kColorTargetCount> extra_images{};
    std::array<VkDeviceMemory, prosper::gpu::kColorTargetCount> extra_memories{};
    std::array<VkImageView, prosper::gpu::kColorTargetCount> extra_views{};
    std::array<PersistentColorTargetImage*, prosper::gpu::kColorTargetCount> cached_extra{};
    std::array<PersistentColorTargetKey, prosper::gpu::kColorTargetCount> extra_keys{};
    // One decision, consumed by the pre-pass barrier, the attachment's initialLayout and the LOAD op
    // alike. Computed where all three can see it so they cannot disagree about whether this slot's
    // contents are being carried forward.
    std::array<bool, prosper::gpu::kColorTargetCount> load_extra{};
    bool persistent_color1 = persistent_color1_enabled;
    PersistentColorTargetKey color_key1{};
    PersistentColorTargetImage* cached_color1 = nullptr;
    if (persistent_color1) {
        color_key1 = {color_target->persistent_id1, W, H, FMT1};
        auto [found, inserted] = persistent_color_target_cache().try_emplace(color_key1);
        cached_color1 = &found->second;
        cached_color1->last_use = color_target_generation;
    }

    VkImage img1 = cached_color1 ? cached_color1->image : VK_NULL_HANDLE;
    VkDeviceMemory imem1 = cached_color1 ? cached_color1->memory : VK_NULL_HANDLE;
    VkImageView view1 = cached_color1 ? cached_color1->view : VK_NULL_HANDLE;
    if (use_color1 && !img1) {
        VkImageCreateInfo color1_ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        color1_ci.imageType = VK_IMAGE_TYPE_2D; color1_ci.format = FMT1;
        color1_ci.extent = {W, H, 1}; color1_ci.mipLevels = 1; color1_ci.arrayLayers = 1;
        color1_ci.samples = VK_SAMPLE_COUNT_1_BIT; color1_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        color1_ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          (persistent_color1 ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u) |
                          ((effective_seed1 || persistent_color1)
                               ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
        if (create_color_target_image(dev, color1_ci, RenderColorTargetCreateSite::Slot1,
                                      &img1) != VK_SUCCESS) {
            if (cached_color1) persistent_color_target_cache().erase(color_key1);
            return out;
        }
        VkMemoryRequirements color1_requirements{};
        vkGetImageMemoryRequirements(dev, img1, &color1_requirements);
        VkMemoryAllocateInfo color1_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        color1_allocation.allocationSize = color1_requirements.size;
        color1_allocation.memoryTypeIndex = pick(color1_requirements.memoryTypeBits, 0);
        if (persistent_color1) {
            const VkDeviceSize limit = persistent_color_target_limit();
            while (!avoid_cache_eviction &&
                   (persistent_color_target_cache().size() >
                            persistent_color_target_count_limit() ||
                    color1_requirements.size > limit ||
                    persistent_color_target_bytes() > limit - color1_requirements.size) &&
                   evict_persistent_color_target(ctx, color_target_generation)) {}
            if (color1_requirements.size <= limit &&
                persistent_color_target_cache().size() <=
                    persistent_color_target_count_ceiling(avoid_cache_eviction) &&
                persistent_color_target_bytes() <= limit - color1_requirements.size &&
                vkAllocateMemory(dev, &color1_allocation, nullptr, &imem1) == VK_SUCCESS) {
                cached_color1->bytes = color1_requirements.size;
                persistent_color_target_bytes() += color1_requirements.size;
            } else {
                vkDestroyImage(dev, img1, nullptr);
                img1 = VK_NULL_HANDLE;
                persistent_color_target_cache().erase(color_key1);
                cached_color1 = nullptr;
                persistent_color1 = false;
                color1_ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  (effective_seed1 ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
                if (create_color_target_image(dev, color1_ci,
                                              RenderColorTargetCreateSite::Slot1Fallback,
                                              &img1) != VK_SUCCESS)
                    return out;
                vkGetImageMemoryRequirements(dev, img1, &color1_requirements);
                color1_allocation.allocationSize = color1_requirements.size;
                color1_allocation.memoryTypeIndex = pick(color1_requirements.memoryTypeBits, 0);
            }
        }
        if (!imem1)
            imem1 = allocate_transient_render_memory(
                dev, color1_allocation.allocationSize, color1_allocation.memoryTypeIndex);
        VkImageViewCreateInfo color1_view_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        color1_view_ci.image = img1; color1_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        color1_view_ci.format = FMT1;
        color1_view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const bool color1_ready = imem1 &&
            vkBindImageMemory(dev, img1, imem1, 0) == VK_SUCCESS &&
            vkCreateImageView(dev, &color1_view_ci, nullptr, &view1) == VK_SUCCESS && view1;
        if (!color1_ready) {
            if (view1) vkDestroyImageView(dev, view1, nullptr);
            vkDestroyImage(dev, img1, nullptr);
            if (cached_color1) {
                if (cached_color1->bytes)
                    persistent_color_target_bytes() -= cached_color1->bytes;
                if (imem1) vkFreeMemory(dev, imem1, nullptr);
                persistent_color_target_cache().erase(color_key1);
            } else if (imem1) {
                release_transient_render_memory(dev, imem1);
            }
            return out;
        }
        if (cached_color1) {
            cached_color1->image = img1;
            cached_color1->memory = imem1;
            cached_color1->view = view1;
        }
    }
    const bool load_cached_color1 = cached_color1 && cached_color1->valid &&
                                    color_target->load_existing1 && !effective_seed1;
    if (use_color1) {
        extra_images[1] = img1;
        extra_memories[1] = imem1;
        extra_views[1] = view1;
    }
    // Slots 2..7 retain their allocation across render groups when the caller names a persistent id,
    // exactly as slots 0 and 1 do. Falls back to a transient image whenever the cache is unavailable
    // or over its limits, which is the same fallback slot 1 takes -- a title that exceeds the budget
    // loses accumulation on those slots rather than failing to render.
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        const uint64_t slot_id = color_target ? color_target->persistent_id_slots[slot] : 0;
        const bool want_persistent = persistent_color_targets_enabled && slot_id;
        if (want_persistent) {
            extra_keys[slot] = {slot_id, W, H, color_formats[slot]};
            auto [found, inserted] = persistent_color_target_cache().try_emplace(extra_keys[slot]);
            (void)inserted;
            cached_extra[slot] = &found->second;
            cached_extra[slot]->last_use = color_target_generation;
            extra_images[slot] = cached_extra[slot]->image;
            extra_memories[slot] = cached_extra[slot]->memory;
            extra_views[slot] = cached_extra[slot]->view;
        }
        if (extra_images[slot]) continue;   // retained from an earlier group
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D; ci.format = color_formats[slot];
        ci.extent = {W, H, 1}; ci.mipLevels = 1; ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        const bool slot_seeded = color_target && color_target->seed_slots[slot];
        ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   (cached_extra[slot] ? (VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT) : 0u) |
                   (slot_seeded ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
        if (create_color_target_image(dev, ci, RenderColorTargetCreateSite::SlotExtra,
                                      &extra_images[slot]) != VK_SUCCESS) {
            if (cached_extra[slot]) {
                persistent_color_target_cache().erase(extra_keys[slot]);
                cached_extra[slot] = nullptr;
            }
            return out;
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(dev, extra_images[slot], &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = pick(requirements.memoryTypeBits, 0);
        if (cached_extra[slot]) {
            const VkDeviceSize limit = persistent_color_target_limit();
            while (!avoid_cache_eviction &&
                   (persistent_color_target_cache().size() >
                            persistent_color_target_count_limit() ||
                    requirements.size > limit ||
                    persistent_color_target_bytes() > limit - requirements.size) &&
                   evict_persistent_color_target(ctx, color_target_generation)) {}
            if (requirements.size <= limit &&
                persistent_color_target_cache().size() <=
                    persistent_color_target_count_ceiling(avoid_cache_eviction) &&
                persistent_color_target_bytes() <= limit - requirements.size &&
                vkAllocateMemory(dev, &allocation, nullptr, &extra_memories[slot]) == VK_SUCCESS) {
                cached_extra[slot]->bytes = requirements.size;
                persistent_color_target_bytes() += requirements.size;
            } else {
                // Over budget: drop back to a transient image for this slot, exactly as slot 1 does.
                vkDestroyImage(dev, extra_images[slot], nullptr);
                extra_images[slot] = VK_NULL_HANDLE;
                persistent_color_target_cache().erase(extra_keys[slot]);
                cached_extra[slot] = nullptr;
                ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           (slot_seeded ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
                if (create_color_target_image(dev, ci,
                                              RenderColorTargetCreateSite::SlotExtraFallback,
                                              &extra_images[slot]) != VK_SUCCESS)
                    return out;
                vkGetImageMemoryRequirements(dev, extra_images[slot], &requirements);
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = pick(requirements.memoryTypeBits, 0);
            }
        }
        if (!extra_memories[slot])
            extra_memories[slot] = allocate_transient_render_memory(
                dev, allocation.allocationSize, allocation.memoryTypeIndex);
        VkImageViewCreateInfo view_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_ci.image = extra_images[slot]; view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format = color_formats[slot];
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const bool ready = extra_memories[slot] &&
            vkBindImageMemory(dev, extra_images[slot], extra_memories[slot], 0) == VK_SUCCESS &&
            vkCreateImageView(dev, &view_ci, nullptr, &extra_views[slot]) == VK_SUCCESS &&
            extra_views[slot];
        if (!ready) {
            if (extra_views[slot]) vkDestroyImageView(dev, extra_views[slot], nullptr);
            vkDestroyImage(dev, extra_images[slot], nullptr);
            if (cached_extra[slot]) {
                if (cached_extra[slot]->bytes)
                    persistent_color_target_bytes() -= cached_extra[slot]->bytes;
                if (extra_memories[slot]) vkFreeMemory(dev, extra_memories[slot], nullptr);
                persistent_color_target_cache().erase(extra_keys[slot]);
            } else if (extra_memories[slot]) {
                release_transient_render_memory(dev, extra_memories[slot]);
            }
            extra_images[slot] = VK_NULL_HANDLE;
            extra_memories[slot] = VK_NULL_HANDLE;
            extra_views[slot] = VK_NULL_HANDLE;
            return out;
        }
        if (cached_extra[slot]) {
            cached_extra[slot]->image = extra_images[slot];
            cached_extra[slot]->memory = extra_memories[slot];
            cached_extra[slot]->view = extra_views[slot];
        }
    }
    for (uint32_t slot = 2; slot < color_count; ++slot)
        load_extra[slot] = cached_extra[slot] && cached_extra[slot]->valid &&
                           color_target && color_target->load_existing_slots[slot];
    if (color_target && PROSPER_ENV_ON("PROSPER_BACKEND_LOAD_LOG")) {
        if (use_color1)
            fprintf(stderr,
                    "[backend-load] slot=1 id=0x%llx %ux%u fmt=%d cached=%d valid=%d "
                    "load_existing=%d seed=%d readback=%d -> load=%d\n",
                    (unsigned long long)color_target->persistent_id1, W, H, (int)FMT1,
                    cached_color1 != nullptr, cached_color1 ? (int)cached_color1->valid : -1,
                    (int)color_target->load_existing1, effective_seed1 != nullptr,
                    (int)color_target->readback1, (int)load_cached_color1);
        for (uint32_t slot = 2; slot < color_count; ++slot)
            fprintf(stderr,
                    "[backend-load] slot=%u id=0x%llx %ux%u fmt=%d cached=%d valid=%d "
                    "load_existing=%d seed=%d readback=%d -> load=%d\n",
                    slot,
                    (unsigned long long)color_target->persistent_id_slots[slot], W, H,
                    (int)color_formats[slot], cached_extra[slot] != nullptr,
                    cached_extra[slot] ? (int)cached_extra[slot]->valid : -1,
                    (int)color_target->load_existing_slots[slot],
                    color_target->seed_slots[slot] != nullptr,
                    (int)color_target->readback_slots[slot], (int)load_extra[slot]);
    }

    if (use_ds && !dimg) {
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D; dci.format = DFMT; dci.extent = {W, H, 1};
        dci.mipLevels = 1; dci.arrayLayers = 1; dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL;
        dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;   // sampled depth bridge (#1275)
        // #3045: same class of defect as the sampled-texture upload path, and found while
        // auditing this function's other vkCreateImage sites for the same review. Unlike the
        // color-target sites above (which discard the VkResult but do check `!img`/`!img1`/
        // `!extra_images[slot]` before touching the handle further), this one had NO guard at
        // all -- a real failure fed a definitely-null `dimg` straight into
        // vkGetImageMemoryRequirements, the exact undefined behaviour #3045 reports. Match the
        // sibling return-out-on-failure convention instead.
        const VkResult ds_image_result = vkCreateImage(dev, &dci, nullptr, &dimg);
        if (ds_image_result != VK_SUCCESS || !dimg) {
            dimg = VK_NULL_HANDLE;
            static std::atomic<uint32_t> ds_create_failure_logs{0};
            if (ds_create_failure_logs.fetch_add(1, std::memory_order_relaxed) < 32)
                std::fprintf(stderr,
                    "[ds-create-failed] vkCreateImage result=%d extent=%ux%u fmt=%d\n",
                    (int)ds_image_result, W, H, (int)DFMT);
            if (cached_ds) persistent_ds_cache().erase(ds_key);
            return out;
        }
        VkMemoryRequirements dr; vkGetImageMemoryRequirements(dev, dimg, &dr);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dr.size; dai.memoryTypeIndex = pick(dr.memoryTypeBits, 0);
        if (cached_ds) vkAllocateMemory(dev, &dai, nullptr, &dmem);
        else dmem = allocate_transient_render_memory(dev, dai.allocationSize,
                                                      dai.memoryTypeIndex);
        vkBindImageMemory(dev, dimg, dmem, 0);
        VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dvci.image = dimg; dvci.viewType = VK_IMAGE_VIEW_TYPE_2D; dvci.format = DFMT;
        dvci.subresourceRange = {DASPECT, 0, 1, 0, 1};
        // #3210: the result used to be discarded and `dview` used unchecked. It is
        // value-initialized, so unlike the render pass below the failed value was at least
        // deterministic -- but a null view still went into fbviews[ds_attachment] and from there
        // into the framebuffer. Drop the pass the way the color-target sites do, and tear the
        // half-created target down rather than caching it: a cached entry with a null view breaks
        // every later LOAD of this identity, the same defect #1383 records for the color path.
        if (create_render_image_view_checked(dev, dvci, RenderVkObjectCreateSite::DepthStencilView,
                                             &dview) != VK_SUCCESS) {
            vkDestroyImage(dev, dimg, nullptr);
            dimg = VK_NULL_HANDLE;
            if (cached_ds) {
                if (dmem) vkFreeMemory(dev, dmem, nullptr);
                persistent_ds_cache().erase(ds_key);
            } else if (dmem) {
                release_transient_render_memory(dev, dmem);
            }
            dmem = VK_NULL_HANDLE;
            return out;
        }
        if (cached_ds) { cached_ds->image = dimg; cached_ds->memory = dmem; cached_ds->view = dview; }
    }

    std::array<VkAttachmentDescription, prosper::gpu::kColorTargetCount + 1> att{};
    att[0].format = FMT; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    // Seeded or persistent: the attachment already holds valid pixels before this pass, so LOAD them.
    att[0].loadOp = (seed_rgba || load_cached_color) ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                     : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = (seed_rgba || load_cached_color)
        ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = persistent_color ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    if (use_color1) {
        att[1].format = FMT1; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
        att[1].loadOp = (effective_seed1 || load_cached_color1)
            ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[1].initialLayout = (effective_seed1 || load_cached_color1)
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        att[1].finalLayout = persistent_color1 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                               : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        // LOAD a retained slot whose contents are valid, so a G-buffer built across several render
        // groups accumulates instead of each group clearing its predecessor's work.
        const bool load_slot = load_extra[slot] ||
                               (color_target && color_target->seed_slots[slot]);
        att[slot].format = color_formats[slot];
        att[slot].samples = VK_SAMPLE_COUNT_1_BIT;
        att[slot].loadOp = load_slot ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[slot].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[slot].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[slot].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[slot].initialLayout = load_slot ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                            : VK_IMAGE_LAYOUT_UNDEFINED;
        att[slot].finalLayout = cached_extra[slot] ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                   : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    att[ds_attachment].format = DFMT; att[ds_attachment].samples = VK_SAMPLE_COUNT_1_BIT;
    // A guest-identified DS surface survives calls. New attachments get a defined initial value;
    // existing ones LOAD. Explicit DB_RENDER_CONTROL clears execute at their draw below, preserving
    // command order instead of being promoted to an unconditional pass-start clear (#518).
    // Depth and stencil have independent guest lifetimes even when Vulkan stores them in one D32S8
    // image. Using stencil must not make an untouched depth plane valid: Unity can stencil-prime a
    // surface under an ALWAYS, read-only depth test and only later use reverse-Z depth (#540).
    att[ds_attachment].loadOp = depth_was_valid ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[ds_attachment].storeOp = persistent_ds ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[ds_attachment].stencilLoadOp = format_has_stencil
        ? (stencil_was_valid ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR)
        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[ds_attachment].stencilStoreOp = (persistent_ds && format_has_stencil)
        ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[ds_attachment].initialLayout = self_sampled_depth
        ? self_depth_layout
        : (ds_layout_initialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                 : VK_IMAGE_LAYOUT_UNDEFINED);
    att[ds_attachment].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    std::array<VkAttachmentReference, prosper::gpu::kColorTargetCount> ar{};
    for (uint32_t slot = 0; slot < color_count; ++slot)
        ar[slot] = {slot, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dar{
        ds_attachment,
        self_sampled_depth ? self_depth_layout
                           : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = color_count; sub.pColorAttachments = ar.data();
    if (use_ds) sub.pDepthStencilAttachment = &dar;
    // #2945 -- EXPLICIT external subpass dependencies. Without these the render pass gets Vulkan's
    // DEFAULT external dependencies, and the outgoing default is
    //   srcStageMask=ALL_COMMANDS, srcAccessMask=<all writes>,
    //   dstStageMask=BOTTOM_OF_PIPE, dstAccessMask=0
    // i.e. it makes this pass's writes AVAILABLE and makes them visible to NOTHING. Every
    // attachment above whose finalLayout is TRANSFER_SRC_OPTIMAL is then read by
    // vkCmdCopyImageToBuffer a few commands later with no dependency at all -- and the final layout
    // transition itself is a write of the whole image, which on RADV can be a DCC decompress.
    // Synchronization validation names it exactly: `SYNC-HAZARD-READ-AFTER-WRITE ...
    // vkCmdCopyImageToBuffer reads VkImage <...>, which was previously written during an image
    // layout transition initiated by vkCmdEndRenderPass` -- 10 of them on one offline replay.
    //
    // The barrier the readback path DOES emit (`transition_color_to_readback`) is guarded on
    // `persistent`, so it covers only the persistent-target route; a transient target's copy has
    // never been ordered against the pass that filled it. Declaring the dependency here rather than
    // beside the copy covers every consumer of the pass's output in one place.
    //
    // THESE MASKS ARE DELIBERATELY ALL_COMMANDS / MEMORY_READ|MEMORY_WRITE, and narrowing them is a
    // REGRESSION, not an optimisation. Declaring an explicit external dependency REPLACES the
    // implicit one, and the implicit INCOMING default is
    //   srcStageMask=TOP_OF_PIPE, dstStageMask=ALL_COMMANDS, srcAccessMask=0,
    //   dstAccessMask=<all reads and writes>
    // -- so a pass that samples a texture a transfer just uploaded, or reads a buffer a compute
    // dispatch just wrote, is relying on that blanket visibility. The first version of this change
    // set dstAccessMask to the ATTACHMENT accesses only, which silently removes visibility for every
    // SHADER_READ in the pass. Both dependencies must stay a SUPERSET of the implicit defaults: they
    // may only add ordering, never remove it.
    //
    // That requirement is the Vulkan contract, not an empirical result. Three rung-6 snapshot guards
    // did fail on the narrow build, and this comment used to cite them -- but they fail identically
    // on plain master (#2950), so that run had no control and proved nothing about these masks.
    // Do not re-derive the rule from a guard run on this machine; derive it from the defaults above.
    //
    // PROSPER_NO_RENDERPASS_EXTERNAL_DEPS=1 restores the defaulted behaviour, so the A/B for this
    // change is single-variable on ONE binary. Same reason PROSPER_NO_INDEX_ARENA and the
    // PROSPER_NO_BACKEND_* family exist. It is a bisection lever, not a tunable: leaving it set
    // reinstates the race.
    static const bool no_renderpass_external_deps =
        getenv("PROSPER_NO_RENDERPASS_EXTERNAL_DEPS") != nullptr;
    constexpr VkAccessFlags kAllAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    // ALL_COMMANDS on the EXTERNAL side, ALL_GRAPHICS on the SUBPASS side, and the asymmetry is
    // required rather than stylistic. VUID-VkRenderPassCreateInfo-pDependencies-00837 and -00838:
    // when a dependency's src/dstSubpass is NOT VK_SUBPASS_EXTERNAL, that side's stage mask may
    // contain only stages the subpass's bind point supports, and ALL_COMMANDS includes compute and
    // host. The first version of this change used ALL_COMMANDS on both sides of both dependencies
    // and was rejected 923 times across 15 tests by CI's validation scan -- which is the only gate
    // in the tree that would have caught it: syncval on the replay path was clean, all 303 local
    // ctest cases passed, and the guards were already failing for #2950's reasons.
    //
    // Nothing is lost by the narrowing. Subpass 0 is a graphics subpass, so ALL_GRAPHICS covers
    // every stage that can execute inside it; the dependency remains a superset of the implicit
    // defaults in the only sense that matters, which is that nothing the pass actually does escapes
    // the ordering.
    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;   // EXTERNAL side
    deps[0].dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;   // subpass side
    deps[0].srcAccessMask = kAllAccess;
    deps[0].dstAccessMask = kAllAccess;
    deps[0].dependencyFlags = 0;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;   // subpass side
    deps[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;   // EXTERNAL side
    deps[1].srcAccessMask = kAllAccess;
    deps[1].dstAccessMask = kAllAccess;
    deps[1].dependencyFlags = 0;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = color_count + (use_ds ? 1u : 0u); rpci.pAttachments = att.data();
    rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    if (!no_renderpass_external_deps) {
        rpci.dependencyCount = static_cast<uint32_t>(deps.size());
        rpci.pDependencies = deps.data();
    }
    // #3210: both of these were declared WITHOUT an initializer and used with no guard at all --
    // `rp` into fbci.renderPass, every VkGraphicsPipelineCreateInfo::renderPass, the render-pass
    // begin info and vkDestroyRenderPass; `fb` straight into vkCmdBeginRenderPass. Vulkan writes an
    // output handle only on success, so a failed create left an arbitrary stack word in them, and
    // handing a non-handle to vkDestroyRenderPass is undefined. The helpers null the output before
    // the call, so the indeterminate read is gone whatever the driver does; here we drop the pass,
    // the same failure path the color-target creates above take.
    VkRenderPass rp = VK_NULL_HANDLE;
    if (create_render_pass_checked(dev, rpci, &rp) != VK_SUCCESS) return out;
    std::array<VkImageView, prosper::gpu::kColorTargetCount + 1> fbviews{};
    fbviews[0] = view;
    for (uint32_t slot = 1; slot < color_count; ++slot) fbviews[slot] = extra_views[slot];
    if (use_ds) fbviews[ds_attachment] = dview;
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = color_count + (use_ds ? 1u : 0u);
    fbci.pAttachments = fbviews.data(); fbci.width = W; fbci.height = H; fbci.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (create_framebuffer_checked(dev, fbci, &fb) != VK_SUCCESS) {
        // `rp` is owned entirely by this scope and nothing references it yet, so unlike the wider
        // transient state (which the deferred cleanup below would have taken) it can and must be
        // destroyed here rather than leaked.
        vkDestroyRenderPass(dev, rp, nullptr);
        return out;
    }

    auto mkmod = [&](const std::vector<uint32_t>& c) -> VkShaderModule {
        VkShaderModuleCreateInfo s{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        s.codeSize = c.size() * 4; s.pCode = c.data(); VkShaderModule m = VK_NULL_HANDLE;
        // #3210 case 3, the mild one: the result was discarded, but `m` is initialized above and
        // the caller already skips the draw on `!v.vs || !v.fs`. Only the report is new -- the
        // control flow is deliberately unchanged.
        create_render_shader_module_checked(dev, s, &m);
        return m; };
    // Per-draw Vulkan objects stay alive until the call or explicit submission batch completes.
    const auto timing_target_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    struct DV {
        VkShaderModule vs = VK_NULL_HANDLE, gs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> dsets;
        VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE;
        VkBuffer ibuf = VK_NULL_HANDLE; VkDeviceMemory ibmem = VK_NULL_HANDLE;   // index buffer (indexed draws)
        // Non-zero when the index data lives in a shared arena slice rather than a dedicated
        // allocation. `iarena` decides ownership at teardown: an arena slice is owned by the arena
        // and must NOT be destroyed per draw, which is the one way this optimisation corrupts
        // rather than merely slows (#2253). Mirrors SharedBufferUpload::arena on the storage path.
        VkDeviceSize ioffset = 0;
        bool iarena = false;
        // PROSPER_INDEX_ECHO diagnostic only: the host-visible address the index bytes were written
        // to, so the record loop can read back exactly what the GPU will fetch from (v.ibuf,
        // v.ioffset). Never read outside the diagnostic.
        const void* imapped = nullptr;
        VkViewport viewport{};
        VkRect2D scissor{};
        float line_width = 1.0f;
        float depth_bias_constant = 0.0f;
        float depth_bias_clamp = 0.0f;
        float depth_bias_slope = 0.0f;
        VkStencilOpState stencil_front{};
        VkStencilOpState stencil_back{};
        uint32_t n_sets = 1, vcount = 3, icount = 0, instance_count = 1;
        int32_t vertex_offset = 0;
        bool use_desc = false, ok = false, pipeline_cached = false;
    };
    struct TextureUploadKey {
        const uint8_t* pixels = nullptr;
        uint64_t render_target_id = 0;
        void* borrowed_compute_image = nullptr;
        uint32_t width = 0, height = 0, depth = 1;
        uint32_t img_dim = 1;
        uint32_t sample_count = 1;
        uint32_t mip_levels = 1;   // effective uploaded chain length (#1272)
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool storage_image = false;
        std::array<uint32_t, 4> uniform_color_bits{};
        bool operator==(const TextureUploadKey& other) const {
            return pixels == other.pixels && render_target_id == other.render_target_id &&
                   borrowed_compute_image == other.borrowed_compute_image &&
                   width == other.width && height == other.height && depth == other.depth &&
                   img_dim == other.img_dim && sample_count == other.sample_count &&
                   mip_levels == other.mip_levels &&
                   format == other.format && storage_image == other.storage_image &&
                   uniform_color_bits == other.uniform_color_bits;
        }
    };
    struct TextureUploadKeyHash {
        size_t operator()(const TextureUploadKey& key) const {
            size_t h = std::hash<const uint8_t*>{}(key.pixels);
            h ^= std::hash<uint64_t>{}(key.render_target_id) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= std::hash<void*>{}(key.borrowed_compute_image) +
                 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.width) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.height) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.depth) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.img_dim) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.sample_count) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.mip_levels) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.format) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.storage_image) + 0x9e3779b9u + (h << 6) + (h >> 2);
            for (uint32_t bits : key.uniform_color_bits)
                h ^= static_cast<size_t>(bits) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct SharedTextureUpload {
        TextureUploadKey key;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        uint64_t persistent_id = 0;
        uint64_t persistent_version = 0;
        VkDeviceSize image_bytes = 0;
        bool persistent_hit = false;
        bool persistent_refresh = false;
        bool uniform_clear = false;
        VkClearColorValue uniform_color{};
        bool borrowed_target = false;
        // Color-attachment feedback cannot borrow the attachment image itself. Snapshot its
        // pre-pass contents into this upload's distinct sampled image with a queue-ordered GPU copy.
        bool feedback_snapshot = false;
        VkImage feedback_source = VK_NULL_HANDLE;
        VkImageLayout feedback_source_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Independent one-level renderer targets copied into this upload's mip levels. Sources
        // retain their persistent-cache layout after the copy; a zero image is generated from the
        // preceding destination level.
        bool assembled_target_mips = false;
        std::array<VkImage, 16> target_mip_images{};
        std::array<VkImageLayout, 16> target_mip_layouts{};
        bool borrowed_compute = false;
        bool stacked_compute = false;
        VkImage stacked_compute_source = VK_NULL_HANDLE;
        uint32_t stacked_compute_layers = 0;
        VkImageLayout borrowed_compute_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::shared_ptr<void> borrowed_compute_lease;
        // Sampled depth bridge (#1275): image borrowed from the persistent DS cache. The view uses
        // the DEPTH aspect of ds_format, and the call transitions the image DS-attachment ->
        // shader-read around its passes.
        bool borrowed_ds = false;
        bool borrowed_ds_stencil = false;   // sample the stencil plane rather than the depth plane
        bool borrowed_ds_feedback = false; // same image is this pass's read-only depth attachment
        VkFormat ds_format = VK_FORMAT_UNDEFINED;
        bool direct_memory = false;
        std::vector<std::function<void(const uint8_t*, size_t)>> storage_writebacks;
    };
    struct PersistentTextureKey {
        uint64_t id = 0;
        uint32_t width = 0, height = 0, depth = 1, img_dim = 1, sample_count = 1;
        uint32_t mip_levels = 1;   // a cached image must match the requested chain length (#1272)
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool operator==(const PersistentTextureKey&) const = default;
    };
    struct PersistentTextureKeyHash {
        size_t operator()(const PersistentTextureKey& key) const {
            size_t h = std::hash<uint64_t>{}(key.id);
            auto mix = [&](uint32_t value) {
                h ^= static_cast<size_t>(value) + 0x9e3779b9u + (h << 6) + (h >> 2);
            };
            mix(key.width); mix(key.height); mix(key.depth); mix(key.img_dim);
            mix(key.sample_count);
            mix(key.mip_levels);
            mix(static_cast<uint32_t>(key.format));
            return h;
        }
    };
    struct TextureBindingKey {
        std::array<uint64_t, 22> words{};
        bool operator==(const TextureBindingKey&) const = default;
    };
    struct TextureBindingKeyHash {
        size_t operator()(const TextureBindingKey& key) const {
            uint64_t hash = 1469598103934665603ull;
            for (uint64_t word : key.words) {
                hash ^= word;
                hash *= 1099511628211ull;
            }
            return static_cast<size_t>(hash);
        }
    };
    struct PersistentTextureBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        uint64_t last_use = 0;
    };
    struct PersistentTextureImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;
        uint64_t last_use = 0;
        uint64_t content_version = 0;
        bool content_valid = true;
        std::unordered_map<TextureBindingKey, PersistentTextureBinding,
                           TextureBindingKeyHash> bindings;
    };
    static std::unordered_map<PersistentTextureKey, PersistentTextureImage,
                              PersistentTextureKeyHash> persistent_texture_images;
    static VkDeviceSize persistent_texture_bytes = 0;
    // Exact sum of each retained image's binding count. This domain is protected by the
    // BackendPersistentResourceGuard above; update on successful insertion/erasure, not on
    // content invalidation (which leaves the bindings resident until the image is retired).
    // Recounting every cached image after every pass made statistics an O(cache size) hot loop.
    static size_t persistent_texture_binding_entries = 0;
    static uint64_t persistent_texture_generation = 0;
    constexpr size_t persistent_texture_max_entries = 1024;
    const uint64_t texture_generation = ++persistent_texture_generation;
    const bool persistent_textures_enabled =
        getenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES") == nullptr;
    const VkDeviceSize persistent_texture_limit = persistent_texture_cache_limit();
    const bool share_backend_resources =
        getenv("PROSPER_NO_BACKEND_RESOURCE_SHARE") == nullptr;
    const bool reuse_host_buffers = render_host_buffer_pool_enabled();
    const bool buffer_verify_enabled = getenv("PROSPER_BUFVERIFY") != nullptr;
    struct SharedBufferKey {
        const uint32_t* words = nullptr;
        size_t count = 0;
        uint64_t identity = 0;
        uint64_t hash = 0;
        uint64_t unique_tag = 0;
        bool operator==(const SharedBufferKey& other) const {
            return identity == other.identity && hash == other.hash &&
                   unique_tag == other.unique_tag &&
                   count == other.count &&
                   (!count || std::memcmp(words, other.words, count * sizeof(uint32_t)) == 0);
        }
    };
    struct SharedBufferKeyHash {
        size_t operator()(const SharedBufferKey& key) const {
            return static_cast<size_t>(key.hash ^ key.identity ^
                (key.unique_tag * 0x9e3779b97f4a7c15ull));
        }
    };
    struct SharedBufferUpload {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize offset = 0;
        VkDeviceSize range = 0;
        VkDeviceSize bytes = 0;
        VkDeviceSize allocation_bytes = 0;
        bool pooled = false;
        bool arena = false;
    };
    struct SharedBufferArena {
        RenderHostBuffer buffer;
        VkDeviceSize used = 0;
    };
    struct SharedTextureBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        bool persistent = false;
    };
    struct SharedPipelineLayout {
        VkPipelineLayout handle = VK_NULL_HANDLE;
        bool persistent = false;
    };
    std::vector<SharedBufferUpload> shared_buffers;
    // PROSPER_BUFVERIFY=1 — re-read every uploaded storage buffer from its MAPPED, device-visible
    // allocation just before the pass is submitted, and compare it byte for byte against the guest
    // words it was built from. PROSPER_BUFLOG reports only the SOURCE words, so it cannot see the
    // destination at all; this closes that half.
    //
    // Scope, stated narrowly on purpose. The destination `range` is assigned the same byte count the
    // memcpy used, so a SHORT upload is structurally inexpressible here and a zero from this check
    // says nothing about truncation — that is `[buffer-truncated]`'s and BUFLOG's question. What it
    // does detect is the destination being clobbered after the memcpy and before submission: an
    // overlapping arena slice, a stray write, a pooled buffer handed out twice. End-of-pass placement
    // is what makes that case visible; verifying at upload time would miss all of it.
    struct BufferVerifyRecord {
        const uint32_t* words = nullptr;
        size_t word_count = 0;
        uint64_t identity = 0;
        uint32_t set = 0;
        uint32_t binding = 0;
        size_t upload_index = 0;
    };
    std::vector<BufferVerifyRecord> buffer_verify_records;
    std::vector<SharedBufferArena> shared_buffer_arenas;
    std::unordered_map<SharedBufferKey, size_t, SharedBufferKeyHash> shared_buffer_indices;
    // Per-call repeat-reference memo (#1268): draws in one pass batch overwhelmingly re-reference
    // the same guest ranges (same VB/UB across hundreds of draws), and the SharedBufferKey lookup
    // pays a FULL content hash per reference because the hash is part of the key. Within one call
    // the referenced guest memory is stable modulo cross-thread racing writes — which are equally
    // nondeterministic on real hardware (the GPU samples the buffer once per draw at execution),
    // so resolving a repeated (words, count, identity) reference to the first upload is within the
    // same latitude the hardware has. First reference still hashes + memcmps as before.
    struct BufferRefMemoKey {
        const uint32_t* words = nullptr;
        size_t count = 0;
        uint64_t identity = 0;
        bool operator==(const BufferRefMemoKey& other) const {
            return words == other.words && count == other.count && identity == other.identity;
        }
    };
    struct BufferRefMemoKeyHash {
        size_t operator()(const BufferRefMemoKey& key) const {
            return static_cast<size_t>((reinterpret_cast<uintptr_t>(key.words) >> 2) ^
                (key.count * 0x9e3779b97f4a7c15ull) ^ key.identity);
        }
    };
    std::unordered_map<BufferRefMemoKey, size_t, BufferRefMemoKeyHash> buffer_ref_memo;
    std::vector<SharedTextureBinding> shared_texture_bindings;
    std::unordered_map<TextureBindingKey, size_t, TextureBindingKeyHash>
        shared_texture_binding_indices;
    std::unordered_map<std::vector<uint64_t>, VkDescriptorSetLayout, BackendWordVectorHash>
        shared_descriptor_set_layouts;
    std::unordered_map<std::vector<uint64_t>, SharedPipelineLayout, BackendWordVectorHash>
        shared_pipeline_layouts;
    uint64_t resource_unique_tag = 0;
    const uint32_t zero_buffer_word = 0;
    const VkDeviceSize storage_buffer_alignment = ctx.storage_buffer_alignment;
    const bool use_buffer_arena = reuse_host_buffers &&
        !PROSPER_ENV_ON("PROSPER_NO_BACKEND_BUFFER_ARENA");
    auto align_storage_offset = [storage_buffer_alignment](VkDeviceSize value) {
        const VkDeviceSize remainder = value % storage_buffer_alignment;
        if (!remainder) return value;
        const VkDeviceSize padding = storage_buffer_alignment - remainder;
        return value <= UINT64_MAX - padding ? value + padding : VkDeviceSize{UINT64_MAX};
    };
    auto acquire_buffer_arena_slice = [&](VkDeviceSize bytes, SharedBufferUpload& upload) {
        for (SharedBufferArena& arena : shared_buffer_arenas) {
            const VkDeviceSize offset = align_storage_offset(arena.used);
            if (offset != UINT64_MAX && offset <= arena.buffer.bytes &&
                bytes <= arena.buffer.bytes - offset) {
                upload.buffer = arena.buffer.buffer;
                upload.mapped = arena.buffer.mapped;
                upload.offset = offset;
                upload.arena = true;
                arena.used = offset + bytes;
                return true;
            }
        }
        VkDeviceSize request = std::max(render_host_buffer_arena_size(), bytes);
        if (request < bytes) request = bytes;
        RenderHostBuffer buffer = acquire_render_host_buffer(ctx, request);
        if (!buffer.buffer || !buffer.memory || !buffer.mapped) {
            destroy_render_host_buffer(ctx.dev, buffer);
            return false;
        }
        shared_buffer_arenas.push_back({buffer, bytes});
        upload.buffer = buffer.buffer;
        upload.mapped = buffer.mapped;
        upload.offset = 0;
        upload.arena = true;
        return true;
    };
    auto handle_bits = [](auto handle) {
        uint64_t bits = 0;
        static_assert(sizeof(handle) <= sizeof(bits));
        std::memcpy(&bits, &handle, sizeof(handle));
        return bits;
    };
    auto float_bits = [](float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    std::vector<DV> dv(draws.size());
    // Most draws use the frontend's immutable resource vector verbatim. Copying every FrameResource
    // here copied owned buffer payloads, callbacks and shared owners for thousands of draws before
    // the backend could even consult its upload caches. Only a shader that needs prosper's synthetic
    // GDS binding requires an augmented vector; all ordinary draws borrow the caller-owned vector for
    // this synchronous backend call.
    std::vector<std::vector<FrameResource>> augmented_resources(draws.size());
    std::vector<const std::vector<FrameResource>*> effective_resources(draws.size());
    for (size_t i = 0; i < draws.size(); ++i) {
        if (fragment_uses_internal_gds_memoized(draws[i].fs_identity, draws[i].fs_words())) {
            augmented_resources[i] = draws[i].R;
            FrameResource gds;
            gds.set = 1;
            gds.binding = 0;
            gds.is_internal_gds = true;
            augmented_resources[i].push_back(std::move(gds));
            effective_resources[i] = &augmented_resources[i];
        } else {
            effective_resources[i] = &draws[i].R;
        }
    }
    std::vector<SharedTextureUpload> texture_uploads;
    std::unordered_map<TextureUploadKey, size_t, TextureUploadKeyHash> texture_upload_indices;
    const bool share_texture_uploads = getenv("PROSPER_NO_BACKEND_TEXTURE_SHARE") == nullptr;
    size_t texture_references = 0;
    size_t persistent_texture_hits = 0;
    size_t persistent_texture_misses = 0;
    double setup_shader_ms = 0.0;
    double setup_fixed_ms = 0.0;
    double res_fixed_index_upload_ms = 0.0;
    double res_fixed_blend_ms = 0.0;
    double res_fixed_depth_stencil_ms = 0.0;
    double res_fixed_viewport_ms = 0.0;
    double res_fixed_stages_ms = 0.0;
    double res_fixed_prologue_ms = 0.0;
    double res_prologue_subgroup_scan_ms = 0.0;
    double setup_resources_ms = 0.0;
    double setup_pipeline_ms = 0.0;
    // #1284 sub-attribution of setup_resources_ms; see BackendRenderTimingStats for what each covers.
    double res_texture_ms = 0.0;
    double res_texture_upload_ms = 0.0;
    double res_texture_bind_ms = 0.0;
    double res_buffer_ms = 0.0;
    // Nested INSIDE res_buffer_ms. See BackendRenderTimingStats for why these five exist rather than
    // the original two, and why the remainder is reported signed.
    double res_buffer_acquire_ms = 0.0;
    double res_buffer_copy_ms = 0.0;
    double res_buffer_create_ms = 0.0;
    double res_buffer_index_find_ms = 0.0;
    double res_buffer_index_insert_ms = 0.0;
    double res_buffer_hash_ms = 0.0;
    double res_descriptor_ms = 0.0;
    auto setup_elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    // Scope-guard timer, so a bucket stays correct across the `continue`/`break` exits the resource
    // loop already uses. Reads the clock only when timing is enabled; the whole sub-attribution is
    // inert (two predictable branches per resource) on a default run.
    struct ResourcePhaseTimer {
        bool enabled;
        double* sink;
        TimingClock::time_point begin;
        ResourcePhaseTimer(bool en, double* s)
            : enabled(en), sink(s),
              begin(en ? TimingClock::now() : TimingClock::time_point{}) {}
        ResourcePhaseTimer(const ResourcePhaseTimer&) = delete;
        ResourcePhaseTimer& operator=(const ResourcePhaseTimer&) = delete;
        ~ResourcePhaseTimer() {
            if (enabled)
                *sink += std::chrono::duration<double, std::milli>(TimingClock::now() - begin)
                             .count();
        }
    };
    BackendPipelineCacheStats& pipeline_stats = backend_pipeline_cache_stats_storage();
    pipeline_stats = {};
    auto& pipeline_cache = persistent_pipeline_cache();
    const uint64_t pipeline_generation = ++persistent_pipeline_generation();
    const bool pipeline_cache_enabled = persistent_pipeline_cache_enabled();
    const size_t pipeline_cache_limit = persistent_pipeline_cache_limit();
    const bool pipeline_layout_cache_enabled = share_backend_resources &&
        persistent_pipeline_layout_cache_enabled();
    const size_t pipeline_layout_cache_limit = persistent_pipeline_layout_cache_limit();
    const uint64_t pipeline_layout_generation = ++persistent_pipeline_layout_generation();
    auto& persistent_pipeline_layouts = persistent_backend_pipeline_layout_cache();
    auto evict_persistent_pipeline_layout = [&]() {
        auto victim = persistent_pipeline_layouts.end();
        for (auto it = persistent_pipeline_layouts.begin();
             it != persistent_pipeline_layouts.end(); ++it) {
            if (it->second.last_use == pipeline_layout_generation) continue;
            if (victim == persistent_pipeline_layouts.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == persistent_pipeline_layouts.end()) return false;
        vkDestroyPipelineLayout(dev, victim->second.handle, nullptr);
        persistent_pipeline_layouts.erase(victim);
        ++resource_reuse_stats.persistent_pipeline_layout_evictions;
        return true;
    };
    auto evict_pipeline = [&]() {
        auto victim = pipeline_cache.end();
        for (auto it = pipeline_cache.begin(); it != pipeline_cache.end(); ++it) {
            if (it->second.last_use == pipeline_generation) continue;
            if (victim == pipeline_cache.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == pipeline_cache.end()) return false;
        vkDestroyPipeline(dev, victim->second.pipeline, nullptr);
        pipeline_cache.erase(victim);
        ++pipeline_stats.evictions;
        return true;
    };
    uint64_t descriptor_sets = 0;
    uint64_t storage_buffers = 0;
    uint64_t sampled_images = 0;
    uint64_t storage_images = 0;
    for (const auto* resource_ptr : effective_resources) {
        const auto& resources = *resource_ptr;
        if (resources.empty()) continue;
        uint32_t set_count = 1;
        for (const FrameResource& resource : resources) {
            set_count = std::max(set_count, resource.set + 1);
            if (!resource.is_texture()) {
                // N per array, not 1 per resource: the pool is sized from these counters, and an
                // N-entry array that reserved 1 fails inside vkAllocateDescriptorSets with
                // VK_ERROR_OUT_OF_POOL_MEMORY -- an error with no visible connection to arrays.
                storage_buffers += resource.written_descriptor_count();
            } else if (resource.is_storage_image) {
                ++storage_images;
            } else {
                ++sampled_images;
            }
        }
        descriptor_sets += set_count;
    }
    VkDescriptorPool shared_descriptor_pool = VK_NULL_HANDLE;
    if (descriptor_sets) {
        VkDescriptorPoolSize sizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             static_cast<uint32_t>(std::max<uint64_t>(storage_buffers, 1))},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             static_cast<uint32_t>(std::max<uint64_t>(sampled_images, 1))},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             static_cast<uint32_t>(std::max<uint64_t>(storage_images, 1))},
        };
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = static_cast<uint32_t>(descriptor_sets);
        pool_info.poolSizeCount = 3;
        pool_info.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(dev, &pool_info, nullptr, &shared_descriptor_pool) ==
            VK_SUCCESS) {
            resource_reuse_stats.descriptor_pools = 1;
        }
    }
    // Pass 1: create each draw's shader modules, descriptors (with texture staging upload), and pipeline.
    const bool backend_trace = PROSPER_ENV_ON("PROSPER_BACKEND_TRACE");
    // PROSPER_WAVE64_SKIP_CENSUS=1 -- which DRAWS the required-subgroup-size gate drops, at which
    // render target, and WHY (#2429, #2448). The existing "requires subgroup size 64" line sits inside
    // a dedupe guard keyed on shader identity while the `continue` that drops the draw is outside it,
    // so that message counts DISTINCT SHADERS and the skipped-DRAW count is measured nowhere. A census
    // on #2448 divided one by the other and reported "1.5% of draws"; the units did not match and the
    // figure was withdrawn.
    //
    // The gate is SEVEN disjuncts, not one, so naming it "wave64" over-attributes: two of them are not
    // width problems at all (a subgroup FEATURE shortfall, and internal-GDS use without fragment
    // stores/atomics). That distinction decides whether a wave32 lowering would recover the draw, which
    // is the judgement this number exists to inform, so the reason mask is recorded per skip rather
    // than summed away.
    //
    // Attribution by pass extent W x H is exact rather than heuristic: one shader serves many targets,
    // but each DRAW belongs to exactly one pass. `seen` counts EVERY draw, so the denominator is a real
    // total and not a copy of the numerator.
    //
    // Each report is a CUMULATIVE SNAPSHOT of the counters, not a delta, so only the LAST block in a
    // log is a total for the run. Summing the blocks double-counts -- an awk over every line of a
    // 15,000-draw run reports 30,100. Read the final block.
    //
    // Emitted at observation 100 and every 5000 after, never at exit. The FIRST report matters as much
    // as the cadence: with a threshold alone, a run smaller than the interval prints NOTHING, and that
    // silence is indistinguishable from the variable being unset, from a typo, and from this code never
    // running. That is the defect this instrument was rejected for in review, and the shape of #2456.
    static const bool wave64_census = getenv("PROSPER_WAVE64_SKIP_CENSUS") != nullptr;
    struct Wave64Census {
        std::mutex mx;
        std::unordered_map<uint64_t, uint64_t> seen;
        std::unordered_map<uint64_t, std::unordered_map<uint32_t, uint64_t>> skipped;
        uint64_t observations = 0;
        static uint64_t key(uint32_t w, uint32_t h) { return (uint64_t(w) << 32) | h; }
        void report_locked() {
            std::vector<uint64_t> keys;
            keys.reserve(seen.size());
            for (const auto& kv : seen) keys.push_back(kv.first);
            std::sort(keys.begin(), keys.end());
            for (uint64_t k : keys) {
                uint64_t dropped = 0;
                auto it = skipped.find(k);
                if (it != skipped.end())
                    for (const auto& rm : it->second) dropped += rm.second;
                std::fprintf(stderr, "[wave64-census] %ux%u draws=%llu skipped=%llu\n",
                             (uint32_t)(k >> 32), (uint32_t)(k & 0xffffffffu),
                             (unsigned long long)seen[k], (unsigned long long)dropped);
                if (it != skipped.end())
                    for (const auto& rm : it->second)
                        std::fprintf(stderr, "[wave64-census]   %ux%u reason=0x%02x n=%llu\n",
                                     (uint32_t)(k >> 32), (uint32_t)(k & 0xffffffffu),
                                     rm.first, (unsigned long long)rm.second);
            }
        }
        void note_draw(uint32_t w, uint32_t h) {
            std::lock_guard<std::mutex> lk(mx);
            ++seen[key(w, h)];
            ++observations;
            if (observations == 100 || (observations % 5000) == 0) report_locked();
        }
        void note_skip(uint32_t w, uint32_t h, uint32_t reason_mask) {
            std::lock_guard<std::mutex> lk(mx);
            ++skipped[key(w, h)][reason_mask];
        }
    };
    static Wave64Census wave64_stats;
    for (size_t di = 0; di < draws.size(); di++) {
        // Denominator: every draw this pass considers, recorded before any skip path can divert it.
        if (wave64_census) wave64_stats.note_draw(W, H);
        const auto setup_begin = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        const BackendDraw& bd = draws[di];
        const std::vector<uint32_t>& bd_vs = bd.vs_words();
        const std::vector<uint32_t>& bd_gs = bd.gs_words();
        const std::vector<uint32_t>& bd_fs = bd.fs_words();
        DV& v = dv[di];
        const auto subgroup_scan_begin =
            timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        // Both of these walk the ENTIRE SPIR-V module, and build_bds calls them once per draw on a
        // module that changes only when the shader does. Measured at 7.14 ms/submit -- 58% of the
        // pre-index span and ~4% of a Blue Prince gameplay frame (#2254, partitioned by #2252).
        //
        // Memoized on fs_identity, the same key and the same argument as #2259: DrawItem identities
        // come from the shader-recompile cache as `cache.next_identity++`
        // (src/gpu/execute/gpu_executor.cpp:1255 graphics, :1441 compute; sole assignment at :437), a
        // monotonic counter that is never reset, never decremented and untouched by eviction. So an
        // identity denotes one module for the life of the process -- an evicted shader recompiles to
        // a NEW identity, which misses here and refills, and the stale entry is simply never looked
        // up again. The counter starts at 1, so identity 0 is never minted and the fallthrough below
        // cannot collide with a real module.
        //
        // Both memoized values are pure functions of the module bytes alone -- neither reads device
        // state -- so a hit is exactly what the call would have returned. The device-dependent half
        // (available_fragment_subgroup_features, from ctx.subgroup_operations) is computed BELOW and
        // is deliberately not memoized: it is cheap, and folding it in would key device state on a
        // shader identity.
        struct SubgroupScanEntry { uint32_t size; uint32_t features; };
        static thread_local std::unordered_map<uint64_t, SubgroupScanEntry> subgroup_scan_memo;
        constexpr size_t kSubgroupScanMemoMaxEntries = 4096;
        uint32_t required_fragment_subgroup_size = 0;
        uint32_t required_fragment_subgroup_features = 0;
        // PROSPER_NO_SUBGROUP_SCAN_MEMO: opt out, so the A/B for this change is single-variable on
        // ONE binary rather than a comparison of two builds. Kept as a permanent bisection lever for
        // the same reason PROSPER_NO_INDEX_ARENA (#2258) and the PROSPER_NO_BACKEND_* family exist.
        static const bool no_subgroup_scan_memo = getenv("PROSPER_NO_SUBGROUP_SCAN_MEMO") != nullptr;
        bool subgroup_scan_memoized = false;
        if (bd.fs_identity && !no_subgroup_scan_memo) {
            const auto found = subgroup_scan_memo.find(bd.fs_identity);
            if (found != subgroup_scan_memo.end()) {
                required_fragment_subgroup_size = found->second.size;
                required_fragment_subgroup_features = found->second.features;
                subgroup_scan_memoized = true;
            }
        }
        if (!subgroup_scan_memoized) {
            required_fragment_subgroup_size =
                prosper::gpu::fragment_spirv_required_subgroup_size(bd_fs);
            required_fragment_subgroup_features =
                prosper::gpu::fragment_spirv_required_subgroup_features(bd_fs);
            if (bd.fs_identity && !no_subgroup_scan_memo) {
                // Cleared wholesale rather than aged: there is no LRU bookkeeping worth paying for
                // on a path this hot. A non-zero clear count means this memo has STOPPED WORKING --
                // the scans are being paid again in full, plus the churn -- not that a threshold was
                // brushed. 523 distinct shaders per submit leaves ample headroom against 4,096.
                if (subgroup_scan_memo.size() >= kSubgroupScanMemoMaxEntries)
                    subgroup_scan_memo.clear();
                subgroup_scan_memo.emplace(bd.fs_identity,
                                           SubgroupScanEntry{required_fragment_subgroup_size,
                                                             required_fragment_subgroup_features});
            }
        }
        if (timing_enabled)
            res_prologue_subgroup_scan_ms +=
                setup_elapsed_ms(subgroup_scan_begin, TimingClock::now());
        uint32_t available_fragment_subgroup_features = 0;
        if (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT)
            available_fragment_subgroup_features |= prosper::gpu::kFragmentSubgroupVote;
        if (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)
            available_fragment_subgroup_features |= prosper::gpu::kFragmentSubgroupArithmetic;
        if (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)
            available_fragment_subgroup_features |= prosper::gpu::kFragmentSubgroupShuffle;
        if (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BALLOT_BIT)
            available_fragment_subgroup_features |= prosper::gpu::kFragmentSubgroupBallot;
        const bool uses_internal_gds =
            fragment_uses_internal_gds_memoized(bd.fs_identity, bd_fs);
        bool fragment_subgroup_skip = required_fragment_subgroup_size &&
            (!ctx.subgroup_size_control ||
             required_fragment_subgroup_size < ctx.min_subgroup_size ||
             required_fragment_subgroup_size > ctx.max_subgroup_size ||
             !(ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
             !(ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
             !prosper::gpu::fragment_subgroup_features_supported(
                 required_fragment_subgroup_features,
                 available_fragment_subgroup_features) ||
             (uses_internal_gds && !ctx.fragment_stores_atomics));
        uint32_t subgroup_reasons = UINT32_MAX;
        if (fragment_subgroup_skip && bd.allow_native_fragment_vote_width &&
            (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
            prosper::gpu::fragment_subgroup_features_supported(
                required_fragment_subgroup_features,
                available_fragment_subgroup_features) &&
            !(uses_internal_gds && !ctx.fragment_stores_atomics)) {
            subgroup_reasons =
                prosper::gpu::fragment_spirv_required_subgroup_reasons(bd_fs);
            // This is deliberately title-scoped and narrower than PROSPER_WAVE64_APPROX: only a
            // WaveAny with no lane, ballot, shuffle or scalar-reduction qualifier is admitted.
            // The source branch's reviewed bank route preserves both world and characters under
            // this exact classifier; every unreviewed title remains on the strict master path.
            if (subgroup_reasons == prosper::gpu::kFragmentWaveReasonWaveAny) {
                const uint64_t shader_key = bd.fs_identity
                    ? bd.fs_identity : hash_buffer_words(bd_fs.data(), bd_fs.size());
                static std::mutex native_width_log_mutex;
                static std::unordered_set<uint64_t> native_width_logged;
                std::lock_guard<std::mutex> lock(native_width_log_mutex);
                if (native_width_logged.insert(shader_key).second)
                    std::fprintf(stderr,
                                 "[render] GTA V native-width fragment vote: subgroup %u -> %u "
                                 "(why=0x%x)\n",
                                 required_fragment_subgroup_size, ctx.max_subgroup_size,
                                 subgroup_reasons);
                // Omitting the required-size pNext below selects the device's native fragment
                // subgroup. On the current NVIDIA host that is 32 lanes.
                required_fragment_subgroup_size = 0;
                fragment_subgroup_skip = false;
            }
        }
        if (fragment_subgroup_skip) {
            const uint64_t shader_key = bd.fs_identity
                ? bd.fs_identity : hash_buffer_words(bd_fs.data(), bd_fs.size());
            static std::mutex log_mutex;
            static std::unordered_set<uint64_t> logged;
            std::lock_guard<std::mutex> lock(log_mutex);
            if (logged.insert(shader_key).second) {
                // WHY the width was required, decoded (#2147). `required-ops` cannot answer it:
                // those are Vote/Arithmetic/Shuffle/Ballot CAPABILITY bits, and the lane-id path declares
                // none of them — so a shader needing 64 for lane IDENTITY (which can never run at
                // 32, since SubgroupLocalInvocationId IS the guest lane id) printed identically to
                // one needing it only for a branch-guard vote (which may be width-agnostic). Those
                // have opposite prospects under a wave32 lowering, and without this field the
                // census that decides whether such a lowering is worth writing cannot be taken.
                //
                // Inside the dedupe guard, so it costs once per distinct shader, not per draw.
                const uint32_t why = subgroup_reasons != UINT32_MAX
                    ? subgroup_reasons
                    : prosper::gpu::fragment_spirv_required_subgroup_reasons(bd_fs);
                char why_text[160];
                if (why == UINT32_MAX) {
                    // Absent, not none. A module built before #2147 carries no marker, and printing
                    // that as 0 would assert nothing required the width — impossible for a module
                    // that requires one.
                    std::snprintf(why_text, sizeof why_text, "unrecorded (pre-#2147 module)");
                } else {
                    int n = std::snprintf(why_text, sizeof why_text, "0x%x", why);
                    const struct { uint32_t bit; const char* name; } kReasonNames[] = {
                        {prosper::gpu::kFragmentWaveReasonLaneId,     " lane-id"},
                        {prosper::gpu::kFragmentWaveReasonWaveAny,    " wave-any"},
                        {prosper::gpu::kFragmentWaveReasonDppRow16,   " dpp16"},
                        {prosper::gpu::kFragmentWaveReasonPermLane32, " permlane32"},
                        {prosper::gpu::kFragmentWaveReasonReadLane64, " readlane64"},
                        {prosper::gpu::kFragmentWaveReasonShuffle,    " shuffle"},
                        {prosper::gpu::kFragmentWaveReasonWaveBallot, " wave-ballot"},
                        {prosper::gpu::kFragmentWaveReasonScalarReduce, " scalar-reduce"},
                    };
                    for (const auto& entry : kReasonNames)
                        if ((why & entry.bit) && n > 0 && n < static_cast<int>(sizeof why_text))
                            n += std::snprintf(why_text + n, sizeof why_text - n, "%s", entry.name);
                    if (!why && n > 0 && n < static_cast<int>(sizeof why_text))
                        std::snprintf(why_text + n, sizeof why_text - n, " none");
                }
                std::fprintf(stderr,
                             "[render] skip draw=%zu fs=%016llx: fragment shader requires subgroup size %u "
                             "(device range %u..%u required-stages=0x%x subgroup-stages=0x%x "
                             "ops=0x%x required-ops=0x%x control=%d gds=%d fragment-atomics=%d "
                             "why=%s)\n",
                             static_cast<size_t>(bd.draw_index != UINT64_MAX
                                 ? bd.draw_index : di),
                             static_cast<unsigned long long>(shader_key),
                             required_fragment_subgroup_size, ctx.min_subgroup_size,
                             ctx.max_subgroup_size, ctx.required_subgroup_size_stages,
                             ctx.subgroup_stages, ctx.subgroup_operations,
                             required_fragment_subgroup_features,
                             static_cast<int>(ctx.subgroup_size_control),
                             static_cast<int>(uses_internal_gds),
                             static_cast<int>(ctx.fragment_stores_atomics), why_text);
            }
            // Numerator, counted OUTSIDE the dedupe guard above: once per dropped DRAW rather than once
            // per distinct shader. The draw was already counted in the denominator at the top of the
            // loop, so this records only the skip.
            //
            // The mask names WHICH of the seven disjuncts fired. Bits 0x40 (internal GDS without
            // fragment stores/atomics) and 0x20 (subgroup FEATURE shortfall) are NOT width problems, so
            // a wave32 lowering would not recover those draws; summing them into one "wave64" total
            // would answer a different question than the one asked.
            //
            // Lock nesting is deliberate and one-directional: `log_mutex` above is still held here (its
            // guard is scoped to the enclosing block), so the order is always log_mutex ->
            // wave64_stats.mx, and nothing takes them the other way. Preserve that if you add a lock
            // inside the census.
            if (wave64_census) {
                uint32_t reason_mask = 0;
                if (!ctx.subgroup_size_control)                                          reason_mask |= 0x01;
                if (required_fragment_subgroup_size < ctx.min_subgroup_size)             reason_mask |= 0x02;
                if (required_fragment_subgroup_size > ctx.max_subgroup_size)             reason_mask |= 0x04;
                if (!(ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT)) reason_mask |= 0x08;
                if (!(ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT))               reason_mask |= 0x10;
                if (!prosper::gpu::fragment_subgroup_features_supported(
                        required_fragment_subgroup_features,
                        available_fragment_subgroup_features))                           reason_mask |= 0x20;
                if (uses_internal_gds && !ctx.fragment_stores_atomics)                   reason_mask |= 0x40;
                wave64_stats.note_skip(W, H, reason_mask);
            }
            continue;
        }
        if (uses_internal_gds && !render_internal_gds_buffer().buffer) {
            static std::once_flag logged;
            std::call_once(logged, [] {
                std::fprintf(stderr,
                             "[render] skip draw: failed to allocate persistent GDS buffer\n");
            });
            continue;
        }
        if (backend_trace) {
            fprintf(stderr,
                    "[backend-trace] draw=%zu/%zu begin extent=%ux%u vs=%zu gs=%zu fs=%zu "
                    "vs_id=%016llx fs_id=%016llx resources=%zu\n",
                    di, draws.size(), W, H, bd_vs.size(), bd_gs.size(), bd_fs.size(),
                    (unsigned long long)bd.vs_identity,
                    (unsigned long long)bd.fs_identity, bd.R.size());
            fflush(stderr);
        }
        // Pipeline hits do not need temporary VkShaderModules. Defer module creation until after the
        // persistent lookup; the fixed/resource setup below is also required by the hit pipeline.
        const auto setup_shaders_ready = setup_begin;
        const prosper::gpu::ResolvedPipelineState* ps = bd.ps;
        v.vcount = bd.vcount;
        v.instance_count = bd.instance_count;
        v.vertex_offset = bd.vertex_offset;
        v.scissor = {{0, 0}, {W, H}};
        // PROSPER_IGNORE_EMPTY_SCISSOR (#1287 bring-up diagnostic): render draws whose resolved
        // scissor is empty with a full-target scissor instead, to A/B whether they carry the
        // missing post/shadow content. Off by default; not a fix.
        static const bool ignore_empty_scissor = getenv("PROSPER_IGNORE_EMPTY_SCISSOR") != nullptr;
        const bool scissor_empty = ps && ps->has_scissor &&
            (ps->scissor_right <= ps->scissor_left || ps->scissor_bottom <= ps->scissor_top);
        if (ps && ps->has_scissor && !(ignore_empty_scissor && scissor_empty)) {
            const int64_t left = std::clamp<int64_t>(ps->scissor_left, 0, W);
            const int64_t top = std::clamp<int64_t>(ps->scissor_top, 0, H);
            const int64_t right = std::clamp<int64_t>(ps->scissor_right, left, W);
            const int64_t bottom = std::clamp<int64_t>(ps->scissor_bottom, top, H);
            v.scissor.offset = {static_cast<int32_t>(left), static_cast<int32_t>(top)};
            v.scissor.extent = {static_cast<uint32_t>(right - left),
                                static_cast<uint32_t>(bottom - top)};
        }
        if (timing_enabled) res_fixed_prologue_ms += setup_elapsed_ms(setup_begin, TimingClock::now());
        // Indexed draw: upload the 32-bit index data to a host-visible VkIndexBuffer now; the record
        // pass binds it and issues vkCmdDrawIndexed instead of vkCmdDraw.
        if (!bd.indices.empty()) {
            // Per INDEXED DRAW: create a buffer, allocate memory, bind, map, copy, unmap. Six
            // Vulkan entry points and one allocation on every draw that carries indices, and until
            // this timer none of it was separable from fixed-function state translation.
            const ResourcePhaseTimer phase_index_upload(timing_enabled, &res_fixed_index_upload_ms);
            VkDeviceSize isz = (VkDeviceSize)bd.indices.size() * 4;
            // Prefer a slice of the shared host-buffer arena. The storage path next door has used
            // this since #1284 and #2246's partition measured what it is worth there: `create=0.00`
            // in every window of every run, because the arena absorbs the transient path entirely.
            // Index buffers were the one per-draw allocation left, at ~7.8 us/draw (#2253).
            //
            // Alignment: arena slices are aligned to storage_buffer_alignment, which is a
            // power of two >= 16 on every device prosper runs on and therefore a multiple of the 4
            // bytes vkCmdBindIndexBuffer requires for VK_INDEX_TYPE_UINT32. The `(offset % 4) == 0`
            // term below is a GUARD, not an assertion: a violation quietly takes the dedicated-buffer
            // fallback rather than failing, which is the safer behaviour because a bad bind offset is
            // undefined behaviour rather than a reliably reported error. It must NOT become an
            // assert -- when the guard rejects, acquire_buffer_arena_slice has already advanced
            // arena.used, so the slice is leaked for the remainder of the pass. Unreachable while
            // storage_buffer_alignment >= 4, i.e. on every real device.
            // PROSPER_NO_INDEX_ARENA: opt out of the index arena ONLY, leaving the storage
            // arena on. Without it the single-variable A/B for this change does not exist:
            // PROSPER_NO_BACKEND_BUFFER_ARENA disables both, so it measures "arena vs no arena"
            // rather than "index arena vs dedicated allocation" -- in the first attempt it moved
            // `resources` from 43.70 to 69.92 ms alongside the index term, which is a different
            // change. Kept as a permanent bisection lever for the same reason PROSPER_NO_BACKEND_*
            // exist.
            static const bool no_index_arena = getenv("PROSPER_NO_INDEX_ARENA") != nullptr;
            if (use_buffer_arena && !no_index_arena) {
                SharedBufferUpload islice;
                if (acquire_buffer_arena_slice(isz, islice) && islice.arena && islice.mapped &&
                    (islice.offset % 4) == 0) {
                    std::memcpy(static_cast<uint8_t*>(islice.mapped) + islice.offset,
                                bd.indices.data(), (size_t)isz);
                    v.ibuf = islice.buffer;
                    v.ioffset = islice.offset;
                    v.iarena = true;
                    v.imapped = static_cast<const uint8_t*>(islice.mapped) + islice.offset;
                    v.icount = (uint32_t)bd.indices.size();
                }
            }
            // Fallback: a dedicated buffer, exactly as before. Reached when the arena is disabled
            // (PROSPER_NO_BACKEND_BUFFER_ARENA, which is therefore still a working A/B lever) or
            // when a slice could not be obtained. Keeping it means an arena failure degrades to the
            // old cost rather than dropping the draw.
            if (!v.iarena) {
            VkBufferCreateInfo ibci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ibci.size = isz; ibci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            static const bool echo_index_usage = getenv("PROSPER_BUFFER_ECHO") != nullptr;
            if (echo_index_usage) ibci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // #2945 echo
            vkCreateBuffer(dev, &ibci, nullptr, &v.ibuf);
            VkMemoryRequirements imr; vkGetBufferMemoryRequirements(dev, v.ibuf, &imr);
            VkMemoryAllocateInfo imai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; imai.allocationSize = imr.size;
            imai.memoryTypeIndex = pick(imr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            v.ibmem = allocate_transient_render_memory(dev, imai.allocationSize,
                                                        imai.memoryTypeIndex);
            vkBindBufferMemory(dev, v.ibuf, v.ibmem, 0);
            void* ip = nullptr; vkMapMemory(dev, v.ibmem, 0, isz, 0, &ip);
            std::memcpy(ip, bd.indices.data(), (size_t)isz); vkUnmapMemory(dev, v.ibmem);
            v.icount = (uint32_t)bd.indices.size();
            }
        }
        const auto fixed_stages_begin = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        VkPipelineShaderStageCreateInfo st[3]{};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_fragment_subgroup{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = v.vs; st[0].pName = "main";
        const uint32_t fragment_stage_index = bd_gs.empty() ? 1u : 2u;
        if (!bd_gs.empty()) {
            st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            st[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT; st[1].module = v.gs; st[1].pName = "main";
        }
        st[fragment_stage_index] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        st[fragment_stage_index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[fragment_stage_index].module = v.fs;
        st[fragment_stage_index].pName = "main";
        if (required_fragment_subgroup_size) {
            required_fragment_subgroup.requiredSubgroupSize = required_fragment_subgroup_size;
            st[fragment_stage_index].pNext = &required_fragment_subgroup;
        }
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = ps ? (VkPrimitiveTopology)ps->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
#ifdef __APPLE__
        // Metal always has primitive restart enabled and MoltenVK rejects primitiveRestartEnable=VK_FALSE
        // with VK_ERROR_FEATURE_NOT_PRESENT. Force it on; harmless here because these draws don't use the
        // strip-restart sentinel index (0xFFFF/0xFFFFFFFF), and it has no effect on list topologies.
        ia.primitiveRestartEnable = VK_TRUE;
#endif
        // Default: full-target viewport. When the resolved state carries the guest's PA_CL_VPORT transform,
        // honor it — a guest yscale < 0 arrives as a negative viewport_h (Vulkan core-1.1 flipped viewport),
        // reproducing the hardware's Y orientation (#38; each draw item keeps its own resolved viewport).
        if (timing_enabled) res_fixed_stages_ms += setup_elapsed_ms(fixed_stages_begin, TimingClock::now());
        // viewport/scissor/raster begins here; the index upload above is timed separately.
        const auto fixed_viewport_begin = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
        if (ps && ps->has_viewport)
            vp = {ps->viewport_x, ps->viewport_y, ps->viewport_w, ps->viewport_h, ps->min_depth, ps->max_depth};
        v.viewport = vp;
        VkPipelineViewportStateCreateInfo vpst{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpst.viewportCount = 1; vpst.pViewports = &vp; vpst.scissorCount = 1; vpst.pScissors = &sc;
        // Viewports are draw state, not a shader/render-pass compatibility axis. Baking the camera's
        // sub-pixel viewport into every pipeline key made GTA V create 20-30 nominally new graphics
        // pipelines on every 1440p gameplay frame. Vulkan 1.0 makes viewport dynamic in core; record
        // the resolved per-draw value beside the already-dynamic scissor.
        const VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_LINE_WIDTH,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        };
        VkPipelineDynamicStateCreateInfo dynamic_state{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(std::size(dynamic_states));
        dynamic_state.pDynamicStates = dynamic_states;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        // Honor the guest's PA_SU_SC_MODE_CNTL cull/front-face/polygon mode (#456). Resolve encodes these
        // as the Vk enumerators; an absent register resolves to the same NONE/CCW/FILL default above, so
        // the null-ps (test) path and any draw that never programs it are byte-identical. PROSPER_NO_CULL
        // forces CULL_NONE back on; PROSPER_FLIP_FRONT_FACE preserves culling and toggles only winding.
        // Together they isolate a cull-mode problem from a front-face translation problem without a rebuild.
        if (ps) { rs.cullMode  = getenv("PROSPER_NO_CULL") ? VK_CULL_MODE_NONE : (VkCullModeFlags)ps->cull_mode;
                  rs.frontFace = getenv("PROSPER_FLIP_FRONT_FACE")
                      ? (ps->front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                  : VK_FRONT_FACE_CLOCKWISE)
                      : (VkFrontFace)ps->front_face;
                  rs.polygonMode = (VkPolygonMode)ps->polygon_mode; }
        // Depth bias (#1349): the guest's PA_SU_POLY_OFFSET_* — shadow-map passes need it against
        // acne. Clamp requires the depthBiasClamp device feature; without it a non-zero clamp is
        // dropped to 0 (bias still applies, unclamped — the safe direction). PROSPER_NO_DEPTH_BIAS
        // is the A/B diagnostic, symmetric with PROSPER_NO_CULL above.
        if (ps && ps->depth_bias_enable && !PROSPER_ENV_ON("PROSPER_NO_DEPTH_BIAS")) {
            rs.depthBiasEnable         = VK_TRUE;
            rs.depthBiasConstantFactor = ps->depth_bias_constant;
            rs.depthBiasSlopeFactor    = ps->depth_bias_slope;
            rs.depthBiasClamp = render_vk_ctx().depth_bias_clamp_enabled ? ps->depth_bias_clamp : 0.0f;
        }
        v.line_width = rs.lineWidth;
        v.depth_bias_constant = rs.depthBiasConstantFactor;
        v.depth_bias_clamp = rs.depthBiasClamp;
        v.depth_bias_slope = rs.depthBiasSlopeFactor;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        const auto fixed_viewport_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) res_fixed_viewport_ms += setup_elapsed_ms(fixed_viewport_begin, fixed_viewport_ready);
        std::array<VkPipelineColorBlendAttachmentState,
                   prosper::gpu::kColorTargetCount> cba{};
        for (uint32_t slot = 0; slot < color_count; ++slot) cba[slot].colorWriteMask = 0xF;
        if (ps) {
            cba[0].colorWriteMask = ps->color_write_mask;
            cba[0].blendEnable    = ps->blend_enable ? VK_TRUE : VK_FALSE;
            if (getenv("PROSPER_NO_BLEND")) cba[0].blendEnable = VK_FALSE;   // diag: isolate blend compositing
            cba[0].srcColorBlendFactor = (VkBlendFactor)ps->src_color_blend_factor;
            cba[0].dstColorBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor;
            cba[0].colorBlendOp        = (VkBlendOp)ps->color_blend_op;
            // Alpha channel uses its OWN resolved factors (#381): resolve set these from the separate
            // ALPHA_* blend fields when SEPARATE_ALPHA_BLEND was programmed, else it already mirrored the
            // color factors — so this is correct in both cases without guessing here.
            cba[0].srcAlphaBlendFactor = (VkBlendFactor)ps->src_alpha_blend_factor;
            cba[0].dstAlphaBlendFactor = (VkBlendFactor)ps->dst_alpha_blend_factor;
            cba[0].alphaBlendOp        = (VkBlendOp)ps->alpha_blend_op;
            cba[1].colorWriteMask = ps->color1_write_mask;
            cba[1].blendEnable = ps->blend1_enable ? VK_TRUE : VK_FALSE;
            if (getenv("PROSPER_NO_BLEND")) cba[1].blendEnable = VK_FALSE;
            cba[1].srcColorBlendFactor = (VkBlendFactor)ps->src_color_blend_factor1;
            cba[1].dstColorBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor1;
            cba[1].colorBlendOp = (VkBlendOp)ps->color_blend_op1;
            cba[1].srcAlphaBlendFactor = (VkBlendFactor)ps->src_alpha_blend_factor1;
            cba[1].dstAlphaBlendFactor = (VkBlendFactor)ps->dst_alpha_blend_factor1;
            cba[1].alphaBlendOp = (VkBlendOp)ps->alpha_blend_op1;
            for (uint32_t slot = 2; slot < color_count; ++slot) {
                const auto& target = ps->color_targets[slot];
                cba[slot].colorWriteMask = target.write_mask;
                cba[slot].blendEnable = target.blend_enable ? VK_TRUE : VK_FALSE;
                if (getenv("PROSPER_NO_BLEND")) cba[slot].blendEnable = VK_FALSE;
                cba[slot].srcColorBlendFactor =
                    static_cast<VkBlendFactor>(target.src_color_blend_factor);
                cba[slot].dstColorBlendFactor =
                    static_cast<VkBlendFactor>(target.dst_color_blend_factor);
                cba[slot].colorBlendOp = static_cast<VkBlendOp>(target.color_blend_op);
                cba[slot].srcAlphaBlendFactor =
                    static_cast<VkBlendFactor>(target.src_alpha_blend_factor);
                cba[slot].dstAlphaBlendFactor =
                    static_cast<VkBlendFactor>(target.dst_alpha_blend_factor);
                cba[slot].alphaBlendOp = static_cast<VkBlendOp>(target.alpha_blend_op);
            }
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        if (ps && ps->logic_op_enable) {
            if (ctx.logic_op_enabled) {
                cb.logicOpEnable = VK_TRUE;
                cb.logicOp = static_cast<VkLogicOp>(ps->logic_op);
            } else {
                static std::once_flag logged;
                std::call_once(logged, [] {
                    fprintf(stderr, "[gpu] Vulkan device lacks logicOp support -> COPY fallback\n");
                });
            }
        }
        cb.attachmentCount = color_count; cb.pAttachments = cba.data();
        const auto fixed_blend_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) res_fixed_blend_ms += setup_elapsed_ms(fixed_viewport_ready, fixed_blend_ready);
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (ps && ps->depth_test_enable) {
            dss.depthTestEnable  = VK_TRUE;
            dss.depthWriteEnable = ps->depth_write_enable ? VK_TRUE : VK_FALSE;
            dss.depthCompareOp   = (VkCompareOp)ps->depth_compare_op;
            // UE4 repeats its reverse-Z depth prepass in a separately translated base-pass shader.
            // A one-ULP position difference between those shaders makes exact EQUAL reject the whole
            // base pass, although the guest hardware accepts the pair. Preserve occlusion by relaxing
            // only a read-only EQUAL against an already-populated, explicitly reverse-Z surface:
            // GEQUAL still rejects geometry behind the prepass instead of disabling depth outright.
            const bool reverse_z_equal_compat = persistent_ds && depth_was_valid &&
                !ps->depth_write_enable && ps->depth_compare_op == VK_COMPARE_OP_EQUAL &&
                ps->has_depth_clear && ps->depth_clear_value <= 0.5f;
            if (reverse_z_equal_compat) {
                dss.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
                if (PROSPER_ENV_ON("PROSPER_DSLOG"))
                    fprintf(stderr, "[ds] reverse-Z read-only EQUAL -> GEQUAL compatibility\n");
            }
        }
        if (ps && ps->stencil_enable) {
            // Wire the front/back stencil op-state so masks clip (e.g. the title shimmer tests the
            // stencil the logo draw wrote). ref/compareMask/writeMask are baked (not dynamic).
            dss.stencilTestEnable = VK_TRUE;
            auto mkop = [&](int fb) {
                VkStencilOpState s{};
                s.failOp      = (VkStencilOp)ps->stencil_fail_op[fb];
                s.passOp      = (VkStencilOp)ps->stencil_pass_op[fb];
                s.depthFailOp = (VkStencilOp)ps->stencil_depth_fail_op[fb];
                s.compareOp   = (VkCompareOp)ps->stencil_compare_op[fb];
                // PROSPER_STENCIL_MIRROR=1 (diagnostic A/B ONLY — NOT hardware semantics): mirror
                // the asymmetric compare ops, i.e. evaluate `stencil OP ref` instead of Vulkan's
                // `ref OP stencil`. RDNA2 STENCILFUNC is ref-on-left 1:1 with VkCompareOp (radeonsi
                // programs PIPE_FUNC straight into the field), so this switch deliberately DIVERGES
                // from hardware — its use is isolating whether a suspect draw's coverage is
                // stencil-gated (flipping GREATER<->LESS suppresses/expands it) without touching
                // any other state. Symmetric ops (EQUAL/NOTEQUAL/ALWAYS/NEVER) are unaffected.
                if (PROSPER_ENV_ON("PROSPER_STENCIL_MIRROR")) {
                    switch (s.compareOp) {
                        case VK_COMPARE_OP_LESS:             s.compareOp = VK_COMPARE_OP_GREATER; break;
                        case VK_COMPARE_OP_GREATER:          s.compareOp = VK_COMPARE_OP_LESS; break;
                        case VK_COMPARE_OP_LESS_OR_EQUAL:    s.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; break;
                        case VK_COMPARE_OP_GREATER_OR_EQUAL: s.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL; break;
                        default: break;
                    }
                }
                s.compareMask = ps->stencil_compare_mask[fb];
                s.writeMask   = ps->stencil_write_mask[fb];
                // AMD splits the stencil reference: STENCILTESTVAL is the COMPARE reference, but a REPLACE
                // op writes STENCILOPVAL. Vulkan has one `reference` for both. When this draw REPLACEs
                // (its whole purpose is to WRITE a mask value, and its compare is typically ALWAYS so the
                // compare-ref is irrelevant), use STENCILOPVAL so the mask is written with the value the
                // game intended — else a mask-write with TESTVAL=0/OPVAL=1 writes 0 and every later
                // test==1 draw is wrongly culled (PPSA02664's whole UI vanished; #270).
                const uint32_t REPLACE = 2;   // VK_STENCIL_OP_REPLACE
                bool does_replace = (ps->stencil_pass_op[fb] == REPLACE || ps->stencil_fail_op[fb] == REPLACE ||
                                     ps->stencil_depth_fail_op[fb] == REPLACE);
                s.reference   = does_replace ? ps->stencil_op_val[fb] : ps->stencil_ref[fb];
                if (does_replace && s.compareOp == VK_COMPARE_OP_ALWAYS)
                    if (const char* v = getenv("PROSPER_STENCIL_REPLACE"))
                        s.reference = static_cast<uint32_t>(strtoul(v, nullptr, 0)) & 0xFFu;
                return s;
            };
            dss.front = mkop(0); dss.back = mkop(1);
            if (PROSPER_ENV_ON("PROSPER_STENCILLOG"))
                fprintf(stderr, "[stencil] front{cmp=%u ref=%u opval=%u cmask=0x%x wmask=0x%x fail=%u pass=%u zfail=%u} back{cmp=%u ref=%u fail=%u pass=%u zfail=%u} vkref=%u/%u cull=%u depth_test=%d\n",
                        ps->stencil_compare_op[0], ps->stencil_ref[0], ps->stencil_op_val[0], ps->stencil_compare_mask[0], ps->stencil_write_mask[0],
                        ps->stencil_fail_op[0], ps->stencil_pass_op[0], ps->stencil_depth_fail_op[0],
                        ps->stencil_compare_op[1], ps->stencil_ref[1],
                        ps->stencil_fail_op[1], ps->stencil_pass_op[1], ps->stencil_depth_fail_op[1],
                        dss.front.reference, dss.back.reference, (unsigned)ps->cull_mode, (int)ps->depth_test_enable);
        }
        v.stencil_front = dss.front;
        v.stencil_back = dss.back;
        // Descriptor resources for this draw (two-set: VS=set0, PS=set1 — same layout as the single path).
        const auto fixed_dss_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) res_fixed_depth_stencil_ms += setup_elapsed_ms(fixed_blend_ready, fixed_dss_ready);
        const auto setup_fixed_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) setup_fixed_ms += setup_elapsed_ms(setup_shaders_ready, setup_fixed_ready);
        const auto& R = *effective_resources[di];
        v.use_desc = !R.empty();
        for (auto& r : R) v.n_sets = std::max(v.n_sets, r.set + 1);
        std::vector<VkDescriptorSetLayout> dsls(v.n_sets, VK_NULL_HANDLE);
        v.dsets.assign(v.n_sets, VK_NULL_HANDLE);
        std::vector<std::vector<uint64_t>> descriptor_layout_keys(v.n_sets);
        if (v.use_desc) {
            std::vector<VkDescriptorSetLayoutBinding> lb(R.size());
            // One contiguous run of VkDescriptorBufferInfo per resource, arity entries long. Sized
            // ONCE, before anything takes an address into it: `wr[].pBufferInfo` points in here and
            // must stay valid until vkUpdateDescriptorSets, so growing this vector while
            // materialising entries would silently invalidate pointers already stored in `wr` --
            // which Vulkan reads as garbage descriptors rather than reporting as an error.
            std::vector<uint32_t> dbi_offset(R.size(), 0);
            {
                uint32_t running = 0;
                for (size_t i = 0; i < R.size(); i++) {
                    dbi_offset[i] = running;
                    running += R[i].descriptor_arity();
                }
                (void)running;
            }
            const size_t dbi_total = R.empty()
                ? 0 : static_cast<size_t>(dbi_offset.back()) + R.back().descriptor_arity();
            std::vector<VkDescriptorBufferInfo> dbi(dbi_total);
                // Resolve one storage-buffer payload to an index in `shared_buffers` -- the memo lookup,
                // the size-gated content dedup, and the arena/pool/create upload. Extracted verbatim
                // from the per-resource site below so it can be called MORE THAN ONCE per resource: an
                // indexed descriptor array (#2412 stage 5) needs one upload per table entry, and each
                // of them wants exactly this dedup and arena behaviour.
                //
                // Pure refactor, no behavioural change. Two mechanical transformations: the fatal
                // upload path used to `break` the caller's resource loop and now returns false for the
                // caller to do the same, and `r.set`/`r.binding`/`r.buffer_identity` became parameters
                // so the body no longer reaches for the enclosing resource.
                auto resolve_buffer_upload =
                    [&](const uint32_t* words, size_t word_count, uint64_t identity,
                        uint32_t set, uint32_t binding, size_t& buffer_index_out) -> bool {
                if (!words || !word_count) {
                    words = &zero_buffer_word;
                    word_count = 1;
                }
                // PROSPER_BUFLOG=1: per-binding upload provenance — set/binding, word count,
                // and the first dwords as floats. Ground truth for "which bytes did the shader
                // see" when a fetch-offset defect is suspected (the #1287 palette-UV audit).
                static const bool buflog_enabled = getenv("PROSPER_BUFLOG") != nullptr;
                if (buflog_enabled) {
                    static int buflog_count = 0;
                    if (buflog_count++ < 400) {
                        const float* fp = reinterpret_cast<const float*>(words);
                        fprintf(stderr,
                                "[buflog] set=%u binding=%u words=%zu id=%llx f0=%g f1=%g f2=%g\n",
                                set, binding, word_count,
                                (unsigned long long)identity,
                                word_count > 0 ? fp[0] : 0.f, word_count > 1 ? fp[1] : 0.f,
                                word_count > 2 ? fp[2] : 0.f);
                    }
                }
                ++resource_reuse_stats.buffer_references;
                ++backend_hash_stats_totals().references;
                const bool shareable = share_backend_resources && identity != 0;
                size_t buffer_index = SIZE_MAX;
                const BufferRefMemoKey memo_key{words, word_count, identity};
                if (shareable) {
                    auto memo_found = buffer_ref_memo.find(memo_key);
                    if (memo_found != buffer_ref_memo.end()) {
                        buffer_index = memo_found->second;
                        ++resource_reuse_stats.buffer_ref_memo_hits;
                        ++backend_hash_stats_totals().memo_hits;
                    }
                }
                if (buffer_index != SIZE_MAX) {
                    // repeat reference within this call — resolved without hashing (#1268)
                } else {
                // A unique-tag key can never match an existing entry (the tag differs from every
                // other key and operator== checks all scalars before the memcmp), so its content
                // hash contributes nothing to the lookup — skip it (#1268).
                //
                // Large buffers also take a unique tag: cross-pointer content dedup is kept for
                // SMALL buffers only (<= kSharedBufferHashDedupMaxDwords). Measured live on Blue
                // Prince's loading submits, the content-hash lookup had ZERO dedup hits across
                // 556K hashed references while costing ~1.8 GiB/s of FNV over ~97 KiB average
                // payloads — pure waste at exactly the draw volume where it hurts. Repeat
                // references to a large range still resolve through the per-call memo above;
                // what a large buffer loses is only the merge of two DIFFERENT-pointer,
                // identical-content references within one call, which the live data shows never
                // happens. Small buffers (per-draw UBO-sized, the tests' contract) keep the full
                // hash + memcmp dedup exactly as before.
                constexpr size_t kSharedBufferHashDedupMaxDwords = 1024;   // 4 KiB
                const bool hash_dedup =
                    shareable && word_count <= kSharedBufferHashDedupMaxDwords;
                uint64_t content_hash = 0;
                if (hash_dedup) {
                    const ResourcePhaseTimer phase_hash(timing_enabled, &res_buffer_hash_ms);
                    content_hash = hash_buffer_words(words, word_count);
                    ++resource_reuse_stats.buffer_hash_calls;
                    resource_reuse_stats.buffer_hash_dwords += word_count;
                    ++backend_hash_stats_totals().hash_calls;
                    backend_hash_stats_totals().hash_dwords += word_count;
                } else if (shareable) {
                    ++resource_reuse_stats.buffer_hash_skipped_large;
                    resource_reuse_stats.buffer_skipped_large_dwords += word_count;
                    ++backend_hash_stats_totals().skipped_large;
                } else {
                    ++resource_reuse_stats.buffer_hash_skipped_unique;
                    ++backend_hash_stats_totals().skipped_unique;
                }
                SharedBufferKey buffer_key{
                    words, word_count, identity, content_hash,
                    hash_dedup ? 0 : ++resource_unique_tag};
                auto buffer_found = shared_buffer_indices.end();
                {
                    const ResourcePhaseTimer phase_index(timing_enabled,
                                                         &res_buffer_index_find_ms);
                    buffer_found = shared_buffer_indices.find(buffer_key);
                }
                if (buffer_found != shared_buffer_indices.end()) {
                    buffer_index = buffer_found->second;
                } else {
                    buffer_index = shared_buffers.size();
                    SharedBufferUpload upload;
                    const VkDeviceSize bytes = static_cast<VkDeviceSize>(word_count) * 4;
                    if (use_buffer_arena) {
                        const ResourcePhaseTimer phase_acquire(timing_enabled,
                                                               &res_buffer_acquire_ms);
                        acquire_buffer_arena_slice(bytes, upload);
                    }
                    if (!upload.arena && reuse_host_buffers) {
                        const ResourcePhaseTimer phase_acquire(timing_enabled,
                                                               &res_buffer_acquire_ms);
                        RenderHostBuffer pooled = acquire_render_host_buffer(ctx, bytes);
                        upload.buffer = pooled.buffer;
                        upload.memory = pooled.memory;
                        upload.mapped = pooled.mapped;
                        upload.bytes = pooled.bytes;
                        upload.allocation_bytes = pooled.allocation_bytes;
                        upload.pooled = upload.buffer && upload.memory && upload.mapped;
                    }
                    if (!upload.arena && !upload.pooled) {
                        // create+allocate+map+copy, all of it. Neither `acquire` nor `copy` saw
                        // this branch, so a payload that misses both the arena and the pool cost
                        // a full Vulkan allocation that read as free.
                        const ResourcePhaseTimer phase_create(timing_enabled,
                                                              &res_buffer_create_ms);
                        RenderBufferUploadFailureStep failure =
                            create_transient_storage_buffer_upload(
                                ctx, words, bytes, upload.buffer, upload.memory);
                        if (failure != RenderBufferUploadFailureStep::None) {
                            static std::atomic<uint32_t> failure_logs{0};
                            if (failure_logs.fetch_add(1, std::memory_order_relaxed) < 32)
                                std::fprintf(
                                    stderr,
                                    "[buffer-upload-failed] set=%u binding=%u requested=%llu "
                                    "step=%s — substituting one zero word\n",
                                    set, binding, (unsigned long long)bytes,
                                    render_buffer_upload_failure_name(failure));
                            upload = {};
                            failure = create_transient_storage_buffer_upload(
                                ctx, &zero_buffer_word, sizeof(zero_buffer_word),
                                upload.buffer, upload.memory);
                            if (failure != RenderBufferUploadFailureStep::None) {
                                if (failure_logs.fetch_add(1, std::memory_order_relaxed) < 32)
                                    std::fprintf(
                                        stderr,
                                        "[buffer-upload-failed] set=%u binding=%u zero-word "
                                        "fallback step=%s — skipping draw\n",
                                        set, binding,
                                        render_buffer_upload_failure_name(failure));
                                return false;   // caller breaks its loop; see the call site
                            }
                            upload.range = sizeof(zero_buffer_word);
                            ++resource_reuse_stats.buffer_upload_fallbacks;
                        } else {
                            upload.range = bytes;
                        }
                    } else {
                        const ResourcePhaseTimer phase_copy(timing_enabled,
                                                            &res_buffer_copy_ms);
                        resource_reuse_stats.buffer_upload_bytes += static_cast<uint64_t>(bytes);
                        std::memcpy(static_cast<uint8_t*>(upload.mapped) + upload.offset,
                                    words, static_cast<size_t>(bytes));
                        upload.range = bytes;
                    }
                    {
                        const ResourcePhaseTimer phase_index(timing_enabled,
                                                             &res_buffer_index_insert_ms);
                        shared_buffers.push_back(upload);
                        shared_buffer_indices.emplace(buffer_key, buffer_index);
                        if (buffer_verify_enabled)
                            buffer_verify_records.push_back(
                                {words, word_count, identity, set, binding, buffer_index});
                    }
                    ++resource_reuse_stats.unique_buffers;
                    ++backend_hash_stats_totals().unique_buffers;
                }
                if (shareable) {
                    const ResourcePhaseTimer phase_index(timing_enabled,
                                                         &res_buffer_index_find_ms);
                    buffer_ref_memo.emplace(memo_key, buffer_index);
                }
                }
                buffer_index_out = buffer_index;
                return true;
                };

            std::vector<VkDescriptorImageInfo> dii(R.size());
            std::vector<VkWriteDescriptorSet> wr(R.size());
            bool buffer_resources_ready = true;
            for (size_t i = 0; i < R.size(); i++) {
                const FrameResource& r = R[i];
                lb[i] = {}; lb[i].binding = r.binding;
                // What the WRITE will supply, not what the resource declares (#2477). See
                // `written_descriptor_count()`. Loud once if a producer ever asks for an array on a
                // class whose write path cannot build one -- silently declaring 1 would drop the
                // extra entries and read as handled.
                lb[i].descriptorCount = r.written_descriptor_count();
                if (r.descriptor_arity() != r.written_descriptor_count()) {
                    static std::once_flag warned;
                    std::call_once(warned, [&] {
                        std::fprintf(stderr,
                                     "[render] UNSUPPORTED descriptor array: binding %u declares %u "
                                     "entries but its class writes one (texture=%d storage-image=%d "
                                     "internal-gds=%d). Declaring 1 so the layout and the write agree; "
                                     "entries 1..%u are IGNORED. Arrays are implemented for storage "
                                     "buffers only -- see #2477.\n",
                                     r.binding, r.descriptor_arity(),
                                     (int)r.is_texture(), (int)r.is_storage_image,
                                     (int)r.is_internal_gds, r.descriptor_arity() - 1u);
                    });
                }
                if (r.is_texture()) {
                    const ResourcePhaseTimer phase_texture(timing_enabled, &res_texture_ms);
                    lb[i].descriptorType = r.is_storage_image
                        ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                        : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    // A set-0 texture belongs to the VERTEX shader (build_R tags VS resources into set 0,
                    // PS into set 1). stageFlags must include every stage that reads the binding, so a
                    // vertex texture fetch (displacement/heightmap, GPU vertex animation) needs
                    // VERTEX_BIT — a fragment-only hardcode made set-0 textures invisible to the VS,
                    // yielding undefined samples / a validation error (#376). Match the storage-buffer path.
                    lb[i].stageFlags = (r.set == 0)
                        ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
                    texture_references++;
                    // Every ACTIVE bound slot, not just 0 and 1. A sampled renderer mip chain must
                    // not copy any level that is also an attachment of this pass; the copy is
                    // recorded before the render pass and would observe the previous contents.
                    auto target_is_feedback = [&](uint64_t target_id,
                                                  uint32_t target_w,
                                                  uint32_t target_h) {
                        if (!color_target || !target_id) return false;
                        uint64_t bases[prosper::gpu::kColorTargetCount]{};
                        bool active[prosper::gpu::kColorTargetCount]{};
                        bases[0] = color_target->persistent_id;
                        active[0] = persistent_color;
                        bases[1] = color_target->persistent_id1;
                        active[1] = persistent_color1;
                        for (uint32_t slot = 2; slot < color_count; ++slot) {
                            bases[slot] = color_target->persistent_id_slots[slot];
                            active[slot] = cached_extra[slot] != nullptr;
                        }
                        return prosper::frontend::mrt_target_view_feedback(
                            bases, active, color_count, target_id,
                            target_w, target_h, W, H);
                    };
                    const bool target_feedback = target_is_feedback(
                        r.persistent_render_target_id, r.tw, r.th);
                    bool target_mip_feedback = target_feedback;
                    const uint32_t requested_target_mips = std::min<uint32_t>(
                        r.persistent_render_target_mip_count,
                        static_cast<uint32_t>(r.persistent_render_target_mip_ids.size()));
                    for (uint32_t level = 1;
                         !target_mip_feedback && level < requested_target_mips; ++level)
                        target_mip_feedback = target_is_feedback(
                            r.persistent_render_target_mip_ids[level],
                            std::max(r.tw >> level, 1u),
                            std::max(r.th >> level, 1u));
                    // #1272: effective generated-mip chain — bounded by the chain the T# itself
                    // declares (declared_mip_levels; 1 = historical single-level behavior), and
                    // restricted to plain-2D RGBA8 sampled guest textures. Cube/volume stacks,
                    // storage images, RTT-backed bindings, and other formats keep one level.
                    uint32_t tex_mip_levels = 1;
                    // PROSPER_MIPLOG=1: report every texture binding's mip-eligibility inputs —
                    // which gate leg starves a declared chain down to one level (#1287 moiré).
                    static const bool miplog_enabled = getenv("PROSPER_MIPLOG") != nullptr;
                    if (miplog_enabled && r.tw >= 64 && !r.is_storage_image)
                        fprintf(stderr,
                                "[miplog] %ux%ux%u dim=%u declared_mips=%u fmt=%d rtt=%llu ds=%llu "
                                "rgba=%d min_lod=%.1f max_lod=%.1f mag=%u min=%u mip=%u\n",
                                r.tw, r.th, r.td, r.img_dim, r.declared_mip_levels,
                                (int)r.texture_format,
                                (unsigned long long)r.persistent_render_target_id,
                                (unsigned long long)r.persistent_depth_target_id,
                                r.tex_rgba != nullptr, r.min_lod, r.max_lod,
                                r.mag_filter, r.min_filter, r.mip_filter);
                    const VkFormat sampled_format = backend_color_format(r.texture_format);
                    bool generated_mip_format_supported =
                        sampled_format == VK_FORMAT_R8G8B8A8_UNORM;
                    if (sampled_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                        VkFormatProperties properties{};
                        vkGetPhysicalDeviceFormatProperties(phys, sampled_format, &properties);
                        constexpr VkFormatFeatureFlags required =
                            VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                            VK_FORMAT_FEATURE_BLIT_DST_BIT |
                            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
                        generated_mip_format_supported =
                            (properties.optimalTilingFeatures & required) == required;
                    }
                    std::array<VkImage, 16> renderer_mip_images{};
                    std::array<VkImageLayout, 16> renderer_mip_layouts{};
                    uint32_t renderer_mip_sources = 0;
                    if (!r.is_storage_image && !target_mip_feedback && r.img_dim == 1u &&
                        r.td == 1u && r.sample_count == 1u && generated_mip_format_supported &&
                        requested_target_mips > 1u &&
                        requested_target_mips == r.declared_mip_levels) {
                        for (uint32_t level = 0; level < requested_target_mips; ++level) {
                            const uint64_t id = r.persistent_render_target_mip_ids[level];
                            if (!id) continue;
                            auto* target = find_persistent_color_target(
                                id, std::max(r.tw >> level, 1u),
                                std::max(r.th >> level, 1u), sampled_format);
                            if (!target || !target->image ||
                                target->layout == VK_IMAGE_LAYOUT_UNDEFINED)
                                continue;
                            target->last_use = color_target_generation;
                            renderer_mip_images[level] = target->image;
                            renderer_mip_layouts[level] = target->layout;
                            ++renderer_mip_sources;
                        }
                    }
                    const bool assemble_renderer_mips = renderer_mip_sources > 1u &&
                        renderer_mip_images[0] != VK_NULL_HANDLE;
                    if (!r.is_storage_image && r.img_dim == 1 && r.td == 1 &&
                        ((!r.persistent_render_target_id && r.tex_rgba) ||
                         assemble_renderer_mips) &&
                        // Widening this format gate requires BLIT_SRC/BLIT_DST +
                        // SAMPLED_IMAGE_FILTER_LINEAR on the new format (the blit cascade below).
                        generated_mip_format_supported &&
                        r.declared_mip_levels > 1 && (r.tw > 1 || r.th > 1)) {
                        uint32_t full = 1;
                        for (uint32_t m = r.tw > r.th ? r.tw : r.th; m > 1; m >>= 1) full++;
                        tex_mip_levels = r.declared_mip_levels < full ? r.declared_mip_levels : full;
                        // Symmetry with the 16-region copy cap below; unreachable from guest data
                        // (T# extents are rejected above 16384 -> <= 15 levels).
                        if (tex_mip_levels > 16u) tex_mip_levels = 16u;
                    }
                    // A DS-bridged resource shares the id slot: both are guest plane addresses, and
                    // pixels stays null for either direct bind, so distinct surfaces cannot collide.
                    std::array<uint32_t, 4> uniform_color_bits{};
                    if (r.has_uniform_color)
                        for (size_t channel = 0; channel < uniform_color_bits.size(); ++channel)
                            std::memcpy(&uniform_color_bits[channel], &r.uniform_color[channel],
                                        sizeof(uint32_t));
                    const TextureUploadKey texture_key{
                        r.tex_rgba,
                        r.persistent_render_target_id ? r.persistent_render_target_id
                                                      : r.persistent_depth_target_id,
                        r.borrowed_compute_image,
                        r.tw, r.th, r.td, r.img_dim, r.sample_count,
                        tex_mip_levels, backend_color_format(r.texture_format), r.is_storage_image,
                        uniform_color_bits};
                    size_t upload_index = SIZE_MAX;
                    if (share_texture_uploads) {
                        auto found = texture_upload_indices.find(texture_key);
                        if (found != texture_upload_indices.end()) upload_index = found->second;
                    }
                    if (upload_index == SIZE_MAX) {
                        const ResourcePhaseTimer phase_upload(timing_enabled,
                                                              &res_texture_upload_ms);
                        upload_index = texture_uploads.size();
                        texture_uploads.push_back({});
                        SharedTextureUpload& upload = texture_uploads.back();
                        upload.key = texture_key;
                        if (share_texture_uploads) texture_upload_indices.emplace(texture_key, upload_index);

                        if (!r.is_storage_image && r.borrowed_compute_image &&
                            r.borrowed_compute_device == static_cast<void*>(dev) &&
                            r.borrowed_compute_image_layout !=
                                static_cast<uint32_t>(VK_IMAGE_LAYOUT_UNDEFINED) &&
                            r.borrowed_compute_image_lease) {
                            if (r.borrowed_compute_vertical_stack_layers) {
                                upload.stacked_compute = true;
                                upload.stacked_compute_source =
                                    static_cast<VkImage>(r.borrowed_compute_image);
                                upload.stacked_compute_layers =
                                    r.borrowed_compute_vertical_stack_layers;
                            } else {
                                upload.image = static_cast<VkImage>(r.borrowed_compute_image);
                                upload.borrowed_compute = true;
                            }
                            upload.borrowed_compute_layout = static_cast<VkImageLayout>(
                                r.borrowed_compute_image_layout);
                            upload.borrowed_compute_lease = r.borrowed_compute_image_lease;
                        }

                        if (assemble_renderer_mips) {
                            upload.assembled_target_mips = true;
                            upload.target_mip_images = renderer_mip_images;
                            upload.target_mip_layouts = renderer_mip_layouts;
                        }
                        if (!upload.borrowed_compute && !r.is_storage_image &&
                            !upload.assembled_target_mips && persistent_color_targets_enabled &&
                            r.persistent_render_target_id && r.img_dim == 1) {
                            if (auto* target = find_persistent_color_target(
                                    r.persistent_render_target_id, r.tw, r.th,
                                    backend_color_format(r.texture_format))) {
                                target->last_use = color_target_generation;
                                if (target_feedback) {
                                    upload.feedback_snapshot = true;
                                    upload.feedback_source = target->image;
                                    upload.feedback_source_layout = target->layout;
                                    // Earlier barriers in this command buffer move a retained
                                    // attachment that will LOAD to COLOR_ATTACHMENT_OPTIMAL. Record
                                    // the layout the copy actually sees, not the cache's pre-call
                                    // resting layout.
                                    if ((target == cached_color && load_cached_color) ||
                                        (target == cached_color1 && load_cached_color1)) {
                                        upload.feedback_source_layout =
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                                    } else {
                                        for (uint32_t slot = 2; slot < color_count; ++slot)
                                            if (target == cached_extra[slot] && load_extra[slot])
                                                upload.feedback_source_layout =
                                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                                    }
                                } else {
                                    upload.image = target->image;
                                    upload.image_bytes = target->bytes;
                                    upload.borrowed_target = true;
                                }
                                ++color_target_stats.sampled_hits;
                            }
                        }
                        // Sampled depth bridge (#1275): the T# addresses a depth plane rendered
                        // into a persistent DS image (never written back to guest memory). Bind
                        // that image's depth aspect directly. The pass's own DS attachment must
                        // not be borrowed as a sampled input (feedback) — cached_ds identifies it.
                        if (!upload.image && !upload.borrowed_target && !upload.borrowed_compute &&
                            !r.is_storage_image &&
                            r.persistent_depth_target_id && r.img_dim == 1) {
                            const PersistentDsSampled sampled_ds = find_persistent_ds_sampled(
                                r.persistent_depth_target_id, r.tw, r.th);
                            const bool same_pass_feedback = self_sampled_depth && cached_ds &&
                                sampled_ds.image == cached_ds;
                            if (sampled_ds.image &&
                                (same_pass_feedback || !cached_ds ||
                                 sampled_ds.image->image != cached_ds->image)) {
                                upload.image = sampled_ds.image->image;
                                upload.borrowed_ds = true;
                                upload.borrowed_ds_feedback = same_pass_feedback;
                                upload.ds_format = sampled_ds.format;
                                upload.borrowed_ds_stencil =
                                    sampled_ds.aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
                            }
                        }
                        upload.persistent_id = r.is_storage_image || upload.borrowed_ds ||
                                upload.feedback_snapshot
                            ? 0 : r.persistent_texture_id;
                        upload.persistent_version = r.persistent_texture_version;
                        const PersistentTextureKey persistent_key{
                            r.persistent_texture_id, r.tw, r.th, r.td, r.img_dim,
                            r.sample_count, texture_key.mip_levels,
                            backend_color_format(r.texture_format)};
                        if (!r.is_storage_image && !upload.borrowed_target &&
                            !upload.borrowed_compute && !upload.borrowed_ds &&
                            persistent_textures_enabled &&
                            r.persistent_texture_id) {
                            auto cached = persistent_texture_images.find(persistent_key);
                            // A failed queued upload may have left this owned allocation in an
                            // unknown layout with undefined contents. The failure callback cannot
                            // destroy it while an indeterminate submission might still be in flight;
                            // a later usable backend call can safely retire it before rebuilding.
                            if (cached != persistent_texture_images.end() &&
                                !cached->second.content_valid) {
                                for (const auto& [key, binding] : cached->second.bindings) {
                                    (void)key;
                                    if (binding.sampler)
                                        vkDestroySampler(dev, binding.sampler, nullptr);
                                    if (binding.view)
                                        vkDestroyImageView(dev, binding.view, nullptr);
                                }
                                if (cached->second.image)
                                    vkDestroyImage(dev, cached->second.image, nullptr);
                                if (cached->second.memory)
                                    vkFreeMemory(dev, cached->second.memory, nullptr);
                                persistent_texture_bytes -= cached->second.bytes;
                                persistent_texture_binding_entries -= cached->second.bindings.size();
                                persistent_texture_images.erase(cached);
                                cached = persistent_texture_images.end();
                            }
                            if (cached != persistent_texture_images.end()) {
                                cached->second.last_use = texture_generation;
                                upload.image = cached->second.image;
                                upload.image_bytes = cached->second.bytes;
                                // Immutable callers keep version zero and the historical exact-ID
                                // contract. Versioned callers may refresh the same allocation after
                                // frontend byte validation proves that its contents changed.
                                if (upload.persistent_version &&
                                    cached->second.content_version !=
                                        upload.persistent_version) {
                                    upload.persistent_refresh = true;
                                    cached->second.content_version =
                                        upload.persistent_version;
                                    active_submission.add_failure_cleanup(
                                        [persistent_key]() {
                                            auto found = persistent_texture_images.find(
                                                persistent_key);
                                            if (found != persistent_texture_images.end())
                                                found->second.content_valid = false;
                                        });
                                    ++persistent_texture_misses;
                                } else {
                                    upload.persistent_hit = true;
                                    ++persistent_texture_hits;
                                }
                            } else {
                                ++persistent_texture_misses;
                            }
                        }

                        if (!upload.persistent_hit && !upload.borrowed_target &&
                            !upload.borrowed_compute && !upload.borrowed_ds) {
                            if (!upload.persistent_refresh) {
                                VkImageCreateInfo tci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                                const bool texture_3d = r.img_dim == 2;
                                tci.imageType = texture_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
                                tci.format = backend_color_format(r.texture_format);
                                tci.extent = {r.tw, r.th, r.td};
                                tci.mipLevels = upload.key.mip_levels;
                                tci.arrayLayers = r.sample_count;
                                tci.samples = VK_SAMPLE_COUNT_1_BIT;
                                tci.tiling = VK_IMAGE_TILING_OPTIMAL;
                                tci.usage = (r.is_storage_image ? VK_IMAGE_USAGE_STORAGE_BIT
                                                              : VK_IMAGE_USAGE_SAMPLED_BIT) |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                            (r.is_storage_image
                                                 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u) |
                                            (upload.key.mip_levels > 1
                                                 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
                                // #3045: the result used to be discarded here, so a device
                                // rejecting the request (VK_ERROR_OUT_OF_DEVICE_MEMORY, or an
                                // extent/array-layer count past a real device's limits -- Vulkan
                                // Core only guarantees maxImageArrayLayers >= 256, well under the
                                // 2048 backend ceiling in #3043) left upload.image as
                                // VK_NULL_HANDLE and fed straight into vkGetImageMemoryRequirements,
                                // which is undefined behaviour on a null image. Check the result and
                                // skip this draw the same way a fatal buffer-upload failure does
                                // above (buffer_resources_ready = false; break;) instead of
                                // propagating the null handle.
                                const VkResult image_create_result =
                                    consume_render_texture_create_failure_once()
                                        ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                                        : vkCreateImage(dev, &tci, nullptr, &upload.image);
                                if (image_create_result != VK_SUCCESS || !upload.image) {
                                    upload.image = VK_NULL_HANDLE;
                                    static std::atomic<uint32_t> texture_create_failure_logs{0};
                                    if (texture_create_failure_logs.fetch_add(
                                            1, std::memory_order_relaxed) < 32)
                                        std::fprintf(
                                            stderr,
                                            "[texture-upload-failed] set=%u binding=%u "
                                            "vkCreateImage result=%d extent=%ux%ux%u mips=%u "
                                            "layers=%u fmt=%d -- skipping draw\n",
                                            r.set, r.binding, (int)image_create_result,
                                            tci.extent.width, tci.extent.height, tci.extent.depth,
                                            tci.mipLevels, tci.arrayLayers, (int)tci.format);
                                    buffer_resources_ready = false;
                                    break;
                                }
                                VkMemoryRequirements tr;
                                vkGetImageMemoryRequirements(dev, upload.image, &tr);
                                upload.image_bytes = tr.size;
                                VkMemoryAllocateInfo tai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                                tai.allocationSize = tr.size;
                                tai.memoryTypeIndex = pick(tr.memoryTypeBits, 0);
                                const bool retain = !r.is_storage_image &&
                                    persistent_textures_enabled && upload.persistent_id &&
                                    tr.size <= persistent_texture_limit;
                                if (retain &&
                                    vkAllocateMemory(dev, &tai, nullptr, &upload.memory) ==
                                        VK_SUCCESS)
                                    upload.direct_memory = true;
                                if (!upload.memory) {
                                    upload.persistent_id = 0;
                                    upload.memory = allocate_transient_render_memory(
                                        dev, tai.allocationSize, tai.memoryTypeIndex);
                                }
                                vkBindImageMemory(dev, upload.image, upload.memory, 0);
                            }

                            // #1272: staging carries level 0 only; levels 1..N-1 are produced on the
                            // GPU by a linear-filtered vkCmdBlitImage cascade at upload time (see the
                            // copy site). A CPU box filter here was the first implementation and
                            // collapsed titles that re-upload large mip-eligible textures per frame
                            // (Evergate's title froze the publish rate — snapshot-gate catch).
                            if (r.has_uniform_color) {
                                upload.uniform_clear = true;
                                std::copy(r.uniform_color.begin(), r.uniform_color.end(),
                                          upload.uniform_color.float32);
                            } else if (!upload.feedback_snapshot &&
                                       !upload.assembled_target_mips && !upload.stacked_compute) {
                                const VkDeviceSize tbytes =
                                    static_cast<VkDeviceSize>(r.tw) * r.th * r.td *
                                    r.sample_count *
                                    backend_color_bytes_per_pixel(r.texture_format);
                                VkBufferCreateInfo stci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                                stci.size = tbytes;
                                stci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    (r.is_storage_image
                                         ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0u);
                                vkCreateBuffer(dev, &stci, nullptr, &upload.staging);
                                VkMemoryRequirements sr;
                                vkGetBufferMemoryRequirements(dev, upload.staging, &sr);
                                VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                                sai.allocationSize = sr.size;
                                sai.memoryTypeIndex = pick(sr.memoryTypeBits,
                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                                upload.staging_memory = allocate_transient_render_memory(
                                    dev, sai.allocationSize, sai.memoryTypeIndex);
                                vkBindBufferMemory(dev, upload.staging, upload.staging_memory, 0);
                                void* sp = nullptr;
                                vkMapMemory(dev, upload.staging_memory, 0, tbytes, 0, &sp);
                                if (r.tex_rgba) {
                                    std::memcpy(sp, r.tex_rgba, static_cast<size_t>(tbytes));
                                } else {
                                    // Only a declined/missed depth-plane borrow reaches the creation
                                    // path with no CPU pixels (#1275: the bridge deliberately carries
                                    // none — e.g. the consumer samples its own bound DS attachment, or
                                    // the plane was invalidated between the frontend gate and this
                                    // lookup). Bind well-defined zeros — the value the guest-byte
                                    // decode of an unwritten depth address produced — and say so.
                                    std::memset(sp, 0, static_cast<size_t>(tbytes));
                                    static int declined_logged = 0;
                                    if (declined_logged++ < 16)
                                        fprintf(stderr,
                                                "[dsbridge] borrow declined/missed for 0x%llx %ux%u -> "
                                                "zero texture\n",
                                                (unsigned long long)r.persistent_depth_target_id,
                                                r.tw, r.th);
                                }
                                vkUnmapMemory(dev, upload.staging_memory);
                            }
                        }
                    }
                    if (r.is_storage_image && r.storage_image_writeback &&
                        texture_uploads[upload_index].storage_writebacks.empty())
                        texture_uploads[upload_index].storage_writebacks.push_back(
                            r.storage_image_writeback);
                    const SharedTextureUpload& upload = texture_uploads[upload_index];
                    VkImageViewCreateInfo tvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                    // NOTE(#263): r.srgb carries whether the T# is a gamma-encoded (sRGB) surface, but we
                    // deliberately keep the view UNORM. This whole renderer works in gamma/sRGB space
                    // end-to-end (this target is UNORM, the frontend blit + swapchain are UNORM), with NO
                    // linear->sRGB encode at present. Sampling an sRGB texture as UNORM passes its encoded
                    // bytes straight through, which MATCHES real-hardware output for pass-through content.
                    // Flipping this to VK_FORMAT_R8G8B8A8_SRGB would apply sRGB->linear on sample with no
                    // matching encode on store -> linear values into a UNORM swapchain -> too dark. A
                    // correct sRGB fix is a coordinated linear-working-space + output-encode change (see the
                    // #263 discussion), NOT a per-view format flip. r.srgb is decoded now as groundwork.
                    tvci.image = upload.image;
                    // #325: a guest 2D_ARRAY takes an ARRAY view even at one layer. The view type
                    // has to agree with what the consuming SPIR-V declared, and the recompiler
                    // decides that from the T# (`res->img_dim == 5`) -- which it can do without
                    // knowing how many layers the uploader managed to decode. Keying the view on
                    // the layer COUNT instead would silently disagree whenever an array resolved to
                    // a single layer, binding a 2D view under an `Arrayed=1` OpTypeImage. A
                    // one-layer 2D_ARRAY view is perfectly legal and samples layer 0.
                    tvci.viewType = r.img_dim == 2 ? VK_IMAGE_VIEW_TYPE_3D
                        : (r.guest_array || r.sample_count > 1u) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                 : VK_IMAGE_VIEW_TYPE_2D;
                    tvci.format = backend_color_format(r.texture_format);
                    // T# DST_SEL channel remap (#261): map each SQ_SEL to a VkComponentSwizzle. Identity
                    // (the default, and the narrow/font path) yields IDENTITY == a no-op. PROSPER_NO_SWIZZLE
                    // forces identity for A/B testing against the pre-swizzle behavior.
                    auto vkswz = [](uint32_t s) -> VkComponentSwizzle {
                        switch (s) {
                            case 0:  return VK_COMPONENT_SWIZZLE_ZERO;
                            case 1:  return VK_COMPONENT_SWIZZLE_ONE;
                            case 4:  return VK_COMPONENT_SWIZZLE_R;
                            case 5:  return VK_COMPONENT_SWIZZLE_G;
                            case 6:  return VK_COMPONENT_SWIZZLE_B;
                            case 7:  return VK_COMPONENT_SWIZZLE_A;
                            default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                        }
                    };
                    // Vulkan component mappings do not apply to storage-image accesses.
                    if (!r.is_storage_image && !getenv("PROSPER_NO_SWIZZLE")) {
                        const auto swizzle = backend_sampled_component_swizzle(r);
                        tvci.components = {vkswz(swizzle[0]), vkswz(swizzle[1]),
                                           vkswz(swizzle[2]), vkswz(swizzle[3])};
                    }
                    tvci.subresourceRange =
                        {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels,
                         0, r.sample_count};
                    // Sampled depth bridge (#1275): a borrowed DS image is viewed through its own
                    // depth format's DEPTH aspect (one level — DS surfaces have no mip chains here);
                    // the sampled value arrives in R. The T# swizzle above still applies.
                    if (upload.borrowed_ds) {
                        // The view format stays the image's combined DS format -- a depth/stencil
                        // image admits no format reinterpretation -- and the aspect mask alone
                        // selects the plane. Exactly one aspect, as a sampled view of a combined
                        // format requires (VUID-VkDescriptorImageInfo-imageView-01976).
                        tvci.format = upload.ds_format;
                        tvci.subresourceRange = {
                            upload.borrowed_ds_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_DEPTH_BIT,
                            0, 1, 0, 1};
                    }
                    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                    // Honor the game's decoded S# (r.mag/min/mip_filter, r.addr_uvw) instead of a fixed
                    // LINEAR/clamp sampler — point-sampled art (pixel-art titles) no longer gets a blurred
                    // per-texel outline, and real wrap modes work. Gen5 CLAMP enum -> Vk address mode.
                    auto vkflt  = [](uint32_t f){ return f ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; };
                    auto vkaddr = [](uint32_t c) -> VkSamplerAddressMode {
                        switch (c) {
                            case 0:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                            case 1:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                            case 6: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // 2,3,4,5: clamp-ish
                        }
                    };
                    sci.magFilter = vkflt(r.mag_filter); sci.minFilter = vkflt(r.min_filter);
                    sci.mipmapMode = r.mip_filter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    // Sampled depth bridge (#1275): LINEAR filtering of depth formats is an OPTIONAL
                    // Vulkan format feature, and the manual-compare lowering takes single taps
                    // anyway (filter-then-compare would differ from hardware's compare-then-filter
                    // PCF regardless). Force NEAREST for borrowed depth views.
                    if (upload.borrowed_ds) {
                        sci.magFilter = VK_FILTER_NEAREST;
                        sci.minFilter = VK_FILTER_NEAREST;
                        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    }
                    sci.addressModeU = vkaddr(r.addr_uvw[0]);
                    sci.addressModeV = vkaddr(r.addr_uvw[1]);
                    sci.addressModeW = vkaddr(r.addr_uvw[2]);
                    // Remaining S# fields (#262), applied where valid on this color combined-image-sampler.
                    // Defaults (border 0 / LOD 0,0 / bias 0) reproduce the previous fixed sampler exactly.
                    //   border color: only bites with CLAMP_TO_BORDER wrap. 3 = register/custom (needs
                    //     VK_EXT_custom_border_color); fall back to opaque-black.
                    //   LOD min/max/bias: honored; harmless with our single uploaded mip.
                    // Anisotropy (#275): applied when the S# requests a ratio, the device feature is
                    // enabled, and filtering is linear (Vulkan requires anisotropyEnable only with linear
                    // mag/min filters). maxAnisotropy = 1<<ratio, clamped to the device ceiling. ratio 0
                    // (isotropic) leaves anisotropyEnable false -> the sampler is unchanged.
                    if (r.max_aniso_ratio > 0 && aniso_enabled &&
                        sci.magFilter == VK_FILTER_LINEAR && sci.minFilter == VK_FILTER_LINEAR) {
                        sci.anisotropyEnable = VK_TRUE;
                        float want = (float)(1u << r.max_aniso_ratio);
                        sci.maxAnisotropy = want < max_aniso_limit ? want : max_aniso_limit;
                    }
                    // NOT applied here (need machinery the current path lacks — decoded under GFXLOG only):
                    //   depth_compare_func (needs a depth/shadow sampler over a depth image).
                    // FORCE_UNNORMALIZED is handled in the recompiler by scaling only the spatial
                    // sample coordinates; keeping this sampler normalized preserves wrap and LOD.
                    switch (r.border_color_type) {
                        case 1:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;
                        case 2:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; break;
                        case 3:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;   // custom unsupported
                        default: sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; break;
                    }
                    sci.minLod = r.min_lod; sci.maxLod = r.max_lod; sci.mipLodBias = r.lod_bias;
                    TextureBindingKey binding_key;
                    binding_key.words = {
                        handle_bits(upload.image), static_cast<uint64_t>(tvci.viewType),
                        static_cast<uint64_t>(tvci.format),
                        static_cast<uint64_t>(tvci.components.r),
                        static_cast<uint64_t>(tvci.components.g),
                        static_cast<uint64_t>(tvci.components.b),
                        static_cast<uint64_t>(tvci.components.a),
                        static_cast<uint64_t>(r.is_storage_image),
                        static_cast<uint64_t>(sci.magFilter),
                        static_cast<uint64_t>(sci.minFilter),
                        static_cast<uint64_t>(sci.mipmapMode),
                        static_cast<uint64_t>(sci.addressModeU),
                        static_cast<uint64_t>(sci.addressModeV),
                        static_cast<uint64_t>(sci.addressModeW),
                        static_cast<uint64_t>(sci.borderColor),
                        static_cast<uint64_t>(float_bits(sci.minLod)),
                        static_cast<uint64_t>(float_bits(sci.maxLod)),
                        static_cast<uint64_t>(float_bits(sci.mipLodBias)),
                        static_cast<uint64_t>(sci.anisotropyEnable),
                        static_cast<uint64_t>(float_bits(sci.maxAnisotropy)),
                        static_cast<uint64_t>(r.sample_count),
                        share_backend_resources ? 0 : ++resource_unique_tag,
                    };
                    ++resource_reuse_stats.texture_binding_references;
                    size_t binding_index = SIZE_MAX;
                    auto binding_found = shared_texture_binding_indices.find(binding_key);
                    if (binding_found != shared_texture_binding_indices.end()) {
                        binding_index = binding_found->second;
                    } else {
                        const ResourcePhaseTimer phase_bind(timing_enabled, &res_texture_bind_ms);
                        binding_index = shared_texture_bindings.size();
                        SharedTextureBinding binding;
                        const bool persistent_bindings_enabled = share_backend_resources &&
                            persistent_textures_enabled &&
                            !PROSPER_ENV_ON("PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS");
                        auto persistent_image = persistent_texture_images.end();
                        if (persistent_bindings_enabled &&
                            (upload.persistent_hit || upload.persistent_refresh) &&
                            upload.persistent_id && !r.is_storage_image) {
                            persistent_image = persistent_texture_images.find(
                                {upload.persistent_id, upload.key.width, upload.key.height,
                                 upload.key.depth, upload.key.img_dim,
                                 upload.key.sample_count, upload.key.mip_levels,
                                 upload.key.format});
                        }
                        if (persistent_image != persistent_texture_images.end()) {
                            auto cached_binding = persistent_image->second.bindings.find(binding_key);
                            if (cached_binding != persistent_image->second.bindings.end()) {
                                cached_binding->second.last_use = texture_generation;
                                binding.view = cached_binding->second.view;
                                binding.sampler = cached_binding->second.sampler;
                                binding.persistent = true;
                                ++resource_reuse_stats.persistent_texture_binding_hits;
                            } else {
                                ++resource_reuse_stats.persistent_texture_binding_misses;
                                constexpr size_t max_bindings_per_texture = 32;
                                if (!avoid_cache_eviction &&
                                    persistent_image->second.bindings.size() >=
                                        max_bindings_per_texture) {
                                    auto victim = persistent_image->second.bindings.end();
                                    for (auto it = persistent_image->second.bindings.begin();
                                         it != persistent_image->second.bindings.end(); ++it) {
                                        if (it->second.last_use == texture_generation) continue;
                                        if (victim == persistent_image->second.bindings.end() ||
                                            it->second.last_use < victim->second.last_use)
                                            victim = it;
                                    }
                                    if (victim != persistent_image->second.bindings.end()) {
                                        if (victim->second.sampler)
                                            vkDestroySampler(dev, victim->second.sampler, nullptr);
                                        if (victim->second.view)
                                            vkDestroyImageView(dev, victim->second.view, nullptr);
                                        persistent_image->second.bindings.erase(victim);
                                        --persistent_texture_binding_entries;
                                        ++resource_reuse_stats.persistent_texture_binding_evictions;
                                    }
                                }
                                // #3210: both results used to be discarded, and a null
                                // view/sampler went into VkDescriptorImageInfo and on to
                                // vkUpdateDescriptorSets. Worse on THIS branch than on the
                                // transient one below: the emplace a few lines down is the only
                                // writer of PersistentTextureImage::bindings, so a null cached
                                // here was retained and re-served to every later draw with the
                                // same key. Skipping the draw before the emplace is what keeps
                                // the map free of nulls, which is why the cache-hit path above
                                // needs no guard of its own.
                                const bool persistent_binding_ready =
                                    create_render_image_view_checked(
                                        dev, tvci, RenderVkObjectCreateSite::TextureViewPersistent,
                                        &binding.view) == VK_SUCCESS &&
                                    create_render_sampler_checked(
                                        dev, sci,
                                        RenderVkObjectCreateSite::TextureSamplerPersistent,
                                        &binding.sampler) == VK_SUCCESS;
                                if (!persistent_binding_ready) {
                                    // Nothing has taken ownership yet -- the binding is not in
                                    // shared_texture_bindings and not in the persistent map -- so
                                    // whichever half succeeded is destroyed here.
                                    if (binding.sampler)
                                        vkDestroySampler(dev, binding.sampler, nullptr);
                                    if (binding.view)
                                        vkDestroyImageView(dev, binding.view, nullptr);
                                    buffer_resources_ready = false;
                                    break;
                                }
                                if (persistent_image->second.bindings.size() <
                                    max_bindings_per_texture) {
                                    const auto [entry, inserted] =
                                        persistent_image->second.bindings.emplace(
                                            binding_key, PersistentTextureBinding{
                                                binding.view, binding.sampler, texture_generation});
                                    (void)entry;
                                    if (inserted) ++persistent_texture_binding_entries;
                                    binding.persistent = true;
                                }
                            }
                        } else {
                            // #3210: same unchecked pair on the transient branch. A storage image
                            // legitimately gets NO sampler (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                            // ignores it), so only the view is unconditionally required here --
                            // treating a null sampler as a failure would drop every storage-image
                            // draw. Skip the draw the way #3045's upload failure does.
                            const bool binding_ready =
                                create_render_image_view_checked(
                                    dev, tvci, RenderVkObjectCreateSite::TextureView,
                                    &binding.view) == VK_SUCCESS &&
                                (r.is_storage_image ||
                                 create_render_sampler_checked(
                                     dev, sci, RenderVkObjectCreateSite::TextureSampler,
                                     &binding.sampler) == VK_SUCCESS);
                            if (!binding_ready) {
                                if (binding.sampler)
                                    vkDestroySampler(dev, binding.sampler, nullptr);
                                if (binding.view)
                                    vkDestroyImageView(dev, binding.view, nullptr);
                                buffer_resources_ready = false;
                                break;
                            }
                        }
                        shared_texture_bindings.push_back(binding);
                        shared_texture_binding_indices.emplace(binding_key, binding_index);
                        ++resource_reuse_stats.unique_texture_bindings;
                    }
                    const SharedTextureUpload& descriptor_upload = texture_uploads[upload_index];
                    const VkImageLayout image_layout = r.is_storage_image
                        ? VK_IMAGE_LAYOUT_GENERAL
                        : descriptor_upload.borrowed_ds_feedback
                            ? self_depth_layout
                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    dii[i] = {shared_texture_bindings[binding_index].sampler,
                              shared_texture_bindings[binding_index].view, image_layout};
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                    wr[i].descriptorType = lb[i].descriptorType; wr[i].pImageInfo = &dii[i];
                } else {
                    const ResourcePhaseTimer phase_buffer(timing_enabled, &res_buffer_ms);
                    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                    if (r.is_internal_gds) {
                        const RenderHostBuffer& gds = render_internal_gds_buffer();
                        dbi[i] = {gds.buffer, 0, 64u * 1024u};
                        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        wr[i].pBufferInfo = &dbi[i];
                        continue;
                    }
                    // One upload per descriptor this binding occupies. Uniform over entries on
                    // purpose: an earlier revision took entry 0 from `dwords` and entries 1..N-1 from
                    // `table_entries`, which leaves `table_entries[0]` counted-but-unused and is the
                    // same two-sources-of-truth defect the comment on `table_entries` warns about.
                    //
                    // With `table_entries` empty -- every producer today -- arity is 1 and this makes
                    // exactly one call with exactly the arguments the single-descriptor path always
                    // used, `r.buffer_identity` included, so the memo and dedup behaviour is unchanged.
                    const uint32_t arity = r.descriptor_arity();
                    bool entries_ready = true;
                    for (uint32_t e = 0; e < arity; e++) {
                        const uint32_t* entry_words = nullptr;
                        size_t entry_count = 0;
                        uint64_t entry_identity = 0;
                        if (r.table_entries.empty()) {
                            entry_words = r.buffer_words_data();
                            entry_count = r.buffer_word_count();
                            entry_identity = r.buffer_identity;
                        } else {
                            const std::vector<uint32_t>& entry = r.table_entries[e];
                            entry_words = entry.empty() ? nullptr : entry.data();
                            entry_count = entry.size();
                            // Identity 0 keeps each table entry conservatively distinct. Entries of one
                            // array are different guest buffers; sharing one identity across them would
                            // let the per-call memo collapse them into a single descriptor, so the
                            // shader would read entry 0 for every index.
                            entry_identity = 0;
                        }
                        size_t entry_index = SIZE_MAX;
                        // False is the fatal upload path: the zero-word fallback itself failed, so this
                        // draw cannot be bound. Same effect as the `break` this replaced.
                        if (!resolve_buffer_upload(entry_words, entry_count, entry_identity, r.set,
                                                  r.binding, entry_index)) {
                            entries_ready = false;
                            break;
                        }
                        dbi[dbi_offset[i] + e] = {shared_buffers[entry_index].buffer,
                                                  shared_buffers[entry_index].offset,
                                                  shared_buffers[entry_index].range};
                    }
                    if (!entries_ready) { buffer_resources_ready = false; break; }
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding;
                    wr[i].descriptorCount = arity;
                    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wr[i].pBufferInfo = &dbi[dbi_offset[i]];
                }
            }
            if (!buffer_resources_ready) continue;
            const ResourcePhaseTimer phase_descriptor(timing_enabled, &res_descriptor_ms);
            for (uint32_t s = 0; s < v.n_sets; s++) {
                std::vector<VkDescriptorSetLayoutBinding> slb;
                for (size_t i = 0; i < R.size(); i++) if (R[i].set == s) slb.push_back(lb[i]);
                std::sort(slb.begin(), slb.end(), [](const auto& left, const auto& right) {
                    return left.binding < right.binding;
                });
                std::vector<uint64_t> layout_key;
                layout_key.reserve(1 + slb.size() * 4);
                layout_key.push_back(share_backend_resources ? 0 : ++resource_unique_tag);
                for (const auto& binding : slb) {
                    layout_key.push_back(binding.binding);
                    layout_key.push_back(binding.descriptorType);
                    layout_key.push_back(binding.descriptorCount);
                    layout_key.push_back(binding.stageFlags);
                }
                descriptor_layout_keys[s] = layout_key;
                ++resource_reuse_stats.descriptor_set_layout_references;
                auto layout_found = shared_descriptor_set_layouts.find(layout_key);
                if (layout_found != shared_descriptor_set_layouts.end()) {
                    dsls[s] = layout_found->second;
                } else {
                    VkDescriptorSetLayoutCreateInfo layout_info{
                        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                    layout_info.bindingCount = static_cast<uint32_t>(slb.size());
                    layout_info.pBindings = slb.data();
                    vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &dsls[s]);
                    shared_descriptor_set_layouts.emplace(std::move(layout_key), dsls[s]);
                    ++resource_reuse_stats.unique_descriptor_set_layouts;
                }
            }
            VkDescriptorSetAllocateInfo allocate_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocate_info.descriptorPool = shared_descriptor_pool;
            allocate_info.descriptorSetCount = v.n_sets;
            allocate_info.pSetLayouts = dsls.data();
            const VkResult descriptor_alloc = vkAllocateDescriptorSets(dev, &allocate_info,
                                                                       v.dsets.data());
            // Read once, not per draw: getenv takes a process-wide lock on Windows and this is the
            // per-draw resource loop (#2214).
            static const bool descriptor_echo = getenv("PROSPER_BUFFER_ECHO") != nullptr;
            if (descriptor_echo) {
                std::fprintf(stderr, "[desc-echo] draw=%zu alloc=%d sets=%u", di,
                             (int)descriptor_alloc, v.n_sets);
                for (uint32_t s = 0; s < v.n_sets; ++s)
                    std::fprintf(stderr, " set%u=%p", s, (void*)v.dsets[s]);
                for (size_t i = 0; i < R.size(); i++)
                    std::fprintf(stderr, " [b%u type=%d cnt=%u buf=%p off=%llu range=%llu]",
                                 R[i].binding, (int)wr[i].descriptorType, wr[i].descriptorCount,
                                 wr[i].pBufferInfo ? (void*)wr[i].pBufferInfo->buffer : nullptr,
                                 wr[i].pBufferInfo
                                     ? (unsigned long long)wr[i].pBufferInfo->offset : 0ull,
                                 wr[i].pBufferInfo
                                     ? (unsigned long long)wr[i].pBufferInfo->range : 0ull);
                std::fprintf(stderr, "\n");
            }
            for (size_t i = 0; i < R.size(); i++)
                wr[i].dstSet = v.dsets[R[i].set];
            vkUpdateDescriptorSets(dev, static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
        }
        const auto setup_resources_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) setup_resources_ms += setup_elapsed_ms(setup_fixed_ready, setup_resources_ready);
        std::vector<uint64_t> pipeline_layout_key;
        size_t pipeline_layout_key_words = 1;
        if (v.use_desc) {
            for (const auto& descriptor_layout_key : descriptor_layout_keys)
                pipeline_layout_key_words += 1 + descriptor_layout_key.size();
        }
        pipeline_layout_key.reserve(pipeline_layout_key_words);
        pipeline_layout_key.push_back(share_backend_resources ? 0 : ++resource_unique_tag);
        if (v.use_desc) {
            for (const auto& descriptor_layout_key : descriptor_layout_keys) {
                pipeline_layout_key.push_back(descriptor_layout_key.size());
                pipeline_layout_key.insert(pipeline_layout_key.end(),
                                           descriptor_layout_key.begin(),
                                           descriptor_layout_key.end());
            }
        }
        ++resource_reuse_stats.pipeline_layout_references;
        auto pipeline_layout_found = shared_pipeline_layouts.find(pipeline_layout_key);
        if (pipeline_layout_found != shared_pipeline_layouts.end()) {
            v.layout = pipeline_layout_found->second.handle;
        } else {
            SharedPipelineLayout shared_layout;
            const bool can_persist_pipeline_layout = pipeline_layout_cache_enabled &&
                pipeline_layout_cache_limit;
            auto persistent_layout = can_persist_pipeline_layout
                ? persistent_pipeline_layouts.find(pipeline_layout_key)
                : persistent_pipeline_layouts.end();
            if (persistent_layout != persistent_pipeline_layouts.end()) {
                persistent_layout->second.last_use = pipeline_layout_generation;
                shared_layout.handle = persistent_layout->second.handle;
                shared_layout.persistent = true;
                ++resource_reuse_stats.persistent_pipeline_layout_hits;
            } else {
                VkPipelineLayoutCreateInfo layout_info{
                    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                if (v.use_desc) {
                    layout_info.setLayoutCount = v.n_sets;
                    layout_info.pSetLayouts = dsls.data();
                }
                vkCreatePipelineLayout(dev, &layout_info, nullptr, &shared_layout.handle);
                if (can_persist_pipeline_layout) {
                    ++resource_reuse_stats.persistent_pipeline_layout_misses;
                    while (!avoid_cache_eviction &&
                           persistent_pipeline_layouts.size() >= pipeline_layout_cache_limit &&
                           evict_persistent_pipeline_layout()) {}
                    if (shared_layout.handle &&
                        persistent_pipeline_layouts.size() < pipeline_layout_cache_limit) {
                        persistent_pipeline_layouts.emplace(
                            pipeline_layout_key,
                            PersistentBackendPipelineLayout{
                                shared_layout.handle, pipeline_layout_generation});
                        shared_layout.persistent = true;
                    }
                }
            }
            v.layout = shared_layout.handle;
            shared_pipeline_layouts.emplace(std::move(pipeline_layout_key), shared_layout);
            ++resource_reuse_stats.unique_pipeline_layouts;
        }
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = bd_gs.empty() ? 2u : 3u; gp.pStages = st;
        gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vpst; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb; gp.pDynamicState = &dynamic_state;
        gp.layout = v.layout; gp.renderPass = rp; gp.subpass = 0;
        if (ps && (ps->depth_test_enable || ps->stencil_enable)) gp.pDepthStencilState = &dss;
        ++pipeline_stats.references;
        bool create_pipeline = true;
        PersistentPipelineKey pipeline_key;
        if (pipeline_cache_enabled && pipeline_cache_limit) {
            auto append = [&](uint32_t word) {
                if (pipeline_key.word_count < PersistentPipelineKey::kInlineWords)
                    pipeline_key.inline_words[pipeline_key.word_count] = word;
                else
                    pipeline_key.overflow_words.push_back(word);
                ++pipeline_key.word_count;
                pipeline_key.hash ^= word;
                pipeline_key.hash *= 1099511628211ull;
            };
            auto append_float = [&](float value) {
                uint32_t word = 0;
                static_assert(sizeof word == sizeof value);
                memcpy(&word, &value, sizeof word);
                append(word);
            };
            append(12); // dynamic viewport/bias/stencil values; descriptor arity #2471; MRT formats #1390
            append(W); append(H); append(color_count);
            for (uint32_t slot = 0; slot < color_count; ++slot)
                append(static_cast<uint32_t>(color_formats[slot]));
            append(use_ds); append(static_cast<uint32_t>(DFMT));
            const bool exact_shader_identities = bd.vs_identity && bd.fs_identity;
            append(bd.vs_identity != 0);
            if (bd.vs_identity) {
                append(static_cast<uint32_t>(bd.vs_identity));
                append(static_cast<uint32_t>(bd.vs_identity >> 32));
            } else {
                append(static_cast<uint32_t>(bd_vs.size()));
                for (uint32_t word : bd_vs) append(word);
            }
            append(bd.fs_identity != 0);
            if (bd.fs_identity) {
                append(static_cast<uint32_t>(bd.fs_identity));
                append(static_cast<uint32_t>(bd.fs_identity >> 32));
            } else {
                append(static_cast<uint32_t>(bd_fs.size()));
                for (uint32_t word : bd_fs) append(word);
            }
            append(required_fragment_subgroup_size);
            append(static_cast<uint32_t>(bd_gs.size()));
            for (uint32_t word : bd_gs) append(word);
            append(v.use_desc); append(v.n_sets);
            // Descriptor ARITY is keyed on BOTH branches, and it has to be (#2471). A cached
            // VkPipeline was created under one VkPipelineLayout and is replayed under whatever
            // `v.layout` resolves to now; the two are keyed independently (`pipeline_layout_key`,
            // :5698), so any axis that reaches the layout and not this key silently pairs a
            // pipeline with an incompatible layout — VUID-vkCmdDraw*-None-08600, "descriptorCount
            // N doesn't match M", raised at the draw rather than at either cache.
            //
            // Arity is exactly such an axis, on both branches and for different reasons:
            //   * a pair of shader-cache identities names the exact compile keys, including every
            //     descriptor's class and binding, but NOT its count — a fixed vertex-fetch VS
            //     reading one binding is a byte-identical module at arity 1 or arity 3;
            //   * the fallback branch below used to hardcode `append(1)` as the descriptorCount,
            //     correct only while every binding held one descriptor.
            // So it is appended here, once, ahead of the split: two copies of the arity rule is the
            // same drift risk that put a hardcoded 1 in the fallback in the first place. `R.size()`
            // comes along because the arity list is meaningless without its length.
            append(static_cast<uint32_t>(R.size()));
            for (size_t i = 0; i < R.size(); ++i) append(R[i].descriptor_arity());
            append(!exact_shader_identities);
            if (!exact_shader_identities) {
                for (size_t i = 0; i < R.size(); ++i) {
                    const bool texture = R[i].is_texture();
                    const VkDescriptorType descriptor_type = !texture
                        ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                        : (R[i].is_storage_image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
                    const VkShaderStageFlags stage_flags = !texture || R[i].set == 0
                        ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
                    append(R[i].set); append(R[i].binding); append(descriptor_type);
                    append(stage_flags);   // count already keyed above, for both branches
                }
            }
            append(ia.topology); append(ia.primitiveRestartEnable);
            append(rs.polygonMode); append(rs.cullMode); append(rs.frontFace);
            append(rs.depthBiasEnable);
            append(cb.logicOpEnable); append(cb.logicOp);
            for (uint32_t attachment = 0; attachment < color_count; ++attachment) {
                append(cba[attachment].blendEnable); append(cba[attachment].srcColorBlendFactor);
                append(cba[attachment].dstColorBlendFactor); append(cba[attachment].colorBlendOp);
                append(cba[attachment].srcAlphaBlendFactor); append(cba[attachment].dstAlphaBlendFactor);
                append(cba[attachment].alphaBlendOp); append(cba[attachment].colorWriteMask);
            }
            append(gp.pDepthStencilState != nullptr);
            if (gp.pDepthStencilState) {
                append(dss.depthTestEnable); append(dss.depthWriteEnable); append(dss.depthCompareOp);
                append(dss.depthBoundsTestEnable); append(dss.stencilTestEnable);
                auto append_stencil = [&](const VkStencilOpState& stencil) {
                    append(stencil.failOp); append(stencil.passOp); append(stencil.depthFailOp);
                    append(stencil.compareOp);
                };
                append_stencil(dss.front); append_stencil(dss.back);
                append_float(dss.minDepthBounds); append_float(dss.maxDepthBounds);
            }
            auto found = pipeline_cache.find(pipeline_key);
            if (found != pipeline_cache.end()) {
                found->second.last_use = pipeline_generation;
                v.pipe = found->second.pipeline;
                v.pipeline_cached = true;
                v.ok = true;
                create_pipeline = false;
                ++pipeline_stats.hits;
            } else {
                ++pipeline_stats.misses;
            }
            // PROSPER_PIPEKEY_LOG (#2471): pair the graphics-pipeline cache decision with the
            // pipeline LAYOUT it will run under. The two are keyed independently — `pipeline_key`
            // here, `pipeline_layout_key` at :5698 — and nothing checks that a cache hit's stored
            // pipeline was created under a layout equivalent to `v.layout`. When they diverge the
            // symptom is VUID-vkCmdDraw*-None-08600 ("descriptorCount N doesn't match M") at draw
            // time, thousands of lines away from either cache, so the two decisions have to be
            // readable side by side. Prints the per-resource arity because arity is the axis that
            // reaches the layout without reaching this key.
            static const bool pipekey_log = getenv("PROSPER_PIPEKEY_LOG") != nullptr;
            if (pipekey_log) {
                // Unbounded on purpose: a fixed char buffer would truncate the arity list on a
                // draw with many resources, and a truncated list is exactly the case this
                // diagnostic exists to read. It allocates only when the variable is set.
                std::string arity_text = "[";
                for (size_t i = 0; i < R.size(); ++i) {
                    if (i) arity_text += ' ';
                    arity_text += std::to_string(R[i].binding);
                    arity_text += ':';
                    arity_text += std::to_string(R[i].descriptor_arity());
                }
                arity_text += ']';
                std::fprintf(stderr,
                             "[pipekey] draw=%zu key=%016llx words=%u %s exact_ids=%d "
                             "use_desc=%d n_sets=%u arity=%s layout=%p pipe=%p\n",
                             di, (unsigned long long)pipeline_key.hash, pipeline_key.word_count,
                             create_pipeline ? "MISS" : "HIT ",
                             (int)exact_shader_identities, (int)v.use_desc, v.n_sets,
                             arity_text.c_str(), (void*)v.layout, (void*)v.pipe);
                std::fflush(stderr);
            }
        } else {
            ++pipeline_stats.bypasses;
        }
        const auto setup_pipeline_key_ready = timing_enabled
            ? TimingClock::now() : TimingClock::time_point{};
        auto setup_pipeline_create_begin = setup_pipeline_key_ready;
        if (create_pipeline) {
            const auto setup_shader_begin = timing_enabled
                ? TimingClock::now() : TimingClock::time_point{};
            if (backend_trace) {
                fprintf(stderr, "[backend-trace] draw=%zu create-shaders begin\n", di);
                fflush(stderr);
            }
            v.vs = mkmod(bd_vs); v.gs = bd_gs.empty() ? VK_NULL_HANDLE : mkmod(bd_gs);
            v.fs = mkmod(bd_fs);
            if (backend_trace) {
                fprintf(stderr,
                        "[backend-trace] draw=%zu create-shaders end vs=%p gs=%p fs=%p\n",
                        di, (void*)v.vs, (void*)v.gs, (void*)v.fs);
                fflush(stderr);
            }
            const auto setup_shader_ready = timing_enabled
                ? TimingClock::now() : TimingClock::time_point{};
            if (timing_enabled)
                setup_shader_ms += setup_elapsed_ms(setup_shader_begin, setup_shader_ready);
            if (!v.vs || !v.fs || (!bd_gs.empty() && !v.gs)) {
                if (timing_enabled)
                    setup_pipeline_ms += setup_elapsed_ms(
                        setup_resources_ready, setup_pipeline_key_ready);
                continue;   // rejected SPIR-V -> skip this draw
            }
            st[0].module = v.vs;
            if (!bd_gs.empty()) st[1].module = v.gs;
            st[fragment_stage_index].module = v.fs;
            setup_pipeline_create_begin = setup_shader_ready;
            if (backend_trace) {
                fprintf(stderr, "[backend-trace] draw=%zu create-pipeline begin\n", di);
                fflush(stderr);
            }
            const VkResult pipeline_result = vkCreateGraphicsPipelines(
                    dev, VK_NULL_HANDLE, 1, &gp, nullptr, &v.pipe);
            if (backend_trace) {
                fprintf(stderr,
                        "[backend-trace] draw=%zu create-pipeline end result=%d pipeline=%p\n",
                        di, (int)pipeline_result, (void*)v.pipe);
                fflush(stderr);
            }
            if (pipeline_result == VK_SUCCESS) {
                v.ok = true;
                bool retain = pipeline_cache_enabled && pipeline_cache_limit;
                while (retain && !avoid_cache_eviction &&
                       pipeline_cache.size() >= pipeline_cache_limit)
                    if (!evict_pipeline()) retain = false;
                if (retain && pipeline_cache.size() >= pipeline_cache_limit) retain = false;
                if (retain) {
                    pipeline_cache.emplace(std::move(pipeline_key),
                                           PersistentPipeline{v.pipe, pipeline_generation});
                    v.pipeline_cached = true;
                }
            }
        }
        pipeline_stats.entries = pipeline_cache.size();
        const auto setup_pipeline_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) {
            setup_pipeline_ms += setup_elapsed_ms(setup_resources_ready, setup_pipeline_key_ready);
            setup_pipeline_ms += setup_elapsed_ms(setup_pipeline_create_begin, setup_pipeline_ready);
        }
    }

    BackendTextureUploadStats& texture_stats = backend_texture_upload_stats_storage();
    texture_stats = {};
    texture_stats.references = texture_references;
    texture_stats.persistent_hits = persistent_texture_hits;
    texture_stats.persistent_misses = persistent_texture_misses;
    for (const auto& upload : texture_uploads) {
        if (!upload.staging && !upload.uniform_clear && !upload.assembled_target_mips &&
            !upload.feedback_snapshot)
            continue;
        ++texture_stats.unique_uploads;
        texture_stats.upload_bytes += static_cast<uint64_t>(upload.key.width) * upload.key.height *
                                      upload.key.depth * upload.key.sample_count *
                                      backend_color_bytes_per_pixel(upload.key.format);
    }

    const auto timing_draws_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    // Persistent MRT0/MRT1 attachments retain their independent readback contracts. In particular,
    // merely requesting the complete MRT shape must not materialize either GPU-resident image on the
    // CPU. Slots 2+ are currently transient backend attachments, so a caller requesting them through
    // BackendMrtOutputs still needs those planes copied before the images are released.
    // #2283. Split into two questions that used to be one.
    //
    // WOULD a readback be requested -- unchanged, and it is what still drives the flush below. The
    // scope of this change is the COPY only: `readback_requested` also feeds
    // `synchronous_results_requested` and therefore `flush_now`, which gates persistent attachment
    // publication. Removing a flush is a real win (each is a queue submit plus a full CPU-GPU fence
    // wait) and it is a different change with different risk, so it is deliberately NOT taken here.
    const bool readback_color0_wanted = prosper::frontend::is_color_target_readback_wanted(
        color_target != nullptr,
        color_target ? color_target->persistent_id : 0,
        persistent_color,
        color_target ? color_target->readback : false);
    const bool readback_color1_wanted = use_color1 &&
        prosper::frontend::is_color_target_readback_wanted(
            color_target != nullptr,
            color_target ? color_target->persistent_id1 : 0,
            persistent_color1,
            color_target ? color_target->readback1 : false);
    // Slots 2+ ask per slot, exactly as slots 0 and 1 do. `color_count > 2` alone would force a
    // readback of every higher slot on every segment of a split pass, including the ones whose
    // pixels are thrown away.
    const bool readback_extra_wanted = [&] {
        for (uint32_t slot = 2; slot < color_count; ++slot) {
            const bool persistent_slot = cached_extra[slot] != nullptr;
            if (prosper::frontend::is_color_target_readback_wanted(
                    color_target != nullptr,
                    color_target ? color_target->persistent_id_slots[slot] : 0,
                    persistent_slot,
                    color_target ? color_target->readback_slots[slot] : false))
                return true;
        }
        return false;
    }();
    const bool readback_requested_for_flush =
        readback_color0_wanted || readback_color1_wanted || readback_extra_wanted;

    // ...and will one actually be PERFORMED. Blue Prince renders 457 depth-only passes with no
    // colour base per route; every one reads back a fully black surface that the frontend then
    // never looks at (`live_renderer.cpp:6082`/`:6118` consume the pixels only under `if (base ...)`,
    // and there is no else). That is up to 8 MB copied and discarded per pass.
    const bool readback_color0 = want_color_readback && readback_color0_wanted;
    const bool readback_color1 = want_color_readback && readback_color1_wanted;
    const bool readback_requested = readback_color0 || readback_color1 ||
                                    (want_color_readback && readback_extra_wanted);
    const bool storage_writeback_requested = std::any_of(
        texture_uploads.begin(), texture_uploads.end(),
        [](const SharedTextureUpload& upload) {
            return !upload.storage_writebacks.empty();
        });
    // Deliberately the WOULD-BE value, not the gated one: this change must not alter when the
    // backend flushes. Verified by the flush-reason counters, which are unchanged in an A/B.
    const bool synchronous_results_requested =
        readback_requested_for_flush || storage_writeback_requested;
    const bool flush_now = !submission_batch || synchronous_results_requested ||
                           flush_submission_batch;
    // Attributed in the same order the condition above evaluates, so exactly one bucket is charged
    // per flush and their sum is the flush count -- the arithmetic a reader will check first.
    uint64_t flush_reason_no_batch = 0, flush_reason_readback = 0;
    uint64_t flush_reason_storage = 0, flush_reason_explicit = 0;
    if (flush_now) {
        if (!submission_batch) flush_reason_no_batch = 1;
        // The WOULD-BE value: this bucket answers "why did the backend flush", and the flush is
        // still caused by a readback being wanted even when #2283 then skips performing it.
        // Attributing it to `explicit` instead would silently move counts between buckets and make
        // the flush-reason histogram lie about an unchanged synchronization path.
        else if (readback_requested_for_flush) flush_reason_readback = 1;
        else if (storage_writeback_requested) flush_reason_storage = 1;
        else flush_reason_explicit = 1;
    }
    // Sizing the readback buffer to the maximum extent over the selected slots captures the common
    // case of unbound higher MRT slots without disturbing the absolute offsets (#3276).
    readback_bytes = static_cast<VkDeviceSize>(
        prosper::frontend::compute_active_readback_bytes<prosper::gpu::kColorTargetCount>(
            color_count, color_offsets, color_bytes, [&](size_t slot) {
                if (slot == 0) return readback_color0;
                if (slot == 1) return readback_color1;
                const bool persistent_slot = cached_extra[slot] != nullptr;
                return want_color_readback &&
                    prosper::frontend::is_color_target_readback_wanted(
                        color_target != nullptr,
                        color_target ? color_target->persistent_id_slots[slot] : 0,
                        persistent_slot,
                        color_target ? color_target->readback_slots[slot] : false);
            }));

    VkBuffer rb = VK_NULL_HANDLE;
    VkDeviceMemory bmem = VK_NULL_HANDLE;
    if (readback_requested) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = readback_bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &rb);
        VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, rb, &br);
        VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; bai.allocationSize = br.size;
    // Prefer cached host memory for GPU -> CPU readback. Discrete NVIDIA exposes an earlier coherent,
    // write-combined BAR type and a later HOST_CACHED type; the generic first-match selector chose the
    // former, making an 8 MiB 1080p read take roughly 570 ms on Windows. Upload buffers deliberately keep
    // the write-combined type. Integrated GPUs and portability drivers may not expose HOST_CACHED, so fall
    // back to the original required flags.
    constexpr VkMemoryPropertyFlags host_coherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        bai.memoryTypeIndex = pick(br.memoryTypeBits, host_coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (bai.memoryTypeIndex == UINT32_MAX)
            bai.memoryTypeIndex = pick(br.memoryTypeBits, host_coherent);
        bmem = allocate_transient_render_memory(dev, bai.allocationSize, bai.memoryTypeIndex);
        vkBindBufferMemory(dev, rb, bmem, 0);
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    if (timing_enabled)
        active_submission.begin_gpu_timestamp(
            dev, cmd, ctx.timestamp_period_ns, ctx.timestamp_valid_bits);
    if (load_cached_color) {
        VkImageMemoryBarrier load{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        load.oldLayout = cached_color->layout;
        load.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        load.image = img;
        load.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // A previous target pass may be an earlier command buffer in the same queue submission.
        // Command-buffer order does not itself make its attachment writes visible, so include the
        // producer access/stage as well as the layouts in which an already-flushed target can rest.
        load.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        load.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &load);
    }
    // Slots 2..7 take the same pre-pass transition as slots 0 and 1, from the layout the retained
    // image is ACTUALLY in. Recording COLOR_ATTACHMENT_OPTIMAL as the attachment's initialLayout
    // without this barrier is undefined: the image rests in SHADER_READ_ONLY_OPTIMAL after its
    // previous group's readback restore.
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        if (!load_extra[slot]) continue;
        VkImageMemoryBarrier load{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        load.oldLayout = cached_extra[slot]->layout;
        load.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        load.image = extra_images[slot];
        load.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        load.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        load.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &load);
    }
    if (load_cached_color1) {
        VkImageMemoryBarrier load{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        load.oldLayout = cached_color1->layout;
        load.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        load.image = img1;
        load.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        load.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        load.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &load);
    }
    // Preload the color attachment with the seed pixels (render-target persistence): staging upload +
    // transition to COLOR_ATTACHMENT_OPTIMAL, matching att[0]'s LOAD/initialLayout above.
    VkBuffer seedbuf = VK_NULL_HANDLE; VkDeviceMemory seedmem = VK_NULL_HANDLE;
    if (seed_rgba) {
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &seedbuf);
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, seedbuf, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        seedmem = allocate_transient_render_memory(dev, sai.allocationSize,
                                                    sai.memoryTypeIndex);
        vkBindBufferMemory(dev, seedbuf, seedmem, 0);
        void* sp = nullptr; vkMapMemory(dev, seedmem, 0, bytes, 0, &sp);
        memcpy(sp, seed_rgba, (size_t)bytes); vkUnmapMemory(dev, seedmem);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = img; s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.srcAccessMask = 0; s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, seedbuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = img; s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    VkBuffer seedbuf1 = VK_NULL_HANDLE; VkDeviceMemory seedmem1 = VK_NULL_HANDLE;
    if (use_color1 && effective_seed1) {
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes1; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &seedbuf1);
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, seedbuf1, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        seedmem1 = allocate_transient_render_memory(dev, sai.allocationSize, sai.memoryTypeIndex);
        vkBindBufferMemory(dev, seedbuf1, seedmem1, 0);
        void* sp = nullptr; vkMapMemory(dev, seedmem1, 0, bytes1, 0, &sp);
        memcpy(sp, effective_seed1, static_cast<size_t>(bytes1));
        vkUnmapMemory(dev, seedmem1);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = extra_images[1]; s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, seedbuf1, extra_images[1],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = extra_images[1]; s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    // Slots 2..7 seed the same way slot 1 does. Split segments rely on this whenever the slots are
    // not persistent: it is the only channel by which a later physical pass inherits an earlier
    // one's pixels.
    std::array<VkBuffer, prosper::gpu::kColorTargetCount> extra_seedbufs{};
    std::array<VkDeviceMemory, prosper::gpu::kColorTargetCount> extra_seedmems{};
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        if (!color_target || !color_target->seed_slots[slot]) continue;
        const VkDeviceSize slot_bytes = color_bytes[slot];
        if (!slot_bytes) continue;
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = slot_bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &extra_seedbufs[slot]);
        if (!extra_seedbufs[slot]) continue;
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, extra_seedbufs[slot], &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        extra_seedmems[slot] =
            allocate_transient_render_memory(dev, sai.allocationSize, sai.memoryTypeIndex);
        if (!extra_seedmems[slot]) { vkDestroyBuffer(dev, extra_seedbufs[slot], nullptr);
                                     extra_seedbufs[slot] = VK_NULL_HANDLE; continue; }
        vkBindBufferMemory(dev, extra_seedbufs[slot], extra_seedmems[slot], 0);
        void* sp = nullptr; vkMapMemory(dev, extra_seedmems[slot], 0, slot_bytes, 0, &sp);
        memcpy(sp, color_target->seed_slots[slot], static_cast<size_t>(slot_bytes));
        vkUnmapMemory(dev, extra_seedmems[slot]);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = extra_images[slot];
        s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, extra_seedbufs[slot], extra_images[slot],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = extra_images[slot];
        s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    // Upload each distinct texture once. Draw descriptors may use separate views/samplers over the
    // same image, preserving per-binding swizzle and sampler state without duplicating pixel storage.
    for (const auto& upload : texture_uploads) {
        if (!upload.staging && !upload.uniform_clear && !upload.assembled_target_mips &&
            !upload.stacked_compute && !upload.feedback_snapshot)
            continue;  // exact-validated persistent image already has shader-read layout
        VkImageMemoryBarrier b0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b0.oldLayout = upload.persistent_refresh
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b0.image = upload.image;
        b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels,
                               0, upload.key.sample_count};
        b0.srcAccessMask = upload.persistent_refresh ? VK_ACCESS_SHADER_READ_BIT : 0;
        b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cmd,
            upload.persistent_refresh
                ? (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b0);
        if (upload.feedback_snapshot) {
            VkImageMemoryBarrier source_to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            source_to_copy.oldLayout = upload.feedback_source_layout;
            source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            source_to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                           VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT |
                                           VK_ACCESS_TRANSFER_WRITE_BIT;
            source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            source_to_copy.image = upload.feedback_source;
            source_to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, nullptr, 0, nullptr, 1, &source_to_copy);
            VkImageCopy copy{};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.extent = {upload.key.width, upload.key.height, 1};
            vkCmdCopyImage(
                cmd, upload.feedback_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            VkImageMemoryBarrier source_restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            source_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            source_restore.newLayout = upload.feedback_source_layout;
            source_restore.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            source_restore.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                           VK_ACCESS_SHADER_READ_BIT;
            source_restore.srcQueueFamilyIndex = source_restore.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            source_restore.image = upload.feedback_source;
            source_restore.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(
                cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &source_restore);
        } else if (upload.stacked_compute) {
            VkImageMemoryBarrier source_to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            source_to_copy.oldLayout = upload.borrowed_compute_layout;
            source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            source_to_copy.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            source_to_copy.image = upload.stacked_compute_source;
            source_to_copy.subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, upload.stacked_compute_layers};
            vkCmdPipelineBarrier(
                cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &source_to_copy);
            const uint32_t face_height = upload.stacked_compute_layers
                ? upload.key.height / upload.stacked_compute_layers : 0u;
            std::vector<VkImageCopy> regions(upload.stacked_compute_layers);
            for (uint32_t layer = 0; layer < upload.stacked_compute_layers; ++layer) {
                VkImageCopy& copy = regions[layer];
                copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
                copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.dstOffset = {0, static_cast<int32_t>(layer * face_height), 0};
                copy.extent = {upload.key.width, face_height, 1};
            }
            vkCmdCopyImage(
                cmd, upload.stacked_compute_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<uint32_t>(regions.size()), regions.data());
            VkImageMemoryBarrier source_restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            source_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            source_restore.newLayout = upload.borrowed_compute_layout;
            source_restore.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            source_restore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            source_restore.srcQueueFamilyIndex = source_restore.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            source_restore.image = upload.stacked_compute_source;
            source_restore.subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, upload.stacked_compute_layers};
            vkCmdPipelineBarrier(
                cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &source_restore);
        } else if (upload.uniform_clear) {
            const VkImageSubresourceRange range{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels,
                0, upload.key.sample_count};
            vkCmdClearColorImage(cmd, upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &upload.uniform_color, 1, &range);
        } else if (!upload.assembled_target_mips) {
            VkBufferImageCopy tc{};
            tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   upload.key.sample_count};
            tc.imageExtent = {upload.key.width, upload.key.height, upload.key.depth};
            vkCmdCopyBufferToImage(cmd, upload.staging, upload.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tc);
        }
        if (upload.assembled_target_mips) {
            // Each source is an independent one-level persistent color target resting in its
            // recorded cache layout. Zero the destination first, then copy only levels the guest
            // actually rendered and restore every source immediately. A missing level must remain
            // unavailable/black: deriving it from a neighbour invents guest output and amplified
            // GTA V's incomplete bloom chain into a full-screen glare.
            const VkClearColorValue missing_level_clear{};
            const VkImageSubresourceRange missing_level_range{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels, 0, 1};
            vkCmdClearColorImage(cmd, upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &missing_level_clear, 1, &missing_level_range);
            // #3248: the clear and the per-level copies below both WRITE this image, and every
            // barrier inside the loop names the copy SOURCE. Ordering the two writes needs a memory
            // dependency on the DESTINATION -- see record_transfer_write_after_write_barrier for why
            // the resulting corruption is invisible in the output.
            record_transfer_write_after_write_barrier(cmd, upload.image, missing_level_range);
            for (uint32_t level = 0; level < upload.key.mip_levels; ++level) {
                const uint32_t level_w = std::max(upload.key.width >> level, 1u);
                const uint32_t level_h = std::max(upload.key.height >> level, 1u);
                if (upload.target_mip_images[level]) {
                    VkImageMemoryBarrier source_to_copy{
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    source_to_copy.oldLayout = upload.target_mip_layouts[level];
                    source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    source_to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                                   VK_ACCESS_SHADER_READ_BIT |
                                                   VK_ACCESS_TRANSFER_READ_BIT;
                    source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                        VK_QUEUE_FAMILY_IGNORED;
                    source_to_copy.image = upload.target_mip_images[level];
                    source_to_copy.subresourceRange = {
                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, nullptr, 0, nullptr, 1, &source_to_copy);
                    VkImageCopy copy{};
                    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.extent = {level_w, level_h, 1};
                    vkCmdCopyImage(
                        cmd, upload.target_mip_images[level],
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, upload.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                    VkImageMemoryBarrier source_restore{
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    source_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    source_restore.newLayout = upload.target_mip_layouts[level];
                    source_restore.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    source_restore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                                   VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    source_restore.srcQueueFamilyIndex = source_restore.dstQueueFamilyIndex =
                        VK_QUEUE_FAMILY_IGNORED;
                    source_restore.image = upload.target_mip_images[level];
                    source_restore.subresourceRange = {
                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    vkCmdPipelineBarrier(
                        cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &source_restore);
                }
            }
        }
        // #1272: generate levels 1..N-1 with a linear-filtered blit cascade (GPU-side, once per
        // upload — a CPU box filter here collapsed titles that re-upload large textures per frame).
        // Each source level transitions DST->SRC before feeding the next; the final barrier below
        // then flips the whole chain to shader-read. RGBA8 linear-blit support is mandatory Vulkan.
        for (uint32_t l = 1; !upload.uniform_clear && !upload.assembled_target_mips &&
             !upload.stacked_compute && !upload.feedback_snapshot &&
             l < upload.key.mip_levels; l++) {
            VkImageMemoryBarrier bs{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            bs.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            bs.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bs.image = upload.image;
            bs.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 1,
                                   0, upload.key.sample_count};
            bs.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bs.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &bs);
            const int32_t sw = (int32_t)(upload.key.width >> (l - 1) ? upload.key.width >> (l - 1) : 1u);
            const int32_t sh = (int32_t)(upload.key.height >> (l - 1) ? upload.key.height >> (l - 1) : 1u);
            const int32_t dw = (int32_t)(upload.key.width >> l ? upload.key.width >> l : 1u);
            const int32_t dh = (int32_t)(upload.key.height >> l ? upload.key.height >> l : 1u);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0,
                                   upload.key.sample_count};
            blit.srcOffsets[1] = {sw, sh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0,
                                   upload.key.sample_count};
            blit.dstOffsets[1] = {dw, dh, 1};
            vkCmdBlitImage(cmd, upload.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                           VK_FILTER_LINEAR);
        }
        if (!upload.uniform_clear && !upload.assembled_target_mips &&
            !upload.stacked_compute && !upload.feedback_snapshot &&
            upload.key.mip_levels > 1) {
            // Levels 0..N-2 sit in TRANSFER_SRC after feeding the cascade; return them to
            // TRANSFER_DST so the single final-layout barrier below covers the whole chain.
            VkImageMemoryBarrier br{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            br.image = upload.image;
            br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   upload.key.mip_levels - 1, 0,
                                   upload.key.sample_count};
            br.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &br);
        }
        VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b1.newLayout = upload.key.storage_image
            ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b1.image = upload.image;
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels,
                               0, upload.key.sample_count};
        b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            (upload.key.storage_image ? VK_ACCESS_SHADER_WRITE_BIT : 0);
        // The dst stage must cover EVERY stage that samples this image. #376 made set-0 textures
        // VS-visible (stageFlags VERTEX|FRAGMENT), so a vertex texture fetch reads it in the VERTEX
        // stage — a FRAGMENT-only barrier leaves the transfer-write→shader-read dependency unordered
        // for that stage (SYNC-HAZARD-READ-AFTER-WRITE; garbage vertex fetch on GPUs that don't
        // over-synchronize). Include the vertex stage to match the binding's stageFlags (#454).
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b1);
    }
    // Compute-owned typed storage results rest in GENERAL. Borrow each exact image once per call,
    // make the completed compute/transfer writes visible to graphics sampling, and restore GENERAL
    // after the pass so the compute cache's layout contract remains true.
    std::vector<std::pair<VkImage, VkImageLayout>> borrowed_compute_images;
    for (const SharedTextureUpload& upload : texture_uploads) {
        if (!upload.borrowed_compute || !upload.image) continue;
        if (std::any_of(borrowed_compute_images.begin(), borrowed_compute_images.end(),
                        [&](const auto& entry) { return entry.first == upload.image; }))
            continue;
        borrowed_compute_images.push_back({upload.image, upload.borrowed_compute_layout});
        VkImageMemoryBarrier to_sampled{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_sampled.oldLayout = upload.borrowed_compute_layout;
        to_sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sampled.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sampled.srcQueueFamilyIndex = to_sampled.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        to_sampled.image = upload.image;
        to_sampled.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_sampled);
    }
    // Same-pass depth feedback (#1186): make prior attachment writes visible to the shader and
    // enter the layout declared by both the sampled descriptor and this render pass. The render
    // pass returns the image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL at the end, preserving the cache
    // contract for every later call. Include both aspects in the transition for D32S8; the mixed
    // layout keeps depth read-only while allowing the stencil writes detected above.
    if (self_sampled_depth) {
        VkImageMemoryBarrier to_feedback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_feedback.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_feedback.newLayout = self_depth_layout;
        to_feedback.image = dimg;
        to_feedback.subresourceRange = {DASPECT, 0, 1, 0, 1};
        to_feedback.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        to_feedback.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    (stencil_may_be_written
                                         ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0u);
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_feedback);
    }
    // Sampled depth bridge (#1275): borrowed DS images live in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    // between passes. Transition each (once) to SHADER_READ_ONLY for this pass's sampling; the
    // matching post-pass barrier below returns it so the next depth pass LOADs it unchanged.
    // Both aspects of a combined image must transition together.
    std::vector<std::pair<VkImage, VkFormat>> borrowed_ds_images;
    for (const SharedTextureUpload& upload : texture_uploads) {
        if (!upload.borrowed_ds || upload.borrowed_ds_feedback || !upload.image) continue;
        if (std::any_of(borrowed_ds_images.begin(), borrowed_ds_images.end(),
                        [&](const auto& entry) { return entry.first == upload.image; }))
            continue;
        borrowed_ds_images.push_back({upload.image, upload.ds_format});
        const bool ds_has_stencil = upload.ds_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkImageMemoryBarrier to_sampled{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_sampled.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sampled.image = upload.image;
        to_sampled.subresourceRange = {
            VK_IMAGE_ASPECT_DEPTH_BIT |
                (ds_has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u), 0, 1, 0, 1};
        to_sampled.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_sampled);
    }
    // Clear color: the caller's clear_rgba (game fast-clear / black on the live path), else the
    // legacy diagnostic blue. PROSPER_CLEAR_DEBUG forces blue back on even when a color is passed.
    float cc[4] = {0.0f, 0.0f, 1.0f, 1.0f};   // diagnostic blue
    if (clear_rgba && getenv("PROSPER_CLEAR_DEBUG") == nullptr)
        for (int i = 0; i < 4; i++) cc[i] = clear_rgba[i];
    float cc1[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    if (clear_rgba1 && getenv("PROSPER_CLEAR_DEBUG") == nullptr)
        for (int i = 0; i < 4; ++i) cc1[i] = clear_rgba1[i];
    std::array<VkClearValue, prosper::gpu::kColorTargetCount + 1> clear{};
    clear[0].color = {{cc[0], cc[1], cc[2], cc[3]}};
    if (use_color1) clear[1].color = {{cc1[0], cc1[1], cc1[2], cc1[3]}};
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        float value[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        for (const auto& draw : logical_draws) {
            if (!draw.ps || !draw.ps->color_targets[slot].has_clear) continue;
            std::copy(std::begin(draw.ps->color_targets[slot].clear),
                      std::end(draw.ps->color_targets[slot].clear), value);
            break;
        }
        clear[slot].color = {{value[0], value[1], value[2], value[3]}};
    }
    clear[ds_attachment].depthStencil = {depth_clear, stencil_clear}; // guest DB clear/default
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}};
    rpbi.clearValueCount = color_count + (use_ds ? 1u : 0u);
    rpbi.pClearValues = clear.data();
    if (PROSPER_ENV_ON("PROSPER_PIPELOG")) {   // diag: how many draws' pipelines built + will be recorded
        int nok = 0; for (auto& v : dv) if (v.ok) nok++;
        fprintf(stderr, "[pipe] %zu draws, %d pipelines OK, use_depth=%d use_stencil=%d; counts:", dv.size(), nok, (int)use_depth, (int)use_stencil);
        for (auto& v : dv) fprintf(stderr, " %s%u", v.ok ? "" : "SKIP", v.icount ? v.icount : v.vcount);
        fprintf(stderr, "\n");
    }
    // Per-draw "fragment funnel" (PROSPER_DRAW_STATS): wrap each recorded draw in pipeline-statistics
    // + occlusion queries to show WHERE its pixels vanish (geometry clipped away, never rasterized,
    // depth/stencil-rejected, or survived) — objective per-draw truth, no oracle needed. Read-only, and
    // only active with the env var AND a real flush in THIS call (so the results are ready to read back).
    // Query-pool RESET must be recorded outside a render pass, so it happens here.
    const RenderVkCtx& ds_ctx = render_vk_ctx();
    const bool draw_stats = PROSPER_ENV_ON("PROSPER_DRAW_STATS") && ds_ctx.pipeline_stats_enabled &&
                            !dv.empty() && flush_now;
    VkQueryPool ds_stats_pool = VK_NULL_HANDLE, ds_occ_pool = VK_NULL_HANDLE;
    if (draw_stats) {
        const uint32_t nq = static_cast<uint32_t>(dv.size());
        VkQueryPoolCreateInfo sp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        sp.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS; sp.queryCount = nq;
        sp.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        VkQueryPoolCreateInfo op{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        op.queryType = VK_QUERY_TYPE_OCCLUSION; op.queryCount = nq;
        if (vkCreateQueryPool(dev, &sp, nullptr, &ds_stats_pool) == VK_SUCCESS &&
            vkCreateQueryPool(dev, &op, nullptr, &ds_occ_pool) == VK_SUCCESS) {
            vkCmdResetQueryPool(cmd, ds_stats_pool, 0, nq);
            vkCmdResetQueryPool(cmd, ds_occ_pool, 0, nq);
        } else {
            if (ds_stats_pool) { vkDestroyQueryPool(dev, ds_stats_pool, nullptr); ds_stats_pool = VK_NULL_HANDLE; }
            if (ds_occ_pool)   { vkDestroyQueryPool(dev, ds_occ_pool,   nullptr); ds_occ_pool   = VK_NULL_HANDLE; }
        }
    }
    const bool ds_active = ds_stats_pool != VK_NULL_HANDLE && ds_occ_pool != VK_NULL_HANDLE;

    // Geometry probe (PROSPER_GEOM_PROBE=N): capture draw N's post-transform clip-space vertices via
    // transform feedback and report where they land (degenerate / off-screen / behind-camera / NaN).
    // Requires VK_EXT_transform_feedback + the last pre-rasterization shader xfb-decorated (gated on
    // the same env var in gpu_executor). Only active with the env var, TF support, AND a flush here.
    const char* geom_env = PROSPER_ENV_VALUE("PROSPER_GEOM_PROBE");
    uint64_t geom_target = 0;
    const bool geom_target_valid = geom_env && gpu::parse_diagnostic_draw_id(geom_env, geom_target);
    const auto geom_target_label = static_cast<unsigned long long>(geom_target);
    size_t geom_item = SIZE_MAX;
    if (geom_target_valid) {
        for (size_t i = 0; i < draws.size(); ++i)
            if (draws[i].draw_index == geom_target) {
                geom_item = i;
                break;
            }
        // Direct backend tests and the single-draw wrapper have no semantic DrawItem. Preserve their
        // positional diagnostic behavior only when the caller supplied no semantic IDs at all.
        if (geom_item == SIZE_MAX &&
            std::none_of(draws.begin(), draws.end(), [](const BackendDraw& draw) {
                return draw.draw_index != UINT64_MAX;
            }) && geom_target < draws.size())
            geom_item = static_cast<size_t>(geom_target);
    }
    // #3248: arm only when the module the backend is about to hand Vulkan actually declares the
    // capture. Transform feedback records nothing without it, and the probe used to report that
    // silence as "the draw produced no primitives" -- a wrong answer, not a missing one. Every
    // refusal below says which precondition failed, so the instrument never returns silently.
    const bool geom_xfb_declared = geom_item != SIZE_MAX && [&] {
        // The last pre-rasterization stage is the generated geometry stage when there is one, and
        // the vertex stage otherwise -- transform feedback captures that stage and no other.
        const BackendDraw& probed = draws[geom_item];
        return gpu::spirv_declares_xfb_capture(
            probed.gs_words().empty() ? probed.vs_words() : probed.gs_words());
    }();
    if (geom_item != SIZE_MAX && flush_now) {
        // Both reasons are reported independently, and the undeclared one is COUNTED independently
        // of transform-feedback support: it is a property of the module, so a device without the
        // extension must not make the check look like it passed.
        // Printed once per process per reason -- a live title flushes many times a second and the
        // condition cannot change between them, so repeating it would bury the run's real output.
        // The COUNTERS are exact regardless; they are what the test reads.
        static std::atomic<bool> said_undeclared{false}, said_no_extension{false};
        if (!geom_xfb_declared) {
            backend_geom_probe_undeclared_count().fetch_add(1, std::memory_order_relaxed);
            if (!said_undeclared.exchange(true, std::memory_order_relaxed))
                fprintf(stderr,
                        "[geom-probe] draw=%llu: REFUSING to arm -- the last pre-rasterization "
                        "stage does not declare OpExecutionMode Xfb, so transform feedback would "
                        "capture nothing and the result would read as 'no primitives'. The shader "
                        "must be recompiled with PROSPER_GEOM_PROBE set (and the draw must not take "
                        "a path that skips the decoration). Reported once per run.\n",
                        geom_target_label);
        }
        if (!ds_ctx.transform_feedback_enabled &&
            !said_no_extension.exchange(true, std::memory_order_relaxed))
            fprintf(stderr, "[geom-probe] draw=%llu: REFUSING to arm -- this device has no "
                            "VK_EXT_transform_feedback, so nothing can be captured. "
                            "Reported once per run.\n",
                    geom_target_label);
    }
    const bool geom_probe = geom_item != SIZE_MAX && geom_xfb_declared &&
                            ds_ctx.transform_feedback_enabled && flush_now;
    static auto p_bindxfb  = reinterpret_cast<PFN_vkCmdBindTransformFeedbackBuffersEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT"));
    static auto p_beginxfb = reinterpret_cast<PFN_vkCmdBeginTransformFeedbackEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT"));
    static auto p_endxfb   = reinterpret_cast<PFN_vkCmdEndTransformFeedbackEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT"));
    VkBuffer geom_buf = VK_NULL_HANDLE; VkDeviceMemory geom_mem = VK_NULL_HANDLE;
    VkBuffer geom_counter = VK_NULL_HANDLE; VkDeviceMemory geom_counter_mem = VK_NULL_HANDLE;
    uint32_t geom_cap = 0;   // buffer capacity in vertices; the counter buffer gives the exact count written
    if (geom_probe && p_bindxfb && p_beginxfb && p_endxfb && dv[geom_item].ok) {
        const auto& tv = dv[geom_item];
        const uint64_t per_inst = tv.icount ? tv.icount : tv.vcount;
        // Transform feedback records DECOMPOSED primitives: a triangle strip/fan of N verts emits up to
        // ~3*(N-2) individual vertices. Over-size by 3x so no records are dropped; the counter buffer
        // reports how many were actually written so we never read the uninitialized tail.
        uint64_t total = per_inst * (tv.instance_count ? tv.instance_count : 1u) * 3u;
        if (total > (1u << 20)) total = (1u << 20);   // cap at 1M vertices (16 MiB)
        geom_cap = static_cast<uint32_t>(total);
        std::string gerr;
        if (geom_cap == 0 ||
            !persistent_ds_transfer_buffer(ds_ctx, static_cast<VkDeviceSize>(geom_cap) * 16,
                                           VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
                                           geom_buf, geom_mem, gerr) ||
            !persistent_ds_transfer_buffer(ds_ctx, 16,
                                           VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
                                           geom_counter, geom_counter_mem, gerr)) {
            if (geom_buf) { vkDestroyBuffer(dev, geom_buf, nullptr); vkFreeMemory(dev, geom_mem, nullptr); }
            if (geom_counter) { vkDestroyBuffer(dev, geom_counter, nullptr); vkFreeMemory(dev, geom_counter_mem, nullptr); }
            geom_buf = VK_NULL_HANDLE; geom_mem = VK_NULL_HANDLE;
            geom_counter = VK_NULL_HANDLE; geom_counter_mem = VK_NULL_HANDLE; geom_cap = 0;
        }
    }
    const bool geom_active = geom_buf != VK_NULL_HANDLE && geom_counter != VK_NULL_HANDLE;
    if (geom_active) backend_geom_probe_armed_count().fetch_add(1, std::memory_order_relaxed);

    // ONE render pass (cleared once): record every realized draw with its own pipeline + descriptors.
    backend_pass_timing_begin(dev, cmd, ctx.timestamp_period_ns, ctx.timestamp_valid_bits,
                              rpbi.renderArea.extent.width, rpbi.renderArea.extent.height,
                              color_target ? color_target->persistent_id : 0, dv.size(),
                              [&] {   // #2283: colour-writing draws, counted not inferred
                                  // Gate-checked HERE, not only inside the callee. Arguments are
                                  // evaluated before the call, so without this the loop ran on every
                                  // pass of every DEFAULT run -- an O(draws) scan added to the hot
                                  // path by a diagnostic that was switched off. Exactly the cost
                                  // this instrument exists to find.
                                  if (!backend_pass_timing_enabled()) return size_t{0};
                                  size_t n = 0;
                                  for (size_t i = 0; i < dv.size() && i < draws.size(); ++i)
                                      if (draws[i].ps && draws[i].ps->color_write_mask) ++n;
                                  return n;
                              }());   // #2333
    // Every dynamic state these pipelines declare, set once per draw. It is a lambda rather than
    // eleven inline calls because the PROSPER_DRAW_ISO re-render below replays the SAME pipelines
    // into a second command buffer, and it set NONE of this (#3248): the isolation pass therefore
    // ran under undefined viewport, scissor, line width, depth bias and stencil state while
    // reporting which draw paints a pixel -- a diagnostic answering about a render that is not the
    // one it is isolating. The validation layer says so directly, VUID-vkCmdDraw-None-07831 and
    // -07832, but no ctest exercised the path so nothing ever saw it. One recorder, both loops.
    //
    // WHAT IS STILL DIFFERENT, because a diagnostic described as fixed is one nobody re-checks: the
    // isolation loop does NOT replay the per-draw vkCmdClearAttachments depth/stencil clear a few
    // lines below. A submit whose draws carry a guest depth or stencil clear is therefore still
    // isolated against different depth contents from the pass it is naming a culprit in. Dynamic
    // state was the part the layer could see; this part it cannot, and it is not fixed here.
    auto record_draw_dynamic_state = [](VkCommandBuffer command, const DV& v) {
        vkCmdSetViewport(command, 0, 1, &v.viewport);
        vkCmdSetScissor(command, 0, 1, &v.scissor);
        vkCmdSetLineWidth(command, v.line_width);
        vkCmdSetDepthBias(command, v.depth_bias_constant, v.depth_bias_clamp,
                          v.depth_bias_slope);
        vkCmdSetStencilCompareMask(command, VK_STENCIL_FACE_FRONT_BIT,
                                   v.stencil_front.compareMask);
        vkCmdSetStencilCompareMask(command, VK_STENCIL_FACE_BACK_BIT,
                                   v.stencil_back.compareMask);
        vkCmdSetStencilWriteMask(command, VK_STENCIL_FACE_FRONT_BIT,
                                 v.stencil_front.writeMask);
        vkCmdSetStencilWriteMask(command, VK_STENCIL_FACE_BACK_BIT,
                                 v.stencil_back.writeMask);
        vkCmdSetStencilReference(command, VK_STENCIL_FACE_FRONT_BIT,
                                 v.stencil_front.reference);
        vkCmdSetStencilReference(command, VK_STENCIL_FACE_BACK_BIT,
                                 v.stencil_back.reference);
    };
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    for (size_t di = 0; di < dv.size(); di++) {
        auto& v = dv[di];
        const auto* ps = draws[di].ps;
        if (use_ds && ps &&
            (effective_depth_clear(ps) ||
             stencil_clear_effective(ps->stencil_clear_enable, ps->stencil_enable,
                                     ps->stencil_write_mask[0], ps->stencil_write_mask[1]))) {
            VkClearAttachment dsc{};
            if (effective_depth_clear(ps)) dsc.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (stencil_clear_effective(ps->stencil_clear_enable, ps->stencil_enable,
                                        ps->stencil_write_mask[0], ps->stencil_write_mask[1]) &&
                format_has_stencil)
                dsc.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            dsc.clearValue.depthStencil = {ps->depth_clear_value, ps->stencil_clear_value};
            VkClearRect rect{v.scissor, 0, 1};
            // A fully clipped draw legitimately has a zero-area dynamic scissor, but Vulkan requires
            // vkCmdClearAttachments rectangles to have non-zero width and height (VUID 02682/02683).
            if (dsc.aspectMask && rect.rect.extent.width && rect.rect.extent.height)
                vkCmdClearAttachments(cmd, 1, &dsc, 1, &rect);
        }
        if (!v.ok) continue;
        if (ds_active) {
            vkCmdBeginQuery(cmd, ds_stats_pool, static_cast<uint32_t>(di), 0);
            vkCmdBeginQuery(cmd, ds_occ_pool, static_cast<uint32_t>(di),
                            ds_ctx.occlusion_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
        record_draw_dynamic_state(cmd, v);
        if (v.use_desc) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
        const bool geom_here = geom_active && di == geom_item;
        if (geom_here) {
            VkDeviceSize off = 0, sz = static_cast<VkDeviceSize>(geom_cap) * 16;
            p_bindxfb(cmd, 0, 1, &geom_buf, &off, &sz);
            p_beginxfb(cmd, 0, 0, nullptr, nullptr);
        }
        // PROSPER_INDEX_ECHO=1 (investigation instrument, #3374): the index bytes at exactly the
        // (buffer, offset) pair about to be bound, against the indices prosper decoded for this draw,
        // plus the draw parameters that decide the fetched range. It answers "is this draw about to
        // ask for vertices outside its own vertex buffer" -- a draw that consumed another draw's index
        // slice, or whose vertexOffset pushes gl_VertexIndex past the bound range, makes every vertex
        // load return 0 under robustness, which is indistinguishable at the pixel from wrong vertex
        // data and produces no Vulkan validation message either.
        //
        // SCOPE, stated narrowly. This reads the HOST mapping of the slice, not device memory, so a
        // clean result retires "prosper computed or placed the wrong indices" and says nothing about
        // what the GPU read back. `mismatched` compares the arena against the same vector the memcpy
        // sourced from, so it cannot express a wrong DECODE either -- only a clobber between the copy
        // and recording. The device-side question belongs to PROSPER_BUFFER_ECHO, which copies index
        // slices back through the GPU; note that its `echo_count` starts at min(16, shared_buffers)
        // and its index loop runs only while that is below 16, so at >=16 storage uploads -- which is
        // every real frame; 346 of them here -- it echoes zero index slices (#3376). That gap is why
        // this host-side reading was needed at all.
        static const bool index_echo = getenv("PROSPER_INDEX_ECHO") != nullptr;
        if (index_echo && v.icount) {
            const std::vector<uint32_t>& want = draws[di].indices;
            uint32_t want_max = 0;
            for (uint32_t x : want) want_max = std::max(want_max, x);
            std::fprintf(stderr,
                         "[index-echo] draw=%zu icount=%u vcount=%u voff=%d inst=%u arena=%d "
                         "ibuf=%p ioff=%llu want_n=%zu want_max=%u",
                         di, v.icount, v.vcount, v.vertex_offset, v.instance_count, (int)v.iarena,
                         (void*)v.ibuf, (unsigned long long)v.ioffset, want.size(), want_max);
            if (v.imapped) {
                const uint32_t* got = static_cast<const uint32_t*>(v.imapped);
                uint32_t got_max = 0, mismatched = 0;
                for (uint32_t k = 0; k < v.icount; k++) {
                    got_max = std::max(got_max, got[k]);
                    if (k < want.size() && got[k] != want[k]) ++mismatched;
                }
                std::fprintf(stderr, " got_max=%u mismatched=%u got[0..7]=", got_max, mismatched);
                for (uint32_t k = 0; k < v.icount && k < 8; k++)
                    std::fprintf(stderr, "%s%u", k ? "," : "", got[k]);
            } else {
                std::fprintf(stderr, " got=<dedicated buffer, unmapped>");
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        if (v.icount) {
            vkCmdBindIndexBuffer(cmd, v.ibuf, v.ioffset, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, v.icount, v.instance_count, 0, v.vertex_offset, 0);
        } else {
            vkCmdDraw(cmd, v.vcount, v.instance_count,
                      static_cast<uint32_t>(v.vertex_offset), 0);
        }
        if (geom_here) { VkDeviceSize coff = 0; p_endxfb(cmd, 0, 1, &geom_counter, &coff); }
        if (ds_active) {
            vkCmdEndQuery(cmd, ds_occ_pool, static_cast<uint32_t>(di));
            vkCmdEndQuery(cmd, ds_stats_pool, static_cast<uint32_t>(di));
        }
    }
    vkCmdEndRenderPass(cmd);
    // #2944: the geometry probe maps both transform-feedback buffers below. Transform feedback does
    // not write through the transfer stage, so this pair carries its own source scope -- the vertex
    // records and the counter the extension writes at vkCmdEndTransformFeedbackEXT. Recorded here
    // rather than beside p_endxfb because a buffer memory barrier is not permitted inside a render
    // pass instance.
    if (geom_active) {
        record_host_read_barrier(cmd, geom_buf, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                 VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT);
        record_host_read_barrier(cmd, geom_counter, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                 VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT);
    }
    // PROSPER_BUFFER_ECHO=1 (#2945) -- ground truth for "what did the GPU see in this draw's
    // storage buffers". Every other instrument reads the HOST side: PROSPER_BUFLOG prints the source
    // words, --dump-resource prints the capture's bytes, and both were byte-identical across runs
    // that rendered and runs whose vertices collapsed. This copies the bound slices back THROUGH THE
    // GPU, in the same command buffer as the draw and immediately after the render pass ends, so
    // the values printed are the ones the shader's descriptors resolved to. It has to be after
    // vkCmdEndRenderPass: vkCmdCopyBuffer is a transfer command and is not permitted inside a
    // render pass instance. Opt-in and bounded: first 16 slices, 64 bytes each.
    //
    // Gated on `flush_now` as well, and that gate is load-bearing rather than tidy: teardown below
    // is unconditional, while the print is not, so on the ordinary BATCHED path (flush_now false)
    // the copies would be recorded into `cmd`, the echo buffer destroyed immediately, and the batch
    // would then submit a command buffer referencing a dead VkBuffer -- a use-after-free that also
    // printed nothing, so the diagnostic would have been both unsafe and silent exactly where it
    // was least likely to be noticed. Arming only on a pass that submits its own work keeps the
    // buffer's lifetime inside the fence this function waits on.
    static const bool buffer_echo_requested = getenv("PROSPER_BUFFER_ECHO") != nullptr;
    const bool buffer_echo = buffer_echo_requested && flush_now;
    if (buffer_echo_requested && !flush_now) {
        static std::atomic<int> deferred_notice{0};
        if (deferred_notice.fetch_add(1) == 0)
            std::fprintf(stderr, "[buffer-echo] this pass defers its submission to a later batch; "
                                 "the echo only arms on a pass that submits its own work\n");
    }
    VkBuffer echo_buffer = VK_NULL_HANDLE;
    VkDeviceMemory echo_memory = VK_NULL_HANDLE;
    void* echo_mapped = nullptr;
    size_t echo_count = 0;
    std::vector<size_t> echo_index_slice;
    bool echo_ready = false;
    constexpr VkDeviceSize kEchoStride = 64;
    constexpr size_t kEchoMax = 16;
    if (buffer_echo && !shared_buffers.empty()) {
        VkBufferCreateInfo eci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        eci.size = kEchoStride * kEchoMax;
        eci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryRequirements er{};
        if (vkCreateBuffer(dev, &eci, nullptr, &echo_buffer) == VK_SUCCESS) {
            vkGetBufferMemoryRequirements(dev, echo_buffer, &er);
            VkMemoryAllocateInfo eai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            eai.allocationSize = er.size;
            eai.memoryTypeIndex = render_memory_type(
                ctx.phys, er.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (eai.memoryTypeIndex != UINT32_MAX &&
                vkAllocateMemory(dev, &eai, nullptr, &echo_memory) == VK_SUCCESS &&
                vkBindBufferMemory(dev, echo_buffer, echo_memory, 0) == VK_SUCCESS &&
                vkMapMemory(dev, echo_memory, 0, VK_WHOLE_SIZE, 0, &echo_mapped) == VK_SUCCESS) {
                std::memset(echo_mapped, 0xCD, static_cast<size_t>(kEchoStride * kEchoMax));
                echo_ready = true;
            }
        }
    }
    if (echo_ready) {
        echo_count = std::min(kEchoMax, shared_buffers.size());
        for (size_t i = 0; i < echo_count; ++i) {
            VkBufferCopy copy{};
            copy.srcOffset = shared_buffers[i].offset;
            copy.dstOffset = kEchoStride * i;
            copy.size = std::min<VkDeviceSize>(kEchoStride, shared_buffers[i].range);
            if (!copy.size) continue;
            vkCmdCopyBuffer(cmd, shared_buffers[i].buffer, echo_buffer, 1, &copy);
        }
        // ...and the INDEX buffers, which live outside `shared_buffers`. A draw whose
        // gl_VertexIndex is out of range makes every vertex-buffer load return 0 under
        // robustness, which is indistinguishable at the pixel from "the vertex data is
        // wrong" -- so the index bytes have to be read back from the GPU too.
        for (size_t d = 0; d < dv.size() && echo_count < kEchoMax; ++d) {
            if (!dv[d].ibuf || !dv[d].icount) continue;
            VkBufferCopy copy{};
            copy.srcOffset = dv[d].ioffset;
            copy.dstOffset = kEchoStride * echo_count;
            copy.size = std::min<VkDeviceSize>(kEchoStride,
                                               VkDeviceSize(dv[d].icount) * 4);
            echo_index_slice.push_back(echo_count);
            ++echo_count;
            vkCmdCopyBuffer(cmd, dv[d].ibuf, echo_buffer, 1, &copy);
        }
        record_host_read_barrier(cmd, echo_buffer);   // #2944
    }
    backend_pass_timing_end(cmd);   // #2333
    // Fence waits used to provide the device-memory dependency between every target call. Batched
    // command buffers deliberately remove those intermediate waits, and command-buffer/submission
    // order alone permits action commands to overlap. Publish persistent attachment writes here so
    // any later command buffer in the queue can sample or LOAD them without relying on driver-wide
    // serialization. The final render-pass layouts are already correct; these same-layout barriers
    // supply the missing availability/visibility dependency.
    auto publish_persistent_color = [&](bool persistent, bool readback, VkImage image) {
        if (!persistent || readback) return;
        VkImageMemoryBarrier color_ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        color_ready.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        color_ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        color_ready.image = image;
        color_ready.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        color_ready.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        color_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &color_ready);
    };
    publish_persistent_color(persistent_color, readback_color0, img);
    publish_persistent_color(persistent_color1, readback_color1, img1);
    // Retained higher slots need the same availability/visibility barrier when no readback copy
    // performed it. Keyed on whether THIS slot's readback path actually ran, since the new public
    // contract admits want_color_readback=false and per-slot readback_slots -- in those branches a
    // persistent slot 2+ would otherwise reach SHADER_READ_ONLY_OPTIMAL with no barrier making its
    // writes visible to a later command buffer that samples or LOADs it.
    for (uint32_t slot = 2; slot < color_count; ++slot)
        publish_persistent_color(
            cached_extra[slot] != nullptr,
            readback_requested && (!color_target || color_target->readback_slots[slot]),
            extra_images[slot]);
    if (persistent_ds) {
        VkImageMemoryBarrier ds_ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ds_ready.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ds_ready.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ds_ready.image = dimg;
        ds_ready.subresourceRange = {DASPECT, 0, 1, 0, 1};
        ds_ready.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        ds_ready.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &ds_ready);
    }
    // Sampled depth bridge (#1275): return each borrowed DS image to the layout every depth pass
    // expects, so the bridge is invisible to the existing persistent-DS contract.
    for (const auto& [borrowed, borrowed_format] : borrowed_ds_images) {
        VkImageMemoryBarrier to_ds{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_ds.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_ds.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_ds.image = borrowed;
        to_ds.subresourceRange = {
            VK_IMAGE_ASPECT_DEPTH_BIT |
                (borrowed_format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                 : 0u),
            0, 1, 0, 1};
        to_ds.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_ds.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_ds);
    }
    for (const auto& [borrowed, saved_layout] : borrowed_compute_images) {
        VkImageMemoryBarrier restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        restore.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        restore.newLayout = saved_layout;
        restore.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        restore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        restore.srcQueueFamilyIndex = restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore.image = borrowed;
        restore.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &restore);
    }
    // A writable graphics storage image is architecturally guest memory, not a callback-local
    // texture. Copy its final texels back through the upload's coherent staging allocation before
    // releasing the image. The callback below restores the guest surface layout and publishes the
    // ordinary guest-write invalidation event. This deliberately forces one synchronization boundary
    // for storage-image calls; retaining GPU images without alias/import/write invalidation would be
    // faster but would still lose visibility at graphics->compute/DMA/CPU boundaries.
    for (SharedTextureUpload& upload : texture_uploads) {
        if (upload.storage_writebacks.empty() || !upload.image || !upload.staging) continue;
        VkImageMemoryBarrier to_readback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_readback.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_readback.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_readback.image = upload.image;
        to_readback.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_readback.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        to_readback.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_readback);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {upload.key.width, upload.key.height, upload.key.depth};
        vkCmdCopyImageToBuffer(cmd, upload.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               upload.staging, 1, &copy);
        record_host_read_barrier(cmd, upload.staging);   // #2944
        VkImageMemoryBarrier to_general{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general.image = upload.image;
        to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_general.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_general);
    }
    if (readback_requested) {
        auto transition_color_to_readback = [&](bool persistent, bool readback, VkImage image) {
            if (!persistent || !readback) return;
            VkImageMemoryBarrier to_readback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_readback.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_readback.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_readback.image = image;
            to_readback.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            to_readback.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            to_readback.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_readback);
        };
        transition_color_to_readback(persistent_color, readback_color0, img);
        transition_color_to_readback(persistent_color1, readback_color1, img1);
        for (uint32_t slot = 2; slot < color_count; ++slot) {
            const bool persistent_slot = cached_extra[slot] != nullptr;
            const bool slot_readback = want_color_readback &&
                prosper::frontend::is_color_target_readback_wanted(
                    color_target != nullptr,
                    color_target ? color_target->persistent_id_slots[slot] : 0,
                    persistent_slot,
                    color_target ? color_target->readback_slots[slot] : false);
            transition_color_to_readback(
                persistent_slot, slot_readback, extra_images[slot]);
        }
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {W, H, 1};
        if (readback_color0)
            vkCmdCopyImageToBuffer(
                cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp);
        if (readback_color1) {
            VkBufferImageCopy cp1 = cp; cp1.bufferOffset = color_offsets[1];
            vkCmdCopyImageToBuffer(
                cmd, img1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp1);
        }
        for (uint32_t slot = 2; slot < color_count; ++slot) {
            const bool persistent_slot = cached_extra[slot] != nullptr;
            const bool slot_readback = want_color_readback &&
                prosper::frontend::is_color_target_readback_wanted(
                    color_target != nullptr,
                    color_target ? color_target->persistent_id_slots[slot] : 0,
                    persistent_slot,
                    color_target ? color_target->readback_slots[slot] : false);
            if (!slot_readback) continue;
            VkBufferImageCopy extra_copy = cp;
            extra_copy.bufferOffset = color_offsets[slot];
            vkCmdCopyImageToBuffer(cmd, extra_images[slot],
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   rb, 1, &extra_copy);
        }
        // #2944: one whole-buffer dependency covers every colour slot copied above -- they are
        // offsets into one allocation, which the host maps and reads once the batch completes.
        record_host_read_barrier(cmd, rb);
        auto restore_persistent_color = [&](bool persistent, bool readback, VkImage image) {
            if (!persistent || !readback) return;
            VkImageMemoryBarrier to_sample{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_sample.image = image;
            to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            to_sample.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_sample);
        };
        restore_persistent_color(persistent_color, readback_color0, img);
        restore_persistent_color(persistent_color1, readback_color1, img1);
        for (uint32_t slot = 2; slot < color_count; ++slot)
            restore_persistent_color(
                cached_extra[slot] != nullptr,
                !color_target || color_target->readback_slots[slot], extra_images[slot]);
    }
    if (timing_enabled && flush_now)
        active_submission.end_gpu_timestamp(cmd);
    if (buffer_verify_enabled) {
        // Compare the mapped allocation against the source words. A mismatch names the first
        // differing BYTE, because what this is used to localize is an offset, not a boolean.
        //
        // WHAT IT CAN AND CANNOT SEE, precisely, because the difference decides what a zero from it
        // is allowed to retire. On the branch this examines, `upload.range` is assigned the same
        // `bytes` that drove the memcpy and that `word_count` describes, so `source_bytes` and
        // `device_bytes` are equal BY CONSTRUCTION: a short upload is structurally inexpressible
        // here and this check can never observe one. What it does observe is the destination range
        // being CLOBBERED after the memcpy and before the pass is submitted -- an overlapping arena
        // slice, a stray write, a pooled buffer handed out twice. Truncation is BUFLOG's and
        // `[buffer-truncated]`'s question; do not quote a zero from here against it.
        //
        // The other direction is a false POSITIVE: `#1268` above deliberately tolerates cross-thread
        // guest writes to the source words, so a guest that rewrites them after the memcpy produces
        // a MISMATCH that is not a prosper defect. Check the guest before blaming the upload.
        uint64_t parsed = 0;
        bool arming_rejected = false;
        auto strict = [&](const char* name, size_t& out, size_t unset) {
            const char* text = getenv(name);
            if (!text || !*text) { out = unset; return; }
            // The UINT32_MAX bound is not pedantry: `binding` is compared as uint32_t below, so a
            // value above it would truncate and aim the control at a DIFFERENT binding while still
            // printing a confident MUTATED line -- the same silent-misaim this strict parse exists
            // to prevent, one range check further out.
            if (prosper::diag::parse_u64_auto_base(text, &parsed) && parsed <= UINT32_MAX) {
                out = (size_t)parsed;
                return;
            }
            std::fprintf(stderr,
                         "[bufverify] %s=\"%s\" is not a number in [0, 4294967295] (decimal or 0x "
                         "hex); the control is DISARMED rather than aimed somewhere unintended\n",
                         name, text);
            arming_rejected = true;
            out = unset;
        };
        size_t mutate_offset = SIZE_MAX, mutate_min_bytes = 0, mutate_binding = SIZE_MAX;
        strict("PROSPER_BUFVERIFY_MUTATE", mutate_offset, SIZE_MAX);
        strict("PROSPER_BUFVERIFY_MUTATE_BINDING", mutate_binding, SIZE_MAX);
        strict("PROSPER_BUFVERIFY_MUTATE_MINBYTES", mutate_min_bytes, 0);
        if (arming_rejected) mutate_offset = SIZE_MAX;

        // The control. A clean zero from the comparison is worth exactly what its ability to report a
        // dirty one is worth, and nothing in the normal path ever exercises the mismatch branch.
        // PROSPER_BUFVERIFY_MUTATE corrupts one byte of a device copy so the very next line must name
        // that offset. It is AIMABLE because a control on a 128-byte uniform buffer says nothing
        // about a 7 KiB vertex buffer: a control drawn from a different size class than the null
        // tests the discriminator, not the domain.
        const BufferVerifyRecord* mutate_target = nullptr;
        for (const BufferVerifyRecord& cand : buffer_verify_records) {
            if (mutate_binding != SIZE_MAX && cand.binding != (uint32_t)mutate_binding) continue;
            if (cand.word_count * sizeof(uint32_t) < mutate_min_bytes) continue;
            if (cand.upload_index >= shared_buffers.size()) continue;
            if (!shared_buffers[cand.upload_index].mapped) continue;
            if (mutate_offset >= (size_t)shared_buffers[cand.upload_index].range) continue;
            mutate_target = &cand;
            break;
        }
        if (mutate_offset != SIZE_MAX && mutate_target) {
            const SharedBufferUpload& up = shared_buffers[mutate_target->upload_index];
            uint8_t* device = static_cast<uint8_t*>(up.mapped) + (size_t)up.offset;
            device[mutate_offset] = (uint8_t)(device[mutate_offset] ^ 0xFF);
            std::fprintf(stderr,
                         "[bufverify] MUTATED device byte %zu of set=%u binding=%u source_bytes=%zu "
                         "-- the next line must report a mismatch at exactly that offset\n",
                         mutate_offset, mutate_target->set, mutate_target->binding,
                         mutate_target->word_count * sizeof(uint32_t));
        } else if (mutate_offset != SIZE_MAX) {
            // Silence here is indistinguishable from a clean run, which is the whole failure this
            // control exists to prevent. Say so.
            std::fprintf(stderr,
                         "[bufverify] CONTROL DID NOT FIRE: no recorded buffer matched "
                         "binding=%s minbytes=%zu with a host mapping and offset %zu in range. "
                         "The \"0 mismatched\" below is UNVALIDATED\n",
                         mutate_binding == SIZE_MAX ? "any" : std::to_string(mutate_binding).c_str(),
                         mutate_min_bytes, mutate_offset);
        }

        size_t checked = 0, mismatched = 0, skipped_unmapped = 0;
        for (const BufferVerifyRecord& rec : buffer_verify_records) {
            if (rec.upload_index >= shared_buffers.size()) { ++skipped_unmapped; continue; }
            const SharedBufferUpload& up = shared_buffers[rec.upload_index];
            // A transient (non-arena, non-pooled) upload unmaps its memory before returning, so it
            // has no host mapping to re-read and is skipped. That is a WHOLE CLASS, not an oddity:
            // PROSPER_NO_BACKEND_BUFFER_POOL sends every upload down that path, and without the
            // count below such a run would print an authoritative-looking "0 mismatched" having
            // verified nothing at all.
            if (!up.mapped || !rec.words || !rec.word_count) { ++skipped_unmapped; continue; }
            const size_t source_bytes = rec.word_count * sizeof(uint32_t);
            const size_t device_bytes = (size_t)up.range;
            const uint8_t* device = static_cast<const uint8_t*>(up.mapped) + (size_t)up.offset;
            const uint8_t* source = reinterpret_cast<const uint8_t*>(rec.words);
            ++checked;
            const size_t common = source_bytes < device_bytes ? source_bytes : device_bytes;
            if (std::memcmp(device, source, common) == 0 && source_bytes == device_bytes) continue;
            size_t first_bad = SIZE_MAX;
            for (size_t b = 0; b < common; ++b)
                if (device[b] != source[b]) { first_bad = b; break; }
            ++mismatched;
            if (mismatched <= 32)
                std::fprintf(stderr,
                             "[bufverify] MISMATCH set=%u binding=%u id=%llx source_bytes=%zu "
                             "device_bytes=%zu first_differing_byte=%s (vertex %s at stride 32)\n",
                             rec.set, rec.binding, (unsigned long long)rec.identity,
                             source_bytes, device_bytes,
                             first_bad == SIZE_MAX ? "none (size differs only)"
                                                   : std::to_string(first_bad).c_str(),
                             first_bad == SIZE_MAX ? "-" : std::to_string(first_bad / 32).c_str());
        }
        if (!checked && !buffer_verify_records.empty())
            std::fprintf(stderr,
                         "[bufverify] NOTHING WAS VERIFIED: all %zu recorded buffer(s) lack a host "
                         "mapping (transient uploads). A zero from this run means the check never "
                         "ran, not that the uploads are correct\n",
                         buffer_verify_records.size());
        std::fprintf(stderr,
                     "[bufverify] %zu buffer(s) re-read from device memory, %zu mismatched, "
                     "%zu skipped (no host mapping)\n", checked, mismatched, skipped_unmapped);
    }

    vkEndCommandBuffer(cmd);

    // Publish newly uploaded exact-version textures before a later command buffer in the same batch
    // is recorded. The image upload and every consumer remain ordered in the eventual queue submit.
    // Do not evict while an earlier command buffer is pending: it may still reference the candidate.
    auto evict_persistent_texture = [&]() {
        auto victim = persistent_texture_images.end();
        for (auto it = persistent_texture_images.begin();
             it != persistent_texture_images.end(); ++it) {
            if (it->second.last_use == texture_generation) continue;
            if (victim == persistent_texture_images.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == persistent_texture_images.end()) return false;
        for (const auto& [key, binding] : victim->second.bindings) {
            if (binding.sampler) vkDestroySampler(dev, binding.sampler, nullptr);
            if (binding.view) vkDestroyImageView(dev, binding.view, nullptr);
        }
        vkDestroyImage(dev, victim->second.image, nullptr);
        vkFreeMemory(dev, victim->second.memory, nullptr);
        persistent_texture_bytes -= victim->second.bytes;
        persistent_texture_binding_entries -= victim->second.bindings.size();
        persistent_texture_images.erase(victim);
        return true;
    };
    for (auto& upload : texture_uploads) {
        if (!upload.direct_memory || !upload.persistent_id || upload.persistent_hit ||
            upload.persistent_refresh) continue;
        const PersistentTextureKey key{upload.persistent_id, upload.key.width, upload.key.height,
                                       upload.key.depth, upload.key.img_dim,
                                       upload.key.sample_count, upload.key.mip_levels,
                                       upload.key.format};
        while (!avoid_cache_eviction &&
               (persistent_texture_images.size() >= persistent_texture_max_entries ||
                (upload.image_bytes <= persistent_texture_limit &&
                 persistent_texture_bytes > persistent_texture_limit - upload.image_bytes)) &&
               evict_persistent_texture()) {}
        if (upload.image_bytes <= persistent_texture_limit &&
            persistent_texture_images.size() < persistent_texture_max_entries &&
            persistent_texture_bytes <= persistent_texture_limit - upload.image_bytes) {
            auto [cached, inserted] = persistent_texture_images.emplace(
                key, PersistentTextureImage{upload.image, upload.memory, upload.image_bytes,
                                            texture_generation, upload.persistent_version, true});
            if (inserted) {
                active_submission.add_failure_cleanup([key]() {
                    auto found = persistent_texture_images.find(key);
                    if (found != persistent_texture_images.end())
                        found->second.content_valid = false;
                });
                persistent_texture_bytes += upload.image_bytes;
                upload.image = VK_NULL_HANDLE;
                upload.memory = VK_NULL_HANDLE;
            }
        }
    }
    if (!avoid_cache_eviction)
        while (persistent_texture_bytes > persistent_texture_limit &&
               evict_persistent_texture()) {}
    resource_reuse_stats.persistent_texture_binding_entries = persistent_texture_binding_entries;
    texture_stats.persistent_cached_bytes = persistent_texture_bytes;

    const auto timing_recorded = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    active_submission.enqueue(cmd);
    if (cached_ds) {
        active_submission.add_failure_cleanup([cached_ds]() {
            cached_ds->layout_initialized = false;
            cached_ds->depth_valid = false;
            cached_ds->stencil_valid = false;
        });
        // PROSPER_DS_SLICE_CENSUS=1 — per (base, slice): how many passes attached it, and how many
        // of those actually claimed a depth write. A slice whose entry exists but never becomes
        // valid is either never really written or is having its write disclaimed here, and those
        // are different defects. GTA V's cube shadows show exactly that shape: every face has a
        // cache entry, and slice 0 is never valid.
        if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS")) {
            static std::mutex mutex;
            struct SliceTally { uint64_t passes = 0, claimed = 0, used_depth = 0, meaningful = 0; };
            static std::map<std::pair<uint64_t, uint32_t>, SliceTally> tally;
            static uint64_t n = 0;
            std::lock_guard lock(mutex);
            auto& row = tally[{ds_key.dr, ds_key.slice}];
            ++row.passes;
            if (use_depth) ++row.used_depth;
            if (depth_used_meaningfully) ++row.meaningful;
            if (use_depth && depth_used_meaningfully) ++row.claimed;
            // Increment SEQUENCED before the test -- see the same fix in live_renderer.cpp.
            // `(++n) & (n - 1)` has no sequencing between the operands of `&`, so this
            // throttle's cadence was undefined. Do not fold these back together.
            ++n;
            if ((n & (n - 1)) == 0 && n >= 256) {
                fprintf(stderr, "[ds-slice] after %llu DS passes:\n", (unsigned long long)n);
                for (const auto& e : tally)
                    fprintf(stderr,
                            "[ds-slice]   base=0x%llx slice=%u passes=%llu use_depth=%llu "
                            "meaningful=%llu claimed_valid=%llu%s\n",
                            (unsigned long long)e.first.first, e.first.second,
                            (unsigned long long)e.second.passes,
                            (unsigned long long)e.second.used_depth,
                            (unsigned long long)e.second.meaningful,
                            (unsigned long long)e.second.claimed,
                            e.second.claimed ? "" : "   <-- never claims a depth write");
            }
        }
        cached_ds->layout_initialized = true;
        cached_ds->depth_valid |= use_depth && depth_used_meaningfully;
        cached_ds->stencil_valid |= use_stencil;
        // Sampled depth bridge (#1275): recency for find_persistent_ds_sampled — two valid
        // entries can share a plane address (a surface re-keyed D32 -> D32S8 keeps its old
        // entry), and the most recently written one is the live truth.
        note_persistent_ds_depth_write(*cached_ds, use_depth, depth_may_be_written,
                                       depth_write_command_order);
    }
    if (cached_color) {
        active_submission.add_failure_cleanup([cached_color]() {
            cached_color->valid = false;
        });
        cached_color->valid = true;
        cached_color->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (cached_color1) {
        active_submission.add_failure_cleanup([cached_color1]() {
            cached_color1->valid = false;
        });
        cached_color1->valid = true;
        cached_color1->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    for (uint32_t slot = 2; slot < color_count; ++slot) {
        PersistentColorTargetImage* retained = cached_extra[slot];
        if (!retained) continue;
        active_submission.add_failure_cleanup([retained]() { retained->valid = false; });
        retained->valid = true;
        // The layout the image is ACTUALLY left in: the pass ends it in SHADER_READ_ONLY_OPTIMAL and
        // the readback restore returns it there. Recording COLOR_ATTACHMENT_OPTIMAL here while the
        // image sat in TRANSFER_SRC_OPTIMAL is what made the next group's initialLayout a lie --
        // VUID-vkCmdDraw-None-09600, invisible on RADV because the pixels happened to survive.
        retained->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    BackendSubmissionBatchResult batch_result;
    if (flush_now)
        batch_result = active_submission.submit_and_wait(dev, queue, backend_trace);
    if (echo_mapped) {
        const uint8_t* bytes = static_cast<const uint8_t*>(echo_mapped);
        for (size_t i = 0; i < echo_count; ++i) {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(bytes + kEchoStride * i);
            const bool is_index =
                std::find(echo_index_slice.begin(), echo_index_slice.end(), i) !=
                echo_index_slice.end();
            if (is_index) {
                std::fprintf(stderr,
                             "[buffer-echo] INDEX slice=%zu gpu-read=%u %u %u %u %u %u\n",
                             i, words[0], words[1], words[2], words[3], words[4], words[5]);
                continue;
            }
            std::fprintf(stderr,
                         "[buffer-echo] slice=%zu buf=%p off=%llu range=%llu gpu-read="
                         "%08x %08x %08x %08x\n",
                         i, (void*)shared_buffers[i].buffer,
                         (unsigned long long)shared_buffers[i].offset,
                         (unsigned long long)shared_buffers[i].range,
                         words[0], words[1], words[2], words[3]);
        }
    }
    if (echo_mapped) vkUnmapMemory(dev, echo_memory);
    if (echo_memory) vkFreeMemory(dev, echo_memory, nullptr);
    if (echo_buffer) vkDestroyBuffer(dev, echo_buffer, nullptr);
    const auto timing_gpu_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const bool batch_completed = !flush_now ||
        (batch_result.submit_result == VK_SUCCESS && batch_result.wait_result == VK_SUCCESS);

    // Fragment-funnel readback (PROSPER_DRAW_STATS): one line per realized draw showing where its
    // pixels vanished. ds_active implies flush_now (the pool-creation gate above uses the same flush
    // condition), so `cmd` has completed and the results are ready. Pools are destroyed unconditionally.
    if (ds_active) {
        const uint32_t nq = static_cast<uint32_t>(dv.size());
        if (batch_completed) {
            std::vector<uint64_t> sres(static_cast<size_t>(nq) * 5, 0);  // 4 statistics + availability
            std::vector<uint64_t> ores(static_cast<size_t>(nq) * 2, 0);  // occlusion samples + availability
            vkGetQueryPoolResults(dev, ds_stats_pool, 0, nq, sres.size() * sizeof(uint64_t), sres.data(),
                                  5 * sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            vkGetQueryPoolResults(dev, ds_occ_pool, 0, nq, ores.size() * sizeof(uint64_t), ores.data(),
                                  2 * sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            for (uint32_t i = 0; i < nq; i++) {
                if (!sres[i * 5 + 4]) continue;  // unavailable -> draw was skipped (v.ok == false)
                const uint64_t verts = sres[i * 5 + 0], prims = sres[i * 5 + 1],
                               clip = sres[i * 5 + 2], fs = sres[i * 5 + 3];
                const uint64_t samp = ores[i * 2 + 1] ? ores[i * 2 + 0] : 0;
                // Funnel classification, checked in pipeline order. `samples` (occlusion) is the ground
                // truth for "survived the depth+stencil test" — it is counted even when the fragment
                // shader is optimised out for a colour-write-disabled (stencil-only) draw, so it must be
                // tested before fs_inv or such draws look falsely dead.
                const char* tag =
                    (prims == 0) ? "NO-GEOMETRY(no primitives)" :
                    (clip  == 0) ? "GEOMETRY-VANISH(clipped/degenerate/offscreen)" :
                    (samp  >  0) ? "passed-samples(colour/stencil written)" :
                    (fs    >  0) ? "TEST-KILLED(depth/stencil rejected all)" :
                                   "NO-RASTER(cull/scissor/zero-area)";
                const uint64_t draw_index = draws[i].draw_index != UINT64_MAX
                    ? draws[i].draw_index : i;
                fprintf(stderr,
                        "[draw-stats] draw=%llu verts=%llu prims=%llu after_clip=%llu fs_inv=%llu samples=%llu %s\n",
                        (unsigned long long)draw_index,
                        (unsigned long long)verts, (unsigned long long)prims,
                        (unsigned long long)clip, (unsigned long long)fs, (unsigned long long)samp, tag);
            }
        }
    }

    // Geometry-probe readback: report where the probed draw's post-transform clip-space vertices landed.
    // geom_active implies flush_now (same gate as buffer creation), so `cmd` has completed here.
    if (geom_active) {
        if (batch_completed) {
            // The counter buffer holds the byte count transform feedback actually wrote; read only that
            // many vertices so the uninitialized tail of the (3x-oversized) buffer never pollutes stats.
            uint32_t written = 0;
            void* cp = nullptr;
            if (vkMapMemory(dev, geom_counter_mem, 0, 16, 0, &cp) == VK_SUCCESS) {
                uint32_t bytes = 0; std::memcpy(&bytes, cp, sizeof bytes);
                written = std::min(bytes / 16u, geom_cap);
                vkUnmapMemory(dev, geom_counter_mem);
            }
            void* gp = nullptr;
            if (written && vkMapMemory(dev, geom_mem, 0, static_cast<VkDeviceSize>(written) * 16, 0, &gp) == VK_SUCCESS) {
                const float* pos = static_cast<const float*>(gp);
                float minx=1e30f,maxx=-1e30f,miny=1e30f,maxy=-1e30f,minz=1e30f,maxz=-1e30f,minw=1e30f,maxw=-1e30f;
                uint32_t nan=0, wle0=0, offscreen=0, clipped=0, finite=0;
                bool all_same = written > 0;
                for (uint32_t i = 0; i < written; i++) {
                    const float x=pos[i*4+0], y=pos[i*4+1], z=pos[i*4+2], w=pos[i*4+3];
                    if (!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||!std::isfinite(w)) { nan++; continue; }
                    finite++;
                    minx=std::min(minx,x); maxx=std::max(maxx,x); miny=std::min(miny,y); maxy=std::max(maxy,y);
                    minz=std::min(minz,z); maxz=std::max(maxz,z); minw=std::min(minw,w); maxw=std::max(maxw,w);
                    if (w <= 0.0f) wle0++;
                    if (std::fabs(x) > std::fabs(w) || std::fabs(y) > std::fabs(w)) offscreen++;
                    // A vertex is on-screen only if it is in front of the camera (w>0) and inside the
                    // clip cube (|x|,|y| <= w). Everything else is clipped and cannot rasterize.
                    if (!(w > 0.0f && std::fabs(x) <= w && std::fabs(y) <= w)) clipped++;
                    if (i && (x!=pos[0]||y!=pos[1]||z!=pos[2]||w!=pos[3])) all_same = false;
                }
                const uint32_t onscreen = finite - clipped;
                // Classification is descriptive (read it WITH the funnel, which owns the vanishes verdict):
                // a large quad whose verts sit just outside the cube still rasterizes via clipping, so
                // "all verts outside" is not the same as "renders nothing" — the bbox tells them apart.
                const char* tag =
                    finite == 0             ? "ALL-NAN/INF(numeric defect)" :
                    all_same                ? "DEGENERATE(all verts collapse to one clip point)" :
                    wle0 == finite          ? "ALL-BEHIND-CAMERA(w<=0: transform/w defect)" :
                    clipped == finite       ? "ALL-VERTS-OUTSIDE-CLIP-CUBE(see bbox: off-screen shift or oversized quad)" :
                    onscreen * 20u < finite ? "MOSTLY-OUTSIDE(<5% verts on-screen; see bbox)" :
                                              "on-screen(geometry spread across clip space)";
                // Shader I/O tap mode (PROSPER_SHADER_TAP): the captured "positions" are actually the tapped
                // intermediate VGPR (dst..dst+3) at that PC, so print the raw hex too (values are often
                // integers/bitfields, not clip floats) and skip the meaningless clip classification.
                // Gated on the REQUEST, not on whether the redirect happened -- and those differ
                // (#2064). tap_vec is set only when the instruction at tap_pc is walked, so a
                // tap_pc after this shader's EXP POS0 exports the REAL clip position while this
                // header still says "the tapped VGPR". The recompiler now prints
                // `[shader-tap] NOT APPLIED ...` naming the PC in that case, so the two lines
                // must be read together: this header states what was ASKED FOR, and the ABSENCE
                // of a NOT APPLIED line is what says it was delivered.
                const bool is_tap = PROSPER_ENV_ON("PROSPER_SHADER_TAP");
                if (is_tap)
                    fprintf(stderr, "[geom-probe] draw=%llu SHADER-TAP REQUESTED (see any [shader-tap] NOT APPLIED line for this shader, which means these are the REAL clip positions -- #2064): values below are the tapped VGPR "
                                    "(dst+3) at that PC, not clip positions (bbox/tags meaningless)\n",
                            geom_target_label);
                else
                    fprintf(stderr, "[geom-probe] draw=%llu verts-written=%u finite=%u on-screen=%u clipped=%u "
                                    "(offscreen=%u w<=0=%u nan/inf=%u)\n"
                                    "[geom-probe]   clip-bbox x[%g,%g] y[%g,%g] z[%g,%g] w[%g,%g] -> %s\n",
                            geom_target_label, written, finite, onscreen, clipped, offscreen, wle0, nan,
                            minx,maxx, miny,maxy, minz,maxz, minw,maxw, tag);
                for (uint32_t i = 0; i < written && i < (is_tap ? 8u : 4u); i++) {
                    if (is_tap) {
                        uint32_t h[4]; std::memcpy(h, &pos[i*4], 16);
                        fprintf(stderr, "[geom-probe]   v%u = float(%g, %g, %g, %g) hex(%08x %08x %08x %08x)\n",
                                i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3], h[0], h[1], h[2], h[3]);
                    } else {
                        fprintf(stderr, "[geom-probe]   v%u = (%g, %g, %g, %g)\n",
                                i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3]);
                    }
                }
                // Geometry-health metrics (#1257): the transform-feedback buffer already holds every
                // post-transform vertex in primitive-assembly order, so consecutive triples ARE the
                // rasterized triangles (TF decomposes strips/fans into triangles). Report the tells that
                // localize a geometry/vertex-fetch bug — the exact signals that cracked GTA #1163's
                // vertex-count inflation: how many DISTINCT positions the draw really has (a shared VB pool
                // read past its real range collapses most verts onto a few points), what fraction of
                // triangles are DEGENERATE (zero-area stitching / collapsed fetch), and how many triangles
                // are exact DUPLICATES (the same triangle rasterized N times = pure overdraw). Overdraw
                // itself is the funnel's job (occlusion samples > covered pixels): read this WITH
                // PROSPER_DRAW_STATS. Skipped in tap mode (pos holds VGPR values, not positions).
                if (!is_tap && written >= 3) {
                    std::vector<std::array<float, 4>> verts;   // finite positions, for unique/multiplicity
                    verts.reserve(written);
                    for (uint32_t i = 0; i < written; i++) {
                        const float x = pos[i*4+0], y = pos[i*4+1], z = pos[i*4+2], w = pos[i*4+3];
                        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w))
                            verts.push_back({x, y, z, w});
                    }
                    std::sort(verts.begin(), verts.end());
                    uint32_t unique_pos = 0, max_mult = 0, run = 0;
                    for (size_t i = 0; i < verts.size(); i++) {
                        if (i == 0 || verts[i] != verts[i-1]) { unique_pos++; run = 1; } else run++;
                        max_mult = std::max(max_mult, run);
                    }
                    // Per-triangle (over FINITE triangles only): degenerate (near-zero NDC area) + exact
                    // duplicates (canonical key). Non-finite triangles are a numeric defect already reported
                    // by the probe's nan/inf count, and are skipped here so no NaN enters std::sort (a NaN
                    // breaks the strict-weak-ordering -> UB); their vertices are also excluded from `verts`.
                    const uint32_t ntri = written / 3u;
                    uint32_t degenerate = 0, finite_tri = 0;
                    std::vector<std::array<float, 12>> tris;
                    tris.reserve(ntri);
                    for (uint32_t t = 0; t < ntri; t++) {
                        const float* p = &pos[t*3*4];
                        bool fin = true;
                        for (int k = 0; k < 12; k++) if (!std::isfinite(p[k])) { fin = false; break; }
                        if (!fin) continue;
                        finite_tri++;
                        auto ndc = [&](int k, int c) {   // p[k*4+3] is finite here
                            float v = p[k*4+c], w = p[k*4+3];
                            return w != 0.0f ? v / w : v;
                        };
                        const float ax = ndc(0,0), ay = ndc(0,1), bx = ndc(1,0), by = ndc(1,1),
                                    cx = ndc(2,0), cy = ndc(2,1);
                        const float area2 = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay);
                        if (std::fabs(area2) < 1e-10f) degenerate++;
                        // Canonical key: the triangle's three (x,y,z,w) vertices sorted, so a duplicate
                        // triangle in any winding/rotation collides.
                        std::array<std::array<float,4>,3> v3 = {{
                            {p[0],p[1],p[2],p[3]}, {p[4],p[5],p[6],p[7]}, {p[8],p[9],p[10],p[11]} }};
                        std::sort(v3.begin(), v3.end());
                        tris.push_back({v3[0][0],v3[0][1],v3[0][2],v3[0][3],
                                        v3[1][0],v3[1][1],v3[1][2],v3[1][3],
                                        v3[2][0],v3[2][1],v3[2][2],v3[2][3]});
                    }
                    std::sort(tris.begin(), tris.end());
                    uint32_t dup_tri = 0;
                    for (size_t i = 1; i < tris.size(); i++) if (tris[i] == tris[i-1]) dup_tri++;
                    const uint32_t real_tri = finite_tri - degenerate;
                    const char* health =
                        finite && unique_pos <= 2                      ? "COLLAPSED(<=2 distinct positions - fetch/transform returns a constant)" :
                        finite_tri && degenerate * 5u >= finite_tri*4u ? "DEGENERATE-HEAVY(>=80% zero-area - strip-stitching read as list, or wrong count/stride)" :
                        dup_tri && dup_tri * 4u >= real_tri            ? "DUPLICATE-TRIANGLES(exact repeats = pure overdraw - likely over-count/wrong vertex range)" :
                                                                         "ok(check PROSPER_DRAW_STATS for overdraw: samples>pixels)";
                    fprintf(stderr, "[geom-health] draw=%llu verts=%u unique-pos=%u (max-mult=%u) "
                                    "triangles=%u degenerate=%u(%.0f%%) real=%u duplicate-tri=%u -> %s\n",
                            geom_target_label, (unsigned)verts.size(), unique_pos, max_mult, finite_tri, degenerate,
                            finite_tri ? 100.0 * degenerate / finite_tri : 0.0, real_tri, dup_tri, health);
                }
                // PROSPER_GEOM_PROBE_DUMP=path (gated, off by default): write EVERY post-transform vertex
                // (x,y,z,w in primitive-assembly order — for a triangle list, consecutive triples are the
                // rasterized triangles) as CSV, so per-triangle overlap/degeneracy can be analyzed offline
                // (e.g. GTA #1163's stencil over-count from self-overlapping mask triangles).
                if (const char* dp = PROSPER_ENV_VALUE("PROSPER_GEOM_PROBE_DUMP")) {
                    if (FILE* f = fopen(dp, "w")) {
                        fprintf(f, "i,x,y,z,w\n");
                        for (uint32_t i = 0; i < written; i++)
                            fprintf(f, "%u,%.7g,%.7g,%.7g,%.7g\n",
                                    i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3]);
                        fclose(f);
                        fprintf(stderr, "[geom-probe]   wrote %u verts -> %s\n", written, dp);
                    }
                }
                vkUnmapMemory(dev, geom_mem);
            } else if (!written) {
                // Supportable only because arming proved OpExecutionMode Xfb is present on the last
                // pre-rasterization stage (#3248); without that check this line was a wrong answer
                // rather than a null one.
                fprintf(stderr, "[geom-probe] draw=%llu: transform feedback wrote 0 vertices "
                                "(the capture was declared, so the draw produced no primitives)\n",
                        geom_target_label);
            }
        }
    }

    if (storage_writeback_requested && batch_completed) {
        for (SharedTextureUpload& upload : texture_uploads) {
            if (upload.storage_writebacks.empty() || !upload.staging ||
                !upload.staging_memory)
                continue;
            const size_t storage_bytes = static_cast<size_t>(upload.key.width) *
                upload.key.height * upload.key.depth * upload.key.sample_count *
                backend_color_bytes_per_pixel(upload.key.format);
            void* mapped = nullptr;
            if (!storage_bytes ||
                vkMapMemory(dev, upload.staging_memory, 0, storage_bytes, 0,
                            &mapped) != VK_SUCCESS ||
                !mapped)
                continue;
            const auto* pixels = static_cast<const uint8_t*>(mapped);
            for (const auto& writeback : upload.storage_writebacks)
                writeback(pixels, storage_bytes);
            vkUnmapMemory(dev, upload.staging_memory);
        }
    }
    if (readback_requested && batch_completed) {
        void* mp = nullptr; vkMapMemory(dev, bmem, 0, readback_bytes, 0, &mp);
        const auto* readback = static_cast<const uint8_t*>(mp);
        // A range assignment constructs directly from the mapped pixels. resize()+memcpy first zeroed the
        // entire 8.3 MiB 1080p vector even though every byte was immediately overwritten.
        if (readback_color0)
            out.assign(readback, readback + static_cast<size_t>(bytes));
        if (readback_color1)
            (out_rgba1 ? *out_rgba1 : mrt_outputs->colors[1]).assign(
                readback + static_cast<size_t>(color_offsets[1]),
                readback + static_cast<size_t>(color_offsets[1] + color_bytes[1]));
        if (mrt_outputs)
            for (uint32_t slot = 2; slot < color_count; ++slot) {
                const bool persistent_slot = cached_extra[slot] != nullptr;
                const bool slot_readback = want_color_readback &&
                    prosper::frontend::is_color_target_readback_wanted(
                        color_target != nullptr,
                        color_target ? color_target->persistent_id_slots[slot] : 0,
                        persistent_slot,
                        color_target ? color_target->readback_slots[slot] : false);
                if (!slot_readback) continue;
                mrt_outputs->colors[slot].assign(
                    readback + static_cast<size_t>(color_offsets[slot]),
                    readback + static_cast<size_t>(color_offsets[slot] + color_bytes[slot]));
            }
        vkUnmapMemory(dev, bmem);
        color_target_stats.readbacks = persistent_color ? 1 : 0;
    }
    const auto timing_readback_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};

    // PROSPER_DRAW_ISO + PROSPER_ISO_AT="x,y": per-draw kill isolation (generalizes the #240 title harness
    // to any submit / any target pixel). On the FIRST submit whose rendered pixel at (x,y) is lit
    // (non-background), re-render THIS exact submit once per killed-draw index and report which draw lights
    // that pixel — the kill index that turns (x,y) dark is the culprit. Reuses the built pipelines/
    // descriptors; a fresh clear each pass. Env-gated, no default behavior. Used to locate a stray primitive
    // such as the #298 menu focus-ring sliver. Dumps iso_kill_<k>.bmp to PROSPER_FRAME_DIR.
    if (batch_completed && FMT == VK_FORMAT_R8G8B8A8_UNORM &&
        getenv("PROSPER_DRAW_ISO") && getenv("PROSPER_ISO_AT")) {
        static bool iso_done = false;
        int tx = -1, ty = -1; sscanf(getenv("PROSPER_ISO_AT"), "%d,%d", &tx, &ty);
        // Optional PROSPER_ISO_RGB="r,g,b" (+ PROSPER_ISO_TOL, default 45): the target submit is the first
        // whose pixel at (x,y) matches that color within tol — robust against an earlier full-screen submit
        // (e.g. the intro cutscene) that merely lights the pixel a different color. Unset -> any non-background.
        int wr = -1, wg = 0, wb = 0, tol = getenv("PROSPER_ISO_TOL") ? atoi(getenv("PROSPER_ISO_TOL")) : 45;
        if (getenv("PROSPER_ISO_RGB")) sscanf(getenv("PROSPER_ISO_RGB"), "%d,%d,%d", &wr, &wg, &wb);
        auto lit_at = [&](const std::vector<uint8_t>& buf) -> bool {
            if (tx < 0 || ty < 0 || (uint32_t)tx >= W || (uint32_t)ty >= H) return false;
            const uint8_t* p = &buf[((size_t)ty * W + tx) * 4];
            if (wr >= 0) return abs((int)p[0]-wr) <= tol && abs((int)p[1]-wg) <= tol && abs((int)p[2]-wb) <= tol;
            return p[0] > 40 || p[1] > 40 || p[2] > 40;
        };
        // Optional second reference pixel: reported alongside the target so we can tell whether the culprit
        // draw ALSO paints a legit element (e.g. the active focus ring) or only the stray pixel.
        int rx = -1, ry = -1; if (getenv("PROSPER_ISO_AT2")) sscanf(getenv("PROSPER_ISO_AT2"), "%d,%d", &rx, &ry);
        if (!iso_done && lit_at(out)) {
            iso_done = true;
            const char* fd = getenv("PROSPER_FRAME_DIR"); std::string dir = fd ? fd : ".";
            fprintf(stderr, "[iso] submit lights (%d,%d): %zu draws; re-rendering per killed draw\n", tx, ty, dv.size());
            // Characterize every draw in the target submit (blend/write-mask/viewport/textures/vertex count).
            for (size_t di = 0; di < dv.size(); di++) {
                const prosper::gpu::ResolvedPipelineState* ps = draws[di].ps; DV& v = dv[di];
                fprintf(stderr, "[iso]  draw#%zu %s cnt=%u", di, v.ok ? "OK" : "SKIP", v.icount ? v.icount : v.vcount);
                if (ps) fprintf(stderr, " blend=%d src=%u dst=%u cwm=0x%x vp_y=%.0f vp_h=%.0f depth=%d/%d",
                                (int)ps->blend_enable, ps->src_color_blend_factor, ps->dst_color_blend_factor,
                                ps->color_write_mask, ps->viewport_y, ps->viewport_h,
                                (int)ps->depth_test_enable, (int)ps->depth_write_enable);
                int nt = 0; for (const auto& r : draws[di].R) if (r.is_texture()) { fprintf(stderr, " tex%d=%ux%u", nt, r.tw, r.th); nt++; }
                fprintf(stderr, "\n");
            }
            VkFenceCreateInfo iso_fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence iso_fence = VK_NULL_HANDLE;
            if (vkCreateFence(dev, &iso_fence_info, nullptr, &iso_fence) != VK_SUCCESS)
                iso_fence = VK_NULL_HANDLE;
            bool iso_submission_pending = false;
            VkBufferImageCopy cp2{}; cp2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp2.imageExtent = {W, H, 1};
            for (int kk = -1; iso_fence && kk < (int)dv.size(); kk++) {
                VkCommandBuffer c2 = VK_NULL_HANDLE;
                VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
                if (vkAllocateCommandBuffers(dev, &ai, &c2) != VK_SUCCESS ||
                    vkBeginCommandBuffer(c2, &cbbi) != VK_SUCCESS) {
                    if (c2) vkFreeCommandBuffers(dev, pool, 1, &c2);
                    break;
                }
                vkCmdBeginRenderPass(c2, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
                for (size_t di = 0; di < dv.size(); di++) { auto& v = dv[di]; if (!v.ok) continue; if ((int)di == kk) continue;
                    vkCmdBindPipeline(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
                    record_draw_dynamic_state(c2, v);   // #3248: same state as the pass being isolated
                    if (v.use_desc) vkCmdBindDescriptorSets(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
                    if (v.icount) { vkCmdBindIndexBuffer(c2, v.ibuf, v.ioffset, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(c2, v.icount, v.instance_count, 0, v.vertex_offset, 0); }
                    else vkCmdDraw(c2, v.vcount, v.instance_count, static_cast<uint32_t>(v.vertex_offset), 0);
                }
                vkCmdEndRenderPass(c2);
                backend_draw_iso_pass_count().fetch_add(1, std::memory_order_relaxed);
                vkCmdCopyImageToBuffer(c2, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp2);
                record_host_read_barrier(c2, rb);   // #2944
                const bool recorded = vkEndCommandBuffer(c2) == VK_SUCCESS;
                const bool fence_reset = recorded &&
                    vkResetFences(dev, 1, &iso_fence) == VK_SUCCESS;
                VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si2.commandBufferCount = 1; si2.pCommandBuffers = &c2;
                const bool submitted = fence_reset &&
                    render_locked_queue_submit(queue, 1, &si2, iso_fence) == VK_SUCCESS;
                bool finished = submitted &&
                    vkWaitForFences(dev, 1, &iso_fence, VK_TRUE,
                                    5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
                if (submitted && !finished)
                    finished = render_locked_queue_wait_idle(queue) == VK_SUCCESS;
                const BackendSubmissionState iso_state =
                    backend_submission_state(submitted, finished);
                if (iso_state == BackendSubmissionState::Pending) {
                    // The diagnostic command uses this call's shared pool, framebuffer, pipelines,
                    // descriptors, images, and readback buffer. Retain that complete cleanup scope,
                    // its fence, and c2 when neither wait proves completion.
                    if (cached_ds) {
                        cached_ds->layout_initialized = false;
                        cached_ds->depth_valid = false;
                        cached_ds->stencil_valid = false;
                    }
                    if (cached_color) cached_color->valid = false;
                    active_submission.abandon_pending_resources();
                    iso_submission_pending = true;
                    break;
                }
                if (iso_state != BackendSubmissionState::Complete) {
                    vkFreeCommandBuffers(dev, pool, 1, &c2);
                    break;
                }
                std::vector<uint8_t> px(bytes); void* m2 = nullptr;
                if (vkMapMemory(dev, bmem, 0, bytes, 0, &m2) != VK_SUCCESS || !m2) {
                    vkFreeCommandBuffers(dev, pool, 1, &c2);
                    break;
                }
                for (VkDeviceSize i = 0; i < bytes; i++) px[i] = ((const uint8_t*)m2)[i]; vkUnmapMemory(dev, bmem);
                const uint8_t* tp = &px[((size_t)ty * W + tx) * 4];
                bool lit = tp[0] > 40 || tp[1] > 40 || tp[2] > 40;
                // Total gold pixels (ring color) in the frame: if killing a draw drops this only by the
                // sliver's ~size, that draw paints ONLY the sliver; a large drop means it also paints the ring.
                size_t gold_n = 0;
                for (size_t q = 0; q < (size_t)W * H; q++) { const uint8_t* g = &px[q * 4];
                    if (g[0] > 140 && g[1] > 100 && g[2] < 90 && (int)g[0] - (int)g[2] > 60) gold_n++; }
                char ref[96] = "";
                if (rx >= 0 && ry >= 0 && (uint32_t)rx < W && (uint32_t)ry < H) {
                    const uint8_t* rp = &px[((size_t)ry * W + rx) * 4];
                    snprintf(ref, sizeof ref, "  ref(%d,%d)=%u,%u,%u %s", rx, ry, rp[0], rp[1], rp[2],
                             (rp[0] > 40 || rp[1] > 40 || rp[2] > 40) ? "lit" : "DARK");
                }
                fprintf(stderr, "[iso]  kill=%d -> (%d,%d) rgb=%u,%u,%u %s  gold_px=%zu%s\n", kk, tx, ty, tp[0], tp[1], tp[2],
                        lit ? "LIT" : "dark <<< THIS DRAW paints the pixel", gold_n, ref);
                char fn[512]; snprintf(fn, sizeof fn, "%s/iso_kill_%d.bmp", dir.c_str(), kk); dump_bmp(fn, px, W, H);
                vkFreeCommandBuffers(dev, pool, 1, &c2);
            }
            if (iso_fence && !iso_submission_pending)
                vkDestroyFence(dev, iso_fence, nullptr);
            fprintf(stderr, "[iso] done: the kill index marked 'dark' is the draw painting (%d,%d)\n", tx, ty);
        }
    }

    color_target_stats.cached_bytes = persistent_color_target_bytes();
    color_target_stats.cached_entries = persistent_color_target_cache().size();
    resource_reuse_stats.persistent_pipeline_layout_entries =
        persistent_pipeline_layouts.size();
    const bool transient_color = cached_color == nullptr;
    const bool transient_color1 = use_color1 && cached_color1 == nullptr;
    // Per-slot twin of the two flags above, so the teardown lambda can tell a retained slot from a
    // transient one without capturing the cache pointers themselves.
    std::array<bool, prosper::gpu::kColorTargetCount> transient_extra{};
    for (uint32_t slot = 2; slot < color_count; ++slot)
        transient_extra[slot] = cached_extra[slot] == nullptr;
    const bool transient_ds = use_ds && cached_ds == nullptr;
    const RenderVkCtx* ctx_ptr = &ctx;
    active_submission.add_cleanup(
        [dev, pool, dv = std::move(dv), shared_descriptor_pool,
         shared_pipeline_layouts = std::move(shared_pipeline_layouts),
         shared_descriptor_set_layouts = std::move(shared_descriptor_set_layouts),
         shared_texture_bindings = std::move(shared_texture_bindings),
         shared_buffers = std::move(shared_buffers),
         shared_buffer_arenas = std::move(shared_buffer_arenas),
         texture_uploads = std::move(texture_uploads), seedbuf, seedmem, seedbuf1, seedmem1,
         extra_seedbufs, extra_seedmems,
         rb, bmem, fb, rp, transient_color, view, img, imem, transient_color1, view1, img1,
         transient_extra,
         imem1, color_count, extra_views, extra_images, extra_memories,
         transient_ds, dview, dimg, dmem, ds_stats_pool, ds_occ_pool,
         geom_buf, geom_mem, geom_counter, geom_counter_mem, ctx_ptr,
         color_target_generation]() mutable {
            vkDestroyCommandPool(dev, pool, nullptr);
            if (ds_stats_pool) vkDestroyQueryPool(dev, ds_stats_pool, nullptr);
            if (ds_occ_pool) vkDestroyQueryPool(dev, ds_occ_pool, nullptr);
            if (geom_buf) vkDestroyBuffer(dev, geom_buf, nullptr);
            if (geom_mem) vkFreeMemory(dev, geom_mem, nullptr);
            if (geom_counter) vkDestroyBuffer(dev, geom_counter, nullptr);
            if (geom_counter_mem) vkFreeMemory(dev, geom_counter_mem, nullptr);
            for (auto& v : dv) {
                if (v.pipe && !v.pipeline_cached) vkDestroyPipeline(dev, v.pipe, nullptr);
                // An arena slice is owned by the arena and released with it; destroying the
                // shared buffer here would free storage still referenced by other draws.
                if (!v.iarena) {
                    if (v.ibuf) vkDestroyBuffer(dev, v.ibuf, nullptr);
                    if (v.ibmem) release_transient_render_memory(dev, v.ibmem);
                }
                if (v.vs) vkDestroyShaderModule(dev, v.vs, nullptr);
                if (v.gs) vkDestroyShaderModule(dev, v.gs, nullptr);
                if (v.fs) vkDestroyShaderModule(dev, v.fs, nullptr);
            }
            if (shared_descriptor_pool)
                vkDestroyDescriptorPool(dev, shared_descriptor_pool, nullptr);
            for (const auto& [key, layout] : shared_pipeline_layouts)
                if (layout.handle && !layout.persistent)
                    vkDestroyPipelineLayout(dev, layout.handle, nullptr);
            for (const auto& [key, layout] : shared_descriptor_set_layouts)
                if (layout) vkDestroyDescriptorSetLayout(dev, layout, nullptr);
            for (const SharedTextureBinding& binding : shared_texture_bindings) {
                if (!binding.persistent) {
                    if (binding.sampler) vkDestroySampler(dev, binding.sampler, nullptr);
                    if (binding.view) vkDestroyImageView(dev, binding.view, nullptr);
                }
            }
            for (const SharedBufferUpload& upload : shared_buffers) {
                if (upload.arena) {
                    continue;
                } else if (upload.pooled) {
                    release_render_host_buffer(
                        dev, {upload.buffer, upload.memory, upload.mapped,
                              upload.bytes, upload.allocation_bytes});
                } else {
                    if (upload.buffer) vkDestroyBuffer(dev, upload.buffer, nullptr);
                    if (upload.memory) release_transient_render_memory(dev, upload.memory);
                }
            }
            for (SharedBufferArena& arena : shared_buffer_arenas)
                release_render_host_buffer(dev, arena.buffer);
            for (auto& upload : texture_uploads) {
                if (upload.image && !upload.persistent_hit && !upload.persistent_refresh &&
                    !upload.borrowed_target &&
                    !upload.borrowed_compute && !upload.borrowed_ds)
                    vkDestroyImage(dev, upload.image, nullptr);
                if (upload.memory) {
                    if (upload.direct_memory) vkFreeMemory(dev, upload.memory, nullptr);
                    else release_transient_render_memory(dev, upload.memory);
                }
                if (upload.staging) vkDestroyBuffer(dev, upload.staging, nullptr);
                if (upload.staging_memory)
                    release_transient_render_memory(dev, upload.staging_memory);
            }
            if (seedbuf) vkDestroyBuffer(dev, seedbuf, nullptr);
            if (seedmem) release_transient_render_memory(dev, seedmem);
            if (seedbuf1) vkDestroyBuffer(dev, seedbuf1, nullptr);
            if (seedmem1) release_transient_render_memory(dev, seedmem1);
            for (uint32_t slot = 2; slot < prosper::gpu::kColorTargetCount; ++slot) {
                if (extra_seedbufs[slot]) vkDestroyBuffer(dev, extra_seedbufs[slot], nullptr);
                if (extra_seedmems[slot])
                    release_transient_render_memory(dev, extra_seedmems[slot]);
            }
            if (rb) vkDestroyBuffer(dev, rb, nullptr);
            if (bmem) release_transient_render_memory(dev, bmem);
            vkDestroyFramebuffer(dev, fb, nullptr);
            vkDestroyRenderPass(dev, rp, nullptr);
            if (transient_color) {
                vkDestroyImageView(dev, view, nullptr);
                vkDestroyImage(dev, img, nullptr);
                release_transient_render_memory(dev, imem);
            }
            if (transient_color1) {
                vkDestroyImageView(dev, view1, nullptr);
                vkDestroyImage(dev, img1, nullptr);
                release_transient_render_memory(dev, imem1);
            }
            for (uint32_t slot = 2; slot < color_count; ++slot) {
                if (!transient_extra[slot]) continue;   // retained for the next render group
                vkDestroyImageView(dev, extra_views[slot], nullptr);
                vkDestroyImage(dev, extra_images[slot], nullptr);
                release_transient_render_memory(dev, extra_memories[slot]);
            }
            if (transient_ds) {
                vkDestroyImageView(dev, dview, nullptr);
                vkDestroyImage(dev, dimg, nullptr);
                release_transient_render_memory(dev, dmem);
            }
            while (!backend_has_unproven_submission() &&
                   (persistent_color_target_cache().size() > persistent_color_target_count_limit() ||
                    persistent_color_target_bytes() > persistent_color_target_limit()) &&
                   evict_persistent_color_target(*ctx_ptr, color_target_generation)) {}
        });
    if (flush_now) active_submission.complete();
    if (timing_enabled) {
        const auto timing_done = TimingClock::now();
        auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        BackendRenderTimingStats& call_timing = backend_render_timing_stats_storage();
        call_timing.calls = 1;
        call_timing.draws = draws.size();
        call_timing.command_buffers = batch_result.command_buffers;
        call_timing.queue_submits = batch_result.queue_submits;
        call_timing.flush_no_batch = flush_reason_no_batch;
        call_timing.flush_readback = flush_reason_readback;
        call_timing.flush_storage_writeback = flush_reason_storage;
        call_timing.flush_explicit = flush_reason_explicit;
        call_timing.fence_waits = batch_result.fence_waits;
        call_timing.gpu_timestamp_samples = batch_result.gpu_timestamp_samples;
        call_timing.target_ms = ms(timing_start, timing_target_ready);
        call_timing.draw_setup_ms = ms(timing_target_ready, timing_draws_ready);
        call_timing.record_upload_ms = ms(timing_draws_ready, timing_recorded);
        call_timing.gpu_wait_ms = ms(timing_recorded, timing_gpu_done);
        call_timing.gpu_device_ms = batch_result.gpu_device_ms;
        call_timing.readback_ms = ms(timing_gpu_done, timing_readback_done);
        call_timing.cleanup_ms = ms(timing_readback_done, timing_done);
        call_timing.setup_shader_ms = setup_shader_ms;
        call_timing.setup_fixed_ms = setup_fixed_ms;
        call_timing.res_fixed_index_upload_ms = res_fixed_index_upload_ms;
        call_timing.res_fixed_blend_ms = res_fixed_blend_ms;
        call_timing.res_fixed_depth_stencil_ms = res_fixed_depth_stencil_ms;
        call_timing.res_fixed_viewport_ms = res_fixed_viewport_ms;
        call_timing.res_fixed_stages_ms = res_fixed_stages_ms;
        call_timing.res_fixed_prologue_ms = res_fixed_prologue_ms;
        call_timing.res_prologue_subgroup_scan_ms = res_prologue_subgroup_scan_ms;
        call_timing.setup_resources_ms = setup_resources_ms;
        call_timing.res_texture_ms = res_texture_ms;
        call_timing.res_texture_upload_ms = res_texture_upload_ms;
        call_timing.res_texture_bind_ms = res_texture_bind_ms;
        call_timing.res_buffer_ms = res_buffer_ms;
        call_timing.res_buffer_acquire_ms = res_buffer_acquire_ms;
        call_timing.res_buffer_copy_ms = res_buffer_copy_ms;
        call_timing.res_buffer_create_ms = res_buffer_create_ms;
        call_timing.res_buffer_index_find_ms = res_buffer_index_find_ms;
        call_timing.res_buffer_index_insert_ms = res_buffer_index_insert_ms;
        call_timing.res_buffer_hash_ms = res_buffer_hash_ms;
        call_timing.res_descriptor_ms = res_descriptor_ms;
        call_timing.setup_pipeline_ms = setup_pipeline_ms;
        // The live F8 caller consumes call_timing above. Only the explicit environment switch owns
        // the backend lifetime aggregates and optional periodic stderr windows below.
        if (!timing_mode.log) return out;
        struct TimingTotals {
            uint64_t calls = 0, draws = 0;
            uint64_t command_buffers = 0, queue_submits = 0, fence_waits = 0;
            uint64_t gpu_timestamp_samples = 0;
            uint64_t texture_references = 0, texture_uploads = 0, texture_upload_bytes = 0;
            uint64_t persistent_hits = 0, persistent_misses = 0, persistent_cached_bytes = 0;
            uint64_t texture_binding_references = 0, unique_texture_bindings = 0;
            uint64_t buffer_references = 0, unique_buffers = 0;
            uint64_t descriptor_layout_references = 0, unique_descriptor_layouts = 0;
            uint64_t pipeline_layout_references = 0, unique_pipeline_layouts = 0;
            uint64_t pipeline_references = 0, pipeline_hits = 0, pipeline_misses = 0;
            uint64_t pipeline_bypasses = 0, pipeline_entries = 0, pipeline_evictions = 0;
            double target = 0, draw_setup = 0, record = 0, gpu_wait = 0, gpu_device = 0;
            double readback = 0, cleanup = 0;
            double setup_shader = 0, setup_fixed = 0, setup_resources = 0, setup_pipeline = 0;
        };
        static TimingTotals totals;
        static TimingTotals window;
        auto accumulate = [&](TimingTotals& timing) {
            timing.calls++;
            timing.draws += draws.size();
            timing.command_buffers += call_timing.command_buffers;
            timing.queue_submits += call_timing.queue_submits;
            timing.fence_waits += call_timing.fence_waits;
            timing.gpu_timestamp_samples += call_timing.gpu_timestamp_samples;
            timing.texture_references += texture_stats.references;
            timing.texture_uploads += texture_stats.unique_uploads;
            timing.texture_upload_bytes += texture_stats.upload_bytes;
            timing.persistent_hits += texture_stats.persistent_hits;
            timing.persistent_misses += texture_stats.persistent_misses;
            timing.persistent_cached_bytes = texture_stats.persistent_cached_bytes;
            timing.texture_binding_references += resource_reuse_stats.texture_binding_references;
            timing.unique_texture_bindings += resource_reuse_stats.unique_texture_bindings;
            timing.buffer_references += resource_reuse_stats.buffer_references;
            timing.unique_buffers += resource_reuse_stats.unique_buffers;
            timing.descriptor_layout_references +=
                resource_reuse_stats.descriptor_set_layout_references;
            timing.unique_descriptor_layouts +=
                resource_reuse_stats.unique_descriptor_set_layouts;
            timing.pipeline_layout_references += resource_reuse_stats.pipeline_layout_references;
            timing.unique_pipeline_layouts += resource_reuse_stats.unique_pipeline_layouts;
            timing.pipeline_references += pipeline_stats.references;
            timing.pipeline_hits += pipeline_stats.hits;
            timing.pipeline_misses += pipeline_stats.misses;
            timing.pipeline_bypasses += pipeline_stats.bypasses;
            timing.pipeline_entries = pipeline_stats.entries;
            timing.pipeline_evictions += pipeline_stats.evictions;
            timing.target += call_timing.target_ms;
            timing.draw_setup += call_timing.draw_setup_ms;
            timing.record += call_timing.record_upload_ms;
            timing.gpu_wait += call_timing.gpu_wait_ms;
            timing.gpu_device += call_timing.gpu_device_ms;
            timing.readback += call_timing.readback_ms;
            timing.cleanup += call_timing.cleanup_ms;
            timing.setup_shader += call_timing.setup_shader_ms;
            timing.setup_fixed += call_timing.setup_fixed_ms;
            timing.setup_resources += call_timing.setup_resources_ms;
            timing.setup_pipeline += call_timing.setup_pipeline_ms;
        };
        accumulate(totals);
        accumulate(window);
        static const bool print_backend_timing_windows =
            getenv("PROSPER_BACKEND_TIMING_WINDOWS") != nullptr;
        if (print_backend_timing_windows && totals.calls % 25 == 0) {
            const double n = static_cast<double>(totals.calls);
            const double total = totals.target + totals.draw_setup + totals.record +
                                 totals.gpu_wait + totals.readback + totals.cleanup;
            fprintf(stderr,
                    "[render-timing] backend calls=%llu draws=%llu avg_ms: total=%.2f target=%.2f "
                    "draw_setup=%.2f record_upload=%.2f gpu_wait=%.2f gpu_device=%.2f "
                    "gpu_overhead=%.2f readback=%.2f cleanup=%.2f\n",
                    (unsigned long long)totals.calls, (unsigned long long)totals.draws, total / n,
                    totals.target / n, totals.draw_setup / n, totals.record / n,
                    totals.gpu_wait / n, totals.gpu_device / n,
                    std::max(0.0, totals.gpu_wait - totals.gpu_device) / n,
                    totals.readback / n, totals.cleanup / n);
            fprintf(stderr,
                    "[render-timing] backend synchronization command_buffers=%llu queue_submits=%llu "
                    "fence_waits=%llu gpu_timestamps=%llu\n",
                    (unsigned long long)totals.command_buffers,
                    (unsigned long long)totals.queue_submits,
                    (unsigned long long)totals.fence_waits,
                    (unsigned long long)totals.gpu_timestamp_samples);
            fprintf(stderr,
                    "[render-timing] draw_setup avg_ms: shaders=%.2f fixed=%.2f resources=%.2f pipeline=%.2f\n",
                    totals.setup_shader / n, totals.setup_fixed / n,
                    totals.setup_resources / n, totals.setup_pipeline / n);
            fprintf(stderr,
                    "[render-timing] backend textures refs=%llu uploads=%llu %.1f MiB "
                    "persistent=%llu/%llu cache=%.1f MiB\n",
                    (unsigned long long)totals.texture_references,
                    (unsigned long long)totals.texture_uploads,
                    totals.texture_upload_bytes / (1024.0 * 1024.0),
                    (unsigned long long)totals.persistent_hits,
                    (unsigned long long)totals.persistent_misses,
                    totals.persistent_cached_bytes / (1024.0 * 1024.0));
            fprintf(stderr,
                    "[render-timing] backend pipelines refs=%llu hits=%llu misses=%llu bypass=%llu "
                    "entries=%llu evictions=%llu\n",
                    (unsigned long long)totals.pipeline_references,
                    (unsigned long long)totals.pipeline_hits,
                    (unsigned long long)totals.pipeline_misses,
                    (unsigned long long)totals.pipeline_bypasses,
                    (unsigned long long)totals.pipeline_entries,
                    (unsigned long long)totals.pipeline_evictions);
            const RenderMemoryPoolStats pool = render_memory_pool_stats();
            fprintf(stderr,
                    "[render-timing] memory_pool hits=%llu misses=%llu cached=%zu %.1f MiB "
                    "discarded=%llu\n",
                    (unsigned long long)pool.hits, (unsigned long long)pool.misses,
                    pool.cached_allocations,
                    static_cast<double>(pool.cached_bytes) / (1024.0 * 1024.0),
                    (unsigned long long)pool.discarded);
            const RenderHostBufferPoolStats buffer_pool = render_host_buffer_pool_stats();
            fprintf(stderr,
                    "[render-timing] backend_buffer_pool hits=%llu misses=%llu cached=%zu %.1f MiB "
                    "evictions=%llu\n",
                    (unsigned long long)buffer_pool.hits,
                    (unsigned long long)buffer_pool.misses,
                    buffer_pool.cached_buffers,
                    static_cast<double>(buffer_pool.cached_bytes) / (1024.0 * 1024.0),
                    (unsigned long long)buffer_pool.evictions);
            const double wn = static_cast<double>(window.calls);
            const double window_total = window.target + window.draw_setup + window.record +
                                        window.gpu_wait + window.readback + window.cleanup;
            fprintf(stderr,
                    "[render-window] backend calls=%llu draws=%.1f avg_ms: total=%.2f target=%.2f "
                    "draw_setup=%.2f record_upload=%.2f gpu_wait=%.2f gpu_device=%.2f "
                    "gpu_overhead=%.2f readback=%.2f cleanup=%.2f\n",
                    (unsigned long long)window.calls, window.draws / wn, window_total / wn,
                    window.target / wn, window.draw_setup / wn, window.record / wn,
                    window.gpu_wait / wn, window.gpu_device / wn,
                    std::max(0.0, window.gpu_wait - window.gpu_device) / wn,
                    window.readback / wn, window.cleanup / wn);
            fprintf(stderr,
                    "[render-window] backend synchronization command_buffers=%.1f queue_submits=%.1f "
                    "fence_waits=%.1f gpu_timestamps=%.1f\n",
                    window.command_buffers / wn, window.queue_submits / wn,
                    window.fence_waits / wn, window.gpu_timestamp_samples / wn);
            fprintf(stderr,
                    "[render-window] draw_setup avg_ms: shaders=%.2f fixed=%.2f resources=%.2f pipeline=%.2f\n",
                    window.setup_shader / wn, window.setup_fixed / wn,
                    window.setup_resources / wn, window.setup_pipeline / wn);
            fprintf(stderr,
                    "[render-window] backend textures refs=%.1f uploads=%.1f %.1f MiB "
                    "persistent=%.1f/%.1f cache=%.1f MiB\n",
                    window.texture_references / wn, window.texture_uploads / wn,
                    window.texture_upload_bytes / (wn * 1024.0 * 1024.0),
                    window.persistent_hits / wn, window.persistent_misses / wn,
                    window.persistent_cached_bytes / (1024.0 * 1024.0));
            fprintf(stderr,
                    "[render-window] backend resources texture_bindings=%.1f/%.1f "
                    "buffers=%.1f/%.1f descriptor_layouts=%.1f/%.1f "
                    "pipeline_layouts=%.1f/%.1f\n",
                    window.texture_binding_references / wn,
                    window.unique_texture_bindings / wn,
                    window.buffer_references / wn, window.unique_buffers / wn,
                    window.descriptor_layout_references / wn,
                    window.unique_descriptor_layouts / wn,
                    window.pipeline_layout_references / wn,
                    window.unique_pipeline_layouts / wn);
            fprintf(stderr,
                    "[render-window] backend pipelines refs=%.1f hits=%.1f misses=%.1f bypass=%.1f "
                    "entries=%llu evictions=%.1f\n",
                    window.pipeline_references / wn, window.pipeline_hits / wn,
                    window.pipeline_misses / wn, window.pipeline_bypasses / wn,
                    (unsigned long long)window.pipeline_entries, window.pipeline_evictions / wn);
            window = {};
        }
    }
    // NB: dev/instance are the persistent RenderVkCtx — do NOT destroy them here (reused across calls).
    return out;
}

// A Vulkan render pass has one depth layout for every draw. If an earlier draw writes a persistent
// guest depth plane and a later draw samples that same plane, keeping both in one pass makes the
// attachment writable and the sampled-depth bridge must reject the alias. Split exactly at those
// write/sample transitions (in either direction): queue order plus the persistent color/DS caches
// preserve guest draw order, while each resulting pass has a legal, unambiguous depth layout.
inline size_t depth_feedback_split_index(std::span<const BackendDraw> draws,
                                         uint32_t W, uint32_t H) {
    std::unordered_set<uint64_t> sampled_depth;
    std::unordered_set<uint64_t> written_depth;
    for (size_t i = 0; i < draws.size(); ++i) {
        const BackendDraw& draw = draws[i];
        const bool samples_prior_write = std::any_of(
            draw.R.begin(), draw.R.end(), [&](const FrameResource& resource) {
                return resource.persistent_depth_target_id && resource.img_dim == 1u &&
                    resource.tw == W && resource.th == H &&
                    written_depth.contains(resource.persistent_depth_target_id);
            });
        const bool writes_depth = draw.ps && persistent_ds_pass_may_write_depth(
            draw.ps->depth_clear_enable, draw.ps->depth_test_enable,
            draw.ps->depth_write_enable, draw.ps->depth_compare_op);
        const bool writes_prior_sample = writes_depth &&
            ((draw.ps->depth_read_base && sampled_depth.contains(draw.ps->depth_read_base)) ||
             (draw.ps->depth_write_base && sampled_depth.contains(draw.ps->depth_write_base)));
        if (i && (samples_prior_write || writes_prior_sample)) return i;

        for (const FrameResource& resource : draw.R)
            if (resource.persistent_depth_target_id && resource.img_dim == 1u &&
                resource.tw == W && resource.th == H)
                sampled_depth.insert(resource.persistent_depth_target_id);
        if (writes_depth) {
            if (draw.ps->depth_read_base) written_depth.insert(draw.ps->depth_read_base);
            if (draw.ps->depth_write_base) written_depth.insert(draw.ps->depth_write_base);
        }
    }
    return draws.size();
}

// The contract one SEGMENT of a depth-feedback-split pass renders under.
//
// Extracted so the segment decisions are testable at the site that makes them. Every field here was
// once derived inline, and two of them were wrong in ways no end-to-end assertion caught: non-final
// segments were handed `mrt_outputs == nullptr`, so their colour_count collapsed to one attachment
// and every MRT1..7 export in them was discarded; and later segments never asked to LOAD the higher
// slots, so a fresh slot cleared at the second segment.
//
// The invariant the tests pin: the SHAPE is identical across segments and only the pixel
// destination and the load/readback flags differ.
struct SplitSegmentContract {
    BackendColorTarget target{};
    uint32_t color_count = 1;
    bool has_target = false;
};
inline SplitSegmentContract split_segment_contract(
    const BackendColorTarget* whole, const BackendMrtOutputs* whole_mrt, bool first, bool final,
    const std::array<const uint8_t*, prosper::gpu::kColorTargetCount>& carried_slots = {}) {
    SplitSegmentContract out;
    // The attachment shape never varies by segment. A pass is five-MRT for all of its segments or
    // none of them; splitting is a synchronisation boundary, not a change of render target.
    out.color_count = whole_mrt ? whole_mrt->color_count : 1u;
    // A caller that named no persistent identities still needs a target when higher slots have to
    // be carried between segments: the seed and readback flags live on it, and they are the only
    // channel by which a non-persistent MRT2+ survives a physical pass boundary. A synthesised
    // target is behaviourally identical to no target -- every identity is zero, so `persistent_color`
    // and its siblings stay false and no cached image is consulted -- except that it can carry.
    // > 1, not > 2: slot 1 needs carrying too when the caller takes it through BackendMrtOutputs.
    // The threshold was 2 and a two-attachment split therefore synthesised no carrier at all, so
    // MRT1 was cleared by the next segment even with every other part of the carry correct.
    const bool carries_slots = out.color_count > 1u;
    if (!whole && !carries_slots) return out;
    if (whole) out.target = *whole;
    out.has_target = true;
    if (!first) {
        // Every later segment LOADS everything, or it erases what its predecessors drew.
        out.target.load_existing = true;
        out.target.load_existing1 = true;
        out.target.load_existing_slots.fill(true);
        // LOADing is only sufficient when the slot's image actually SURVIVES, and a non-zero
        // identity does not establish that: persistence also requires
        // persistent_color_targets_enabled (an env switch) and a successful cache/budget
        // allocation, and the slot-creation path explicitly falls back to a transient image while
        // leaving the identity non-zero. Seeding whatever the previous segment produced is correct
        // in every one of those cases and merely redundant when the image did survive, so the
        // decision is made conservatively rather than from an identity that cannot answer it.
        for (uint32_t slot = 2; slot < prosper::gpu::kColorTargetCount; ++slot)
            out.target.seed_slots[slot] = carried_slots[slot];
        // Slot 1 travels the same way when the caller takes it through BackendMrtOutputs rather
        // than out_rgba1 -- see the carry below.
        if (carried_slots[1]) out.target.seed_rgba1_slot = carried_slots[1];
    }
    if (!final) {
        // A non-final segment's slot-0 pixels are discarded, so do not copy them out.
        out.target.readback = false;
        out.target.readback1 = false;
        out.target.readback_slots.fill(false);
        // ...but every slot this segment may have to CARRY must be copied out, or there is nothing
        // to seed the next segment with. That is slot 1 (which the live renderer takes through
        // BackendMrtOutputs) and every active slot 2+, unconditionally: whether a slot's image
        // actually survives depends on persistent_color_targets_enabled and on a cache/budget
        // allocation that can fall back to a transient image while leaving the identity non-zero,
        // and none of that is visible from here.
        //
        // Decided HERE rather than by the caller amending the result. It was written as an
        // override immediately after this function returned, which left the helper's own test
        // asserting `non-final segments read back no slot` -- the exact opposite of the effective
        // contract, in the test whose job is to document it.
        for (uint32_t slot = 2; slot < out.color_count; ++slot)
            out.target.readback_slots[slot] = true;
        if (out.color_count > 1) out.target.readback1 = true;
    } else if (!whole) {
        // A synthesised carrier target on the final segment represents a caller with whole == nullptr.
        // It must read back the requested outputs just as a whole == nullptr non-split pass does.
        out.target.readback = true;
        out.target.readback1 = out.color_count > 1;
        for (uint32_t slot = 2; slot < out.color_count; ++slot)
            out.target.readback_slots[slot] = true;
    }
    return out;
}

// Logical multi-draw entry. Most calls remain one Vulkan render pass. A depth feedback transition
// becomes multiple ordered passes without copying the (often large) BackendDraw shader/resource
// payloads. Intermediate color is retained on the GPU when possible and carried through CPU pixels
// only on the established non-persistent/MRT path.
inline std::vector<uint8_t> render_draws_rgba(const std::vector<BackendDraw>& draws,
                                              uint32_t W, uint32_t H,
                                              const uint8_t* seed_rgba = nullptr,
                                              const float* clear_rgba = nullptr,
                                              bool persist_depth_stencil = false,
                                              const BackendColorTarget* color_target = nullptr,
                                              const uint8_t* seed_rgba1 = nullptr,
                                              const float* clear_rgba1 = nullptr,
                                              std::vector<uint8_t>* out_rgba1 = nullptr,
                                              BackendSubmissionBatch* submission_batch = nullptr,
                                              bool flush_submission_batch = true,
                                              BackendMrtOutputs* mrt_outputs = nullptr,
                                              bool want_color_readback = true) {   // #2283
    const std::span<const BackendDraw> all(draws);
    if (!persist_depth_stencil ||
        depth_feedback_split_index(all, W, H) == all.size())
        return render_draw_pass_rgba(all, W, H, seed_rgba, clear_rgba,
                                     persist_depth_stencil, color_target, seed_rgba1,
                                     clear_rgba1, out_rgba1, submission_batch,
                                     flush_submission_batch, {}, mrt_outputs,
                                     want_color_readback);

    BackendColorTargetStats aggregate_color{};
    BackendTextureUploadStats aggregate_textures{};
    BackendResourceReuseStats aggregate_resources{};
    BackendPipelineCacheStats aggregate_pipelines{};
    BackendRenderTimingStats aggregate_timing{};
    std::vector<uint8_t> carried_color0;
    std::array<std::vector<uint8_t>, prosper::gpu::kColorTargetCount> carried_slots;
    std::array<const uint8_t*, prosper::gpu::kColorTargetCount> carried_slot_ptrs{};
    std::vector<uint8_t> carried_color1;
    const uint8_t* next_seed0 = seed_rgba;
    const uint8_t* next_seed1 = seed_rgba1;

    auto add_stats = [&] {
        const BackendColorTargetStats color = backend_color_target_stats();
        aggregate_color.writes += color.writes;
        aggregate_color.write_hits += color.write_hits;
        aggregate_color.sampled_hits += color.sampled_hits;
        aggregate_color.readbacks += color.readbacks;
        aggregate_color.cached_bytes = color.cached_bytes;
        aggregate_color.cached_entries = color.cached_entries;

        const BackendTextureUploadStats textures = backend_texture_upload_stats();
        aggregate_textures.references += textures.references;
        aggregate_textures.unique_uploads += textures.unique_uploads;
        aggregate_textures.upload_bytes += textures.upload_bytes;
        aggregate_textures.persistent_hits += textures.persistent_hits;
        aggregate_textures.persistent_misses += textures.persistent_misses;
        aggregate_textures.persistent_cached_bytes = textures.persistent_cached_bytes;

        const BackendResourceReuseStats resources = backend_resource_reuse_stats();
#define PROSPER_SUM_RESOURCE_STAT(field) aggregate_resources.field += resources.field
        PROSPER_SUM_RESOURCE_STAT(buffer_references);
        PROSPER_SUM_RESOURCE_STAT(unique_buffers);
        PROSPER_SUM_RESOURCE_STAT(texture_binding_references);
        PROSPER_SUM_RESOURCE_STAT(unique_texture_bindings);
        PROSPER_SUM_RESOURCE_STAT(descriptor_set_layout_references);
        PROSPER_SUM_RESOURCE_STAT(unique_descriptor_set_layouts);
        PROSPER_SUM_RESOURCE_STAT(pipeline_layout_references);
        PROSPER_SUM_RESOURCE_STAT(unique_pipeline_layouts);
        PROSPER_SUM_RESOURCE_STAT(descriptor_pools);
        PROSPER_SUM_RESOURCE_STAT(persistent_pipeline_layout_hits);
        PROSPER_SUM_RESOURCE_STAT(persistent_pipeline_layout_misses);
        PROSPER_SUM_RESOURCE_STAT(persistent_pipeline_layout_evictions);
        PROSPER_SUM_RESOURCE_STAT(persistent_texture_binding_hits);
        PROSPER_SUM_RESOURCE_STAT(persistent_texture_binding_misses);
        PROSPER_SUM_RESOURCE_STAT(persistent_texture_binding_evictions);
        PROSPER_SUM_RESOURCE_STAT(buffer_hash_calls);
        PROSPER_SUM_RESOURCE_STAT(buffer_hash_dwords);
        PROSPER_SUM_RESOURCE_STAT(buffer_hash_skipped_unique);
        PROSPER_SUM_RESOURCE_STAT(buffer_hash_skipped_large);
        PROSPER_SUM_RESOURCE_STAT(buffer_ref_memo_hits);
        PROSPER_SUM_RESOURCE_STAT(buffer_skipped_large_dwords);
        PROSPER_SUM_RESOURCE_STAT(buffer_upload_bytes);
#undef PROSPER_SUM_RESOURCE_STAT
        aggregate_resources.persistent_pipeline_layout_entries =
            resources.persistent_pipeline_layout_entries;
        aggregate_resources.persistent_texture_binding_entries =
            resources.persistent_texture_binding_entries;

        const BackendPipelineCacheStats pipelines = backend_pipeline_cache_stats();
        aggregate_pipelines.references += pipelines.references;
        aggregate_pipelines.hits += pipelines.hits;
        aggregate_pipelines.misses += pipelines.misses;
        aggregate_pipelines.bypasses += pipelines.bypasses;
        aggregate_pipelines.entries = pipelines.entries;
        aggregate_pipelines.evictions += pipelines.evictions;

        if (getenv("PROSPER_RENDER_TIMING")) {
            const BackendRenderTimingStats timing = backend_render_timing_stats();
#define PROSPER_SUM_TIMING_STAT(field) aggregate_timing.field += timing.field
            PROSPER_SUM_TIMING_STAT(calls);
            PROSPER_SUM_TIMING_STAT(draws);
            PROSPER_SUM_TIMING_STAT(command_buffers);
            PROSPER_SUM_TIMING_STAT(queue_submits);
            PROSPER_SUM_TIMING_STAT(fence_waits);
            PROSPER_SUM_TIMING_STAT(gpu_timestamp_samples);
            PROSPER_SUM_TIMING_STAT(target_ms);
            PROSPER_SUM_TIMING_STAT(draw_setup_ms);
            PROSPER_SUM_TIMING_STAT(record_upload_ms);
            PROSPER_SUM_TIMING_STAT(gpu_wait_ms);
            PROSPER_SUM_TIMING_STAT(gpu_device_ms);
            PROSPER_SUM_TIMING_STAT(readback_ms);
            PROSPER_SUM_TIMING_STAT(cleanup_ms);
            PROSPER_SUM_TIMING_STAT(setup_shader_ms);
            PROSPER_SUM_TIMING_STAT(setup_fixed_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_index_upload_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_blend_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_depth_stencil_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_viewport_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_stages_ms);
            PROSPER_SUM_TIMING_STAT(res_fixed_prologue_ms);
            PROSPER_SUM_TIMING_STAT(flush_no_batch);
            PROSPER_SUM_TIMING_STAT(flush_readback);
            PROSPER_SUM_TIMING_STAT(flush_storage_writeback);
            PROSPER_SUM_TIMING_STAT(flush_explicit);
            PROSPER_SUM_TIMING_STAT(res_prologue_subgroup_scan_ms);
            PROSPER_SUM_TIMING_STAT(setup_resources_ms);
            PROSPER_SUM_TIMING_STAT(res_texture_ms);
            PROSPER_SUM_TIMING_STAT(res_texture_upload_ms);
            PROSPER_SUM_TIMING_STAT(res_texture_bind_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_acquire_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_copy_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_create_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_index_find_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_index_insert_ms);
            PROSPER_SUM_TIMING_STAT(res_buffer_hash_ms);
            PROSPER_SUM_TIMING_STAT(res_descriptor_ms);
            PROSPER_SUM_TIMING_STAT(setup_pipeline_ms);
#undef PROSPER_SUM_TIMING_STAT
        }
    };

    size_t begin = 0;
    while (begin < all.size()) {
        const std::span<const BackendDraw> remaining = all.subspan(begin);
        const size_t relative_end = depth_feedback_split_index(remaining, W, H);
        const size_t end = begin + relative_end;
        const bool final = end == all.size();

        const SplitSegmentContract segment = split_segment_contract(
            color_target, mrt_outputs, begin == 0, final, carried_slot_ptrs);
        const BackendColorTarget* segment_target_ptr =
            segment.has_target ? &segment.target : nullptr;
        std::vector<uint8_t> intermediate_color1;
        std::vector<uint8_t>* segment_out1 = out_rgba1
            ? (final ? out_rgba1 : &intermediate_color1) : nullptr;
        // The shape travels with every segment; only the pixel destination differs.
        BackendMrtOutputs intermediate_mrt;
        intermediate_mrt.color_count = segment.color_count;
        BackendMrtOutputs* segment_mrt = mrt_outputs
            ? (final ? mrt_outputs : &intermediate_mrt) : nullptr;
        std::vector<uint8_t> rendered = render_draw_pass_rgba(
            all.subspan(begin, end - begin), W, H, next_seed0,
            begin ? nullptr : clear_rgba, true, segment_target_ptr, next_seed1,
            begin ? nullptr : clear_rgba1, segment_out1, submission_batch,
            final ? flush_submission_batch : false, all,
            segment_mrt, want_color_readback);
        add_stats();

        if (final) {
            backend_color_target_stats_storage() = aggregate_color;
            backend_texture_upload_stats_storage() = aggregate_textures;
            backend_resource_reuse_stats_storage() = aggregate_resources;
            backend_pipeline_cache_stats_storage() = aggregate_pipelines;
            if (getenv("PROSPER_RENDER_TIMING"))
                backend_render_timing_stats_storage() = aggregate_timing;
            return rendered;
        }

        // A persistent color target intentionally returns no pixels here; the next pass LOADs its
        // queued image. Transient and MRT paths read back and carry their exact output as a seed.
        if (!rendered.empty()) carried_color0 = std::move(rendered);
        else carried_color0.clear();
        next_seed0 = carried_color0.empty() ? nullptr : carried_color0.data();
        // Slot 1 arrives in `intermediate_color1` under the out_rgba1 API and in
        // `intermediate_mrt.colors[1]` under the BackendMrtOutputs API. Carry whichever is
        // populated: keying on `out_rgba1` alone dropped MRT1 on every non-final segment of a live
        // renderer split, because that caller uses the other API.
        if (out_rgba1 && !intermediate_color1.empty())
            carried_color1 = std::move(intermediate_color1);
        else if (!intermediate_mrt.colors[1].empty())
            carried_color1 = std::move(intermediate_mrt.colors[1]);
        else
            carried_color1.clear();
        next_seed1 = carried_color1.empty() ? nullptr : carried_color1.data();
        // Carry each non-persistent higher slot's pixels into the next segment. Without this the
        // final segment created fresh transient attachments and cleared away everything earlier
        // segments drew into MRT2+ -- silently, since the shape and the flags were already right.
        for (uint32_t slot = 2; slot < prosper::gpu::kColorTargetCount; ++slot) {
            if (final) break;
            carried_slots[slot] = std::move(intermediate_mrt.colors[slot]);
            carried_slot_ptrs[slot] =
                carried_slots[slot].empty() ? nullptr : carried_slots[slot].data();
        }
        // Slot 1's pointer travels in the same table, so the contract can seed it without a second
        // channel. `next_seed1` still serves the out_rgba1 API; both end at the same upload.
        carried_slot_ptrs[1] = out_rgba1 ? nullptr : next_seed1;
        begin = end;
    }
    return {};
}

// Single-draw entry — a thin wrapper over render_draws_rgba, preserving the exact signature/behavior the
// recompiled-shader render tests rely on. Builds one draw's set-tagged resources (`gres`, or the legacy
// cbuf/vbuf@bindings 2/3 mirrored into both sets + optional `tex` in set 1).
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr,
                                                 const std::vector<uint32_t>* vbuf = nullptr,
                                                 const std::vector<uint32_t>* cbuf = nullptr,
                                                 const TexDesc* tex = nullptr,
                                                 const std::vector<FrameResource>* gres = nullptr,
                                                 uint32_t vcount = 3) {
    BackendDraw d; d.vs = vert; d.fs = frag; d.ps = ps; d.vcount = vcount;
    if (gres && !gres->empty()) { d.R = *gres; }
    else {
        for (uint32_t s2 = 0; s2 < 2; s2++) {
            FrameResource b2; b2.binding = 2; b2.set = s2; if (cbuf) b2.dwords = *cbuf; d.R.push_back(std::move(b2));
            FrameResource b3; b3.binding = 3; b3.set = s2; if (vbuf) b3.dwords = *vbuf; d.R.push_back(std::move(b3));
        }
        if (tex) { FrameResource t; t.binding = tex->binding; t.set = 1; t.tex_rgba = tex->rgba; t.tw = tex->w; t.th = tex->h; t.max_aniso_ratio = tex->max_aniso_ratio; d.R.push_back(std::move(t)); }
    }
    return render_draws_rgba({std::move(d)}, W, H);
}

} // namespace prosper::test
