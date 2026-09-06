// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "shared/live/live_renderer.hpp"
#include "diagnostics/env_cache.hpp"   // PROSPER_ENV_ON / _VALUE: cached reads on per-draw paths
#include "diagnostics/env_numeric.hpp" // #3253: a typo must not select a different setting
#include "gpu/resources/metadata_kind_correlation.hpp"  // positive metadata-kind correlation (pure, tested)
#include "gpu/diagnostics/watch_list.hpp"                 // strict 0x-only watch parsing
#include "gpu/diagnostics/draw_program_skip.hpp"          // PROSPER_SKIP_DRAW_PROGRAM / census
#include "gpu/diagnostics/link_list_census.hpp"          // PROSPER_DRAW_LINKSCAN
#include "gpu/capture/writer_provenance.hpp"              // who last wrote a censused range
#include "shared/rtt/rtt_authority.hpp"
#include "shared/rtt/rtt_injection.hpp"
#include "shared/rtt/rtt_scale.hpp"
#include "shared/rtt/mrt_extent.hpp"
#include "shared/rtt/mrt_binding.hpp"               // which MRT slot may join a pass (measured extents only)
#include "shared/present/readback_policy.hpp"
#include "shared/diagnostics/capture_renderer_policy.hpp"
#include "shared/texture/write_watch_policy.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/decode_scratch.hpp"     // pooled full-surface decode intermediates
#include "shared/live/live_target_format.hpp"       // the one LiveTargetPixelFormat mapping (exhaustive)
#include "shared/perf/performance_capture.hpp"      // bounded F8 post-trigger renderer timing
#include "shared/perf/performance_timing_gate.hpp"  // turn on render_runner's existing backend clocks
#include "shared/perf/performance_timing_policy.hpp" // retain timing across split semantic submits

#include "gpu/execute/gpu_execute.hpp"          // DrawItem, set_submit_renderer
#include "gpu/timeline/gpu_timeline.hpp"         // phase-gated detailed-capture policy
#include "gpu/capture/writer_provenance.hpp"
#include "gpu/capture/gpu_capture.hpp"          // temporal RTT capture/replay seeds
#include "gpu/texture/guest_texture_layout.hpp" // exact pitch for HLE-produced guest textures
#include "gpu/texture/tile.hpp"                 // detile_surface / tiled_surface_bytes / detile_elements
#include "gpu/texture/bc_decode.hpp"            // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/resources/shader_resources.hpp"     // ShaderResourceTable / ResourceClass
#include "gpu/recompiler/rdna2_to_spirv.hpp"       // recompile_fragment (diagnostic solid-color PS)
#include "gpu/present/videoout_present.hpp"     // present_front_index (flip-anchored present selection)
#include "shared/present/present_blit.hpp"             // GPU scanout handoff (#1270 unified-device present)
#include "shared/present/present_blit_policy.hpp"      // flip-anchored scanout publication policy
#include "shared/present/present_extent.hpp"           // the publish extent contract with the caller (#1986)
#include "shared/media/avplayer_plane_policy.hpp"    // which sampled resource is AvPlayer's NV12 chroma plane
#include "shared/present/guest_scanout_present.hpp"    // publishing the guest's own flipped buffer (#1968)
#include "shared/diagnostics/diagnostic_window.hpp"        // census window by callback ordinal or by elapsed time
#include "gpu/diagnostics/diag_ratelimit.hpp"       // ordinal + sparse tail for capped diagnostics
#include "host/memory/guest_write_watch.hpp"
#include "fixtures/render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp

#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <tuple>
#include <map>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);
#ifdef _WIN32
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);
#endif
// VideoOut scanout registry (hle_graphics.cpp) — which guest buffer the game most recently FLIPPED
// to screen. The flip fires during the Dcb fold (agc_dcb_set_flip -> prosper_vo_flip_from_gpu),
// BEFORE the submit's execute_and_present, so at render time these identify this frame's scanout VA.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint64_t prosper_vo_buffer_addr(int i);
extern "C" uint64_t prosper_vo_flip_count();

namespace prosper::frontend {

// Render-to-texture surface cache (#167): CB_COLOR0_BASE -> the RGBA pixels we last rendered into it.
// The game renders its scene into a color target then samples that address as a texture in a later
// composite pass. Guest memory at that address is never populated on our (CPU-read) side, so without
// this the composite samples zeros and the frame is black. We cache each submit's rendered pixels under
// its render-target base and inject them when a subsequent draw samples a texture at a matching base.
namespace {
// Default ceiling on a single non-texture (vertex/index/storage/constant) buffer upload. This is
// not borrowed from any other path — it exists only to bound a corrupt descriptor, and it is sized
// against what a 64 MiB read already costs elsewhere (~16K guest_readable page probes). A guest
// descriptor legitimately declares multi-megabyte vertex streams, and anything clamped away reads
// as zeros in the shader and collapses geometry, so the bound stays far above real content and any
// short upload is reported (#1427). PROSPER_MAX_BUFFER_UPLOAD_MB can lower it for an A/B.
constexpr uint32_t kMaxBufferUploadBytes = 64u << 20;

struct RttSurf {
    std::shared_ptr<const std::vector<uint8_t>> rgba;
    // Uniform DCC fast-clears remain compact until a consumer genuinely needs CPU bytes. Graphics
    // sampling and attachment LOADs can realize the value directly on the GPU.
    bool has_uniform_color = false;
    std::array<float, 4> uniform_color{};
    uint32_t w = 0, h = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    // Raw guest CB_COLOR format before backend_color_format() canonicalizes the Vulkan attachment.
    // Consumers need this to compose their T# DST_SEL with the host image's component order.
    VkFormat guest_format = VK_FORMAT_UNDEFINED;
    bool gpu_valid = false;
    // A color target can be cleared by a compute write to its DCC metadata rather than by a
    // color-plane write. Remember the sampled descriptor's metadata range so that write can
    // invalidate the retained CPU/GPU target just like a write to the color plane itself.
    uint64_t dcc_metadata_addr = 0;
    uint64_t dcc_metadata_bytes = 0;
    bool dcc_metadata_dirty = false;
};

bool materialize_uniform_rtt(RttSurf& surface) {
    if (!surface.has_uniform_color || !surface.w || !surface.h) return false;
    const VkFormat format = prosper::test::backend_color_format(surface.format);
    const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(format);
    const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
    if (!bpp || texels > SIZE_MAX / bpp) return false;
    std::vector<uint8_t> pixels(static_cast<size_t>(texels) * bpp);
    if (format == VK_FORMAT_R8G8B8A8_UNORM) {
        uint8_t native[4];
        for (uint32_t channel = 0; channel < 4; ++channel)
            native[channel] = static_cast<uint8_t>(std::clamp(
                surface.uniform_color[channel], 0.0f, 1.0f) * 255.0f + 0.5f);
        prosper::frontend::fill_repeating_pixel(pixels, native, sizeof(native));
    } else if (format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        uint16_t native[4];
        for (uint32_t channel = 0; channel < 4; ++channel)
            native[channel] = prosper::gpu::float_to_half(surface.uniform_color[channel]);
        prosper::frontend::fill_repeating_pixel(
            pixels, reinterpret_cast<const uint8_t*>(native), sizeof(native));
    } else if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        const uint32_t native =
            static_cast<uint32_t>(prosper::gpu::float_to_f11(surface.uniform_color[0])) |
            (static_cast<uint32_t>(prosper::gpu::float_to_f11(surface.uniform_color[1])) << 11) |
            (static_cast<uint32_t>(prosper::gpu::float_to_f10(surface.uniform_color[2])) << 22);
        prosper::frontend::fill_repeating_pixel(
            pixels, reinterpret_cast<const uint8_t*>(&native), sizeof(native));
    } else {
        return false;
    }
    surface.rgba = std::make_shared<const std::vector<uint8_t>>(std::move(pixels));
    return true;
}

using RttCache = std::unordered_map<uint64_t, RttSurf>;

// A queued guest write carries WHO made it. The origin is a thread-local set around
// notify_guest_gpu_write, but DS/RTT invalidation runs later at drain time -- on whatever thread
// drains, long after that thread-local was reset. So an invalidation diagnostic that reads the
// thread-local reports a constant: measured on a routed GTA V boot, `[ds] invalidate` printed
// `origin=gpu` for 30,458 of 30,458 invalidations, including ones made by writers that DO tag
// themselves. The field looked like attribution and was structurally incapable of it.
//
// Capturing the origin at QUEUE time is what makes it real, and it is why the origin must live in
// the record rather than be re-read at the far end of the seam.
struct PendingGuestGpuWrite {
    uint64_t addr = 0;
    uint64_t size = 0;
    const char* origin = "unknown";   // string literal or static storage; never an owned buffer
};

struct PendingGuestGpuWrites {
    std::mutex mutex;
    std::vector<PendingGuestGpuWrite> ranges;
    bool overflowed = false;
};

PendingGuestGpuWrites& pending_guest_gpu_writes() {
    static PendingGuestGpuWrites pending;
    return pending;
}

void queue_guest_gpu_write(uint64_t addr, uint64_t size, const char* origin) {
    auto& pending = pending_guest_gpu_writes();
    // The origin arrives AS DATA from the notifier. Reading the thread-local here instead would be
    // correct for the plain notifier and wrong for the byte-preserving one, which knows a
    // classification the thread-local never carried.
    if (!origin) origin = "unknown";
    std::lock_guard<std::mutex> lock(pending.mutex);
    constexpr size_t kMaxPendingRanges = 65536;
    if (pending.ranges.size() < kMaxPendingRanges)
        pending.ranges.push_back(PendingGuestGpuWrite{addr, size, origin});
    else
        pending.overflowed = true;
}

// Does this draw bind `addr` as ANY active colour target?
//
// The direct-live sampling decision excluded `draw.color0_base` alone. That was complete only while
// slots above 1 could not be persistent; they can now, so sampling an ACTIVE MRT2+ target would
// otherwise borrow the very image the draw is writing -- one VkImage used as shader-read and colour
// attachment at once, bypassing the CPU fallback that exists for exactly this case.
//
// "Active" matches the pass-grouping definition: a bound base whose write mask is non-zero. Slots 0
// and 1 keep their named-field fallbacks, since a draw may carry either form.
// The backend's notion of a defined colour format, supplied to the shared policy so that header
// stays backend-free. One mapping, used by grouping and feedback alike.
bool mrt_format_defined(uint32_t raw) {
    return prosper::test::backend_color_format(static_cast<VkFormat>(raw)) != VK_FORMAT_UNDEFINED;
}

bool draw_binds_color_target(const prosper::gpu::DrawItem& draw, uint64_t addr,
                             uint32_t sampled_width = 0u, uint32_t sampled_height = 0u) {
    return prosper::frontend::mrt_draw_binds_target_view(
        draw, addr, sampled_width, sampled_height, mrt_format_defined);
}

// PROSPER_RTT_INVALIDATE_WATCH=<0xaddr>[,<0xaddr>…] — why a renderer-owned surface's CPU snapshot is,
// or is not, being invalidated.
//
// The question it answers is the one left when a composite input is empty and the renderer serves its
// own snapshot for it. `PROSPER_RTT_GUESTPEEK` cannot answer it: every renderer-owned surface reports
// 0% non-zero GUEST bytes whether it renders correctly or not, because the renderer owns the pixels.
// So "the guest memory is empty" is not evidence about production, and the live question becomes
// whether the cached copy is ever refreshed.
//
// It reports both directions, which is the point. A write that OVERLAPS prints its effect and its
// origin; a write that misses prints nothing, but the periodic line states how many writes were
// examined and how many hit — so "never invalidated" is distinguishable from "the instrument never
// ran", and a watched address absent from the cache says so explicitly rather than looking like a
// silent zero.
//
// Addresses are parsed by the shared STRICT parser (`watch_list.hpp`), which requires an explicit `0x`,
// full token consumption, no overflow and no zero address, and arms NOTHING on a malformed spec. An
// earlier revision of this comment claimed `strtoull(..., 0)` was deliberate "so `0x` is required" —
// that was simply false, since base 0 falls back to DECIMAL, and the comment made the bug read as
// checked. The same trap is recorded for PROSPER_TARGET_WATCH, which now uses the same parser.
struct RttInvalidateWatch {
    std::vector<uint64_t> addrs;
    std::atomic<uint64_t> writes_examined{0};
    std::atomic<uint64_t> writes_hitting{0};
};

// The counters are atomics, so the watch is populated IN PLACE rather than returned from an
// initializer lambda -- an atomic is neither copyable nor movable.
std::vector<uint64_t> parse_rtt_invalidate_watch_addrs() {
    std::vector<uint64_t> addrs;
    const char* spec = getenv("PROSPER_RTT_INVALIDATE_WATCH");
    if (!spec || !*spec) return addrs;
    // Strict: an explicit 0x prefix, full consumption, no overflow, no zero address. A rejected spec
    // arms NOTHING and says so, rather than watching a decimal-parsed address and reporting success.
    if (!prosper::gpu::parse_hex_watch_list(spec, addrs)) {
        fprintf(stderr,
                "[rtt-inval] ignoring malformed PROSPER_RTT_INVALIDATE_WATCH=\"%s\" "
                "(expected 0x-prefixed hex addresses, comma separated) -- NOT armed\n", spec);
        addrs.clear();
        return addrs;
    }
    fprintf(stderr, "[rtt-inval] watching %zu address(es)\n", addrs.size());
    return addrs;
}

RttInvalidateWatch& rtt_invalidate_watch() {
    static RttInvalidateWatch watch;
    static const bool once = [] {
        watch.addrs = parse_rtt_invalidate_watch_addrs();
        return true;
    }();
    (void)once;
    return watch;
}

const char* rtt_guest_write_effect_name(prosper::frontend::LiveRttGuestWriteEffect effect) {
    switch (effect) {
        case prosper::frontend::LiveRttGuestWriteEffect::color_plane:   return "color-plane";
        case prosper::frontend::LiveRttGuestWriteEffect::dcc_metadata:  return "dcc-metadata";
        default:                                                       return "none";
    }
}

// #3407 reset-site census: which code path nulls a renderer surface's CPU publication. Shares
// PROSPER_RTT_REMAT_LOG with the read-side census so one run answers both halves -- "is the readback
// repeated" and "what invalidates it".
inline void rtt_remat_note_reset(const char* site, uint64_t addr) {
    static const bool on = std::getenv("PROSPER_RTT_REMAT_LOG") != nullptr;
    if (!on) return;
    static std::mutex m;
    static std::map<std::pair<std::string, uint64_t>, uint64_t> rows;
    std::lock_guard<std::mutex> lk(m);
    uint64_t& n = rows[{std::string(site), addr}];
    if (++n % 32 == 0)
        std::fprintf(stderr, "[rtt-reset] site=%s addr=0x%llx count=%llu\n",
                     site, (unsigned long long)addr, (unsigned long long)n);
}

void invalidate_cpu_rtt_guest_write(RttCache& cache, uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    auto& watch = rtt_invalidate_watch();
    if (!watch.addrs.empty()) {
        watch.writes_examined.fetch_add(1, std::memory_order_relaxed);
        bool write_touched_a_watched_surface = false;
        for (const uint64_t wanted : watch.addrs) {
            const auto entry = cache.find(wanted);
            if (entry == cache.end()) {
                // A watched address that is not a cached RTT surface at all. Reported once so an
                // absence of invalidations is not read as "nothing ever wrote it".
                static std::mutex absent_mutex;
                static std::set<uint64_t> absent_reported;
                bool first = false;
                { std::lock_guard<std::mutex> lock(absent_mutex);
                  first = absent_reported.insert(wanted).second; }
                if (first)
                    fprintf(stderr,
                            "[rtt-inval] 0x%llx is NOT in the RTT cache at this drain -- no snapshot "
                            "to invalidate\n", (unsigned long long)wanted);
                continue;
            }
            const RttSurf& surface = entry->second;
            const uint64_t bpp = prosper::test::backend_color_bytes_per_pixel(surface.format);
            const uint64_t pixels = static_cast<uint64_t>(surface.w) * surface.h;
            const uint64_t bytes = pixels > UINT64_MAX / bpp ? UINT64_MAX : pixels * bpp;
            const auto effect = prosper::frontend::live_rtt_guest_write_effect(
                wanted, bytes, surface.dcc_metadata_addr, surface.dcc_metadata_bytes, addr, size);
            if (effect == prosper::frontend::LiveRttGuestWriteEffect::none) continue;
            // Counted ONCE per queued write, not once per matching address: with several watched
            // addresses a per-address counter can exceed writes_examined, which makes the ratio
            // nonsense in exactly the summary line that exists to make a zero trustworthy.
            if (!write_touched_a_watched_surface) {
                write_touched_a_watched_surface = true;
                watch.writes_hitting.fetch_add(1, std::memory_order_relaxed);
            }
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 64)
                fprintf(stderr,
                        "[rtt-inval] 0x%llx %ux%u snapshot invalidated: effect=%s by write "
                        "addr=0x%llx size=%llu origin=%s\n",
                        (unsigned long long)wanted, surface.w, surface.h,
                        rtt_guest_write_effect_name(effect),
                        (unsigned long long)addr, (unsigned long long)size,
                        prosper::gpu::guest_gpu_write_origin());
        }
        // Periodic totals, so a zero is a measurement rather than a silence.
        const uint64_t examined = watch.writes_examined.load(std::memory_order_relaxed);
        if (examined && (examined & (examined - 1)) == 0 && examined >= 1024)
            fprintf(stderr, "[rtt-inval] examined=%llu writes, %llu touched a watched surface\n",
                    (unsigned long long)examined,
                    (unsigned long long)watch.writes_hitting.load(std::memory_order_relaxed));
    }
    for (auto it = cache.begin(); it != cache.end();) {
        const RttSurf& surface = it->second;
        const uint64_t bpp = prosper::test::backend_color_bytes_per_pixel(surface.format);
        const uint64_t pixels = static_cast<uint64_t>(surface.w) * surface.h;
        const uint64_t bytes = pixels > UINT64_MAX / bpp ? UINT64_MAX : pixels * bpp;
        const auto effect = prosper::frontend::live_rtt_guest_write_effect(
            it->first, bytes, surface.dcc_metadata_addr, surface.dcc_metadata_bytes, addr, size);
        if (effect == prosper::frontend::LiveRttGuestWriteEffect::color_plane) {
            rtt_remat_note_reset("guest-color-plane-erase", it->first);
            it = cache.erase(it);
        } else if (effect == prosper::frontend::LiveRttGuestWriteEffect::dcc_metadata) {
            // Keep the surface identity/extent long enough to materialize a uniform DCC clear from
            // the descriptor in the following graphics span, but never LOAD or sample stale pixels.
            prosper::test::invalidate_persistent_color_target(it->first);
            rtt_remat_note_reset("dcc-metadata-write", it->first);
            it->second.rgba.reset();
            it->second.has_uniform_color = false;
            it->second.gpu_valid = false;
            it->second.dcc_metadata_dirty = true;
            ++it;
        } else {
            ++it;
        }
    }
}

void register_cpu_rtt_dcc_metadata(
    RttCache& cache, const std::vector<prosper::gpu::DrawItem>& items) {
    for (const auto& item : items) {
        const prosper::gpu::ShaderResourceTable* tables[] = {
            item.vrt.get(), item.prt.get(),
        };
        for (const auto* table : tables) {
            if (!table) continue;
            for (const auto& resource : table->resources) {
                if (!resource.compression_enabled || !resource.gpu_addr ||
                    !resource.metadata_addr)
                    continue;
                auto surface = cache.find(resource.gpu_addr);
                if (surface == cache.end()) continue;
                const uint64_t metadata_bytes =
                    prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
                if (!metadata_bytes) continue;
                surface->second.dcc_metadata_addr = resource.metadata_addr;
                surface->second.dcc_metadata_bytes = metadata_bytes;
            }
        }
    }
}

void drain_guest_gpu_writes(RttCache& cache, bool invalidate_ds) {
    auto& pending = pending_guest_gpu_writes();
    std::vector<PendingGuestGpuWrite> ranges;
    bool overflowed = false;
    {
        std::lock_guard<std::mutex> lock(pending.mutex);
        ranges.swap(pending.ranges);
        overflowed = pending.overflowed;
        pending.overflowed = false;
    }
    if (overflowed) {
        for (auto& [key, target] : prosper::test::persistent_color_target_cache()) {
            (void)key;
            target.valid = false;
        }
        if (invalidate_ds)
            for (auto& [key, image] : prosper::test::persistent_ds_cache()) {
                (void)key;
                image.depth_valid = false;
                image.stencil_valid = false;
            }
        cache.clear();
        return;
    }
    for (const auto& write : ranges) {
        // Restore the writer's own origin for the duration of this range's invalidation, so the
        // diagnostics inside report who actually wrote the bytes rather than the drain thread's
        // default. Cleared afterwards so a queued origin never leaks into an unrelated later write.
        prosper::gpu::set_guest_gpu_write_origin(write.origin);
        if (invalidate_ds)
            prosper::test::invalidate_persistent_ds_guest_write(write.addr, write.size);
        prosper::test::invalidate_persistent_color_target_guest_write(write.addr, write.size);
        invalidate_cpu_rtt_guest_write(cache, write.addr, write.size);
        prosper::gpu::set_guest_gpu_write_origin(nullptr);
    }
}

bool capture_color_format(VkFormat format, prosper::gpu::GpuCaptureColorFormat& captured) {
    format = prosper::test::backend_color_format(format);
    if (format == VK_FORMAT_R8G8B8A8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm;
    else if (format == VK_FORMAT_R16G16B16A16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba16Float;
    else if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        captured = prosper::gpu::GpuCaptureColorFormat::R11G11B10Float;
    else if (format == VK_FORMAT_R8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::R8Unorm;
    else if (format == VK_FORMAT_R32_UINT)
        captured = prosper::gpu::GpuCaptureColorFormat::R32Uint;
    else if (format == VK_FORMAT_R32_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::R32Float;
    else if (format == VK_FORMAT_R8G8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::Rg8Unorm;
    else if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba32Float;
    else if (format == VK_FORMAT_R16G16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rg16Float;
    else if (format == VK_FORMAT_R16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::R16Float;
    else
        return false;
    return true;
}

VkFormat replay_color_format(prosper::gpu::GpuCaptureColorFormat format) {
    if (format == prosper::gpu::GpuCaptureColorFormat::Rgba16Float)
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R11G11B10Float)
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    if (format == prosper::gpu::GpuCaptureColorFormat::R8Unorm)
        return VK_FORMAT_R8_UNORM;
    if (format == prosper::gpu::GpuCaptureColorFormat::R32Uint)
        return VK_FORMAT_R32_UINT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R32Float)
        return VK_FORMAT_R32_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rg8Unorm)
        return VK_FORMAT_R8G8_UNORM;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rgba32Float)
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rg16Float)
        return VK_FORMAT_R16G16_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R16Float)
        return VK_FORMAT_R16_SFLOAT;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

std::vector<uint8_t> inspection_rgba8(const std::vector<uint8_t>& pixels,
                                      uint32_t width, uint32_t height, VkFormat format) {
    const size_t texels = static_cast<size_t>(width) * height;
    if (format == VK_FORMAT_R8G8B8A8_UNORM && pixels.size() == texels * 4)
        return pixels;
    if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && pixels.size() == texels * 4) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            uint32_t packed = 0;
            std::memcpy(&packed, pixels.data() + texel * 4, sizeof(packed));
            const float values[3] = {
                prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)),
                prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
                prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)),
            };
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float value = values[channel];
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R8_UNORM && pixels.size() == texels) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            const uint8_t value = pixels[texel];
            rgba[texel * 4 + 0] = value;
            rgba[texel * 4 + 1] = value;
            rgba[texel * 4 + 2] = value;
            rgba[texel * 4 + 3] = value;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R8G8_UNORM && pixels.size() == texels * 2) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            rgba[texel * 4 + 0] = pixels[texel * 2 + 0];
            rgba[texel * 4 + 1] = pixels[texel * 2 + 1];
            rgba[texel * 4 + 2] = 0;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R16G16_SFLOAT && pixels.size() == texels * 4) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            for (uint32_t channel = 0; channel < 2; ++channel) {
                uint16_t half = 0;
                std::memcpy(&half, pixels.data() + texel * 4 + channel * 2, sizeof(half));
                const float value = prosper::gpu::half_to_float(half);
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
            rgba[texel * 4 + 2] = 0;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R16_SFLOAT && pixels.size() == texels * 2) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            uint16_t half = 0;
            std::memcpy(&half, pixels.data() + texel * 2, sizeof(half));
            const float value = prosper::gpu::half_to_float(half);
            const uint8_t visible = !std::isfinite(value) || value <= 0.0f ? 0
                : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            rgba[texel * 4 + 0] = visible;
            rgba[texel * 4 + 1] = visible;
            rgba[texel * 4 + 2] = visible;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT && pixels.size() == texels * 16) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                float value = 0.0f;
                std::memcpy(&value, pixels.data() + texel * 16 + channel * 4, sizeof(value));
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
        }
        return rgba;
    }
    // R16G16B16A16_UNORM: 16-bit fixed-point per channel -> 8-bit (high byte). Without this, a 64-bit
    // UNORM color target (a common non-float HDR/deep surface) inspected to black regardless of content,
    // which is misleading for RTT diagnostics.
    if (format == VK_FORMAT_R16G16B16A16_UNORM && pixels.size() == texels * 8) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel)
            for (uint32_t channel = 0; channel < 4; ++channel) {
                uint16_t v = 0;
                std::memcpy(&v, pixels.data() + texel * 8 + channel * 2, sizeof(v));
                rgba[texel * 4 + channel] = static_cast<uint8_t>(v >> 8);
            }
        return rgba;
    }
    if (format != VK_FORMAT_R16G16B16A16_SFLOAT || pixels.size() != texels * 8)
        return {};
    std::vector<uint8_t> rgba(texels * 4);
    for (size_t texel = 0; texel < texels; ++texel) {
        for (uint32_t channel = 0; channel < 4; ++channel) {
            uint16_t half = 0;
            std::memcpy(&half, pixels.data() + texel * 8 + channel * 2, sizeof(half));
            const float value = prosper::gpu::half_to_float(half);
            rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
        }
    }
    return rgba;
}

// #1330: gpu_replay sets PROSPER_PREFIX_INSPECT for its ordered-prefix modes (--draw N:M,
// --draw-steps, --through-operation) before the first render, so a prefix ending on a non-RGBA8
// color pass publishes that surface (inspection-converted) instead of a stale earlier RGBA8 pass.
// Never set outside those diagnostic replays; cached once — the flag is process-lifetime.
bool prefix_inspect_publish() {
    static const bool enabled = getenv("PROSPER_PREFIX_INSPECT") != nullptr;
    return enabled;
}

// Pixel decoding is a pure function of these fields for guest-backed textures. The identity map keyed
// by this tuple lives for one SUBMIT (#1691), not one renderer callback. One callback is one graphics
// span, and a submit is split into a new span at every interleaved compute/DMA operation, so a
// span-scoped map re-resolved every identity once per span: Blue Prince interleaves 21-22 dispatches
// through one frame, and its 56 distinct identities were resolved 853 times.
//
// The span split exists precisely because an interleaved operation can rewrite guest texture bytes.
// Retaining an entry past that boundary therefore requires proof that no such write landed on the
// decoded range, which the ordered in-submit journal (guest_gpu_writes_since) supplies directly and
// more precisely than the span boundary did. An entry reused inside its own span keeps the historical
// guarantee unchanged and needs no query. Writable storage-image callbacks still explicitly invalidate
// every overlapping entry after publishing their results, and live RTT / compute-imported inputs are
// excluded from the map entirely because an earlier pass in the same callback can replace their pixels.
struct TextureDecodeKey {
    uint64_t gpu_addr = 0;
    uint64_t host_data = 0;
    uint64_t host_data_size = 0;
    uint32_t size = 0;
    uint64_t source_span_bytes = 0;
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0, height = 0, depth = 1;
    uint32_t sample_count = 1;
    uint32_t tile_mode = 0;
    uint32_t linear_row_pitch_bytes = 0;
    uint32_t img_dim = 0;
    uint32_t mip_tail_bytes = 0, mip_tail_x = 0, mip_tail_y = 0;
    uint32_t layer_stride_bytes = 0, layer_mip_offset_bytes = 0;
    uint32_t max_uncompressed_block_size = 0, max_compressed_block_size = 0;
    uint32_t dcc_flags = 0;
    uint64_t metadata_addr = 0;
    uint64_t metadata_host_data = 0;
    uint64_t metadata_host_data_size = 0;
    bool in_mip_tail = false;
    bool preserve_narrow_channels = false;
    // The same guest T# can be consumed through incompatible shader interfaces. In particular,
    // graphics IMAGE_LOAD uses a formatless raw-uvec4 image (16 host bytes per texel), while an
    // image atomic over that descriptor requires exact R32ui (4 host bytes per texel). These are
    // distinct decoded identities even when every guest descriptor field and source span matches.
    uint32_t reflected_image_numeric_class = 0;
    uint32_t reflected_storage_image_format = 0;
    uint32_t reflected_image_dim = 0;
    bool reflected_image_arrayed = false;
    bool reflected_image_multisampled = false;
    bool reflected_image_writable = false;
    // Renderer-owned layers that are newer than the guest/compute backing participate in the
    // decoded identity. This is zero for ordinary textures.
    uint64_t renderer_overlay_version = 0;
    bool operator==(const TextureDecodeKey&) const = default;
};

struct TextureDecodeKeyHash {
    size_t operator()(const TextureDecodeKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash ^= static_cast<size_t>(value);
            hash *= 1099511628211ull;
            hash ^= static_cast<size_t>(value >> 32);
            hash *= 1099511628211ull;
        };
        mix(key.gpu_addr); mix(key.host_data); mix(key.host_data_size); mix(key.size);
        mix(key.source_span_bytes);
        mix(key.cls); mix(key.format); mix(key.num_components); mix(key.width); mix(key.height); mix(key.depth);
        mix(key.sample_count);
        mix(key.tile_mode); mix(key.linear_row_pitch_bytes); mix(key.img_dim);
        mix(key.mip_tail_bytes); mix(key.mip_tail_x); mix(key.mip_tail_y);
        mix(key.layer_stride_bytes); mix(key.layer_mip_offset_bytes);
        mix(key.max_uncompressed_block_size);
        mix(key.max_compressed_block_size); mix(key.dcc_flags); mix(key.metadata_addr);
        mix(key.metadata_host_data); mix(key.metadata_host_data_size);
        mix(key.in_mip_tail);
        mix(key.preserve_narrow_channels);
        mix(key.reflected_image_numeric_class);
        mix(key.reflected_storage_image_format);
        mix(key.reflected_image_dim);
        mix(key.reflected_image_arrayed);
        mix(key.reflected_image_multisampled);
        mix(key.reflected_image_writable);
        mix(key.renderer_overlay_version);
        return hash;
    }
};

// ---- AvPlayer NV12 chroma-plane recognition diagnostic (#2005) --------------------------------
//
// The classification itself lives in avplayer_plane_policy.hpp (with its own unit test); this is
// only its live log. PROSPER_AVPCHROMA_LOG=1 emits one line per distinct narrow Unorm8 sampled
// texture — both NV12 planes qualify, whatever DIM they declare — naming every field the predicate
// reads and the clause that decided it. Off by default. A wrong verdict here is silent: a chroma
// plane sent down the coverage broadcast still produces correct luma, detail and geometry, so no
// draw census, colour count or non-black metric can see it.
using prosper::frontend::AvpChromaReason;
using prosper::frontend::AvpChromaVerdict;
using prosper::frontend::avp_chroma_reason_name;
using prosper::frontend::classify_avplayer_chroma_plane;

bool avp_chroma_log() {
    static const bool enabled = getenv("PROSPER_AVPCHROMA_LOG") != nullptr;
    return enabled;
}

void log_avp_chroma_candidate(const prosper::gpu::ShaderResource& r,
                              uint32_t tw, uint32_t th, const AvpChromaVerdict& v) {
    static std::mutex mx;
    static std::set<uint64_t> seen;
    {
        std::lock_guard<std::mutex> lock(mx);
        const uint64_t identity = r.gpu_addr ^ (static_cast<uint64_t>(tw) << 12) ^
                                  (static_cast<uint64_t>(th) << 34) ^
                                  (static_cast<uint64_t>(r.num_components) << 56);
        if (!seen.insert(identity).second) return;
    }
    // A one-component plane is a luma candidate, never a chroma one: report it as context rather
    // than as a rejected chroma plane, so the pair can be read off one log without confusion.
    const char* verdict = r.num_components == 1u ? "luma-plane-candidate "
                        : v.match               ? "CHROMA "
                                                : "coverage-broadcast ";
    fprintf(stderr,
            "[avpchroma] addr=%llx %ux%u fmt=%u ncomp=%u tile=%u dim=%u depth=%u "
            "lstride=%u lmipoff=%u miptail=%u size=%u swz=%u,%u,%u,%u "
            "row_bytes=%u pitch_field=%u registered=%u resolved=%u luma=%llx -> %s%s\n",
            (unsigned long long)r.gpu_addr, tw, th, static_cast<unsigned>(r.format),
            r.num_components, r.tile_mode, r.img_dim, r.depth,
            r.layer_stride_bytes, r.layer_mip_offset_bytes, r.in_mip_tail ? 1u : 0u, r.size,
            r.swizzle[0], r.swizzle[1], r.swizzle[2], r.swizzle[3],
            v.row_bytes, r.linear_row_pitch_bytes, v.registered_pitch, v.resolved_pitch,
            (unsigned long long)v.sibling_luma_addr,
            verdict, avp_chroma_reason_name(v.reason));
    fflush(stderr);
}

// Milliseconds since the first ARMED census check of this run — the origin for the `ms:` form of
// PROSPER_PASS_LOG / PROSPER_DUMP_PERSISTENT (diagnostic_window.hpp). Lazily started, so an
// unarmed run never reads the clock and the origin is the same first-callback moment
// PROSPER_GPU_CAPTURE_AFTER_MS uses, letting a capture and a census be aimed at one instant.
static uint64_t diagnostic_elapsed_ms() {
    static const auto start = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count());
}

struct DecodedTexture {
    const uint8_t* pixels = nullptr;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t persistent_id = 0;
    uint64_t persistent_version = 0;
    // Everything below exists so the entry can outlive the span that produced it (#1691).
    // `span` is the renderer-callback ordinal that decoded these pixels: reuse inside that same span
    // is the historical contract and needs no proof. Crossing a span boundary requires
    // guest_gpu_writes_since(snapshot, source_addr, source_size) == Unchanged over the exact range
    // the decode read, so an interleaved compute/DMA/EOP write to the backing forces a fresh resolve
    // instead of serving pixels the guest has since replaced (the #780 failure shape).
    // `texstore_slot` pins the scratch slot when the pixels are NOT owned by the persistent cache;
    // without the pin the next span would hand the same slot to an unrelated decode and this entry
    // would silently alias another texture's bytes.
    uint64_t span = 0;
    prosper::gpu::GuestGpuWriteSnapshot snapshot;
    uint64_t source_addr = 0;
    uint64_t source_size = 0;
    size_t texstore_slot = SIZE_MAX;
    // Exact 2D-MSAA resolves use an owned plane-major allocation rather than a reusable texstore
    // slot. Retaining the owner here gives same-submit cache hits the same lifetime guarantee as a
    // pinned scratch slot, without pinning one ~32 MiB vector per distinct surface forever.
    std::shared_ptr<const std::vector<uint8_t>> pixels_owner;
    // Pixel bytes do not define their Vulkan interpretation. Restore these on every hit rather than
    // re-deriving only the historical sampled R8/RG8/FP16 cases and accidentally reverting a
    // portable UINT storage image to the FrameResource default RGBA8_UNORM.
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    bool storage_image_contract_valid = true;
    // Decoded array layers behind `pixels`. 1 for every non-array texture, which is why this can
    // sit last and default: the positional initializers below stay correct without naming it.
    uint32_t layers = 1;
    // Readable bytes behind `pixels`. The backend's multi-layer span contract REQUIRES this -- it
    // refuses to build a layered image without proof that every layer is readable -- and the
    // renderer had exactly one assignment to fr.tex_byte_size in the whole file, on the MSAA path.
    // So every array served from cache arrived claiming zero readable bytes and was rejected.
    size_t pixels_bytes = 0;
};

// Always-on identity-scope accounting; see TextureDecodeScopeStats in the header.
thread_local TextureDecodeScopeStats g_texture_decode_scope{};

// #325: the decoded-array footprint budget. Read once at file scope rather than inside the decode,
// because it is CONFIGURATION, not a diagnostic gate -- a getenv lexically enclosing a report makes
// that report depend on two switches, which check_diag_gates.py rightly rejects (TWO-GATE).
inline uint64_t array_decode_budget_bytes() {
    static const uint64_t bytes = [] {
        // atoll turned `-1` into a budget of 0xFFFFFFFFFFF00000 -- i.e. UNBOUNDED, the one setting
        // this knob exists to prevent -- and `1gib` into 1 MiB, which reads in the log as a
        // deliberate choice. Refuse both and keep 1 GiB (#3267).
        const char* e = getenv("PROSPER_ARRAY_DECODE_BUDGET_MIB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_ARRAY_DECODE_BUDGET_MIB", e, 1024ull, UINT64_MAX >> 20, "MiB");
        return mib << 20;
    }();
    return bytes;
}

struct PersistentDecodedTexture {
    // Decoded array layers behind `pixels`; 1 for every non-array texture. The persistent cache is
    // the one that actually serves a world-texture atlas -- a submit-local entry is rebuilt from it
    // -- so a layer count that stops here is a layer count the shader never sees (#325).
    uint32_t layers = 1;
    uint64_t source_addr = 0;
    size_t source_size = 0;
    size_t source_prefix_size = 0;
    bool source_matches_pixels = false;
    std::vector<uint8_t> source_prefix;
    std::shared_ptr<const std::vector<uint8_t>> pixels;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t last_use = 0;
    uint64_t persistent_id = 0;
    uint64_t persistent_version = 0;
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    bool storage_image_contract_valid = true;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    prosper::host::GuestWriteWatch source_watch;
    uint32_t source_watch_dirty_count = 0;
    uint32_t source_watch_stable_validations = 0;
    bool source_watch_only = false;
    bool source_watch_disabled = false;

    size_t bytes() const { return source_prefix.size() + (pixels ? pixels->size() : 0); }
};

uint64_t host_physical_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0) return 0;
    const uint64_t count = static_cast<uint64_t>(pages);
    const uint64_t bytes = static_cast<uint64_t>(page_bytes);
    return count <= UINT64_MAX / bytes ? count * bytes : UINT64_MAX;
#endif
}
}

size_t texture_decode_cache_limit_bytes(const char* override_mib,
                                        uint64_t physical_memory_bytes) {
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    constexpr uint64_t kMinBytes = 1024ull * kMiB;
    constexpr uint64_t kMaxBytes = 4096ull * kMiB;
    uint64_t bytes = kMinBytes;
    if (override_mib) {
        const uint64_t mib = strtoull(override_mib, nullptr, 10);
        bytes = std::min<uint64_t>(mib, SIZE_MAX / kMiB) * kMiB;
    } else if (physical_memory_bytes) {
        bytes = std::clamp(physical_memory_bytes / 8u, kMinBytes, kMaxBytes);
        bytes -= bytes % kMiB;
    }
    return static_cast<size_t>(std::min<uint64_t>(bytes, SIZE_MAX));
}

bool submit_local_texture_decode_reusable(uint64_t entry_span, uint64_t current_span,
                                          uint64_t entry_source_addr, uint64_t entry_source_size,
                                          uint64_t current_source_addr, uint64_t current_source_size,
                                          prosper::gpu::GuestGpuWriteQuery journal_query) {
    if (entry_span == current_span) return true;
    return current_source_size != 0 &&
        entry_source_addr == current_source_addr &&
        entry_source_size == current_source_size &&
        journal_query == prosper::gpu::GuestGpuWriteQuery::Unchanged;
}

TextureDecodeScopeStats texture_decode_scope_stats() { return g_texture_decode_scope; }
void reset_texture_decode_scope_stats() { g_texture_decode_scope = {}; }

bool sampled_msaa_fetch_shape_supported(const prosper::gpu::ShaderResource& resource,
                                        bool is_storage_image,
                                        bool reflected_msaa_fetch) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::ResourceClass;
    return resource.cls == ResourceClass::Texture && !is_storage_image &&
        resource.img_dim == 6u && resource.sample_count == 4u &&
        resource.tile_mode == 24u && resource.format == DataFormat::Float32 &&
        resource.num_components == 1u && resource.depth == 1u &&
        resource.declared_mip_levels == 1u && !resource.in_mip_tail &&
        resource.mip_tail_bytes == 0u && resource.layer_stride_bytes == 0u &&
        resource.layer_mip_offset_bytes == 0u && reflected_msaa_fetch;
}

bool native_r11_sampled_upload_supported(const prosper::gpu::ShaderResource& resource) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::ResourceClass;
    return resource.cls == ResourceClass::Texture && resource.img_dim == 1u &&
        resource.depth == 1u && resource.format == DataFormat::Float10_11_11 &&
        resource.num_components == 3u && !resource.compression_enabled && !resource.srgb;
}

struct DiagnosticAddressSelector {
    bool configured = false;
    bool valid = true;
    std::vector<uint64_t> addresses;

    bool includes(uint64_t address) const {
        return !configured ||
            (valid && std::find(addresses.begin(), addresses.end(), address) != addresses.end());
    }
};

DiagnosticAddressSelector parse_diagnostic_address_selector(const char* env_name) {
    DiagnosticAddressSelector selector;
    // Each caller stores the result in a function-local static, so this generic parser keeps the
    // same one-read-per-process contract as PROSPER_ENV_VALUE without passing a runtime name into
    // that macro's captureless lambda.
    const char* spec = std::getenv(env_name);
    if (!spec || !*spec) return selector;
    selector.configured = true;
    selector.valid = prosper::gpu::parse_hex_watch_list(spec, selector.addresses);
    if (!selector.valid) {
        std::fprintf(stderr,
                     "[render] ignoring malformed %s=\"%s\" (expected 0x-prefixed hex "
                     "addresses, comma separated) -- selector matches NOTHING\n",
                     env_name, spec);
        selector.addresses.clear();
    } else {
        std::fprintf(stderr, "[render] %s selects %zu address(es)\n",
                     env_name, selector.addresses.size());
    }
    return selector;
}

const DiagnosticAddressSelector& native_r11_sampled_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_NATIVE_R11_SAMPLED_ONLY");
    return selector;
}

const DiagnosticAddressSelector& renderer_mip_chain_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_RENDERER_MIP_CHAIN_ONLY");
    return selector;
}

const DiagnosticAddressSelector& rtt_no_seed_target_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_RTT_NOSEED_TARGET");
    return selector;
}

RendererMipChainLayout renderer_mip_chain_layout(uint64_t level_zero_address,
                                                 uint32_t width, uint32_t height,
                                                 uint32_t bytes_per_texel,
                                                 uint32_t tile_mode,
                                                 uint32_t declared_mip_levels) {
    RendererMipChainLayout result;
    if (!level_zero_address || !width || !height || !bytes_per_texel ||
        declared_mip_levels < 2u || declared_mip_levels > result.level_ids.size())
        return result;
    const uint32_t max_mip = declared_mip_levels - 1u;
    const prosper::gpu::TiledMipLevelLayout level_zero =
        prosper::gpu::tiled_mip_level_layout(
            width, height, bytes_per_texel, tile_mode, max_mip, 0u);
    // This helper starts from a descriptor selecting allocation level zero. A level-zero tail view
    // cannot identify the allocation origin by subtraction and remains on the established fallback.
    if (!level_zero.supported || level_zero.in_tail ||
        level_zero.byte_offset > level_zero_address)
        return result;
    const uint64_t allocation_base = level_zero_address - level_zero.byte_offset;
    for (uint32_t level = 0; level < declared_mip_levels; ++level) {
        const prosper::gpu::TiledMipLevelLayout layout =
            prosper::gpu::tiled_mip_level_layout(
                width, height, bytes_per_texel, tile_mode, max_mip, level);
        if (!layout.supported ||
            (!layout.in_tail && layout.byte_offset > UINT64_MAX - allocation_base))
            return {};
        result.level_ids[level] = layout.in_tail
            ? allocation_base : allocation_base + layout.byte_offset;
    }
    result.level_count = declared_mip_levels;
    return result;
}

bool persistent_texture_decode_cache_eligible(bool guest_decode_candidate,
                                              bool compute_image_hit,
                                              bool is_storage_image,
                                              bool cache_disabled,
                                              bool compression_supported,
                                              size_t cache_limit,
                                              size_t source_size) {
    return guest_decode_candidate && !compute_image_hit && !is_storage_image &&
        !cache_disabled && compression_supported && cache_limit && source_size;
}

// PROSPER_RTT_ALIAS=<sampled>:<served>[,<sampled>:<served>…] — diagnostic ONLY, off by default.
//
// Serve one sampled surface from another surface's cache entry. It answers exactly one question, and
// only by experiment: when a consumer samples an address nothing writes, is that address a distinct
// buffer whose producer is missing, or the SAME buffer the renderer already holds under a different
// base?  Those two have identical signatures in every census -- the pass graph, the target census
// and the write watches all report "written by nothing" either way -- and they call for opposite
// work: find the missing producer, or fix a base decode.
//
// GTA V PPSA04263 is the worked case. Its final composite samples the 4K HDR scene colour
// 0x2063380000 24x per frame, and across 6,561 frames that address is a render target exactly once,
// for a clear. Meanwhile the lit scene sits in 0x20431c0000, which the composite never reads.
// Aliasing one onto the other decides between the two readings in a single run.
//
// This is not a repair and must never become one: it makes a consumer read bytes the guest did not
// point it at, so a picture appearing under it is evidence about the DECODE, never a fix. Leave it
// unset outside an A/B.
// PROSPER_SCAN_DESCRIPTOR=<hex surface base>[,<hex base>…] — count CANDIDATE descriptor words in
// guest memory whose value matches a surface's base field, once, from the render thread.
//
// READ THE NEGATIVES CAREFULLY. This is a candidate-word scan, not a descriptor enumerator, and it is
// not exhaustive in three separate ways:
//   * COVERAGE. It probes 2 MiB-aligned spans in a fixed window and keeps only spans that are
//     readable IN FULL, so a descriptor in a partially mapped span, or in a mapping outside the
//     window, is never examined. The report prints the spans kept and skipped so the coverage is
//     visible rather than assumed.
//   * VALIDATION. A matching dword is only the value a base field WOULD hold; the following words are
//     printed but not checked against a descriptor layout. A hit is a candidate, not proof that a
//     descriptor lives there, and an ordinary integer can match.
//   * DIRECTION. A descriptor does not encode read/write intent -- that lives in the shader
//     instruction -- so a hit cannot distinguish a producer's view of a surface from a consumer's.
// **A zero therefore cannot establish that no producer exists.** It bounds where one was not found,
// which is a weaker and different claim.
//
// COST: it runs SYNCHRONOUSLY on the render callback and may read several GiB, so the frame it fires
// on stalls visibly. Opt-in and one-shot for that reason; never leave it set for a timing run.
//
// The question it exists for is the one left over when every write path comes back empty: a surface
// nothing writes is either one the guest genuinely never produces, or one whose producer prosper
// failed to RESOLVE -- and an unresolved producer is invisible to every instrument that enumerates
// resolved tables. `PROSPER_DYNTRACE_FAIL` prints only the descriptors the const-fold evaluated;
// measured on GTA V's failing kernels it printed 15 of 21 image ops for 0x205b5e8600 and 4 of 6 for
// 0x205b657200, so its null covered most of that population and not all of it.
//
// A descriptor's base field holds the address shifted right by 8, so a surface's candidates are
// findable by matching that dword. Use the result as a lead to follow, not as a census.
void scan_for_descriptors_once(uint64_t submit_index) {
    struct Spec { std::vector<uint64_t> bases; uint64_t at_submit = 0; };
    static const Spec spec = [] {
        Spec parsed;
        const char* text = getenv("PROSPER_SCAN_DESCRIPTOR");
        if (!text || !*text) return parsed;
        for (const char* cursor = text; *cursor;) {
            char* end = nullptr;
            const uint64_t value = strtoull(cursor, &end, 16);
            if (end == cursor) break;
            parsed.bases.push_back(value);
            cursor = (*end == ',') ? end + 1 : end;
        }
        // SUBMITS, not frames: the caller passes g_this_submit. Named for what it counts -- the two
        // differ by more than an order of magnitude on a routed run, so a threshold read as frames
        // fires at a moment the operator did not ask for.
        const char* at = getenv("PROSPER_SCAN_DESCRIPTOR_AT_SUBMIT");
        parsed.at_submit = at ? strtoull(at, nullptr, 10) : 6000;
        return parsed;
    }();
    if (spec.bases.empty()) return;
    static std::atomic<bool> done{false};
    if (submit_index < spec.at_submit || done.exchange(true)) return;

    // The guest's direct-memory mapping, discovered by probing rather than assumed: walk 2 MiB steps
    // over the region PS5 titles map and keep the readable spans.
    constexpr uint64_t kStep = 2ull << 20;
    constexpr uint64_t kBegin = 0x2000000000ull, kEnd = 0x2200000000ull;
    size_t spans = 0, skipped_spans = 0, scanned_mb = 0;
    std::map<uint64_t, uint64_t> hits;   // surface base -> count
    std::map<uint64_t, std::map<std::string, uint64_t>> shapes;  // base -> shape -> count
    for (uint64_t page = kBegin; page < kEnd; page += kStep) {
        if (!prosper::gpu::guest_readable(page, static_cast<uint32_t>(kStep))) {
            ++skipped_spans;   // unmapped OR only partially mapped -- not examined
            continue;
        }
        ++spans; scanned_mb += kStep >> 20;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(page));
        const size_t count = kStep / sizeof(uint32_t);
        for (size_t i = 0; i < count; ++i) {
            for (const uint64_t base : spec.bases) {
                if (words[i] != static_cast<uint32_t>(base >> 8)) continue;
                ++hits[base];
                // A bare dword match is not yet a descriptor: the same bit pattern occurs in ordinary
                // data. Decode the eight words that would follow if this were a T# and group by their
                // SHAPE, so a real descriptor set stands out from coincidence and two descriptors that
                // describe the surface differently (a producer's view versus a consumer's) separate.
                if (i + 8 > count) continue;
                char sig[128];
                snprintf(sig, sizeof sig, "w1=%08x w2=%08x w3=%08x w4=%08x w7=%08x",
                         words[i + 1], words[i + 2], words[i + 3], words[i + 4], words[i + 7]);
                shapes[base][sig] += 1;
            }
        }
    }
    fprintf(stderr,
            "[descr-scan] examined %zu MiB over %zu fully-readable spans at submit %llu; %zu span(s) "
            "SKIPPED (unmapped or partially mapped) -- negatives are NOT exhaustive\n",
            scanned_mb, spans, (unsigned long long)submit_index, skipped_spans);
    for (const uint64_t base : spec.bases) {
        fprintf(stderr,
                "[descr-scan] 0x%llx: %llu CANDIDATE word(s) match its base field, %zu distinct "
                "following shape(s) -- unvalidated, and direction (producer vs consumer) unknown\n",
                (unsigned long long)base, (unsigned long long)hits[base], shapes[base].size());
        size_t shown = 0;
        for (const auto& [sig, n] : shapes[base]) {
            if (shown++ >= 8) break;
            fprintf(stderr, "[descr-scan]     x%-4llu %s\n", (unsigned long long)n, sig.c_str());
        }
    }
}

uint64_t texture_rtt_alias(uint64_t gpu_address) {
    struct Alias { uint64_t from, to; };
    static const std::vector<Alias> aliases = [] {
        std::vector<Alias> parsed;
        const char* spec = getenv("PROSPER_RTT_ALIAS");
        if (!spec || !*spec) return parsed;
        for (const char* cursor = spec; *cursor;) {
            char* end = nullptr;
            const uint64_t from = strtoull(cursor, &end, 16);
            if (end == cursor || *end != ':') break;
            cursor = end + 1;
            const uint64_t to = strtoull(cursor, &end, 16);
            if (end == cursor) break;
            parsed.push_back({from, to});
            fprintf(stderr, "[rtt-alias] sampling 0x%llx will be served from 0x%llx\n",
                    (unsigned long long)from, (unsigned long long)to);
            cursor = (*end == ',') ? end + 1 : end;
        }
        return parsed;
    }();
    for (const Alias& alias : aliases)
        if (alias.from == gpu_address) return alias.to;
    return gpu_address;
}

uint64_t texture_decode_source_address(uint64_t gpu_address,
                                       uint32_t image_dimension,
                                       bool in_mip_tail,
                                       uint32_t layer_mip_offset_bytes) {
    gpu_address = texture_rtt_alias(gpu_address);
    if (image_dimension != 5u || in_mip_tail ||
        gpu_address > UINT64_MAX - layer_mip_offset_bytes)
        return gpu_address;
    return gpu_address + layer_mip_offset_bytes;
}

bool texture_source_snapshot_can_follow_watch(bool source_matches_pixels,
                                              bool validation_audit_enabled,
                                              bool watch_active,
                                              size_t retained_source_bytes,
                                              size_t expected_source_bytes) {
    return !source_matches_pixels && !validation_audit_enabled && watch_active &&
        expected_source_bytes != 0 && retained_source_bytes == expected_source_bytes;
}

uint32_t buffer_upload_bytes(uint32_t declared_bytes) {
    // PROSPER_MAX_BUFFER_UPLOAD_MB=N lowers the ceiling (1..64 MiB) so one build can reproduce the
    // #1427 collapse and its fix back to back; unset/invalid keeps the full ceiling.
    static const uint32_t ceiling = [] {
        const char* value = getenv("PROSPER_MAX_BUFFER_UPLOAD_MB");
        if (!value || !*value) return kMaxBufferUploadBytes;
        const unsigned long megabytes = strtoul(value, nullptr, 10);
        if (megabytes < 1 || megabytes > 64) return kMaxBufferUploadBytes;
        return static_cast<uint32_t>(megabytes) << 20;
    }();
    return std::min(declared_bytes, ceiling) & ~3u;
}

// #2215 instrument -- see live_renderer.hpp for why this is a process-wide slot rather than a
// thread_local (the sampler asks from a different thread than the one it is asking about).
std::atomic<unsigned long>& renderer_callback_tid() {
    static std::atomic<unsigned long> tid{0};
    return tid;
}

bool tid_is_in_renderer_callback(unsigned long native_tid) {
    const unsigned long current = renderer_callback_tid().load(std::memory_order_relaxed);
    return current != 0 && current == native_tid;
}

// Every callback ENTRY, not the ~0.6% of them a sampler happens to catch. See the self-check in
// hle_kernel.cpp: a zero here is real evidence that the strong override never linked, whereas a zero
// count of sampler sightings is just the ordinary outcome at this base rate (#2347).
std::atomic<unsigned long long> g_renderer_callback_entries{0};

extern "C" unsigned long long prosper_renderer_callback_entries() {
    return g_renderer_callback_entries.load(std::memory_order_relaxed);
}

ScopedRendererCallbackTid::ScopedRendererCallbackTid() {
    g_renderer_callback_entries.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    const unsigned long self = (unsigned long)GetCurrentThreadId();
#else
    const unsigned long self = (unsigned long)(uintptr_t)pthread_self();
#endif
    previous = renderer_callback_tid().exchange(self, std::memory_order_relaxed);
}

ScopedRendererCallbackTid::~ScopedRendererCallbackTid() {
    renderer_callback_tid().store(previous, std::memory_order_relaxed);
}

// The strong definition that overrides prosper_core's weak stub once the live renderer is in
// the link. Keeping the override here rather than in prosper_core is what lets a build WITHOUT
// the renderer answer 0 truthfully instead of failing to link (#2215).
extern "C" int prosper_thread_in_renderer_callback(unsigned long native_tid) {
    return prosper::frontend::tid_is_in_renderer_callback(native_tid) ? 1 : 0;
}

void register_live_renderer(const std::string& frame_dir, bool dump_bmps_requested,
                            const std::string& title_id) {
    // Keep the legacy global disable authoritative for every frontend, including callers with their
    // own explicit opt-in such as PROSPER_APP_DUMP_FRAMES.
    const bool dump_bmps = frame_dump_request_allowed(
        dump_bmps_requested, PROSPER_ENV_VALUE("PROSPER_NO_FRAME_DUMPS"));
    // GTA V's reviewed Performance-mode route contains Wave64 fragment programs whose only
    // remaining width reason is a control-flow WaveAny. The backend may run that narrow class at
    // NVIDIA's native Wave32 only for this title; every other title keeps master's fail-visible
    // exact-width contract, and ballots/lane identity/scalar reductions stay exact even here.
    const bool gta5_native_fragment_vote = title_id == "PPSA04263";
    // Create (and thereby PUBLISH) the renderer's Vulkan device up front so the compute backend can
    // adopt it (#1091). Compute initializes lazily on its first dispatch, and titles routinely
    // dispatch before their first draw -- without this the compute device would be created first and
    // the two would never share. Only reached when the live renderer is registered, so headless
    // compute-only use (tests/gpu/recompiler/test_game_compute.cpp) still creates its own device.
    (void)prosper::test::render_vk_ctx();
    static RttCache g_rtt;   // render-to-texture cache (#167)
    // Shared final-callback ordinal for PROSPER_PASS_LOG (increments where dp_submit does).
    static std::atomic<uint64_t> g_pass_log_submit{0};
    // The census windows those two switches open. Parsed once here rather than per callback so the
    // `ms:` form's latch is one object: PROSPER_PASS_LOG is consulted from two places about the SAME
    // ordinal, and a per-site window would latch them onto different callbacks. Same single-present-
    // thread contract as `g_rtt` and the `dp_submit`/`warned` statics below.
    static prosper::frontend::DiagnosticWindow g_pass_log_window{
        prosper::frontend::parse_diagnostic_window(PROSPER_ENV_VALUE("PROSPER_PASS_LOG"))};
    static prosper::frontend::DiagnosticWindow g_persist_window{
        prosper::frontend::parse_diagnostic_window(PROSPER_ENV_VALUE("PROSPER_DUMP_PERSISTENT"))};
    // Match boot_trace's progression-diagnostic contract: callers may register the graphics
    // renderer while deliberately leaving compute unregistered. This keeps semantic dispatches
    // visible without letting screenshot/prosper-app registration silently undo the A/B.
    if (!PROSPER_ENV_ON("PROSPER_NO_COMPUTE")) register_live_compute();
    const char* ds_invalidate = PROSPER_ENV_VALUE("PROSPER_DS_GUEST_WRITE_INVALIDATE");
    const bool invalidate_ds = !ds_invalidate || strcmp(ds_invalidate, "0");
    prosper::gpu::set_guest_gpu_write_observer(
        [](uint64_t addr, uint64_t size, const char* origin) {
            queue_guest_gpu_write(addr, size, origin);
        });
    // Resource tables are built before the submit reaches this callback. Publish the renderer's
    // default mode now so unmapped render-target descriptors remain available for RTT injection.
    // Outside a registered renderer, resource decoding retains its strict unknown-format policy.
    // PROSPER_RTT_PERTARGET is read LIVE here even though this runs once: the very next lines SET
    // it, so it is a variable this process writes to itself. Caching it would be correct today
    // (each macro use site has its own static, and :1067 reads after this one) but it is correct by
    // ordering rather than by construction, and it is the only thing that would put this name in
    // the cached-vs-armed intersection below. A gate with one standing exception is a gate people
    // stop reading. This site is init-time, so a live read costs nothing.
    if (!PROSPER_ENV_ON("PROSPER_RTT_SINGLE_TARGET") && !getenv("PROSPER_RTT_PERTARGET")) {
#ifdef _WIN32
        _putenv_s("PROSPER_RTT_PERTARGET", "1");
#else
        setenv("PROSPER_RTT_PERTARGET", "1", 1);
#endif
    }
    static std::atomic<int> frame_no{0};
    // Register the capture RTT-seed readers whenever the live renderer is up — not only under the
    // capture env vars — so the interactive F9 frame grab (request_interactive_capture_bundle, which has
    // no env var) can seed the renderer-owned RTTs its frame SAMPLES. Registration is free: these lambdas
    // are only invoked while a capture is actually in flight; a normal render run never calls them.
    // Without the seed, a submit that samples a deferred/temporal renderer-owned RGBA16F target (Blue
    // Prince's presenting pass) replays black because that input is captured as all-zeros (#1291).
    {
        auto materialize_current_rtt = [](uint64_t addr, RttSurf& surface,
                                          std::string& error) {
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            const uint32_t bytes_per_pixel =
                prosper::test::backend_color_bytes_per_pixel(format);
            if (surface.rgba && prosper::frontend::live_rtt_cpu_snapshot_matches(
                    surface.w, surface.h, bytes_per_pixel, surface.rgba->size()))
                return true;
            if (materialize_uniform_rtt(surface)) return true;
            if (!surface.gpu_valid) return false;
            std::vector<uint8_t> materialized;
            if (!prosper::test::readback_persistent_color_target(
                    addr, surface.w, surface.h, format, materialized, error))
                return false;
            if (!prosper::frontend::live_rtt_cpu_snapshot_matches(
                    surface.w, surface.h, bytes_per_pixel, materialized.size())) {
                error = "persistent RTT readback byte count does not match its current identity";
                return false;
            }
            surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                std::move(materialized));
            return true;
        };
        prosper::gpu::set_gpu_capture_rtt_seed_reader(
            [invalidate_ds, materialize_current_rtt](
                uint64_t addr, prosper::gpu::GpuCaptureRttSeed& seed) {
                drain_guest_gpu_writes(g_rtt, invalidate_ds);
                auto it = g_rtt.find(addr); if (it == g_rtt.end()) return false;
                prosper::gpu::GpuCaptureColorFormat captured_format;
                if (!capture_color_format(it->second.format, captured_format)) return false;
                std::string error;
                if (!materialize_current_rtt(addr, it->second, error)) return false;
                seed.guest_addr = addr; seed.width = it->second.w; seed.height = it->second.h;
                seed.format = captured_format;
                seed.rgba = *it->second.rgba; return true;
            });
        prosper::gpu::set_gpu_capture_rtt_seed_snapshot_reader(
            [invalidate_ds, materialize_current_rtt](
                std::vector<prosper::gpu::GpuCaptureRttSeed>& seeds, std::string& error) {
                drain_guest_gpu_writes(g_rtt, invalidate_ds);
                seeds.reserve(g_rtt.size());
                for (auto& [addr, surface] : g_rtt) {
                    prosper::gpu::GpuCaptureColorFormat captured_format;
                    if (!capture_color_format(surface.format, captured_format)) continue;
                    std::string readback_error;
                    if (!materialize_current_rtt(addr, surface, readback_error)) {
                        if (!readback_error.empty()) {
                            char detail[256];
                            std::snprintf(detail, sizeof(detail),
                                          "RTT 0x%llx %ux%u snapshot failed: %s",
                                          static_cast<unsigned long long>(addr),
                                          surface.w, surface.h, readback_error.c_str());
                            error = detail;
                            return false;
                        }
                        continue;
                    }
                    prosper::gpu::GpuCaptureRttSeed seed;
                    seed.guest_addr = addr; seed.width = surface.w; seed.height = surface.h;
                    seed.format = captured_format;
                    seed.rgba = *surface.rgba; seeds.push_back(std::move(seed));
                }
                return true;
            });
    }
    // Deferred RTT readback (#1284) can leave a persistent Vulkan image as the ONLY copy of a
    // target's pixels. When the backend evicts such an image it reads the pixels back first and
    // hands them here, so the CPU cache regains the authoritative copy the eager path used to keep.
    prosper::test::persistent_color_target_evict_sink() =
        [](uint64_t id, uint32_t width, uint32_t height, VkFormat format,
           std::vector<uint8_t>&& pixels) {
            auto it = g_rtt.find(id);
            if (it == g_rtt.end()) return;
            RttSurf& surface = it->second;
            if (surface.w != width || surface.h != height ||
                prosper::test::backend_color_format(surface.format) !=
                    prosper::test::backend_color_format(format))
                return;
            const size_t expected = static_cast<size_t>(width) * height *
                prosper::test::backend_color_bytes_per_pixel(
                    prosper::test::backend_color_format(format));
            if (pixels.size() != expected) return;
            if (!surface.rgba || surface.rgba->size() != expected)
                surface.rgba =
                    std::make_shared<const std::vector<uint8_t>>(std::move(pixels));
            surface.gpu_valid = false;   // the image is being destroyed
        };
    // The compute backend must not sample a surface whose CURRENT pixels live in this renderer's
    // RTT cache (raw guest memory is then empty/stale — the Dead Cells 642x362 lesson): publish the
    // exact-match identity and immutable CPU snapshot used by live compute (#590).
    // Establish WHICH kind of compression metadata an address is. The decision is a pure function over
    // retained state (metadata_kind_correlation.hpp); this lambda only GATHERS that state, so the rule
    // itself is mutation-testable at the site that ships rather than through this registration.
    //
    // Both answers are positive correlations over metadata AND resource identity. An earlier version
    // answered DCC by elimination -- "not Float32x1, therefore colour" -- which classified an
    // uncorrelated Uint32x1 raw alias as DCC and let its all-0xff bytes authorize reading the base.
    prosper::gpu::set_metadata_kind_query(
        [](const prosper::gpu::MetadataKindRequest& request) {
            std::vector<prosper::gpu::RetainedDepthCorrelation> depth;
            for (const auto& [key, image] : prosper::test::persistent_ds_cache()) {
                (void)image;
                depth.push_back({key.dr, key.dw, key.sr, key.sw, key.htile});
            }
            std::vector<prosper::gpu::RetainedColorCorrelation> color;
            for (const auto& [addr, surface] : g_rtt)
                color.push_back({addr, surface.dcc_metadata_addr});
            return prosper::gpu::correlate_compression_metadata_kind(request, depth, color);
        });

    prosper::gpu::set_live_target_query([invalidate_ds](uint64_t addr) {
        drain_guest_gpu_writes(g_rtt, invalidate_ds);
        auto it = g_rtt.find(addr);
        if (it == g_rtt.end()) return false;
        const RttSurf& surface = it->second;
        const VkFormat format = prosper::test::backend_color_format(surface.format);
        const uint32_t bytes_per_pixel =
            prosper::test::backend_color_bytes_per_pixel(format);
        const bool has_cpu_snapshot = (surface.rgba &&
            prosper::frontend::live_rtt_cpu_snapshot_matches(
                surface.w, surface.h, bytes_per_pixel, surface.rgba->size())) ||
            surface.has_uniform_color;
        return prosper::frontend::live_rtt_compute_authoritative(
            surface.gpu_valid, has_cpu_snapshot);
    });
    prosper::gpu::set_live_target_reader(
        [invalidate_ds](uint64_t addr, prosper::gpu::LiveTargetSnapshot& snapshot) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            auto it = g_rtt.find(addr);
            if (it == g_rtt.end() || !it->second.w || !it->second.h) return false;
            RttSurf& surface = it->second;
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            const uint32_t bytes_per_pixel = prosper::test::backend_color_bytes_per_pixel(format);
            const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
            if (!bytes_per_pixel || texels > UINT64_MAX / bytes_per_pixel) return false;
            const uint64_t expected = texels * bytes_per_pixel;
            if ((!surface.rgba || surface.rgba->size() != expected) &&
                surface.has_uniform_color)
                materialize_uniform_rtt(surface);
            // PROSPER_RTT_REMAT_LOG=1 -- diagnostic only (#3407). A renderer-owned surface that
            // re-materializes on EVERY compute read is doing a full GPU->CPU readback per read, and
            // it also hands the consumer a fresh buffer each time, which is why a pointer-identity
            // conversion cache measured a structural 0% hit rate. This reports the split so the
            // question "is the readback repeated, and why" is answered by counting rather than by
            // inferring it from consumer-side pointers.
            {
                static const bool remat_log = std::getenv("PROSPER_RTT_REMAT_LOG") != nullptr;
                if (remat_log) {
                    static std::mutex m;
                    struct Counts { uint64_t reads, reuse, null_remat, size_remat, no_gpu; };
                    static std::map<uint64_t, Counts> rows;
                    std::lock_guard<std::mutex> lk(m);
                    Counts& c = rows[addr];
                    ++c.reads;
                    const bool bad = !surface.rgba || surface.rgba->size() != expected;
                    if (!bad) ++c.reuse;
                    else if (!surface.gpu_valid) ++c.no_gpu;
                    else if (!surface.rgba) ++c.null_remat;
                    else ++c.size_remat;
                    if (c.reads % 64 == 0)
                        std::fprintf(stderr,
                                     "[rtt-remat] addr=0x%llx reads=%llu reuse=%llu "
                                     "remat-null=%llu remat-size=%llu no-gpu=%llu bytes=%llu\n",
                                     (unsigned long long)addr,
                                     (unsigned long long)c.reads, (unsigned long long)c.reuse,
                                     (unsigned long long)c.null_remat,
                                     (unsigned long long)c.size_remat,
                                     (unsigned long long)c.no_gpu,
                                     (unsigned long long)expected);
                }
            }
            // Graphics-to-graphics intermediates normally stay in the persistent Vulkan target.
            // Compute cannot import that color attachment directly, so materialize its current
            // pixels only when an ordered compute dispatch actually consumes the surface.
            if ((!surface.rgba || surface.rgba->size() != expected) && surface.gpu_valid) {
                std::vector<uint8_t> materialized;
                std::string error;
                if (prosper::test::readback_persistent_color_target(
                        addr, surface.w, surface.h, format, materialized, error) &&
                    materialized.size() == expected) {
                    surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                        std::move(materialized));
                } else {
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        std::fprintf(stderr,
                                     "[rtt] live compute target readback failed: base=0x%llx "
                                     "extent=%ux%u error=%s\n",
                                     static_cast<unsigned long long>(addr), surface.w, surface.h,
                                     error.c_str());
                    return false;
                }
            }
            if (!surface.rgba || expected != surface.rgba->size()) return false;
            snapshot.width = surface.w;
            snapshot.height = surface.h;
            if (!prosper::frontend::live_target_pixel_format_from_vk(format, snapshot.format))
                return false;
            snapshot.pixels = surface.rgba;
            return true;
        });
    // Phase 2 of #1091 (#1095): with one shared device the compute backend can sample the renderer's
    // persistent image in place. Offer it only while gpu_valid proves that image is current. A CPU
    // snapshot may coexist as an ordered readback mirror; CPU-newer publications clear gpu_valid,
    // while guest GPU writes erase the cache entry before this callback runs (#780).
    // One dispatch can bind the same target through several descriptors, so pins are counted: the
    // compute backend releases once per successful import and the entry drops at zero.
    struct PinnedImport { uint32_t width, height; VkFormat format; uint32_t count; };
    static std::unordered_map<uint64_t, PinnedImport> pinned_imports;
    // gpu_replay registers a renderer from more than one entry point, so this can run twice in a
    // process. Pins are taken and released within a single dispatch, so the map is empty between
    // them; clear it anyway so a second registration cannot inherit counts for a cache that no
    // longer holds those entries.
    pinned_imports.clear();
    const bool direct_bind = !PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_RTT_BIND");
    prosper::gpu::set_live_target_image_importer(
        [invalidate_ds, direct_bind](uint64_t addr,
                                     const prosper::gpu::LiveTargetImageRequest& request,
                                     prosper::gpu::LiveTargetImageImport& import) {
            if (!direct_bind) return false;
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            // `allow_depth` is set only for a proven one-component Float32 sample or Uint32 raw-bit
            // view of an ordinary 2D descriptor. Prefer the matching DS plane before consulting the
            // address-only color registry: guest allocations are reused, and a stale RttSurf at the
            // same base must not hide the newer persistent depth image. Persistent DS entries are
            // not evicted, and compute runs in ordered submit execution, so unlike the bounded color
            // cache this path needs no pin.
            if (request.allow_depth && request.width && request.height) {
                const prosper::test::PersistentDsSampled sampled =
                    prosper::test::find_persistent_ds_sampled(
                        addr, request.width, request.height,
                        request.render_scale ? request.render_scale : 1u,
                        request.normalized_sampling);
                const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
                if (sampled.image && ctx.ok) {
                    import.width = sampled.width;
                    import.height = sampled.height;
                    import.kind = prosper::gpu::LiveTargetImageImport::Kind::Depth;
                    import.native_format = static_cast<uint32_t>(sampled.format);
                    import.image = sampled.image->image;
                    import.device = ctx.dev;
                    import.layout = static_cast<uint32_t>(
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    return true;
                }
            }
            auto it = g_rtt.find(addr);
            if (it == g_rtt.end()) return false;
            RttSurf& surface = it->second;
            if (!surface.w || !surface.h) return false;
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            // Map the backend format explicitly and fail closed. A direct bind hands the consumer a
            // real VkImage, so an unrecognized format must decline rather than be reported as rgba8:
            // the consumer would then build a mismatched view over the renderer's image.
            prosper::gpu::LiveTargetPixelFormat pixel_format;
            if (!prosper::frontend::live_target_pixel_format_from_vk(format, pixel_format))
                return false;
            const uint32_t bytes_per_pixel = prosper::test::backend_color_bytes_per_pixel(format);
            if (!bytes_per_pixel) return false;
            const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
            if (texels > UINT64_MAX / bytes_per_pixel) return false;
            const uint64_t expected = texels * bytes_per_pixel;
            const bool has_cpu_snapshot = surface.rgba && surface.rgba->size() == expected;
            if (!prosper::frontend::live_rtt_gpu_importable(surface.gpu_valid,
                                                             has_cpu_snapshot))
                return false;
            const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
            if (!ctx.ok) return false;
            prosper::test::PersistentColorTargetImage* target =
                prosper::test::find_persistent_color_target(addr, surface.w, surface.h, format);
            if (!target || !target->image || target->layout == VK_IMAGE_LAYOUT_UNDEFINED) return false;
            if (!prosper::test::pin_persistent_color_target(addr, surface.w, surface.h, format))
                return false;
            target->last_use = ++prosper::test::persistent_color_target_generation();
            // The unpin key is {addr, w, h, format}. A repeat import that disagreed with the
            // recorded key would corrupt the outstanding pin's release, so decline instead.
            PinnedImport& pin = pinned_imports[addr];
            if (pin.count &&
                (pin.width != surface.w || pin.height != surface.h || pin.format != format)) {
                prosper::test::unpin_persistent_color_target(addr, surface.w, surface.h, format);
                return false;
            }
            pin = {surface.w, surface.h, format, pin.count + 1};
            import.width = surface.w;
            import.height = surface.h;
            import.format = pixel_format;
            import.image = target->image;
            import.device = ctx.dev;
            import.layout = static_cast<uint32_t>(target->layout);
            import.transfer_dst = true;
            return true;
        },
        [](uint64_t addr) {
            auto pinned = pinned_imports.find(addr);
            if (pinned == pinned_imports.end()) return;
            PinnedImport& pin = pinned->second;
            prosper::test::unpin_persistent_color_target(addr, pin.width, pin.height, pin.format);
            if (--pin.count == 0) pinned_imports.erase(pinned);
        });
    prosper::gpu::set_live_target_image_written_notifier(
        [invalidate_ds](const prosper::gpu::LiveTargetImageWrite& write) {
            auto it = g_rtt.find(write.gpu_addr);
            if (it == g_rtt.end()) return;
            // Name the write's format exhaustively. Reporting an unmapped format as RGBA8 does not
            // merely mislabel it: the mirror-identity check below then rejects a target the compute
            // dispatch really did write, the entry stays invalidated by the ordinary guest-write
            // path, and every later consumer samples guest bytes prosper never wrote. That is the
            // whole Syberia black-3D-menu defect, and #773 in a different format.
            const VkFormat format = prosper::frontend::live_target_pixel_format_vk(write.format);
            if (format == VK_FORMAT_UNDEFINED) return;
            if (!prosper::frontend::live_rtt_mirror_identity_matches(
                    it->first, it->second.w, it->second.h,
                    static_cast<uint32_t>(
                        prosper::test::backend_color_format(it->second.format)),
                    write.gpu_addr, write.width, write.height,
                    static_cast<uint32_t>(format)))
                return;

            // The compute backend also wrote exact guest bytes. Process that ordinary notification
            // first so every color/depth/view alias becomes stale, then restore only the persistent
            // image that received the queue-ordered device copy. If the image disappeared, leave the
            // invalidation in force and let the next graphics use rebuild from the correct guest bytes.
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            if (!prosper::test::restore_persistent_color_target_after_mirrored_write(
                    write.gpu_addr, write.width, write.height, format))
                return;
            RttSurf& published = g_rtt[write.gpu_addr];
            rtt_remat_note_reset("mirrored-write", write.gpu_addr);
            published.rgba.reset();
            published.has_uniform_color = false;
            published.w = write.width;
            published.h = write.height;
            published.format = format;
            published.guest_format = format;
            published.gpu_valid = true;
        });
    prosper::gpu::set_live_target_byte_range_reader(
        [invalidate_ds](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            if (!addr || !bytes) return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
            for (auto& [base, surface] : g_rtt) {
                if (addr < base) continue;
                if (!surface.w || !surface.h) {
                    if (addr == base) return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                    continue;
                }
                const VkFormat format = prosper::test::backend_color_format(surface.format);
                const uint64_t bpp = prosper::test::backend_color_bytes_per_pixel(format);
                const uint64_t pixels = static_cast<uint64_t>(surface.w) * surface.h;
                if (!bpp || pixels > UINT64_MAX / bpp) continue;
                const uint64_t target_bytes = pixels * bpp;
                const uint64_t offset = addr - base;
                if (offset >= target_bytes) continue;
                if ((!surface.rgba || surface.rgba->size() != target_bytes) &&
                    surface.has_uniform_color)
                    materialize_uniform_rtt(surface);
                if ((!surface.rgba || surface.rgba->size() != target_bytes) &&
                    surface.gpu_valid) {
                    std::vector<uint8_t> materialized;
                    std::string error;
                    if (prosper::test::readback_persistent_color_target(
                            base, surface.w, surface.h, format, materialized, error) &&
                        materialized.size() == target_bytes) {
                        surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                            std::move(materialized));
                    } else {
                        static std::atomic<int> warned{0};
                        if (warned.fetch_add(1) < 24)
                            std::fprintf(stderr,
                                         "[rtt] ordered DMA target readback failed: base=0x%llx "
                                         "extent=%ux%u error=%s\n",
                                         static_cast<unsigned long long>(base), surface.w, surface.h,
                                         error.c_str());
                        return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                    }
                }
                if (!surface.rgba || surface.rgba->size() != target_bytes ||
                    bytes > target_bytes - offset)
                    return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                output.assign(surface.rgba->begin() + static_cast<size_t>(offset),
                              surface.rgba->begin() + static_cast<size_t>(offset + bytes));
                return prosper::gpu::LiveTargetByteReadResult::Success;
            }
            return prosper::gpu::LiveTargetByteReadResult::NotFound;
        });
    // Register unconditionally, as for RTT seeds above: the interactive F9 bundle has no capture
    // environment variable and snapshots persistent depth/stencil state only after it is armed.
    // Normal rendering pays nothing because the callback is otherwise never invoked (#1307).
    prosper::gpu::set_gpu_capture_ds_seed_snapshot_reader(
        [](std::vector<prosper::gpu::GpuCaptureDsSeed>& seeds, std::string& error) {
            return prosper::test::snapshot_persistent_ds_images(seeds, error);
        });
    if (getenv("PROSPER_GPU_REPLAY_RTT_SEEDS"))
        prosper::gpu::set_gpu_replay_rtt_seed_writer([](const prosper::gpu::GpuCaptureRttSeed& seed, std::string& error) {
            const VkFormat format = replay_color_format(seed.format);
            const uint64_t expected = static_cast<uint64_t>(seed.width) * seed.height *
                                      prosper::test::backend_color_bytes_per_pixel(format);
            if (!seed.guest_addr || !seed.width || !seed.height || expected != seed.rgba.size()) {
                error = "invalid temporal RTT seed"; return false;
            }
            RttSurf& surface = g_rtt[seed.guest_addr];
            surface.w = seed.width; surface.h = seed.height;
            surface.format = format;
            // Capture seeds contain canonical backend pixels. Older capture versions do not carry
            // the producing CB_COLOR order, so retain the exact format they do name.
            surface.guest_format = format;
            surface.rgba = std::make_shared<const std::vector<uint8_t>>(seed.rgba);
            surface.has_uniform_color = false;
            surface.gpu_valid = false;
            surface.dcc_metadata_dirty = false;
            prosper::test::invalidate_persistent_color_target(seed.guest_addr);
            return true;
        });
    if (getenv("PROSPER_GPU_REPLAY_DS_SEEDS"))
        prosper::gpu::set_gpu_replay_ds_seed_writer(
            [](const prosper::gpu::GpuCaptureDsSeed& seed, std::string& error) {
                return prosper::test::restore_persistent_ds_image(seed, error);
            });
    // A real command stream renders each CB_COLOR0_BASE into its own surface. Keep the old flattened
    // compositor only as a diagnostic fallback; it cannot preserve post chains or target extents.
    static const bool pertarget = getenv("PROSPER_RTT_PERTARGET") != nullptr ||
                                  PROSPER_ENV_VALUE("PROSPER_RTT_SINGLE_TARGET") == nullptr;
    static const bool rtt_on = getenv("PROSPER_RTT") != nullptr || pertarget;
    static const bool timeline_capture_requested =
        getenv("PROSPER_GPU_TIMELINE_CAPTURE") != nullptr;
    static const bool timeline_capture_phase_gated = timeline_capture_requested &&
        prosper::gpu::gpu_timeline_capture_is_after_compute_gated();
    static const bool timeline_capture_permits_live_targets =
        timeline_capture_allows_persistent_targets(
            timeline_capture_requested, timeline_capture_phase_gated);
    // Retain intermediate color targets on the GPU by default. Captures and per-target pixel
    // diagnostics require authoritative CPU pixels at every pass, so they retain the established
    // readback path. The explicit opt-out keeps a direct A/B and a recovery switch for driver issues.
    // Bundle replay normally requests CPU-visible RTT checkpoints so its inspection/export modes can
    // observe every pass. That fallback cannot carry MRT2..7 between independent render groups (the
    // public CPU seed parameters historically cover only MRT0/MRT1), and therefore is not an oracle
    // for a deferred frame that accumulates a wider G-buffer. This explicit diagnostic makes replay
    // use the production GPU-resident contract while still restoring the bundle's boundary seeds.
    // It is deliberately opt-in: export/readback diagnostics keep their established behaviour.
    static const bool replay_live_targets =
        PROSPER_ENV_VALUE("PROSPER_GPU_REPLAY_LIVE_TARGETS") != nullptr;
    static const bool live_gpu_targets = pertarget &&
        !PROSPER_ENV_VALUE("PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS") && !getenv("PROSPER_GPU_CAPTURE") &&
        timeline_capture_permits_live_targets && !PROSPER_ENV_VALUE("PROSPER_GPU_REPLAY_EXPORT_RTT") &&
        (!getenv("PROSPER_GPU_REPLAY_RTT_SEEDS") || replay_live_targets) &&
        !PROSPER_ENV_VALUE("PROSPER_DUMP_SAMPLED_RTT") &&
        !PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS") && !getenv("PROSPER_DUMP_RTGROUPS_RGBA") &&
        !PROSPER_ENV_VALUE("PROSPER_DUMP_DRAWSTEPS") &&
        !PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") && !PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_DIM") &&
        !PROSPER_ENV_VALUE("PROSPER_RTTLOG");
    if (timeline_capture_phase_gated)
        fprintf(stderr, "[render] phase-gated timeline capture retains live targets; "
                        "capture readback is on demand\n");
    static const bool defer_intermediate_scanout = live_gpu_targets &&
        !PROSPER_ENV_VALUE("PROSPER_NO_INTERMEDIATE_SCANOUT_DEFER");
    // Ordered passes share one queue and keep their attachments GPU-resident, so submit the callback's
    // command buffers in one batch by default. Keep a direct recovery/A-B switch for driver issues.
    static const bool batch_backend_submits = live_gpu_targets &&
        !PROSPER_ENV_VALUE("PROSPER_NO_BACKEND_BATCH_SUBMITS");
    if (live_gpu_targets)
        fprintf(stderr, "[render] persistent GPU color targets enabled (experimental%s)\n",
                replay_live_targets ? ", replay parity" : "");
    if (batch_backend_submits)
        fprintf(stderr, "[render] backend target-submit batching enabled (experimental)\n");
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps, invalidate_ds, gta5_native_fragment_vote](const std::vector<prosper::gpu::DrawItem>& items,
                               uint32_t w, uint32_t h) -> prosper::gpu::RenderedFrame {
            using RC = prosper::gpu::ResourceClass;
            // #2215 instrument: publish which thread is inside a submit-render callback right
            // now, so the thread sampler can attribute its samples EXACTLY instead of guessing
            // from a six-frame host-stack walk (a sample taken deep in ucrtbase loses the
            // renderer frame off the end of that walk and is misfiled as "outside").
            //
            // Measured: 2.92 submits/flip x 181.8 ms = 530 ms/flip inside these callbacks
            // against 938 ms/flip wall -- so ~43% of the collapsed frame is time when this
            // callback is NOT running, and nothing has ever instrumented that half.
            //
            // One slot, not a set: the renderer already documents a single-present-thread
            // contract (see g_rtt and the dp_submit statics above). A second concurrent caller
            // would overwrite it, so the reader treats a mismatch as "outside" rather than
            // asserting -- a wrong attribution is better than a crash in a diagnostic.
            prosper::frontend::ScopedRendererCallbackTid scoped_renderer_tid;
            // Compute fast-clears update a target's DCC metadata between graphics spans. Associate
            // those ranges before draining the ordered guest-write notifications so a metadata write
            // cannot leave the previous frame's retained target authoritative.
            register_cpu_rtt_dcc_metadata(g_rtt, items);
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            const prosper::gpu::LiveRenderPhase phase = prosper::gpu::live_render_phase();
            static const size_t write_watch_promotion_budget_bytes = [] {
                // 0 is not "off": an empty budget makes WriteWatchPromotionBudget::try_consume
                // return true unconditionally, i.e. unbounded arming per submit. A typo must keep
                // the default rather than select that (#3253).
                const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB");
                const uint64_t mib = prosper::diag::env_u64_or_default_capped(
                    "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", value, 8ull,
                    SIZE_MAX / (1024ull * 1024ull), "MiB");
                return static_cast<size_t>(mib * (1024ull * 1024ull));
            }();
            static thread_local prosper::frontend::WriteWatchPromotionBudget
                write_watch_promotion_budget;
            if (phase.first_span)
                write_watch_promotion_budget.reset(write_watch_promotion_budget_bytes);
            struct PinnedScanout {
                uint64_t id = 0;
                uint32_t width = 0, height = 0;
                VkFormat format = VK_FORMAT_UNDEFINED;
            };
            struct PinnedRendererMipTarget {
                uint64_t id = 0;
                uint32_t width = 0, height = 0;
                VkFormat format = VK_FORMAT_UNDEFINED;
                // A mipmapped T# has named this exact independently rendered level. Keep the pin
                // through the backend call that consumes it, then release it at the logical
                // submit's final span. Unconsumed producers deliberately cross submit boundaries.
                bool consumed = false;
            };
            static thread_local std::vector<PinnedScanout> pinned_scanouts;
            static thread_local std::vector<PinnedRendererMipTarget>
                pinned_renderer_mip_targets;
            auto release_pinned_scanouts = [&] {
                for (const PinnedScanout& target : pinned_scanouts)
                    prosper::test::unpin_persistent_color_target(
                        target.id, target.width, target.height, target.format);
                pinned_scanouts.clear();
            };
            auto release_consumed_renderer_mip_targets = [&] {
                auto target = pinned_renderer_mip_targets.begin();
                while (target != pinned_renderer_mip_targets.end()) {
                    if (!target->consumed) {
                        ++target;
                        continue;
                    }
                    prosper::test::unpin_persistent_color_target(
                        target->id, target->width, target->height, target->format);
                    target = pinned_renderer_mip_targets.erase(target);
                }
            };
            auto pin_renderer_mip_target = [&](uint64_t id, uint32_t width, uint32_t height,
                                               VkFormat format, bool gpu_valid) {
                if (!gpu_valid || !id || !width || !height ||
                    !prosper::frontend::renderer_mip_target_requires_submit_retention(format))
                    return;
                const auto already_pinned = std::find_if(
                    pinned_renderer_mip_targets.begin(), pinned_renderer_mip_targets.end(),
                    [&](const PinnedRendererMipTarget& target) {
                        return target.id == id && target.width == width &&
                               target.height == height && target.format == format;
                    });
                if (already_pinned != pinned_renderer_mip_targets.end()) {
                    // A new production supersedes any earlier consumption of the same identity.
                    // It must remain retained for the next mip consumer, which may be in a later
                    // logical submit.
                    already_pinned->consumed = false;
                    return;
                }
                // This is a narrow bridge for independently rendered packed-HDR mip levels, not a
                // second unbounded residency cache. GTA V's observed scene uses 22 identities;
                // retaining substantially more without one being consumed indicates a different
                // workload and must fall back visibly to the ordinary LRU contract.
                constexpr size_t kMaxPendingRendererMipTargets = 64;
                if (pinned_renderer_mip_targets.size() >= kMaxPendingRendererMipTargets) {
                    const PinnedRendererMipTarget oldest = pinned_renderer_mip_targets.front();
                    prosper::test::unpin_persistent_color_target(
                        oldest.id, oldest.width, oldest.height, oldest.format);
                    pinned_renderer_mip_targets.erase(pinned_renderer_mip_targets.begin());
                    static std::atomic<uint64_t> overflows{0};
                    const uint64_t occurrence = overflows.fetch_add(1) + 1;
                    if (prosper::diag_should_print(occurrence))
                        fprintf(stderr,
                                "[renderer-mips] pending producer cap reached #%llu; "
                                "released unconsumed target=0x%llx extent=%ux%u\n",
                                (unsigned long long)occurrence,
                                (unsigned long long)oldest.id, oldest.width, oldest.height);
                }
                if (prosper::test::pin_persistent_color_target(id, width, height, format)) {
                    pinned_renderer_mip_targets.push_back({id, width, height, format, false});
                    return;
                }
                // The target was proven GPU-valid immediately before this call, so a failed pin
                // means frontend/backend cache identity drift. Report it even without diagnostics:
                // silently continuing would recreate the missing-middle-mip corruption this pin
                // exists to prevent.
                static std::atomic<uint64_t> failures{0};
                const uint64_t occurrence = failures.fetch_add(1) + 1;
                if (prosper::diag_should_print(occurrence))
                    fprintf(stderr,
                            "[renderer-mips] submit-retention pin FAILED #%llu: "
                            "target=0x%llx extent=%ux%u format=%d\n",
                            (unsigned long long)occurrence, (unsigned long long)id,
                            width, height, (int)format);
            };
            auto consume_renderer_mip_chain = [&](const RendererMipChainLayout& chain,
                                                  uint32_t width, uint32_t height,
                                                  VkFormat format) {
                size_t matched = 0;
                for (uint32_t level = 0; level < chain.level_count; ++level) {
                    const uint64_t id = chain.level_ids[level];
                    const uint32_t level_width = std::max(width >> level, 1u);
                    const uint32_t level_height = std::max(height >> level, 1u);
                    for (PinnedRendererMipTarget& target : pinned_renderer_mip_targets) {
                        if (target.id != id || target.width != level_width ||
                            target.height != level_height || target.format != format)
                            continue;
                        if (!target.consumed) ++matched;
                        target.consumed = true;
                        break;
                    }
                }
                return matched;
            };
            // A terminal callback normally releases every pin. Recover conservatively if an earlier
            // submit was aborted after rendering but before frontend finalization.
            if (phase.first_span && !pinned_scanouts.empty()) release_pinned_scanouts();
            if (phase.first_span && !pinned_renderer_mip_targets.empty())
                release_consumed_renderer_mip_targets();
            // PROSPER_RENDER_FIRST=<N>: skip the slow (~400x) Vulkan render for the first N GPU submits, so
            // the game reaches a LATE scene (e.g. the level1 cutscene, which only starts submitting after
            // ~5000 title-loop submits) at native speed before we begin rendering/dumping. Returning {}
            // means "not rendered this submit". Without this, rendering from boot is far too slow to ever
            // reach a post-loading-screen scene.
            static std::atomic<int> g_submit_idx{0};
            static int g_render_first = getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0;
            // PROSPER_RENDER_DELAY_MS=<N>: wall-clock warmup for titles whose useful scene begins after
            // a variable number of submits. The guest and command decoder keep running at native speed;
            // only the synchronous Vulkan work is skipped. The clock starts at the first GPU submit.
            static const int64_t g_render_delay_ms = getenv("PROSPER_RENDER_DELAY_MS")
                ? std::max<int64_t>(0, atoll(PROSPER_ENV_VALUE("PROSPER_RENDER_DELAY_MS"))) : 0;
            static const auto g_render_delay_start = std::chrono::steady_clock::now();
            static std::atomic<bool> g_render_delay_announced{false};
            // PROSPER_RENDER_LAST=<N>: stop rendering after submit N (default: unbounded). Bounds the render
            // window so a diagnostic slice at a late stall (RENDER_FIRST..RENDER_LAST) does not accumulate
            // unbounded RTT/GPU resources across tens of thousands of submits (which OOM-kills the process).
            static int g_render_last = getenv("PROSPER_RENDER_LAST") ? atoi(getenv("PROSPER_RENDER_LAST")) : INT_MAX;
            static thread_local int g_this_submit = -1;
            static thread_local bool g_force_this_submit = false;
            if (phase.first_span || g_this_submit < 0) {
                g_this_submit = g_submit_idx++;
                g_force_this_submit = false;
            }
            if (g_this_submit > g_render_last) return {};
            static const int g_rttlog_min_submit = getenv("PROSPER_RTTLOG_MIN_SUBMIT")
                ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MIN_SUBMIT"))) : 0;
            static const int g_rttlog_max_submit = getenv("PROSPER_RTTLOG_MAX_SUBMIT")
                ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MAX_SUBMIT"))) : INT_MAX;
            scan_for_descriptors_once(static_cast<uint64_t>(g_this_submit < 0 ? 0 : g_this_submit));
            const bool rtt_log_in_range =
                g_this_submit >= g_rttlog_min_submit && g_this_submit <= g_rttlog_max_submit;
            const bool rtt_log = PROSPER_ENV_VALUE("PROSPER_RTTLOG") && rtt_log_in_range;
            using RenderClock = std::chrono::steady_clock;
            struct RenderTiming {
                uint64_t callbacks = 0;
                double total_ms = 0, prelude_ms = 0, pass_ms = 0;
                double build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
                double dcc_materialize_ms = 0;
                uint64_t dcc_materialize_surfaces = 0, dcc_materialize_bytes = 0;
                uint64_t backend_calls = 0, backend_draws = 0;
                uint64_t backend_command_buffers = 0, backend_queue_submits = 0;
                uint64_t backend_fence_waits = 0;
                // Why each of the ~16 flushes a submit happened (#2276). Exclusive and summing to
                // the flush count, so the arithmetic is checkable from the printed line alone.
                uint64_t flush_no_batch = 0, flush_readback = 0;
                uint64_t flush_storage_writeback = 0, flush_explicit = 0;
                uint64_t backend_gpu_timestamp_samples = 0;
                double backend_target_ms = 0, backend_draw_setup_ms = 0;
                double backend_record_upload_ms = 0, backend_gpu_wait_ms = 0;
                double backend_gpu_device_ms = 0;
                double backend_readback_ms = 0, backend_cleanup_ms = 0;
                double backend_setup_shader_ms = 0, backend_setup_fixed_ms = 0;
                double backend_res_fixed_index_upload_ms = 0;
                double backend_res_fixed_blend_ms = 0;
                double backend_res_fixed_depth_stencil_ms = 0;
                double backend_res_fixed_viewport_ms = 0;
                double backend_res_fixed_stages_ms = 0;
                double backend_res_fixed_prologue_ms = 0;
                double backend_res_prologue_subgroup_scan_ms = 0;
                double backend_setup_resources_ms = 0, backend_setup_pipeline_ms = 0;
                // #1284: sub-attribution of backend_setup_resources_ms. res_texture/res_buffer
                // cover the whole per-resource branch; the upload/bind pair is nested inside
                // res_texture and covers only cache-MISS work, so the remainder is what a cache
                // hit still pays per reference. res_descriptor is the per-draw set alloc/update.
                double backend_res_texture_ms = 0, backend_res_texture_upload_ms = 0;
                double backend_res_texture_bind_ms = 0, backend_res_buffer_ms = 0;
                double backend_res_buffer_acquire_ms = 0, backend_res_buffer_copy_ms = 0;
                double backend_res_buffer_create_ms = 0;
                double backend_res_buffer_index_find_ms = 0;
                double backend_res_buffer_index_insert_ms = 0;
                double backend_res_buffer_hash_ms = 0;
                double backend_res_descriptor_ms = 0;
                uint64_t backend_pipeline_refs = 0, backend_pipeline_hits = 0;
                uint64_t backend_pipeline_misses = 0, backend_pipeline_bypasses = 0;
                uint64_t backend_pipeline_entries = 0, backend_pipeline_evictions = 0;
                uint64_t color_target_writes = 0, color_target_write_hits = 0;
                uint64_t color_target_sample_hits = 0, color_target_readbacks = 0;
                uint64_t color_target_cached_bytes = 0, color_target_cached_entries = 0;
                uint64_t textures = 0, texture_reuses = 0, buffers = 0, buffer_views = 0;
                uint64_t persistent_hits = 0, persistent_misses = 0, persistent_invalidations = 0;
                uint64_t persistent_submit_reuses = 0, persistent_validations = 0;
                uint64_t persistent_validation_bytes = 0;
                uint64_t persistent_watch_reuses = 0, persistent_watch_dirty = 0;
                uint64_t persistent_watch_unknown = 0, persistent_watch_disabled = 0;
                // WHY in-submit reuse never fires (#2289). persistent_submit_reuses is 0 across
                // 55,820 hits on PPSA25009, and the two ways that happens have OPPOSITE prospects:
                // Unknown means prosper is not tracking guest writes on this path at all (fixable,
                // and worth ~9x, since the same texture is referenced 9.5 times per submit), while
                // Overlap means the guest genuinely rewrote those bytes and the revalidation is
                // CORRECT. A single zero cannot tell those apart, so count them separately.
                uint64_t persistent_submit_unknown = 0, persistent_submit_overlap = 0;
                // ...and WHY the Unknowns were unknown (#2289). Two structurally different
                // causes that a single Unknown count cannot separate, and the difference
                // decides whether the cost is addressable by the submit journal AT ALL:
                //   unarmed  the journal is not armed on this thread -> missing instrumentation
                //   stale    it IS armed, but the snapshot came from an EARLIER submit. That is
                //            not a defect: the journal is intra-submit by construction, so a
                //            cross-submit question can only ever answer Unknown through it,
                //            and only a cross-submit WATCH can answer it at all.
                uint64_t persistent_submit_unarmed = 0, persistent_submit_stale = 0;
                // #2283's blocking arm. publish_selected_fmt0 must stay 0 over a full route for
                // the disabled-target readback skip to be safe by construction. `unknown` is
                // separate so a selection whose source pass could not be identified is never
                // silently counted as a clean one.
                uint64_t publish_selected = 0, publish_selected_fmt0 = 0;
                uint64_t publish_selected_unknown = 0, publish_candidate_fmt0 = 0;
                uint64_t texture_bytes = 0, buffer_bytes = 0, buffer_materialized_bytes = 0;
                double texture_ms = 0, buffer_ms = 0;
                // Attribution of texture_ms by OUTCOME class (#2262). After #2259 removed the
                // reflection cost this is the largest leaf of build_R -- 22.56 ms/submit on a Blue
                // Prince gameplay window, 55% of build_R -- and it is one span with nothing inside
                // it, so "4,449 references at an 89% reuse rate cost 22.56 ms" is as far as anyone
                // can currently get.
                //
                // An attribution rather than a phase partition, deliberately: these classes are
                // already computed as locals on the per-resource path and are decided in a fixed
                // order, so classifying in that same order makes them mutually exclusive by
                // construction -- whereas carving a 2,800-line branch into timed phases invites the
                // double-counting #2246 and #2250 had to be careful about.
                double tex_rtt_ms = 0, tex_compute_ms = 0, tex_local_ms = 0;
                double tex_persist_hit_ms = 0, tex_persist_reuse_ms = 0, tex_persist_miss_ms = 0;
                uint64_t tex_rtt_n = 0, tex_compute_n = 0, tex_local_n = 0;
                uint64_t tex_persist_hit_n = 0, tex_persist_reuse_n = 0, tex_persist_miss_n = 0;
                // The cache entry was FOUND and its guest bytes had changed, so the decode ran
                // again. It had no bucket of its own, which put it in `other` alongside "this
                // resource was never cacheable at all" -- two states with opposite fixes. On Stray's
                // title screen 930.78 of the texture leaf's 1110.10 ms landed in that undivided
                // residual, and naming this half is what identified the two 63.75 MiB RGBA16F HDR
                // intermediates that re-decode every frame (#3149 measured the splash, not this).
                double tex_persist_invalid_ms = 0;
                uint64_t tex_persist_invalid_n = 0;
                uint64_t tex_other_n = 0;
                double tex_other_slowest_ms = 0;
                uint64_t tex_other_addr = 0, tex_other_source_bytes = 0;
                uint32_t tex_other_width = 0, tex_other_height = 0, tex_other_depth = 0;
                uint32_t tex_other_format = 0, tex_other_components = 0;
                uint32_t tex_other_tile_mode = 0, tex_other_img_dim = 0, tex_other_class = 0;
                bool tex_other_compute_candidate = false;
                bool tex_other_persistent_candidate = false;
                bool tex_other_compressed = false, tex_other_depth_compare = false;
                bool tex_other_host_backed = false;
                // Exhaustive partition of build_resources_ms, the frontend materialiser (#2215).
                // It had two named leaves -- texture_ms and buffer_ms, both from INSIDE build_R --
                // covering 10.2 of 48.95 ms on a Blue Prince gameplay submit, with no assertion
                // that they were exhaustive. 38.76 ms, 21.1% of the frame, was attributable to
                // nothing. That is not the same as being spent on nothing, and it is
                // indistinguishable from it in a report.
                //
                // These four plus a SIGNED residual cover the per-draw body of build_bds. The
                // residual is published rather than clamped for the reason #2246 records: an
                // unmeasured region does not read as unmeasured, it reads as someone else's number.
                // pass_control is a RESIDUAL, not a measurement: pass_ms - build_resources_ms -
                // backend_ms. On a Blue Prince gameplay window it is 85.57 ms/submit -- 27% of the
                // frame -- attributed to nothing, which is the state build_resources was in before
                // #2250: an unmeasured region does not read as unmeasured, it reads as somebody
                // else's cost.
                //
                // Two spans bound the per-group work either side of the measured pair. They are
                // accumulated where a `continue` cannot skip past them silently: `pre` at the same
                // place build_resources_ms is recorded, `post` at the loop's end. A group that
                // continues before reaching the build contributes to neither and lands in the
                // residual, which is correct -- and visible, rather than folded into a leaf.
                double pass_pre_ms = 0, pass_post_ms = 0;
                double post_stats_ms = 0, post_slot0_ms = 0, post_mrt_ms = 0,
                       post_rest_ms = 0;
                // pre+post measured 0.23 ms of a 43.85 ms pass_control, so the cost is NOT the
                // per-group work either side of the measured pair. head/tail bound the two regions
                // outside the group loop. `tail` subtracts the build_resources+backend the tail
                // itself performs -- those are already excluded from pass_control, so counting them
                // here would make the leaves exceed the parent.
                double pass_head_ms = 0, pass_tail_ms = 0;
                // head+pre+post+tail measured 1.94 ms of 60.15, so the cost is in none of them.
                // The only region left is the group loop itself -- specifically iterations that
                // return before reaching the render, whose time no span above can see. `loop_ms`
                // is the whole loop; subtracting pre+post and the build/backend the loop performed
                // leaves exactly the early-exit time, without having to instrument every `continue`
                // (there are several, and one missed would under-report silently).
                double pass_loop_ms = 0;
                // The group loop has exactly ONE early exit -- the MSAA resolve at :5520 -- so the
                // 49.4 ms difference between `loop` and pre+post+build+backend is entirely resolve
                // work. Split into the three things a resolve does, because they have completely
                // different fixes: `stall` is submit_and_wait (a full GPU pipeline flush), `read` is
                // a whole-surface GPU->CPU readback, `copy` is the GPU-side destination copy that
                // already exists and is presumably cheap.
                double resolve_stall_ms = 0, resolve_read_ms = 0, resolve_copy_ms = 0;
                // The GPU copy's OWN submit_and_wait (#1382 requires it). Separate from `stall`
                // because removing the eager readback relocated cost into it rather than
                // eliminating it -- the copy's flush now drains a queue the eager flush used to.
                // Leaving it unmeasured would put an unmeasured region back inside `loop`, which is
                // the exact defect this partition exists to remove.
                double resolve_copy_stall_ms = 0;
                uint64_t resolve_n = 0, resolve_read_n = 0, resolve_bytes = 0;
                uint64_t pass_groups_seen = 0;
                uint64_t pass_groups = 0;
                double build_r_ms = 0, build_validate_ms = 0;
                double build_poison_ms = 0, build_indices_ms = 0;
                uint64_t build_draws = 0, build_rejected = 0;
                // One level deeper, into build_R's own residual (#2256). `identities` is a SET and
                // not a counter on purpose: its size is how many distinct shaders a submit-scoped
                // memo would have to hold, which is the ceiling on what such a memo could save.
                // It is cleared per submit with the rest of this struct.
                double build_reflect_ms = 0;
                uint64_t build_reflect_calls = 0, build_reflect_words = 0;
                uint64_t build_reflect_unidentified = 0;
                uint64_t build_reflect_memo_hits = 0, build_reflect_memo_clears = 0;
                uint64_t build_reflect_memo_mismatch = 0;
                std::unordered_set<uint64_t> build_reflect_identities;
            };
            struct RttTimingRecord {
                int submit = 0;
                uint64_t target = 0;
                uint32_t width = 0, height = 0;
                size_t draws = 0;
                bool first_span = false, final_span = false;
                bool authoritative_readback = false, deferred_readback = false;
                double measured_ms = 0;
                prosper::test::BackendRenderTimingStats timing;
                prosper::test::BackendColorTargetStats color_target;
            };
            static thread_local RenderTiming pending_timing;
            static thread_local std::vector<RttTimingRecord> pending_rtt_timing;
            const bool perf_capture_timing =
                prosper::perf::interactive_performance_capture().detailed_timing_active();
            const bool timing_capture_only =
                getenv("PROSPER_RENDER_TIMING_CAPTURE_ONLY") != nullptr;
            const bool timing_log_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr &&
                (!timing_capture_only || perf_capture_timing);
            const prosper::frontend::PerformanceTimingMode timing_mode =
                prosper::frontend::performance_timing_mode(timing_log_enabled,
                                                            perf_capture_timing);
            const bool timing_enabled = timing_mode.measure;
            // Read ONCE per span. The guard below sits inside the per-RESOURCE loop, where a
            // getenv costs a locked environment scan for every resource of every draw -- it was
            // the single hottest line in #2215's profile (26 of 349 primary-thread samples).
            //
            // Deliberately a LIVE read rather than PROSPER_ENV_VALUE: tests arm this variable at
            // runtime, so it has to stay observable. Hoisting removes the per-resource cost
            // WITHOUT changing that -- which is why it is the better fix here than caching.
            const char* const render_timing_env = getenv("PROSPER_RENDER_TIMING");
            const bool render_timing_detail =
                ((render_timing_env && strcmp(render_timing_env, "detail") == 0 &&
                  (!timing_capture_only || perf_capture_timing)) ||
                 (perf_capture_timing &&
                  getenv("PROSPER_RENDER_TIMING_DETAIL_CAPTURE") != nullptr));
            // render_runner.h cannot depend on the app capture singleton: it is also compiled into
            // standalone Vulkan tests. This thread-local scope activates its backend clocks only
            // inside this production callback and restores the prior state on every return path.
            prosper::frontend::ScopedInteractivePerformanceTiming scoped_perf_timing(
                perf_capture_timing);
            const bool lightweight_rtt_timing = timing_mode.log && PROSPER_ENV_VALUE("PROSPER_RTT_TIMING");
            static const uint64_t rtt_timing_min_draws = getenv("PROSPER_RTT_TIMING_MIN_DRAWS")
                ? strtoull(PROSPER_ENV_VALUE("PROSPER_RTT_TIMING_MIN_DRAWS"), nullptr, 0) : 0;
            if (timing_enabled && phase.first_span) {
                pending_timing = {};
                pending_rtt_timing.clear();
            }
            const auto callback_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            auto record_backend_timing = [&]
                (const prosper::test::BackendRenderTimingStats& backend,
                 const prosper::test::BackendPipelineCacheStats& pipelines,
                 const prosper::test::BackendResourceReuseStats& reuse) {
                pending_timing.backend_calls += backend.calls;
                pending_timing.backend_draws += backend.draws;
                pending_timing.backend_command_buffers += backend.command_buffers;
                pending_timing.backend_queue_submits += backend.queue_submits;
                pending_timing.backend_fence_waits += backend.fence_waits;
                pending_timing.flush_no_batch += backend.flush_no_batch;
                pending_timing.flush_readback += backend.flush_readback;
                pending_timing.flush_storage_writeback += backend.flush_storage_writeback;
                pending_timing.flush_explicit += backend.flush_explicit;
                pending_timing.backend_gpu_timestamp_samples += backend.gpu_timestamp_samples;
                pending_timing.backend_target_ms += backend.target_ms;
                pending_timing.backend_draw_setup_ms += backend.draw_setup_ms;
                pending_timing.backend_record_upload_ms += backend.record_upload_ms;
                pending_timing.backend_gpu_wait_ms += backend.gpu_wait_ms;
                pending_timing.backend_gpu_device_ms += backend.gpu_device_ms;
                pending_timing.backend_readback_ms += backend.readback_ms;
                pending_timing.backend_cleanup_ms += backend.cleanup_ms;
                pending_timing.backend_setup_shader_ms += backend.setup_shader_ms;
                pending_timing.backend_setup_fixed_ms += backend.setup_fixed_ms;
                pending_timing.backend_res_fixed_index_upload_ms += backend.res_fixed_index_upload_ms;
                pending_timing.backend_res_fixed_blend_ms += backend.res_fixed_blend_ms;
                pending_timing.backend_res_fixed_depth_stencil_ms += backend.res_fixed_depth_stencil_ms;
                pending_timing.backend_res_fixed_viewport_ms += backend.res_fixed_viewport_ms;
                pending_timing.backend_res_fixed_stages_ms += backend.res_fixed_stages_ms;
                pending_timing.backend_res_fixed_prologue_ms += backend.res_fixed_prologue_ms;
                pending_timing.backend_res_prologue_subgroup_scan_ms += backend.res_prologue_subgroup_scan_ms;
                pending_timing.backend_setup_resources_ms += backend.setup_resources_ms;
                pending_timing.backend_res_texture_ms += backend.res_texture_ms;
                pending_timing.backend_res_texture_upload_ms += backend.res_texture_upload_ms;
                pending_timing.backend_res_texture_bind_ms += backend.res_texture_bind_ms;
                pending_timing.backend_res_buffer_ms += backend.res_buffer_ms;
                pending_timing.backend_res_buffer_acquire_ms += backend.res_buffer_acquire_ms;
                pending_timing.backend_res_buffer_copy_ms += backend.res_buffer_copy_ms;
                pending_timing.backend_res_buffer_create_ms += backend.res_buffer_create_ms;
                pending_timing.backend_res_buffer_index_find_ms += backend.res_buffer_index_find_ms;
                pending_timing.backend_res_buffer_index_insert_ms += backend.res_buffer_index_insert_ms;
                pending_timing.backend_res_buffer_hash_ms += backend.res_buffer_hash_ms;
                pending_timing.backend_res_descriptor_ms += backend.res_descriptor_ms;
                pending_timing.backend_setup_pipeline_ms += backend.setup_pipeline_ms;
                pending_timing.backend_pipeline_refs += pipelines.references;
                pending_timing.backend_pipeline_hits += pipelines.hits;
                pending_timing.backend_pipeline_misses += pipelines.misses;
                pending_timing.backend_pipeline_bypasses += pipelines.bypasses;
                pending_timing.backend_pipeline_entries = pipelines.entries;
                pending_timing.backend_pipeline_evictions += pipelines.evictions;
                // Buffer-copy economics. This leaf dominates the Blue Prince FMV collapse -- 539 ms
                // of 802 ms of backend resource setup, 67% (#2215) -- and nothing could say WHY: the
                // counters are collected per call and aggregated into the thread-local storage, and
                // the ONLY consumers in the tree were assertions in test_multidraw_render.cpp. The
                // sibling resource kind has reported its cache economics for ages
                // ("[render-timing] texture_cache hits=... misses=..."), while buffers -- an order of
                // magnitude more expensive in that regime -- reported nothing at all.
                //
                // Per SUBMIT rather than as a window mean, deliberately. The question is which of
                // three mutually exclusive stories produces the copy volume, and they need different
                // fixes, so an average across them answers none of it:
                //   skipped_unique dominates -> buffer_identity == 0, nothing can dedup, fix upstream
                //   skipped_large dominates  -> the 4 KiB hash-dedup cap is working as designed and
                //                               the payloads are genuinely distinct; the fix is not
                //                               dedup at all but not staging through a mapped copy
                //   refs >> memo_hits        -> repeat references within one call are being missed
                if (timing_mode.log) {
                    // CUMULATIVE, not a per-call snapshot. The first draft printed one call's
                    // numbers at diag_should_print's ordinals (first 64, then powers of two) --
                    // which is a UNIFORM sample of a heavily skewed distribution, so it could not
                    // see the expensive calls by construction. The sampled calls summed to ~290 KB
                    // per submit against 30 ms of measured copy: 290 KB cannot take 30 ms, and that
                    // contradiction is the only reason the sampling flaw was caught.
                    //
                    // Totals cannot miss a call. The rate limit now controls how often the running
                    // total is PRINTED, never which calls it counts.
                    static uint64_t refs = 0, uniq = 0, memo = 0, hashed = 0, dwords = 0;
                    static uint64_t skipped_large = 0, skipped_unique = 0, fallbacks = 0, calls = 0;
                    static uint64_t large_dwords = 0, upload_bytes = 0;
                    refs += reuse.buffer_references;      uniq += reuse.unique_buffers;
                    memo += reuse.buffer_ref_memo_hits;   hashed += reuse.buffer_hash_calls;
                    dwords += reuse.buffer_hash_dwords;   fallbacks += reuse.buffer_upload_fallbacks;
                    skipped_large += reuse.buffer_hash_skipped_large;
                    skipped_unique += reuse.buffer_hash_skipped_unique;
                    large_dwords += reuse.buffer_skipped_large_dwords;
                    upload_bytes += reuse.buffer_upload_bytes;
                    if (prosper::diag_should_print(++calls))
                        fprintf(stderr,
                                "[render-timing] buffer_reuse (cumulative over %llu backend calls) "
                                "refs=%llu unique=%llu memo_hits=%llu hashed=%llu (%.1f MiB) "
                                "skipped_large=%llu (%.1f MiB) skipped_unique=%llu "
                                "COPIED=%.1f MiB upload_fallbacks=%llu\n",
                                (unsigned long long)calls, (unsigned long long)refs,
                                (unsigned long long)uniq, (unsigned long long)memo,
                                (unsigned long long)hashed, (double)dwords * 4.0 / (1024.0 * 1024.0),
                                (unsigned long long)skipped_large,
                                (double)large_dwords * 4.0 / (1024.0 * 1024.0),
                                (unsigned long long)skipped_unique,
                                (double)upload_bytes / (1024.0 * 1024.0),
                                (unsigned long long)fallbacks);
                }
            };
            auto append_rtt_timing = [](std::string& output, const RttTimingRecord& record) {
                const prosper::test::BackendRenderTimingStats& timing = record.timing;
                const double detail_ms = timing.total_ms();
                char line[512];
                const int length = snprintf(
                    line, sizeof(line),
                    "[rtt-timing] submit=%d target=0x%llx extent=%ux%u draws=%zu "
                    "span=%d/%d authoritative=%d deferred=%d cmd=%llu submit=%llu wait=%llu "
                    "measured=%.2f detail=%.2f other=%.2f target_setup=%.2f "
                    "draw_setup=%.2f record_upload=%.2f batch_wait=%.2f "
                    "batch_device=%.2f batch_overhead=%.2f batch_timestamps=%llu "
                    "readback=%.2f cleanup=%.2f gpu_target=%llu load=%llu sample=%llu cpu=%llu\n",
                    record.submit, (unsigned long long)record.target,
                    record.width, record.height, record.draws,
                    record.first_span, record.final_span, record.authoritative_readback,
                    record.deferred_readback,
                    (unsigned long long)timing.command_buffers,
                    (unsigned long long)timing.queue_submits,
                    (unsigned long long)timing.fence_waits,
                    record.measured_ms, detail_ms,
                    record.measured_ms - detail_ms,
                    timing.target_ms, timing.draw_setup_ms, timing.record_upload_ms,
                    timing.gpu_wait_ms, timing.gpu_device_ms,
                    std::max(0.0, timing.gpu_wait_ms - timing.gpu_device_ms),
                    (unsigned long long)timing.gpu_timestamp_samples,
                    timing.readback_ms, timing.cleanup_ms,
                    (unsigned long long)record.color_target.writes,
                    (unsigned long long)record.color_target.write_hits,
                    (unsigned long long)record.color_target.sampled_hits,
                    (unsigned long long)record.color_target.readbacks);
                if (length > 0)
                    output.append(line, std::min<size_t>(length, sizeof(line) - 1));
            };
            auto print_rtt_timing = [&](const RttTimingRecord& record) {
                std::string output;
                append_rtt_timing(output, record);
                if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
            };
            // PROSPER_SUBMITLOG: print the GPU-submit index periodically (at native speed, before the slow
            // render) so it can be correlated with guest-side log lines (e.g. a MsgDialog wait) to find the
            // exact submit at which a scene appears — for aiming PROSPER_RENDER_FIRST at it.
            if (phase.first_span && PROSPER_ENV_ON("PROSPER_SUBMITLOG") && (g_this_submit % 1000 == 0))
                fprintf(stderr, "[submit] index=%d (%zu draw items)\n", g_this_submit, items.size());
            if (const char* sd = PROSPER_ENV_VALUE("PROSPER_SUBMITLOG_DIM")) {
                uint32_t sw = 0, sh = 0;
                if (sscanf(sd, "%ux%u", &sw, &sh) == 2)
                    for (const auto& it : items)
                        if (it.color0_width == sw && it.color0_height == sh) {
                            fprintf(stderr, "[submit] index=%d target=0x%llx extent=%ux%u (%zu draw items)\n",
                                    g_this_submit, (unsigned long long)it.color0_base, sw, sh, items.size());
                            break;
                        }
            }
            bool force_target = false;
            if (const char* td = PROSPER_ENV_VALUE("PROSPER_RENDER_TARGET_DIM")) {
                uint32_t tw = 0, th = 0;
                if (sscanf(td, "%ux%u", &tw, &th) == 2)
                    for (const auto& it : items)
                        if (it.color0_width == tw && it.color0_height == th) { force_target = true; break; }
            }
            if (const char* rd = PROSPER_ENV_VALUE("PROSPER_RENDER_RESOURCE_DIM")) {
                uint32_t rw = 0, rh = 0;
                if (sscanf(rd, "%ux%u", &rw, &rh) == 2)
                    for (const auto& it : items) {
                        auto has_dim = [&](const prosper::gpu::ShaderResourceTable* table) {
                            if (!table) return false;
                            for (const auto& r : table->resources)
                                if (r.width == rw && r.height == rh) return true;
                            return false;
                        };
                        if (has_dim(it.vrt.get()) || has_dim(it.prt.get())) { force_target = true; break; }
                    }
            }
            // Ordered submits can contain several graphics spans separated by compute work. Once a
            // diagnostic target selects one span, keep rendering the rest of that transaction so the
            // final span can recover and return the scanout assembled in the persistent RTT cache.
            g_force_this_submit |= force_target;
            // A one-time offscreen producer may occur before a much later consumer render window.
            // Render that target even before the warmup ends so its RTT cache entry survives skipped
            // intermediate submits (#526). Submit-count and wall-clock gates are additive.
            const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - g_render_delay_start).count();
            const bool before_delay = elapsed_ms < g_render_delay_ms;
            if (!before_delay && g_render_delay_ms > 0 &&
                !g_render_delay_announced.exchange(true, std::memory_order_relaxed)) {
                fprintf(stderr, "[render] wall-clock warmup complete after %lld ms at submit %d\n",
                        (long long)elapsed_ms, g_this_submit);
            }
            if ((g_this_submit < g_render_first || before_delay) && !g_force_this_submit) return {};
            // Dump the FIRST item's recompiled SPIR-V (diagnostic; survives a mid-render crash).
            if (PROSPER_ENV_ON("PROSPER_SHADER_DUMP") && !items.empty()) {
                std::string d = PROSPER_ENV_VALUE("PROSPER_SHADER_DUMP");
                const auto& dump_vs = items[0].vs_words();
                const auto& dump_fs = items[0].fs_words();
                if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(dump_vs.data(), 4, dump_vs.size(), f); fclose(f); }
                if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(dump_fs.data(), 4, dump_fs.size(), f); fclose(f); }
                fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", dump_vs.size(), dump_fs.size()); fflush(stderr);
            }
            // Copy [a, a+n) into dst, but stop at the first 64KB block that is NOT within a reserved guest
            // mapping (prosper_reserved_range_state == 0). A resource's declared size (e.g. a 2048x1024
            // atlas = 8 MB) can run past its real committed backing; an unguarded memcpy then walks off
            // the mapping and SIGSEGVs the render thread mid-copy — /dev/null-write "readable" can't catch
            // it because /dev/null discards without faulting. Returns bytes copied; dst is pre-zeroed so a
            // short copy just leaves a transparent/black tail (only the missing region degrades).
            // How many LEADING bytes of [a, a+n) are mapped, i.e. exactly how many `safe_copy` will
            // deliver. Split out of `safe_copy` so a caller that needs all n of them can read guest
            // memory IN PLACE instead of staging a full-surface copy first (see
            // `direct_resource_source`): the two then cannot disagree about what "readable" means,
            // which they would within a week if the walk were written twice.
            //
            // `safe_copy` used to interleave the per-chunk mapping check with a per-chunk memcpy;
            // this proves the whole prefix first and then copies it in one call. The set of checks
            // and the set of bytes copied are identical -- a mapping that changes mid-walk was
            // equally fatal before -- and one 64 MiB memcpy beats a thousand 64 KiB ones.
            auto safe_span = [](uint64_t a, size_t n) -> size_t {
                if (!n) return 0;
                const size_t PG = 0x10000;   // lazy-commit granularity (64 KB)
#ifdef _WIN32
                // Prepare a complete sparse direct-memory resource once. The per-chunk mapping
                // checks below still stop an over-declared resource at its real guest boundary.
                if (prosper_try_commit_dmem(a, n, 0)) return n;
#endif
                size_t done = 0;
                while (done < n) {
                    uint64_t cur = a + done;
                    if (cur < 0x1000 || prosper_reserved_range_state(cur) == 0) break;
                    done += std::min(n - done, PG - (size_t)(cur & (PG - 1)));
                }
                return done;
            };
            auto safe_copy = [safe_span](uint8_t* dst, uint64_t a, size_t n) -> size_t {
                const size_t done = safe_span(a, n);
                if (done) std::memcpy(dst, (const void*)(uintptr_t)a, done);
                return done;
            };
            // A uniform DCC clear is self-contained in metadata. If the ordered compute span dirtied
            // a retained target's metadata, materialize that clear now so the following graphics pass
            // loads the clear value rather than stale pixels (or an arbitrary fallback clear).
            const auto dcc_materialize_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            uint64_t dcc_materialize_surfaces = 0;
            uint64_t dcc_materialize_bytes = 0;
            for (const auto& item : items) {
                const prosper::gpu::ShaderResourceTable* tables[] = {
                    item.vrt.get(), item.prt.get(),
                };
                for (const auto* table : tables) {
                    if (!table) continue;
                    for (const auto& resource : table->resources) {
                        auto found = g_rtt.find(resource.gpu_addr);
                        if (found == g_rtt.end() || !found->second.dcc_metadata_dirty ||
                            !resource.compression_enabled ||
                            resource.metadata_addr != found->second.dcc_metadata_addr ||
                            resource.width != found->second.w ||
                            resource.height != found->second.h)
                            continue;
                        const uint64_t metadata_bytes =
                            prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
                        if (!metadata_bytes || metadata_bytes > SIZE_MAX) continue;
                        std::vector<uint8_t> metadata(static_cast<size_t>(metadata_bytes));
                        size_t copied = 0;
                        if (resource.dcc_metadata_host_data) {
                            copied = static_cast<size_t>(std::min<uint64_t>(
                                metadata.size(), resource.dcc_metadata_host_data_size));
                            std::memcpy(metadata.data(), resource.dcc_metadata_host_data, copied);
                        } else {
                            copied = safe_copy(metadata.data(), resource.metadata_addr,
                                               metadata.size());
                        }
                        const uint64_t texels = static_cast<uint64_t>(found->second.w) *
                                                found->second.h;
                        const VkFormat format = prosper::test::backend_color_format(
                            found->second.format);
                        const uint32_t bpp =
                            prosper::test::backend_color_bytes_per_pixel(format);
                        if (copied != metadata.size() || !bpp || texels > SIZE_MAX / bpp)
                            continue;
                        uint8_t clear_rgba[4]{};
                        if (!prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                                clear_rgba, 1, metadata.data(), metadata.size(),
                            resource.num_components, resource.alpha_is_on_msb))
                            continue;
                        if (format != VK_FORMAT_R8G8B8A8_UNORM &&
                            format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                            format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                            continue;
                        }
                        rtt_remat_note_reset("dcc-fast-clear", found->first);
                        found->second.rgba.reset();
                        found->second.has_uniform_color = true;
                        for (uint32_t channel = 0; channel < 4; ++channel)
                            found->second.uniform_color[channel] =
                                clear_rgba[channel] ? 1.0f : 0.0f;
                        // PROSPER_DCCLOG=1 -- diagnostic only, no behaviour change. A surface
                        // materialised from a DCC fast-clear code becomes a UNIFORM colour for the
                        // whole target, so if this decode is wrong the entire frame is one wrong
                        // colour with no content -- which is exactly Little Nightmares III's
                        // uniform-yellow presents (#2014). Deduped per (address, decoded colour) so a
                        // run costs a handful of lines.
                        if (const char* dcclog = PROSPER_ENV_VALUE("PROSPER_DCCLOG")) {
                            if (dcclog[0] == '1' && dcclog[1] == '\0') {
                                static std::mutex dcc_mutex;
                                static std::set<std::pair<uint64_t, uint32_t>> dcc_seen;
                                const uint32_t packed = (uint32_t)clear_rgba[0] |
                                    ((uint32_t)clear_rgba[1] << 8) |
                                    ((uint32_t)clear_rgba[2] << 16) |
                                    ((uint32_t)clear_rgba[3] << 24);
                                bool first = false;
                                {
                                    std::lock_guard<std::mutex> lock(dcc_mutex);
                                    first = dcc_seen.emplace((uint64_t)found->first, packed).second;
                                }
                                if (first)
                                    fprintf(stderr,
                                            "[dcclog] addr=0x%llx %ux%u fmt=%d ncomp=%u "
                                            "alpha_msb=%d clear_rgba=(%u,%u,%u,%u) -> uniform=(%.0f,%.0f,%.0f,%.0f)\n",
                                            (unsigned long long)found->first,
                                            found->second.w, found->second.h, (int)format,
                                            resource.num_components, (int)resource.alpha_is_on_msb,
                                            clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3],
                                            found->second.uniform_color[0], found->second.uniform_color[1],
                                            found->second.uniform_color[2], found->second.uniform_color[3]);
                            }
                        }
                        found->second.dcc_metadata_dirty = false;
                        ++dcc_materialize_surfaces;
                        dcc_materialize_bytes += sizeof(found->second.uniform_color);
                    }
                }
            }
            if (timing_enabled) {
                pending_timing.dcc_materialize_ms +=
                    std::chrono::duration<double, std::milli>(
                        RenderClock::now() - dcc_materialize_start).count();
                pending_timing.dcc_materialize_surfaces += dcc_materialize_surfaces;
                pending_timing.dcc_materialize_bytes += dcc_materialize_bytes;
            }
            auto safe_equal = [](const uint8_t* expected, uint64_t a, size_t n,
                                 size_t& compared) -> bool {
                const size_t PG = 0x10000;
#ifdef _WIN32
                if (prosper_try_commit_dmem(a, n, 0)) {
                    compared = n;
                    return std::memcmp(expected, (const void*)(uintptr_t)a, n) == 0;
                }
#endif
                compared = 0;
                while (compared < n) {
                    const uint64_t cur = a + compared;
                    if (cur < 0x1000 || prosper_reserved_range_state(cur) == 0) return true;
                    const size_t chunk = std::min(
                        n - compared, PG - static_cast<size_t>(cur & (PG - 1)));
                    if (std::memcmp(expected + compared,
                                    reinterpret_cast<const void*>(static_cast<uintptr_t>(cur)),
                                    chunk)) return false;
                    compared += chunk;
                }
                return true;
            };
            // Keep decoded texture storage alive across callbacks. The old clear()+emplace(size, 0)
            // released and zero-filled tens of MiB every submit even though the decode paths overwrite
            // all pixels. Reusing same-sized slots avoids both costs; short guest reads explicitly clear
            // their uncovered tail below so stale scratch bytes can never become sampled pixels.
            static thread_local std::vector<std::vector<uint8_t>> texstore;
            // Parallel to `texstore`: a slot is pinned while a retained submit-scoped identity entry
            // still points into it. Handing a pinned slot to a later span's decode would overwrite the
            // very bytes that entry promises, so the allocator below skips them and the pins are
            // released together when the submit's identity map is rebuilt.
            static thread_local std::vector<bool> texstore_pinned;
            size_t texstore_used = 0;
            auto acquire_texstore_slot = [&]() -> size_t {
                while (texstore_used < texstore.size() && texstore_pinned[texstore_used])
                    ++texstore_used;
                if (texstore_used == texstore.size()) {
                    texstore.emplace_back();
                    texstore_pinned.push_back(false);
                }
                return texstore_used++;
            };
            // PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE=1 restores the pre-#1691 span-scoped lifetime on
            // the same build, so a routed A/B measures the change and not two compilers' luck.
            //
            // PROSPER_RESOURCE_HASH_DIM does the same, for a different reason: it correlates each
            // decode's raw and sampled hashes with that range's writers, and it reports from inside
            // the decode path. Retaining an identity across a span boundary would silently drop
            // correlation points a previous build emitted, which would make the instrument disagree
            // with itself across builds while investigating exactly the kind of question it exists
            // for. Keep its output identical to pre-#1691 rather than make it cheaper.
            static const bool submit_decode_scope_disabled =
                PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE") != nullptr ||
                PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") != nullptr;
            static thread_local std::unordered_map<TextureDecodeKey, DecodedTexture,
                                                   TextureDecodeKeyHash> decoded_textures;
            static thread_local uint64_t decode_span_ordinal = 0;
            ++decode_span_ordinal;
            // `phase.first_span` is the normal submit boundary, but it is not the only one: the
            // warmup/window gates above (PROSPER_RENDER_FIRST, PROSPER_RENDER_DELAY_MS,
            // PROSPER_RENDER_LAST) return before this point, so a submit can reach here first at a
            // later span. Tracking the submit ordinal as well keeps the rebuild exact under those
            // recipes — without it the pins and the LRU generation would sit frozen across the whole
            // warmup. Retained entries were never unsafe there (a foreign submit serial makes the
            // journal query Unknown, which refuses reuse before dereferencing anything), but stale
            // state that outlives its submit is not something to leave resting on that.
            static thread_local int decode_scope_submit = -1;
            const bool rebuild_decode_scope = phase.first_span || submit_decode_scope_disabled ||
                g_this_submit != decode_scope_submit;
            decode_scope_submit = g_this_submit;
            const bool use_direct_buffer_views =
                PROSPER_ENV_VALUE("PROSPER_NO_FRONTEND_BUFFER_VIEW") == nullptr;
            static std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>
                persistent_decoded_textures;
            static size_t persistent_decoded_texture_bytes = 0;
            static uint64_t persistent_decode_generation = 0;
            // The persistent cache's LRU generation now advances per SUBMIT rather than per span. That
            // is what keeps a submit-scoped identity entry safe: an entry whose `last_use` equals the
            // current generation is skipped by the eviction scan, so persistent-cache storage a
            // retained pointer refers to cannot be freed under it later in the same submit.
            if (rebuild_decode_scope) ++persistent_decode_generation;
            const uint64_t decode_generation = persistent_decode_generation;
            static thread_local std::vector<std::shared_ptr<const std::vector<uint8_t>>>
                retired_submit_pixels;
            static size_t retired_submit_bytes = 0;
            if (rebuild_decode_scope) {
                decoded_textures.clear();
                texstore_pinned.assign(texstore.size(), false);
                retired_submit_pixels.clear();
                retired_submit_bytes = 0;
            }
            static uint64_t persistent_texture_id = 0;
            static std::vector<uint8_t> persistent_validation_scratch;
            static const size_t persistent_decode_limit = [] {
                const uint64_t physical_bytes = host_physical_memory_bytes();
                const size_t limit = texture_decode_cache_limit_bytes(
                    PROSPER_ENV_VALUE("PROSPER_TEXTURE_DECODE_CACHE_MB"), physical_bytes);
                fprintf(stderr,
                        "[render] decoded texture cache budget = %.1f MiB "
                        "(host physical %.1f GiB)\n",
                        limit / (1024.0 * 1024.0),
                        physical_bytes / (1024.0 * 1024.0 * 1024.0));
                return limit;
            }();
            uint32_t resource_hash_w = 0, resource_hash_h = 0;
            if (const char* dim = PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &resource_hash_w, &resource_hash_h) != 2)
                    resource_hash_w = resource_hash_h = 0;
            uint32_t target_step_w = 0, target_step_h = 0;
            if (const char* dim = PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &target_step_w, &target_step_h) != 2)
                    target_step_w = target_step_h = 0;
            const size_t target_step_min_draws = PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_MIN_DRAWS")
                ? std::max<long>(2, atol(PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_MIN_DRAWS"))) : 2;
            prosper::frontend::RttInjectionCache rtt_injection_cache;
            // Build one draw's set-tagged resources from its VS (set 0) + PS (set 1) tables — read the
            // bytes from 1:1-mapped guest memory, detile textures. (Each constant/vertex buffer + texture
            // gets its own binding; the recompiler declared a storage buffer / image sampler at each.)
            // #2256. `validate_spirv_descriptor_interface` is memoized inside shader_resources.cpp,
            // but its HIT path is O(module size) twice -- an FNV-1a walk of every word to build the
            // key, then a full vector equality compare to confirm it -- under a process-global lock,
            // and build_R calls it twice per draw. Measured on a Blue Prince gameplay submit
            // (PPSA25009, 2,103 draws): 30.37 ms/submit, 53% of build_R and 9.3% of the frame,
            // traversing 27.2 million words. 3,231 calls served 523 distinct shaders, so 84% of that
            // work was re-deriving a key for a module already reflected.
            //
            // The caller already holds a cheap stable key. DrawItem::vs_identity/fs_identity come
            // from the exact shader-recompile cache as `cache.next_identity++`
            // (src/gpu/execute/gpu_executor.cpp:1255) -- a monotonic counter that is NEVER reused, so an
            // identity denotes one module for the life of the process even across cache eviction
            // (an evicted shader recompiles to a NEW identity, which misses here and is refilled).
            //
            // This changes no semantics. The upstream cache already returns its stored report for a
            // matching module regardless of which runtime table was passed, so the report was
            // already a pure function of the SPIR-V; and this call site reads only
            // `report.descriptors` (via find_spirv_descriptor_binding), never `report.issues`.
            //
            // Bounded at the same 4,096 entries the shader cache itself uses. On overflow the map is
            // cleared wholesale rather than evicted by age: there is no LRU bookkeeping to pay for on
            // a path this hot, and 523 distinct shaders per submit leaves ample headroom.
            //
            // The overflow behaviour does NOT degrade gracefully, and `memo_clears` should be read
            // accordingly: a title exceeding the cap would clear and refill repeatedly, paying the
            // full reflection cost again PLUS the map churn. So a non-zero `memo_clears` means this
            // optimisation has STOPPED WORKING, not that a threshold was brushed -- do not read a
            // small non-zero value as benign. Observed 0 across a 13-minute Blue Prince route.
            //
            // The entry copies the whole DescriptorValidationReport although this path reads only
            // `descriptors`. Deliberate: storing the subset would bound the footprint to what is
            // used, but it would also make the entry a lie about what it holds the moment a caller
            // reads `issues`, and at 523 entries against a 4,096 cap the footprint is not the
            // constraint. If that ever inverts, narrow the entry rather than raising the cap.
            struct ReflectMemoEntry {
                prosper::gpu::DescriptorValidationReport report;
                uint32_t set = 0;
                prosper::gpu::SpirvShaderStage stage = prosper::gpu::SpirvShaderStage::Unknown;
            };
            static thread_local std::unordered_map<uint64_t, ReflectMemoEntry> reflect_memo;
            constexpr size_t kReflectMemoMaxEntries = 4096;
            auto build_R = [&](const prosper::gpu::DrawItem& draw,
                               const prosper::gpu::ShaderResourceTable* vrt,
                               const prosper::gpu::ShaderResourceTable* prt) {
              std::vector<prosper::test::FrameResource> R;
              auto add = [&](const prosper::gpu::ShaderResourceTable* t, uint32_t set,
                             const std::vector<uint32_t>& spirv,
                             prosper::gpu::SpirvShaderStage stage,
                             uint64_t shader_identity){
                if (!t) return;
                // #2256. This call is memoized inside shader_resources.cpp, but the memo's HIT path
                // is O(module size) three times over -- an FNV-1a walk to build the key, a full
                // vector equality compare to confirm it, and a copy of the report -- under a
                // process-global lock, twice per draw. Time it as its own leaf of build_R rather
                // than guessing: build_R's residual is the largest unexplained bucket in the
                // renderer (+6.16 of 8.02 ms/submit on RADV, #2250) and this is the biggest fixed
                // per-draw cost inside it.
                //
                // `words` is the falsifiable magnitude: it is exactly what the hash and the memcmp
                // each traverse, so a reader can check the ms against it instead of taking it on
                // faith. `distinct` is the memo's CEILING and is deliberately UNCAPPED -- the last
                // attempt on this path (#2239) died because its justifying probe hit a 4,096-entry
                // cap on precisely this count, and a truncated ceiling reads as a small one.
                const prosper::gpu::DescriptorValidationReport* memoized = nullptr;
                if (shader_identity) {
                    const auto found = reflect_memo.find(shader_identity);
                    // set/stage are checked rather than assumed. They are fixed per module on this
                    // path today (VS -> set 0, PS -> set 1), so a mismatch cannot occur -- but the
                    // key is the identity alone, and a silent wrong answer here would be a wrong
                    // descriptor contract rather than a slow one. A mismatch falls through to the
                    // real call and is counted.
                    if (found != reflect_memo.end() &&
                        found->second.set == set && found->second.stage == stage)
                        memoized = &found->second.report;
                    else if (found != reflect_memo.end()) {
                        ++pending_timing.build_reflect_memo_mismatch;
                        // Loud and unconditional on first occurrence, NOT gated on timing: this
                        // path is unreachable by the argument above, and if the argument is wrong
                        // the consequence is a wrong descriptor contract rather than a slow frame.
                        // A counter only a timing run prints would hide exactly the case that
                        // matters. The fallback is correct either way -- it re-reflects -- so this
                        // reports rather than fails.
                        static thread_local bool warned = false;
                        if (!warned) {
                            warned = true;
                            fprintf(stderr,
                                    "[render] *** shader identity %llu was reflected for set=%u "
                                    "stage=%u and is now asked for set=%u stage=%u -- identity is "
                                    "not stage-unique after all (#2256). Re-reflecting; the "
                                    "descriptor contract is unaffected%s",
                                    (unsigned long long)shader_identity, found->second.set,
                                    (unsigned)found->second.stage, set, (unsigned)stage, "\n");
                        }
                    }
                }
                if (memoized && timing_enabled) ++pending_timing.build_reflect_memo_hits;
                const auto rt0 = (timing_enabled && !memoized)
                    ? RenderClock::now() : RenderClock::time_point{};
                prosper::gpu::DescriptorValidationReport computed;
                if (!memoized)
                    computed = prosper::gpu::validate_spirv_descriptor_interface(
                        spirv, t, set, stage, false);
                if (!memoized && timing_enabled) {
                    pending_timing.build_reflect_ms +=
                        std::chrono::duration<double, std::milli>(
                            RenderClock::now() - rt0).count();
                    ++pending_timing.build_reflect_calls;
                    pending_timing.build_reflect_words += spirv.size();
                    // Identity 0 means the module came from an external/replay path, so no memo
                    // keyed on identity could serve it. Counted separately: folding those into
                    // `distinct` would inflate the ceiling, folding them into the hit population
                    // would inflate the saving. They are neither.
                    if (shader_identity)
                        pending_timing.build_reflect_identities.insert(shader_identity);
                    else
                        ++pending_timing.build_reflect_unidentified;
                }
                if (!memoized && shader_identity) {
                    if (reflect_memo.size() >= kReflectMemoMaxEntries) {
                        reflect_memo.clear();
                        if (timing_enabled) ++pending_timing.build_reflect_memo_clears;
                    }
                    ReflectMemoEntry entry;
                    entry.report = computed;
                    entry.set = set;
                    entry.stage = stage;
                    memoized = &reflect_memo.emplace(shader_identity, std::move(entry))
                                   .first->second.report;
                }
                // Bound to the memo entry when there is one and to the local otherwise. Both outlive
                // the per-resource loop below, and the loop does not touch the map.
                const prosper::gpu::DescriptorValidationReport& reflected =
                    memoized ? *memoized : computed;
                for (auto& r : t->resources) {
                    const prosper::gpu::SpirvDescriptorBinding* reflected_binding =
                        prosper::gpu::find_spirv_descriptor_binding(
                            reflected, set, r.binding);
                    // The front half deliberately retains every descriptor candidate it can prove
                    // while folding guest code. The final recompiler can use only a subset (for
                    // example, pc-specific scalar loads supersede the original broad V#). Extra
                    // runtime bindings are harmless to validation, but materializing one absurd
                    // unused declaration can allocate and zero the 64 MiB safety ceiling per draw.
                    // The generated SPIR-V is authoritative about which bindings the backend needs.
                    if (!reflected_binding) continue;
                    const auto resource_timing_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    bool resource_rtt_hit = false;
                    bool resource_compute_image_hit = false;
                    bool resource_compute_image_candidate = false;
                    bool resource_compute_depth_hybrid = false;
                    uint64_t resource_compute_producer_order = 0;
                    uint32_t resource_compute_depth_overlay_mask = 0;
                    bool resource_local_reuse = false;
                    bool resource_persistent_hit = false;
                    bool resource_persistent_submit_reuse = false;
                    bool resource_persistent_miss = false;
                    bool resource_persistent_invalidation = false;
                    bool resource_buffer_view = false;
                    double resource_buffer_probe_ms = 0.0;
                    double resource_buffer_copy_ms = 0.0;
                    double resource_texture_validation_ms = 0.0;
                    size_t resource_texture_validated_bytes = 0;
                    size_t resource_texture_source_bytes = 0;
                    int resource_texture_submit_query = -1;
                    int resource_texture_watch_query = -1;
                    uint32_t resource_texture_watch_stability = 0;
                    bool resource_texture_exact_validation = false;
                    bool resource_texture_watch_active = false;
                    bool resource_texture_watch_disabled = false;
                    bool resource_texture_watch_only = false;
                    bool resource_has_live_rtt = false;
                    bool resource_has_ds_live = false;
                    bool resource_persistent_candidate = false;
                    size_t resource_persistent_source_size = 0;
                    prosper::test::FrameResource fr; fr.binding = r.binding; fr.set = set;
                    fr.is_storage_image = r.cls == RC::StorageImage;
                    fr.storage_image_numeric_class = reflected_binding->image_numeric_class;
                    fr.storage_image_contract_valid = !fr.is_storage_image ||
                        reflected_binding->image_numeric_class !=
                            prosper::gpu::SpirvImageNumericClass::Unknown;
                    const bool normalized_sampling =
                        reflected_binding->kind ==
                            prosper::gpu::SpirvDescriptorKind::CombinedImageSampler &&
                        reflected_binding->normalized_sampling &&
                        !reflected_binding->texel_access;
                    const bool writable_storage_image =
                        reflected_binding->kind ==
                            prosper::gpu::SpirvDescriptorKind::StorageImage &&
                        reflected_binding->writable;
                    // Captured replays preserve the logical guest address for RT identity but provide
                    // owned bytes here. Production resources leave host_data null and use guest memory.
                    auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) -> size_t {
                        if (!r.host_data) return safe_copy(dst, addr, n);
                        if (addr < r.gpu_addr) return 0;
                        uint64_t off = addr - r.gpu_addr;
                        if (off >= r.host_data_size) return 0;
                        size_t take = static_cast<size_t>(std::min<uint64_t>(n, r.host_data_size - off));
                        std::memcpy(dst, r.host_data + off, take);
                        return take;
                    };
                    // The same bytes `copy_resource` would deliver, read IN PLACE, or nullptr when it
                    // would deliver fewer than `n` of them.
                    //
                    // Guest memory is 1:1 mapped, so a full-surface staging copy buys nothing except a
                    // second address for the same bytes -- and on a 4K HDR intermediate that second
                    // address is 64+ MiB of freshly mmap'd, page-faulted, memset and then munmap'd
                    // anonymous memory per reference. Stray's title screen paid it twice a frame
                    // (#3149's `__memmove_avx512` at 14.5% self, and its kernel page-management
                    // frames, are this).
                    //
                    // Returning nullptr on a SHORT range is the load-bearing half: the decode branches
                    // below rely on `copy_resource`'s partial return to fall back to a linear read or
                    // to a zero-filled tail, so a sparse or over-declared surface must keep taking the
                    // staged path rather than being handed a pointer into a hole.
                    // PROSPER_NO_DIRECT_TEXTURE_SOURCE restores the staged copy on BOTH source kinds.
                    // A lever that disabled only the guest-memory half would still be named after the
                    // whole optimisation, and an A/B under it would attribute the remaining half's
                    // cost to whatever else moved -- the instrument-trap shape this tree records.
                    static const bool no_direct_texture_source =
                        PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_TEXTURE_SOURCE") != nullptr;
                    auto direct_resource_source = [&](uint64_t addr, size_t n) -> const uint8_t* {
                        if (!n || no_direct_texture_source) return nullptr;
                        if (r.host_data) {
                            // Mirrors copy_resource's host_data arm: the same bounds, and `take == n`.
                            if (addr < r.gpu_addr) return nullptr;
                            const uint64_t off = addr - r.gpu_addr;
                            if (off >= r.host_data_size || n > r.host_data_size - off) return nullptr;
                            return r.host_data + off;
                        }
                        if (addr < 0x1000 || addr > UINT64_MAX - n) return nullptr;
                        return safe_span(addr, n) == n
                            ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr))
                            : nullptr;
                    };
                    // The tiled bytes a detiler will read, without staging a copy of them when the
                    // range is fully mapped. `lease` must outlive the returned pointer; it stays
                    // empty on the direct path and holds the staging buffer otherwise, so an early
                    // exit from a decode branch returns it to the pool either way.
                    //
                    // `got` reports what the STAGED path would have delivered, exactly as
                    // `copy_resource` did, so every downstream `got < ...` fallback keeps its
                    // meaning. On the direct path `got == n` by construction: the pointer is only
                    // handed out when the whole range is readable.
                    auto stage_tiled_source =
                        [&](prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr,
                            size_t n, size_t& got) -> const uint8_t* {
                        if (const uint8_t* direct = direct_resource_source(addr, n)) {
                            got = n;
                            return direct;
                        }
                        lease = prosper::frontend::decode_scratch_pool().take(n);
                        got = copy_resource(lease.data(), addr, n);
                        lease.zero_tail(got);   // restores what `std::vector<uint8_t>(n, 0)` gave
                        return lease.data();
                    };
                    // Avoid allocating and copying storage buffers that are already present in stable,
                    // readable unified guest memory. The callback and backend upload are synchronous;
                    // compute/DMA boundaries split graphics spans before this point. The submit-scoped
                    // readability guard proves the complete range; an invalid tail retains the copied
                    // fallback.
                    auto direct_resource = [&](uint64_t addr, size_t n) -> const uint8_t* {
                        if (!n || addr < 0x1000 || addr > UINT64_MAX - n ||
                            (addr & (alignof(uint32_t) - 1))) return nullptr;
                        if (r.host_data) {
                            if (addr < r.gpu_addr) return nullptr;
                            const uint64_t off = addr - r.gpu_addr;
                            if (off > r.host_data_size || n > r.host_data_size - off) return nullptr;
                            const uint8_t* source = r.host_data + off;
                            return (reinterpret_cast<uintptr_t>(source) &
                                    (alignof(uint32_t) - 1)) ? nullptr : source;
                        }
                        // Reuse the executor's submit-scoped range cache; on Windows the same guard
                        // also materializes sparse direct-memory pages when necessary.
                        if (n > UINT32_MAX ||
                            !prosper::gpu::guest_readable(addr, static_cast<uint32_t>(n)))
                            return nullptr;
                        return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr));
                    };
                    auto copy_dcc_metadata = [&](uint8_t* dst, size_t n) -> size_t {
                        if (!r.dcc_metadata_host_data)
                            return safe_copy(dst, r.metadata_addr, n);
                        const size_t take = static_cast<size_t>(std::min<uint64_t>(
                            n, r.dcc_metadata_host_data_size));
                        std::memcpy(dst, r.dcc_metadata_host_data, take);
                        return take;
                    };
                    if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                        uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                        // AvPlayer exposes NV12 as an R8 luma plane followed by an RG8 UV plane.
                        // Which resource IS that chroma plane — and why a candidate was rejected —
                        // is decided by avplayer_plane_policy.hpp, which carries the reasoning and
                        // has its own unit test. A false verdict here is silent: the plane falls
                        // into the legacy narrow coverage broadcast, the shader's V becomes its U,
                        // and the movie keeps correct luma, detail and geometry while every colour
                        // collapses onto one green<->magenta axis. (#2005)
                        const AvpChromaVerdict avplayer_chroma_verdict =
                            classify_avplayer_chroma_plane(r, tw, th, t->resources);
                        const bool avplayer_chroma_layout = avplayer_chroma_verdict.match;
                        // Deliberately NOT filtered by img_dim: a plane the classifier rejects on
                        // exactly that field has to appear in this log, or the log cannot report
                        // the rejection that matters most.
                        if (avp_chroma_log() && r.cls == RC::Texture &&
                            r.format == prosper::gpu::DataFormat::Unorm8 && r.num_components <= 2)
                            log_avp_chroma_candidate(r, tw, th, avplayer_chroma_verdict);
                        // PROSPER_AVPCHROMA_DUMP=<dir> — write each candidate plane's exact guest
                        // source bytes once, at the resolved row pitch. Converting the two dumped
                        // planes on the CPU separates a decode/staging defect (planes wrong) from a
                        // sampling/recognition defect (planes right, picture wrong), which is the
                        // only way to tell a chroma cast's two very different causes apart.
                        if (const char* avp_dump_dir = PROSPER_ENV_VALUE("PROSPER_AVPCHROMA_DUMP")) {
                            if (r.cls == RC::Texture &&
                                r.format == prosper::gpu::DataFormat::Unorm8 &&
                                r.num_components >= 1 && r.num_components <= 2 &&
                                (r.img_dim == 1u || r.img_dim == 5u) &&
                                r.tile_mode == 0u && tw && th) {
                                static std::mutex dump_mx;
                                static std::map<uint64_t, uint64_t> sightings;
                                static uint64_t dump_count = 0;
                                // PROSPER_AVPCHROMA_DUMP_EVERY samples one in N sightings of each
                                // plane allocation (default: only the first). A movie opens on a
                                // fade from black, so the first sighting is a useless all-dark
                                // plane; sampling a series lands somewhere mid-shot without having
                                // to guess a frame index. Capped so a long run cannot fill a disk.
                                const char* every_env = PROSPER_ENV_VALUE("PROSPER_AVPCHROMA_DUMP_EVERY");
                                const uint64_t every = every_env
                                    ? std::max<uint64_t>(1, strtoull(every_env, nullptr, 0)) : 0;
                                uint64_t sighting = 0;
                                bool first = false;
                                {
                                    std::lock_guard<std::mutex> lock(dump_mx);
                                    sighting = ++sightings[r.gpu_addr ^
                                                           (uint64_t{r.num_components} << 56)];
                                    first = every ? (sighting % every == 0 && dump_count < 40)
                                                  : sighting == 1;
                                    if (first) ++dump_count;
                                }
                                if (first) {
                                    const uint32_t bpt = r.num_components;
                                    const uint32_t tight = tw * bpt;
                                    uint32_t pitch = r.linear_row_pitch_bytes;
                                    if (!pitch)
                                        pitch = prosper::gpu::guest_linear_texture_row_pitch(
                                            r.gpu_addr, tight);
                                    if (!pitch)
                                        pitch = static_cast<uint32_t>(
                                            prosper::gpu::linear_sampled_row_pitch(tw, bpt));
                                    std::vector<uint8_t> plane(
                                        static_cast<size_t>(pitch) * th, 0);
                                    const size_t got = copy_resource(plane.data(), r.gpu_addr,
                                                                     plane.size());
                                    char name[512];
                                    snprintf(name, sizeof name,
                                             "%s/avpplane_%ux%u_c%u_p%u_%llx_s%06llu.bin",
                                             avp_dump_dir, tw, th, bpt, pitch,
                                             (unsigned long long)r.gpu_addr,
                                             (unsigned long long)sighting);
                                    if (FILE* f = fopen(name, "wb")) {
                                        fwrite(plane.data(), 1, plane.size(), f);
                                        fclose(f);
                                        fprintf(stderr,
                                                "[avpchroma] dumped %zu/%zu bytes -> %s\n",
                                                got, plane.size(), name);
                                        fflush(stderr);
                                    }
                                }
                            }
                        }
                        const uint64_t sampled_source_addr = texture_decode_source_address(
                            r.gpu_addr, r.img_dim, r.in_mip_tail,
                            r.layer_mip_offset_bytes);
                        const bool sampled_2d_view = r.img_dim == 1u || r.img_dim == 5u;
                        const size_t msaa_tiled_source_span = r.img_dim == 6u
                            ? prosper::gpu::tiled_msaa_surface_bytes(
                                  tw, th, r.tile_mode, 4u, r.sample_count)
                            : 0u;
                        TextureDecodeKey decode_key{
                            r.gpu_addr, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r.host_data)),
                            r.host_data_size, r.size, msaa_tiled_source_span,
                            static_cast<uint32_t>(r.cls),
                            static_cast<uint32_t>(r.format), r.num_components, tw, th, r.depth,
                            r.sample_count,
                            r.tile_mode, r.linear_row_pitch_bytes, r.img_dim,
                            r.mip_tail_bytes, r.mip_tail_x, r.mip_tail_y,
                            r.layer_stride_bytes, r.layer_mip_offset_bytes,
                            r.compression_enabled ? r.max_uncompressed_block_size : 0u,
                            r.compression_enabled ? r.max_compressed_block_size : 0u,
                            r.compression_enabled
                                ? ((r.meta_pipe_aligned ? 1u : 0u) |
                                   (r.write_compress_enabled ? 2u : 0u) | 4u |
                                   (r.alpha_is_on_msb ? 8u : 0u) |
                                   (r.color_transform ? 16u : 0u))
                                : 0u,
                            r.compression_enabled ? r.metadata_addr : 0u,
                            r.compression_enabled
                                ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                                      r.dcc_metadata_host_data))
                                : 0u,
                            r.compression_enabled ? r.dcc_metadata_host_data_size : 0u,
                            r.in_mip_tail,
                            avplayer_chroma_layout,
                            static_cast<uint32_t>(reflected_binding->image_numeric_class),
                            reflected_binding->storage_image_format,
                            reflected_binding->image_dim,
                            reflected_binding->image_arrayed,
                            reflected_binding->image_multisampled,
                            reflected_binding->writable,
                        };
                        if (r.img_dim == 6u) {
                            // GFX10 TYPE=2D_MSAA interleaves the sample coordinate into the tiled
                            // address. It is neither an ordinary 2D texture nor a native Vulkan
                            // multisample upload: IMAGE_LOAD names one sample explicitly, so the
                            // recompiler exposes the samples as four layers of a single-sample 2D
                            // array. Materialize exactly that representation here. Keep the first
                            // live contract deliberately as narrow as the observed Asterix surface;
                            // unsupported sample counts/layouts/formats must remain fail-visible.
                            const bool reflected_msaa_fetch =
                                reflected_binding->kind ==
                                    prosper::gpu::SpirvDescriptorKind::CombinedImageSampler &&
                                reflected_binding->image_dim == 1u &&
                                reflected_binding->image_arrayed &&
                                !reflected_binding->image_multisampled &&
                                reflected_binding->texel_access &&
                                !reflected_binding->normalized_sampling;
                            const bool exact_msaa_shape = sampled_msaa_fetch_shape_supported(
                                r, fr.is_storage_image, reflected_msaa_fetch);

                            prosper::gpu::Gfx10HtileMsaaSource htile_source =
                                r.compression_enabled
                                    ? prosper::gpu::Gfx10HtileMsaaSource::Unsupported
                                    : prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase;
                            uint32_t decompressed_htile_value = 0;
                            std::vector<uint8_t> htile_metadata;
                            size_t htile_metadata_got = 0;
                            if (exact_msaa_shape && r.compression_enabled) {
                                const uint64_t metadata_size =
                                    prosper::gpu::gpu_capture_dcc_metadata_footprint(r);
                                htile_metadata.resize(
                                    static_cast<size_t>(metadata_size), 0);
                                htile_metadata_got = metadata_size
                                    ? copy_dcc_metadata(htile_metadata.data(),
                                                        htile_metadata.size())
                                    : 0u;
                                // Mode-24 MSAA metadata is HTILE, not color DCC. The two exact PAL
                                // initialization dwords disable depth compression for depth-only and
                                // depth+stencil surfaces respectively. The metadata itself must prove
                                // one uniform state across the complete AddrLib-sized plane; every
                                // compressed, nonuniform, ambiguous, or short state remains rejected.
                                htile_source = prosper::gpu::gfx10_htile_msaa_source(
                                    htile_metadata.data(), htile_metadata_got,
                                    tw, th, r.tile_mode, 4u, r.sample_count,
                                    r.meta_pipe_aligned);
                                if (htile_source ==
                                    prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase) {
                                    prosper::gpu::gfx10_htile_metadata_is_decompressed(
                                        htile_metadata.data(), htile_metadata_got,
                                        htile_metadata.size(), &decompressed_htile_value);
                                }
                            }

                            const bool uncompressed_base = htile_source ==
                                prosper::gpu::Gfx10HtileMsaaSource::UncompressedBase;
                            const bool depth_zero_fast_clear = htile_source ==
                                prosper::gpu::Gfx10HtileMsaaSource::DepthZeroFastClear;
                            const bool materializable_msaa = exact_msaa_shape &&
                                (uncompressed_base || depth_zero_fast_clear);
                            const size_t tiled_bytes = exact_msaa_shape && uncompressed_base
                                ? msaa_tiled_source_span : 0u;
                            const uint64_t linear_bytes64 =
                                static_cast<uint64_t>(tw) * th * r.sample_count * 4u;
                            if (!materializable_msaa || !linear_bytes64 ||
                                linear_bytes64 > SIZE_MAX) {
                                static uint32_t rejected = 0;
                                const uint32_t ordinal = ++rejected;
                                if (ordinal <= 32u) {
                                    uint32_t htile_first = 0;
                                    uint32_t htile_first_different = 0;
                                    size_t htile_dwords_different = 0;
                                    if (htile_metadata_got == htile_metadata.size() &&
                                        htile_metadata.size() >= sizeof(uint32_t) &&
                                        (htile_metadata.size() % sizeof(uint32_t)) == 0u) {
                                        std::memcpy(&htile_first, htile_metadata.data(),
                                                    sizeof(htile_first));
                                        for (size_t offset = sizeof(uint32_t);
                                             offset < htile_metadata.size();
                                             offset += sizeof(uint32_t)) {
                                            uint32_t value = 0;
                                            std::memcpy(&value, htile_metadata.data() + offset,
                                                        sizeof(value));
                                            if (value != htile_first) {
                                                if (!htile_dwords_different)
                                                    htile_first_different = value;
                                                ++htile_dwords_different;
                                            }
                                        }
                                    }
                                    fprintf(stderr,
                                            "[render-msaa-reject] ordinal=%u set=%u binding=%u "
                                            "addr=0x%llx "
                                            "%ux%u samples=%u fmt=%u/%u tile=%u compression=%d "
                                            "shape=%d reflected=%d base-uncompressed=%d "
                                            "depth-zero-fast-clear=%d "
                                            "htile=%zu/%zu first=0x%08x first-different=0x%08x "
                                            "dwords-different=%zu\n",
                                            ordinal, set, r.binding,
                                            (unsigned long long)r.gpu_addr,
                                            tw, th, r.sample_count, (unsigned)r.format,
                                            r.num_components, r.tile_mode,
                                            static_cast<int>(r.compression_enabled),
                                            static_cast<int>(exact_msaa_shape),
                                            static_cast<int>(reflected_msaa_fetch),
                                            static_cast<int>(uncompressed_base),
                                            static_cast<int>(depth_zero_fast_clear),
                                            htile_metadata_got, htile_metadata.size(), htile_first,
                                            htile_first_different, htile_dwords_different);
                                }
                                continue;
                            }

                            // Captured backing and DCC metadata cannot use the one-range journal
                            // proof across callback spans. They still reuse safely inside the span
                            // that decoded them. Plain guest backing may cross spans only while the
                            // ordered write journal proves the complete padded tiled allocation was
                            // untouched.
                            const uint64_t cross_span_source_size =
                                !r.host_data && !r.compression_enabled ? tiled_bytes : 0u;
                            auto reused = decoded_textures.find(decode_key);
                            if (reused != decoded_textures.end()) {
                                const bool same_span =
                                    reused->second.span == decode_span_ordinal;
                                const prosper::gpu::GuestGpuWriteQuery journal_query = same_span
                                    ? prosper::gpu::GuestGpuWriteQuery::Unknown
                                    : prosper::gpu::guest_gpu_writes_since(
                                          reused->second.snapshot,
                                          reused->second.source_addr,
                                          reused->second.source_size);
                                if (!submit_local_texture_decode_reusable(
                                        reused->second.span, decode_span_ordinal,
                                        reused->second.source_addr, reused->second.source_size,
                                        sampled_source_addr, cross_span_source_size,
                                        journal_query) || !reused->second.pixels_owner) {
                                    decoded_textures.erase(reused);
                                    reused = decoded_textures.end();
                                    ++g_texture_decode_scope.invalidations;
                                }
                            }

                            if (reused != decoded_textures.end()) {
                                fr.tex_rgba_owner = reused->second.pixels_owner;
                                fr.tex_rgba = reused->second.pixels;
                                resource_local_reuse = true;
                                if (reused->second.span == decode_span_ordinal) {
                                    ++g_texture_decode_scope.same_span_reuses;
                                } else {
                                    ++g_texture_decode_scope.cross_span_reuses;
                                    reused->second.snapshot =
                                        prosper::gpu::guest_gpu_write_snapshot();
                                }
                                if (timing_enabled) pending_timing.texture_reuses++;
                            } else {
                                std::vector<uint8_t> tiled;
                                size_t copied = 0;
                                if (uncompressed_base) {
                                    tiled.resize(tiled_bytes, 0);
                                    copied = copy_resource(
                                        tiled.data(), sampled_source_addr, tiled.size());
                                }
                                std::vector<uint8_t> linear(
                                    static_cast<size_t>(linear_bytes64), 0);
                                prosper::gpu::Gfx10HtileMsaaSource realized_source =
                                    prosper::gpu::Gfx10HtileMsaaSource::Unsupported;
                                const bool materialized = r.compression_enabled
                                    ? prosper::gpu::materialize_gfx10_htile_msaa_surface(
                                          linear.data(), linear.size(),
                                          tiled.empty() ? nullptr : tiled.data(), copied,
                                          htile_metadata.data(), htile_metadata_got,
                                          tw, th, r.tile_mode, 4u, r.sample_count,
                                          r.meta_pipe_aligned, &realized_source)
                                    : (copied == tiled.size() &&
                                       prosper::gpu::detile_msaa_surface(
                                           linear.data(), tiled.data(), copied, tw, th,
                                           r.tile_mode, 4u, r.sample_count));
                                if (!materialized ||
                                    (r.compression_enabled && realized_source != htile_source)) {
                                    static uint32_t short_sources = 0;
                                    if (short_sources++ < 32u)
                                        fprintf(stderr,
                                                "[render-msaa-reject] set=%u binding=%u addr=0x%llx "
                                                "source=%zu/%zu bytes\n",
                                                set, r.binding, (unsigned long long)r.gpu_addr,
                                                copied, tiled.size());
                                    continue;
                                }

                                fr.tex_rgba_owner =
                                    std::make_shared<const std::vector<uint8_t>>(std::move(linear));
                                fr.tex_rgba = fr.tex_rgba_owner->data();
                                ++g_texture_decode_scope.decodes;
                                DecodedTexture decoded;
                                decoded.pixels = fr.tex_rgba;
                                decoded.output_height = th;
                                decoded.span = decode_span_ordinal;
                                decoded.snapshot = prosper::gpu::guest_gpu_write_snapshot();
                                decoded.source_addr = sampled_source_addr;
                                decoded.source_size = cross_span_source_size;
                                decoded.pixels_owner = fr.tex_rgba_owner;
                                decoded.texture_format = VK_FORMAT_R32_SFLOAT;
                                decoded_textures.emplace(decode_key, std::move(decoded));
                            }
                            fr.tex_byte_size = fr.tex_rgba_owner->size();
                            fr.tw = tw;
                            fr.th = th;
                            fr.td = 1u;
                            fr.img_dim = r.img_dim;
                            fr.sample_count = r.sample_count;
                            fr.declared_mip_levels = 1u;
                            fr.texture_format = VK_FORMAT_R32_SFLOAT;
                            resource_texture_source_bytes = depth_zero_fast_clear
                                ? htile_metadata.size() : tiled_bytes;
                            if (decompressed_htile_value && getenv("PROSPER_GFXLOG")) {
                                static uint32_t logged = 0;
                                if (logged++ < 16u)
                                    fprintf(stderr,
                                            "[render-msaa] addr=0x%llx %ux%u samples=%u "
                                            "decompressed-htile=0x%08x source=%zu bytes\n",
                                            (unsigned long long)r.gpu_addr, tw, th,
                                            r.sample_count, decompressed_htile_value, tiled_bytes);
                            }
                            if (depth_zero_fast_clear && getenv("PROSPER_GFXLOG")) {
                                static uint32_t logged_zero_clear = 0;
                                const uint32_t ordinal = ++logged_zero_clear;
                                if (ordinal <= 16u)
                                    fprintf(stderr,
                                            "[render-msaa] ordinal=%u addr=0x%llx %ux%u "
                                            "samples=%u depth-zero-fast-clear metadata=%zu bytes\n",
                                            ordinal, (unsigned long long)r.gpu_addr, tw, th,
                                            r.sample_count, htile_metadata.size());
                            }

                            fr.mag_filter = r.mag_filter;
                            fr.min_filter = r.min_filter;
                            fr.mip_filter = r.mip_filter;
                            fr.addr_uvw[0] = r.addr_uvw[0];
                            fr.addr_uvw[1] = r.addr_uvw[1];
                            fr.addr_uvw[2] = r.addr_uvw[2];
                            fr.border_color_type = r.border_color_type;
                            fr.min_lod = r.min_lod;
                            fr.max_lod = r.max_lod;
                            fr.lod_bias = r.lod_bias;
                            fr.max_aniso_ratio = r.max_aniso_ratio;
                            for (int k = 0; k < 4; ++k) fr.swizzle[k] = r.swizzle[k];
                        } else {
                        auto live_rtt = rtt_on ? g_rtt.find(sampled_source_addr) : g_rtt.end();
                        static const uint32_t render_scale = [] {
                            const char* e = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
                            const long v = e ? std::strtol(e, nullptr, 10) : 1;
                            return v > 0 ? static_cast<uint32_t>(v) : 1u;
                        }();
                        // Deferred RTT readback (#1284): a GPU-resident target consumed in a way the
                        // GPU bind below cannot serve (extent mismatch or storage image) materializes
                        // its CPU copy here on demand. Exact 2D attachment feedback is served by the
                        // backend's prior-version GPU snapshot rather than by a synchronous readback.
                        // The producer
                        // ran in an earlier batch — same-batch CPU consumers force the eager readback
                        // at defer time — so the queue-ordered copy reads current pixels.
                        if (live_rtt != g_rtt.end() && live_gpu_targets && r.img_dim != 2u &&
                            !r.in_mip_tail && live_rtt->second.gpu_valid) {
                            RttSurf& surface = live_rtt->second;
                            const bool sampled_extent_compatible =
                                prosper::frontend::rtt_sampled_extent_compatible(
                                    tw, th, surface.w, surface.h, render_scale,
                                    normalized_sampling);
                            // Must use the SAME all-slot rule as the gate it feeds. On colour-0
                            // alone an MRT2+ feedback collision set direct_serves=true, which
                            // suppressed the lazy CPU materialisation; the corrected gate below then
                            // refused the direct image because the collision is real, and the
                            // resource fell through to stale guest bytes with no snapshot to use.
                            const bool direct_serves = prosper::frontend::mrt_direct_serves(
                                draw, sampled_source_addr, tw, th,
                                fr.is_storage_image, r.img_dim,
                                sampled_extent_compatible,
                                prosper::test::find_persistent_color_target(
                                    sampled_source_addr, surface.w, surface.h,
                                    surface.format) != nullptr,
                                mrt_format_defined,
                                /*feedback_copy_supported=*/true);
                            const VkFormat surface_format =
                                prosper::test::backend_color_format(surface.format);
                            const uint32_t surface_bpp =
                                prosper::test::backend_color_bytes_per_pixel(surface_format);
                            const uint64_t surface_texels =
                                static_cast<uint64_t>(surface.w) * surface.h;
                            if (!direct_serves && surface.w && surface.h && surface_bpp &&
                                surface_texels <= UINT64_MAX / surface_bpp &&
                                (!surface.rgba ||
                                 surface.rgba->size() != surface_texels * surface_bpp)) {
                                std::vector<uint8_t> materialized;
                                std::string error;
                                if (prosper::test::readback_persistent_color_target(
                                        sampled_source_addr, surface.w, surface.h, surface_format,
                                        materialized, error) &&
                                    materialized.size() == surface_texels * surface_bpp) {
                                    surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                                        std::move(materialized));
                                } else {
                                    static std::atomic<int> warned{0};
                                    if (warned.fetch_add(1) < 24)
                                        fprintf(stderr,
                                                "[rtt] lazy sampled target readback failed: "
                                                "base=0x%llx extent=%ux%u error=%s\n",
                                                (unsigned long long)sampled_source_addr, surface.w,
                                                surface.h, error.c_str());
                                }
                            }
                        }
                        static const bool retain_cpu_rtt_snapshots =
                            PROSPER_ENV_VALUE("PROSPER_NO_RTT_SNAPSHOT_BORROW") == nullptr;
                        // Pixel-mutating/inspection diagnostics intentionally retain their owned
                        // scratch copy. Normal consumers retain exact immutable snapshots directly
                        // and share each scaled materialization within this submit callback.
                        static const bool cpu_rtt_copy_diagnostics =
                            PROSPER_ENV_VALUE("PROSPER_DUMP_SAMPLED_RTT") || PROSPER_ENV_VALUE("PROSPER_DUMP_RAWTEX") ||
                            getenv("PROSPER_GFXLOG") || PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") ||
                            PROSPER_ENV_VALUE("PROSPER_PALETTELOG") || PROSPER_ENV_VALUE("PROSPER_TESTTEX") ||
                            PROSPER_ENV_VALUE("PROSPER_TESTLUT") || PROSPER_ENV_VALUE("PROSPER_TESTLUT32") ||
                            PROSPER_ENV_VALUE("PROSPER_DUMP_TEX") || PROSPER_ENV_VALUE("PROSPER_DUMP_ATLAS") ||
                            PROSPER_ENV_VALUE("PROSPER_KILL_RING");
                        const bool uniform_cpu_diagnostic_path =
                            live_rtt != g_rtt.end() &&
                            prosper::frontend::live_rtt_uniform_uses_cpu_diagnostic_path(
                                live_rtt->second.has_uniform_color,
                                cpu_rtt_copy_diagnostics);
                        if (uniform_cpu_diagnostic_path)
                            materialize_uniform_rtt(live_rtt->second);
                        const bool has_cpu_live_rtt = !fr.is_storage_image && sampled_2d_view &&
                            !r.in_mip_tail &&
                            live_rtt != g_rtt.end() &&
                            live_rtt->second.w && live_rtt->second.h && live_rtt->second.rgba &&
                            prosper::frontend::live_rtt_cpu_snapshot_matches(
                                live_rtt->second.w, live_rtt->second.h,
                                prosper::test::backend_color_bytes_per_pixel(
                                    prosper::test::backend_color_format(live_rtt->second.format)),
                                live_rtt->second.rgba->size());
                        // Match the established CPU injection gate below. Cube descriptors and
                        // mismatched 2D views can also retain identical materialized bytes; 3D
                        // volumes, storage images, and mip tails remain on their specialized paths.
                        const bool retain_cpu_live_rtt = retain_cpu_rtt_snapshots &&
                            !cpu_rtt_copy_diagnostics && !fr.is_storage_image &&
                            r.img_dim != 2u && !r.in_mip_tail && live_rtt != g_rtt.end() &&
                            live_rtt->second.w && live_rtt->second.h && live_rtt->second.rgba &&
                            prosper::frontend::live_rtt_cpu_snapshot_matches(
                                live_rtt->second.w, live_rtt->second.h,
                                prosper::test::backend_color_bytes_per_pixel(
                                    live_rtt->second.format),
                                live_rtt->second.rgba->size());
                        // A retained render target was created for color-attachment + sampled usage,
                        // not storage usage. Storage images therefore take the decoded/upload path.
                        const bool has_gpu_live_rtt = !fr.is_storage_image && live_gpu_targets &&
                            r.img_dim == 1u &&
                            live_rtt != g_rtt.end() && live_rtt->second.gpu_valid &&
                            prosper::frontend::rtt_sampled_extent_compatible(
                                tw, th, live_rtt->second.w, live_rtt->second.h, render_scale,
                                normalized_sampling) &&
                            prosper::test::find_persistent_color_target(
                                sampled_source_addr, live_rtt->second.w, live_rtt->second.h,
                                live_rtt->second.format) != nullptr;
                        const bool has_uniform_live_rtt = prosper::frontend::mrt_uniform_live_serves(
                            draw, sampled_source_addr, tw, th,
                            /*preconditions=*/!fr.is_storage_image &&
                                !uniform_cpu_diagnostic_path && sampled_2d_view &&
                                !r.in_mip_tail && live_rtt != g_rtt.end() &&
                                live_rtt->second.has_uniform_color && live_rtt->second.w &&
                                live_rtt->second.h &&
                                prosper::frontend::rtt_sampled_extent_compatible(
                                    tw, th, live_rtt->second.w, live_rtt->second.h, render_scale,
                                    normalized_sampling),
                            mrt_format_defined);
                        const bool has_live_rtt =
                            has_cpu_live_rtt || has_gpu_live_rtt || has_uniform_live_rtt;
                        // A dim-5 base-slice view may need the CPU injection path rather than a direct
                        // Vulkan bind, but the selected renderer target is still authoritative even if
                        // an on-demand readback cannot currently materialize it. Never validate/cache a
                        // guest decode for that identity: renderer-only writes cannot dirty guest pages.
                        const bool has_live_rtt_authority = has_live_rtt ||
                            (!fr.is_storage_image && r.img_dim == 5u && !r.in_mip_tail &&
                             live_rtt != g_rtt.end());
                        resource_has_live_rtt = has_live_rtt_authority;
                        // Resolve renderer-owned depth before considering guest-byte texture
                        // decoding. A sampled depth attachment has no authoritative color payload
                        // in guest memory: the retained Vulkan image is the source of truth. The old
                        // ordering first copied DCC metadata and probed the persistent CPU decode
                        // cache for every such binding, then discarded that work when this bridge
                        // won below. Deferred workloads can bind hundreds of depth views per submit,
                        // so select the exact GPU path up front.
                        const prosper::test::PersistentDsSampled sampled_ds =
                            !has_live_rtt && !fr.is_storage_image && r.img_dim == 1u &&
                                    r.cls == RC::Texture
                                ? prosper::test::find_persistent_ds_sampled(
                                      r.gpu_addr, tw, th, render_scale, normalized_sampling)
                                : prosper::test::PersistentDsSampled{};
                        const bool has_ds_live = sampled_ds.image != nullptr;
                        resource_has_ds_live = has_ds_live;
                        // A binding rejected by the gate above never reaches the lookup, so it is
                        // absent from every statistic the lookup keeps -- it reads as "we hold no
                        // such surface" when we may hold it and be perfectly valid. Name the
                        // failing sub-condition, once per (address, reason).
                        if (PROSPER_ENV_ON("PROSPER_DSBRIDGE_LOG") && !has_ds_live) {
                            uint32_t held_w = 0, held_h = 0;
                            if (!(!has_live_rtt && !fr.is_storage_image && r.img_dim == 1u &&
                                  r.cls == RC::Texture) &&
                                prosper::test::is_retained_ds_plane(r.gpu_addr, &held_w, &held_h)) {
                                static std::mutex gate_mutex;
                                static std::set<std::pair<uint64_t, int>> reported;
                                const int reason = has_live_rtt ? 0
                                    : fr.is_storage_image ? 1
                                    : r.img_dim != 1u ? 2 : 3;
                                bool first = false;
                                {
                                    std::lock_guard lock(gate_mutex);
                                    first = reported.emplace(r.gpu_addr, reason).second;
                                }
                                if (first)
                                    fprintf(stderr,
                                            "[dsbridge] GATED addr=0x%llx T#=%ux%ux%u dim=%u "
                                            "cls=%u fmt=%u (retained DS %ux%u) never reached the "
                                            "lookup: %s\n",
                                            (unsigned long long)r.gpu_addr, tw, th, r.depth,
                                            r.img_dim, (unsigned)r.cls, (unsigned)r.format,
                                            held_w, held_h,
                                            reason == 0 ? "has_live_rtt (colour cache claims it)"
                                            : reason == 1 ? "is_storage_image"
                                            : reason == 2 ? "img_dim != 1 (not a plain 2D view)"
                                                          : "cls != Texture");
                            }
                        }
                        const bool is_cube = r.img_dim == 3u;   // CUBE: six faces stacked vertically (#273)
                        const bool is_volume = r.img_dim == 2u;
                        // #325: a guest 2D_ARRAY. Cubes stack their six faces into height because
                        // six is small; an array cannot -- this title's level atlas is 256 slices,
                        // and 512x131072 exceeds every device's maxImageDimension2D. So arrays ride
                        // the backend's real layer channel (`sample_count`) instead, which
                        // image creation and the view already follow.
                        // prosper decodes BC to RGBA8 on the CPU, so a compressed array inflates 4x
                        // and is then held in BOTH host RAM (the persistent cache) and VRAM. This
                        // title's world atlas is 512x512x256 Bc7 = 256 MiB decoded, and another
                        // resource reaches 464 MiB. Above a budget, decode the base slice only --
                        // the pre-#325 behaviour -- rather than failing: an allocation failure here
                        // is silent, because vkCreateImage's result is discarded and the null handle
                        // is used anyway (#3045).
                        const uint64_t array_budget_bytes = array_decode_budget_bytes();
                        const uint64_t array_footprint =
                            (uint64_t)tw * th * 4ull * (r.depth ? r.depth : 1u);
                        // The SHAPE question -- is this a layered array? -- is what the
                        // recompiler also answers, so it must not depend on anything the recompiler
                        // cannot see. The budget below decides only how many layers we DECODE.
                        const bool guest_array =
                            prosper::gpu::guest_texture_is_uploaded_array(r.img_dim, r.depth,
                                                                          r.format) &&
                            r.cls == RC::Texture;
                        fr.guest_array = guest_array;
                        const bool is_array = guest_array &&
                            array_footprint <= array_budget_bytes;
                        if (guest_array && array_footprint > array_budget_bytes) {
                            static uint32_t over_budget_reports = 0;
                            if (over_budget_reports++ < 16u)
                                fprintf(stderr,
                                        "[array-budget] binding=%u %ux%u x%u layers = %llu MiB "
                                        "decoded exceeds %llu MiB; sampling the base slice only\n",
                                        r.binding, tw, th, r.depth,
                                        (unsigned long long)(array_footprint >> 20),
                                        (unsigned long long)(array_budget_bytes >> 20));
                        }
                        const uint32_t decoded_layers =
                            is_array ? (r.depth ? r.depth : 1u) : 1u;
                        // GTA V retains each shadow-cube face as a renderer-owned D32 image, then
                        // samples the cube through a UNORM16 descriptor. Guest memory is not the
                        // authority for these pixels. Select the complete six-face generation now
                        // and include it in the decode identity; unchanged cubes can then reuse the
                        // already quantized vertical stack without six synchronous depth readbacks.
                        prosper::test::PersistentDsCubeSelection retained_depth_cube;
                        bool retained_depth_cube_cache_candidate = false;
                        if (is_cube && r.depth == 6u &&
                            r.format == prosper::gpu::DataFormat::Unorm16 &&
                            r.num_components == 1u && r.cls == RC::Texture && !r.host_data &&
                            !has_live_rtt && !has_ds_live && !fr.is_storage_image &&
                            prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                            if (r.layer_stride_bytes)
                                prosper::test::note_ds_layer_stride(
                                    r.gpu_addr, tw, th, r.layer_stride_bytes);
                            retained_depth_cube =
                                prosper::test::select_persistent_ds_cube_depth(
                                    r.gpu_addr, tw, th);
                            retained_depth_cube_cache_candidate =
                                retained_depth_cube.present_mask != 0 &&
                                retained_depth_cube.overlay_version != 0;
                            if (retained_depth_cube_cache_candidate)
                                decode_key.renderer_overlay_version =
                                    (static_cast<uint64_t>(retained_depth_cube.present_mask) << 56) |
                                    (retained_depth_cube.overlay_version & 0x00ffffffffffffffull);
                        }
                        const uint32_t persistent_pitch = PROSPER_ENV_VALUE("PROSPER_PITCH")
                            ? static_cast<uint32_t>(atoi(getenv("PROSPER_PITCH"))) : 0;
                        const uint32_t persistent_bc_block_bytes =
                            prosper::gpu::bc_block_bytes(r.format);
                        const bool persistent_unorm8_texture =
                            r.format == prosper::gpu::DataFormat::Unorm8 &&
                            r.num_components >= 1 && r.num_components <= 4;
                        // UNORM16 inputs follow the same deterministic guest-byte decode contract.
                        // Retaining the already-decoded pixels does not change the existing channel
                        // expansion/swizzle semantics; it only avoids repeating them for immutable data.
                        const bool persistent_unorm16_texture =
                            r.format == prosper::gpu::DataFormat::Unorm16 &&
                            r.num_components >= 1 && r.num_components <= 2;
                        // Float16/HDR sampled textures decode to RGBA8 via a dedicated branch below but were
                        // excluded from the persistent decode cache, so HDR art was re-detiled every frame
                        // (#1177, the largest `notsampled` bucket). The cache machinery is format-agnostic —
                        // it stores the decoded RGBA8 and validates against the raw source bytes — so caching
                        // fp16 is correctness-preserving as long as persistent_source_size counts the fp16
                        // source (2 B/component) exactly, which is what the decode reads.
                        // Must match the fp16 DECODE branch's `f16` predicate exactly (bpt in {2,4,8} =
                        // 1/2/4 components); RGB16F (3 comp, bpt 6) takes a different decode path that reads
                        // a different byte count, so it must NOT be cached here or the exact-byte source
                        // validation would compare the wrong region.
                        const bool persistent_fp16_texture =
                            r.format == prosper::gpu::DataFormat::Float16 &&
                            (r.num_components == 1 || r.num_components == 2 || r.num_components == 4);
                        // Float32 sampled textures narrow to RGBA16F below. Like the existing fp16 path,
                        // the decoded result is fully determined by the exact source bytes and descriptor
                        // shape, so immutable versions can use the same validated cross-submit cache.
                        // This matters for titles that repeatedly bind large Float32 post-process inputs:
                        // otherwise every callback re-detiles and scalar-narrows the complete surface.
                        const bool persistent_fp32_texture =
                            r.format == prosper::gpu::DataFormat::Float32 &&
                            (r.num_components == 1 || r.num_components == 2 || r.num_components == 4);
                        // Exact single-channel integer compute outputs may be consumed immediately
                        // by graphics. Their guest bytes and native Vulkan texels have the same
                        // width, so the existing submit journal can authorize a device-local image
                        // lease instead of a writeback -> detile -> upload round trip.
                        const bool persistent_uint8_texture =
                            r.format == prosper::gpu::DataFormat::Uint8 &&
                            r.num_components == 1;
                        const bool persistent_uint32_texture =
                            r.format == prosper::gpu::DataFormat::Uint32 &&
                            r.num_components == 1;
                        // Packed 32-bit sampled formats are also pure decode inputs.  Keeping their
                        // decoded pixels is especially important for full-resolution HDR intermediates:
                        // unpacking R11G11B10F scalar-by-scalar on every callback otherwise dominates an
                        // entire frame.  Their encoded source is exactly one dword per texel regardless
                        // of the logical component count.
                        const bool persistent_packed32_texture =
                            (r.format == prosper::gpu::DataFormat::Float10_11_11 &&
                             r.num_components == 3) ||
                            (r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 &&
                             r.num_components == 4);
                        // Real source bytes per texel. It also determines the visible row size used to
                        // resolve a guest-backed linear image's pitch below: ordinary sampled images use
                        // GFX10's 256-byte alignment, while exact HLE-producer provenance may stay tight.
                        uint32_t sampled_source_bpt =
                            prosper::gpu::data_format_bytes(r.format) *
                            (r.num_components ? r.num_components : 1u);
                        const bool sampled_source_f16 =
                            r.format == prosper::gpu::DataFormat::Float16 &&
                            (sampled_source_bpt == 2 || sampled_source_bpt == 4 || sampled_source_bpt == 8);
                        const bool sampled_source_f32 =
                            r.cls == RC::Texture && r.img_dim != 3u && !r.compression_enabled &&
                            r.format == prosper::gpu::DataFormat::Float32 &&
                            (sampled_source_bpt == 4 || sampled_source_bpt == 8 ||
                             sampled_source_bpt == 16);
                        if (sampled_source_bpt == 0 ||
                            (sampled_source_bpt > 4 && !sampled_source_f16 && !sampled_source_f32))
                            sampled_source_bpt = 4;
                        const bool persistent_format_supported =
                            persistent_unorm8_texture || persistent_fp16_texture ||
                            persistent_unorm16_texture || persistent_fp32_texture ||
                            persistent_uint8_texture || persistent_uint32_texture ||
                            persistent_packed32_texture ||
                            (persistent_bc_block_bytes != 0 && !is_volume);
                        // GTA V's five shadow cubes are ordinary tiled Z16 allocations. The fallback
                        // decoder below reads all six faces from one exact descriptor footprint, so
                        // unlike the broad non-BC cube class that regressed Plucky Squire this one
                        // narrow shape can use exact-byte validation safely.
                        const bool exact_unorm16_cube = is_cube && r.depth == 6u &&
                            persistent_unorm16_texture && r.num_components == 1u &&
                            !r.compression_enabled;
                        const bool persistent_sampled_texture = texture_decode_cache_candidate(
                            has_live_rtt_authority, has_ds_live, r.host_data != nullptr, r.img_dim,
                            r.cls == RC::Texture, persistent_format_supported,
                            persistent_bc_block_bytes != 0, exact_unorm16_cube);
                        resource_persistent_candidate = persistent_sampled_texture ||
                            retained_depth_cube_cache_candidate;
                        const bool persistent_source_is_tiled =
                            persistent_sampled_texture && !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                            prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                        // A sampled LINEAR (tile_mode 0) 2D surface (or the selected base slice of a
                        // 2D array) whose tight row is not 256-aligned
                        // is pitch-padded: RDNA2 requires a sampled linear texture's row pitch to be aligned
                        // (256 B here), so its real pitch aligns either texel rows or BC block rows. The
                        // decode below must read it row-by-row at that pitch (else every row drifts ->
                        // horizontal scramble, e.g. Dead Cells' 348-wide Motion Twin splash whose real
                        // pitch is 384 texels). Guards keep
                        // this narrow and regression-safe: 2D or the already-selected 2D-array base slice
                        // only (volume/cube layouts untouched); tile_mode == 0 exactly, so
                        // unrecognized/actually-tiled modes are NOT strided (they fall to the contiguous
                        // read + auto-detile pass); exact HLE-producer provenance can select a tight guest
                        // layout (AvPlayer NV12); !host_data keeps ordinary CPU-uploaded test fixtures
                        // contiguous unless capture replay supplies an explicit guest-layout pitch.
                        // The tightly-packed
                        // LINEAR_GENERAL layout is a buffer/copy layout, not a sampled-texture layout, so it
                        // does not reach here. r.size is the TIGHT extent (tw*th*bpp), so it cannot gate this.
                        const uint32_t linear_row_width = persistent_bc_block_bytes
                            ? tw / 4u + static_cast<uint32_t>(tw % 4u != 0u) : tw;
                        const uint32_t linear_row_count = persistent_bc_block_bytes
                            ? th / 4u + static_cast<uint32_t>(th % 4u != 0u) : th;
                        const uint32_t linear_row_element_bytes = persistent_bc_block_bytes
                            ? persistent_bc_block_bytes : sampled_source_bpt;
                        const size_t linear_dst_row =
                            static_cast<size_t>(linear_row_width) * linear_row_element_bytes;
                        const uint32_t registered_linear_pitch = linear_dst_row <= UINT32_MAX
                            ? prosper::gpu::guest_linear_texture_row_pitch(
                                  r.gpu_addr, static_cast<uint32_t>(linear_dst_row))
                            : 0;
                        size_t linear_src_row = r.linear_row_pitch_bytes
                            ? r.linear_row_pitch_bytes
                            : (registered_linear_pitch
                                   ? registered_linear_pitch
                                   : prosper::gpu::linear_sampled_row_pitch(
                                         linear_row_width, linear_row_element_bytes));
                        if (const char* lp = PROSPER_ENV_VALUE("PROSPER_LINPITCH"))
                            linear_src_row =
                                (size_t)strtoull(lp, nullptr, 0) * linear_row_element_bytes;
                        const bool linear_padded_read =
                            r.cls == RC::Texture && (r.img_dim == 1u || r.img_dim == 5u) &&
                            r.tile_mode == 0 &&
                            (!r.host_data || r.linear_row_pitch_bytes != 0) &&
                            !r.compression_enabled &&
                            linear_dst_row != 0 && linear_src_row > linear_dst_row;
                        // Dynamic single-channel video/coverage surfaces are already exactly what a
                        // VK_FORMAT_R8_UNORM sampled image consumes. Expanding every byte to RGBA on the
                        // CPU multiplied Astro Bot's 3840x3240 FMV traffic by four and cost 26-31 ms per
                        // frame. Keep the historical grayscale broadcast through the image-view swizzle.
                        // img_dim is tested through avp_plane_is_one_layer_2d rather than
                        // `== 1u`: a title may declare a single-channel plane as a ONE-LAYER 2D
                        // ARRAY (dim 5, depth 1), which is the same memory and the same sampled
                        // values but missed this fast path and was CPU-expanded to RGBA8 --
                        // R-Type Delta (PPSA26414) ships its AvPlayer luma plane that way, so
                        // every movie frame moved 4x the bytes it needed (#2034). The helper
                        // also requires no layer stride and no layer mip offset, so a REAL
                        // multi-layer array still fails and keeps the expanded path.
                        const bool native_r8_sampled =
                            r.cls == RC::Texture && prosper::frontend::avp_plane_is_one_layer_2d(r) &&
                            r.format == prosper::gpu::DataFormat::Unorm8 &&
                            r.num_components == 1 && r.tile_mode == 0u &&
                            !r.compression_enabled && !linear_padded_read;
                        // AvPlayer's exact chroma plane is interleaved RG8. Keeping it native halves
                        // the upload bytes and avoids CPU RGBA expansion while the descriptor swizzle
                        // preserves U/V. A pitch-padded source is copied row-by-row below.
                        //
                        // LINEAR ONLY, and the classifier no longer implies it: since #2731 a
                        // recognised chroma plane may be GPU-TILED (Sonic Origins stages both NV12
                        // planes SW_64KB_S). This branch is a straight copy with no de-swizzle, so a
                        // tiled plane taken through it would upload micro-tile order as if it were
                        // scanline order -- a woven picture, worse than the collapse it replaced.
                        // A tiled plane instead falls to the `bpt < 4` path below, which detiles at
                        // the real 2-byte element size and then preserves both U and V because
                        // avplayer_chroma_layout is set.
                        const bool native_rg8_sampled = avplayer_chroma_layout && r.tile_mode == 0u;
                        // R11G11B10F is already a native Vulkan sampled format. Preserve the packed
                        // words after detiling instead of expanding to RGBA8: the old conversion
                        // clamped every HDR channel above 1.0, destroying environment-light energy.
                        // GTA V's 1024x512 lighting map has 2,536 of 6,123 sampled channels above
                        // 1.0 (1,791 above 16.0), so this was a semantic collapse, not quantization.
                        const bool native_r11_eligible =
                            native_r11_sampled_upload_supported(r);
                        const bool native_r11_sampled = native_r11_eligible &&
                            PROSPER_ENV_VALUE("PROSPER_NO_NATIVE_R11_SAMPLED") == nullptr &&
                            native_r11_sampled_selector().includes(r.gpu_addr);
                        if (native_r11_eligible &&
                            PROSPER_ENV_VALUE("PROSPER_NATIVE_R11_SAMPLED_LOG") != nullptr) {
                            static std::set<uint64_t> logged_native_r11;
                            if (logged_native_r11.insert(r.gpu_addr).second)
                                std::fprintf(stderr,
                                             "[native-r11] addr=0x%llx %ux%ux%u tile=%u "
                                             "declared_mips=%u selected=%u\n",
                                             static_cast<unsigned long long>(r.gpu_addr),
                                             r.width, r.height, r.depth, r.tile_mode,
                                             r.declared_mip_levels,
                                             native_r11_sampled ? 1u : 0u);
                        }
                        // Storage-image atomics require a typed integer Vulkan view. Keep R32_UINT
                        // texels byte-exact through the existing 4-B read/detile path instead of
                        // silently normalizing the view to RGBA8_UNORM.
                        const bool native_r32ui_storage =
                            r.cls == RC::StorageImage && r.img_dim == 1u &&
                            r.format == prosper::gpu::DataFormat::Uint32 &&
                            r.num_components == 1 && !r.compression_enabled;
                        // The portable graphics recompiler declares formatless uvec4 storage images.
                        // Their UINT Sampled Type is not compatible with the renderer's historical
                        // RGBA8_UNORM upload even when the same undefined read happened to return useful
                        // pixels. Materialize the same raw-channel ABI as live_compute instead.
                        const bool portable_raw_uvec4_storage =
                            fr.is_storage_image &&
                            reflected_binding->image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Uint &&
                            reflected_binding->storage_image_format == 0u;
                        const uint32_t portable_storage_guest_texel =
                            portable_raw_uvec4_storage
                                ? storage_image_guest_texel_bytes(
                                      r.format, r.num_components ? r.num_components : 1u)
                                : 0u;
                        const bool portable_storage_shape =
                            !r.compression_enabled &&
                            !reflected_binding->image_arrayed &&
                            !reflected_binding->image_multisampled &&
                            ((reflected_binding->image_dim == 1u && r.img_dim == 1u) ||
                             (reflected_binding->image_dim == 2u && r.img_dim == 2u &&
                              r.depth != 0u));
                        // A storage-image contract failure does not skip one binding -- the backend
                        // returns an EMPTY batch, dropping every draw submitted with it
                        // (render_runner.h, `return out;` in the pre-Vulkan validation loop). So
                        // "which sub-condition failed" is the difference between one unusable
                        // resource and a whole frame's geometry going missing, and the existing line
                        // reports only `materialized=0`, which is the conjunction.
                        if (portable_raw_uvec4_storage &&
                            (!portable_storage_guest_texel || !portable_storage_shape)) {
                            static std::set<std::tuple<uint32_t, uint32_t, uint32_t>> reported;
                            if (reported.emplace(fr.set, fr.binding,
                                                 static_cast<uint32_t>(r.format)).second)
                                std::fprintf(stderr,
                                    "[render] storage-image contract: set=%u binding=%u "
                                    "portable-uvec4 REJECTED guest-texel=%u shape=%u "
                                    "(writable=%u compressed=%u arrayed=%u multisampled=%u "
                                    "reflected-dim=%u guest-dim=%u depth=%u fmt=%u comps=%u)\n",
                                    fr.set, fr.binding, portable_storage_guest_texel,
                                    portable_storage_shape ? 1u : 0u,
                                    writable_storage_image ? 1u : 0u,
                                    r.compression_enabled ? 1u : 0u,
                                    reflected_binding->image_arrayed ? 1u : 0u,
                                    reflected_binding->image_multisampled ? 1u : 0u,
                                    reflected_binding->image_dim, r.img_dim, r.depth,
                                    static_cast<unsigned>(r.format), r.num_components);
                            fr.storage_image_contract_valid = false;
                        }
                        if (fr.is_storage_image &&
                            reflected_binding->image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Uint &&
                            reflected_binding->storage_image_format ==
                                prosper::gpu::kSpirvImageFormatR32ui &&
                            !native_r32ui_storage)
                            fr.storage_image_contract_valid = false;
                        if (fr.is_storage_image &&
                            reflected_binding->image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Sint)
                            fr.storage_image_contract_valid = false;
                        const VkFormat decoded_texture_format = has_cpu_live_rtt
                            ? live_rtt->second.format
                            : (portable_raw_uvec4_storage
                                   ? VK_FORMAT_R32G32B32A32_UINT
                               : (native_r32ui_storage ? VK_FORMAT_R32_UINT
                               : (native_r8_sampled ? VK_FORMAT_R8_UNORM
                               : (native_rg8_sampled ? VK_FORMAT_R8G8_UNORM
                               : (native_r11_sampled ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                               : (sampled_source_f32 ? VK_FORMAT_R16G16B16A16_SFLOAT
                                                     : VK_FORMAT_R8G8B8A8_UNORM))))));
                        const bool persistent_source_matches_pixels =
                            native_r8_sampled || native_rg8_sampled ||
                            (persistent_unorm8_texture && r.num_components == 4 &&
                             !linear_padded_read &&
                             !prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                             (!PROSPER_ENV_VALUE("PROSPER_DETILE") || atoi(PROSPER_ENV_VALUE("PROSPER_DETILE")) == 0));
                        const size_t persistent_base_source_size = [&] {
                            if (!persistent_sampled_texture) return size_t{0};
                            if (is_cube)
                                return layered_cube_source_size(
                                    persistent_bc_block_bytes != 0 || exact_unorm16_cube,
                                    sampled_source_addr,
                                    prosper::gpu::gpu_capture_resource_footprint(r));
                            // One SURFACE's bytes. For a layered array the decode reads many of
                            // these, so the layered span is applied below -- see #2998.
                            const size_t surface_bytes = [&] {
                            if (persistent_bc_block_bytes) {
                                const uint32_t bw = (tw + 3) / 4;
                                const uint32_t bh = (th + 3) / 4;
                                return persistent_source_is_tiled
                                    ? prosper::gpu::tiled_elements_bytes(
                                          bw, bh, persistent_bc_block_bytes, r.tile_mode)
                                    : (linear_padded_read
                                           ? linear_src_row * linear_row_count
                                           : static_cast<size_t>(bw) * bh *
                                                 persistent_bc_block_bytes);
                            }
                            if (persistent_packed32_texture)
                                return persistent_source_is_tiled
                                    ? (is_volume
                                           ? prosper::gpu::tiled_volume_bytes(
                                                 tw, th, r.depth, r.tile_mode, 4u)
                                           : prosper::gpu::tiled_surface_bytes(
                                                 tw, th, r.tile_mode, persistent_pitch, 4u))
                                    : static_cast<size_t>(tw) * th *
                                          (is_volume ? r.depth : 1u) * 4u;
                            // fp32 sources are 4 B/component, fp16/unorm16 are 2, and unorm8 is 1.
                            const uint32_t source_component_bytes =
                                (persistent_fp32_texture || persistent_uint32_texture) ? 4u
                                : ((persistent_fp16_texture || persistent_unorm16_texture) ? 2u : 1u);
                            const uint32_t source_bpt = r.num_components * source_component_bytes;
                            if (persistent_source_is_tiled)
                                return is_volume
                                    ? prosper::gpu::tiled_volume_bytes(
                                          tw, th, r.depth, r.tile_mode, source_bpt)
                                    : prosper::gpu::tiled_surface_bytes(
                                          tw, th, r.tile_mode, persistent_pitch, source_bpt);
                            // Forced detiling treats a nominally linear Unorm8x4 descriptor as tiled; its
                            // decode source is then not the linear byte range computed here. This exclusion
                            // is unorm8-specific — fp16 (which can also land at source_bpt==4 for RG16F)
                            // always reads exactly tw*th*source_bpt and must NOT be excluded.
                            if (persistent_unorm8_texture && source_bpt == 4 &&
                                !persistent_source_matches_pixels) return size_t{0};
                            if (linear_padded_read) return linear_src_row * th;
                            return static_cast<size_t>(tw) * th *
                                (is_volume ? r.depth : 1u) * source_bpt;
                            }();
                            // An uploaded ARRAY's decode reads EVERY layer, so the range the
                            // persistent cache validates has to span them all. Returning one
                            // surface made it compare 262144 of 90177536 bytes -- 0.29% -- of Tomb
                            // Raider's 256-layer world atlas: every layer above the first then
                            // changed invisibly, and a decode taken while the atlas was about 5%
                            // filled was reused for the rest of the run, which is why interiors
                            // sampled slices that held the previous occupants of that memory
                            // (#2998). The `is_cube` branch above has always spanned its six faces;
                            // the array path was added later without the matching change.
                            //
                            // A surface_bytes of 0 is a deliberate "do not cache this" signal from
                            // the branches above, so it must not be turned into a nonzero span.
                            if (guest_array)
                                return prosper::frontend::layered_array_source_size(
                                    surface_bytes, r.layer_stride_bytes, decoded_layers);
                            return surface_bytes;
                        }();
                        // DCC code 0xff means the base allocation contains ordinary uncompressed
                        // texels.  Such an image is just as cacheable as a descriptor with DCC disabled,
                        // provided we re-check the complete metadata plane before every reuse.  Other
                        // metadata states remain on the existing fast-clear/unsupported paths: caching
                        // them from base bytes alone would miss a metadata-only content transition.
                        const uint64_t sampled_dcc_metadata_size =
                            !has_ds_live && r.compression_enabled
                            ? prosper::gpu::gpu_capture_dcc_metadata_footprint(r)
                            : 0u;
                        std::vector<uint8_t> sampled_dcc_metadata(
                            static_cast<size_t>(sampled_dcc_metadata_size), 0);
                        const size_t sampled_dcc_metadata_got = sampled_dcc_metadata_size
                            ? copy_dcc_metadata(sampled_dcc_metadata.data(),
                                                sampled_dcc_metadata.size())
                            : 0u;
                        const bool persistent_dcc_uncompressed = r.compression_enabled &&
                            sampled_dcc_metadata_got == sampled_dcc_metadata.size() &&
                            !sampled_dcc_metadata.empty() &&
                            std::all_of(sampled_dcc_metadata.begin(), sampled_dcc_metadata.end(),
                                        [](uint8_t code) { return code == 0xff; });
                        uint8_t persistent_dcc_clear_pixel[4]{};
                        const bool persistent_dcc_fast_clear = r.compression_enabled &&
                            sampled_dcc_metadata_got == sampled_dcc_metadata.size() &&
                            prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                                persistent_dcc_clear_pixel, 1,
                                sampled_dcc_metadata.data(), sampled_dcc_metadata.size(),
                                r.num_components, r.alpha_is_on_msb);
                        const uint64_t persistent_source_addr = persistent_dcc_fast_clear
                            ? r.metadata_addr : sampled_source_addr;
                        const size_t persistent_source_size = persistent_dcc_fast_clear
                            ? sampled_dcc_metadata.size() : persistent_base_source_size;
                        resource_persistent_source_size = persistent_source_size;
                        resource_texture_source_bytes = persistent_source_size;
                        // A successful typed-storage compute dispatch may still own this exact
                        // sampled image on the shared Vulkan device. Prefer that image only for the
                        // exact native formats the graphics backend preserves today. The compute cache
                        // independently requires the complete descriptor key plus a current-submit
                        // journal or page-watch proof; a miss falls through to the existing exact
                        // guest-byte decode/cache path.
                        prosper::frontend::LiveComputeImageImport compute_image_import;
                        const VkFormat compute_image_format = static_cast<VkFormat>(
                            prosper::frontend::live_compute_graphics_import_native_format(
                                r.format, r.num_components));
                        // A cube decoder's ordinary source span covers one face, while its retained
                        // Uint16 2D-array producer is keyed by all six. Keep that exception beside
                        // the importer; ordinary decoded formats continue to use the exact span the
                        // persistent cache validates.
                        const uint64_t compute_image_guest_bytes =
                            prosper::frontend::live_compute_graphics_import_guest_bytes(
                                r, persistent_source_size);
                        const bool compute_image_shape =
                            (r.img_dim == 1u && r.depth == 1u) ||
                            (r.img_dim == 2u && r.depth != 0u) ||
                            (r.img_dim == 3u && r.depth == 6u);
                        const bool compute_image_candidate =
                            !PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_COMPUTE_IMAGE_BIND") &&
                            !has_live_rtt && !has_ds_live && r.cls == RC::Texture &&
                            compute_image_shape && !r.in_mip_tail &&
                            r.declared_mip_levels == 1u && !r.srgb &&
                            (!r.depth_compare || (r.img_dim == 3u && r.depth == 6u)) &&
                            !r.host_data &&
                            compute_image_guest_bytes &&
                            (!r.compression_enabled || persistent_dcc_uncompressed) &&
                            compute_image_format != VK_FORMAT_UNDEFINED;
                        resource_compute_image_candidate = compute_image_candidate;
                        if (compute_image_candidate &&
                            prosper::frontend::import_live_compute_storage_image(
                                r, compute_image_guest_bytes, compute_image_import)) {
                            const prosper::test::RenderVkCtx& render_context =
                                prosper::test::render_vk_ctx();
                            resource_compute_image_hit = compute_image_import.valid() &&
                                compute_image_import.native_format ==
                                    static_cast<uint32_t>(compute_image_format) &&
                                compute_image_import.width == tw &&
                                compute_image_import.height == th &&
                                compute_image_import.depth == r.depth && render_context.ok &&
                                compute_image_import.device ==
                                    static_cast<void*>(render_context.dev);
                            // #3307: a borrow the renderer then discards costs exactly what a miss
                            // costs, and the compute backend cannot see it happen.
                            prosper::frontend::live_compute_record_image_borrow_renderer_verdict(
                                resource_compute_image_hit);
                            if (!resource_compute_image_hit) compute_image_import = {};
                        }
                        if (resource_compute_image_hit &&
                            compute_image_import.vertical_stack_layers == 6u) {
                            // The compute image is the exact five-face copy result. A later graphics
                            // span can render the remaining face into a retained D32 attachment at
                            // the same guest cube address. Merge only attachment writes ordered
                            // after this producer; older retained faces are precisely what compute
                            // replaced and must not win merely because they still reside in cache.
                            if (r.layer_stride_bytes)
                                prosper::test::note_ds_layer_stride(
                                    r.gpu_addr, tw, th, r.layer_stride_bytes);
                            const auto overlay =
                                prosper::test::select_persistent_ds_cube_depth_after(
                                    r.gpu_addr, tw, th,
                                    compute_image_import.producer_command_order);
                            if (overlay.present_mask) {
                                resource_compute_depth_hybrid = true;
                                resource_compute_producer_order =
                                    compute_image_import.producer_command_order;
                                resource_compute_depth_overlay_mask = overlay.present_mask;
                                decode_key.renderer_overlay_version = overlay.overlay_version;
                                compute_image_import = {}; // release the lease; CPU builds the hybrid
                                resource_compute_image_hit = false;
                            }
                        }
                        auto copy_persistent_source = [&](uint8_t* dst, size_t bytes) {
                            return persistent_dcc_fast_clear
                                ? copy_dcc_metadata(dst, bytes)
                                : copy_resource(dst, sampled_source_addr, bytes);
                        };
                        const bool guest_persistent_cache_eligible =
                            persistent_texture_decode_cache_eligible(
                                persistent_sampled_texture,
                                resource_compute_image_hit || resource_compute_depth_hybrid,
                                fr.is_storage_image,
                                PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") != nullptr,
                                !r.compression_enabled || persistent_dcc_uncompressed ||
                                    persistent_dcc_fast_clear,
                                persistent_decode_limit, persistent_source_size);
                        const bool retained_depth_cube_cache_eligible =
                            retained_depth_cube_cache_candidate &&
                            !PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") &&
                            persistent_decode_limit != 0;
                        const bool persistent_cache_eligible =
                            guest_persistent_cache_eligible ||
                            retained_depth_cube_cache_eligible;
                        // Range a submit-scoped identity entry may be re-proved against across a span
                        // boundary (#1691). It is exactly the range the persistent decode cache
                        // validates for this identity — the same bytes `validate_exact()` compares and
                        // the same range its own in-submit reuse queries — so the fast path never
                        // asserts more than the cache it short-circuits. Ineligible resources
                        // (captured replay backing, storage images, unsupported DCC states, cache
                        // disabled) have no such established range, so they keep the pre-#1691
                        // span-local lifetime instead of being retained against an unverified extent.
                        const uint64_t cross_span_source_size = persistent_cache_eligible
                            ? persistent_source_size : 0u;
                        static const bool cross_submit_watch_enabled =
                            !PROSPER_ENV_VALUE("PROSPER_NO_CROSS_SUBMIT_TEXTURE_WRITE_WATCH");
                        // Page-protection watches have a fixed setup/query cost and may need to
                        // resolve every alias of every covered page.  For small textures an exact
                        // byte comparison is both simpler and cheaper; reserve dirty tracking for
                        // sources large enough for it to amortize.  Keep the cutoff tunable for
                        // host/platform profiling without changing the cache's correctness policy.
                        static const size_t cross_submit_watch_min_bytes = [] {
                            // Same family, same sentinel: 0 makes every source large enough, so a
                            // typo would widen eligibility rather than disable the knob (#3253).
                            const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_MIN_KB");
                            const uint64_t kib = prosper::diag::env_u64_or_default_capped(
                                "PROSPER_TEXTURE_WRITE_WATCH_MIN_KB", value, 1024ull,
                                SIZE_MAX / 1024ull, "KiB");
                            return static_cast<size_t>(kib * 1024ull);
                        }();
                        const bool cross_submit_watch_eligible = cross_submit_watch_enabled &&
                            persistent_source_size >= cross_submit_watch_min_bytes;
                        static const size_t cross_submit_watch_defer_min_bytes = [] {
                            // 0 means "defer nothing": should_promote_write_watch returns true
                            // immediately, arming every source on first sight. This is the knob
                            // #3253 names first, because it is the one an A/B is most likely to
                            // mistype (`=8mb`, `=8 KB`, a stray quote) -- and a bare strtoull would
                            // hand back the MOST aggressive arm instead of the one asked for.
                            const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB");
                            const uint64_t kib = prosper::diag::env_u64_or_default_capped(
                                "PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", value, 8192ull,
                                SIZE_MAX / 1024ull, "KiB");
                            return static_cast<size_t>(kib * 1024ull);
                        }();
                        static const uint32_t cross_submit_watch_promotion_validations = [] {
                            // 0 makes `stable >= 0` always true, so every deferred source promotes
                            // on its first acquisition -- the aggressive end, not the off end.
                            const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS");
                            return static_cast<uint32_t>(prosper::diag::env_u64_or_default_capped(
                                "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", value, 3ull, UINT32_MAX,
                                "unchanged validations"));
                        }();
                        static const bool audit_cross_submit_watch =
                            PROSPER_ENV_VALUE("PROSPER_AUDIT_CROSS_SUBMIT_TEXTURE_WRITE_WATCH") != nullptr;
                        prosper::host::GuestWriteWatch pending_source_watch;
                        // `narrow_decode_done` prevents the generic 32-bpp detiler from touching an already
                        // expanded narrow surface. `narrow_done` separately records the legacy coverage
                        // broadcast whose RGBA pixels intentionally replace the T# swizzle. UNORM16 expands
                        // to (R,0,0,1) and must retain the real descriptor swizzle, so the two facts differ.
                        bool narrow_done = false;
                        bool narrow_decode_done = false;
                        auto reused = (has_live_rtt || resource_compute_image_hit)
                            ? decoded_textures.end() : decoded_textures.find(decode_key);
                        // A retained entry from an EARLIER span in this submit is usable only while
                        // its range is still this binding's range AND the ordered journal proves
                        // nothing wrote it. Overlap (an interleaved compute/DMA/EOP write landed on
                        // the backing), a moved range, and Unknown (no journal, overflowed, or a
                        // different submit) all drop the entry, releasing its scratch pin, and the
                        // resolve falls through to the persistent cache's own validation exactly as
                        // it did before the map was widened. The journal query is skipped for a
                        // same-span hit: its verdict cannot change the answer, and this runs once
                        // per texture reference.
                        if (reused != decoded_textures.end()) {
                            const bool same_span = reused->second.span == decode_span_ordinal;
                            // Unknown, not Unchanged: the predicate ignores this on a same-span
                            // hit, so the value is unobservable today — but Unchanged is the
                            // PERMISSIVE verdict, and if the two same-span tests ever drift apart
                            // the placeholder would silently authorise reuse. Unknown fails closed
                            // at identical cost.
                            prosper::gpu::GuestGpuWriteQuery journal_query =
                                prosper::gpu::GuestGpuWriteQuery::Unknown;
                            if (!same_span) {
                                journal_query = prosper::gpu::guest_gpu_writes_since(
                                    reused->second.snapshot, reused->second.source_addr,
                                    reused->second.source_size);
                            }
                            if (!submit_local_texture_decode_reusable(
                                    reused->second.span, decode_span_ordinal,
                                    reused->second.source_addr, reused->second.source_size,
                                    persistent_source_addr, cross_span_source_size,
                                    journal_query)) {
                                if (reused->second.texstore_slot < texstore_pinned.size())
                                    texstore_pinned[reused->second.texstore_slot] = false;
                                decoded_textures.erase(reused);
                                reused = decoded_textures.end();
                                ++g_texture_decode_scope.invalidations;
                            }
                        }
                        DecodedTexture persistent_reuse;
                        const DecodedTexture* decoded_reuse = reused != decoded_textures.end()
                            ? &reused->second : nullptr;
                        resource_local_reuse = decoded_reuse != nullptr;
                        if (decoded_reuse) {
                            if (decoded_reuse->span == decode_span_ordinal) {
                                ++g_texture_decode_scope.same_span_reuses;
                            } else {
                                ++g_texture_decode_scope.cross_span_reuses;
                                // Unchanged was just proven up to here, so this instant is a valid new
                                // baseline. Advancing it keeps each later query scanning only the
                                // writes since the previous use instead of the whole submit journal.
                                reused->second.snapshot = prosper::gpu::guest_gpu_write_snapshot();
                            }
                        }
                        if (!decoded_reuse && persistent_cache_eligible) {
                            auto cached = persistent_decoded_textures.find(decode_key);
                            if (cached != persistent_decoded_textures.end() &&
                                cached->second.source_addr == persistent_source_addr &&
                                cached->second.source_size == persistent_source_size) {
                                resource_texture_watch_active =
                                    static_cast<bool>(cached->second.source_watch);
                                resource_texture_watch_disabled =
                                    cached->second.source_watch_disabled;
                                resource_texture_watch_only = cached->second.source_watch_only;
                                resource_texture_watch_stability =
                                    cached->second.source_watch_stable_validations;
                                auto validate_exact = [&] {
                                    // A successfully promoted Linux watch may own the mutation proof
                                    // without retaining a second encoded copy. Dirty/Unknown must miss;
                                    // there is deliberately no probabilistic hash fallback here.
                                    if (cached->second.source_watch_only) return false;
                                    resource_texture_exact_validation = true;
                                    const auto validation_start = timing_enabled
                                        ? RenderClock::now() : RenderClock::time_point{};
                                    bool matches = false;
                                    size_t validated_bytes = 0;
                                    if (cached->second.source_matches_pixels) {
                                        matches = safe_equal(
                                            cached->second.pixels ? cached->second.pixels->data() : nullptr,
                                            persistent_source_addr,
                                            persistent_source_size, validated_bytes) &&
                                            validated_bytes == cached->second.source_prefix_size;
                                    } else if (!PROSPER_ENV_VALUE("PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY") &&
                                               cached->second.source_prefix.size() ==
                                                   persistent_source_size) {
                                        // The cached prefix owns the complete encoded texture. Compare
                                        // guest memory directly against it: copying the same 100+ MiB
                                        // working set into a scratch buffer before memcmp doubled the
                                        // validation traffic on every Evergate frame. safe_equal keeps
                                        // the same sparse-page/readability guards and exact byte check.
                                        matches = safe_equal(
                                            cached->second.source_prefix.data(), persistent_source_addr,
                                            persistent_source_size, validated_bytes) &&
                                            validated_bytes == persistent_source_size;
                                    } else {
                                        persistent_validation_scratch.resize(persistent_source_size);
                                        validated_bytes = copy_persistent_source(
                                            persistent_validation_scratch.data(),
                                            persistent_source_size);
                                        matches =
                                            validated_bytes == cached->second.source_prefix_size &&
                                            validated_bytes == cached->second.source_prefix.size() &&
                                            (validated_bytes == 0 || !std::memcmp(
                                                persistent_validation_scratch.data(),
                                                cached->second.source_prefix.data(), validated_bytes));
                                    }
                                    resource_texture_validated_bytes += validated_bytes;
                                    if (timing_enabled) {
                                        pending_timing.persistent_validations++;
                                        pending_timing.persistent_validation_bytes += validated_bytes;
                                        resource_texture_validation_ms +=
                                            std::chrono::duration<double, std::milli>(
                                                RenderClock::now() - validation_start).count();
                                    }
                                    return matches;
                                };
                                static const bool submit_reuse_enabled =
                                    !PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE");
                                static const bool audit_submit_reuse =
                                    PROSPER_ENV_VALUE("PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE") != nullptr;
                                const prosper::gpu::GuestGpuWriteQuery submit_query =
                                    submit_reuse_enabled
                                    ? prosper::gpu::guest_gpu_writes_since(
                                        cached->second.validation_snapshot, persistent_source_addr,
                                        persistent_source_size)
                                    : prosper::gpu::GuestGpuWriteQuery::Unknown;
                                resource_texture_submit_query =
                                    static_cast<int>(submit_query);
                                if (timing_enabled && submit_reuse_enabled) {
                                    if (submit_query == prosper::gpu::GuestGpuWriteQuery::Unknown) {
                                        pending_timing.persistent_submit_unknown++;
                                        if (prosper::gpu::guest_gpu_write_tracking_active())
                                            pending_timing.persistent_submit_stale++;
                                        else
                                            pending_timing.persistent_submit_unarmed++;
                                    } else if (submit_query == prosper::gpu::GuestGpuWriteQuery::Overlap)
                                        pending_timing.persistent_submit_overlap++;
                                }
                                const bool submit_unchanged = submit_reuse_enabled &&
                                    submit_query == prosper::gpu::GuestGpuWriteQuery::Unchanged;
                                prosper::host::GuestWriteWatchQuery watch_query =
                                    prosper::host::GuestWriteWatchQuery::Unknown;
                                if (!submit_unchanged && cross_submit_watch_eligible &&
                                    !cached->second.source_watch_disabled) {
                                    watch_query = cached->second.source_watch.query();
                                    resource_texture_watch_query =
                                        static_cast<int>(watch_query);
                                    if (timing_enabled) {
                                        if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty)
                                            pending_timing.persistent_watch_dirty++;
                                        else if (watch_query == prosper::host::GuestWriteWatchQuery::Unknown)
                                            pending_timing.persistent_watch_unknown++;
                                    }
                                    if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty &&
                                        ++cached->second.source_watch_dirty_count >= 2) {
                                        cached->second.source_watch.reset();
                                        cached->second.source_watch_disabled = true;
                                        if (timing_enabled)
                                            pending_timing.persistent_watch_disabled++;
                                    } else if (watch_query ==
                                               prosper::host::GuestWriteWatchQuery::Unchanged) {
                                        cached->second.source_watch_dirty_count = 0;
                                    }
                                }
                                const bool watch_unchanged = !submit_unchanged &&
                                    watch_query == prosper::host::GuestWriteWatchQuery::Unchanged;
                                const bool watch_ready =
                                    prosper::frontend::should_promote_write_watch(
                                        persistent_source_size,
                                        cached->second.source_watch_stable_validations,
                                        cross_submit_watch_defer_min_bytes,
                                        cross_submit_watch_promotion_validations);
                                const bool may_arm_watch = cached->second.source_watch ||
                                    (!submit_unchanged && !watch_unchanged &&
                                     cross_submit_watch_eligible &&
                                     !cached->second.source_watch_disabled && watch_ready &&
                                     write_watch_promotion_budget.try_consume(
                                         persistent_source_size));
                                if (!submit_unchanged && !watch_unchanged &&
                                    cross_submit_watch_eligible &&
                                    !cached->second.source_watch_disabled &&
                                    may_arm_watch) {
                                    // Arm before reading. A concurrent CPU write during or after the
                                    // authoritative comparison then dirties this registration instead of
                                    // landing in an unprotected compare-to-rearm window.
                                    if (!cached->second.source_watch.rearm())
                                        cached->second.source_watch =
                                            prosper::host::GuestWriteWatch::create(
                                                persistent_source_addr, persistent_source_size);
                                }
                                bool content_matches = retained_depth_cube_cache_eligible;
                                if (retained_depth_cube_cache_eligible) {
                                    // The key carries the newest selected retained-depth write.
                                    // Finding this entry is the exact renderer-authority proof;
                                    // there are intentionally no guest bytes to compare.
                                    resource_texture_exact_validation = false;
                                } else if (submit_unchanged || watch_unchanged) {
                                    const bool audit = submit_unchanged
                                        ? audit_submit_reuse : audit_cross_submit_watch;
                                    content_matches = audit ? validate_exact() : true;
                                    if (!content_matches) {
                                        fprintf(stderr,
                                                "[render] %s texture validation audit failed "
                                                "addr=0x%llx bytes=%zu\n",
                                                submit_unchanged ? "in-submit" : "cross-submit-watch",
                                                (unsigned long long)r.gpu_addr,
                                                persistent_source_size);
                                    } else {
                                        if (submit_unchanged) {
                                            resource_persistent_submit_reuse = true;
                                            if (timing_enabled)
                                                pending_timing.persistent_submit_reuses++;
                                        } else if (timing_enabled) {
                                            pending_timing.persistent_watch_reuses++;
                                        }
                                    }
                                } else {
                                    content_matches = validate_exact();
                                }
                                if (!submit_unchanged && !watch_unchanged) {
                                    cached->second.source_watch_stable_validations =
                                        prosper::frontend::update_write_watch_stability(
                                            cached->second.source_watch_stable_validations,
                                            content_matches,
                                            cross_submit_watch_promotion_validations);
                                }
                                resource_texture_watch_active =
                                    static_cast<bool>(cached->second.source_watch);
                                resource_texture_watch_disabled =
                                    cached->second.source_watch_disabled;
                                resource_texture_watch_only = cached->second.source_watch_only;
                                resource_texture_watch_stability =
                                    cached->second.source_watch_stable_validations;
                                if (content_matches) {
                                    cached->second.last_use = decode_generation;
                                    cached->second.validation_snapshot =
                                        prosper::gpu::guest_gpu_write_snapshot();
                                    static const bool keep_source_snapshots =
                                        PROSPER_ENV_VALUE("PROSPER_KEEP_TEXTURE_SOURCE_SNAPSHOTS") != nullptr;
                                    if (!keep_source_snapshots &&
                                        texture_source_snapshot_can_follow_watch(
                                            cached->second.source_matches_pixels,
                                            audit_submit_reuse || audit_cross_submit_watch,
                                            static_cast<bool>(cached->second.source_watch),
                                            cached->second.source_prefix.size(),
                                            cached->second.source_size)) {
                                        const size_t released = cached->second.source_prefix.size();
                                        std::vector<uint8_t>().swap(cached->second.source_prefix);
                                        cached->second.source_watch_only = true;
                                        persistent_decoded_texture_bytes -= released;
                                    }
                                    persistent_reuse = {cached->second.pixels ? cached->second.pixels->data() : nullptr,
                                                        cached->second.output_height,
                                                        cached->second.narrow,
                                                        cached->second.persistent_id,
                                                        cached->second.persistent_version,
                                                        decode_span_ordinal,
                                                        cached->second.validation_snapshot,
                                                        cached->second.source_addr,
                                                        cross_span_source_size,
                                                        SIZE_MAX,
                                                        cached->second.pixels};
                                    persistent_reuse.texture_format =
                                        cached->second.texture_format;
                                    persistent_reuse.storage_image_contract_valid =
                                        cached->second.storage_image_contract_valid;
                                    persistent_reuse.layers = cached->second.layers;
                                    persistent_reuse.pixels_bytes = cached->second.pixels ? cached->second.pixels->size() : 0;
                                    decoded_reuse = &persistent_reuse;
                                    resource_persistent_hit = true;
                                    if (timing_enabled) pending_timing.persistent_hits++;
                                } else {
                                    resource_persistent_invalidation = true;
                                    if (timing_enabled)
                                        pending_timing.persistent_invalidations++;
                                }
                            } else {
                                // Establish the mutation boundary before the initial source read/decode.
                                // If registration is unsupported, the empty watch keeps all later reuse on
                                // the exact fallback.
                                if (cross_submit_watch_eligible &&
                                    prosper::frontend::should_promote_write_watch(
                                        persistent_source_size, 0,
                                        cross_submit_watch_defer_min_bytes,
                                        cross_submit_watch_promotion_validations))
                                    pending_source_watch = prosper::host::GuestWriteWatch::create(
                                        persistent_source_addr, persistent_source_size);
                                resource_persistent_miss = true;
                                if (timing_enabled) {
                                    pending_timing.persistent_misses++;
                                }
                            }
                        }
                        // PROSPER_BIND_LOG=<min-cb>: log every sampled-resource binding decision for
                        // 3 final callbacks (same ordinal as PROSPER_PASS_LOG) — which path serves
                        // each guest address, so a black consumer input can be attributed.
                        static const char* const bind_log = getenv("PROSPER_BIND_LOG");
                        if (bind_log) {
                            const uint64_t at = g_pass_log_submit.load(std::memory_order_relaxed);
                            const uint64_t bl_min = std::strtoull(bind_log, nullptr, 0);
                            if (at >= bl_min && at < bl_min + 3u)
                                fprintf(stderr,
                                        "[bind] cb=%llu addr=0x%llx %ux%u dim=%u fmt=%u path=%s "
                                        "rtt(w=%u h=%u gpu=%d rgba=%d)\n",
                                        (unsigned long long)at, (unsigned long long)r.gpu_addr,
                                        tw, th, r.img_dim, (unsigned)r.format,
                                        has_gpu_live_rtt          ? "gpu-bind"
                                        : has_uniform_live_rtt    ? "uniform-clear"
                                        : has_cpu_live_rtt        ? "cpu-rtt"
                                        : resource_compute_image_hit ? "compute-bind"
                                        : decoded_reuse           ? "decode-cache"
                                                                  : "decode",
                                        live_rtt != g_rtt.end() ? live_rtt->second.w : 0,
                                        live_rtt != g_rtt.end() ? live_rtt->second.h : 0,
                                        live_rtt != g_rtt.end() ? (int)live_rtt->second.gpu_valid
                                                                : -1,
                                        live_rtt != g_rtt.end() ? (int)(bool)live_rtt->second.rgba
                                                                : -1);
                        }
                        // Sampled depth bridge (#1275): the T# addresses the depth plane of a
                        // surface prosper rendered into a persistent Vulkan DS image. Guest memory
                        // never receives that depth, so the guest-byte decode below would sample
                        // zeros (every shadow compare passes -> unshadowed, overbright scenes).
                        // Bind the retained depth image directly instead. The guest T# keeps its
                        // native extent under PROSPER_RENDER_SCALE>1 while the renderer-owned image
                        // is uniformly smaller; normalized depth sampling maps onto that image just
                        // like the color-target bridge above. Keep the actual image extent in the
                        // backend resource so its exact cache lookup remains unambiguous.
                        if (has_gpu_live_rtt) {
                            fr.persistent_render_target_id = sampled_source_addr;
                            fr.tw = live_rtt->second.w;
                            fr.th = live_rtt->second.h;
                            fr.td = 1; fr.img_dim = r.img_dim;
                            fr.texture_format = live_rtt->second.format;
                            // CB_COLOR renders one mip view per target, while this T# samples the
                            // complete allocation. Reconstruct the target identities and retain
                            // every level that is still authoritative on the GPU. The backend will
                            // copy them into one sampled mip image. Expose only the contiguous
                            // renderer-owned prefix: inventing a missing tail by filtering the last
                            // live level corrupts temporal/bloom chains whose real producer has not
                            // run yet. Requiring at least two live levels keeps the
                            // historical zero-copy bind for ordinary single-target resources.
                            uint32_t mip_bytes_per_texel =
                                prosper::gpu::data_format_bytes(r.format) *
                                (r.num_components ? r.num_components : 1u);
                            if ((r.format == prosper::gpu::DataFormat::Float10_11_11 &&
                                 r.num_components == 3u) ||
                                (r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 &&
                                 r.num_components == 4u))
                                mip_bytes_per_texel = 4u;
                            const bool renderer_mip_chain_selected =
                                PROSPER_ENV_VALUE("PROSPER_NO_RENDERER_MIP_CHAIN") == nullptr &&
                                renderer_mip_chain_selector().includes(sampled_source_addr);
                            const RendererMipChainLayout mip_chain = renderer_mip_chain_selected
                                ? renderer_mip_chain_layout(
                                      sampled_source_addr, tw, th, mip_bytes_per_texel,
                                      r.tile_mode, r.declared_mip_levels)
                                : RendererMipChainLayout{};
                            const VkFormat mip_format = prosper::test::backend_color_format(
                                live_rtt->second.format);
                            const size_t retained_mips_consumed = mip_chain.level_count > 1u
                                ? consume_renderer_mip_chain(mip_chain, fr.tw, fr.th, mip_format)
                                : 0u;
                            uint32_t live_mip_levels = 0;
                            uint32_t live_mip_mask = 0;
                            uint32_t present_mip_mask = 0;
                            if (mip_chain.level_count > 1u) {
                                for (uint32_t level = 0; level < mip_chain.level_count; ++level) {
                                    const uint32_t level_w = std::max(fr.tw >> level, 1u);
                                    const uint32_t level_h = std::max(fr.th >> level, 1u);
                                    auto* retained = prosper::test::find_persistent_color_target(
                                        mip_chain.level_ids[level], level_w, level_h,
                                        mip_format, false);
                                    if (retained) present_mip_mask |= 1u << level;
                                    if (retained && retained->valid) {
                                        fr.persistent_render_target_mip_ids[level] =
                                            mip_chain.level_ids[level];
                                        ++live_mip_levels;
                                        live_mip_mask |= 1u << level;
                                    }
                                }
                            }
                            uint32_t exposed_mip_levels = 0;
                            while (exposed_mip_levels < mip_chain.level_count &&
                                   (live_mip_mask & (1u << exposed_mip_levels)) != 0u)
                                ++exposed_mip_levels;
                            if (exposed_mip_levels > 1u &&
                                fr.persistent_render_target_mip_ids[0] ==
                                    fr.persistent_render_target_id) {
                                fr.persistent_render_target_mip_count = exposed_mip_levels;
                                fr.declared_mip_levels = exposed_mip_levels;
                            } else {
                                fr.persistent_render_target_mip_ids = {};
                            }
                            if (PROSPER_ENV_VALUE("PROSPER_RENDERER_MIP_CHAIN_LOG") != nullptr &&
                                r.declared_mip_levels > 1u) {
                                static std::set<uint64_t> logged_renderer_mips;
                                if (logged_renderer_mips.insert(sampled_source_addr).second) {
                                    std::fprintf(stderr,
                                                 "[renderer-mips] addr=0x%llx %ux%u fmt=%u "
                                                 "tile=%u declared=%u layout=%u live=%u "
                                                 "present=0x%x valid=0x%x exposed=%u "
                                                 "selected=%u retained=%zu shader=%llu set=%u binding=%u "
                                                 "lod=%.3f..%.3f bias=%.3f filter=%u/%u/%u "
                                                 "cache=%zu/%.1fMiB limit=%zu/%.1fMiB ids=",
                                                 static_cast<unsigned long long>(sampled_source_addr),
                                                 tw, th, static_cast<unsigned>(r.format), r.tile_mode,
                                                 r.declared_mip_levels, mip_chain.level_count,
                                                 live_mip_levels, present_mip_mask, live_mip_mask,
                                                 exposed_mip_levels,
                                                 renderer_mip_chain_selected ? 1u : 0u,
                                                 retained_mips_consumed,
                                                 static_cast<unsigned long long>(shader_identity),
                                                 set, r.binding, r.min_lod, r.max_lod, r.lod_bias,
                                                 r.mag_filter, r.min_filter, r.mip_filter,
                                                 prosper::test::persistent_color_target_cache().size(),
                                                 prosper::test::persistent_color_target_bytes() /
                                                     (1024.0 * 1024.0),
                                                 prosper::test::persistent_color_target_count_limit(),
                                                 prosper::test::persistent_color_target_limit() /
                                                     (1024.0 * 1024.0));
                                    for (uint32_t level = 0; level < mip_chain.level_count; ++level)
                                        std::fprintf(stderr, "%s0x%llx", level ? "," : "",
                                                     static_cast<unsigned long long>(
                                                         mip_chain.level_ids[level]));
                                    std::fprintf(stderr, "\n");
                                }
                            }
                            resource_rtt_hit = true;
                        } else if (has_uniform_live_rtt) {
                            fr.has_uniform_color = true;
                            fr.uniform_color = live_rtt->second.uniform_color;
                            fr.tw = live_rtt->second.w;
                            fr.th = live_rtt->second.h;
                            fr.td = 1;
                            fr.img_dim = r.img_dim;
                            fr.texture_format = live_rtt->second.format;
                            resource_rtt_hit = true;
                        } else if (resource_compute_image_hit) {
                            fr.borrowed_compute_image = compute_image_import.image;
                            fr.borrowed_compute_device = compute_image_import.device;
                            fr.borrowed_compute_image_layout = compute_image_import.layout;
                            fr.borrowed_compute_image_lease =
                                std::move(compute_image_import.lease);
                            fr.borrowed_compute_vertical_stack_layers =
                                compute_image_import.vertical_stack_layers;
                            fr.tw = compute_image_import.width;
                            fr.th = compute_image_import.height *
                                std::max(compute_image_import.vertical_stack_layers, 1u);
                            fr.td = compute_image_import.vertical_stack_layers
                                ? 1u : compute_image_import.depth;
                            fr.img_dim = r.img_dim;
                            fr.texture_format = compute_image_format;
                        } else if (has_ds_live) {
                            fr.persistent_depth_target_id = r.gpu_addr;
                            fr.tw = sampled_ds.width;
                            fr.th = sampled_ds.height;
                            fr.td = 1; fr.img_dim = r.img_dim;
                            resource_rtt_hit = true;
                            if (PROSPER_ENV_ON("PROSPER_DSBRIDGE_LOG")) {
                                static int consumer_logged = 0;
                                if (consumer_logged++ < 24)
                                    fprintf(stderr,
                                            "[dsbridge] consumer draw target=0x%llx samples "
                                            "0x%llx %ux%u cwm=%x scissor=[%d,%d)-[%d,%d) "
                                            "dcmp=%d func=%u\n",
                                            (unsigned long long)draw.color0_base,
                                            (unsigned long long)r.gpu_addr, tw, th,
                                            draw.ps.color_write_mask, draw.ps.scissor_left,
                                            draw.ps.scissor_top, draw.ps.scissor_right,
                                            draw.ps.scissor_bottom, (int)r.depth_compare,
                                            r.depth_compare_func);
                            }
                        } else if (decoded_reuse) {
                            fr.tex_rgba = decoded_reuse->pixels;
                            fr.tex_rgba_owner = decoded_reuse->pixels_owner;
                            fr.tw = tw;
                            fr.th = decoded_reuse->output_height;
                            // #325: the layer count travels with the decoded pixels. Without this
                            // the hot path -- a cache hit, which is how a world-texture atlas is
                            // served on all but its first reference -- published a single-layer view
                            // over a buffer holding every slice, so the shader could only ever read
                            // slice 0 however correct the decode and the SPIR-V were.
                            // Guarded on `is_array` so this cannot touch the MSAA plane-array
                            // resources that share the channel: for those, sample_count is already
                            // established by the resource setup, and republishing a cached value
                            // would be a behaviour change nobody asked for.
                            if (is_array) {
                                fr.sample_count = decoded_reuse->layers;
                                fr.tex_byte_size = decoded_reuse->pixels_bytes;
                            }
                            // #2998: does the GUEST fill high slices after we decoded the atlas?
                            // The decode happens once and is then served from cache, so a guest that
                            // streams slices in later is invisible to every instrument that samples
                            // the DECODE. This samples guest memory directly on the reuse path,
                            // which runs on every reference, and reports the first time a high slice
                            // becomes non-zero. If that ever fires, the cache is serving stale
                            // pixels and the decode simply happened too early.
                            // #2998: what BACKS this atlas? If the guest maps texture data in
                            // (a file mapping, or several separate mappings), a short or partial
                            // mapping explains an allocation whose tail reads zero far better than
                            // "the game never wrote it". Printed once per address, from the host
                            // process's own map, because the guest VA is mapped into it.
                            if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_SLICEMAPS")) {
                                static std::set<uint64_t> shown;
                                if (shown.insert(r.gpu_addr).second) {
                                    const uint64_t lo = r.gpu_addr;
                                    const uint64_t hi = r.gpu_addr +
                                        (uint64_t)r.layer_stride_bytes * r.depth;
                                    FILE* m = fopen("/proc/self/maps", "r");
                                    char line[512];
                                    uint32_t printed = 0, overlapping = 0;
                                    while (m && fgets(line, sizeof line, m)) {
                                        unsigned long long a = 0, b = 0;
                                        if (sscanf(line, "%llx-%llx", &a, &b) != 2) continue;
                                        if (b <= lo || a >= hi) continue;
                                        ++overlapping;
                                        if (printed >= 8u) continue;   // counted below, never hidden
                                        fprintf(stderr, "[slicemaps] 0x%llx..0x%llx overlaps: %s",
                                                (unsigned long long)lo, (unsigned long long)hi, line);
                                        ++printed;
                                    }
                                    // The COUNT is the measurement -- a span split across many
                                    // mappings is precisely the partial-residency case, and a
                                    // silently truncated list would read as "one clean mapping".
                                    if (overlapping > printed)
                                        fprintf(stderr, "[slicemaps]   ... %u overlapping mapping(s) "
                                                        "total, %u shown\n", overlapping, printed);
                                    if (m) fclose(m);
                                    if (!printed)
                                        fprintf(stderr, "[slicemaps] 0x%llx..0x%llx has NO mapping "
                                                        "in /proc/self/maps\n",
                                                (unsigned long long)lo, (unsigned long long)hi);
                                }
                            }
                            // #2998: a coarse occupancy map of the GUEST allocation itself, in
                            // 256 KiB buckets, independent of any stride assumption. Every previous
                            // measurement addressed slices with layer_stride, so a wrong stride made
                            // "slice N is zero" trivially true by probing outside the data. This
                            // asks the only question that needs no layout at all: WHERE in this
                            // allocation is there content?
                            if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_OCCUPANCY")) {
                                static std::set<uint64_t> done;
                                if (done.insert(r.gpu_addr).second) {
                                    const uint64_t span = (uint64_t)r.layer_stride_bytes * r.depth;
                                    const uint64_t bucket = 256u * 1024u;
                                    const uint64_t n = span / bucket;
                                    if (!n) {
                                        // filled=0/0 read as "scanned and empty". It is neither.
                                        fprintf(stderr,
                                                "[occupancy] 0x%llx span=%llu bytes -- NO complete "
                                                "256KiB bucket (layer_stride=%u depth=%u); "
                                                "NOTHING SCANNED\n",
                                                (unsigned long long)r.gpu_addr,
                                                (unsigned long long)span,
                                                r.layer_stride_bytes, r.depth);
                                        fflush(stderr);
                                    } else {
                                    // Scan every byte of each bucket, not a head sample. This probed
                                    // only the first 512 B of each 256 KiB bucket (0.195%), so a
                                    // bucket whose content began past that window read as EMPTY --
                                    // a false negative that reached TOMB_RAIDER_STATUS.md as
                                    // "not in this allocation at ANY layout".
                                    const uint64_t scan_n = n < 512u ? n : 512u;
                                    std::vector<uint8_t> probe(64u * 1024u);
                                    std::string map;
                                    uint64_t filled = 0, last_filled = 0, unreadable = 0;
                                    for (uint64_t i = 0; i < scan_n; ++i) {
                                        bool any = false, short_read = false;
                                        for (uint64_t off = 0; off < bucket && !any; off += probe.size()) {
                                            const size_t want = (size_t)std::min<uint64_t>(
                                                probe.size(), bucket - off);
                                            const size_t got = copy_resource(
                                                probe.data(), r.gpu_addr + i * bucket + off, want);
                                            for (size_t k = 0; k < got && !any; ++k)
                                                any = probe[k] != 0;
                                            if (got < want) { short_read = true; break; }
                                        }
                                        if (any) { ++filled; last_filled = i; }
                                        else if (short_read) ++unreadable;
                                        // '?' is NOT '.': prosper could not read it, so it is
                                        // unknown, not known-empty.
                                        map += any ? '#' : (short_read ? '?' : '.');
                                    }
                                    fprintf(stderr,
                                            "[occupancy] 0x%llx span=%llu MiB bucket=256KiB "
                                            "FULL-BUCKET scan filled=%llu/%llu scanned "
                                            "(of %llu total) last_filled_bucket=%llu "
                                            "unreadable=%llu ('?' = unreadable, NOT empty)\n%s\n",
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)(span >> 20),
                                            (unsigned long long)filled,
                                            (unsigned long long)scan_n, (unsigned long long)n,
                                            (unsigned long long)last_filled,
                                            (unsigned long long)unreadable, map.c_str());
                                    fflush(stderr);
                                    }
                                }
                            }
                            if (is_array && r.depth > 64u && PROSPER_ENV_ON("PROSPER_SLICEWATCH")) {
                                // Per-address, not global: with two arrays in flight a global flag
                                // lets the first silence the second's report entirely.
                                static std::map<uint64_t, uint64_t> watched_by_addr;
                                static std::set<uint64_t> said_zero, said_nz;
                                uint64_t& watched = watched_by_addr[r.gpu_addr];
                                const uint64_t stride = r.layer_stride_bytes
                                    ? r.layer_stride_bytes
                                    : (uint64_t)tw * th;  // conservative
                                const uint64_t probe_slice = r.depth - 8u;
                                const uint64_t addr = r.gpu_addr + probe_slice * stride;
                                uint8_t buf[256] = {0};
                                const size_t got = copy_resource(buf, addr, sizeof buf);
                                bool nz = false;
                                for (size_t k = 0; k < got && !nz; ++k) nz = buf[k] != 0;
                                ++watched;
                                if (!nz && said_zero.insert(r.gpu_addr).second) {
                                    fprintf(stderr, "[slicewatch] slice %llu of 0x%llx reads ZERO "
                                                    "(got=%zu of %zu) on reference %llu%s\n",
                                            (unsigned long long)probe_slice,
                                            (unsigned long long)r.gpu_addr, got, sizeof buf,
                                            (unsigned long long)watched,
                                            r.layer_stride_bytes ? "" : " [probe used the tw*th "
                                                                        "stride FALLBACK, not a "
                                                                        "guest-declared stride]");
                                }
                                if (nz && said_nz.insert(r.gpu_addr).second) {
                                    // A transition needs a PRIOR zero reading for this address.
                                    // Without one this is just the first observation, and saying
                                    // "the guest filled it after our decode" would assert an
                                    // ordering nothing here measured.
                                    const bool saw_zero_first = said_zero.count(r.gpu_addr) != 0;
                                    fprintf(stderr,
                                            saw_zero_first
                                              ? "[slicewatch] slice %llu of 0x%llx BECAME NON-ZERO "
                                                "on reference %llu -- it read ZERO earlier, so the "
                                                "guest filled it after that reading%s\n"
                                              : "[slicewatch] slice %llu of 0x%llx reads NON-ZERO on "
                                                "reference %llu -- FIRST observation, no prior ZERO "
                                                "reading, so this is NOT an observed transition%s\n",
                                            (unsigned long long)probe_slice,
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)watched,
                                            r.layer_stride_bytes ? "" : " [probe used the tw*th "
                                                                        "stride FALLBACK, not a "
                                                                        "guest-declared stride]");
                                }
                            }
                            fr.td = is_volume ? r.depth : 1u;
                            fr.img_dim = r.img_dim;
                            fr.texture_format = decoded_reuse->texture_format;
                            fr.storage_image_contract_valid =
                                decoded_reuse->storage_image_contract_valid;
                            // #1272: plain 2D guest textures only — cube outputs stack 6 faces into
                            // one 2D image (fr.th != th), and volumes keep their own path; generated
                            // mips across face/slice boundaries would bleed.
                            if (!is_volume && fr.th == th)
                                fr.declared_mip_levels = r.declared_mip_levels;
                            narrow_done = decoded_reuse->narrow;
                            fr.persistent_texture_id = decoded_reuse->persistent_id;
                            fr.persistent_texture_version = decoded_reuse->persistent_version;
                            if (PROSPER_ENV_ON("PROSPER_DETILE_STATS") && resource_persistent_hit) {
                                static uint64_t bc_cube_hit_total = 0;
                                static std::unordered_map<uint64_t, uint64_t> bc_cube_hits;
                                const uint64_t hit_footprint =
                                    prosper::gpu::gpu_capture_resource_footprint(r);
                                const bool expensive_bc_cube = r.img_dim == 3u &&
                                    texture_decode_miss_is_expensive_block(
                                        r.format == prosper::gpu::DataFormat::Bc6,
                                        persistent_source_size,
                                        static_cast<size_t>(std::min<uint64_t>(
                                            hit_footprint, SIZE_MAX)));
                                if (expensive_bc_cube) {
                                    const uint64_t address_hit_ordinal = ++bc_cube_hits[r.gpu_addr];
                                    const uint64_t global_hit_ordinal = address_hit_ordinal == 1u
                                        ? ++bc_cube_hit_total : bc_cube_hit_total;
                                    if (should_report_texture_decode_hit(
                                            global_hit_ordinal, address_hit_ordinal,
                                            expensive_bc_cube)) {
                                        fprintf(stderr,
                                                "[detile-hit] ordinal=%llu address-ordinal=%llu "
                                                "addr=0x%llx key=0x%zx %ux%ux%u dim=%u fmt=%u/%u "
                                                "tile=%u footprint=%llu source=%zu "
                                                "cache=persistent-hit id=%llu version=%llu "
                                                "validation=%s validated=%zu submit-query=%d "
                                                "watch-query=%d watch-active=%d watch-only=%d\n",
                                                (unsigned long long)global_hit_ordinal,
                                                (unsigned long long)address_hit_ordinal,
                                                (unsigned long long)r.gpu_addr,
                                                TextureDecodeKeyHash{}(decode_key),
                                                tw, th, r.depth, r.img_dim,
                                                static_cast<unsigned>(r.format), r.num_components,
                                                r.tile_mode,
                                                (unsigned long long)hit_footprint,
                                                persistent_source_size,
                                                (unsigned long long)fr.persistent_texture_id,
                                                (unsigned long long)fr.persistent_texture_version,
                                                resource_texture_exact_validation ? "exact" : "skip",
                                                resource_texture_validated_bytes,
                                                resource_texture_submit_query,
                                                resource_texture_watch_query,
                                                static_cast<int>(resource_texture_watch_active),
                                                static_cast<int>(resource_texture_watch_only));
                                        fflush(stderr);
                                    }
                                }
                            }
                            // Only a persistent-cache hit needs to be recorded here; a submit-local
                            // hit is already in the map under this key, and emplacing over it would
                            // build and discard a node on every repeat reference. Persistent-hit
                            // pixels are owned by the persistent entry, so the bytes live in storage
                            // the scratch allocator never recycles and no pin is needed; the retained
                            // range is the one that cache itself validates for this identity.
                            if (!resource_local_reuse)
                                decoded_textures.insert_or_assign(
                                    decode_key,
                                    DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                                   fr.persistent_texture_id,
                                                   fr.persistent_texture_version,
                                                   decode_span_ordinal,
                                    prosper::gpu::guest_gpu_write_snapshot(),
                                    persistent_source_addr,
                                    cross_span_source_size, SIZE_MAX,
                                    fr.tex_rgba_owner,
                                                   fr.texture_format,
                                                   fr.storage_image_contract_valid,
                                                   fr.sample_count, fr.tex_byte_size});
                            if (timing_enabled) pending_timing.texture_reuses++;
                        } else {
                        // PROSPER_DETILE_STATS: this branch is the texture-decode MISS path — the cache
                        // had no entry for decode_key, so we are about to read guest memory and CPU-detile
                        // this surface. Counting decodes per guest address answers the #1177 redundancy
                        // question: if a handful of addresses accumulate huge counts, the same immutable
                        // surface is re-detiled every frame (a cache-key instability / live-RTT re-decode);
                        // if counts stay near 1 the cost is a large one-time working set instead.
                        if (PROSPER_ENV_ON("PROSPER_DETILE_STATS")) {
                            struct DecodeShape {
                                uint32_t width = 0, height = 0, depth = 0;
                                uint32_t image_dimension = 0, format = 0, components = 0;
                                uint32_t tile_mode = 0, declared_bytes = 0;
                            };
                            struct DecodeAddressState {
                                size_t last_key_hash = 0;
                                uint64_t key_changes = 0;
                                bool has_key = false;
                            };
                            static std::unordered_map<uint64_t, uint64_t> ds_count;  // guest addr -> times decoded
                            static std::unordered_map<uint64_t, DecodeShape> ds_shape;
                            static std::unordered_map<uint64_t, DecodeAddressState> ds_address_state;
                            static uint64_t ds_total = 0;
                            // Classify WHY this decode reached the miss path (so the redundancy is actionable):
                            // rtt          - a renderer-owned RTT decoded on the CPU (has_live_rtt)
                            // notsampled   - not a candidate class (host data / format / unsupported dim)
                            // compression  - DCC-compressed source whose metadata is unsupported
                            // size0        - persistent_source_size==0 (force-detiled Unorm8x4)
                            // invalidated  - eligible AND a cache entry exists, but validate_exact() rejected it
                            // cold         - eligible but no cache entry (first use / LRU-evicted)
                            static uint64_t r_rtt = 0, r_notsampled = 0, r_compression = 0,
                                            r_size0 = 0, r_inval = 0, r_cold = 0, r_other = 0;
                            // For the rtt bucket, record WHICH has_gpu_live_rtt condition failed (so a
                            // registered render target that could be sampled straight from the GPU image
                            // instead fell back to CPU detile). storage/notvalid/dimmismatch/self/noptarget.
                            static uint64_t rtt_storage = 0, rtt_notvalid = 0, rtt_dimmismatch = 0,
                                            rtt_self = 0, rtt_noptarget = 0, rtt_unknown = 0;
                            // For the notsampled bucket, record WHY the guest texture is not a persistent-
                            // cache candidate (host data / unsupported dimension / class / format) plus a
                            // (format<<4|components) histogram of the format-excluded
                            // ones, to see whether FP16/HDR art dominates (a cache-eligibility gap, #1177).
                            static uint64_t ns_hostdata = 0, ns_notdim1 = 0, ns_notclass = 0, ns_fmt = 0;
                            static std::unordered_map<uint32_t, uint64_t> ns_fmt_hist;
                            const uint64_t address_ordinal = ++ds_count[r.gpu_addr];
                            ds_shape[r.gpu_addr] = {
                                tw, th, r.depth, r.img_dim, static_cast<uint32_t>(r.format),
                                r.num_components, r.tile_mode, r.size};
                            const uint64_t global_ordinal = ++ds_total;
                            const size_t key_hash = TextureDecodeKeyHash{}(decode_key);
                            DecodeAddressState& address_state = ds_address_state[r.gpu_addr];
                            const bool key_changed = address_state.has_key &&
                                address_state.last_key_hash != key_hash;
                            if (key_changed) ++address_state.key_changes;
                            const size_t previous_key_hash = address_state.last_key_hash;
                            address_state.last_key_hash = key_hash;
                            address_state.has_key = true;

                            const auto matching_entry = persistent_decoded_textures.find(decode_key);
                            const bool matching_cache_entry =
                                matching_entry != persistent_decoded_textures.end() &&
                                matching_entry->second.source_addr == persistent_source_addr &&
                                matching_entry->second.source_size == persistent_source_size;
                            const bool compression_supported = !r.compression_enabled ||
                                persistent_dcc_uncompressed || persistent_dcc_fast_clear;
                            const bool cache_disabled =
                                PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_DECODE_CACHE") != nullptr;
                            const TextureDecodeMissReason miss_reason = texture_decode_miss_reason(
                                has_live_rtt, persistent_sampled_texture, compression_supported,
                                persistent_source_size, cache_disabled, persistent_decode_limit,
                                matching_cache_entry, persistent_cache_eligible);
                            if (miss_reason == TextureDecodeMissReason::LiveRenderTarget) {
                                r_rtt++;
                                if (fr.is_storage_image) rtt_storage++;
                                else if (!live_rtt->second.gpu_valid) rtt_notvalid++;
                                else if (live_rtt->second.w != tw || live_rtt->second.h != th)
                                    rtt_dimmismatch++;
                                else if (draw_binds_color_target(draw, sampled_source_addr, tw, th))
                                    rtt_self++;
                                else if (prosper::test::find_persistent_color_target(
                                             sampled_source_addr, tw, th,
                                             live_rtt->second.format) == nullptr)
                                    rtt_noptarget++;   // evicted from / absent in the backend target cache
                                else rtt_unknown++;
                            }
                            else if (miss_reason == TextureDecodeMissReason::UnsupportedCandidate) {
                                r_notsampled++;
                                if (r.host_data) ns_hostdata++;
                                else if (r.img_dim != 1u) ns_notdim1++;
                                else if (r.cls != RC::Texture) ns_notclass++;
                                else { ns_fmt++;
                                    ns_fmt_hist[((uint32_t)r.format << 4) |
                                                (r.num_components & 0xFu)]++; }
                            }
                            else if (miss_reason == TextureDecodeMissReason::UnsupportedCompression)
                                r_compression++;
                            else if (miss_reason == TextureDecodeMissReason::EmptySource)
                                r_size0++;
                            else if (miss_reason == TextureDecodeMissReason::ContentInvalidated)
                                r_inval++;
                            else if (miss_reason == TextureDecodeMissReason::ColdOrEvicted)
                                r_cold++;
                            else
                                r_other++;

                            const uint64_t diagnostic_footprint =
                                prosper::gpu::gpu_capture_resource_footprint(r);
                            const size_t diagnostic_footprint_size = static_cast<size_t>(
                                std::min<uint64_t>(diagnostic_footprint, SIZE_MAX));
                            const bool expensive_bc6 = texture_decode_miss_is_expensive_block(
                                r.format == prosper::gpu::DataFormat::Bc6,
                                persistent_source_size, diagnostic_footprint_size);
                            if (should_report_texture_decode_miss(
                                    global_ordinal, address_ordinal, expensive_bc6)) {
                                auto reason_name = [](TextureDecodeMissReason reason) {
                                    switch (reason) {
                                        case TextureDecodeMissReason::LiveRenderTarget: return "rtt";
                                        case TextureDecodeMissReason::UnsupportedCandidate:
                                            return "unsupported-candidate";
                                        case TextureDecodeMissReason::UnsupportedCompression:
                                            return "unsupported-compression";
                                        case TextureDecodeMissReason::EmptySource: return "empty-source";
                                        case TextureDecodeMissReason::CacheDisabled:
                                            return "cache-disabled";
                                        case TextureDecodeMissReason::CacheLimitZero:
                                            return "cache-limit-zero";
                                        case TextureDecodeMissReason::ContentInvalidated:
                                            return "content-invalidated";
                                        case TextureDecodeMissReason::ColdOrEvicted:
                                            return "cold-or-evicted";
                                        case TextureDecodeMissReason::Other: return "other";
                                    }
                                    return "unknown";
                                };
                                fprintf(stderr,
                                        "[detile-miss] ordinal=%llu address-ordinal=%llu "
                                        "addr=0x%llx key=0x%zx previous-key=0x%zx "
                                        "key-changes=%llu reason=%s %ux%ux%u dim=%u fmt=%u/%u "
                                        "tile=%u declared=%u footprint=%llu source=%zu "
                                        "candidate=%d eligible=%d entry=%d "
                                        "validation=%s validated=%zu submit-query=%d "
                                        "watch-query=%d watch-active=%d watch-disabled=%d "
                                        "watch-only=%d watch-stable=%u\n",
                                        (unsigned long long)global_ordinal,
                                        (unsigned long long)address_ordinal,
                                        (unsigned long long)r.gpu_addr, key_hash,
                                        previous_key_hash,
                                        (unsigned long long)address_state.key_changes,
                                        reason_name(miss_reason), tw, th, r.depth, r.img_dim,
                                        static_cast<unsigned>(r.format), r.num_components,
                                        r.tile_mode, r.size,
                                        (unsigned long long)diagnostic_footprint,
                                        persistent_source_size,
                                        static_cast<int>(persistent_sampled_texture),
                                        static_cast<int>(persistent_cache_eligible),
                                        static_cast<int>(matching_cache_entry),
                                        resource_texture_exact_validation ? "exact" : "skip",
                                        resource_texture_validated_bytes,
                                        resource_texture_submit_query,
                                        resource_texture_watch_query,
                                        static_cast<int>(resource_texture_watch_active),
                                        static_cast<int>(resource_texture_watch_disabled),
                                        static_cast<int>(resource_texture_watch_only),
                                        resource_texture_watch_stability);
                                fflush(stderr);
                            }
                            if ((ds_total % 3000) == 0) {
                                std::vector<std::pair<uint64_t, uint64_t>> v(ds_count.begin(), ds_count.end());
                                std::sort(v.begin(), v.end(),
                                          [](auto& a, auto& b) { return a.second > b.second; });
                                fprintf(stderr, "[detile-stats] %llu decodes, %zu distinct addrs; top re-decoded:",
                                        (unsigned long long)ds_total, ds_count.size());
                                for (int i = 0; i < 8 && i < (int)v.size(); i++) {
                                    const DecodeShape& shape = ds_shape[v[i].first];
                                    fprintf(stderr,
                                            " 0x%llx x%llu[%ux%ux%u dim=%u fmt=%u/%u tile=%u "
                                            "decl=%u]",
                                            (unsigned long long)v[i].first,
                                            (unsigned long long)v[i].second,
                                            shape.width, shape.height, shape.depth,
                                            shape.image_dimension, shape.format, shape.components,
                                            shape.tile_mode, shape.declared_bytes);
                                }
                                fprintf(stderr, "\n[detile-stats] miss reasons: rtt=%llu notsampled=%llu "
                                        "compression=%llu size0=%llu invalidated=%llu cold=%llu other=%llu\n",
                                        (unsigned long long)r_rtt, (unsigned long long)r_notsampled,
                                        (unsigned long long)r_compression, (unsigned long long)r_size0,
                                        (unsigned long long)r_inval, (unsigned long long)r_cold,
                                        (unsigned long long)r_other);
                                fprintf(stderr, "[detile-stats] rtt-fail: storage=%llu notvalid=%llu "
                                        "dimmismatch=%llu self=%llu noptarget=%llu unknown=%llu\n",
                                        (unsigned long long)rtt_storage, (unsigned long long)rtt_notvalid,
                                        (unsigned long long)rtt_dimmismatch, (unsigned long long)rtt_self,
                                        (unsigned long long)rtt_noptarget, (unsigned long long)rtt_unknown);
                                std::vector<std::pair<uint32_t, uint64_t>> fh(ns_fmt_hist.begin(),
                                                                             ns_fmt_hist.end());
                                std::sort(fh.begin(), fh.end(),
                                          [](auto& a, auto& b) { return a.second > b.second; });
                                fprintf(stderr, "[detile-stats] notsampled-why: hostdata=%llu notdim1=%llu "
                                        "notclass=%llu fmt=%llu; top fmt/comp:",
                                        (unsigned long long)ns_hostdata, (unsigned long long)ns_notdim1,
                                        (unsigned long long)ns_notclass, (unsigned long long)ns_fmt);
                                for (int i = 0; i < 6 && i < (int)fh.size(); i++)
                                    fprintf(stderr, " fmt=%u/comp=%u x%llu",
                                            fh[i].first >> 4, fh[i].first & 0xF,
                                            (unsigned long long)fh[i].second);
                                fprintf(stderr, "\n");
                                fflush(stderr);
                            }
                        }
                        const size_t volume_texels = (size_t)tw * th * (is_volume ? r.depth : 1u);
                        fr.texture_format = decoded_texture_format;
                        const uint32_t output_bpp =
                            prosper::test::backend_color_bytes_per_pixel(decoded_texture_format);
                        size_t nb = volume_texels * output_bpp *
                            (is_cube ? 6u : (is_array ? decoded_layers : 1u));
                        size_t linear_source_prefix_size = 0;
                        ++g_texture_decode_scope.decodes;
                        const size_t texture_slot = acquire_texstore_slot();
                        std::vector<uint8_t>& texture_pixels = texstore[texture_slot];
                        // Set false once the persistent cache takes ownership of these bytes; while it
                        // is true a retained identity entry must pin `texture_slot`, or the next span
                        // would decode an unrelated texture into the very slot it points at.
                        bool decoded_pixels_in_texstore = false;
                        if (retain_cpu_live_rtt) texture_pixels.clear();
                        else texture_pixels.resize(nb);
                        auto copy_linear_padded_rows = [&](uint8_t* dst, size_t dst_row,
                                                           uint32_t rows) {
                            size_t total = 0;
                            for (uint32_t y = 0; y < rows; ++y) {
                                uint8_t* drow = dst + (size_t)y * dst_row;
                                const size_t got = copy_resource(
                                    drow, sampled_source_addr + (uint64_t)y * linear_src_row,
                                    dst_row);
                                total += got;
                                if (got < dst_row) std::fill(drow + got, drow + dst_row, 0);
                            }
                            return total;
                        };
                        // #325: the same padded-row read, but from an arbitrary base, so an ARRAY
                        // slice can use it. A linear array's rows are pitch-padded exactly as a
                        // single linear surface's are -- `linear_padded_read` already names
                        // img_dim 5 -- and the slice loop was reading them flat, which is what made
                        // slice 0 of a linear BC array decode differently than it used to.
                        auto copy_linear_padded_rows_from = [&](uint8_t* dst, uint64_t base,
                                                                size_t dst_row, uint32_t rows) {
                            size_t total = 0;
                            for (uint32_t y = 0; y < rows; ++y) {
                                uint8_t* drow = dst + (size_t)y * dst_row;
                                const size_t got = copy_resource(
                                    drow, base + (uint64_t)y * linear_src_row, dst_row);
                                total += got;
                                if (got < dst_row) std::fill(drow + got, drow + dst_row, 0);
                            }
                            return total;
                        };
                        // PROSPER_TEXCOMMIT: log, once per texture base, how much of the sampled surface is
                        // COMMITTED guest memory (the same reserved_range_state safe_copy stops at). If the
                        // level's backgrounds read ~0% committed, they're GPU-DMA'd pages the CPU never
                        // touched, so we read zeros -> the scene samples black (#300 black-gameplay probe).
                        if (PROSPER_ENV_ON("PROSPER_TEXCOMMIT")) {
                            // Walk the SOURCE extent, never `nb` (the DECODED byte count computed just
                            // above for `texture_pixels`): for a block-compressed format `nb` is 4x+ the
                            // real guest bytes (BC7 decodes 1 source byte/texel to 4 RGBA8 bytes), and an
                            // array multiplies that again by the layer count. Scanning `nb` over
                            // `sampled_source_addr` both mis-measured "committed" (it counted whatever
                            // guest memory happened to follow the real texture) and read guest memory
                            // past the real allocation (#3053). `gpu_capture_resource_footprint()` is the
                            // same T#-driven source-byte computation the capture/replay path already uses
                            // for this resource, so it is correct here too regardless of format.
                            const size_t src_bytes = texcommit_scan_extent(
                                prosper::gpu::gpu_capture_resource_footprint(r));
                            const size_t PG = 0x10000; size_t committed = 0;
                            for (uint64_t a = sampled_source_addr;
                                 a < sampled_source_addr + src_bytes; a += PG)
                                if (a >= 0x1000 && prosper_reserved_range_state(a) != 0) committed += PG;
                            static std::set<uint64_t> tcseen;
                            if (tcseen.insert(r.gpu_addr).second) {
                                // Also sample the first 8 dwords and count non-zero bytes over the whole
                                // SOURCE surface (src_bytes, not nb): zero content => the texture was
                                // allocated but never filled (a GPU-side upload/copy we don't execute);
                                // non-zero => it's a decode/tiling problem. tile_mode tells tiled vs linear.
                                uint32_t w0[8] = {0}; size_t nzb = 0;
                                if (committed) {
                                    const uint8_t* p =
                                        (const uint8_t*)(uintptr_t)sampled_source_addr;
                                    for (int i = 0; i < 8; i++) w0[i] = ((const uint32_t*)p)[i];
                                    for (size_t i = 0; i < src_bytes; i += 997) nzb += (p[i] != 0);   // sparse scan
                                }
                                fprintf(stderr, "[texcommit] tex 0x%llx %ux%u f%u tile=%u nb=%zu src=%zu "
                                        "committed=%zu%% nz~%zu/%zu first=%08x %08x %08x %08x\n",
                                        (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format, r.tile_mode,
                                        nb, src_bytes,
                                        src_bytes ? (size_t)(100 * committed / src_bytes) : 100, nzb,
                                        src_bytes / 997,
                                        w0[0], w0[1], w0[2], w0[3]);
                            }
                        }
                        // Real bytes-per-texel of the SAMPLED surface. Single/dual-channel textures (an R8
                        // font/coverage atlas — this game's 2048x1024 c1 surface) are NOT 4 B/texel; reading
                        // them as RGBA8 packs adjacent texels into one pixel and over-reads the allocation,
                        // which is why glyph text rendered as solid white/black BLOCKS (#102). bpt drives a
                        // narrow read+expand path below. StorageImage / unknown formats keep 4 B (bpt=0->4).
                        uint32_t bpt = sampled_source_bpt;
                        // fp16/fp32 surfaces use their real source element size below. Guest fp16 keeps
                        // the historical UNORM8 conversion, while fp32 narrows to native RGBA16F so small
                        // values and HDR range survive. Renderer-owned RTTs bypass both conversions.
                        const bool f16 = sampled_source_f16;
                        const bool f32 = sampled_source_f32;
                        bool f16_done = false, f32_done = false;
                        // RTT (#167): if this texture's base is a color target we rendered into, inject those
                        // pixels (nearest-scaled to tw x th) instead of reading empty guest memory.
                        // Direct compute imports already own the complete sampled image. Mark the
                        // CPU materializer complete as well; otherwise it performs a discarded
                        // guest decode (and, for a cube, synchronous retained-depth readbacks)
                        // before the backend binds the borrowed image.
                        bool rtt_hit = resource_compute_image_hit;
                        // Why a sampled resource never consulted the renderer-owned RTT cache. Without
                        // this, a draw that reads a target prosper rendered but takes the guest-decode
                        // path instead is invisible in the log: no "sample tex" line is emitted at all,
                        // and the draw silently samples empty guest memory.
                        if (rtt_log && (fr.is_storage_image || !rtt_on || is_volume || r.in_mip_tail))
                            fprintf(stderr,
                                    "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> RTT PATH SKIPPED "
                                    "(storage=%d rtt_on=%d volume=%d mip_tail=%d)\n",
                                    (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format,
                                    (int)fr.is_storage_image, (int)rtt_on, (int)is_volume,
                                    (int)r.in_mip_tail);
                        if (!fr.is_storage_image && rtt_on && !is_volume && !r.in_mip_tail) {
                            auto rit = g_rtt.find(sampled_source_addr);
                            if (rit != g_rtt.end() && rit->second.w && rit->second.h && rit->second.rgba &&
                                !rit->second.rgba->empty()) {
                                const RttSurf& s = rit->second;
                                const uint32_t rtt_bpp = prosper::test::backend_color_bytes_per_pixel(s.format);
                                if (retain_cpu_live_rtt &&
                                    (fr.tex_rgba_owner = rtt_injection_cache.materialize(
                                         s.rgba, tw, th, s.w, s.h, rtt_bpp))) {
                                    fr.tex_rgba = fr.tex_rgba_owner->data();
                                    fr.texture_format = s.format;
                                    rtt_hit = true;
                                    resource_rtt_hit = true;
                                } else if (prosper::frontend::inject_rtt_pixels(
                                               texture_pixels, tw, th, *s.rgba,
                                               s.w, s.h, rtt_bpp)) {
                                    fr.texture_format = s.format;
                                    rtt_hit = true;
                                    resource_rtt_hit = true;
                                    // PROSPER_DUMP_SAMPLED_RTT (#710/#320): dump the exact RTT-layer
                                    // pixels a draw samples, disambiguating a dark layer from a later
                                    // composite/tint that darkens otherwise-correct input.
                                    if (PROSPER_ENV_ON("PROSPER_DUMP_SAMPLED_RTT")) {
                                        static std::set<uint64_t> seen;
                                        if (seen.insert(sampled_source_addr).second) {
                                            const std::vector<uint8_t> inspected = inspection_rgba8(
                                                texture_pixels, tw, th, s.format);
                                            size_t nz = 0, rgbnz = 0;
                                            for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                                                if (inspected[p] || inspected[p+1] || inspected[p+2]) rgbnz++;
                                                for (int k = 0; k < 4; k++) nz += (inspected[p+k] != 0);
                                            }
                                            const char* dd = getenv("PROSPER_FRAME_DIR");
                                            char fn[512]; snprintf(fn, sizeof fn, "%s/sampledrtt_%llx_%ux%u.bmp",
                                                                   dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th);
                                            prosper::test::dump_bmp(fn, inspected, tw, th);
                                            fprintf(stderr, "[sampledrtt] addr=0x%llx %ux%u rgb_nonblack=%zu/%u -> %s\n",
                                                    (unsigned long long)sampled_source_addr,
                                                    tw, th, rgbnz, tw*th, fn);
                                        }
                                    }
                                } // malformed/incomplete RTT bytes => miss; decode guest backing below
                            }
                            // Report HOW the binding resolved, not merely whether the COLOUR cache
                            // had it. This line used to print "miss" for a resource the depth or
                            // stencil bridge had already resolved, because `rtt_hit` speaks only for
                            // g_rtt -- so a correctly-bridged 4K depth buffer and a texture decoded
                            // from guest zeros were the same word. Reading a pass's inputs off this
                            // log therefore over-counted missing inputs, which is the one thing the
                            // log exists to answer.
                            // PROSPER_RTT_GUESTPEEK=1 — for each distinct sampled surface, how
                            // much of the GUEST backing is non-zero, next to how prosper resolved
                            // it. This separates two states that every other signal renders
                            // identically: a surface nothing ever wrote (guest bytes zero, cache
                            // zero) from one the guest DID write through a path prosper did not
                            // observe (guest bytes populated, cache zero) -- the second is a
                            // prosper defect and the first is not, and they call for opposite work.
                            // Read the guest VA directly rather than r.host_data: on this path
                            // host_data is null for every sampled texture (measured: zero peeks in a
                            // full routed run), so keying on it produced complete silence -- which
                            // reads as "the guest backing is empty" and is instead "we never
                            // looked". Guest code runs natively here, so a readability-checked VA is
                            // the authoritative view of that memory.
                            if (rtt_log && PROSPER_ENV_ON("PROSPER_RTT_GUESTPEEK") &&
                                sampled_source_addr &&
                                prosper::gpu::guest_readable(sampled_source_addr, 4096)) {
                                // Re-peek on a power-of-two schedule per address rather than once.
                                // A once-only peek reports the FIRST sighting, and a surface's first
                                // sighting is routinely its emptiest moment -- sampled during
                                // start-up before the guest has written it. That reads as "the guest
                                // never wrote this", which is the exact conclusion the peek exists to
                                // test, so the instrument would confirm it by construction. The
                                // schedule keeps the volume bounded (about 20 lines per surface over
                                // a million samples) while making the steady state observable.
                                static std::mutex peek_mutex;
                                static std::map<uint64_t, uint64_t> peek_counts;
                                uint64_t seen = 0;
                                {
                                    std::lock_guard lock(peek_mutex);
                                    seen = ++peek_counts[sampled_source_addr];
                                }
                                if ((seen & (seen - 1)) == 0) {
                                    size_t window = std::min<size_t>(
                                        prosper::gpu::gpu_capture_resource_footprint(r),
                                        size_t(1) << 20);
                                    while (window &&
                                           !prosper::gpu::guest_readable(
                                               sampled_source_addr,
                                               static_cast<uint32_t>(window)))
                                        window /= 2;
                                    size_t nz = 0;
                                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(
                                        static_cast<uintptr_t>(sampled_source_addr));
                                    for (size_t i = 0; i < window; ++i) nz += bytes[i] != 0;
                                    fprintf(stderr,
                                            "[rtt-guestpeek] addr=0x%llx %ux%u fmt=%u sample#%llu "
                                            "srt=0x%x sgpr=%u pc=%u "
                                            "guest non-zero=%zu/%zu bytes (%.1f%%) resolved=%s\n",
                                            (unsigned long long)sampled_source_addr, tw, th,
                                            (unsigned)r.format, (unsigned long long)seen,
                                            // Descriptor PROVENANCE, which decides whether a wrong
                                            // address can be stale at all. sgpr_base valid means the
                                            // descriptor came from THIS draw's user data and is fresh
                                            // by construction; srt_offset valid means it was loaded
                                            // from a table, which a first-fold read can pin.
                                            r.srt_offset, r.sgpr_base, r.fetch_pc,
                                            nz, window,
                                            window ? 100.0 * nz / window : 0.0,
                                            has_uniform_live_rtt ? "HIT-UNIFORM"
                                                : rtt_hit ? (has_gpu_live_rtt ? "HIT-GPU"
                                                                              : "HIT-CPU")
                                                : has_ds_live ? "DS" : "miss");
                                }
                            }
                            if (rtt_log)
                                fprintf(stderr,
                                        "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> %s "
                                        "(cache_size=%zu)\n",
                                        (unsigned long long)sampled_source_addr, tw, th,
                                        (unsigned)r.format,
                                        // Which hit path, not merely that one hit. A UNIFORM-colour
                                        // surface samples as one flat value everywhere, so a shader
                                        // reading it produces correct geometry with no texture
                                        // detail at all -- visually a smooth gradient inside a
                                        // correct mask. That is indistinguishable from a real hit
                                        // in a log that prints one word for all three paths, and it
                                        // is exactly what GTA V's lighting output looks like.
                                        has_uniform_live_rtt ? "HIT-UNIFORM"
                                            : rtt_hit ? (has_gpu_live_rtt ? "HIT-GPU" : "HIT-CPU")
                                            : has_ds_live
                                                ? (sampled_ds.aspect == VK_IMAGE_ASPECT_STENCIL_BIT
                                                       ? "DS-STENCIL" : "DS-DEPTH")
                                                : "miss",
                                        g_rtt.size());
                        }
                        // CUBE texture (#273 — DOLL's title reflection probes / skybox): decode six
                        // independent thin-2D faces into the stacked w x 6h image addressed by the
                        // cube-sample lowering. A selected mip is not six tightly packed levels: each
                        // face owns a complete aligned mip chain, so layer_stride/mip_offset select the
                        // same level from each face. Packed-tail levels retain each face-chain base and
                        // use the tail coordinates. BCn and native-width fp16 follow the same layout.
                        // GFX8-GFX10 embeds four self-contained 0/1 fast clears in DCC metadata. A
                        // uniform metadata surface means every compression block has that value, so it
                        // can be materialized without interpreting compressed base bytes. Uniform 0xff
                        // means an emulated writer published ordinary uncompressed base texels and the
                        // normal format/detile path below is authoritative.
                        bool dcc_fast_clear_done = false;
                        bool dcc_uncompressed = false;
                        if (r.compression_enabled && !rtt_hit && !fr.is_storage_image) {
                            const uint64_t metadata_bytes = sampled_dcc_metadata_size;
                            const std::vector<uint8_t>& metadata = sampled_dcc_metadata;
                            const size_t metadata_got = sampled_dcc_metadata_got;
                            uint8_t clear_code = 0;
                            dcc_uncompressed = persistent_dcc_uncompressed;
                            if (!dcc_uncompressed && metadata_got == metadata.size() &&
                                prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                                    texture_pixels.data(), texture_pixels.size() / 4,
                                    metadata.data(), metadata.size(), r.num_components,
                                    r.alpha_is_on_msb, &clear_code)) {
                                dcc_fast_clear_done = true;
                                static std::set<std::pair<uint64_t, uint64_t>> decoded_dcc_images;
                                if (decoded_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second)
                                    fprintf(stderr,
                                            "[render] DCC fast-clear addr=0x%llx meta=0x%llx "
                                            "%ux%ux%u fmt=%u code=0x%02x bytes=%zu\n",
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)r.metadata_addr,
                                            tw, th, r.depth, (unsigned)r.format, clear_code,
                                            metadata.size());
                            } else if (!dcc_uncompressed) {
                                static std::set<std::pair<uint64_t, uint64_t>> warned_dcc_images;
                                if (warned_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second) {
                                    // WHICH KIND this metadata is, by positive correlation with
                                    // retained renderer state -- never inferred from the descriptor.
                                    // T# WORD6[21] (`compression_enabled`) is set for depth/stencil
                                    // and colour alike, and AMD PAL puts GetHtile256BAddr() in the
                                    // same `meta_data_address` field it uses for DCC, so the
                                    // descriptor cannot separate them. This branch assumed DCC,
                                    // which is how GTA V's main depth (0x2052ac0000, HTILE at
                                    // 0x2055310000) came to be reported as "DCC-compressed" (#2674).
                                    //
                                    // This is the FIRST production caller of the correlator anywhere
                                    // in the tree. It was easy to assume the compute path already
                                    // used it -- a query is registered for it in this file -- but
                                    // registering a query is not calling one, and on master every
                                    // reference to `classify_compression_metadata_kind` outside its
                                    // own definition is a test.
                                    //
                                    // Classified HERE, inside the first-sighting guard, rather than
                                    // beside the footprint: the query walks the retained DS cache and
                                    // the RTT map, which is O(surfaces) and this path runs per
                                    // sampled texture. Behind the guard it runs at most once per
                                    // (base, metadata) pair for the process, so the hot path is
                                    // unchanged.
                                    //
                                    // Used ONLY TO REPORT. It deliberately does not change which
                                    // footprint is taken or what is authorized: the HTILE helper is
                                    // fail-closed on sample_count != 4 and this surface is
                                    // single-sample, so routing it there today returns 0 exactly as
                                    // the DCC helper does, while changing the path every other title
                                    // takes. Widening that gate needs its own evidence (#2674).
                                    const prosper::gpu::CompressionMetadataKind kind =
                                        prosper::gpu::classify_compression_metadata_kind(
                                            prosper::gpu::MetadataKindRequest{
                                                r.metadata_addr, r.gpu_addr, r.format,
                                                r.num_components, r.img_dim});
                                    // Name the KIND, and print the inputs the sizing helpers
                                    // actually gate on. The previous line called every declined
                                    // surface "DCC-compressed" and omitted `samples` and
                                    // `pipe_aligned` -- two of the three conditions in
                                    // gfx10_htile_msaa_metadata_bytes(). The DCC sizer gates only on
                                    // tile mode and bytes-per-texel; it takes `pipe_aligned` and
                                    // explicitly discards it (`(void)pipe_aligned` in tile.cpp), so
                                    // that field is diagnostic for the HTILE route only. Either way
                                    // the old line could not say WHICH gate failed, and a reader
                                    // chasing a depth surface was sent looking for a DCC defect.
                                    //
                                    // `first=` is printed only when bytes were actually read.
                                    // metadata=0/0 means the footprint was zero, so nothing was
                                    // fetched and the first byte is an unread default, not a
                                    // measurement of the plane -- reporting 0x00 there stated a fact
                                    // about our own buffer as though it were one about the guest.
                                    //
                                    // `ds_live=` disambiguates the TWO independent reasons the
                                    // footprint can be zero, which the old line could not separate:
                                    // the size expression is gated on `!has_ds_live` outright, or the
                                    // sizing helper fail-closed on the shape. Without it,
                                    // "metadata=0/0" is compatible with both and a reader is free to
                                    // pick the one that suits their hypothesis -- which is how a gate
                                    // analysis gets written against the wrong gate. One was.
                                    //
                                    // Read `ds_live=0` as "the size expression called a helper", NOT
                                    // as "no retained DS image exists". The lookup above is itself
                                    // gated on `!has_live_rtt && !is_storage_image && img_dim == 1 &&
                                    // cls == Texture`, so a binding the gate rejects never reaches it
                                    // and reads 0 for a reason that has nothing to do with residency
                                    // -- on an `img_dim == 6` surface it is structurally 0.
                                    //
                                    // `dim=` and `ncomp=` are printed because the ROUTING, not the
                                    // sample count, decides which gate applies:
                                    // `dcc_metadata_footprint` reaches the HTILE sizer only for
                                    // `img_dim == 6 && Float32 && num_components == 1`, and
                                    // otherwise uses the DCC sizer, which fail-closes on tile mode
                                    // and never looks at `sample_count`. All three conjuncts must be
                                    // visible: `fmt=` cannot stand in for `ncomp=`, because it
                                    // prints the DataFormat enum and `Float32` is reached from raw
                                    // IMG_FMT 22/64/74/77 with one, two, three or four components.
                                    // With any conjunct missing, "the sample count blocked it" and
                                    // "it was never routed to the HTILE sizer at all" stay
                                    // indistinguishable -- which they were, and a gate analysis was
                                    // written against the wrong gate because of it.
                                    const char* kind_name =
                                        kind == prosper::gpu::CompressionMetadataKind::Htile
                                            ? "HTILE"
                                            : kind == prosper::gpu::CompressionMetadataKind::Dcc
                                                  ? "DCC"
                                                  : "UNCORRELATED";
                                    char first_text[24] = "(unread)";
                                    if (metadata_got)
                                        std::snprintf(first_text, sizeof first_text, "0x%02x",
                                                      metadata[0]);
                                    fprintf(stderr,
                                            "[render] compressed sampled image kind=%s draw=%llu "
                                            "order=%llu fs=0x%llx fs-id=%llu target=0x%llx "
                                            "addr=0x%llx "
                                            "meta=0x%llx %ux%ux%u fmt=%u ncomp=%u dim=%u tile=%u "
                                            "samples=%u pipe_aligned=%u ds_live=%u is unsupported; "
                                            "metadata=%zu/%llu first=%s\n",
                                            kind_name,
                                            (unsigned long long)draw.draw_index,
                                            (unsigned long long)draw.command_order,
                                            (unsigned long long)draw.fs_guest_addr,
                                            (unsigned long long)draw.fs_identity,
                                            (unsigned long long)draw.color0_base,
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)r.metadata_addr,
                                            tw, th, r.depth, (unsigned)r.format, r.num_components,
                                            r.img_dim, r.tile_mode, r.sample_count,
                                            r.meta_pipe_aligned ? 1u : 0u,
                                            has_ds_live ? 1u : 0u, metadata_got,
                                            (unsigned long long)metadata_bytes, first_text);
                                }
                            }
                        }
                        // PROSPER_DS_UNBRIDGED_FAR=1 — DIAGNOSTIC ONLY, never a shipped default.
                        //
                        // A depth surface prosper rendered into a retained DS image but could not
                        // bridge decodes from guest memory the renderer never wrote, i.e. zeros. For
                        // a shadow map zero is the NEAR plane, so every shadow test reads "occluded"
                        // and the lighting pass outputs black over a perfectly good G-buffer. Filling
                        // with far instead flips that to "unoccluded", which is equally wrong as
                        // output but is a decisive discriminator: if the scene lights up, the
                        // unbridged depth surface is the cause; if it does not, the hypothesis is
                        // dead and no cube-assembly work is justified. It detects its own invalidity
                        // in both directions, and it reports whether it fired at all.
                        //
                        // TWO POLES, and which one means "unoccluded" is not knowable a priori: a
                        // reversed-Z depth buffer stores near at 1.0, so filling 0xff is the SAME
                        // claim as the zeros it was meant to contradict. The experiment is only
                        // decisive if both poles are run, so the fill byte is a parameter
                        // (PROSPER_DS_UNBRIDGED_FILL, default 0xff) rather than a constant.
                        //
                        // Restricted to cubes by PROSPER_DS_UNBRIDGED_FAR=cube. The first run filled
                        // every unbridged retained DS plane including the main 4K depth, which is
                        // depth-tested against by the HUD -- the radar vanished, so the arm measured
                        // the confound as much as the subject.
                        static const std::string far_mode =
                            PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FAR")
                                ? PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FAR") : "";
                        static const uint8_t far_fill = PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL")
                            ? static_cast<uint8_t>(strtoul(
                                  getenv("PROSPER_DS_UNBRIDGED_FILL"), nullptr, 0))
                            : 0xffu;
                        // PROSPER_DS_UNBRIDGED_FILL_F32=<float> — fill FLOAT TEXELS, not bytes.
                        //
                        // The byte fill above is correct for a UNORM depth, where 0xff really is 1.0.
                        // It is NOT correct for a Float32 depth (`fmt=1`), where 0xffffffff is
                        // **NaN**: every comparison against it is false, so the "far" pole is not far,
                        // it is a third failure mode. A two-pole run over {0x00, 0xff} on a Float32
                        // surface therefore tests {0.0f, NaN} and never tests 1.0f at all -- which
                        // silently voids the discriminator this lever exists to be. One such run was
                        // published as a falsification before the arithmetic was checked.
                        static const bool have_f32_fill =
                            PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL_F32") != nullptr;
                        static const float f32_fill = have_f32_fill
                            ? strtof(PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FILL_F32"), nullptr)
                            : 1.0f;
                        if (!rtt_hit && !has_ds_live && !fr.is_storage_image &&
                            !far_mode.empty() && far_mode != "0" &&
                            (far_mode != "cube" || r.img_dim == 3u) &&
                            prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                            const bool f32_path = have_f32_fill &&
                                r.format == prosper::gpu::DataFormat::Float32 &&
                                texture_pixels.size() % sizeof(float) == 0;
                            if (f32_path) {
                                float* texels = reinterpret_cast<float*>(texture_pixels.data());
                                std::fill(texels, texels + texture_pixels.size() / sizeof(float),
                                          f32_fill);
                            } else {
                                std::fill(texture_pixels.begin(), texture_pixels.end(), far_fill);
                            }
                            // PER-ADDRESS count, not just a global one. The report prints once per
                            // address, so a global running total cannot say how many of a given
                            // surface's missed samples the fill actually reached -- and that is
                            // exactly the number a reader needs: this lever only fires when
                            // `is_retained_ds_plane` holds, so it can cover an arbitrary fraction of
                            // the misses a run observes. Without it, "filled, and nothing changed"
                            // is not a falsification, because the fill may have reached 2 of 161.
                            static std::mutex far_mutex;
                            static std::map<uint64_t, uint64_t> far_hits;
                            static std::atomic<uint64_t> far_count{0};
                            const uint64_t n = far_count.fetch_add(1) + 1;
                            bool first = false;
                            uint64_t per_addr = 0;
                            {
                                std::lock_guard lock(far_mutex);
                                per_addr = ++far_hits[r.gpu_addr];
                                first = per_addr == 1;
                            }
                            // Re-report on a power-of-two schedule so coverage is visible without
                            // flooding: 1, 2, 4, 8 ... fills per address.
                            first = first || (per_addr & (per_addr - 1)) == 0;
                            char fill_text[32];
                            if (f32_path)
                                std::snprintf(fill_text, sizeof fill_text, "%gf", f32_fill);
                            else
                                std::snprintf(fill_text, sizeof fill_text, "0x%02x",
                                              (unsigned)far_fill);
                            if (first)
                                fprintf(stderr,
                                        "[ds-far] addr=0x%llx %ux%ux%u dim=%u fmt=%u path=%s "
                                        "filled=%s (unbridged retained DS plane; "
                                        "fills-this-addr=%llu total=%llu)\n",
                                        (unsigned long long)r.gpu_addr, tw, th, r.depth, r.img_dim,
                                        (unsigned)r.format, f32_path ? "f32" : "byte",
                                        fill_text, (unsigned long long)per_addr,
                                        (unsigned long long)n);
                            rtt_hit = true;   // do not overwrite it with a guest-byte decode below
                            resource_rtt_hit = true;
                        }
                        bool cube_done = dcc_fast_clear_done && is_cube;
                        // Deliberately NOT folded into cube_done: the two publish through different
                        // channels, and reusing one flag would multiply an array's height by six.
                        bool array_done = false;
                        // CUBE DEPTH BRIDGE. A guest depth cube is ONE six-layer allocation whose
                        // faces prosper retains as separate DS images (each keyed by its
                        // DB_DEPTH_VIEW slice). The recompiler lowers a cube sample onto a single
                        // vertically stacked w x 6h image, so the faces are gathered and restacked
                        // here -- nothing can bind six images as one cube.
                        //
                        // Without this the sample fell through to a guest-byte decode of memory the
                        // renderer never writes, i.e. zeros. For an omnidirectional shadow map zero
                        // is one pole of the depth range, so every shadow test answered the same way
                        // everywhere; a two-pole experiment showed the value can drive GTA V's
                        // lighting output from its normal content to exactly zero, so this input is
                        // load-bearing rather than cosmetic.
                        //
                        // The guest surface is Z16 (DB_Z_INFO.FORMAT=1) while prosper canonicalises
                        // host attachments to D32_SFLOAT, so the conversion is NUMERIC -- clamp to
                        // [0,1] and quantise -- not a bit reinterpretation, and it must not invert:
                        // reversed-Z is already carried by the stored values and by the comparison
                        // direction. CONFIDENCE: MED on the 8-bit staging below; the value is
                        // faithful, the precision is not the guest's, and a shadow compare is
                        // precision-sensitive. Recorded rather than hidden.
                        bool cube_depth_bridged = false;
                        if (is_cube && !rtt_hit && !resource_compute_depth_hybrid &&
                            !fr.is_storage_image && r.depth == 6u &&
                            prosper::test::is_retained_ds_plane(r.gpu_addr)) {
                            std::array<std::vector<float>, 6> faces;
                            uint32_t slices_found = 0;
                            std::string cube_error;
                            // Publish the consumer descriptor's layer stride so the DS invalidation
                            // can address each face's own bytes. This is the only authority this
                            // title offers: it never programs DB_DEPTH_SLICE.
                            if (r.layer_stride_bytes)
                                prosper::test::note_ds_layer_stride(r.gpu_addr, tw, th,
                                                                    r.layer_stride_bytes);
                            uint32_t present_mask = retained_depth_cube.present_mask;
                            uint32_t known_mask = retained_depth_cube.known_mask;
                            for (uint32_t face = 0; face < 6u; ++face)
                                slices_found += (present_mask >> face) & 1u;
                            // Selection above is metadata-only. Do not synchronously read back four
                            // or five resident faces merely to discover that the sixth is absent and
                            // discard every byte into the guest fallback. Only a complete selection
                            // can enter this all-renderer bridge; the compute/DS hybrid is handled by
                            // its independently ordered path below.
                            const bool complete = retained_depth_cube_cache_candidate &&
                                prosper::test::read_persistent_ds_cube_depth(
                                    r.gpu_addr, tw, th, faces, slices_found, cube_error,
                                    &present_mask, &known_mask);
                            // Residency DISTRIBUTION, not a first-sight snapshot. "1 of 6 faces"
                            // seen once is consistent with three different worlds: the guest
                            // amortises faces across frames, prosper's invalidation evicts them
                            // faster than they are re-rendered, or the sample simply landed
                            // mid-sequence. A histogram over every sample separates them -- if the
                            // count never reaches 6, no timing argument survives.
                            {
                                static std::mutex cube_mutex;
                                static std::map<std::pair<uint64_t, uint32_t>, uint64_t> hist;
                                static std::map<uint64_t, std::pair<uint32_t, uint32_t>> masks;
                                static uint64_t samples = 0;
                                std::lock_guard lock(cube_mutex);
                                ++hist[{r.gpu_addr, slices_found}];
                                auto& seen_masks = masks[r.gpu_addr];
                                seen_masks.first |= present_mask;   // any slice ever VALID
                                seen_masks.second |= known_mask;    // any slice ever KNOWN
                                // Increment SEQUENCED before the test: `(++samples) & (samples - 1)`
                                // reads and modifies `samples` with no sequencing between the
                                // operands of `&`, so the throttle's cadence was whatever the
                                // optimiser chose. That matters here beyond tidiness -- CLAUDE.md
                                // tells readers to reason about this power-of-two schedule when
                                // interpreting the log, and that reasoning is only sound if the
                                // schedule is defined. Do not fold these back together.
                                ++samples;
                                if ((samples & (samples - 1)) == 0 && samples >= 64) {
                                    fprintf(stderr,
                                            "[cube-depth] residency after %llu cube samples "
                                            "(faces resident -> times seen):\n",
                                            (unsigned long long)samples);
                                    for (const auto& e : hist)
                                        fprintf(stderr,
                                                "[cube-depth]   addr=0x%llx faces=%u/6 x%llu%s\n",
                                                (unsigned long long)e.first.first, e.first.second,
                                                (unsigned long long)e.second,
                                                e.first.second == 6u ? "  <-- bridgeable" : "");
                                    for (const auto& e : masks)
                                        fprintf(stderr,
                                                "[cube-depth]   addr=0x%llx slices ever VALID=0x%02x "
                                                "ever KNOWN=0x%02x (bit n = slice n; 0x3f = all six)"
                                                "\n",
                                                (unsigned long long)e.first, e.second.first,
                                                e.second.second);
                                }
                            }
                            (void)cube_error;
                            if (complete && texture_pixels.size() >=
                                    static_cast<size_t>(tw) * th * 6u * 4u) {
                                const bool ctiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                                    !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                                auto face_base = [&](uint32_t face, size_t selected_span) {
                                    const uint64_t stride = r.layer_stride_bytes
                                        ? r.layer_stride_bytes : selected_span;
                                    return r.gpu_addr + static_cast<uint64_t>(face) * stride;
                                };
                                for (uint32_t face = 0; face < 6u; ++face) {
                                    const std::vector<float>& src = faces[face];
                                    uint8_t* dst = texture_pixels.data() +
                                        static_cast<size_t>(face) * tw * th * 4u;
                                    if (!src.empty()) {
                                        for (size_t i = 0; i < static_cast<size_t>(tw) * th &&
                                                           i < src.size(); ++i) {
                                            float d = src[i];
                                            d = d < 0.0f ? 0.0f : (d > 1.0f ? 1.0f : d);
                                            const uint8_t q = static_cast<uint8_t>(d * 255.0f + 0.5f);
                                            dst[i * 4 + 0] = q;
                                            dst[i * 4 + 1] = q;
                                            dst[i * 4 + 2] = q;
                                            dst[i * 4 + 3] = 0xffu;
                                        }
                                    } else {
                                        const uint32_t source_bpt = bpt ? bpt : 2u;
                                        const size_t linear_bytes = static_cast<size_t>(tw) * th * source_bpt;
                                        const size_t surface_bytes = ctiled
                                            ? prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, source_bpt)
                                            : linear_bytes;
                                        const uint64_t selected_addr = face_base(face, surface_bytes);
                                        std::vector<uint8_t> traw(surface_bytes, 0);
                                        const size_t got = copy_resource(traw.data(), selected_addr, surface_bytes);
                                        std::vector<uint8_t> linear(linear_bytes, 0);
                                        if (got >= linear.size() && ctiled) {
                                            prosper::gpu::detile_surface(
                                                linear.data(), traw.data(), tw, th, r.tile_mode, 0, source_bpt);
                                        } else if (got >= linear.size()) {
                                            std::memcpy(linear.data(), traw.data(), linear.size());
                                        }
                                        if (got >= linear.size()) {
                                            const uint16_t* u16 = reinterpret_cast<const uint16_t*>(linear.data());
                                            for (size_t i = 0; i < static_cast<size_t>(tw) * th; ++i) {
                                                const uint8_t q = static_cast<uint8_t>((u16[i] >> 8) & 0xffu);
                                                dst[i * 4 + 0] = q;
                                                dst[i * 4 + 1] = q;
                                                dst[i * 4 + 2] = q;
                                                dst[i * 4 + 3] = 0xffu;
                                            }
                                        } else {
                                            for (size_t i = 0; i < static_cast<size_t>(tw) * th; ++i) {
                                                dst[i * 4 + 0] = 0xffu;
                                                dst[i * 4 + 1] = 0xffu;
                                                dst[i * 4 + 2] = 0xffu;
                                                dst[i * 4 + 3] = 0xffu;
                                            }
                                        }
                                    }
                                }
                                cube_depth_bridged = true;
                                rtt_hit = true;          // do not overwrite with a guest-byte decode
                                resource_rtt_hit = true;
                                // The loop above wrote all SIX faces into texture_pixels. Publishing
                                // is driven by `cube_done` -- `fr.th = cube_done ? th * 6u : th` at
                                // the upload seam -- so leaving it false uploaded one face and
                                // silently dropped five, and the stacked-face lowering could never
                                // see them. The ordinary cube path sets this for the same reason.
                                cube_done = true;
                            }
                        }
                        if ((is_cube || is_array) && !rtt_hit && !dcc_fast_clear_done &&
                            !cube_depth_bridged) {
                            const uint32_t cb = prosper::gpu::bc_block_bytes(r.format);
                            const bool ctiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                                !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            auto face_base = [&](uint32_t face, size_t selected_span) {
                                const uint64_t stride = r.layer_stride_bytes
                                    ? r.layer_stride_bytes : selected_span;
                                return r.gpu_addr + static_cast<uint64_t>(face) * stride;
                            };
                            const uint32_t slice_count = is_cube ? 6u : decoded_layers;
                            std::vector<uint32_t> slice_short(slice_count, 0);
                            uint32_t slice_short_count = 0;
                            // Reports the GUEST's declared layout only. Deliberately not the
                            // decoded slice count: that is what PROSPER_ARRAY_DECODE_BUDGET_MIB
                            // measures, and a diagnostic that prints a field a different switch
                            // decided is one whose output cannot be trusted on its own -- which
                            // check_diag_gates.py enforces (TWO-GATE).
                            if (!is_cube && PROSPER_ENV_ON("PROSPER_SLICESTRIDE")) {
                                static std::unordered_set<uint64_t> reported;
                                if (reported.insert(r.gpu_addr).second)
                                    fprintf(stderr,
                                            "[slicestride] addr=0x%llx %ux%u guest_depth=%u fmt=%u "
                                            "layer_stride=%u mip_off=%u in_mip_tail=%d "
                                            "mip_tail_bytes=%u declared_mips=%u size=%u\n",
                                            (unsigned long long)r.gpu_addr, tw, th, r.depth,
                                            (unsigned)r.format, r.layer_stride_bytes,
                                            r.layer_mip_offset_bytes, (int)r.in_mip_tail,
                                            r.mip_tail_bytes, r.declared_mip_levels, r.size);
                            }
                            for (uint32_t fface = 0; fface < slice_count; fface++) {
                                uint8_t* slice = texture_pixels.data() + (size_t)fface * tw * th * 4;
                                if (cb) {
                                    uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                                    size_t comp = (size_t)bw * bh * cb;
                                    const size_t surface_bytes = ctiled
                                        ? prosper::gpu::tiled_elements_bytes(
                                              bw, bh, cb, r.tile_mode)
                                        : comp;
                                    const size_t selected_span = r.in_mip_tail
                                        ? r.mip_tail_bytes : surface_bytes;
                                    const uint64_t selected_addr = face_base(fface, selected_span) +
                                        (r.in_mip_tail ? 0u : r.layer_mip_offset_bytes);
                                    std::vector<uint8_t> lin(comp, 0);
                                    // #2998: keep the byte count. safe_copy stops at the first
                                    // uncommitted 64 KiB page and leaves the tail zero-filled, so a
                                    // slice prosper could not READ is byte-identical to one the
                                    // guest never WROTE -- and every "the atlas is mostly empty"
                                    // conclusion turns on telling those apart. The non-BC branch
                                    // below already checks its `got`; this one discarded it.
                                    size_t slice_got = 0;
                                    if (ctiled) {
                                        std::vector<uint8_t> traw(selected_span, 0);
                                        slice_got = copy_resource(
                                            traw.data(), selected_addr, selected_span);
                                        if (r.in_mip_tail)
                                            prosper::gpu::detile_elements_level(
                                                lin.data(), traw.data(), traw.size(), bw, bh, cb,
                                                r.tile_mode, r.mip_tail_x, r.mip_tail_y);
                                        else
                                            prosper::gpu::detile_elements(
                                                lin.data(), traw.data(), traw.size(), bw, bh, cb,
                                                r.tile_mode);
                                    } else if (linear_padded_read) {
                                        slice_got = copy_linear_padded_rows_from(
                                            lin.data(), selected_addr,
                                            static_cast<size_t>(bw) * cb, bh);
                                    } else {
                                        slice_got = copy_resource(lin.data(), selected_addr, comp);
                                    }
                                    if (slice_short_count < slice_count &&
                                        slice_got < (ctiled ? selected_span : comp))
                                        slice_short[slice_short_count++] = fface;
                                    std::vector<uint8_t> face((size_t)tw * th * 4, 0);
                                    prosper::gpu::bc_decode_surface(face.data(), lin.data(), lin.size(), tw, th, r.format);
                                    std::memcpy(slice, face.data(), face.size());
                                } else {
                                    const uint32_t source_bpt = bpt;
                                    const size_t linear_bytes =
                                        static_cast<size_t>(tw) * th * source_bpt;
                                    const size_t surface_bytes = ctiled
                                        ? prosper::gpu::tiled_surface_bytes(
                                              tw, th, r.tile_mode, 0, source_bpt)
                                        : linear_bytes;
                                    const size_t selected_span = r.in_mip_tail
                                        ? r.mip_tail_bytes : surface_bytes;
                                    const uint64_t selected_addr = face_base(fface, selected_span) +
                                        (r.in_mip_tail ? 0u : r.layer_mip_offset_bytes);
                                    std::vector<uint8_t> linear(linear_bytes, 0);
                                    size_t nbc_got = 0, nbc_want = 0;
                                    if (ctiled) {
                                        std::vector<uint8_t> traw(selected_span, 0);
                                        const size_t got = copy_resource(
                                            traw.data(), selected_addr, selected_span);
                                        nbc_got = got; nbc_want = selected_span;
                                        if (got < linear.size()) {
                                            copy_resource(
                                                linear.data(), selected_addr, linear.size());
                                        } else if (r.in_mip_tail) {
                                            prosper::gpu::detile_surface_level(
                                                linear.data(), traw.data(), got, tw, th,
                                                r.tile_mode, source_bpt,
                                                r.mip_tail_x, r.mip_tail_y);
                                        } else {
                                            prosper::gpu::detile_surface(
                                                linear.data(), traw.data(), tw, th,
                                                r.tile_mode, 0, source_bpt);
                                        }
                                    } else if (linear_padded_read) {
                                        nbc_want = static_cast<size_t>(tw) * source_bpt * th;
                                        // #325: use the SAME pitch the single-surface path uses
                                        // (`linear_src_row`), not one recomputed from tw/bpt here.
                                        // They can differ -- the single-surface figure honours a
                                        // registered pitch and its own row width -- and slice 0 of
                                        // an array must decode byte-identically to the way that
                                        // same surface decoded before it was treated as an array.
                                        nbc_got = copy_linear_padded_rows_from(
                                            linear.data(), selected_addr,
                                            static_cast<size_t>(tw) * source_bpt, th);
                                    } else if (r.layer_stride_bytes) {
                                        const size_t row_pitch = r.linear_row_pitch_bytes
                                            ? r.linear_row_pitch_bytes
                                            : prosper::gpu::linear_sampled_row_pitch(
                                                  tw, source_bpt);
                                        for (uint32_t y = 0; y < th; ++y)
                                            nbc_got += copy_resource(
                                                linear.data() + static_cast<size_t>(y) * tw * source_bpt,
                                                selected_addr + static_cast<uint64_t>(y) * row_pitch,
                                                static_cast<size_t>(tw) * source_bpt);
                                        nbc_want = static_cast<size_t>(tw) * source_bpt * th;
                                    } else {
                                        nbc_got = copy_resource(linear.data(), selected_addr,
                                                                linear.size());
                                        nbc_want = linear.size();
                                    }
                                    // Measured, not assumed: this branch left slice_short_count
                                    // untouched, so an uncompressed array printed short_reads=0
                                    // having counted nothing -- a clean result the instrument had
                                    // not earned, which is the exact failure it exists to prevent.
                                    if (slice_short_count < slice_count &&
                                        nbc_want && nbc_got < nbc_want)
                                        slice_short[slice_short_count++] = fface;
                                    if (f16) {
                                        const uint32_t nc = source_bpt / 2;
                                        for (size_t texel = 0; texel < (size_t)tw * th; ++texel) {
                                            uint8_t* pixel = slice + texel * 4;
                                            for (uint32_t c = 0; c < 4; ++c) {
                                                float value = c == 3 ? 1.0f : 0.0f;
                                                if (c < nc) {
                                                    uint16_t half = 0;
                                                    std::memcpy(
                                                        &half,
                                                        linear.data() + texel * source_bpt + c * 2,
                                                        sizeof(half));
                                                    value = prosper::gpu::half_to_float(half);
                                                }
                                                pixel[c] = !std::isfinite(value) || value <= 0.0f
                                                    ? 0u : (value >= 1.0f
                                                        ? 255u
                                                        : static_cast<uint8_t>(value * 255.0f + 0.5f));
                                            }
                                        }
                                    } else if (source_bpt == 4) {
                                        std::memcpy(slice, linear.data(), linear.size());
                                    } else {
                                        const uint32_t component_bytes =
                                            prosper::gpu::data_format_bytes(r.format);
                                        const uint32_t nc = r.num_components ? r.num_components : 1u;
                                        for (size_t texel = 0; texel < (size_t)tw * th; ++texel) {
                                            uint8_t* pixel = slice + texel * 4;
                                            pixel[0] = pixel[1] = pixel[2] = 0;
                                            pixel[3] = 255;
                                            for (uint32_t c = 0; c < std::min(nc, 4u); ++c) {
                                                if (component_bytes == 1) {
                                                    pixel[c] = linear[texel * source_bpt + c];
                                                } else if (component_bytes == 2) {
                                                    uint16_t value = 0;
                                                    std::memcpy(
                                                        &value,
                                                        linear.data() + texel * source_bpt + c * 2,
                                                        sizeof(value));
                                                    pixel[c] = prosper::gpu::unorm16_to_unorm8(value);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // The loop above filled `slice_count` slices. A cube publishes them through
                            // HEIGHT (fr.th = th*6) and an array through LAYERS (fr.sample_count), so the
                            // two completion flags must stay distinct.
                            if (PROSPER_ENV_ON("PROSPER_SLICEMAP") && !is_cube && slice_count > 1) {
                                // #2998: which decoded slices hold content, across the WHOLE range.
                                // A prefix cannot answer the question -- an array whose first slices
                                // decode and whose later ones are empty is identical, in every
                                // per-layer checksum of the first eight, to one that decoded fully.
                                //
                                // Two things this MUST report alongside the map, because without
                                // them its output is not interpretable:
                                //  - the sample rate. "empty" here means "no non-zero byte at a
                                //    sampled offset", and the sample is sparse, so a slice with
                                //    little content can read empty. It is a lower bound on content,
                                //    never an exact count.
                                //  - the SHORT READS. safe_copy stops at the first uncommitted page
                                //    and zero-fills the rest, so a slice prosper could not READ is
                                //    byte-identical to one the guest never WROTE. Reporting empties
                                //    without this invites exactly the wrong conclusion, which is the
                                //    one it invited from me.
                                const size_t lsz = (size_t)tw * th * 4u;
                                const size_t step = 61u;
                                std::string map;
                                uint32_t nonempty = 0;
                                for (uint32_t L = 0; L < slice_count; ++L) {
                                    if ((size_t)(L + 1) * lsz > texture_pixels.size()) break;
                                    const uint8_t* q = texture_pixels.data() + (size_t)L * lsz;
                                    bool any = false;
                                    for (size_t k = 0; k < lsz && !any; k += step) any = (q[k] != 0);
                                    nonempty += any;
                                    map += any ? '#' : '.';
                                }
                                fprintf(stderr,
                                        "[slicemap] addr=0x%llx %ux%u slices=%u nonempty>=%u "
                                        "(1 byte in %zu sampled) short_reads=%u%s map=%s\n",
                                        (unsigned long long)r.gpu_addr, tw, th, slice_count,
                                        nonempty, step, slice_short_count,
                                        slice_short_count ? " <- EMPTY MAY MEAN UNREADABLE" : "",
                                        map.c_str());
                                if (slice_short_count)
                                    fprintf(stderr, "[slicemap]   first short slice=%u\n",
                                            slice_short[0]);
                            }
                            if (is_cube) cube_done = true; else array_done = true;
                            if (resource_compute_depth_hybrid && !decoded_reuse) {
                                std::array<std::vector<float>, 6> overlay_faces;
                                uint32_t overlay_mask = 0, known_mask = 0;
                                std::string overlay_error;
                                const bool overlay_ok =
                                    prosper::test::read_persistent_ds_cube_depth_after(
                                        r.gpu_addr, tw, th,
                                        resource_compute_producer_order,
                                        overlay_faces, overlay_mask, known_mask,
                                        overlay_error);
                                if (overlay_ok && overlay_mask ==
                                        resource_compute_depth_overlay_mask) {
                                    for (uint32_t face = 0; face < 6u; ++face) {
                                        if (!(overlay_mask & (1u << face))) continue;
                                        const std::vector<float>& src = overlay_faces[face];
                                        uint8_t* dst = texture_pixels.data() +
                                            static_cast<size_t>(face) * tw * th * 4u;
                                        for (size_t i = 0;
                                             i < static_cast<size_t>(tw) * th && i < src.size();
                                             ++i) {
                                            const float d = std::clamp(src[i], 0.0f, 1.0f);
                                            const uint8_t q = static_cast<uint8_t>(
                                                d * 255.0f + 0.5f);
                                            dst[i * 4 + 0] = q;
                                            dst[i * 4 + 1] = q;
                                            dst[i * 4 + 2] = q;
                                            dst[i * 4 + 3] = 0xffu;
                                        }
                                    }
                                    static std::mutex hybrid_log_mutex;
                                    static std::map<uint64_t, uint64_t> hybrid_counts;
                                    std::lock_guard lock(hybrid_log_mutex);
                                    const uint64_t n = ++hybrid_counts[r.gpu_addr];
                                    if ((n & (n - 1)) == 0)
                                        std::fprintf(
                                            stderr,
                                            "[cube-depth] compute/DS hybrid addr=0x%llx "
                                            "producer-order=%llu overlay=0x%02x known=0x%02x "
                                            "(x%llu)\n",
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)
                                                resource_compute_producer_order,
                                            overlay_mask, known_mask,
                                            (unsigned long long)n);
                                } else {
                                    static uint64_t hybrid_failures = 0;
                                    if (((++hybrid_failures) & (hybrid_failures - 1)) == 0)
                                        std::fprintf(
                                            stderr,
                                            "[cube-depth] compute/DS hybrid FAILED "
                                            "addr=0x%llx expected=0x%02x got=0x%02x "
                                            "known=0x%02x error=%s (x%llu)\n",
                                            (unsigned long long)r.gpu_addr,
                                            resource_compute_depth_overlay_mask,
                                            overlay_mask, known_mask,
                                            overlay_error.c_str(),
                                            (unsigned long long)hybrid_failures);
                                }
                            }
                        }
                        // Block-compressed (BC1/2/3): read the (possibly tiled) compressed blocks, block-
                        // detile, and decode to RGBA8 in-place. The blocks are the tiled element (BC3 =
                        // 16 bytes -> SW_4KB_S 16x16-block micro-tiles). #121.
                        const uint32_t bcb = (rtt_hit || cube_done || array_done || dcc_fast_clear_done)
                            ? 0u : prosper::gpu::bc_block_bytes(r.format);
                        if (rtt_hit || cube_done || array_done || dcc_fast_clear_done) { /* pixels already materialized */ }
                        else if (portable_raw_uvec4_storage) {
                            // Storage-image tiling describes the compact guest texels, not the expanded
                            // 16-byte Vulkan representation. Detile at the real guest width first, then
                            // convert every channel into the raw VGPR dword carried by the uvec4 image.
                            const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                                !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            const uint32_t materialize_tile_mode = tiled ? r.tile_mode : 0u;
                            const uint32_t materialize_depth = is_volume ? r.depth : 1u;
                            const size_t source_bytes = storage_image_raw_uvec4_source_bytes(
                                r.format, r.num_components ? r.num_components : 1u,
                                tw, th, materialize_depth, materialize_tile_mode,
                                r.in_mip_tail, r.mip_tail_bytes);
                            std::vector<uint8_t> source(source_bytes);
                            bool decoded = fr.storage_image_contract_valid;
                            const size_t got = decoded && source_bytes
                                ? copy_resource(
                                      source.data(), sampled_source_addr, source_bytes)
                                : 0u;
                            const bool materialized = decoded && got == source_bytes &&
                                storage_image_materialize_raw_uvec4(
                                    source.data(), got, r.format,
                                    r.num_components ? r.num_components : 1u,
                                    tw, th, materialize_depth, materialize_tile_mode,
                                    r.in_mip_tail, r.mip_tail_bytes,
                                    r.mip_tail_x, r.mip_tail_y,
                                reinterpret_cast<uint32_t*>(texture_pixels.data()),
                                texture_pixels.size() / sizeof(uint32_t));
                            decoded = decoded && materialized;
                            if (!decoded) {
                                static std::set<std::tuple<uint32_t, uint32_t, uint32_t>> reported;
                                if (reported.emplace(fr.set, fr.binding,
                                                     static_cast<uint32_t>(r.format)).second)
                                    std::fprintf(stderr,
                                        "[render] storage-image materialize failed: set=%u binding=%u "
                                        "source=%zu got=%zu output=%zu fmt=%u comps=%u "
                                        "extent=%ux%ux%u tile=%u tail=%u contract=%u\n",
                                        fr.set, fr.binding, source_bytes, got,
                                        texture_pixels.size(), static_cast<unsigned>(r.format),
                                        r.num_components, tw, th, materialize_depth,
                                        materialize_tile_mode, r.in_mip_tail ? 1u : 0u,
                                        fr.storage_image_contract_valid ? 1u : 0u);
                                fr.storage_image_contract_valid = false;
                                std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                            }
                        } else if (bcb) {
                            uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                            // Block-detile: tiled_elements_bytes/detile_elements now derive the 4KB
                            // micro-tile geometry from the block size (bpe) internally (#119) — 16-byte
                            // blocks -> 16x16, 8-byte -> 32x16 — so no tile_side is passed here.
                            size_t comp_bytes = (size_t)bw * bh * bcb;
                            auto lin = prosper::frontend::decode_scratch_pool().take(comp_bytes);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = prosper::gpu::tiled_elements_bytes(bw, bh, bcb, r.tile_mode);
                                prosper::frontend::DecodeScratchPool::Lease traw_lease;
                                size_t got = 0;
                                const uint8_t* traw = stage_tiled_source(
                                    traw_lease, sampled_source_addr, tbytes, got);
                                if (r.in_mip_tail) {
                                    lin.zero_all();
                                    prosper::gpu::detile_elements_level(
                                        lin.data(), traw, tbytes, bw, bh, bcb,
                                        r.tile_mode, r.mip_tail_x, r.mip_tail_y);
                                } else {
                                    // detile_elements zero-fills every block the (possibly short)
                                    // source cannot supply, so it covers the whole destination for
                                    // the element sizes its copiers handle -- pinned by test_tile.
                                    if (!prosper::gpu::detile_writes_whole_destination(
                                            r.tile_mode, bcb))
                                        lin.zero_all();
                                    prosper::gpu::detile_elements(
                                        lin.data(), traw, tbytes, bw, bh, bcb, r.tile_mode);
                                }
                            } else if (linear_padded_read) {
                                lin.zero_all();
                                copy_linear_padded_rows(
                                    lin.data(), static_cast<size_t>(bw) * bcb, bh);
                            } else {
                                lin.zero_tail(copy_resource(
                                    lin.data(), sampled_source_addr, comp_bytes));
                            }
                            if (!prosper::gpu::bc_decode_surface(
                                    texture_pixels.data(), lin.data(), lin.size(), tw, th, r.format))
                                std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                        } else if (f32) {
                            // Sampled Float32 resources cannot be reinterpreted as RGBA8: a value such as
                            // 1/8192 has bytes 00 00 00 39 and became a zero red channel. Preserve the float
                            // ordinary values and useful HDR range by narrowing each present component to
                            // native RGBA16F. This deliberately loses Float32 precision and extreme range;
                            // the backend carries the resulting half values losslessly to the sampler.
                            // Power-of-two component counts keep the source texel size compatible with the
                            // supported GFX10 surface detilers (4/8/16 B per texel).
                            const uint32_t nc = bpt / 4;
                            auto flin = prosper::frontend::decode_scratch_pool().take(
                                volume_texels * bpt);
                            const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                                !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            if (tiled) {
                                const size_t tbytes = is_volume
                                    ? prosper::gpu::tiled_volume_bytes(
                                          tw, th, r.depth, r.tile_mode, bpt)
                                    : prosper::gpu::tiled_surface_bytes(
                                          tw, th, r.tile_mode, 0, bpt);
                                prosper::frontend::DecodeScratchPool::Lease traw_lease;
                                size_t got = 0;
                                const uint8_t* traw = stage_tiled_source(
                                    traw_lease, sampled_source_addr, tbytes, got);
                                if (got < flin.size()) {
                                    flin.zero_tail(copy_resource(
                                        flin.data(), sampled_source_addr, flin.size()));
                                } else if (is_volume) {
                                    flin.zero_all();
                                    prosper::gpu::detile_volume(
                                        flin.data(), traw, got, tw, th, r.depth,
                                        r.tile_mode, bpt);
                                } else if (r.in_mip_tail) {
                                    flin.zero_all();
                                    prosper::gpu::detile_surface_level(
                                        flin.data(), traw, got, tw, th, r.tile_mode,
                                        bpt, r.mip_tail_x, r.mip_tail_y);
                                } else {
                                    if (!prosper::gpu::detile_writes_whole_destination(
                                            r.tile_mode, bpt))
                                        flin.zero_all();
                                    prosper::gpu::detile_surface(
                                        flin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                                }
                            } else if (linear_padded_read) {
                                flin.zero_all();
                                copy_linear_padded_rows(flin.data(), (size_t)tw * bpt, th);
                            } else {
                                flin.zero_tail(copy_resource(
                                    flin.data(), sampled_source_addr, flin.size()));
                            }
                            // Was a per-texel/per-channel scalar loop around float_to_half with a
                            // 4-byte and a 2-byte memcpy each -- 8.3 M conversions and 16.6 M small
                            // copies for one 1920x1080 reference, single-threaded, next to an
                            // already-vectorised sibling. The helper below IS that sibling, widened
                            // to the narrower component counts; it keeps the same (0,0,0,1) fill and
                            // the same bit-for-bit rounding/NaN-payload contract.
                            prosper::frontend::pack_float32_to_rgba16f_range(
                                flin.data(), nc, bpt, volume_texels, texture_pixels.data());
                            f32_done = true;
                        } else if (f16) {
                            // fp16 texture (#290 wall 1): read at the REAL bytes-per-texel and detile
                            // with the REAL element size — the old clamp read an 8-B/texel Float16x4
                            // surface at 4 B AND detiled it with bpe=4 against 8-B tiled elements, so
                            // the result was doubly wrong ("confetti" regions, e.g. DOLL's 960x540 /
                            // 480x270 bloom-chain buffers). Guest memory still converts half->UNORM8 on
                            // upload. That path clamps values above 1.0 and remains separate from native
                            // tiled guest-texture upload and the renderer-owned RTT fix in #773.
                            // Missing components read (0,0,0,1) per the hardware rule; the T# DST_SEL
                            // swizzle still applies. CONFIDENCE: MED — the half decode is exact and
                            // unit-tested, but the [0,1] clamp loses >1.0 bloom energy for guest-backed
                            // textures; renderer-owned RTTs retain RGBA16F through the #773 path.
                            const uint32_t nc = bpt / 2;                    // fp16 components per texel
                            // Pooled, so a 4K HDR intermediate does not mmap/fault/munmap 63 MiB per
                            // reference. It arrives holding the PREVIOUS surface, so every arm below
                            // either covers it completely or zeroes what it does not fill.
                            auto hlin = prosper::frontend::decode_scratch_pool().take(
                                volume_texels * bpt);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = is_volume
                                    ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                                    : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                                prosper::frontend::DecodeScratchPool::Lease traw_lease;
                                size_t got = 0;
                                const uint8_t* traw = stage_tiled_source(
                                    traw_lease, sampled_source_addr, tbytes, got);
                                if (got < hlin.size())
                                    hlin.zero_tail(copy_resource(hlin.data(), sampled_source_addr,
                                                                 hlin.size()));  // short backing -> linear fallback
                                else if (is_volume) { hlin.zero_all(); prosper::gpu::detile_volume(
                                    hlin.data(), traw, got, tw, th, r.depth, r.tile_mode, bpt); }
                                else if (r.in_mip_tail) { hlin.zero_all(); prosper::gpu::detile_surface_level(
                                    hlin.data(), traw, got, tw, th, r.tile_mode, bpt,
                                    r.mip_tail_x, r.mip_tail_y); }
                                else {
                                    // The only arm on Stray's hot path, and the only one whose
                                    // destination coverage is asserted (test_tile). Everything else
                                    // pre-zeroes, because a detiler that bounds its writes by a
                                    // SHORT source would otherwise leave the previous tenant visible.
                                    if (!prosper::gpu::detile_writes_whole_destination(
                                            r.tile_mode, bpt))
                                        hlin.zero_all();
                                    prosper::gpu::detile_surface(
                                        hlin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                                }
                            } else if (linear_padded_read) {
                                hlin.zero_all();
                                copy_linear_padded_rows(hlin.data(), (size_t)tw * bpt, th);
                            } else {
                                hlin.zero_tail(copy_resource(hlin.data(), sampled_source_addr,
                                                             hlin.size()));
                            }
                            // Same NaN/negative/positive-infinity clamp and absent-channel defaults
                            // as the historical scalar loop, exhaustively checked over all binary16
                            // inputs by test_game_compute. Large dynamic textures convert in parallel.
                            sampled_float16_to_unorm8_range(
                                hlin.data(), nc, volume_texels, texture_pixels.data());
                            f16_done = true;   // read+detiled at the real element size already
                        } else if (native_r8_sampled) {
                            // Tight linear R8 needs neither detiling nor per-texel expansion. One guarded
                            // copy feeds the backend's 1-byte staging upload; the view swizzle below makes
                            // every sampled component equal R, byte-for-byte matching the old RGBA result.
                            linear_source_prefix_size = copy_resource(
                                texture_pixels.data(), sampled_source_addr,
                                texture_pixels.size());
                            if (linear_source_prefix_size < texture_pixels.size())
                                std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                          texture_pixels.end(), 0);
                            narrow_decode_done = true;
                            narrow_done = true;
                        } else if (native_rg8_sampled) {
                            // RG8 is already the backend's two-byte sampled representation. Strip
                            // physical row padding when present; a contiguous copy would advance the
                            // next visible row into padding and produce a diagonal chroma weave.
                            if (linear_padded_read) {
                                linear_source_prefix_size = copy_linear_padded_rows(
                                    texture_pixels.data(), linear_dst_row, th);
                            } else {
                                linear_source_prefix_size = copy_resource(
                                    texture_pixels.data(), sampled_source_addr,
                                    texture_pixels.size());
                                if (linear_source_prefix_size < texture_pixels.size())
                                    std::fill(texture_pixels.begin() + linear_source_prefix_size,
                                              texture_pixels.end(), 0);
                            }
                            narrow_decode_done = true;
                            narrow_done = true;
                        } else if (bpt < 4) {
                            // Narrow (single/dual-channel) surface: read at the REAL element size and detile
                            // with the matching bpe geometry (1 B -> 64x64, 2 B -> 64x32 micro-tiles, #119),
                            // then expand to RGBA8. Legacy 8-bit coverage resources keep the grayscale
                            // broadcast so shaders can read either .r or .a, while the exact AvPlayer NV12
                            // contract preserves both bytes of its RG8 interleaved U/V plane. A UNORM16
                            // resource instead receives the format-defined missing channels (R,0,0,1), after
                            // which the real T# DST_SEL is applied below. Its R component must be normalized
                            // from both bytes; selecting byte zero makes a smooth ramp a sawtooth (#1186).
                            auto nlin = prosper::frontend::decode_scratch_pool().take(
                                volume_texels * bpt);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = is_volume
                                    ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                                    : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                                prosper::frontend::DecodeScratchPool::Lease traw_lease;
                                size_t got = 0;
                                const uint8_t* traw = stage_tiled_source(
                                    traw_lease, sampled_source_addr, tbytes, got);
                                // PROSPER_DUMP_RAWTILE (narrow path): the single/dual-channel RAW TILED bytes,
                                // once per address, for offline 8-bpp de-swizzle sweeps (the SDF font atlas).
                                if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTILE") && got >= nlin.size() && tw <= 2048 && th <= 1024) {
                                    static std::set<uint64_t> nseen;
                                    if (nseen.insert(r.gpu_addr).second) {
                                        std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                        char bn[512]; snprintf(bn, sizeof bn, "%s/narrowtile_%ux%u_b%u_%llx.bin",
                                                               dd.c_str(), tw, th, bpt, (unsigned long long)r.gpu_addr);
                                        if (FILE* bf = fopen(bn, "wb")) { fwrite(traw, 1, tbytes, bf); fclose(bf);
                                            fprintf(stderr, "[render] narrow raw tiled -> %s (%zu, bpt=%u)\n", bn, tbytes, bpt); fflush(stderr); }
                                    }
                                }
                                if (got < nlin.size())
                                    nlin.zero_tail(copy_resource(nlin.data(), sampled_source_addr,
                                                                 nlin.size()));  // short backing -> linear fallback
                                else if (is_volume) { nlin.zero_all(); prosper::gpu::detile_volume(
                                    nlin.data(), traw, got, tw, th, r.depth, r.tile_mode, bpt); }
                                else if (r.in_mip_tail) { nlin.zero_all(); prosper::gpu::detile_surface_level(
                                    nlin.data(), traw, got, tw, th, r.tile_mode, bpt,
                                    r.mip_tail_x, r.mip_tail_y); }
                                else {
                                    if (!prosper::gpu::detile_writes_whole_destination(
                                            r.tile_mode, bpt))
                                        nlin.zero_all();
                                    prosper::gpu::detile_surface(
                                        nlin.data(), traw, tw, th, r.tile_mode, 0, bpt);
                                }
                            } else if (linear_padded_read) {
                                nlin.zero_all();
                                copy_linear_padded_rows(nlin.data(), (size_t)tw * bpt, th);
                            } else {
                                nlin.zero_tail(copy_resource(nlin.data(), sampled_source_addr,
                                                             nlin.size()));
                            }
                            const bool unorm16 = r.format == prosper::gpu::DataFormat::Unorm16;
                            for (size_t t = 0; t < volume_texels; t++) {
                                uint8_t* p = &texture_pixels[t * 4];
                                if (avplayer_chroma_layout) {
                                    const uint8_t* source = &nlin[t * bpt];
                                    p[0] = source[0];
                                    p[1] = source[1];
                                    p[2] = r.num_components > 2 ? source[2] : 0;
                                    p[3] = 255;
                                } else {
                                    uint8_t v = nlin[t * bpt]; // first (coverage) channel
                                    if (unorm16) {
                                        uint16_t raw;
                                        std::memcpy(&raw, &nlin[t * bpt], sizeof(raw));
                                        v = prosper::gpu::unorm16_to_unorm8(raw);
                                        p[0] = v; p[1] = p[2] = 0; p[3] = 255;
                                    } else {
                                        p[0] = p[1] = p[2] = p[3] = v;
                                    }
                                }
                            }
                            narrow_decode_done = true;   // skip the generic 32-bpp detiler below
                            narrow_done = !unorm16;      // coverage broadcast replaces swizzle; R16 does not
                        } else if (linear_padded_read) {
                            // Pitch-padded linear surface (see linear_padded_read above): read row-by-row at
                            // the 256-byte-aligned source pitch into the tight destination, dropping the
                            // per-row padding. copy_resource is fault-safe when an old capture has a short
                            // backing because its pre-v28 writer omitted the final padded rows.
                            linear_source_prefix_size = copy_linear_padded_rows(
                                texture_pixels.data(), linear_dst_row, th);
                        } else {
                            const size_t got = copy_resource(
                                texture_pixels.data(), sampled_source_addr, nb);
                            linear_source_prefix_size = got;
                            if (got < nb)
                                std::fill(texture_pixels.begin() + got, texture_pixels.end(), 0);
                        }
                        // PROSPER_DUMP_RAWTEX: write the raw tiled RGBA bytes (pre-detile) to a .bin for
                        // offline swizzle experimentation.
                        if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTEX") && !texture_pixels.empty()) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_%ux%u.bin", d.c_str(), tw, th);
                            if (FILE* f = fopen(fn, "wb")) { fwrite(texture_pixels.data(), 1, nb, f); fclose(f);
                                fprintf(stderr, "[render] dumped raw tiled bytes -> %s (%zu)\n", fn, nb); fflush(stderr); }
                        }
                        if (getenv("PROSPER_GFXLOG")) { const uint8_t* b = texture_pixels.data();
                            size_t nz = 0; for (size_t i = 0; i < nb && i < (1u<<16); i++) nz += (b[i] != 0);
                            fprintf(stderr, "[render] tex binding=%u %ux%u first64k-nonzero=%zu\n", r.binding, tw, th, nz); }
                        // AUTO-DETILE: de-swizzle a GPU-tiled sampled surface into the linear texstore,
                        // driven by the T# tile_mode threaded through the resource table (r.tile_mode).
                        bool auto_tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                        const char* dt = PROSPER_ENV_VALUE("PROSPER_DETILE");
                        // BC textures already block-detiled + decoded above; the 32-bpp detiler must not touch them.
                        if (!rtt_hit && !cube_done && !array_done && !dcc_fast_clear_done && !bcb &&
                            !narrow_decode_done &&
                            !f16_done && !f32_done && !portable_raw_uvec4_storage &&
                            !PROSPER_ENV_VALUE("PROSPER_NODETILE") &&
                            (auto_tiled || (!is_volume && dt && atoi(dt) != 0))) {
                            const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                            const uint32_t pitch = PROSPER_ENV_VALUE("PROSPER_PITCH") ? (uint32_t)atoi(PROSPER_ENV_VALUE("PROSPER_PITCH")) : 0;
                            size_t tiled_bytes = is_volume
                                ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, tmode, 4)
                                : prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
                            // The `got < nb` recovery below WRITES into this buffer, so the source
                            // may only be read in place when it cannot be reached -- i.e. when the
                            // padded tiled extent already covers the linear one.
                            prosper::frontend::DecodeScratchPool::Lease tiled_lease;
                            size_t got = 0;
                            const uint8_t* tiled = nullptr;
                            if (tiled_bytes >= nb) {
                                tiled = stage_tiled_source(
                                    tiled_lease, sampled_source_addr, tiled_bytes, got);
                            } else {
                                tiled_lease = prosper::frontend::decode_scratch_pool().take(tiled_bytes);
                                got = copy_resource(
                                    tiled_lease.data(), sampled_source_addr, tiled_bytes);
                                tiled_lease.zero_tail(got);
                                tiled = tiled_lease.data();
                            }
                            if (got < nb) {
                                // Reachable only on the staged path -- the direct branch above
                                // requires tiled_bytes >= nb and then delivers all of them -- so the
                                // lease is already the buffer just filled. Re-staging if it somehow
                                // is not keeps this recovery writing to memory it owns.
                                if (tiled_lease.size() < tiled_bytes) {
                                    tiled_lease = prosper::frontend::decode_scratch_pool().take(tiled_bytes);
                                    got = copy_resource(
                                        tiled_lease.data(), sampled_source_addr, tiled_bytes);
                                    tiled_lease.zero_tail(got);
                                }
                                // The padded tiled buffer's tail (th rounded to whole 32-row tiles) runs past
                                // the real backing: fall back to the width*height linear bytes copied above
                                // rather than an all-zero buffer (which would BLANK the texture).
                                std::memcpy(tiled_lease.data(), texture_pixels.data(),
                                            std::min(nb, tiled_bytes));
                                tiled = tiled_lease.data();
                            }
                            // PROSPER_DUMP_RAWTILE: write the EXACT padded tiled bytes to a .bin (no lossy BMP
                            // round-trip) so the de-swizzle can be reversed offline against a known image (#101).
                            if (PROSPER_ENV_ON("PROSPER_DUMP_RAWTILE") && (frame_no < 200 || (tw <= 2048 && th <= 1024))) {
                                // Early frames by binding, PLUS small textures once per address (the font/UI
                                // atlas can be sampled late — capture whenever first seen).
                                static std::set<uint64_t> rawseen;
                                bool small = tw <= 2048 && th <= 1024;
                                if (!small || rawseen.insert(r.gpu_addr).second) {
                                    std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                    char bn[512];
                                    if (small) snprintf(bn, sizeof bn, "%s/rawtile_%ux%u_%llx.bin", dd.c_str(), tw, th, (unsigned long long)r.gpu_addr);
                                    else snprintf(bn, sizeof bn, "%s/tiled_f%04d_b%u_%ux%u.bin", dd.c_str(), (int)frame_no, r.binding, tw, th);
                                    if (FILE* bf = fopen(bn, "wb")) { fwrite(tiled, 1, tiled_bytes, bf); fclose(bf); }
                                }
                            }
                            if (is_volume) prosper::gpu::detile_volume(
                                texture_pixels.data(), tiled, got, tw, th, r.depth, tmode, 4);
                            else if (r.in_mip_tail) prosper::gpu::detile_surface_level(
                                texture_pixels.data(), tiled, got, tw, th, tmode, 4,
                                r.mip_tail_x, r.mip_tail_y);
                            else prosper::gpu::detile_surface(
                                texture_pixels.data(), tiled, tw, th, tmode, pitch);
                        }
                        // Legacy fallback for packed R11G11B10F shapes outside the exact native upload
                        // contract above. The texel IS 4 bytes, so the generic read + auto-detile already
                        // produced linear packed words; only unsupported shapes still narrow to RGBA8.
                        if (!rtt_hit && !dcc_fast_clear_done && !portable_raw_uvec4_storage &&
                            !native_r11_sampled &&
                            r.format == prosper::gpu::DataFormat::Float10_11_11) {
                            uint8_t* tp = texture_pixels.data();
                            const size_t decoded_texels =
                                volume_texels * (cube_done ? 6u : (array_done ? decoded_layers : 1u));
                            for (size_t t = 0; t < decoded_texels; t++) {
                                uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                                const float fc[3] = { prosper::gpu::f11_to_float((uint16_t)(v & 0x7FFu)),
                                                      prosper::gpu::f11_to_float((uint16_t)((v >> 11) & 0x7FFu)),
                                                      prosper::gpu::f10_to_float((uint16_t)((v >> 22) & 0x3FFu)) };
                                for (int c = 0; c < 3; c++)
                                    tp[t * 4 + c] = (fc[c] != fc[c] || fc[c] <= 0.f) ? 0
                                                  : (fc[c] >= 1.f ? 255 : (uint8_t)(fc[c] * 255.f + 0.5f));
                                tp[t * 4 + 3] = 255;
                            }
                        }
                        // Packed R10G10B10A2 UNORM (GFX10 IMG_FMT 50 / "2_10_10_10_UNORM"):
                        // the generic 4-B read and detile above preserve packed texels; normalize each
                        // field to the RGBA8 image format used by this renderer before sampling.
                        if (!rtt_hit && !dcc_fast_clear_done && !portable_raw_uvec4_storage &&
                            r.format == prosper::gpu::DataFormat::Unorm2_10_10_10) {
                            uint8_t* tp = texture_pixels.data();
                            const size_t decoded_texels =
                                volume_texels * (cube_done ? 6u : (array_done ? decoded_layers : 1u));
                            for (size_t t = 0; t < decoded_texels; t++) {
                                uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                                prosper::gpu::unorm2_10_10_10_to_rgba8(v, tp + t * 4);
                            }
                        }
                        // Focused per-consumer version probe (#586). Hash both the raw guest backing
                        // and the final decoded/RTT-injected pixels so a live draw identifies whether
                        // divergence precedes format conversion or enters through renderer-owned state.
                        // Unset parses to 0x0; require the probe to have been requested (see the
                        // matching guard on the pass hash below).
                        if (resource_hash_w && resource_hash_h &&
                            resource_hash_w == tw && resource_hash_h == th) {
                            const size_t raw_size = std::min<size_t>(
                                r.size ? r.size : volume_texels * 4, 64u << 20);
                            std::vector<uint8_t> raw(raw_size, 0);
                            const size_t raw_got = copy_resource(
                                raw.data(), sampled_source_addr, raw.size());
                            auto fnv = [](const uint8_t* data, size_t size) {
                                uint64_t hash = 1469598103934665603ull;
                                for (size_t i = 0; i < size; ++i) {
                                    hash ^= data[i];
                                    hash *= 1099511628211ull;
                                }
                                return hash;
                            };
                            const uint64_t raw_hash = fnv(raw.data(), raw_got);
                            const uint64_t sample_hash = fnv(texture_pixels.data(), texture_pixels.size());
                            const std::vector<uint8_t> inspected = inspection_rgba8(
                                texture_pixels, tw, th, fr.texture_format);
                            size_t rgb_nonblack = 0, alpha_nonzero = 0;
                            for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                                rgb_nonblack += inspected[p] != 0 || inspected[p + 1] != 0 ||
                                                inspected[p + 2] != 0;
                                alpha_nonzero += inspected[p + 3] != 0;
                            }
                            const auto writer = prosper::gpu::last_guest_write_overlap(
                                sampled_source_addr, raw_size);
                            fprintf(stderr,
                                    "[resource-version] render-submit=%llu draw=%llu order=%llu set=%u bind=%u "
                                    "addr=0x%llx dims=%ux%u class=%u fmt=%u tile=%u dcc=%u meta=0x%llx rtt=%d "
                                    "raw=%zu/%zu:%016llx sample=%zu:%016llx rgb_nonblack=%zu alpha_nonzero=%zu "
                                    "writer=%s/%llu/%llu/%llu/0x%llx\n",
                                    (unsigned long long)g_this_submit,
                                    (unsigned long long)draw.draw_index,
                                    (unsigned long long)draw.command_order,
                                    set, r.binding, (unsigned long long)r.gpu_addr, tw, th,
                                    (unsigned)r.cls, (unsigned)r.format, r.tile_mode,
                                    r.compression_enabled, (unsigned long long)r.metadata_addr,
                                    (int)rtt_hit,
                                    raw_got, raw_size, (unsigned long long)raw_hash,
                                    texture_pixels.size(), (unsigned long long)sample_hash,
                                    rgb_nonblack, alpha_nonzero,
                                    writer ? prosper::gpu::guest_writer_kind_name(writer->kind) : "none",
                                    (unsigned long long)(writer ? writer->submit : 0),
                                    (unsigned long long)(writer ? writer->item : 0),
                                    (unsigned long long)(writer ? writer->order : 0),
                                    (unsigned long long)(writer ? writer->identity : 0));
                            if (PROSPER_ENV_ON("PROSPER_DUMP_RESOURCE_VERSION")) {
                                static std::set<std::pair<uint64_t, uint64_t>> dumped_versions;
                                if (dumped_versions.emplace(r.gpu_addr, sample_hash).second) {
                                    const char* dd = getenv("PROSPER_FRAME_DIR");
                                    char fn[512];
                                    snprintf(fn, sizeof fn, "%s/resource_%llx_%ux%u_%016llx.bmp",
                                             dd && *dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th,
                                             (unsigned long long)sample_hash);
                                    prosper::test::dump_bmp(fn, inspected, tw, th);
                                    fprintf(stderr, "[resource-version] dumped decoded sample -> %s\n", fn);
                                }
                            }
                        }
                        // PROSPER_PALETTELOG: compact identity/provenance trace for Unity's 256x16
                        // palette textures. Unlike GFXLOG this is cheap enough for a focused render
                        // window and reveals both descriptor-address and decoded-content changes.
                        if (PROSPER_ENV_ON("PROSPER_PALETTELOG") && tw == 256 && th == 16 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            uint64_t hash = 1469598103934665603ull;
                            size_t rgb_nonblack = 0;
                            for (size_t t = 0; t < (size_t)tw * th; ++t) {
                                const uint8_t* p = &texture_pixels[t * 4];
                                rgb_nonblack += (p[0] != 0 || p[1] != 0 || p[2] != 0);
                                for (unsigned c = 0; c < 4; ++c) {
                                    hash ^= p[c]; hash *= 1099511628211ull;
                                }
                            }
                            fprintf(stderr, "[palette] binding=%u addr=0x%llx fnv=%016llx rgb_nonblack=%zu\n",
                                    r.binding, (unsigned long long)r.gpu_addr,
                                    (unsigned long long)hash, rgb_nonblack);
                        }
                        // Apply the synthetic texture after every decode/conversion step. Applying it before
                        // auto-detile let the real tiled bytes overwrite the checker, producing a false-negative
                        // sampling diagnosis (#522).
                        const char* test_texture = PROSPER_ENV_VALUE("PROSPER_TESTTEX");
                        const char* test_texture_binding = PROSPER_ENV_VALUE("PROSPER_TESTTEX_BINDING");
                        const char* test_texture_draw = PROSPER_ENV_VALUE("PROSPER_TESTTEX_DRAW");
                        const bool test_this_texture = test_texture &&
                            (!test_texture_binding ||
                             strtoul(test_texture_binding, nullptr, 0) == r.binding) &&
                            (!test_texture_draw ||
                             strtoull(test_texture_draw, nullptr, 0) == draw.draw_index);
                        if (test_this_texture) {
                            const uint32_t slices = is_volume ? std::max(r.depth, 1u) : 1u;
                            if (!strcmp(test_texture, "zero")) {
                                std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                            } else if (fr.texture_format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            uint16_t* p = reinterpret_cast<uint16_t*>(
                                                texture_pixels.data() +
                                                (((size_t)z * th + y) * tw + x) * 8);
                                            const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            p[0] = prosper::gpu::float_to_half((float)x / tw);
                                            p[1] = prosper::gpu::float_to_half((float)y / th);
                                            p[2] = prosper::gpu::float_to_half(ck ? 0.8f : 0.16f);
                                            p[3] = prosper::gpu::float_to_half(1.0f);
                                        }
                            } else if (fr.texture_format == VK_FORMAT_R8_UNORM) {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            texture_pixels[((size_t)z * th + y) * tw + x] =
                                                ck ? 200 : 40;
                                        }
                            } else if (fr.texture_format == VK_FORMAT_R8G8_UNORM) {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            uint8_t* p = &texture_pixels[
                                                (((size_t)z * th + y) * tw + x) * 2];
                                            p[0] = ck ? 200 : 40;
                                            p[1] = ck ? 40 : 200;
                                        }
                            } else if (fr.texture_format ==
                                       VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            const float values[3] = {
                                                static_cast<float>(x) / tw,
                                                static_cast<float>(y) / th,
                                                ck ? 8.0f : 0.16f,
                                            };
                                            const uint32_t packed =
                                                static_cast<uint32_t>(
                                                    prosper::gpu::float_to_f11(values[0])) |
                                                (static_cast<uint32_t>(
                                                     prosper::gpu::float_to_f11(values[1])) << 11) |
                                                (static_cast<uint32_t>(
                                                     prosper::gpu::float_to_f10(values[2])) << 22);
                                            std::memcpy(
                                                texture_pixels.data() +
                                                    (((size_t)z * th + y) * tw + x) * 4,
                                                &packed, sizeof(packed));
                                        }
                            } else {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            uint8_t* p = &texture_pixels[
                                                (((size_t)z * th + y) * tw + x) * 4];
                                            bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            p[0] = (uint8_t)(255 * x / tw);
                                            p[1] = (uint8_t)(255 * y / th);
                                            p[2] = ck ? 200 : 40;
                                            p[3] = 255;
                                        }
                            }
                        }
                        // This 256x16 sparse palette is addressed like 16 blue slices, each 16 texels wide.
                        // The identity probe preserves source color through shaders using
                        // u=r/17+b*15/16, v=1-g, distinguishing lookup contents from a broken
                        // source/geometry/sample path (#522).
                        if (PROSPER_ENV_ON("PROSPER_TESTLUT") && tw == 256 && th == 16 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                                p[0] = (uint8_t)((x % 16) * 255 / 15);
                                p[1] = (uint8_t)((15 - y) * 255 / 15);
                                p[2] = (uint8_t)((x / 16) * 255 / 15);
                                p[3] = 255;
                            }
                        }
                        // Unity's post-processing stack flattens a 32^3 grading LUT into a 1024x32
                        // strip (32 red samples per blue slice). This isolates a missing LUT producer
                        // from the persistent post shader and its healthy scene input (#522).
                        if (PROSPER_ENV_ON("PROSPER_TESTLUT32") && tw == 1024 && th == 32 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            for (uint32_t y = 0; y < th; ++y) for (uint32_t x = 0; x < tw; ++x) {
                                uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                                p[0] = (uint8_t)((x % 32) * 255 / 31);
                                p[1] = (uint8_t)(y * 255 / 31);
                                p[2] = (uint8_t)((x / 32) * 255 / 31);
                                p[3] = 255;
                            }
                        }
                        // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                        // bypassing the shader — reveals whether the render target is tiled or linear.
                        if (PROSPER_ENV_ON("PROSPER_DUMP_TEX") && !texture_pixels.empty() && frame_no < 200 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                            prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                        }
                        // PROSPER_DUMP_ATLAS: dump each SMALL sampled texture once per ADDRESS (find the caption
                        // font among same-size UI textures). Capped.
                        if (PROSPER_ENV_ON("PROSPER_DUMP_ATLAS") && !texture_pixels.empty() && tw <= 2048 && th <= 1024 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            static std::unordered_map<uint64_t,int> seen; static int ndumped = 0;
                            if (seen[r.gpu_addr]++ == 0 && ndumped++ < 60) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/tex_%ux%u_%llx_c%u.bmp", d.c_str(), tw, th,
                                                       (unsigned long long)r.gpu_addr, r.num_components);
                                prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            }
                        }
                        // PROSPER_KILL_RING (#1186): null out the concentric-ring light-glow texture
                        // (a 1024x1024 single-channel glow sprite) to A/B whether it is what draws the
                        // see-through concentric-circles pattern over the world. Run with
                        // PROSPER_NO_TEXTURE_DECODE_CACHE=1 so every sample takes this decode path.
                        if (PROSPER_ENV_ON("PROSPER_KILL_RING") && tw == 1024 && th == 1024 &&
                            r.num_components == 1 && r.cls == RC::Texture)
                            std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                        if (!fr.tex_rgba) {
                            fr.tex_rgba = texture_pixels.data();
                            decoded_pixels_in_texstore = true;
                        }
                        fr.tw = tw; fr.th = cube_done ? th * 6u : th;
                        // #325: `sample_count` IS the backend's array-layer channel -- its own field
                        // comment says so for MSAA-as-array -- and the view already selects
                        // VK_IMAGE_VIEW_TYPE_2D_ARRAY whenever it exceeds 1. Set it only once the
                        // slices are actually decoded: claiming layers we did not fill would upload
                        // uninitialized memory, and the backend's span contract would reject it
                        // anyway.
                        if (array_done) {
                            fr.sample_count = decoded_layers;
                            fr.tex_byte_size = texture_pixels.size();
                        }
                        fr.td = is_volume ? r.depth : 1u;
                        fr.img_dim = r.img_dim;
                        // #1272: see the reuse path — plain 2D guest textures only.
                        if (!is_volume && !cube_done)
                            fr.declared_mip_levels = r.declared_mip_levels;
                        if (persistent_cache_eligible) {
                            size_t source_prefix_size = linear_source_prefix_size;
                            if (!persistent_source_matches_pixels && persistent_source_size) {
                                persistent_validation_scratch.resize(persistent_source_size);
                                source_prefix_size = copy_persistent_source(
                                    persistent_validation_scratch.data(),
                                    persistent_source_size);
                            }
                            auto old = persistent_decoded_textures.find(decode_key);
                            // PROSPER_NO_TEXTURE_PREFIX_INHERIT restores the `assign()` spelling:
                            // free the outgoing buffer, allocate a new one. It exists because
                            // WITHOUT it this optimisation has no off-switch, and a discriminator
                            // that cannot disable everything the change does is not a
                            // discriminator -- it silently under-reports the change it is
                            // attributing. Measured on Stray: an A/B armed with
                            // PROSPER_NO_DIRECT_TEXTURE_SOURCE and PROSPER_DECODE_SCRATCH_MB=0
                            // returned the texture leaf to 918.2 ms rather than the 1110.1 ms
                            // baseline, and the missing 191.9 ms was this, still switched on.
                            static const bool no_prefix_inherit =
                                PROSPER_ENV_VALUE("PROSPER_NO_TEXTURE_PREFIX_INHERIT") != nullptr;
                            // The entry this decode replaces holds a source_prefix buffer of the
                            // same shape. Taking its allocation instead of letting `assign` build a
                            // new one is worth naming on a surface the guest rewrites every frame:
                            // for Stray's 63.75 MiB HDR intermediate that pair -- free 63.75 MiB,
                            // then mmap and fault 63.75 MiB back in for the same bytes -- measured
                            // 13.00 ms per invalidation on this host, against 3.5 ms for the copy
                            // it exists to hold. The bytes stored are identical either way; only
                            // the mapping's lifetime changes.
                            std::vector<uint8_t> inherited_source_prefix;
                            uint32_t inherited_watch_dirty_count = 0;
                            uint32_t inherited_watch_stable_validations = 0;
                            bool inherited_watch_disabled = false;
                            prosper::host::GuestWriteWatch inherited_source_watch =
                                std::move(pending_source_watch);
                            uint64_t inherited_persistent_id = 0;
                            uint64_t inherited_persistent_version = 0;
                            if (old != persistent_decoded_textures.end() &&
                                old->second.source_addr == persistent_source_addr &&
                                old->second.source_size == persistent_source_size) {
                                inherited_watch_dirty_count = old->second.source_watch_dirty_count;
                                inherited_watch_stable_validations =
                                    old->second.source_watch_stable_validations;
                                inherited_watch_disabled = old->second.source_watch_disabled;
                                if (!inherited_watch_disabled)
                                    inherited_source_watch = std::move(old->second.source_watch);
                                inherited_persistent_id = old->second.persistent_id;
                                inherited_persistent_version = old->second.persistent_version;
                            }
                            // An identity already established earlier in this submit may be re-decoded
                            // if an interleaved guest GPU write mutated its backing. Only if it was used in
                            // the current submit (last_use == decode_generation) might earlier draws hold
                            // raw pointers to those bytes, so those allocations are retained in submit-scoped
                            // `retired_submit_pixels` until the submit scope completes (#3283).
                            // Retired bytes are tracked separately in `retired_submit_bytes` and deliberately
                            // held resident off-ledger for the submit's duration to ensure pointer safety.
                            if (old != persistent_decoded_textures.end()) {
                                if (old->second.pixels && old->second.last_use == decode_generation) {
                                    retired_submit_bytes += old->second.bytes();
                                    retired_submit_pixels.push_back(old->second.pixels);
                                }
                                persistent_decoded_texture_bytes -= old->second.bytes();
                                // Taken AFTER the ledger has been debited by the old entry's
                                // `bytes()`, which reads `source_prefix.size()`: moving first would
                                // debit zero and leak the entry's bytes from the budget forever.
                                if (!no_prefix_inherit)
                                    inherited_source_prefix = std::move(old->second.source_prefix);
                                persistent_decoded_textures.erase(old);
                            }
                            const size_t required =
                                (persistent_source_matches_pixels ? 0 : source_prefix_size) +
                                texture_pixels.size();
                            while (required <= persistent_decode_limit &&
                                   persistent_decoded_texture_bytes > persistent_decode_limit - required) {
                                auto victim = persistent_decoded_textures.end();
                                for (auto it = persistent_decoded_textures.begin();
                                     it != persistent_decoded_textures.end(); ++it) {
                                    if (it->second.last_use == decode_generation) continue;
                                    if (victim == persistent_decoded_textures.end() ||
                                        it->second.last_use < victim->second.last_use) victim = it;
                                }
                                if (victim == persistent_decoded_textures.end()) break;
                                // Victims are from prior submits (last_use < decode_generation) with no live
                                // references in the current submit, so erasing them releases their memory immediately.
                                persistent_decoded_texture_bytes -= victim->second.bytes();
                                persistent_decoded_textures.erase(victim);
                            }
                            if (required <= persistent_decode_limit &&
                                persistent_decoded_texture_bytes <= persistent_decode_limit - required) {
                                PersistentDecodedTexture cached;
                                cached.source_addr = persistent_source_addr;
                                cached.source_size = persistent_source_size;
                                cached.source_prefix_size = source_prefix_size;
                                cached.source_matches_pixels = persistent_source_matches_pixels;
                                if (!persistent_source_matches_pixels) {
                                    // Byte-for-byte what `assign(scratch.begin(),
                                    // scratch.begin() + source_prefix_size)` produced. `resize`
                                    // before the copy so a SHORT read stores exactly the prefix it
                                    // read -- `source_prefix.size()` is compared against
                                    // `persistent_source_size` on the validation path, so growing
                                    // it to the inherited buffer's extent would silently change
                                    // which entries revalidate. Shrinking keeps the capacity, which
                                    // is the whole point.
                                    cached.source_prefix = std::move(inherited_source_prefix);
                                    cached.source_prefix.resize(source_prefix_size);
                                    if (source_prefix_size)
                                        std::memcpy(cached.source_prefix.data(),
                                                    persistent_validation_scratch.data(),
                                                    source_prefix_size);
                                }
                                cached.pixels = std::make_shared<const std::vector<uint8_t>>(
                                    std::move(texture_pixels));
                                cached.output_height = fr.th;
                                cached.layers = fr.sample_count;
                                cached.narrow = narrow_done;
                                cached.last_use = decode_generation;
                                cached.persistent_id = inherited_persistent_id;
                                if (!cached.persistent_id) {
                                    cached.persistent_id = ++persistent_texture_id;
                                    if (!cached.persistent_id)
                                        cached.persistent_id = ++persistent_texture_id;
                                }
                                cached.persistent_version = inherited_persistent_version + 1;
                                if (!cached.persistent_version) cached.persistent_version = 1;
                                cached.texture_format = fr.texture_format;
                                cached.storage_image_contract_valid =
                                    fr.storage_image_contract_valid;
                                cached.validation_snapshot =
                                    prosper::gpu::guest_gpu_write_snapshot();
                                cached.source_watch_dirty_count = inherited_watch_dirty_count;
                                cached.source_watch_stable_validations =
                                    inherited_watch_stable_validations;
                                cached.source_watch_disabled = inherited_watch_disabled;
                                cached.source_watch = std::move(inherited_source_watch);
                                auto [inserted, ok] = persistent_decoded_textures.emplace(
                                    decode_key, std::move(cached));
                                if (ok)
                                    persistent_decoded_texture_bytes += inserted->second.bytes();
                                fr.tex_rgba = inserted->second.pixels ? inserted->second.pixels->data() : nullptr;
                                fr.tex_rgba_owner = inserted->second.pixels;
                                fr.persistent_texture_id = inserted->second.persistent_id;
                                fr.persistent_texture_version = inserted->second.persistent_version;
                                decoded_pixels_in_texstore = false;
                            }
                        }
                        if (!resource_rtt_hit && !resource_compute_image_hit && !has_live_rtt) {
                            const bool pin_scratch =
                                decoded_pixels_in_texstore && cross_span_source_size != 0;
                            const size_t pinned_slot = pin_scratch ? texture_slot : SIZE_MAX;
                            if (pin_scratch) {
                                texstore_pinned[texture_slot] = true;
                                ++g_texture_decode_scope.scratch_pins;
                            }
                            decoded_textures.insert_or_assign(
                                decode_key, DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                                          fr.persistent_texture_id,
                                                          fr.persistent_texture_version,
                                                          decode_span_ordinal,
                                                          prosper::gpu::guest_gpu_write_snapshot(),
                                                          persistent_source_addr,
                                                          cross_span_source_size, pinned_slot,
                                                          fr.tex_rgba_owner, fr.texture_format,
                                                          fr.storage_image_contract_valid,
                                                          fr.sample_count, fr.tex_byte_size});
                        }
                        }
                        if (native_r32ui_storage && writable_storage_image) {
                            const uint32_t writeback_pitch = PROSPER_ENV_VALUE("PROSPER_PITCH")
                                ? static_cast<uint32_t>(atoi(getenv("PROSPER_PITCH"))) : 0u;
                            const size_t linear_bytes = static_cast<size_t>(tw) * th * 4u;
                            const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                            const size_t guest_bytes = r.in_mip_tail
                                ? r.mip_tail_bytes
                                : (tiled
                                       ? prosper::gpu::tiled_surface_bytes(
                                             tw, th, r.tile_mode, writeback_pitch, 4u)
                                       : linear_bytes);
                            // The exact Astro atomic surface is a base-level 2D R32_UINT image. Keep
                            // the callback fail-closed for malformed/short replay backing; losing a
                            // write is preferable to overrunning an unrelated guest allocation.
                            const bool backing_fits = guest_bytes &&
                                (!r.host_data || guest_bytes <= r.host_data_size);
                            if (backing_fits) {
                                const uint64_t guest_addr = r.gpu_addr;
                                uint8_t* const replay_data = r.host_data;
                                const uint32_t tile_mode = r.tile_mode;
                                const bool in_mip_tail = r.in_mip_tail;
                                const uint32_t mip_tail_x = r.mip_tail_x;
                                const uint32_t mip_tail_y = r.mip_tail_y;
                                fr.storage_image_writeback =
                                    // `decoded_textures` and `texstore_pinned` are thread-local
                                    // statics since #1691 and are deliberately NOT captured: a
                                    // variable with static storage duration cannot appear in a
                                    // capture list (clang rejects it outright), and the body reaches
                                    // the render thread's own instances directly — which is the
                                    // right identity, since this callback runs on that thread.
                                    [guest_addr, replay_data, guest_bytes, linear_bytes,
                                     tw, th, tile_mode,
                                     writeback_pitch, in_mip_tail, mip_tail_x,
                                     mip_tail_y](const uint8_t* pixels, size_t bytes) {
                                        if (!pixels || bytes != linear_bytes) return;
                                        uint8_t* destination = replay_data;
                                        if (!destination) {
                                            if (guest_bytes > UINT32_MAX ||
                                                !prosper::gpu::guest_writable(
                                                    guest_addr,
                                                    static_cast<uint32_t>(guest_bytes)))
                                                return;
                                            prosper::host::guest_write_watch_notify_host_write(
                                                guest_addr, guest_bytes);
                                            destination = reinterpret_cast<uint8_t*>(
                                                static_cast<uintptr_t>(guest_addr));
                                        }
                                        if (in_mip_tail) {
                                            prosper::gpu::tile_surface_level(
                                                destination, guest_bytes, pixels, tw, th,
                                                tile_mode, 4u, mip_tail_x, mip_tail_y);
                                        } else {
                                            prosper::gpu::tile_surface(
                                                destination, pixels, tw, th, tile_mode,
                                                writeback_pitch, 4u);
                                        }
                                        if (guest_addr) {
                                            // Named for the same reason as the compute writebacks:
                                            // a renderer writeback into guest memory is prosper's
                                            // own store, and reads as an anonymous `gpu` write in
                                            // every cache-invalidation diagnostic downstream.
                                            prosper::gpu::set_guest_gpu_write_origin(
                                                "renderer-writeback(rtt-guest-bytes)");
                                            prosper::gpu::notify_guest_gpu_write(
                                                guest_addr, guest_bytes);
                                            prosper::gpu::set_guest_gpu_write_origin(nullptr);
                                        }
                                        // The same guest range can already have a sampled-image decode
                                        // under a different cache key/class. Its pixel representation
                                        // cannot be patched byte-for-byte from R32_UINT, so discard every
                                        // overlapping submit-local decode and let the next pass rebuild it
                                        // from the now-current guest bytes.
                                        if (guest_addr) {
                                            for (auto it = decoded_textures.begin();
                                                 it != decoded_textures.end();) {
                                                const uint64_t cached_addr = it->first.gpu_addr;
                                                const uint64_t cached_bytes = std::max<uint64_t>(
                                                    std::max<uint64_t>(it->first.size,
                                                                       it->first.source_span_bytes),
                                                    1u);
                                                const bool overlaps = cached_addr <= guest_addr
                                                    ? guest_addr - cached_addr < cached_bytes
                                                    : cached_addr - guest_addr < guest_bytes;
                                                if (overlaps) {
                                                    if (it->second.texstore_slot <
                                                        texstore_pinned.size())
                                                        texstore_pinned[
                                                            it->second.texstore_slot] = false;
                                                    it = decoded_textures.erase(it);
                                                } else {
                                                    ++it;
                                                }
                                            }
                                        }
                                    };
                            }
                        }
                        if (portable_raw_uvec4_storage && writable_storage_image &&
                            fr.storage_image_contract_valid) {
                            const bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                                !PROSPER_ENV_VALUE("PROSPER_NODETILE");
                            const uint32_t writeback_tile_mode = tiled ? r.tile_mode : 0u;
                            const uint32_t writeback_depth = is_volume ? r.depth : 1u;
                            const uint32_t components =
                                r.num_components ? r.num_components : 1u;
                            const size_t guest_bytes = storage_image_raw_uvec4_source_bytes(
                                r.format, components, tw, th, writeback_depth,
                                writeback_tile_mode, r.in_mip_tail, r.mip_tail_bytes);
                            const size_t texels = static_cast<size_t>(tw) * th * writeback_depth;
                            const size_t expanded_bytes = texels <= SIZE_MAX / 16u
                                ? texels * 16u : 0u;
                            const bool backing_fits = guest_bytes && expanded_bytes &&
                                (!r.host_data || guest_bytes <= r.host_data_size);
                            if (!backing_fits) {
                                fr.storage_image_contract_valid = false;
                            } else {
                                const uint64_t guest_addr = r.gpu_addr;
                                uint8_t* const replay_data = r.host_data;
                                const auto format = r.format;
                                const bool in_mip_tail = r.in_mip_tail;
                                const uint32_t mip_tail_bytes = r.mip_tail_bytes;
                                const uint32_t mip_tail_x = r.mip_tail_x;
                                const uint32_t mip_tail_y = r.mip_tail_y;
                                fr.storage_image_writeback =
                                    [guest_addr, replay_data, guest_bytes, expanded_bytes,
                                     format, components, tw, th, writeback_depth,
                                     writeback_tile_mode, in_mip_tail, mip_tail_bytes,
                                     mip_tail_x, mip_tail_y](const uint8_t* pixels, size_t bytes) {
                                        if (!pixels || bytes != expanded_bytes) return;
                                        uint8_t* destination = replay_data;
                                        if (!destination) {
                                            if (guest_bytes > UINT32_MAX ||
                                                !prosper::gpu::guest_writable(
                                                    guest_addr,
                                                    static_cast<uint32_t>(guest_bytes)))
                                                return;
                                            prosper::host::guest_write_watch_notify_host_write(
                                                guest_addr, guest_bytes);
                                            destination = reinterpret_cast<uint8_t*>(
                                                static_cast<uintptr_t>(guest_addr));
                                        }
                                        if (!storage_image_writeback_raw_uvec4(
                                                reinterpret_cast<const uint32_t*>(pixels),
                                                bytes / sizeof(uint32_t), format, components,
                                                tw, th, writeback_depth, writeback_tile_mode,
                                                in_mip_tail, mip_tail_bytes,
                                                mip_tail_x, mip_tail_y,
                                                destination, guest_bytes))
                                            return;
                                        if (guest_addr) {
                                            prosper::gpu::set_guest_gpu_write_origin(
                                                "renderer-writeback(portable-uvec4)");
                                            prosper::gpu::notify_guest_gpu_write(
                                                guest_addr, guest_bytes);
                                            prosper::gpu::set_guest_gpu_write_origin(nullptr);
                                            for (auto it = decoded_textures.begin();
                                                 it != decoded_textures.end();) {
                                                const uint64_t cached_addr = it->first.gpu_addr;
                                                const uint64_t cached_bytes = std::max<uint64_t>(
                                                    std::max<uint64_t>(it->first.size,
                                                                       it->first.source_span_bytes),
                                                    1u);
                                                const bool overlaps = cached_addr <= guest_addr
                                                    ? guest_addr - cached_addr < cached_bytes
                                                    : cached_addr - guest_addr < guest_bytes;
                                                if (overlaps) {
                                                    if (it->second.texstore_slot <
                                                        texstore_pinned.size())
                                                        texstore_pinned[
                                                            it->second.texstore_slot] = false;
                                                    it = decoded_textures.erase(it);
                                                } else {
                                                    ++it;
                                                }
                                            }
                                        }
                                    };
                            }
                        }
                        // Carry the decoded S# sampler state (filter/wrap/mip) so the pipeline samples
                        // the way the game asked instead of using fixed LINEAR/clamp state.
                        fr.mag_filter = r.mag_filter; fr.min_filter = r.min_filter; fr.mip_filter = r.mip_filter;
                        // Draw/binding-scoped sampler A/B. This shares TESTTEX's selectors but leaves
                        // the sampled pixels intact, isolating descriptor filtering from texture content.
                        const char* test_filter = PROSPER_ENV_VALUE("PROSPER_TESTTEX_FILTER");
                        const char* test_filter_binding = PROSPER_ENV_VALUE("PROSPER_TESTTEX_BINDING");
                        const char* test_filter_draw = PROSPER_ENV_VALUE("PROSPER_TESTTEX_DRAW");
                        const bool filter_valid = test_filter &&
                            (!strcmp(test_filter, "linear") || !strcmp(test_filter, "point"));
                        if (filter_valid &&
                            (!test_filter_binding ||
                             strtoul(test_filter_binding, nullptr, 0) == r.binding) &&
                            (!test_filter_draw ||
                             strtoull(test_filter_draw, nullptr, 0) == draw.draw_index)) {
                            const uint32_t filter = !strcmp(test_filter, "linear") ? 1u : 0u;
                            fr.mag_filter = fr.min_filter = filter;
                        }
                        fr.addr_uvw[0] = r.addr_uvw[0]; fr.addr_uvw[1] = r.addr_uvw[1]; fr.addr_uvw[2] = r.addr_uvw[2];
                        // Remaining S# sampler fields (#262): border color + LOD clamp/bias (applied where
                        // valid; compare/unnorm stay on ShaderResource for shader-side lowering).
                        fr.border_color_type = r.border_color_type;
                        fr.min_lod = r.min_lod; fr.max_lod = r.max_lod; fr.lod_bias = r.lod_bias;
                        // Anisotropy ratio (#275): applied in render_runner.h when the device supports the
                        // samplerAnisotropy feature and filtering is linear. The Messenger decodes ratio 0
                        // (isotropic) so this is a no-op for it; carries correct behavior for titles that
                        // request anisotropic filtering.
                        fr.max_aniso_ratio = r.max_aniso_ratio;
                        // T# DST_SEL channel remap (#261): narrow coverage paths already broadcast to
                        // every channel, so keep them identity. AvPlayer RG8 preserved U/V above and
                        // still needs its T# mapping (R,G,0,1).
                        if (native_r8_sampled) {
                            fr.swizzle[0]=4; fr.swizzle[1]=4;
                            fr.swizzle[2]=4; fr.swizzle[3]=4;
                        }
                        else if (narrow_done && !avplayer_chroma_layout) {
                            fr.swizzle[0]=4; fr.swizzle[1]=5; fr.swizzle[2]=6; fr.swizzle[3]=7;
                        }
                        else { for (int k=0;k<4;k++) fr.swizzle[k] = r.swizzle[k]; }
                        if (resource_rtt_hit && live_rtt != g_rtt.end())
                            fr.render_target_guest_format = live_rtt->second.guest_format;
                        // PROSPER_ALPHA1: force the sampled alpha to constant 1 (opaque). Diagnostic for a
                        // black scene whose textures decode to real RGB but composite to nothing — if the
                        // level appears with this, the alpha channel (decode or DST_SEL swizzle) is the bug (#300).
                        if (PROSPER_ENV_ON("PROSPER_ALPHA1")) fr.swizzle[3] = 1;
                        }
                    } else {
                        fr.buffer_identity = r.gpu_addr;
                        const prosper::gpu::StorageBufferMaterializationPlan materialization =
                            prosper::gpu::plan_storage_buffer_materialization(
                                *reflected_binding, r);
                        if (!materialization.valid) {
                            fprintf(stderr,
                                    "[buffer-materialization-reject] set=%u binding=%u addr=%llx "
                                    "declared=%u\n",
                                    set, r.binding, (unsigned long long)r.gpu_addr, r.size);
                            // Name the sub-condition, once per (set,binding,address). The line above
                            // collapses every reason into one string, and an investigation cannot act
                            // on that: "the descriptor carries a partial zero-pad marker set" and
                            // "the resource is the wrong class" are different pieces of work. Same
                            // shape and the same justification as [mimg-mip] in rdna2_emit_alu.cpp.
                            //
                            // Deduped rather than gated, because the volume is the whole problem:
                            // Dragon Quest VII emits 549,623 of the line above in one routed run,
                            // ALL of them for a single address (0x20013f1bc0, set 0 binding 10) with
                            // 257 different declared sizes -- so the useful signal is one line
                            // describing that binding, not half a million repetitions of it (#1486).
                            {
                                static std::mutex why_mutex;
                                static std::set<std::tuple<uint32_t, uint32_t, uint64_t>> why_seen;
                                bool first_why = false;
                                {
                                    std::lock_guard<std::mutex> lock(why_mutex);
                                    first_why = why_seen.emplace(set, r.binding,
                                                                 (uint64_t)r.gpu_addr).second;
                                }
                                if (first_why) {
                                    const auto& d = *reflected_binding;
                                    const bool has_logical = d.zero_pad_logical_bytes != 0;
                                    const bool has_binding_pad = d.zero_pad_binding_bytes != 0;
                                    const bool has_semantic =
                                        d.zero_pad_semantic !=
                                        prosper::gpu::StorageBufferTailSemantic::None;
                                    fprintf(stderr,
                                            "[buffer-why] set=%u binding=%u addr=%llx declared=%u "
                                            "markers=%d/%d/%d kind=%d readable=%d writable=%d "
                                            "atomic=%d dynamic=%d required=%llu "
                                            "pad_logical=%u pad_binding=%u semantic=%d "
                                            "res_cls=%d fmt=%d ncomp=%u stride=%u size=%u "
                                            "scalar_dwords=%u\n",
                                            set, r.binding, (unsigned long long)r.gpu_addr, r.size,
                                            (int)has_logical, (int)has_binding_pad,
                                            (int)has_semantic,
                                            (int)d.kind, (int)d.readable, (int)d.writable,
                                            (int)d.atomic_access, (int)d.dynamic_access,
                                            (unsigned long long)d.required_bytes,
                                            d.zero_pad_logical_bytes, d.zero_pad_binding_bytes,
                                            (int)d.zero_pad_semantic,
                                            (int)r.cls, (int)r.format, r.num_components,
                                            r.stride, r.size, r.scalar_buffer_dword_count);
                                }
                            }
                            continue;
                        }
                        // #1427: the guest's declared V#/V-buffer size is the real requirement — a
                        // vertex fetch indexes anywhere inside it. The old 1 MiB clamp silently
                        // truncated larger buffers, so every element past the cap read ZEROS: those
                        // vertices all transformed to the same clip point (the MVP translation
                        // column) and the primitive died as degenerate, with no reject and no log.
                        // On Blue Prince's entrance hall that erased 44 of 248 scene draws —
                        // the tile floor, the far table, most of the room — and read as a shading
                        // defect for weeks. Upload the declared range under a ceiling that exists
                        // only to bound a corrupt descriptor (a 64 MiB read also costs ~16K
                        // guest_readable page probes, so it must stay bounded), and make any
                        // truncation that does happen FAIL-VISIBLE. PROSPER_MAX_BUFFER_UPLOAD_MB
                        // lowers the ceiling for a same-build A/B of this exact defect.
                        const uint32_t requested_bytes = materialization.binding_bytes
                            ? static_cast<uint32_t>(materialization.binding_bytes) : 256u;
                        uint32_t nb = materialization.zero_padded_tail
                            ? static_cast<uint32_t>(materialization.binding_bytes)
                            : prosper::frontend::buffer_upload_bytes(requested_bytes);
                        if (nb < (requested_bytes & ~3u)) {
                            static std::set<uint64_t> truncated_reported;
                            if (truncated_reported.size() < 32 &&
                                truncated_reported.insert(r.gpu_addr).second)
                                fprintf(stderr,
                                        "[buffer-truncated] set=%u binding=%u addr=%llx declared=%u "
                                        "uploaded=%u — fetches past the uploaded range read zeros and "
                                        "collapse geometry (#1427)\n",
                                        set, r.binding, (unsigned long long)r.gpu_addr,
                                        requested_bytes, nb);
                        }
                        // A definitely unmapped source cannot contribute a byte. Preserve the
                        // renderer's established all-zero fallback without allocating/probing the
                        // descriptor's potentially corrupt declared size; robust buffer access makes
                        // accesses beyond this minimum zero as well. Static reflection tells us how
                        // much in-bounds storage the shader can definitely address.
                        const bool unavailable_guest_buffer = !r.host_data &&
                            (r.gpu_addr < 0x1000 ||
                             prosper_reserved_range_state(r.gpu_addr) == 0);
                        if (materialization.zero_padded_tail) {
                            uint8_t logical[2] = {};
                            const uint8_t* logical_source = nullptr;
                            if (r.host_data && r.host_data_size >= sizeof(logical)) {
                                logical_source = r.host_data;
                            } else if (!r.host_data && !unavailable_guest_buffer &&
                                       copy_resource(logical, r.gpu_addr, sizeof(logical)) ==
                                           sizeof(logical)) {
                                logical_source = logical;
                            }
                            fr.dwords.assign(1, 0);
                            if (!logical_source ||
                                !prosper::gpu::materialize_storage_buffer_bytes(
                                    materialization, logical_source, sizeof(logical),
                                    reinterpret_cast<uint8_t*>(fr.dwords.data()),
                                    sizeof(uint32_t))) {
                                fprintf(stderr,
                                        "[buffer-materialization-reject] set=%u binding=%u "
                                        "two-byte source unavailable\n",
                                        set, r.binding);
                                continue;
                            }
                        } else if (unavailable_guest_buffer) {
                            const uint64_t minimum_bytes = std::min<uint64_t>(
                                std::max<uint64_t>(reflected_binding->required_bytes, 256u),
                                kMaxBufferUploadBytes);
                            fr.dwords.assign(static_cast<size_t>((minimum_bytes + 3u) / 4u), 0);
                        } else if (use_direct_buffer_views && nb >= 4) {
                            const auto probe_start = timing_enabled
                                ? RenderClock::now() : RenderClock::time_point{};
                            if (const uint8_t* source = direct_resource(r.gpu_addr, nb)) {
                                fr.dwords_view = reinterpret_cast<const uint32_t*>(source);
                                fr.dwords_view_count = nb / sizeof(uint32_t);
                                resource_buffer_view = true;
                            }
                            if (timing_enabled)
                                resource_buffer_probe_ms =
                                    std::chrono::duration<double, std::milli>(
                                        RenderClock::now() - probe_start).count();
                        }
                        const auto copy_start = timing_enabled
                            ? RenderClock::now() : RenderClock::time_point{};
                        if (!materialization.zero_padded_tail &&
                            !unavailable_guest_buffer && !fr.dwords_view_count &&
                            use_direct_buffer_views) {
                            if (nb >= 4) {
                                fr.dwords.assign(nb / sizeof(uint32_t), 0);
                                if (!copy_resource(reinterpret_cast<uint8_t*>(fr.dwords.data()),
                                                   r.gpu_addr, nb))
                                    fr.dwords.clear();
                            }
                            if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                        } else if (!materialization.zero_padded_tail &&
                                   !unavailable_guest_buffer && !use_direct_buffer_views) {
                            if (nb >= 4) {
                                std::vector<uint8_t> tmp(nb, 0);
                                if (copy_resource(tmp.data(), r.gpu_addr, nb) > 0)
                                    fr.dwords.assign(
                                        reinterpret_cast<const uint32_t*>(tmp.data()),
                                        reinterpret_cast<const uint32_t*>(tmp.data() + nb));
                            }
                            if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                        }
                        if (timing_enabled)
                            resource_buffer_copy_ms = std::chrono::duration<double, std::milli>(
                                RenderClock::now() - copy_start).count();
                        if (render_timing_detail) {
                            const uint64_t detail_min_submit =
                                PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT")
                                    ? strtoull(getenv(
                                          "PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT"), nullptr, 0)
                                    : 0;
                            const double elapsed = std::chrono::duration<double, std::milli>(
                                RenderClock::now() - resource_timing_start).count();
                            static uint64_t detail_buffer_lines = 0;
                            if (static_cast<uint64_t>(g_this_submit) >= detail_min_submit &&
                                elapsed >= 0.5 && detail_buffer_lines++ < 250) {
                                fprintf(stderr,
                                        "[render-timing] buffer draw=%llu set=%u binding=%u "
                                        "addr=0x%llx declared=%u uploaded=%u class=%u direct=%d "
                                        "probe=%.2f copy=%.2f total=%.2f ms\n",
                                        (unsigned long long)draw.draw_index, set, r.binding,
                                        (unsigned long long)r.gpu_addr, requested_bytes, nb,
                                        static_cast<unsigned>(r.cls),
                                        static_cast<int>(resource_buffer_view),
                                        resource_buffer_probe_ms, resource_buffer_copy_ms, elapsed);
                            }
                        }
                        // PROSPER_CBLOG: log each constant buffer's first 4 dwords as floats, once per
                        // address. If a scene draw's color/tint CB is (0,0,0,0), the PS outputs black
                        // regardless of the (correctly-decoded) texture — the #300 black-scene suspect.
                        if (PROSPER_ENV_ON("PROSPER_CBLOG") && r.cls == RC::ConstantBuffer) {
                            static std::set<uint64_t> cbseen;
                            if (cbseen.insert(r.gpu_addr).second) {
                                const uint32_t* words = fr.buffer_words_data();
                                size_t n = fr.buffer_word_count();
                                const float* fp = reinterpret_cast<const float*>(words);
                                fprintf(stderr, "[cb] bind=%u addr=0x%llx size=%u dw=%08x %08x %08x %08x  f=%.3f %.3f %.3f %.3f\n",
                                        r.binding, (unsigned long long)r.gpu_addr, (unsigned)r.size,
                                        n>0?words[0]:0, n>1?words[1]:0, n>2?words[2]:0, n>3?words[3]:0,
                                        n>0?fp[0]:0.f, n>1?fp[1]:0.f, n>2?fp[2]:0.f, n>3?fp[3]:0.f);
                            }
                        }
                    }
                    if (timing_enabled) {
                        const double elapsed = std::chrono::duration<double, std::milli>(
                            RenderClock::now() - resource_timing_start).count();
                        if (fr.is_texture()) {
                            // Classified in the SAME order the path decides them, and exclusively:
                            // a reference satisfied by an earlier class never reaches a later one,
                            // so every reference lands in exactly one bucket or in `other`. The
                            // residual is therefore two independent numbers -- a COUNT of references
                            // nothing above claimed, and a signed remainder of ms -- and they
                            // disagree only if the classification is wrong.
                            if (resource_rtt_hit)                      { pending_timing.tex_rtt_ms += elapsed;           ++pending_timing.tex_rtt_n; }
                            else if (resource_compute_image_hit ||
                                     resource_compute_depth_hybrid)    { pending_timing.tex_compute_ms += elapsed;       ++pending_timing.tex_compute_n; }
                            else if (resource_persistent_submit_reuse) { pending_timing.tex_persist_reuse_ms += elapsed; ++pending_timing.tex_persist_reuse_n; }
                            else if (resource_persistent_hit)          { pending_timing.tex_persist_hit_ms += elapsed;   ++pending_timing.tex_persist_hit_n; }
                            else if (resource_persistent_miss)         { pending_timing.tex_persist_miss_ms += elapsed;  ++pending_timing.tex_persist_miss_n; }
                            else if (resource_persistent_invalidation) { pending_timing.tex_persist_invalid_ms += elapsed; ++pending_timing.tex_persist_invalid_n; }
                            else if (resource_local_reuse)             { pending_timing.tex_local_ms += elapsed;         ++pending_timing.tex_local_n; }
                            else {
                                ++pending_timing.tex_other_n;
                                if (elapsed > pending_timing.tex_other_slowest_ms) {
                                    pending_timing.tex_other_slowest_ms = elapsed;
                                    pending_timing.tex_other_addr = r.gpu_addr;
                                    pending_timing.tex_other_source_bytes =
                                        resource_persistent_source_size;
                                    pending_timing.tex_other_width = r.width;
                                    pending_timing.tex_other_height = r.height;
                                    pending_timing.tex_other_depth = r.depth;
                                    pending_timing.tex_other_format =
                                        static_cast<uint32_t>(r.format);
                                    pending_timing.tex_other_components = r.num_components;
                                    pending_timing.tex_other_tile_mode = r.tile_mode;
                                    pending_timing.tex_other_img_dim = r.img_dim;
                                    pending_timing.tex_other_class =
                                        static_cast<uint32_t>(r.cls);
                                    pending_timing.tex_other_compute_candidate =
                                        resource_compute_image_candidate;
                                    pending_timing.tex_other_persistent_candidate =
                                        resource_persistent_candidate;
                                    pending_timing.tex_other_compressed =
                                        r.compression_enabled;
                                    pending_timing.tex_other_depth_compare = r.depth_compare;
                                    pending_timing.tex_other_host_backed = r.host_data != nullptr;
                                }
                            }
                            pending_timing.textures++;
                            pending_timing.texture_bytes += static_cast<uint64_t>(fr.tw) * fr.th * fr.td *
                                fr.sample_count *
                                prosper::test::backend_color_bytes_per_pixel(fr.texture_format);
                            pending_timing.texture_ms += elapsed;
                            const uint64_t detail_min_submit = PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT")
                                ? strtoull(PROSPER_ENV_VALUE("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT"), nullptr, 0) : 0;
                            // render_timing_detail is the SAME hoist #2214 introduced, already used
                            // at the sibling site a few lines up; this one was missed. It is inside
                            // `if (timing_enabled)`, so it costs a normal run nothing -- but it made
                            // an F8 capture pay a getenv per resource, inflating the enclosing totals
                            // the capture then reports. An instrument that charges for being read.
                            if (render_timing_detail &&
                                static_cast<uint64_t>(g_this_submit) >= detail_min_submit) {
                                static uint64_t detail_lines = 0;
                                if ((elapsed >= 0.5 || resource_persistent_invalidation ||
                                     resource_compute_image_hit ||
                                     resource_compute_depth_hybrid) &&
                                    detail_lines++ < 250) {
                                    const char* cache_state = resource_rtt_hit ? "rtt" :
                                        (resource_compute_depth_hybrid ? "compute-depth-hybrid" :
                                        (resource_compute_image_hit ? "compute-image" :
                                        (resource_local_reuse ? "local" :
                                        (resource_persistent_submit_reuse ? "persistent-submit" :
                                        (resource_persistent_hit ? "persistent-hit" :
                                        (resource_persistent_invalidation ? "persistent-invalid" :
                                        (resource_persistent_miss ? "persistent-miss" : "uncached")))))));
                                    auto submit_query_name = [](int query) {
                                        switch (query) {
                                            case static_cast<int>(
                                                prosper::gpu::GuestGpuWriteQuery::Unchanged):
                                                return "unchanged";
                                            case static_cast<int>(
                                                prosper::gpu::GuestGpuWriteQuery::Overlap):
                                                return "overlap";
                                            case static_cast<int>(
                                                prosper::gpu::GuestGpuWriteQuery::Unknown):
                                                return "unknown";
                                            default: return "none";
                                        }
                                    };
                                    auto watch_query_name = [](int query) {
                                        switch (query) {
                                            case static_cast<int>(
                                                prosper::host::GuestWriteWatchQuery::Unchanged):
                                                return "unchanged";
                                            case static_cast<int>(
                                                prosper::host::GuestWriteWatchQuery::Dirty):
                                                return "dirty";
                                            case static_cast<int>(
                                                prosper::host::GuestWriteWatchQuery::Unknown):
                                                return "unknown";
                                            default: return "none";
                                        }
                                    };
                                    fprintf(stderr,
                                            "[render-timing] texture addr=0x%llx %ux%ux%u out=%ux%ux%u "
                                            "dim=%u fmt=%u comps=%u tile=%u class=%u storage=%d "
                                            "host=%d live=%d depth-live=%d candidate=%d source=%zu "
                                            "compressed=%d depth-compare=%d compute-candidate=%d "
                                            "cache=%s id=%llu validate=%s %.2fms/%zuB/%zuB "
                                            "submit=%s watch=%s active=%d disabled=%d only=%d stable=%u "
                                            "total=%.2f ms\n",
                                            (unsigned long long)r.gpu_addr, r.width, r.height,
                                            r.depth, fr.tw, fr.th, fr.td, r.img_dim,
                                            (unsigned)r.format, r.num_components,
                                            r.tile_mode, static_cast<unsigned>(r.cls),
                                            static_cast<int>(fr.is_storage_image),
                                            static_cast<int>(r.host_data != nullptr),
                                            static_cast<int>(resource_has_live_rtt),
                                            static_cast<int>(resource_has_ds_live),
                                            static_cast<int>(resource_persistent_candidate),
                                            resource_persistent_source_size,
                                            static_cast<int>(r.compression_enabled),
                                            static_cast<int>(r.depth_compare),
                                            static_cast<int>(resource_compute_image_candidate),
                                            cache_state,
                                            (unsigned long long)fr.persistent_texture_id,
                                            resource_texture_exact_validation ? "exact" : "skip",
                                            resource_texture_validation_ms,
                                            resource_texture_validated_bytes,
                                            resource_texture_source_bytes,
                                            submit_query_name(resource_texture_submit_query),
                                            watch_query_name(resource_texture_watch_query),
                                            static_cast<int>(resource_texture_watch_active),
                                            static_cast<int>(resource_texture_watch_disabled),
                                            static_cast<int>(resource_texture_watch_only),
                                            resource_texture_watch_stability, elapsed);
                                }
                            }
                        } else {
                            const size_t buffer_bytes = fr.buffer_word_count() * sizeof(uint32_t);
                            pending_timing.buffers++;
                            pending_timing.buffer_views += resource_buffer_view;
                            pending_timing.buffer_bytes += buffer_bytes;
                            if (!resource_buffer_view)
                                pending_timing.buffer_materialized_bytes += buffer_bytes;
                            pending_timing.buffer_ms += elapsed;
                        }
                    }
                    R.push_back(std::move(fr));
                }
              };
              add(vrt, 0, draw.vs_words(), prosper::gpu::SpirvShaderStage::Vertex,
                  draw.vs_identity);
              add(prt, 1, draw.fs_words(), prosper::gpu::SpirvShaderStage::Fragment,
                  draw.fs_identity);
              // VS resources -> descriptor set 0, PS -> set 1
              return R;
            };
            // Poison mode keeps the draw running while making invalid bindings visually/numerically
            // unmistakable: magenta/cyan texels for images and NaN-like dwords for buffers. Missing,
            // duplicate, wrong-type, and undersized bindings are replaced from the reflected manifest.
            // Read once per submit, not once per call. poison_R is invoked TWICE per draw (VS and
            // PS), so at 2,179 draws in a submit its getenv was 4,358 calls -- and on Windows
            // getenv takes a process-wide lock and walks the environment block, which is why #2214
            // measured +43% from removing exactly this shape from the per-resource path.
            //
            // Hoisted rather than wrapped in PROSPER_ENV_VALUE, and the difference is not stylistic:
            // test_gpu_capture_render.cpp and test_shader_resources.cpp ARM this variable at runtime
            // between phases, and a process-lifetime cache would not observe the second write. Those
            // tests would not fail -- they would go VACUOUS and keep printing [ok] against a stale
            // value, which is the #2214 defect that `cached_env_arming_logic` now gates. A per-submit
            // hoist keeps every runtime write observable from the next submit on.
            const char* const descriptor_validate_mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");
            auto poison_R = [descriptor_validate_mode](
                               std::vector<prosper::test::FrameResource>& resources,
                               const std::vector<uint32_t>& spirv,
                               const prosper::gpu::ShaderResourceTable* table,
                               uint32_t set, prosper::gpu::SpirvShaderStage stage) {
                const char* mode = descriptor_validate_mode;
                if (!mode || strcmp(mode, "poison")) return;
                auto report = prosper::gpu::validate_spirv_descriptor_interface(spirv, table, set, stage, false);
                static const uint8_t poison_tex[16] = {
                    255, 0, 255, 255,   0, 255, 255, 255,
                    0, 255, 255, 255,   255, 0, 255, 255,
                };
                static const uint32_t poison_storage_uint[16] = {
                    0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
                    0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
                    0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
                    0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
                };
                for (const auto& d : report.descriptors) {
                    bool invalid = false;
                    for (const auto& issue : report.issues)
                        if (issue.error && issue.binding == d.binding) { invalid = true; break; }
                    if (!invalid) continue;
                    auto first = std::find_if(resources.begin(), resources.end(), [&](const auto& r) {
                        return r.set == set && r.binding == d.binding;
                    });
                    size_t count = 0;
                    for (const auto& r : resources) if (r.set == set && r.binding == d.binding) ++count;
                    const bool wants_buffer = d.kind == prosper::gpu::SpirvDescriptorKind::StorageBuffer;
                    const bool wants_storage_image =
                        d.kind == prosper::gpu::SpirvDescriptorKind::StorageImage;
                    const uint64_t available = first == resources.end() ? 0 :
                        (first->is_texture() ? (uint64_t)first->tw * first->th * first->td * 4
                                             : first->buffer_word_count() * 4);
                    const bool wrong_type = first != resources.end() &&
                        (wants_buffer ? first->is_texture()
                                      : (!first->is_texture() ||
                                         first->is_storage_image != wants_storage_image));
                    const bool undersized = wants_buffer && available < std::max<uint64_t>(d.required_bytes, 4);
                    resources.erase(std::remove_if(resources.begin(), resources.end(), [&](const auto& r) {
                        return r.set == set && r.binding == d.binding;
                    }), resources.end());
                    prosper::test::FrameResource replacement;
                    replacement.set = set; replacement.binding = d.binding;
                    if (wants_buffer) {
                        size_t words = static_cast<size_t>((std::max<uint64_t>(d.required_bytes, 16) + 3) / 4);
                        // Poison replacements stay deliberately small: this is a diagnostic
                        // substitute for an invalid binding, not a real upload, so it does not
                        // follow build_R's ceiling (which is now the declared range, #1427).
                        words = std::min<size_t>(words, 1u << 18);
                        replacement.dwords.assign(words, 0x7FC0CDCDu);
                    } else {
                        replacement.storage_image_numeric_class = d.image_numeric_class;
                        replacement.storage_image_contract_valid =
                            d.image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Float ||
                            d.image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Uint;
                        if (wants_storage_image &&
                            d.image_numeric_class ==
                                prosper::gpu::SpirvImageNumericClass::Uint) {
                            replacement.tex_rgba = reinterpret_cast<const uint8_t*>(
                                poison_storage_uint);
                            replacement.texture_format = VK_FORMAT_R32G32B32A32_UINT;
                        } else {
                            replacement.tex_rgba = poison_tex;
                        }
                        replacement.tw = 2; replacement.th = 2;
                        replacement.is_storage_image = wants_storage_image;
                        replacement.mag_filter = replacement.min_filter = 0;
                    }
                    fprintf(stderr, "[descriptor] poison set=%u binding=%u type=%s reason=%s%s%s%s\n",
                            set, d.binding, prosper::gpu::spirv_descriptor_kind_name(d.kind),
                            count == 0 ? "missing" : "",
                            count > 1 ? "duplicate" : "",
                            wrong_type ? "wrong-type" : "",
                            undersized ? "undersized" : "");
                    resources.push_back(std::move(replacement));
                }
            };
            // Diagnostic shader/state overrides (computed once, applied to EVERY draw item):
            //   REFVS  -> a known-good fullscreen-triangle VS (isolates the game's real VS).
            //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading). The optional
            //             TESTPS_MATCH file restricts it to one exact recompiled guest PS.
            //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
            //   NOPS   -> bypass the resolved pipeline state (default state).
            #include "refvs.inc"
            const bool refvs = PROSPER_ENV_VALUE("PROSPER_RENDER_REFVS");
            std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
            std::vector<uint32_t> ps_override;
            bool ps_override_is_file = false;   // true only for a valid PROSPER_FS_SPV *file* override
            bool ps_override_is_test = false;
            if (PROSPER_ENV_ON("PROSPER_RENDER_TESTPS")) {
                static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
                    0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
                ps_override = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
                ps_override_is_test = true;
            }
            // Validated SPIR-V file load: require a complete read of a word-aligned file >= 20 bytes
            // (5 words: the minimum SPIR-V header). Validate size BEFORE allocating so a failed ftell
            // (-1 -> huge size_t) cannot trigger a wild allocation, and reject non-word-aligned files
            // rather than silently dropping trailing bytes. Returns false (out untouched) on any failure.
            auto load_spv_file = [](const char* path, std::vector<uint32_t>& out) -> bool {
                FILE* f = fopen(path, "rb");
                if (!f) return false;
                bool ok = false;
                if (fseek(f, 0, SEEK_END) == 0) {
                    long sz = ftell(f);
                    if (sz >= 20 && (sz % 4) == 0 && fseek(f, 0, SEEK_SET) == 0) {
                        std::vector<uint32_t> m(static_cast<size_t>(sz) / 4);
                        if (fread(m.data(), 4, m.size(), f) == m.size()) { out = std::move(m); ok = true; }
                    }
                }
                fclose(f);
                return ok;
            };
            if (const char* fsp = PROSPER_ENV_VALUE("PROSPER_FS_SPV")) {
                std::vector<uint32_t> m;
                if (load_spv_file(fsp, m)) { ps_override = std::move(m); ps_override_is_file = true; }
                else fprintf(stderr, "[fs-spv] PROSPER_FS_SPV='%s' invalid/unreadable -> no file override\n", fsp);
            }
            // PROSPER_FS_SPV_MATCH=<file>: restrict the PROSPER_FS_SPV *file* override to draws whose
            // recompiled fragment SPIR-V EXACTLY equals this file (a per-draw A/B substitution that does
            // not touch draws with a different descriptor contract). FAILS CLOSED: if requested but the
            // file is missing/short/unaligned/unreadable, NO file override is applied — never a silent
            // global fallback (that would recreate the exact hazard this gate exists to prevent). Does
            // NOT gate PROSPER_RENDER_TESTPS, which stays global by design.
            // mode: 0 = not requested (legacy global file override), 1 = loaded+valid (exact match only),
            //       2 = requested but invalid (file override disabled).
            int fs_match_mode = 0;
            std::vector<uint32_t> fs_match;
            if (const char* mp = PROSPER_ENV_VALUE("PROSPER_FS_SPV_MATCH")) {
                fs_match_mode = load_spv_file(mp, fs_match) ? 1 : 2;
                if (fs_match_mode == 2)
                    fprintf(stderr, "[fs-match] PROSPER_FS_SPV_MATCH='%s' invalid/unreadable -> applying NO "
                            "fragment file override (fail closed)\n", mp);
            }
            // PROSPER_RENDER_TESTPS_MATCH=<file> is the geometry half of a per-shader A/B test: replace
            // only that exact guest PS with the known solid output while retaining its real VS, indices,
            // viewport, depth and raster state. As with FS_SPV_MATCH, a bad path fails closed.
            int testps_match_mode = 0;
            std::vector<uint32_t> testps_match;
            if (const char* mp = PROSPER_ENV_VALUE("PROSPER_RENDER_TESTPS_MATCH")) {
                testps_match_mode = load_spv_file(mp, testps_match) ? 1 : 2;
                if (testps_match_mode == 2)
                    fprintf(stderr, "[testps-match] PROSPER_RENDER_TESTPS_MATCH='%s' invalid/unreadable -> "
                            "applying NO test fragment override (fail closed)\n", mp);
            }
            const bool nops = PROSPER_ENV_VALUE("PROSPER_RENDER_NOPS");
            // Assemble backend draws for a subset of the submit's items — one BackendDraw per realized
            // DrawItem with its own resources + fixed-function state (or the diagnostic overrides above).
            // build_R reads the CURRENT g_rtt, so calling this AFTER an earlier target-group has been
            // rendered+stored lets the later group sample that group's pixels (a HIT, not empty memory).
            // PROSPER_SKIP_DRAW="N[,N...]" (diagnostic): drop these semantic draw_index values from
            // every pass — isolate whether a specific draw (e.g. a suspected opaque UI backdrop that
            // hides the composited world) is what corrupts the frame, without touching any state.
            static const char* skip_draws_env = getenv("PROSPER_SKIP_DRAW");
            auto draw_is_skipped = [](uint64_t idx) -> bool {
                if (!skip_draws_env) return false;
                for (const char* s = skip_draws_env; *s;) {
                    char* end = nullptr; unsigned long long v = strtoull(s, &end, 0);
                    if (end != s && v == idx) return true;
                    s = (end && *end) ? end + 1 : end;
                    if (!s || !*s) break;
                }
                return false;
            };
            // PROSPER_SKIP_DRAW_PROGRAM / PROSPER_DRAW_PROGRAM_CENSUS: decline draws by shader
            // PROGRAM identity, and enumerate the programs a title draws with. Both are process
            // singletons configured once from the environment (see draw_program_skip.hpp for the
            // contract and the four limits a reader of a skipped run cannot see in the output).
            // Hoisted here for the same reason descriptor_validate_mode is: the accessor is one
            // function-local-static test, but calling it per draw on a 2,100-draw submit is a
            // measurable cost for a variable nobody set.
            auto& program_skip = prosper::gpu::draw_program_skip_selector();
            const bool program_skip_armed = program_skip.armed();
            const bool program_census = prosper::gpu::draw_program_census_enabled();
            // PROSPER_DRAW_LINKSCAN: the graphics counterpart of PROSPER_COMPUTE_PARENTSCAN.
            // Hoisted for the same reason as the two above -- disarmed it is one bool.
            auto& link_scan = prosper::gpu::draw_link_scan_selector();
            const bool link_scan_armed = prosper::gpu::draw_link_scan_enabled();
            auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group) {
                std::vector<prosper::test::BackendDraw> bds;
                // Once per call, not once per draw -- see the note on descriptor_validate_mode above
                // for why this is a hoist and not a PROSPER_ENV_* cache (test_eop_write.cpp arms
                // PROSPER_GFXLOG at runtime, and a cached read would make its arms vacuous).
                const bool gfxlog = getenv("PROSPER_GFXLOG") != nullptr;
                for (const auto* itp : group) {
                    const auto& it = *itp;
                    if (draw_is_skipped(it.draw_index)) continue;
                    prosper::test::BackendDraw bd;
                    if (refvs) {
                        bd.vs = refvs_spv;
                    } else if (it.vs_shared) {
                        bd.vs_shared = it.vs_shared;
                    } else {
                        bd.vs = it.vs;
                    }
                    bd.gs = refvs ? std::vector<uint32_t>{} : it.gs;
                    // File and synthetic overrides have independent exact-match gates.
                    bool fs_ov = !ps_override.empty();
                    if (fs_ov && ps_override_is_file) {
                        if (fs_match_mode == 1)      fs_ov = (it.fs_words() == fs_match);   // valid match -> exact only
                        else if (fs_match_mode == 2) fs_ov = false;                // requested-but-invalid -> off
                        // mode 0 -> legacy global file override (unchanged)
                    }
                    if (fs_ov && ps_override_is_test) {
                        if (testps_match_mode == 1)      fs_ov = (it.fs_words() == testps_match);
                        else if (testps_match_mode == 2) fs_ov = false;
                        // mode 0 -> legacy global TESTPS override (unchanged)
                    }
                    if (fs_ov && ps_override_is_file && fs_match_mode == 1)
                        fprintf(stderr, "[fs-match] file override applied to draw#%llu\n",
                                (unsigned long long)it.draw_index);
                    if (fs_ov && ps_override_is_test && testps_match_mode == 1)
                        fprintf(stderr, "[testps-match] synthetic override applied to draw#%llu\n",
                                (unsigned long long)it.draw_index);
                    if (fs_ov) {
                        bd.fs = ps_override;
                    } else if (it.fs_shared) {
                        bd.fs_shared = it.fs_shared;
                    } else {
                        bd.fs = it.fs;
                    }
                    bd.vs_identity = refvs ? 0 : it.vs_identity;
                    bd.fs_identity = fs_ov ? 0 : it.fs_identity;
                    bd.allow_native_fragment_vote_width =
                        !fs_ov && gta5_native_fragment_vote;
                    bd.draw_index = it.draw_index;
                    bd.command_order = it.command_order;
                    bd.vcount = refvs ? 3u : it.vertex_count;
                    bd.instance_count = it.instance_count;
                    bd.vertex_offset = refvs ? 0 : it.vertex_offset;
                    bd.ps     = nops ? nullptr : &it.ps;
                    // Five clock reads bounding four spans, only when timing is armed -- ~0.9% of
                    // this bucket at 2,100 draws a submit. It inflates what it measures while
                    // armed, as every timer here does; read the shares, not the totals.
                    const auto bt0 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
                    bd.R      = build_R(it, it.vrt.get(), it.prt.get());
                    const auto bt1 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
                    // && of the positives rather than || of the negations: identical short-circuit
                    // and identical outcome, but it puts the accounting below on a path the
                    // `continue` cannot skip.
                    // descriptor_validate_mode is the per-submit hoist made for poison_R above, and
                    // it serves this pair for the same reason: without it each call re-reads getenv,
                    // and with the variable unset that read IS the whole call. It measured
                    // 5.77 ms/submit in the build_resources partition -- larger than the validation
                    // it was declining to do.
                    const bool contract_ok =
                        prosper::gpu::validate_runtime_descriptor_contract(
                            "VS/backend", bd.vs_words(), it.vrt.get(), 0,
                            prosper::gpu::SpirvShaderStage::Vertex, descriptor_validate_mode) &&
                        prosper::gpu::validate_runtime_descriptor_contract(
                            "PS/backend", bd.fs_words(), it.prt.get(), 1,
                            prosper::gpu::SpirvShaderStage::Fragment, descriptor_validate_mode);
                    const auto bt2 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
                    if (timing_enabled) {
                        pending_timing.build_r_ms +=
                            std::chrono::duration<double, std::milli>(bt1 - bt0).count();
                        pending_timing.build_validate_ms +=
                            std::chrono::duration<double, std::milli>(bt2 - bt1).count();
                    }
                    // A rejected draw has already spent build_R and both validations, and they are
                    // counted above. Dropping it without accounting would move that time into the
                    // residual, where it reads as unattributed work -- the exact defect this
                    // partition exists to make visible.
                    if (!contract_ok) { if (timing_enabled) ++pending_timing.build_rejected; continue; }
                    poison_R(bd.R, bd.vs_words(), it.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex);
                    poison_R(bd.R, bd.fs_words(), it.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment);
                    const auto bt3 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
                    // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
                    // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
                    if (!refvs) bd.indices = it.indices;
                    if (timing_enabled) {
                        const auto bt4 = RenderClock::now();
                        pending_timing.build_poison_ms +=
                            std::chrono::duration<double, std::milli>(bt3 - bt2).count();
                        pending_timing.build_indices_ms +=
                            std::chrono::duration<double, std::milli>(bt4 - bt3).count();
                        // Counted AFTER the reject, so `build_draws` is accepted draws — and the
                        // leaves do not all share it. `build_R` and `validate` ran for
                        // draws + rejected; `poison` and `indices` ran for draws alone. Both
                        // numbers are printed so the reader can pick the right denominator, and
                        // the report line says which is which — a per-draw figure divided by the
                        // wrong one over-states silently, and with rejected=0 nobody would notice
                        // the rule until the first run where it is not.
                        ++pending_timing.build_draws;
                    }
                    if (gfxlog) fprintf(stderr,
                        "[render] item %zu: %zu resources vcount=%u instances=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
                        bds.size(), bd.R.size(), bd.vcount, bd.instance_count, bd.indices.size(), it.ps.topology,
                        it.ps.color_write_mask, (int)it.ps.blend_enable);
                    // RTTLOG per-draw detail (render-window-only, unlike the GFXLOG firehose): enough
                    // state to diagnose a pass whose inputs HIT the RTT cache yet outputs nothing —
                    // blend factors, write mask, viewport, and each PS-sampled texture address (#319).
                    if (rtt_log) {
                        fprintf(stderr, "[rtt]   draw#%llu vs=0x%llx fs=0x%llx tgt=0x%llx "
                                "vcount=%u nidx=%zu topo=%u mask=0x%x "
                                "blend=%d(src=%u dst=%u) vp=%d(%.2f,%.2f %.2fx%.2f) z=%d zw=%d ps=",
                                (unsigned long long)it.draw_index,
                                (unsigned long long)it.vs_guest_addr,
                                (unsigned long long)it.fs_guest_addr,
                                (unsigned long long)it.color0_base, bd.vcount, bd.indices.size(),
                                it.ps.topology, it.ps.color_write_mask, (int)it.ps.blend_enable,
                                it.ps.src_color_blend_factor, it.ps.dst_color_blend_factor,
                                (int)it.ps.has_viewport,
                                it.ps.viewport_x, it.ps.viewport_y, it.ps.viewport_w, it.ps.viewport_h,
                                (int)it.ps.depth_test_enable, (int)it.ps.depth_write_enable);
                        if (it.prt) for (const auto& r : it.prt->resources)
                            if (r.cls == RC::Texture)
                                fprintf(stderr, " tex@%u=0x%llx(%ux%u f%u)", r.binding,
                                        (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
                        fprintf(stderr, "\n");
                    }
                    // PROSPER_DRAW_PROGRAM_CENSUS: which programs does this title actually draw
                    // with? One line per distinct (vs, vs-chain, ps) triple, then powers of two --
                    // bounded by the program count, not the draw count. This is the census that
                    // supplies PROSPER_SKIP_DRAW_PROGRAM's input.
                    if (program_census) {
                        const auto sighting = prosper::gpu::draw_program_census().observe(
                            it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
                        if (sighting.print)
                            fprintf(stderr, "[draw-program] %s vs=0x%llx chain=0x%llx ps=0x%llx "
                                    "draws=%llu distinct=%llu vcount=%u nidx=%zu topo=%u tgt=0x%llx\n",
                                    sighting.first ? "NEW " : "    ",
                                    (unsigned long long)it.vs_guest_addr,
                                    (unsigned long long)it.vs_chain_guest_addr,
                                    (unsigned long long)it.fs_guest_addr,
                                    (unsigned long long)sighting.ordinal,
                                    (unsigned long long)sighting.distinct,
                                    bd.vcount, bd.indices.size(), it.ps.topology,
                                    (unsigned long long)it.color0_base);
                    }
                    // PROSPER_DRAW_LINKSCAN: census the LINKED LISTS this draw's scalar buffers
                    // contain, taken from the exact bytes prosper is about to upload -- not from
                    // guest memory, because the question is what the SHADER sees. Ordered with the
                    // other observers and before the skip, for the trap-166 reason below.
                    if (link_scan_armed &&
                        link_scan.matches(it.vs_guest_addr, it.vs_chain_guest_addr,
                                          it.fs_guest_addr)) {
                        const auto& lcfg = prosper::gpu::draw_link_scan_settings();
                        const uint32_t* heads_words = nullptr; size_t heads_count = 0;
                        const uint32_t* rec_words = nullptr;   size_t rec_count = 0;
                        size_t buffers_seen = 0;
                        for (const auto& fr : bd.R) {
                            if (fr.is_texture() || fr.is_storage_image) continue;
                            const uint32_t* words = fr.buffer_words_data();
                            const size_t nwords = fr.buffer_word_count();
                            if (!words || nwords == 0) continue;
                            ++buffers_seen;
                            // The DECLARED descriptor beside the UPLOADED bytes. A short upload is
                            // the mis-resolved-descriptor case and is invisible in the census alone:
                            // the shader reads zeros past the end and a zero link is not an exit.
                            uint64_t res_addr = 0; uint32_t res_declared = 0; int res_cls = -1;
                            const prosper::gpu::ShaderResourceTable* table =
                                fr.set == 0 ? it.vrt.get() : it.prt.get();
                            if (table)
                                for (const auto& r : table->resources)
                                    if (r.binding == fr.binding) {
                                        res_addr = r.gpu_addr; res_declared = r.size;
                                        res_cls = static_cast<int>(r.cls);
                                        break;
                                    }
                            if (fr.set == 1 && fr.binding == lcfg.heads_binding) {
                                heads_words = words; heads_count = nwords;
                            }
                            if (fr.set == 1 && fr.binding == lcfg.records_binding) {
                                rec_words = words; rec_count = nwords;
                            }
                            bool exhausted = false;
                            const uint32_t scan_ordinal = link_scan.should_scan(
                                it.fs_guest_addr, fr.set, fr.binding,
                                lcfg.max_scans_per_buffer, &exhausted);
                            if (!scan_ordinal) {
                                if (exhausted)
                                    fprintf(stderr,
                                            "[linkscan] capped ps=0x%llx set=%u binding=%u "
                                            "addr=0x%llx after %u scan(s) -- later draws of this "
                                            "buffer are NOT scanned\n",
                                            (unsigned long long)it.fs_guest_addr, fr.set,
                                            fr.binding, (unsigned long long)res_addr,
                                            lcfg.max_scans_per_buffer);
                                continue;
                            }
                            const std::span<const uint32_t> span(words, nwords);
                            const auto hist = prosper::gpu::histogram_words(
                                span, lcfg.encoding.terminator);
                            const auto self = prosper::gpu::census_self_walk(span, lcfg.encoding);
                            fprintf(stderr,
                                    "[linkscan] ps=0x%llx draw#%llu scan=%u set=%u binding=%u "
                                    "cls=%d addr=0x%llx declared=%u uploaded_dw=%zu | "
                                    "hist zero=%u term=%u other=%u first_other=[%u]=0x%08x "
                                    "other_range=0x%08x..0x%08x | self-walk stride=%u next=+%u "
                                    "term=0x%08x records=%u starts=%u terminating=%u cyclic=%u "
                                    "cycle-nodes=%u oob-starts=%u longest=%u\n",
                                    (unsigned long long)it.fs_guest_addr,
                                    (unsigned long long)it.draw_index, scan_ordinal, fr.set,
                                    fr.binding, res_cls, (unsigned long long)res_addr, res_declared,
                                    nwords, hist.zero, hist.terminator, hist.other,
                                    hist.first_other_index, hist.first_other_value, hist.min_other,
                                    hist.max_other, lcfg.encoding.record_stride_dwords,
                                    lcfg.encoding.next_dword_offset, lcfg.encoding.terminator,
                                    self.records, self.starts, self.terminating, self.cyclic,
                                    self.cycle_nodes, self.oob_starts, self.longest);
                            for (uint32_t k = 0; k < self.sample_count; ++k)
                                fprintf(stderr,
                                        "[linkscan]   cycle-member[%u] link=%u -> next=%u\n", k,
                                        self.sample_link[k], self.sample_next[k]);
                            // WHO last wrote these bytes. An all-zero buffer has two completely
                            // different causes -- a producer that never ran, and a producer that
                            // ran and wrote zeros -- and the census alone cannot tell them apart.
                            // The recorder summary is printed with the answer, unconditionally,
                            // because a bounded and per-kind-gated history makes "no writer"
                            // a statement about the INSTRUMENT unless the reader can see which
                            // recorders fired (writer_provenance.hpp's scope list, #2111).
                            if (res_addr && res_declared) {
                                if (!prosper::gpu::writer_provenance_enabled()) {
                                    static bool provenance_note = false;
                                    if (!provenance_note) {
                                        provenance_note = true;
                                        fprintf(stderr,
                                                "[linkscan]   writer provenance is OFF -- set "
                                                "PROSPER_WRITER_PROVENANCE=1 to learn whether an "
                                                "all-zero buffer was never written or was written "
                                                "with zeros\n");
                                    }
                                } else {
                                    const auto writer = prosper::gpu::last_guest_write_overlap(
                                        res_addr, res_declared);
                                    if (writer)
                                        fprintf(stderr,
                                                "[linkscan]   last writer binding=%u kind=%s "
                                                "addr=0x%llx size=%llu submit=%llu item=%llu "
                                                "order=%llu identity=0x%llx | recorders %s\n",
                                                fr.binding,
                                                prosper::gpu::guest_writer_kind_name(writer->kind),
                                                (unsigned long long)writer->addr,
                                                (unsigned long long)writer->size,
                                                (unsigned long long)writer->submit,
                                                (unsigned long long)writer->item,
                                                (unsigned long long)writer->order,
                                                (unsigned long long)writer->identity,
                                                prosper::gpu::guest_write_recorder_summary());
                                    else
                                        fprintf(stderr,
                                                "[linkscan]   last writer binding=%u NONE -- no "
                                                "recorded write overlaps 0x%llx+%u | history=%zu "
                                                "recorders %s\n",
                                                fr.binding, (unsigned long long)res_addr,
                                                res_declared,
                                                prosper::gpu::guest_write_history_size(),
                                                prosper::gpu::guest_write_recorder_summary());
                                }
                            }
                        }
                        if (heads_words && rec_words) {
                            const auto heads = prosper::gpu::census_head_walk(
                                std::span<const uint32_t>(rec_words, rec_count),
                                std::span<const uint32_t>(heads_words, heads_count),
                                lcfg.encoding);
                            fprintf(stderr,
                                    "[linkscan] ps=0x%llx draw#%llu HEAD-WALK heads_binding=%u "
                                    "(%zu dw) records_binding=%u (%zu dw) starts=%u "
                                    "terminating=%u cyclic=%u cycle-nodes=%u oob-starts=%u "
                                    "longest=%u\n",
                                    (unsigned long long)it.fs_guest_addr,
                                    (unsigned long long)it.draw_index, lcfg.heads_binding,
                                    heads_count, lcfg.records_binding, rec_count, heads.starts,
                                    heads.terminating, heads.cyclic, heads.cycle_nodes,
                                    heads.oob_starts, heads.longest);
                        }
                        // Silence would otherwise be indistinguishable between "the program never
                        // drew" and "it drew with no buffers at all". Say which, once.
                        if (buffers_seen == 0) {
                            static std::set<uint64_t> no_buffer_reported;
                            if (no_buffer_reported.insert(it.fs_guest_addr).second)
                                fprintf(stderr,
                                        "[linkscan] ps=0x%llx draw#%llu matched but has NO storage "
                                        "buffer resources -- nothing to census\n",
                                        (unsigned long long)it.fs_guest_addr,
                                        (unsigned long long)it.draw_index);
                        }
                    }
                    // PROSPER_SKIP_DRAW_PROGRAM: withhold this draw from the GPU because one of its
                    // programs is named. LAST in the per-draw build on purpose -- instrument trap
                    // 166: a diagnostic skip ordered before the dump blinds every other instrument
                    // to the thing under suspicion, and the program worth declining is precisely the
                    // one that hangs the GPU. So the draw is fully realized, fully validated and
                    // fully logged by [render]/[rtt]/[draw-program] first; only the Vulkan draw call
                    // is withheld. Its CPU-side cost is therefore real, and the timing partition
                    // above has already counted it in build_draws, which is arithmetically right --
                    // it did spend build_R, validation, poison and indices.
                    if (program_skip_armed) {
                        const auto decision = program_skip.evaluate(
                            it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
                        if (decision.skip) {
                            if (decision.print)
                                fprintf(stderr,
                                        "[draw-decline] program=0x%llx stage=%s "
                                        "reason=skipped-by-selector count=%llu draw#%llu order=%llu "
                                        "vs=0x%llx chain=0x%llx ps=0x%llx vcount=%u nidx=%zu "
                                        "tgt=0x%llx\n",
                                        (unsigned long long)decision.address,
                                        prosper::gpu::draw_program_stage_name(decision.stage),
                                        (unsigned long long)decision.ordinal,
                                        (unsigned long long)it.draw_index,
                                        (unsigned long long)it.command_order,
                                        (unsigned long long)it.vs_guest_addr,
                                        (unsigned long long)it.vs_chain_guest_addr,
                                        (unsigned long long)it.fs_guest_addr,
                                        bd.vcount, bd.indices.size(),
                                        (unsigned long long)it.color0_base);
                            continue;
                        }
                    }
                    bds.push_back(std::move(bd));
                }
                return bds;
            };
            // Clear color for a group: the game's decoded fast-clear taken from the group's first item's
            // resolved pipeline state, or its default opaque black when no fast-clear was programmed.
            // Passing this to render_draws_rgba replaces the old hardcoded debug blue on the live path
            // (#309) — PROSPER_CLEAR_DEBUG still forces blue for spotting unrendered areas.
            auto clear_for = [](const std::vector<const prosper::gpu::DrawItem*>& g) -> const float* {
                return g.empty() ? nullptr : g.front()->ps.clear_color;
            };
            // Declared beside pass_timing_start, not at the loop's end: the group loop lives in an
            // inner scope that closes before the pass span is accumulated, so a tail marker declared
            // there is out of scope where it is needed.
            RenderClock::time_point pass_tail_start{};
            double pass_tail_measured_before = 0.0;
            const auto pass_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            if (timing_enabled)
                pending_timing.prelude_ms += std::chrono::duration<double, std::milli>(
                    pass_timing_start - callback_timing_start).count();
            std::shared_ptr<const std::vector<uint8_t>> selected_pixels;
            // Provenance of whatever `selected_pixels` ends up holding. It travels with the frame to
            // the present layer so a verification gate can tell a composited frame from a
            // republished guest scanout — `--require-composited-frame` asserts prosper rendered
            // something, and #2026 was reverted (#2044) for letting the one frame that assertion
            // exists to reject satisfy it.
            auto frame_origin = prosper::gpu::PresentFrameOrigin::Composited;
            bool published_gpu = false;
            // The extent contract with our caller (#1986). A submit headed for the publish gate is
            // published only as exactly w*h*4 bytes and anything else is silently discarded, so a
            // rendered pass of a different extent is not a present source at all — see
            // present_extent.hpp for what that changes, and PresentSubmitScope in gpu_execute.hpp for
            // why the answer cannot be derived here: render_submit_items (gpu_replay's ordered-prefix
            // inspection, the renderer's own tests) deliberately wants the last pass AT ITS OWN
            // EXTENT, and is indistinguishable from a present by anything visible in this callback.
            // Zero means "no contract", which is that path and not a degenerate case.
            const size_t present_extent_bytes = prosper::gpu::present_submit_in_progress()
                ? static_cast<size_t>(w) * h * 4u : 0u;
            // The chosen source and each candidate's own extent, so the report below can name what was
            // available rather than only that nothing was.
            prosper::frontend::PresentSourceChoice present_choice =
                prosper::frontend::PresentSourceChoice::None;
            uint32_t px_front_w = 0, px_front_h = 0;
            uint32_t px_vo_w = 0, px_vo_h = 0;
            uint32_t px_last_w = 0, px_last_h = 0;
            uint64_t px_front_base = 0, px_vo_base = 0, px_last_base = 0;
            // The raw CB_COLOR0_INFO.FORMAT of the pass that produced each present candidate, so the
            // SELECTED one's format can be reported (#2283). UINT32_MAX is "no candidate recorded",
            // deliberately not 0: 0 is CB_COLOR_INVALID, a real and meaningful value here, and the
            // whole question is how often it reaches the publish source. Absent and disabled must
            // not be the same number.
            constexpr uint32_t kNoPassFormat = UINT32_MAX;
            uint32_t px_front_fmt = kNoPassFormat, px_vo_fmt = kNoPassFormat,
                     px_last_fmt = kNoPassFormat;
            // Fail-visible: a submit that renders yet offers no present-extent source is exactly the
            // Frontiers wall, and before #1968's diagnostics nothing anywhere said so — the publish
            // counter simply stopped while the guest kept submitting. Never expected, so it reports
            // unconditionally; capped per the rate-limit contract (ordinal on every line so the last
            // one bounds the population from below, plus a power-of-two tail so the tail exists at
            // all — src/gpu/diagnostics/diag_ratelimit.hpp, instrument trap 49).
            //
            // The two totals on every line settle a question the frame counter cannot: whether a title
            // whose publish rate looks healthy is publishing FRESH frames or re-serving one retained
            // frame. `frame_seq` climbing says nothing about which (instrument trap 90), and the answer
            // decides where the next investigation looks, so the counters are carried by the instrument
            // rather than left to be inferred.
            static std::atomic<uint64_t> present_frames_stored{0};   // publishable, this submit's own
            static std::atomic<uint64_t> present_frames_served{0};   // the retained frame, re-served
            static std::atomic<uint64_t> present_extent_reports{0};
            auto report_present_extent_shortfall = [&](const char* outcome, size_t offered_bytes) {
                const uint64_t ord = present_extent_reports.fetch_add(1) + 1;
                if (!prosper::diag_should_print(ord)) return;
                auto describe = [](char* out, size_t n, uint32_t cw, uint32_t ch, uint64_t cbase) {
                    if (!cw || !ch) snprintf(out, n, "none");
                    else snprintf(out, n, "%ux%u@0x%llx", cw, ch, (unsigned long long)cbase);
                };
                char front_text[64], vo_text[64], last_text[64];
                describe(front_text, sizeof front_text, px_front_w, px_front_h, px_front_base);
                describe(vo_text, sizeof vo_text, px_vo_w, px_vo_h, px_vo_base);
                describe(last_text, sizeof last_text, px_last_w, px_last_h, px_last_base);
                fprintf(stderr,
                        "[rtt] PRESENT SOURCE EXTENT MISMATCH #%llu: no pass produced a %ux%u "
                        "(%zu-byte) present source — px_front=%s px_vo=%s px_last=%s, offered %zu "
                        "bytes; %s (published so far: fresh=%llu retained=%llu)\n",
                        (unsigned long long)ord, w, h, present_extent_bytes,
                        front_text, vo_text, last_text, offered_bytes, outcome,
                        (unsigned long long)present_frames_stored.load(std::memory_order_relaxed),
                        (unsigned long long)present_frames_served.load(std::memory_order_relaxed));
            };
            if (pertarget) {
                // PER-TARGET RTT: a real frame is a sequence of passes, each rendering into a specific
                // color target (CB_COLOR0_BASE), and a final composite pass SAMPLES the earlier targets.
                // The single-framebuffer path below flattens ALL draws into one image and caches it under
                // only the FIRST item's base — so a draw that targets a different base never gets cached
                // under its OWN address, and a later composite that samples that address misses -> black.
                // Here we group items by color0_base (first-appearance order), render each group into its
                // own framebuffer, and cache each under its base. Because groups render in order and
                // build_bds re-reads g_rtt, a composite group that samples a scene group rendered earlier
                // THIS submit now hits — closing the intra-submit multi-pass gap (and cross-submit too,
                // since g_rtt persists). The final (last) group's pixels — the composite — are presented.
                // PASSES, not groups (#300): a real frame renders in SUBMIT ORDER as A->B->A->B... (a
                // ping-pong RT chain, each pass sampling the previous target). Collapsing all draws with
                // the same color0_base into one group loses that order — an A-pass that samples B then
                // renders before B exists, reading a stale/empty B -> black cascade (and a 4-deep UE4
                // post chain propagates nothing). So split items into CONTIGUOUS same-target runs below
                // and render each in order, caching its RT immediately so the next pass sampling it hits
                // fresh pixels. (Cross-submit persistence via g_rtt is unchanged.)
                // Flip-anchored present selection: the correct frame is whatever the guest FLIPS to
                // screen, i.e. the RTT group whose color-target VA is a registered VideoOut buffer
                // (preferring the CURRENT front buffer — the in-stream SetFlip fires during the Dcb
                // fold, before this render, so present_front_index() is this frame's scanout choice).
                const int      vo_n     = prosper_vo_buffer_count();
                const int      vo_front = prosper::gpu::present_front_index();
                const uint64_t front_va = vo_front >= 0 ? prosper_vo_buffer_addr(vo_front) : 0;
                if (rtt_log) {
                    fprintf(stderr, "[rtt] flip state: front=%d va=0x%llx of %d registered:",
                            vo_front, (unsigned long long)front_va, vo_n);
                    for (int i = 0; i < vo_n && i < 8; i++)
                        fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
                    fprintf(stderr, "\n");
                }
                std::shared_ptr<const std::vector<uint8_t>> px_front;
                std::shared_ptr<const std::vector<uint8_t>> px_vo;
                std::shared_ptr<const std::vector<uint8_t>> px_last;
                // px_front_*/px_vo_*/px_last_* (each candidate's own target extent, and px_last's
                // base) are declared at callback scope above so the final-span report can name which
                // sources this submit actually offered.
                // Render-target persistence (default on; PROSPER_RTT_NOSEED reverts to per-pass blue
                // clear): seed each group's framebuffer with the pixels last rendered into that SAME
                // target VA, so a pass that draws into an already-written target (UE4's UI pass onto
                // the backbuffer, incremental HUD updates) composites OVER the earlier content instead
                // of starting from the diagnostic clear. Real RT memory persists exactly this way.
                static const bool seed_rtt = getenv("PROSPER_RTT_NOSEED") == nullptr;
                const auto& no_seed_targets = rtt_no_seed_target_selector();
                const auto seed_target = [&](uint64_t target) {
                    return seed_rtt && !(no_seed_targets.configured &&
                        no_seed_targets.includes(target));
                };
                // Pass grouping and same-pass feedback detection must not disagree about what an
                // active binding is, so both go through frontends/shared/rtt/mrt_binding.hpp. These
                // were duplicated lambdas; a second, looser copy in the feedback path classified
                // stale named state as a live binding (#2550 review).
                auto color_binding = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
                    return prosper::frontend::mrt_color_binding(draw, slot);
                };
                auto active_format = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
                    return prosper::test::backend_color_format(static_cast<VkFormat>(
                        prosper::frontend::mrt_raw_format(draw, slot)));
                };
                auto active_color = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
                    return prosper::frontend::mrt_active_color(draw, slot, mrt_format_defined);
                };
                auto active_color_count = [](const prosper::gpu::DrawItem& draw) {
                    return prosper::frontend::mrt_active_color_count(draw, mrt_format_defined);
                };
                // #3026 -- the slot-0/1 alias mirror, checked HERE, at the consumer.
                //
                // `DrawItem::color_targets[0]`/`[1]` mirror the named `color0_*`/`color1_*` triples
                // or are absent; every producer honours that and nothing enforced it. The moment
                // one does not, the readers below answer differently: this loop's pass target is
                // the RAW named field (`base`, and `native_w`/`native_h` for the framebuffer
                // extent), grouping is named-first (`mrt_pass_color_binding`), and the
                // active-binding rule behind the attachment count and feedback detection is
                // array-first (`mrt_color_binding`). #3023 is what one such disagreement cost: two
                // draws rendering to different addresses grouped into one pass, and the second
                // draw's surface was discarded with no pass, no published pixels and no diagnostic.
                //
                // It is checked here and not beside the assignments that establish it because an
                // assertion at `realize_draw_item`'s success exit would sit two lines below its own
                // mirror and could never fail. This is the place a divergence would have had to
                // survive to matter.
                //
                // Report-only, deliberately. Which representation should win is undecided, and
                // deciding it here would change the surface the renderer renders to.
                for (const auto& alias_draw : items) {
                    prosper::frontend::mrt_check_color_alias_mirror(
                        alias_draw,
                        [](uint32_t slot,
                           const prosper::gpu::DrawItem::ColorTargetBinding& carried,
                           const prosper::gpu::DrawItem::ColorTargetBinding& named,
                           uint64_t ordinal) {
                            static const bool alias_log =
                                PROSPER_ENV_VALUE("PROSPER_MRT_ALIAS_LOG") != nullptr;
                            if (ordinal > 8 && !alias_log) return;
                            fprintf(stderr,
                                    "[mrt-alias] c%u array=0x%llx %ux%u vs named=0x%llx %ux%u -- "
                                    "the pass renders to the NAMED surface; grouping, feedback and "
                                    "the attachment count may read the other one (#3026) x%llu%s\n",
                                    slot, (unsigned long long)carried.base, carried.width,
                                    carried.height, (unsigned long long)named.base, named.width,
                                    named.height, (unsigned long long)ordinal,
                                    ordinal == 8 && !alias_log
                                        ? "  (further reports need PROSPER_MRT_ALIAS_LOG=1)" : "");
                        });
                }
                size_t pass_i = 0;
                prosper::test::BackendSubmissionBatch backend_submission;
                if (timing_enabled)
                    pending_timing.pass_head_ms += std::chrono::duration<double, std::milli>(
                        RenderClock::now() - pass_timing_start).count();
                const auto pass_loop_start = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                const double pass_loop_measured_before = timing_enabled
                    ? pending_timing.build_resources_ms + pending_timing.backend_ms : 0.0;
                while (pass_i < items.size()) {
                    const auto group_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    // Counted at the TOP, before any `continue` can skip it -- so the difference
                    // from pass_groups is exactly the number of iterations that returned without
                    // rendering, which is the population the missing time must belong to.
                    if (timing_enabled) ++pending_timing.pass_groups_seen;
                    const uint64_t base = items[pass_i].color0_base;
                    const uint32_t requested_color_count = active_color_count(items[pass_i]);
                    // PROSPER_MRT_CENSUS=1 — per slot, why an attachment did or did not become
                    // active. A slot needs all three of base, a known format, and a non-zero write
                    // mask; if any is absent the attachment is dropped silently and the draws that
                    // wrote it produce nothing. On GTA V only c1 is ever published, while an exact
                    // draw census shows 122,028 of 131,072 draws binding a colour target at SLOT 4 --
                    // so one of these three is missing for that slot and nothing said which.
                    if (PROSPER_ENV_ON("PROSPER_MRT_CENSUS")) {
                        struct SlotCensus {
                            std::atomic<uint64_t> seen{0}, has_base{0}, has_format{0},
                                has_mask{0}, active{0};
                        };
                        static std::array<SlotCensus, prosper::gpu::kColorTargetCount> census;
                        static std::atomic<uint64_t> groups{0};
                        const auto& d = items[pass_i];
                        for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
                            auto& c = census[slot];
                            c.seen.fetch_add(1, std::memory_order_relaxed);
                            if (color_binding(d, slot).base)
                                c.has_base.fetch_add(1, std::memory_order_relaxed);
                            if (active_format(d, slot) != VK_FORMAT_UNDEFINED)
                                c.has_format.fetch_add(1, std::memory_order_relaxed);
                            if (d.ps.color_targets[slot].write_mask ||
                                (slot == 0 && d.ps.color_write_mask) ||
                                (slot == 1 && d.ps.color1_write_mask))
                                c.has_mask.fetch_add(1, std::memory_order_relaxed);
                            // The per-slot mask the DrawItem CARRIES, against the one its own two
                            // mask registers IMPLY. render_state.cpp computes the first as
                            // (cb_target_mask & cb_shader_mask) >> (slot*4) & 0xf, so a disagreement
                            // means the per-slot array did not survive the path that built this
                            // DrawItem -- and the array is what decides whether the attachment
                            // exists at all.
                            {
                                const uint32_t implied =
                                    ((d.ps.cb_target_mask & d.ps.cb_shader_mask) >> (slot * 4u))
                                    & 0xfu;
                                const uint32_t carried = d.ps.color_targets[slot].write_mask;
                                if (implied != carried && color_binding(d, slot).base) {
                                    static std::mutex disagree_mutex;
                                    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>,
                                                    uint64_t> disagree;
                                    std::lock_guard lock(disagree_mutex);
                                    if (disagree.size() < 48)
                                        ++disagree[{slot, implied, carried}];
                                    static uint64_t n = 0;
                                    if (++n % 8192 == 0) {
                                        fprintf(stderr,
                                                "[mrt-mask-disagree] per-slot write_mask carried by "
                                                "the DrawItem vs implied by its own registers:\n");
                                        for (const auto& e : disagree)
                                            fprintf(stderr,
                                                    "[mrt-mask-disagree]   c%u implied=0x%x "
                                                    "carried=0x%x  x%llu\n",
                                                    std::get<0>(e.first), std::get<1>(e.first),
                                                    std::get<2>(e.first),
                                                    (unsigned long long)e.second);
                                    }
                                }
                            }
                            if (active_color(d, slot))
                                c.active.fetch_add(1, std::memory_order_relaxed);
                        }
                        // WHICH of the two masks is narrow. The write mask is
                        // cb_target_mask & cb_shader_mask, and both default to 0xffffffff when the
                        // register was never seen -- so a zero nibble means a register really was
                        // programmed narrow, and the fix differs completely depending on which.
                        // Histogrammed only over groups where slot 4 HAS a base, i.e. exactly the
                        // population where the dropped attachment matters.
                        if (color_binding(d, 4).base) {
                            static std::mutex mask_mutex;
                            static std::map<std::pair<uint32_t, uint32_t>, uint64_t> masks;
                            std::lock_guard lock(mask_mutex);
                            if (masks.size() < 64)
                                ++masks[{d.ps.cb_target_mask, d.ps.cb_shader_mask}];
                            static uint64_t reported = 0;
                            if (++reported % 4096 == 0) {
                                fprintf(stderr, "[mrt-census] masks where c4 has a base:\n");
                                for (const auto& e : masks)
                                    fprintf(stderr,
                                            "[mrt-census]   cb_target_mask=0x%08x "
                                            "cb_shader_mask=0x%08x -> effective=0x%08x  x%llu\n",
                                            e.first.first, e.first.second,
                                            e.first.first & e.first.second,
                                            (unsigned long long)e.second);
                            }
                        }
                        // The SHAPE of each distinct pass: its eight slot bases next to the two
                        // mask registers. The aggregate above says how often a slot activates; it
                        // cannot say which surfaces a given pass meant to write, which is what
                        // "buffer X is sampled 23 times and never written" actually needs.
                        {
                            static std::mutex shape_mutex;
                            static std::map<std::array<uint64_t, 10>, uint64_t> shapes;
                            std::array<uint64_t, 10> shape{};
                            for (uint32_t slot = 0; slot < 8u; ++slot)
                                shape[slot] = color_binding(d, slot).base;
                            shape[8] = d.ps.cb_target_mask;
                            shape[9] = d.ps.cb_shader_mask;
                            // PROSPER_MRT_SHAPE_FOR=0xADDR[,...] restricts recording to passes
                            // whose slot-0 base is named. Without it the map fills with startup and
                            // scanout passes long before the gameplay G-buffer pass appears, and an
                            // unfiltered top-10 is then a list of the most FREQUENT shapes rather
                            // than the ones being asked about.
                            static const std::string shape_for =
                                PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR")
                                    ? PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR") : "";
                            bool shape_wanted = shape_for.empty();
                            if (!shape_wanted && shape[0]) {
                                char needle[24];
                                std::snprintf(needle, sizeof needle, "0x%llx",
                                              (unsigned long long)shape[0]);
                                shape_wanted = shape_for.find(needle) != std::string::npos;
                            }
                            if (!shape_wanted) goto shape_done;
                            {
                            std::lock_guard lock(shape_mutex);
                            if (shapes.size() < 96 || shapes.count(shape)) ++shapes[shape];
                            static uint64_t shape_reports = 0;
                            if (++shape_reports % (shape_for.empty() ? 8192 : 256) == 0) {
                                std::vector<std::pair<uint64_t, std::array<uint64_t, 10>>> ranked;
                                for (const auto& e : shapes) ranked.push_back({e.second, e.first});
                                std::sort(ranked.begin(), ranked.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.first > b.first; });
                                fprintf(stderr, "[mrt-shape] %zu distinct pass shapes\n",
                                        shapes.size());
                                for (size_t i = 0; i < ranked.size() && i < 10; ++i) {
                                    fprintf(stderr, "[mrt-shape]   x%-6llu tmask=0x%08llx "
                                            "smask=0x%08llx bases:",
                                            (unsigned long long)ranked[i].first,
                                            (unsigned long long)ranked[i].second[8],
                                            (unsigned long long)ranked[i].second[9]);
                                    for (uint32_t slot = 0; slot < 8u; ++slot)
                                        if (ranked[i].second[slot])
                                            fprintf(stderr, " c%u=0x%llx", slot,
                                                    (unsigned long long)ranked[i].second[slot]);
                                    fprintf(stderr, "\n");
                                }
                            }
                            }
                            shape_done: ;
                        }
                        const uint64_t g = groups.fetch_add(1) + 1;
                        if ((g & (g - 1)) == 0 && g >= 4096) {
                            fprintf(stderr, "[mrt-census] pass groups=%llu\n",
                                    (unsigned long long)g);
                            for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot)
                                fprintf(stderr,
                                        "[mrt-census]   c%u base=%llu format=%llu mask=%llu "
                                        "ACTIVE=%llu\n",
                                        slot,
                                        (unsigned long long)census[slot].has_base.load(),
                                        (unsigned long long)census[slot].has_format.load(),
                                        (unsigned long long)census[slot].has_mask.load(),
                                        (unsigned long long)census[slot].active.load());
                        }
                    }
                    std::array<uint64_t, prosper::gpu::kColorTargetCount> pass_bases{};
                    std::array<VkFormat, prosper::gpu::kColorTargetCount> pass_formats{};
                    for (uint32_t slot = 0; slot < requested_color_count; ++slot) {
                        pass_bases[slot] = slot ? active_color(items[pass_i], slot) : base;
                        pass_formats[slot] = active_format(items[pass_i], slot);
                    }
                    const uint64_t base1 = pass_bases[1];
                    const VkFormat format0 = pass_formats[0];
                    const VkFormat format1 = base1
                        ? pass_formats[1] : VK_FORMAT_UNDEFINED;
                    // The DEPTH/STENCIL attachment is part of a pass's target identity, not just its
                    // colour attachments.
                    //
                    // A render pass has ONE depth attachment, and the backend picks one cached DS
                    // image for the whole grouped call from its first meaningful draw. Grouping on
                    // colour alone therefore let a single call span draws that name different DS
                    // surfaces -- or, for a layered surface, different DB_DEPTH_VIEW SLICES of one
                    // allocation. Every such draw then rendered into the first face the call
                    // happened to select.
                    //
                    // Measured on GTA V (Codex, #2542): one grouped call crosses DB_DEPTH_VIEW from
                    // slice 0 to slice 1 at draw 65, so several guest cube faces were being rendered
                    // into one host face. That also makes any "the cube has six valid faces"
                    // measurement void: the handles existed, their contents did not correspond to
                    // six guest faces.
                    auto ds_identity = [](const prosper::gpu::DrawItem& draw) {
                        return std::tuple(draw.ps.depth_read_base, draw.ps.depth_write_base,
                                          draw.ps.stencil_read_base, draw.ps.stencil_write_base,
                                          draw.ps.htile_data_base,
                                          prosper::test::ds_depth_view_slice_start(
                                              draw.ps.db_depth_view));
                    };
                    const prosper::gpu::DrawItem& pass_head = items[pass_i];
                    const auto ds0 = ds_identity(pass_head);
                    std::vector<const prosper::gpu::DrawItem*> pass;
                    auto same_targets = [&](const prosper::gpu::DrawItem& draw) {
                        if (!prosper::frontend::mrt_same_color_pass(
                                pass_head, draw, mrt_format_defined, active_format))
                            return false;
                        if (ds_identity(draw) != ds0) return false;
                        return true;
                    };
                    // A MODE=RESOLVE draw is answered by a COPY below, not by a render, so its
                    // pass identity is the source/destination pair that copy acts with rather than
                    // the attachment set `same_targets` compares. `mrt_same_resolve_pass` carries
                    // that whole rule -- keeping a resolve out of a scene group, and keeping two
                    // resolves with different destinations out of each other's (#3025) -- and its
                    // header states why each half exists.
                    while (pass_i < items.size() && same_targets(items[pass_i]) &&
                           prosper::frontend::mrt_same_resolve_pass(pass_head, items[pass_i])) {
                        pass.push_back(&items[pass_i]); ++pass_i;
                    }

                    // CB_COLOR_CONTROL.MODE=RESOLVE(3): the guest resolves an MSAA color0 surface into a
                    // single-sample color1 destination (Blue Prince PPSA25009 resolves its 4x-MSAA scene
                    // this way). prosper renders single-sample, so the resolve is a straight copy of the
                    // already-rendered color0 surface into color1. Use the RAW color1_base, not
                    // active_color1(): a fixed-function resolve exports nothing, so its color1_write_mask
                    // is 0 and active_color1 would report no destination. Without this the resolved surface
                    // the display later samples never receives the scene and the frame is a uniform fill.
                    // Census every MODE=RESOLVE pass: source, destination and whether the
                    // destination is one prosper can even express. prosper takes the destination
                    // from `color1_base` alone, so a guest resolving into any other slot is
                    // invisible to the resolve path -- a surface filled that way would read as
                    // "written by nothing" in every other census.
                    if (!pass.empty() && pass.front()->ps.cb_resolve &&
                        PROSPER_ENV_ON("PROSPER_RESOLVE_CENSUS")) {
                        static std::mutex mutex;
                        static std::map<std::pair<uint64_t, uint64_t>, uint64_t> seen;
                        const uint64_t rsrc = pass.front()->color0_base;
                        const uint64_t rdst = pass.front()->color1_base;
                        std::lock_guard lock(mutex);
                        const uint64_t n = ++seen[{rsrc, rdst}];
                        if (n <= 2 || (n & (n - 1)) == 0)
                            fprintf(stderr,
                                    "[resolve-census] src=0x%llx dst(color1)=0x%llx x%llu%s\n",
                                    (unsigned long long)rsrc, (unsigned long long)rdst,
                                    (unsigned long long)n,
                                    rdst ? "" : "  <-- NO EXPRESSIBLE DESTINATION");
                    }
                    static const bool no_resolve = getenv("PROSPER_NO_RESOLVE") != nullptr;
                    if (!pass.empty() && pass.front()->ps.cb_resolve && !no_resolve) {
                        const uint64_t rsrc = pass.front()->color0_base;
                        const uint64_t rdst = pass.front()->color1_base;
                        auto src_it = rsrc ? g_rtt.find(rsrc) : g_rtt.end();
                        if (src_it != g_rtt.end() && src_it->second.has_uniform_color)
                            materialize_uniform_rtt(src_it->second);
                        // Deferred RTT readback (#1284): the resolve is a CPU copy of the source's
                        // pixels, and the source pass usually rendered EARLIER IN THIS BATCH with its
                        // readback deferred. The consumer scan cannot see a resolve (it consumes via
                        // cb_resolve, not a sampled resource), so materialize here: flush the pending
                        // batch first — a mid-batch readback would otherwise return stale pixels —
                        // then read the persistent image back once.
                        // PROSPER_EAGER_RESOLVE_READBACK: restore the pre-#2266 behaviour, where the
                        // CPU mirror was materialised here unconditionally. Kept as a permanent
                        // bisection lever so the A/B for this change is single-variable on ONE binary,
                        // and because this path sits on the #1334/#1382/#1287 evidence chain -- if a
                        // title regresses, the first question is whether the lazy mirror caused it and
                        // that has to be answerable without a rebuild.
                        //
                        // Measured before the change, Blue Prince gameplay window (#2266):
                        //   resolve 5.1/submit [stall=20.76 read=20.63 copy=1.54] 63.3 MiB/submit
                        // 63.3 MiB at 3.1 GB/s -- a PCIe readback rate -- for a CPU mirror that this
                        // file materialises ON DEMAND everywhere else (see the readback at the compute
                        // consumer, whose comment says intermediates "normally stay in the persistent
                        // Vulkan target" and are materialised "only when an ordered compute dispatch
                        // actually consumes the surface"). The resolve was doing eagerly what the rest
                        // of the renderer does lazily, and its own comment already calls the CPU pixels
                        // the failure case: "on failure the shared CPU pixels are the only truth".
                        static const bool eager_resolve_readback =
                            getenv("PROSPER_EAGER_RESOLVE_READBACK") != nullptr;
                        if (src_it != g_rtt.end() && src_it->second.gpu_valid &&
                            src_it->second.w && src_it->second.h && rdst) {
                            RttSurf& src_surface = src_it->second;
                            const VkFormat src_format =
                                prosper::test::backend_color_format(src_surface.format);
                            const uint32_t src_bpp =
                                prosper::test::backend_color_bytes_per_pixel(src_format);
                            const size_t src_bytes = static_cast<size_t>(src_surface.w) *
                                src_surface.h * src_bpp;
                            if (src_bpp && eager_resolve_readback &&
                                (!src_surface.rgba || src_surface.rgba->size() != src_bytes)) {
                                const prosper::test::RenderVkCtx& ctx =
                                    prosper::test::render_vk_ctx();
                                const auto rs0 = timing_enabled
                                    ? RenderClock::now() : RenderClock::time_point{};
                                if (ctx.ok && backend_submission.pending())
                                    backend_submission.submit_and_wait(ctx.dev, ctx.queue, false);
                                const auto rs1 = timing_enabled
                                    ? RenderClock::now() : RenderClock::time_point{};
                                std::vector<uint8_t> materialized;
                                std::string error;
                                const bool read_ok = prosper::test::readback_persistent_color_target(
                                        rsrc, src_surface.w, src_surface.h, src_format,
                                        materialized, error);
                                if (timing_enabled) {
                                    const auto rs2 = RenderClock::now();
                                    pending_timing.resolve_stall_ms +=
                                        std::chrono::duration<double, std::milli>(rs1 - rs0).count();
                                    pending_timing.resolve_read_ms +=
                                        std::chrono::duration<double, std::milli>(rs2 - rs1).count();
                                    ++pending_timing.resolve_read_n;
                                    pending_timing.resolve_bytes += src_bytes;
                                }
                                if (read_ok &&
                                    materialized.size() == src_bytes) {
                                    src_surface.rgba =
                                        std::make_shared<const std::vector<uint8_t>>(
                                            std::move(materialized));
                                } else {
                                    static std::atomic<int> warned{0};
                                    if (warned.fetch_add(1) < 24)
                                        fprintf(stderr,
                                                "[rtt] resolve source readback failed: "
                                                "base=0x%llx extent=%ux%u error=%s\n",
                                                (unsigned long long)rsrc, src_surface.w,
                                                src_surface.h, error.c_str());
                                }
                            }
                        }
                        // A GPU-only source is now the ordinary case, so the destination no longer
                        // requires CPU pixels to exist before the resolve. `gpu_valid` is what the
                        // copy below needs; `rgba` is inherited when the source happens to have it
                        // and is otherwise materialised on demand by the consumer path, exactly as
                        // for every other graphics-to-graphics intermediate.
                        if (rsrc && rdst && src_it != g_rtt.end() &&
                            (src_it->second.rgba || src_it->second.gpu_valid)) {
                            // Copy the source RttSurf out before the g_rtt[rdst] insert: operator[] may
                            // rehash and invalidate src_it before the assignment reads it.
                            RttSurf resolved = src_it->second;   // shares pixels (shared_ptr), no deep copy
                            // The resolve destination has its own descriptor/DCC allocation. Do not
                            // transfer the source surface's metadata identity to an unrelated base.
                            resolved.dcc_metadata_addr = 0;
                            resolved.dcc_metadata_bytes = 0;
                            resolved.dcc_metadata_dirty = false;
                            const uint32_t rw = resolved.w, rh = resolved.h;
                            // #1334: the destination-keyed persistent GPU image did NOT receive these
                            // pixels — inheriting gpu_valid from the source let a later #780 CPU-copy
                            // discard leave consumers importing a stale/zero image (Blue Prince's
                            // compute tonemap read black; #1287/#1381 evidence chain). Copy the
                            // device-local pixels into the destination identity so gpu_valid is
                            // genuinely true; on failure the shared CPU pixels are the only truth.
                            if (resolved.gpu_valid) {
                                // #1382 review: the copy submits out-of-band, so any pending
                                // batched pass — including one targeting the DESTINATION identity
                                // — must reach the queue first, or it would execute after the copy
                                // and overwrite the resolve while gpu_valid=true. Mirror the
                                // source-materialization flush above unconditionally here.
                                {
                                    const prosper::test::RenderVkCtx& copy_ctx =
                                        prosper::test::render_vk_ctx();
                                    const auto cs0 = timing_enabled
                                        ? RenderClock::now() : RenderClock::time_point{};
                                    if (copy_ctx.ok && backend_submission.pending())
                                        backend_submission.submit_and_wait(copy_ctx.dev,
                                                                           copy_ctx.queue, false);
                                    if (timing_enabled)
                                        pending_timing.resolve_copy_stall_ms +=
                                            std::chrono::duration<double, std::milli>(
                                                RenderClock::now() - cs0).count();
                                }
                                std::string copy_error;
                                const auto rc0 = timing_enabled
                                    ? RenderClock::now() : RenderClock::time_point{};
                                const bool copy_ok = prosper::test::copy_persistent_color_target(
                                        rsrc, rdst, rw, rh,
                                        prosper::test::backend_color_format(resolved.format),
                                        copy_error);
                                if (timing_enabled)
                                    pending_timing.resolve_copy_ms +=
                                        std::chrono::duration<double, std::milli>(
                                            RenderClock::now() - rc0).count();
                                if (!copy_ok) {
                                    resolved.gpu_valid = false;
                                    // The CPU mirror is the documented fallback -- "on failure the
                                    // shared CPU pixels are the only truth" -- so with the eager
                                    // readback gone it has to be materialised HERE, on the failure
                                    // path it was always meant for. Without this the destination
                                    // would have neither a valid GPU image nor pixels, which is
                                    // strictly worse than before this change rather than merely
                                    // slower. The flush is required for the same reason the eager
                                    // path needed one: a mid-batch readback returns stale pixels.
                                    if (!resolved.rgba || resolved.rgba->size() !=
                                            static_cast<size_t>(rw) * rh *
                                            prosper::test::backend_color_bytes_per_pixel(
                                                prosper::test::backend_color_format(resolved.format))) {
                                        const prosper::test::RenderVkCtx& fb_ctx =
                                            prosper::test::render_vk_ctx();
                                        if (fb_ctx.ok && backend_submission.pending())
                                            backend_submission.submit_and_wait(fb_ctx.dev,
                                                                               fb_ctx.queue, false);
                                        std::vector<uint8_t> fallback;
                                        std::string fb_error;
                                        if (prosper::test::readback_persistent_color_target(
                                                rsrc, rw, rh,
                                                prosper::test::backend_color_format(resolved.format),
                                                fallback, fb_error))
                                            resolved.rgba =
                                                std::make_shared<const std::vector<uint8_t>>(
                                                    std::move(fallback));
                                    }
                                    static std::atomic<int> warned{0};
                                    if (warned.fetch_add(1) < 8)
                                        fprintf(stderr,
                                                "[msaa] resolve GPU copy failed (%s) — destination "
                                                "keeps CPU pixels only (#1334)\n",
                                                copy_error.c_str());
                                }
                            }
                            g_rtt[rdst] = std::move(resolved);    // dest inherits src content/extent/format
                            if (PROSPER_ENV_ON("PROSPER_MSAA_LOG"))
                                fprintf(stderr, "[msaa] resolve copy 0x%llx -> 0x%llx (%ux%u)\n",
                                        (unsigned long long)rsrc, (unsigned long long)rdst, rw, rh);
                        } else {
                            // A resolve that performs no copy is DROPPED RENDERED CONTENT: the
                            // destination keeps whatever it held before, and every later census
                            // reads it as "written by nothing". Report it unconditionally, bounded
                            // like the copy-failure warning above, so the one remaining way a
                            // resolve can produce nothing is fail-visible rather than silent.
                            // PROSPER_MSAA_LOG still reports every occurrence.
                            static std::atomic<int> skipped{0};
                            if (PROSPER_ENV_VALUE("PROSPER_MSAA_LOG") ||
                                skipped.fetch_add(1) < 8)
                                fprintf(stderr, "[msaa] resolve SKIP src=0x%llx dst=0x%llx (%s)\n",
                                        (unsigned long long)rsrc, (unsigned long long)rdst,
                                        !rsrc || !rdst ? "missing base"
                                                       : "source has no rendered surface");
                        }
                        if (timing_enabled) ++pending_timing.resolve_n;
                        continue;   // resolve is a copy, not an ordinary-draw render
                    }

                    // Gen5 render-target extent (#526). Large scene/scanout surfaces retain the
                    // configured VideoOut render scale; small offscreen targets render at native
                    // resolution so lookup textures (Messenger's 1024x32 grading LUT) preserve every
                    // texel. The viewport was already scaled for the global w/h framebuffer by
                    // execute_gpustate, so correct it from that scale to this pass-local scale.
                    uint32_t native_w = pass.empty() ? 0u : pass.front()->color0_width;
                    uint32_t native_h = pass.empty() ? 0u : pass.front()->color0_height;
                    uint32_t gw = w, gh = h;
                    const uint32_t present_w = prosper::gpu::present_width();
                    const uint32_t present_h = prosper::gpu::present_height();
                    bool is_vo = false;
                    for (int i = 0; i < vo_n && !is_vo; i++)
                        is_vo = base && base == prosper_vo_buffer_addr(i);
                    // VideoOut registration is the visible scanout contract. CB_COLOR0_ATTRIB2 can
                    // describe an overallocated backing surface (Terminator reports 4096x4096 for a
                    // 1920x1080 scanout), which must not become the persistent target extent or the
                    // final cache lookup will reject the rendered frame.
                    if (is_vo && present_w && present_h) {
                        native_w = present_w;
                        native_h = present_h;
                    }
                    // A depth prepass may bind a tiny/dummy color target with CB_TARGET_MASK=0 while
                    // rasterizing the full-size guest depth surface. Keying persistent DS from that
                    // irrelevant color extent creates a small depth image that the later lighting pass
                    // cannot reuse. Recover the attachment extent from the viewport, but only when the
                    // complete inferred extent fits the known presentation surface. Viewport coordinates
                    // position rasterization and are not allocation metadata; accepting translated or
                    // otherwise oversized coordinates here previously created persistent images as large
                    // as 16384x16384 and exhausted the host while Dead Cells loaded its first level.
                    const bool color_disabled = !pass.empty() && std::all_of(
                        pass.begin(), pass.end(), [](const auto* draw) {
                            return draw->ps.color_write_mask == 0;
                        });
                    const bool uses_ds = std::any_of(pass.begin(), pass.end(), [](const auto* draw) {
                        return draw->ps.depth_test_enable || draw->ps.depth_write_enable ||
                               draw->ps.depth_clear_enable || draw->ps.stencil_enable ||
                               draw->ps.stencil_clear_enable;
                    });
                    if (color_disabled && uses_ds && w && h) {
                        float viewport_x = 0.0f, viewport_y = 0.0f;
                        for (const auto* draw : pass) {
                            if (!draw->ps.has_viewport) continue;
                            const float x1 = draw->ps.viewport_x;
                            const float x2 = x1 + draw->ps.viewport_w;
                            const float y1 = draw->ps.viewport_y;
                            const float y2 = y1 + draw->ps.viewport_h;
                            if (std::isfinite(x1) && std::isfinite(x2))
                                viewport_x = std::max(viewport_x, std::max(std::fabs(x1), std::fabs(x2)));
                            if (std::isfinite(y1) && std::isfinite(y2))
                                viewport_y = std::max(viewport_y, std::max(std::fabs(y1), std::fabs(y2)));
                        }
                        const uint32_t max_native_w = present_w ? present_w : w;
                        const uint32_t max_native_h = present_h ? present_h : h;
                        const uint64_t viewport_native_w = static_cast<uint64_t>(
                            std::ceil(viewport_x * static_cast<float>(max_native_w) / w));
                        const uint64_t viewport_native_h = static_cast<uint64_t>(
                            std::ceil(viewport_y * static_cast<float>(max_native_h) / h));
                        // Depth-only surfaces legitimately exceed the presentation extent: Blue
                        // Prince's directional-shadow cascades render translated 512x512/1024x1024
                        // viewports into 2048x2048 and 4096x4096 atlases (#1275). Capping at the
                        // presentation surface collapsed those atlases to 1x1, erasing every shadow.
                        // The per-axis bound (each axis against its own presentation axis, or 4096)
                        // keeps the Dead Cells pathology (translated viewports inferring
                        // 16384x16384, the reason this cap exists) rejected and fail-visible.
                        const bool viewport_extent_valid = viewport_native_w && viewport_native_h &&
                            viewport_native_w <= std::max<uint64_t>(4096u, max_native_w) &&
                            viewport_native_h <= std::max<uint64_t>(4096u, max_native_h);
                        // Log the UNDECIDABLE case too. Gating this on a non-zero derived extent hid
                        // the only outcome that silently changes the DS identity: a depth-only pass
                        // whose draws carry no viewport register derives 0x0, falls through to the
                        // global frame extent, and mints a second cache entry for a surface that
                        // already has a correctly-sized one. A diagnostic that prints only when the
                        // inference succeeded cannot report the inference not happening.
                        if (PROSPER_ENV_ON("PROSPER_DSLOG")) {
                            const size_t with_viewport = static_cast<size_t>(std::count_if(
                                pass.begin(), pass.end(),
                                [](const auto* draw) { return draw->ps.has_viewport; }));
                            fprintf(stderr,
                                    "[ds] viewport-derived extent %llux%llu (presentation %ux%u, "
                                    "%zu/%zu draws with viewport) -> %s\n",
                                    (unsigned long long)viewport_native_w,
                                    (unsigned long long)viewport_native_h,
                                    max_native_w, max_native_h, with_viewport, pass.size(),
                                    viewport_extent_valid ? "accept"
                                        : (viewport_native_w || viewport_native_h) ? "reject"
                                                                                   : "undecidable");
                        }
                        if (viewport_extent_valid) {
                            native_w = std::max(native_w, static_cast<uint32_t>(viewport_native_w));
                            native_h = std::max(native_h, static_cast<uint32_t>(viewport_native_h));
                        }
                    }
                    if (native_w && native_h) {
                        uint64_t native_pixels = (uint64_t)native_w * native_h;
                        uint64_t global_pixels = (uint64_t)w * h;
                        if (native_pixels <= global_pixels) {
                            gw = native_w; gh = native_h;
                        } else if (present_w && present_h) {
                            gw = std::max(1u, (uint32_t)(((uint64_t)native_w * w + present_w / 2) / present_w));
                            gh = std::max(1u, (uint32_t)(((uint64_t)native_h * h + present_h / 2) / present_h));
                        } else if (color_disabled && uses_ds) {
                            // No presentation extent (offline replay): render scale is 1, so a
                            // depth-only surface's viewport-derived extent is exact. Falling back
                            // to the global w/h here silently truncated over-presentation atlases —
                            // Blue Prince's 2048x2048 shadow atlas became a 1920x1080 DS image
                            // whose sampled taps then missed the bridge (#1275). Color passes keep
                            // the historical capture-extent truncation (their over-allocated
                            // CB_COLOR0_ATTRIB2 backings are a different, hash-pinned contract).
                            gw = native_w; gh = native_h;
                        }
                    }
                    std::vector<prosper::gpu::DrawItem> adjusted;
                    std::vector<const prosper::gpu::DrawItem*> render_pass = pass;
                    if (native_w && native_h && present_w && present_h && w && h) {
                        const float ax = ((float)gw / native_w) / ((float)w / present_w);
                        const float ay = ((float)gh / native_h) / ((float)h / present_h);
                        if (ax != 1.0f || ay != 1.0f) {
                            adjusted.reserve(pass.size()); render_pass.clear(); render_pass.reserve(pass.size());
                            for (const auto* src : pass) {
                                adjusted.push_back(*src);
                                auto& item = adjusted.back();
                                prosper::gpu::scale_resolved_render_area(item.ps, ax, ay);
                                render_pass.push_back(&item);
                            }
                        }
                    }
                    const uint8_t* seed = nullptr;
                    const float* retained_uniform_clear = nullptr;
                    bool gpu_seed_available = false;
                    const bool seed_rtt0 = seed_target(base);
                    const VkFormat pass_format = format0;
                    uint32_t mrt_count = requested_color_count;
                    if (PROSPER_ENV_ON("PROSPER_NO_MRT1") || PROSPER_ENV_ON("PROSPER_NO_MRT")) mrt_count = 1;
                    if (!render_pass.empty()) {
                        for (uint32_t slot = 1; slot < mrt_count; ++slot) {
                            const auto binding = color_binding(*render_pass.front(), slot);
                            // Sparse exports retain their native Location through dummy attachments.
                            // Only a real target whose extent is KNOWN to differ from the pass
                            // extent truncates the prefix. A 0x0 on either side means the guest's
                            // CB_COLORn_ATTRIB2 was never seen, not a 0-pixel surface, and an
                            // unmeasured extent is no evidence of a conflict — comparing it for
                            // equality dropped the whole attachment silently (#2114).
                            if (pass_bases[slot] && prosper::frontend::mrt_extent_conflicts(
                                    binding.width, binding.height, native_w, native_h)) {
                                if (rtt_log)
                                    fprintf(stderr,
                                            "[rtt] truncate MRT prefix at c%u target=0x%llx "
                                            "extent=%ux%u; MRT0 is %ux%u\n",
                                            slot, (unsigned long long)pass_bases[slot],
                                            binding.width, binding.height, native_w, native_h);
                                mrt_count = slot;
                                break;
                            }
                            // Fail-visible: name the attachment that was kept on missing extent
                            // data, so an unmeasured extent is a reported state rather than a
                            // silent one in either direction.
                            if (rtt_log && pass_bases[slot] &&
                                (!prosper::frontend::mrt_extent_known(binding.width, binding.height) ||
                                 !prosper::frontend::mrt_extent_known(native_w, native_h)))
                                fprintf(stderr,
                                        "[rtt] keep MRT c%u target=0x%llx on unmeasured extent "
                                        "%ux%u; MRT0 is %ux%u\n",
                                        slot, (unsigned long long)pass_bases[slot],
                                        binding.width, binding.height, native_w, native_h);
                        }
                    }
                    const bool use_color1 = mrt_count > 1;
                    const VkFormat pass_format1 = use_color1 ? format1 : VK_FORMAT_UNDEFINED;
                    const size_t pass_bytes = static_cast<size_t>(gw) * gh *
                        prosper::test::backend_color_bytes_per_pixel(pass_format);
                    if (seed_rtt0 && base) { auto sit = g_rtt.find(base);
                        gpu_seed_available = live_gpu_targets && sit != g_rtt.end() &&
                            sit->second.gpu_valid && sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format &&
                            prosper::test::find_persistent_color_target(
                                base, gw, gh, pass_format) != nullptr;
                        if (!gpu_seed_available && sit != g_rtt.end() &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format && sit->second.rgba &&
                            sit->second.rgba->size() == pass_bytes)
                            seed = sit->second.rgba->data();
                        if (!gpu_seed_available && !seed && sit != g_rtt.end() &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format &&
                            sit->second.has_uniform_color)
                            retained_uniform_clear = sit->second.uniform_color.data();
                        // Gated seed-decision diagnostic: a pass that should LOAD prior target
                        // content but silently falls back to its clear color erases everything the
                        // earlier pass produced (an opaque-black clear wipes a transparent UI RT —
                        // #320's dialogue overlay). Make the decision and its reason visible.
                        if (rtt_log && !seed && !gpu_seed_available) {
                            if (sit == g_rtt.end())
                                fprintf(stderr, "[rtt] seed miss target=0x%llx reason=no-entry\n",
                                        (unsigned long long)base);
                            else
                                fprintf(stderr,
                                        "[rtt] seed miss target=0x%llx reason=mismatch "
                                        "entry=%ux%u fmt=%d rgba=%zu want=%ux%u fmt=%d bytes=%zu\n",
                                        (unsigned long long)base, sit->second.w, sit->second.h,
                                        (int)sit->second.format,
                                        sit->second.rgba ? sit->second.rgba->size() : (size_t)0,
                                        gw, gh, (int)pass_format, pass_bytes);
                        } }
                    struct LaterTargetConsumers {
                        bool sampled_exact = false;
                        bool feedback = false;
                        bool cpu_needed = false;
                        uint32_t storage_references = 0;
                        uint32_t dimension_mismatches = 0;
                        uint32_t extent_mismatches = 0;
                        uint32_t feedback_references = 0;
                    };
                    // A later pass in THIS batch that reads the target's CPU bytes cannot be served
                    // lazily: the producing commands are still unsubmitted when it binds, so a
                    // mid-batch readback would return stale pixels. Only the direct GPU bind (2D
                    // texture, exact extent, not feedback, not storage) is batch-ordered; every
                    // other same-batch consumer forces the eager readback. A later color attachment
                    // LOAD is now direct for both MRT attachments and needs no CPU copy. Volume
                    // (img_dim==2) consumers read guest memory on both paths and never block. pass_i
                    // has already advanced past the current pass, so every scanned item is genuine.
                    // Cross-batch consumers materialize on demand at bind/seed/compute/DMA time
                    // (#1284).
                    const auto inspect_later_consumers = [&](uint64_t target_base) {
                        LaterTargetConsumers result;
                        if (!live_gpu_targets || !target_base) return result;
                        static const uint32_t render_scale = [] {
                            const char* e = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
                            const long v = e ? std::strtol(e, nullptr, 10) : 1;
                            return v > 0 ? static_cast<uint32_t>(v) : 1u;
                        }();
                        for (size_t later = pass_i; later < items.size(); ++later) {
                            auto inspect = [&](const prosper::gpu::ShaderResourceTable* table) {
                                if (!table) return;
                                for (const auto& resource : table->resources) {
                                    if ((resource.cls != RC::Texture &&
                                         resource.cls != RC::StorageImage) ||
                                        resource.gpu_addr != target_base || resource.img_dim == 2u)
                                        continue;
                                    const uint32_t rw = resource.width ? resource.width : 4u;
                                    const uint32_t rh = resource.height ? resource.height : 4u;
                                    bool same_pass_target =
                                        items[later].color0_base == target_base;
                                    for (uint32_t slot = 1;
                                         slot < prosper::gpu::kColorTargetCount; ++slot)
                                        same_pass_target |=
                                            active_color(items[later], slot) == target_base;
                                    const bool sampled_extent_compatible =
                                        prosper::frontend::rtt_sampled_extent_compatible(
                                            rw, rh, gw, gh, render_scale, false);
                                    if (resource.img_dim == 1u && sampled_extent_compatible) {
                                        result.sampled_exact = true;
                                        result.feedback |= same_pass_target;
                                    }
                                    // Exact 2D color feedback is GPU-bindable: the backend copies
                                    // the prior attachment version to a distinct sampled image before
                                    // the render pass. Storage, dimension, and extent mismatches still
                                    // require the CPU representation.
                                    const bool direct_bindable =
                                        resource.cls == RC::Texture && resource.img_dim == 1u &&
                                        sampled_extent_compatible;
                                    if (!direct_bindable) {
                                        // #3415: `dimension` is 100% of the same-batch CPU readbacks
                                        // on PPSA03831 (78,528 counts, 158 GB/run), so the ONE number
                                        // that decides whether the fix is a view type or a real image
                                        // mismatch is the array layer count. A single-layer array is
                                        // legally viewable as VK_IMAGE_VIEW_TYPE_2D_ARRAY over an
                                        // arrayLayers=1 image; a multi-layer one is a different image
                                        // and no view widening can serve it. Report the shape rather
                                        // than inferring it.
                                        if (resource.img_dim != 1u) {
                                            static const bool shape_log =
                                                std::getenv("PROSPER_READBACK_DIM_SHAPE") != nullptr;
                                            if (shape_log) {
                                                static std::mutex m;
                                                static std::map<std::tuple<uint32_t, uint32_t,
                                                                           uint32_t, uint32_t>,
                                                                uint64_t> rows;
                                                std::lock_guard<std::mutex> lk(m);
                                                auto& n = rows[std::make_tuple(
                                                    resource.img_dim, resource.depth,
                                                    resource.sample_count,
                                                    (uint32_t)resource.cls)];
                                                if (++n % 256 == 0)
                                                    std::fprintf(stderr,
                                                        "[readback-dim] img_dim=%u depth=%u "
                                                        "layers(sample_count)=%u cls=%u extent=%ux%u "
                                                        "count=%llu\n",
                                                        resource.img_dim, resource.depth,
                                                        resource.sample_count,
                                                        (unsigned)resource.cls, rw, rh,
                                                        (unsigned long long)n);
                                            }
                                        }
                                        result.cpu_needed = true;
                                        result.storage_references +=
                                            resource.cls == RC::StorageImage;
                                        result.dimension_mismatches += resource.img_dim != 1u;
                                        result.extent_mismatches += !sampled_extent_compatible;
                                        result.feedback_references += same_pass_target;
                                        static const uint64_t diagnose_min_submit = [] {
                                            const char* text = PROSPER_ENV_VALUE(
                                                "PROSPER_READBACK_WHY_MIN_SUBMIT");
                                            return text ? std::strtoull(text, nullptr, 0) : 0ull;
                                        }();
                                        static std::atomic<uint64_t> diagnose_lines{0};
                                        if (PROSPER_ENV_ON("PROSPER_READBACK_WHY") &&
                                            static_cast<uint64_t>(g_this_submit) >=
                                                diagnose_min_submit &&
                                            diagnose_lines.fetch_add(
                                                1, std::memory_order_relaxed) < 64) {
                                            fprintf(stderr,
                                                    "[readback-consumer] submit=%d target=0x%llx "
                                                    "producer-pass=%zu consumer-draw=%zu "
                                                    "class=%s dim=%u extent=%ux%u target-extent=%ux%u "
                                                    "storage=%u dimension=%u extent-mismatch=%u "
                                                    "feedback=%u\n",
                                                    g_this_submit,
                                                    (unsigned long long)target_base, pass_i - 1,
                                                    later,
                                                    resource.cls == RC::StorageImage
                                                        ? "storage" : "texture",
                                                    resource.img_dim, rw, rh, gw, gh,
                                                    resource.cls == RC::StorageImage ? 1u : 0u,
                                                    resource.img_dim != 1u ? 1u : 0u,
                                                    !sampled_extent_compatible ? 1u : 0u,
                                                    same_pass_target ? 1u : 0u);
                                        }
                                    }
                                }
                            };
                            inspect(items[later].vrt.get());
                            inspect(items[later].prt.get());
                        }
                        return result;
                    };
                    const LaterTargetConsumers consumers0 = inspect_later_consumers(base);
                    const LaterTargetConsumers consumers1 = use_color1
                        ? inspect_later_consumers(base1) : LaterTargetConsumers{};
                    std::array<LaterTargetConsumers, prosper::gpu::kColorTargetCount>
                        consumers_slots{};
                    for (uint32_t slot = 2; slot < mrt_count; ++slot)
                        consumers_slots[slot] = inspect_later_consumers(pass_bases[slot]);
                    const bool sampled_exact_later = consumers0.sampled_exact;
                    const bool feedback_later = consumers0.feedback;
                    const bool cpu_needed_same_batch = consumers0.cpu_needed;
                    static const bool defer_rtt_readback =
                        !PROSPER_ENV_VALUE("PROSPER_NO_RTT_READBACK_DEFER");
                    const bool rtt_defer_ok = defer_rtt_readback
                        ? !cpu_needed_same_batch
                        : (sampled_exact_later && !feedback_later);
                    const bool rtt_defer_ok1 = defer_rtt_readback
                        ? !consumers1.cpu_needed
                        : (consumers1.sampled_exact && !consumers1.feedback);
                    // Keep intermediate scanout spans GPU-resident too: they cannot publish until the
                    // final callback, where the cache is materialized on demand if no later scanout
                    // pass already requested CPU pixels. A same-submit DMA asks its producer span for
                    // authoritative readback, and compute consumers use the lazy target reader above.
                    const bool final_gpu_present = phase.final_span &&
                        !phase.authoritative_readback && prosper::gpu::gpu_present_active();
                    const bool defer_readback = live_gpu_targets && vo_n > 0 && base &&
                        !phase.authoritative_readback &&
                        ((is_vo && can_defer_scanout_readback(
                                       phase.allows_deferred_scanout_readback(),
                                       final_gpu_present,
                                       defer_intermediate_scanout, cpu_needed_same_batch)) ||
                         (!is_vo && (base != front_va || final_gpu_present) && rtt_defer_ok));
                    const bool defer_readback1 = live_gpu_targets && vo_n > 0 && use_color1 &&
                        base1 && base1 != base && !phase.authoritative_readback &&
                        (base1 != front_va || final_gpu_present) && rtt_defer_ok1;
                    std::array<bool, prosper::gpu::kColorTargetCount> defer_readback_slots{};
                    for (uint32_t slot = 2; slot < mrt_count; ++slot) {
                        const LaterTargetConsumers& consumers = consumers_slots[slot];
                        const bool rtt_defer_ok_slot = defer_rtt_readback
                            ? !consumers.cpu_needed
                            : (consumers.sampled_exact && !consumers.feedback);
                        const uint64_t slot_base = pass_bases[slot];
                        defer_readback_slots[slot] = live_gpu_targets && vo_n > 0 &&
                            slot_base && !phase.authoritative_readback &&
                            (slot_base != front_va || final_gpu_present) && rtt_defer_ok_slot;
                    }
                    // PROSPER_READBACK_WHY (#1284): classify WHY each non-deferred pass takes the
                    // synchronous CPU readback (75-79 ms/window on Blue Prince's Day One frame).
                    // The dominant reason selects the next optimization; behavior unchanged.
                    if (!defer_readback) {
                        static const bool why = getenv("PROSPER_READBACK_WHY") != nullptr;
                        if (why) {
                            // Counts identify the failing defer condition; bytes weight each bucket
                            // by the actual copy size, since readback time scales with bytes.
                            enum { kNoLive, kNoVo, kNoBase, kAuth, kVoFinal, kFront,
                                   kSameBatchCpu, kNotSampledExact, kFeedback, kBuckets };
                            static std::atomic<uint64_t> counts[kBuckets], bytes[kBuckets];
                            static std::atomic<uint64_t> same_batch_storage{0};
                            static std::atomic<uint64_t> same_batch_dimension{0};
                            static std::atomic<uint64_t> same_batch_extent{0};
                            static std::atomic<uint64_t> same_batch_feedback{0};
                            static std::atomic<uint64_t> c_total{0};
                            int bucket = kNotSampledExact;
                            if (!live_gpu_targets) bucket = kNoLive;
                            else if (vo_n <= 0) bucket = kNoVo;
                            else if (!base) bucket = kNoBase;
                            else if (phase.authoritative_readback) bucket = kAuth;
                            else if (is_vo) bucket = kVoFinal;
                            else if (base == front_va) bucket = kFront;
                            else if (cpu_needed_same_batch) bucket = kSameBatchCpu;
                            else if (sampled_exact_later && feedback_later) bucket = kFeedback;
                            ++counts[bucket];
                            if (bucket == kSameBatchCpu) {
                                same_batch_storage += consumers0.storage_references;
                                same_batch_dimension += consumers0.dimension_mismatches;
                                same_batch_extent += consumers0.extent_mismatches;
                                same_batch_feedback += consumers0.feedback_references;
                            }
                            // #2398: bill bytes only when colour pixels are ACTUALLY requested.
                            // #2283 narrowed the readback to `want_color_readback` (the same
                            // any_of over pass_bases used at the call below) and this accounting
                            // never followed it -- so a depth-only pass, whose slot bases are all
                            // zero and which therefore lands in the `no_base` bucket BY
                            // CONSTRUCTION, was billed a full frame of colour it never copied.
                            //
                            // That is not a rounding error: on Blue Prince it reported 15.6 GB
                            // against `no_base` and made depth passes look like the frame-rate
                            // wall. The count was right; the column that made it actionable was
                            // measuring a world from before #2283. An instrument that drifts from
                            // the code it measures reads as evidence, which is worse than silence.
                            const bool billed_color = std::any_of(
                                pass_bases.begin(),
                                pass_bases.begin() + std::min<size_t>(mrt_count, pass_bases.size()),
                                [](uint64_t slot_base) { return slot_base != 0; });
                            if (billed_color)
                                bytes[bucket] += static_cast<uint64_t>(gw) * gh *
                                    prosper::test::backend_color_bytes_per_pixel(pass_format);
                            static std::atomic<uint64_t> last_report{0};
                            const uint64_t t = ++c_total;
                            if (t >= last_report.load(std::memory_order_relaxed) + 200) {
                                last_report.store(t, std::memory_order_relaxed);
                                static const char* names[kBuckets] = {
                                    "no_live", "no_vo", "no_base", "auth", "vo_final",
                                    "front", "same_batch_cpu", "not_sampled_exact", "feedback"};
                                fprintf(stderr, "[readback-why] total=%llu",
                                        (unsigned long long)t);
                                for (int i = 0; i < kBuckets; ++i)
                                    fprintf(stderr, " %s=%llu/%lluMB", names[i],
                                            (unsigned long long)counts[i].load(),
                                            (unsigned long long)(bytes[i].load() >> 20));
                                fprintf(stderr,
                                        " same_batch_reasons=storage:%llu,dimension:%llu,"
                                        "extent:%llu,feedback:%llu\n",
                                        (unsigned long long)same_batch_storage.load(),
                                        (unsigned long long)same_batch_dimension.load(),
                                        (unsigned long long)same_batch_extent.load(),
                                        (unsigned long long)same_batch_feedback.load());
                            }
                        }
                    }
                    const uint8_t* seed1 = nullptr;
                    const float* retained_uniform_clear1 = nullptr;
                    bool gpu_seed1_available = false;
                    const bool seed_rtt1 = seed_target(base1);
                    if (seed_rtt1 && use_color1) { auto sit = g_rtt.find(base1);
                        gpu_seed1_available = live_gpu_targets && sit != g_rtt.end() &&
                            sit->second.gpu_valid &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format1 &&
                            prosper::test::find_persistent_color_target(
                                base1, gw, gh, pass_format1) != nullptr;
                        if (!gpu_seed1_available && sit != g_rtt.end() &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format1 && sit->second.rgba &&
                            sit->second.rgba->size() == static_cast<size_t>(gw) * gh *
                                prosper::test::backend_color_bytes_per_pixel(pass_format1))
                            seed1 = sit->second.rgba->data();
                        if (!gpu_seed1_available && !seed1 && sit != g_rtt.end() &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format1 &&
                            sit->second.has_uniform_color)
                            retained_uniform_clear1 = sit->second.uniform_color.data(); }
                    if (base1 && !use_color1 && rtt_log) {
                        const auto* first = render_pass.empty() ? nullptr : render_pass.front();
                        fprintf(stderr,
                                "[rtt] skip MRT1 target=0x%llx extent=%ux%u; MRT0 is %ux%u\n",
                                (unsigned long long)base1,
                                first ? first->color1_width : 0u,
                                first ? first->color1_height : 0u, native_w, native_h);
                    }
                    const auto build_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    prosper::test::BackendColorTarget backend_target{
                        base, seed_rtt0, base != 0 && !defer_readback, pass_format};
                    backend_target.persistent_id1 = use_color1 ? base1 : 0;
                    backend_target.load_existing1 = seed_rtt1;
                    backend_target.readback1 = use_color1 && base1 != 0 && !defer_readback1;
                    backend_target.format1 = pass_format1;
                    // Slots 2..7 retain across render groups on the same terms as slots 0 and 1.
                    // A G-buffer built by several groups against one set of allocations otherwise
                    // loses every group's work but the last, because slots above 1 were transient
                    // images cleared per backend call.
                    for (uint32_t slot = 2; slot < mrt_count; ++slot) {
                        backend_target.persistent_id_slots[slot] = pass_bases[slot];
                        backend_target.load_existing_slots[slot] = seed_target(pass_bases[slot]);
                        backend_target.readback_slots[slot] = pass_bases[slot] != 0 && !defer_readback_slots[slot];
                    }
                    auto backend_draws = build_bds(render_pass);
                    const auto build_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    prosper::test::BackendMrtOutputs mrt_outputs;
                    mrt_outputs.color_count = mrt_count;
                    // The colour target is passed whenever ANY slot is bound, not colour-0 alone.
                    // The same union the readback flag below already uses, and for the same reason
                    // the comment there gives: a pass with colour-0 unbound and a higher slot bound
                    // is reachable by construction. On `base != 0` alone such a pass populated
                    // persistent_id_slots[2..7] and then handed the backend a nullptr, so those
                    // active slots stayed transient and were cleared by the next group -- exactly
                    // the defect the persistence contract exists to remove, reintroduced at the one
                    // call site that decides whether the contract is used at all.
                    const bool any_slot_bound = prosper::frontend::mrt_any_slot_bound(
                        pass_bases.data(),
                        static_cast<uint32_t>(std::min<size_t>(mrt_count, pass_bases.size())));
                    std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(
                        backend_draws, gw, gh, seed,
                        retained_uniform_clear ? retained_uniform_clear : clear_for(render_pass), true,
                        live_gpu_targets && any_slot_bound ? &backend_target : nullptr,
                        seed1, retained_uniform_clear1 ? retained_uniform_clear1
                            : (use_color1 ? render_pass.front()->ps.clear_color1 : nullptr),
                        nullptr,
                        batch_backend_submits ? &backend_submission : nullptr,
                        pass_i == items.size(), &mrt_outputs,
                        // #2283: only ask for colour pixels when something will read them.
                        //
                        // Keyed on the UNION of every bound slot, not on colour-0 alone. Review
                        // caught that: `gpx`'s only consumer is indeed inside an `if (base ...)`
                        // pair with no else, but `gpx` is not all the call returns. `mrt_outputs`
                        // is filled by the same call and its consumer is keyed per slot on
                        // `pass_bases[slot]`, where an EMPTY vector is not a no-op -- it calls
                        // `surface.rgba.reset()` and drops that surface's cached pixels.
                        //
                        // `mrt_count` does not depend on `base`, so a pass with colour-0 unbound
                        // and colour-1 bound is reachable by construction -- the same sparse export
                        // hole the MRT loop already acknowledges, at slot 0 instead of slot 1. On
                        // `base != 0` alone such a pass would silently lose a live colour-1 RTT.
                        // Structural rather than observed, which is exactly why the 457/457
                        // depth-only measurement could not clear it: that evidence is all slot 0.
                        //
                        // Depth-only passes have every slot zero, so the win is unchanged.
                        // Keyed on the base rather than the draws' colour write masks for the
                        // separate reason that 457/457 is evidence about THIS route, and a future
                        // title could legally mix a colour-writing draw into a pass that has a base.
                        /*want_color_readback=*/any_slot_bound);
                    const auto backend_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    const prosper::test::BackendColorTargetStats color_target_call =
                        prosper::test::backend_color_target_stats();
                    const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
                        ? prosper::test::backend_render_timing_stats()
                        : prosper::test::BackendRenderTimingStats{};
                    const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
                        ? prosper::test::backend_pipeline_cache_stats()
                        : prosper::test::BackendPipelineCacheStats{};
                    // Captured HERE, beside the pipeline stats, and PASSED IN -- not read inside
                    // record_backend_timing. These are thread_local and published by the backend call;
                    // reading them from wherever the recorder happens to run returns a different,
                    // nearly empty instance. The first draft did exactly that and reported refs=11 for
                    // a 2,102-draw submit -- the kind of small plausible number that gets believed.
                    const prosper::test::BackendResourceReuseStats backend_reuse_stats = timing_enabled
                        ? prosper::test::backend_resource_reuse_stats()
                        : prosper::test::BackendResourceReuseStats{};
                    if (timing_enabled) {
                        pending_timing.build_resources_ms +=
                            std::chrono::duration<double, std::milli>(build_done - build_start).count();
                        pending_timing.backend_ms +=
                            std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                        // Everything this group did BEFORE the measured pair -- target grouping,
                        // format/extent resolution, resolve handling, seed selection. Recorded here
                        // rather than at the loop head so a `continue` between the two points cannot
                        // skip it and silently under-report (#2250's lesson, applied one level up).
                        pending_timing.pass_pre_ms +=
                            std::chrono::duration<double, std::milli>(build_start - group_start).count();
                        ++pending_timing.pass_groups;
                        record_backend_timing(backend_call_timing, backend_pipeline_stats,
                                          backend_reuse_stats);
                        pending_timing.color_target_writes += color_target_call.writes;
                        pending_timing.color_target_write_hits += color_target_call.write_hits;
                        pending_timing.color_target_sample_hits += color_target_call.sampled_hits;
                        pending_timing.color_target_readbacks += color_target_call.readbacks;
                        pending_timing.color_target_cached_bytes = color_target_call.cached_bytes;
                        pending_timing.color_target_cached_entries = color_target_call.cached_entries;
                    }
                    const auto post_stats_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    static const std::string dump_spec =
                        PROSPER_ENV_VALUE("PROSPER_DUMP_PASS")
                            ? PROSPER_ENV_VALUE("PROSPER_DUMP_PASS") : "";
                    static const int dump_pass_every =
                        PROSPER_ENV_VALUE("PROSPER_DUMP_PASS_EVERY")
                            ? std::max(1, atoi(getenv("PROSPER_DUMP_PASS_EVERY"))) : 60;
                    auto pass_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(gpx));
                    if (base && color_target_call.writes) {
                        RttSurf& surface = g_rtt[base];
                        surface.w = gw;
                        surface.h = gh;
                        surface.format = pass_format;
                        surface.guest_format = pass.empty()
                            ? pass_format
                            : static_cast<VkFormat>(
                                  prosper::frontend::mrt_raw_format(*pass.front(), 0));
                        surface.has_uniform_color = false;
                        surface.dcc_metadata_dirty = false;
                        surface.gpu_valid = prosper::test::find_persistent_color_target(
                            base, gw, gh, pass_format) != nullptr;
                        if (!pass_pixels->empty()) surface.rgba = pass_pixels;
                        else surface.rgba.reset();
                        // GTA V builds its packed-HDR bloom pyramid as separate CB_COLOR targets,
                        // then samples them as one mipmapped T#. A target may be hundreds of LRU
                        // insertions old by that final draw. Pin every produced level now, while its
                        // exact identity is known, and release the small set at the submit boundary.
                        pin_renderer_mip_target(base, gw, gh, pass_format, surface.gpu_valid);
                        if (defer_readback && is_vo && surface.gpu_valid) {
                            const auto already_pinned = std::find_if(
                                pinned_scanouts.begin(), pinned_scanouts.end(),
                                [&](const PinnedScanout& target) {
                                    return target.id == base && target.width == gw &&
                                           target.height == gh && target.format == pass_format;
                                });
                            if (already_pinned == pinned_scanouts.end()) {
                                if (prosper::test::pin_persistent_color_target(
                                        base, gw, gh, pass_format)) {
                                    pinned_scanouts.push_back({base, gw, gh, pass_format});
                                } else {
                                    // Pinning is expected to succeed for the target just written. If
                                    // cache state is inconsistent, preserve correctness by immediately
                                    // restoring the authoritative CPU fallback.
                                    std::vector<uint8_t> materialized;
                                    std::string error;
                                    if (prosper::test::readback_persistent_color_target(
                                            base, gw, gh, pass_format, materialized, error))
                                        surface.rgba =
                                            std::make_shared<const std::vector<uint8_t>>(
                                                std::move(materialized));
                                }
                            }
                        }
                    } else if (base && !pass_pixels->empty()) {
                        RttSurf& surface = g_rtt[base];
                        surface.rgba = pass_pixels;
                        surface.w = gw;
                        surface.h = gh;
                        surface.format = pass_format;
                        surface.guest_format = pass.empty()
                            ? pass_format
                            : static_cast<VkFormat>(
                                  prosper::frontend::mrt_raw_format(*pass.front(), 0));
                        surface.gpu_valid = false;
                        surface.has_uniform_color = false;
                        surface.dcc_metadata_dirty = false;
                    }
                    const auto post_slot0_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    for (uint32_t slot = 1; slot < mrt_count; ++slot) {
                        auto& pixels = mrt_outputs.colors[slot];
                        if (!pass_bases[slot]) continue;  // transient attachment for a sparse export hole
                        if (rtt_log && !pixels.empty()) {
                            size_t nz = 0, rgb_nz = 0;
                            for (uint8_t byte : pixels) nz += byte != 0;
                            if (pass_formats[slot] == VK_FORMAT_R8G8B8A8_UNORM)
                                for (size_t p = 0; p + 3 < pixels.size(); p += 4)
                                    rgb_nz += pixels[p] != 0 || pixels[p + 1] != 0 ||
                                              pixels[p + 2] != 0;
                            fprintf(stderr,
                                    "[rtt] pass c%u=0x%llx extent=%ux%u (%zu draws) "
                                    "px_nonzero=%zu rgb_nonblack=%zu\n",
                                    slot, (unsigned long long)pass_bases[slot],
                                    gw, gh, pass.size(), nz, rgb_nz);
                        }
                        // PROSPER_DUMP_PASS covers slots 1..7 too. The first version dumped only
                        // `base`, so the busiest target in GTA V's frame -- 0x2085de0000, written by
                        // 130,290 of 131,072 draws, 96% of them in SLOT 4 -- could not be dumped at
                        // all, and its absence read as "that pass does not run".
                        if (!dump_spec.empty() && pass_bases[slot]) {
                            char needle[24];
                            std::snprintf(needle, sizeof needle, "0x%llx",
                                          (unsigned long long)pass_bases[slot]);
                            if (dump_spec.find(needle) != std::string::npos) {
                                static std::map<uint64_t, size_t> slot_busiest;
                                static std::map<uint64_t, int> slot_counted;
                                const bool slot_peak = pass.size() > slot_busiest[pass_bases[slot]];
                                if (slot_peak) slot_busiest[pass_bases[slot]] = pass.size();
                                if (slot_peak &&
                                    slot_counted[pass_bases[slot]]++ % dump_pass_every == 0) {
                                    const std::vector<uint8_t> shot = inspection_rgba8(
                                        pixels, gw, gh, pass_formats[slot]);
                                    const char* dir = getenv("PROSPER_FRAME_DIR");
                                    char path[512];
                                    std::snprintf(path, sizeof path, "%s/pass_%llx_s%u_%ux%u.bmp",
                                                  dir ? dir : ".",
                                                  (unsigned long long)pass_bases[slot], slot,
                                                  gw, gh);
                                    const size_t want = size_t(gw) * gh * 4;
                                    const bool ok = shot.size() == want &&
                                        prosper::test::dump_bmp(path, shot, gw, gh);
                                    fprintf(stderr,
                                            "[dump-pass] target=0x%llx slot=%u %ux%u src=%zuB -> %s\n",
                                            (unsigned long long)pass_bases[slot], slot, gw, gh,
                                            pixels.size(),
                                            ok ? path : "NO CPU PIXELS (readback deferred)");
                                }
                            }
                        }
                        RttSurf& surface = g_rtt[pass_bases[slot]];
                        surface.w = gw;
                        surface.h = gh;
                        surface.format = pass_formats[slot];
                        surface.guest_format = pass.empty()
                            ? pass_formats[slot]
                            : static_cast<VkFormat>(
                                  prosper::frontend::mrt_raw_format(*pass.front(), slot));
                        surface.has_uniform_color = false;
                        surface.dcc_metadata_dirty = false;
                        // Any slot with a retained target is GPU-valid, not only slot 1. The
                        // `slot == 1` clause dated from when slots above 1 had no persistent image
                        // to be valid about.
                        surface.gpu_valid =
                            prosper::test::find_persistent_color_target(
                                pass_bases[slot], gw, gh, pass_formats[slot]) != nullptr;
                        pin_renderer_mip_target(pass_bases[slot], gw, gh, pass_formats[slot],
                                                surface.gpu_valid);
                        if (!pixels.empty())
                            surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                                std::move(pixels));
                        else
                            surface.rgba.reset();
                    }
                    const auto post_mrt_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    const std::vector<uint8_t>& rendered_pixels = *pass_pixels;
                    // Both dims parse to 0 when PROSPER_RESOURCE_HASH_DIM is UNSET, so an extent
                    // comparison alone made every 0x0 pass satisfy the filter and switch the
                    // diagnostic on by itself. Require it to have actually been requested.
                    if (resource_hash_w && resource_hash_h &&
                        native_w == resource_hash_w && native_h == resource_hash_h &&
                        !rendered_pixels.empty()) {
                        uint64_t hash = 1469598103934665603ull;
                        for (uint8_t byte : rendered_pixels) {
                            hash ^= byte;
                            hash *= 1099511628211ull;
                        }
                        const auto* first = render_pass.empty() ? nullptr : render_pass.front();
                        const auto* last = render_pass.empty() ? nullptr : render_pass.back();
                        fprintf(stderr,
                                "[target-version] render-submit=%llu target=0x%llx dims=%ux%u "
                                "draws=%llu-%llu orders=%llu-%llu seed=%d clear=%d "
                                "rgba=%.3f,%.3f,%.3f,%.3f hash=%016llx\n",
                                (unsigned long long)g_this_submit,
                                (unsigned long long)base, native_w, native_h,
                                (unsigned long long)(first ? first->draw_index : 0),
                                (unsigned long long)(last ? last->draw_index : 0),
                                (unsigned long long)(first ? first->command_order : 0),
                                (unsigned long long)(last ? last->command_order : 0),
                                seed != nullptr, first ? (int)first->ps.has_clear_color : 0,
                                first ? first->ps.clear_color[0] : 0.0f,
                                first ? first->ps.clear_color[1] : 0.0f,
                                first ? first->ps.clear_color[2] : 0.0f,
                                first ? first->ps.clear_color[3] : 0.0f,
                                (unsigned long long)hash);
                    }
                    // Same unset-is-0 trap as the resource hash above, and far more expensive here:
                    // the loop below re-renders every growing draw prefix, so a 0x0 pass silently
                    // turned one pass into O(draws^2) rendering plus a readback and pixel census per
                    // step. Require PROSPER_TARGET_STEP_HASH_DIM to have actually been requested.
                    if (target_step_w && target_step_h &&
                        native_w == target_step_w && native_h == target_step_h &&
                        render_pass.size() >= target_step_min_draws) {
                        for (size_t k = 1; k <= render_pass.size(); ++k) {
                            std::vector<const prosper::gpu::DrawItem*> prefix(
                                render_pass.begin(), render_pass.begin() + k);
                            // PROSPER_TARGET_STEP_HASH_DIM re-renders growing draw prefixes through
                            // build_bds(), so every prefix must resolve its own resources from
                            // scratch. Release the scratch pins with the map they belonged to,
                            // otherwise each prefix would strand another set of slots.
                            texstore_used = 0;
                            texstore_pinned.assign(texstore.size(), false);
                            decoded_textures.clear();
                            std::vector<uint8_t> step = prosper::test::render_draws_rgba(
                                build_bds(prefix), gw, gh, seed, clear_for(prefix), true);
                            uint64_t hash = 1469598103934665603ull;
                            for (uint8_t byte : step) {
                                hash ^= byte;
                                hash *= 1099511628211ull;
                            }
                            size_t dark = 0, near_white = 0;
                            uint64_t rgb_sum = 0;
                            const VkFormat step_format = prosper::test::backend_color_format(
                                static_cast<VkFormat>(prefix.front()->ps.color0_format));
                            const std::vector<uint8_t> step_rgba = inspection_rgba8(
                                step, gw, gh, step_format);
                            for (size_t p = 0; p + 3 < step_rgba.size(); p += 4) {
                                const uint8_t r = step_rgba[p], g = step_rgba[p + 1], b = step_rgba[p + 2];
                                dark += std::max({r, g, b}) < 64;
                                near_white += std::min({r, g, b}) > 240;
                                rgb_sum += r + g + b;
                            }
                            const double mean_rgb = step_rgba.empty() ? 0.0 :
                                (double)rgb_sum / ((step_rgba.size() / 4) * 3);
                            const auto* draw = prefix.back();
                            auto spirv_hash = [](const std::vector<uint32_t>& words) {
                                uint64_t hash = 1469598103934665603ull;
                                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words.data());
                                for (size_t i = 0; i < words.size() * sizeof(uint32_t); ++i) {
                                    hash ^= bytes[i];
                                    hash *= 1099511628211ull;
                                }
                                return hash;
                            };
                            fprintf(stderr,
                                    "[target-step] render-submit=%llu target=0x%llx step=%zu/%zu "
                                    "draw=%llu order=%llu vs=%016llx ps=%016llx mask=0x%x "
                                    "blend=%d/%u/%u hash=%016llx dark=%zu white=%zu mean=%.2f\n",
                                    (unsigned long long)g_this_submit,
                                    (unsigned long long)base, k, render_pass.size(),
                                    (unsigned long long)draw->draw_index,
                                    (unsigned long long)draw->command_order,
                                    (unsigned long long)spirv_hash(draw->vs_words()),
                                    (unsigned long long)spirv_hash(draw->fs_words()),
                                    draw->ps.color_write_mask, (int)draw->ps.blend_enable,
                                    draw->ps.src_color_blend_factor,
                                    draw->ps.dst_color_blend_factor,
                                    (unsigned long long)hash, dark, near_white, mean_rgb);
                        }
                    }
                    const RttTimingRecord rtt_timing_record{
                        g_this_submit, base, gw, gh, pass.size(),
                        phase.first_span, phase.final_span, phase.authoritative_readback,
                        defer_readback,
                        std::chrono::duration<double, std::milli>(
                            backend_done - build_done).count(),
                        backend_call_timing, color_target_call};
                    if (lightweight_rtt_timing) pending_rtt_timing.push_back(rtt_timing_record);
                    else if (timing_enabled && rtt_log) print_rtt_timing(rtt_timing_record);
                    // PROSPER_DUMP_PASS=0xADDR[,0xADDR…] — write what a pass actually produced, as an
                    // image, for the named targets. `rgb_nonblack` alone cannot answer "is the world
                    // there": it is computed from an RGBA8 conversion, so an HDR f16 target whose
                    // values are all below 1/255 reports EXACTLY ZERO while carrying a complete
                    // scene. That is not hypothetical here — GTA V's 0x20431c0000 reports
                    // rgb_nonblack=0 while the very next pass extracts 47.5% non-black from it, so
                    // reading the metric as "black" is a false negative on the whole lighting stage.
                    //
                    // Overwrites one file per target so the last write is the latest frame, and
                    // dumps every PROSPER_DUMP_PASS_EVERY-th occurrence (default 60) because a 4K
                    // BMP is 24 MB and a routed boot renders each of these a few hundred times.
                    {
                        if (!dump_spec.empty() && base) {
                            char needle[24];
                            std::snprintf(needle, sizeof needle, "0x%llx",
                                          (unsigned long long)base);
                            if (dump_spec.find(needle) != std::string::npos) {
                                // Keep the occurrence with the MOST DRAWS, not the last. A
                                // G-buffer target is written by one heavy pass and then touched by
                                // several small ones, so "last write wins" reliably captures a
                                // near-empty tail and reads as "this channel is empty" -- which it
                                // did, for three channels that a draw census showed at 54%.
                                static std::map<uint64_t, size_t> busiest;
                                static std::map<uint64_t, int> counted;
                                const bool new_peak = pass.size() > busiest[base];
                                if (new_peak) busiest[base] = pass.size();
                                if (new_peak && counted[base]++ % dump_pass_every == 0) {
                                    const std::vector<uint8_t> shot = inspection_rgba8(
                                        rendered_pixels, gw, gh, pass_format);
                                    const char* dir = getenv("PROSPER_FRAME_DIR");
                                    char path[512];
                                    std::snprintf(path, sizeof path, "%s/pass_%llx_%ux%u.bmp",
                                                  dir ? dir : ".", (unsigned long long)base,
                                                  gw, gh);
                                    // Report the source size, not just the intent. A pass whose
                                    // readback was deferred has EMPTY cpu pixels, and both this dump
                                    // and `rgb_nonblack` are then reporting on nothing — which reads
                                    // as "the target is black" rather than "we never looked".
                                    const size_t want = size_t(gw) * gh * 4;
                                    const bool ok = shot.size() == want &&
                                        prosper::test::dump_bmp(path, shot, gw, gh);
                                    fprintf(stderr,
                                            "[dump-pass] target=0x%llx %ux%u src=%zuB rgba=%zuB/%zuB"
                                            " -> %s\n",
                                            (unsigned long long)base, gw, gh,
                                            rendered_pixels.size(), shot.size(), want,
                                            ok ? path : "NO CPU PIXELS (readback deferred)");
                                }
                            }
                        }
                    }
                    if (rtt_log) {
                        const std::vector<uint8_t> inspected = inspection_rgba8(
                            rendered_pixels, gw, gh, pass_format);
                        size_t nz = 0, rgb_nz = 0;
                        for (uint8_t b : rendered_pixels) nz += (b != 0);
                        for (size_t p = 0; p + 3 < inspected.size(); p += 4)
                            rgb_nz += (inspected[p] != 0 || inspected[p + 1] != 0 ||
                                       inspected[p + 2] != 0);
                        // `src=` is load-bearing, not decoration. Both counters are computed from
                        // `rendered_pixels`, so a pass whose readback was deferred reports
                        // `px_nonzero=0 rgb_nonblack=0` -- identical to a genuinely black target, and
                        // indistinguishable from it without this field. The `[dump-pass]` line four
                        // lines up already guards exactly this ("as 'the target is black' rather than
                        // 'we never looked'"); this line did not, and readbacks are deferred routinely
                        // -- `[readback-why]` buckets nine distinct reasons.
                        //
                        // It matters most for the conclusion it silently supports. Every "this pass
                        // reads populated inputs and emits nothing" reading on GTA V rests on these
                        // two counters, and "we never looked" produces that reading for free.
                        fprintf(stderr, "[rtt] pass target=0x%llx extent=%ux%u native=%ux%u (%zu draws) "
                                "src=%zuB px_nonzero=%zu rgb_nonblack=%zu cache_size=%zu%s%s\n",
                                (unsigned long long)base, gw, gh, native_w, native_h, pass.size(),
                                rendered_pixels.size(), nz, rgb_nz, g_rtt.size(),
                                is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : "");
                    }
                    // PROSPER_DUMP_DRAWSTEPS: for a pass targeting a SCANOUT buffer, re-render the pass
                    // draw-by-draw (prefix 1, prefix 2, ...) and dump each cumulative result — a one-boot
                    // bisect for "which draw of the final composite blacks the screen" (#319). Diagnostic.
                    if (PROSPER_ENV_ON("PROSPER_DUMP_DRAWSTEPS") && is_vo && pass.size() > 1) {
                        for (size_t k = 1; k <= pass.size(); k++) {
                            std::vector<const prosper::gpu::DrawItem*> prefix(render_pass.begin(), render_pass.begin() + k);
                            std::vector<uint8_t> spx = prosper::test::render_draws_rgba(
                                build_bds(prefix), gw, gh, seed, clear_for(prefix));
                            if (spx.empty()) continue;
                            size_t snz = 0; for (uint8_t b : spx) snz += (b != 0);
                            fprintf(stderr, "[rtt] drawstep %zu/%zu tgt=0x%llx px_nonzero=%zu\n",
                                    k, pass.size(), (unsigned long long)base, snz);
                            const char* dd = getenv("PROSPER_FRAME_DIR");
                            char fn[512]; snprintf(fn, sizeof fn, "%s/drawstep_%04d_%zu.bmp",
                                                   dd ? dd : ".", frame_no.load(), k);
                            prosper::test::dump_bmp(fn, spx, gw, gh);
                        }
                    }
                    // Per-target group dumps into PROSPER_FRAME_DIR. Two INDEPENDENT opt-ins (either may
                    // be set alone); PROSPER_DUMP_RTGROUPS_ADDR optionally limits a long replay to one
                    // target VA. Diagnostic; no default behavior change.
                    //  - PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes>: a 24-bit BMP (alpha dropped) of the
                    //    format-inspected pixels — to eyeball an intermediate pass (e.g. a UI/banner RT).
                    //  - PROSPER_DUMP_RTGROUPS_RGBA: the RAW RGBA8 backend bytes (alpha PRESERVED, no
                    //    format inspection) — needed to reason about premultiplied-alpha UI compositing
                    //    that samples an RT's alpha as a blend factor. Non-RGBA8 targets are skipped
                    //    visibly rather than writing native bytes under a misleading .rgba contract.
                    if ((PROSPER_ENV_ON("PROSPER_DUMP_RTGROUPS") || (getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) &&
                        !rendered_pixels.empty()) {
                        size_t nz = 0; for (uint8_t b : rendered_pixels) nz += (b != 0);
                        const char* address_filter = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS_ADDR");
                        const uint64_t wanted_base = address_filter && *address_filter
                            ? strtoull(address_filter, nullptr, 0) : 0;
                        // Match the address against EVERY MRT slot, not just slot 0.
                        //
                        // `base` is the pass's slot-0 attachment. A G-buffer names different
                        // surfaces per slot, so filtering by a slot-2 address matched ONE pass in a
                        // full routed run for a target the draw census credits with 6,672 draws --
                        // and a near-empty result reads as "that target is empty" rather than as
                        // "the filter asked the wrong question". The dumped pixels are still slot
                        // 0's; what this fixes is WHICH PASSES are selected, so naming any slot of a
                        // G-buffer selects that G-buffer's passes.
                        bool slot_match = !wanted_base || base == wanted_base;
                        uint32_t matched_slot = 0;
                        if (wanted_base && !slot_match)
                            for (uint32_t slot = 1; slot < pass_bases.size(); ++slot)
                                if (pass_bases[slot] == wanted_base) {
                                    slot_match = true; matched_slot = slot; break;
                                }
                        const char* dd = getenv("PROSPER_FRAME_DIR");
                        // Identify the pass by its first..last draw index so multiple passes to the same
                        // target VA in one frame do not silently overwrite each other.
                        const uint64_t pass_d0 = render_pass.empty() ? 0u : render_pass.front()->draw_index;
                        const uint64_t pass_d1 = render_pass.empty() ? 0u : render_pass.back()->draw_index;
                        if (slot_match) {
                            if (wanted_base && matched_slot) {
                                static std::set<std::pair<uint64_t, uint32_t>> slot_seen;
                                if (slot_seen.emplace(wanted_base, matched_slot).second)
                                    fprintf(stderr,
                                            "[rtt] rtgroup filter 0x%llx matched slot %u of a pass "
                                            "whose slot-0 base is 0x%llx; dumped pixels are SLOT 0\n",
                                            (unsigned long long)wanted_base, matched_slot,
                                            (unsigned long long)base);
                            }
                            if (const char* rg = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS"); rg && nz >= (size_t)atol(rg)) {
                                const std::vector<uint8_t> inspected = inspection_rgba8(
                                    rendered_pixels, gw, gh, pass_format);
                                char fn[512]; snprintf(fn, sizeof fn, "%s/rtgrp_%llx_%04d.bmp",
                                                       dd ? dd : ".", (unsigned long long)base, frame_no.load());
                                prosper::test::dump_bmp(fn, inspected, gw, gh);
                            }
                            // Independent of the BMP variable AND its nonzero threshold, so a fully
                            // transparent (all-zero) group is captured too. Self-describing filename
                            // (extent + draw range) and failure-visible: a missing file must not be
                            // mistaken for a transparent/empty result.
                            if ((getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) {
                                const uint64_t expected_bytes_u64 = static_cast<uint64_t>(gw) * gh * 4u;
                                const bool rgba8_format = pass_format == VK_FORMAT_R8G8B8A8_UNORM;
                                const bool size_valid = expected_bytes_u64 <= SIZE_MAX &&
                                    rendered_pixels.size() == static_cast<size_t>(expected_bytes_u64);
                                if (!rgba8_format || !size_valid) {
                                    fprintf(stderr,
                                            "[rtt] rgba-dump skipped target=0x%llx %ux%u draws=%llu..%llu "
                                            "reason=%s format=%d expected=%llu actual=%zu\n",
                                            (unsigned long long)base, gw, gh,
                                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                                            rgba8_format ? "size-mismatch" : "unsupported-format",
                                            static_cast<int>(pass_format),
                                            (unsigned long long)expected_bytes_u64,
                                            rendered_pixels.size());
                                } else {
                                    char rn[600]; snprintf(rn, sizeof rn, "%s/rtgrp_%llx_%ux%u_f%04d_d%04llu-%04llu.rgba",
                                                           dd ? dd : ".", (unsigned long long)base, gw, gh,
                                                           frame_no.load(), (unsigned long long)pass_d0,
                                                           (unsigned long long)pass_d1);
                                    FILE* rf = fopen(rn, "wb");
                                    bool ok = rf != nullptr; size_t wrote = 0;
                                    if (rf) {
                                        wrote = fwrite(rendered_pixels.data(), 1, rendered_pixels.size(), rf);
                                        ok = (wrote == rendered_pixels.size());
                                        if (fclose(rf) != 0) ok = false;
                                    }
                                    fprintf(stderr, "[rtt] rgba-dump %s target=0x%llx %ux%u draws=%llu..%llu "
                                            "bytes=%zu path=%s\n", ok ? "ok" : "FAILED",
                                            (unsigned long long)base, gw, gh,
                                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                                            wrote, rn);
                                }
                            }
                        }
                    }
                    // The POPULATION that the publish zero is a zero OVER, counted BEFORE the
                    // emptiness/format gates below -- so it sees every pass whose colour target
                    // is disabled, not just those that already qualified as candidates.
                    //
                    // This placement is the whole point. A same-source positive control proves
                    // the DISCRIMINATOR fires; it can never prove the DOMAIN contains the case.
                    // Counting the population in the same run is what separates "disabled-target
                    // passes exist and none is ever published" from "none exists, so the zero
                    // says nothing" -- and those are opposite conclusions about whether #2283's
                    // readback skip is safe.
                    if (timing_enabled && !pass.empty() && pass.front()->ps.color0_format == 0)
                        ++pending_timing.publish_candidate_fmt0;
                    if (!rendered_pixels.empty() && pass_format == VK_FORMAT_R8G8B8A8_UNORM) {
                        // Recorded alongside each candidate rather than re-derived at selection
                        // time: by then the pass loop has moved on and `pass` no longer refers to
                        // the pass that produced these pixels (#2283).
                        const uint32_t candidate_fmt =
                            pass.empty() ? kNoPassFormat : pass.front()->ps.color0_format;
                        if (base && base == front_va) {                          // the flipped buffer
                            px_front = pass_pixels; px_front_w = gw; px_front_h = gh;
                            px_front_base = base; px_front_fmt = candidate_fmt;
                        }
                        if (is_vo) {                                            // any registered scanout
                            px_vo = pass_pixels; px_vo_w = gw; px_vo_h = gh;
                            px_vo_base = base; px_vo_fmt = candidate_fmt;
                        }
                        px_last = pass_pixels;                                  // last non-empty (fallback)
                        px_last_w = gw; px_last_h = gh; px_last_base = base; px_last_fmt = candidate_fmt;
                    } else if (!rendered_pixels.empty() && prefix_inspect_publish()) {
                        // #1330: under gpu_replay's ordered-prefix inspection (--draw/--draw-steps/
                        // --through-operation set PROSPER_PREFIX_INSPECT), a prefix ending on a
                        // non-RGBA8 pass (an FP16 HDR scene target) must return THAT surface, not
                        // whatever stale RGBA8 pass ran earlier. Publish an inspection-converted copy
                        // as the weakest fallback: any RGBA8 pass afterwards still overwrites it, and
                        // with the env unset (every live/normal replay run) behavior is byte-identical.
                        std::vector<uint8_t> converted =
                            inspection_rgba8(rendered_pixels, gw, gh, pass_format);
                        if (!converted.empty())
                            px_last = std::make_shared<const std::vector<uint8_t>>(std::move(converted));
                    }
                    // PROSPER_PASS_LOG=<min-submit>|ms:<millis>: per-pass publish provenance for 3
                    // submits — which pass produced pixels, its target identity, and the defer
                    // decision. The `ms:` form aims the window by wall time (diagnostic_window.hpp).
                    // PROSPER_PASS_LOG_NODEFER opens this block too. It was added INSIDE it and
                    // armed by its own switch, so setting only PROSPER_PASS_LOG_NODEFER produced
                    // ZERO lines -- a silent run indistinguishable from "there are no such passes".
                    // That is #2149's class, committed by the author who had been citing #2149 all
                    // session: a diagnostic whose producer and printer are armed by different
                    // switches reports unarmed state as a confident zero.
                    if (PROSPER_ENV_ON("PROSPER_PASS_LOG") ||
                        PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER")) {
                        const uint64_t at = g_pass_log_submit.load(std::memory_order_relaxed);
                        const bool in_window =
                            g_pass_log_window.contains(at, diagnostic_elapsed_ms());
                        // px_nonblack must be counted in the PASS's OWN format. The original loop
                        // stepped 4 bytes and tested bytes `p`, `p+1`, `p+2` — never `p+3` — as if
                        // every target were 8-bit RGBA. Over an 8-byte FP16 texel that reads
                        // {R_lo, R_hi, G_lo} and then {B_lo, B_hi, A_lo}, so whether a texel counts
                        // depends only on where a half-float happens to put its non-zero bytes.
                        // Measured on Sonic Origins' 3840x2160 R16G16B16A16_SFLOAT scene target, whose
                        // texels are RGB bit-zero with alpha 0x3c05 (1.00488): byte 6 is that alpha's
                        // low mantissa byte 0x05, which lands on the second group's `p+2`, so exactly
                        // one of the two groups per texel counted and the line reported 8,294,400 —
                        // precisely w*h, reading as "every pixel has content" for a frame with no
                        // colour in it at all. An alpha of exactly 1.0 would have counted ZERO:
                        // 0x3c00's only non-zero byte is the `p+3` this loop skipped. The number was
                        // never a property of the image, only of its bit layout — and it survived long
                        // enough to become a hypothesis (#1905). Count through the same inspection
                        // conversion the persistent dump uses; -1 says the format has no conversion,
                        // which is honest where a wrong number is not.
                        long long nz = 0;
                        const bool direct_rgba8 = pass_format == VK_FORMAT_R8G8B8A8_UNORM &&
                            rendered_pixels.size() == static_cast<size_t>(gw) * gh * 4u;
                        const std::vector<uint8_t> inspected = direct_rgba8
                            ? std::vector<uint8_t>{}
                            : inspection_rgba8(rendered_pixels, gw, gh, pass_format);
                        const std::vector<uint8_t>& counted =
                            direct_rgba8 ? rendered_pixels : inspected;
                        if (!rendered_pixels.empty() && counted.empty()) nz = -1;
                        else
                            for (size_t p = 0; p + 3 < counted.size(); p += 4)
                                if (counted[p] || counted[p + 1] || counted[p + 2]) nz++;
                        // In-window: every pass. Out of window: only content-bearing (or deferred)
                        // passes, so the publish source can be found without knowing the callback —
                        // plus `nz < 0`, a pass whose format the inspection path cannot convert. That
                        // last case has to stay visible out of window: it is the one where this line
                        // cannot say whether the pass carries content, which is exactly when the
                        // reader needs to know the pass existed.
                        // PROSPER_PASS_LOG_NODEFER: print the NON-DEFERRED passes specifically.
                        // They are the population #2276 needs and the only one this line cannot
                        // currently reach: a deferred pass always prints, a content-bearing pass
                        // prints on `nz > 100`, and the window covers three callbacks -- but a
                        // non-deferred pass with little content satisfies none of those, so 2,999
                        // of them a route were invisible while a classifier counted 21 GB of
                        // readback against them.
                        //
                        // Opt-in and separate from PROSPER_PASS_LOG rather than widening the window,
                        // because the window's three-callback span is deliberate (its comment says
                        // why) and this needs the whole run, not a wider sample. ~9.7 lines a submit.
                        static const bool pass_log_nodefer =
                            PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER");
                        if (in_window || nz > 100 || nz < 0 || defer_readback ||
                            (pass_log_nodefer && !defer_readback))
                            fprintf(stderr,
                                    // `bytes` disambiguates px_nonblack=0, which today means EITHER
                                    // "the readback returned a near-black surface" OR "the readback
                                    // returned nothing and the counting loop never ran". Those are
                                    // opposite facts wearing the same number -- #2255's absent-vs-zero
                                    // collapse, and instrument trap 116's -- and on #2276 it is
                                    // precisely the ambiguity blocking the diagnosis: 2,999 passes a
                                    // route take a ~7.1 MB readback and report px_nonblack under 100,
                                    // and nobody can say from this line whether those surfaces are
                                    // black or absent.
                                    "[pass] cb=%llu pass=%zu/%zu base=0x%llx %ux%u fmt=%d vo=%d "
                                    // raw_fmt is CB_COLOR0_INFO.FORMAT BEFORE backend_color_format
                                    // touches it, and it is the field that decides #2283. That
                                    // converter falls back to R8G8B8A8_UNORM for anything it does
                                    // not recognise -- INCLUDING 0 -- and R8G8B8A8_UNORM is itself
                                    // enumerator 37, so the most common real format and "unknown"
                                    // print identically. `fmt` above therefore cannot distinguish a
                                    // live colour target from a disabled one, and I nearly published
                                    // a correctness claim on it.
                                    //
                                    // raw_fmt=0 is CB_COLOR_INVALID: the target is DISABLED, the
                                    // extent in CB_COLOR0_ATTRIB2 is stale from an earlier pass
                                    // (context registers are sticky), and a base of 0 is correct --
                                    // the readback is then simply waste. raw_fmt non-zero with
                                    // base=0 is the opposite: the guest declared a live colour
                                    // target and prosper does not have its address.
                                    "seed=%d defer=%d writes=%llu px_nonblack=%lld bytes=%zu "
                                    "raw_fmt=%u\n",
                                    (unsigned long long)at, pass_i, items.size(),
                                    (unsigned long long)base, gw, gh, (int)pass_format,
                                    (int)is_vo, (int)seed_rtt0, (int)defer_readback,
                                    (unsigned long long)color_target_call.writes, nz,
                                    rendered_pixels.size(),
                                    pass.empty() ? 0u : pass.front()->ps.color0_format);
                    }
                    // Everything this group did AFTER the backend call returned -- RTT store,
                    // scanout selection, per-pass diagnostics. `backend_done` is in scope only on
                    // the paths that reached the render, which are exactly the paths `pre` counted,
                    // so the two spans cover the same population and their sum is comparable to
                    // pass_groups.
                    if (timing_enabled) {
                        const auto post_done = RenderClock::now();
                        pending_timing.pass_post_ms += std::chrono::duration<double, std::milli>(
                            post_done - backend_done).count();
                        pending_timing.post_stats_ms += std::chrono::duration<double, std::milli>(
                            post_stats_done - backend_done).count();
                        pending_timing.post_slot0_ms += std::chrono::duration<double, std::milli>(
                            post_slot0_done - post_stats_done).count();
                        pending_timing.post_mrt_ms += std::chrono::duration<double, std::milli>(
                            post_mrt_done - post_slot0_done).count();
                        pending_timing.post_rest_ms += std::chrono::duration<double, std::milli>(
                            post_done - post_mrt_done).count();
                    }
                }
                // Everything after the group loop: present selection, scanout resolution, the final
                // render, RTT publication. The tail performs its own build_resources+backend, and
                // pass_control already excludes those -- so subtract what they grew by across this
                // span, or the leaves would exceed their parent and the guard would fire on a
                // correct run.
                if (timing_enabled) {
                    pending_timing.pass_loop_ms += std::chrono::duration<double, std::milli>(
                        RenderClock::now() - pass_loop_start).count() -
                        (pending_timing.build_resources_ms + pending_timing.backend_ms -
                         pass_loop_measured_before);
                    pass_tail_start = RenderClock::now();
                    pass_tail_measured_before =
                        pending_timing.build_resources_ms + pending_timing.backend_ms;
                }
                // Present priority: the flipped front buffer > any registered scanout target > the
                // legacy "last group" fallback (unchanged behavior when no group targets a VO buffer).
                // Under the publish extent contract (#1986) a candidate whose byte count disagrees is
                // not a present source, so a correctly sized lower-priority candidate is preferred over
                // an incorrectly sized higher-priority one, and "nothing qualified" yields no source at
                // all rather than a frame the caller must discard. Outside a publish-gate submit
                // (offline replay, render_submit_items tests) this is byte-for-byte the historical
                // identity priority.
                auto candidate_bytes = [](const std::shared_ptr<const std::vector<uint8_t>>& p) {
                    return p ? p->size() : size_t{0};
                };
                present_choice = prosper::frontend::select_present_source(
                    candidate_bytes(px_front), candidate_bytes(px_vo), candidate_bytes(px_last),
                    present_extent_bytes);
                using prosper::frontend::PresentSourceChoice;
                selected_pixels = present_choice == PresentSourceChoice::Front ? px_front
                                : present_choice == PresentSourceChoice::Vo    ? px_vo
                                : present_choice == PresentSourceChoice::Last  ? px_last
                                : nullptr;
                // #2283's blocking arm, measured rather than argued: is a pass whose colour target is
                // DISABLED (CB_COLOR0_INFO.FORMAT == 0, CB_COLOR_INVALID) ever chosen as the frame
                // that gets published? The proposed fix stops forcing a readback on such a pass, and
                // it is safe by CONSTRUCTION only if the answer is never. Nothing checked it before,
                // and the guard that looks like it would -- pass_format == R8G8B8A8_UNORM at the
                // candidate site -- does NOT exclude a disabled target, because backend_color_format()
                // FALLS BACK to R8G8B8A8_UNORM for anything it does not recognise, including 0.
                if (timing_enabled) {
                    const uint32_t chosen_fmt =
                        present_choice == PresentSourceChoice::Front ? px_front_fmt
                      : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
                      : present_choice == PresentSourceChoice::Last  ? px_last_fmt
                      : kNoPassFormat;
                    if (selected_pixels) {
                        ++pending_timing.publish_selected;
                        if (chosen_fmt == 0) ++pending_timing.publish_selected_fmt0;
                        else if (chosen_fmt == kNoPassFormat) ++pending_timing.publish_selected_unknown;
                    }
                }
                // The one new semantic path: a non-empty higher-priority candidate was passed over on
                // extent. Unreachable today (a VO/front-buffer pass has its extent pinned to the present
                // extent at :5060-5063 before it renders, so those candidates always fit), which is
                // exactly why it gets a report rather than a comment — this PR's whole thesis is that a
                // silent substitution is what cost 31 consecutive submits, and that applies to its own
                // substitution too. If the pin is ever relaxed, this says so instead of quietly changing
                // which surface reaches the screen.
                if (prosper::frontend::present_source_demoted(
                        present_choice, candidate_bytes(px_front), candidate_bytes(px_vo))) {
                    static std::atomic<uint64_t> demotions{0};
                    const uint64_t ord = demotions.fetch_add(1) + 1;
                    if (prosper::diag_should_print(ord))
                        fprintf(stderr,
                                "[rtt] PRESENT SOURCE DEMOTED #%llu: chose %s for a requested %ux%u "
                                "(%zu-byte) frame because a higher-priority candidate did not fit — "
                                "px_front=%ux%u(%zu bytes) px_vo=%ux%u(%zu bytes); this should be "
                                "unreachable while VO passes are pinned to the present extent "
                                "(live_renderer.cpp:5060) — check that pin first\n",
                                (unsigned long long)ord,
                                prosper::frontend::present_source_name(present_choice), w, h,
                                present_extent_bytes, px_front_w, px_front_h,
                                candidate_bytes(px_front), px_vo_w, px_vo_h, candidate_bytes(px_vo));
                }
                {
                    const uint64_t at = g_pass_log_submit.fetch_add(1);
                    if (PROSPER_ENV_ON("PROSPER_PASS_LOG")) {
                        // The same window object and the same ordinal the per-pass report used, so
                        // the two lines describe one set of callbacks — with one seam: if a `ms:`
                        // deadline is crossed between the pass loop and this site, the latching
                        // callback carries `selected=` with no per-pass lines beside it.
                        if (g_pass_log_window.contains(at, diagnostic_elapsed_ms()))
                            fprintf(stderr, "[pass] cb=%llu selected=%s\n", (unsigned long long)at,
                                    prosper::frontend::present_source_name(present_choice));
                    }
                }
                // PROSPER_DUMP_PERSISTENT=<min-submit>|ms:<millis>: read back and dump EVERY
                // persistent color target after this submit's passes render. Unlike
                // PROSPER_RTT*/DUMP_*, this flag is NOT in the live_gpu_targets disable list
                // (:1078-1085), so it observes the NORMAL persistent-render path (all other CPU-pixel
                // diagnostics change that path -> #1103). readback_persistent_color_target restores
                // the image layout, so rendering is unaffected. The `ms:` form aims the window by
                // wall time, because the ordinal below is a renderer-internal counter with no
                // published rate and the states worth censusing are named in seconds
                // (diagnostic_window.hpp).
                if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
                    static std::atomic<uint64_t> dp_submit{0};
                    const uint64_t sub = dp_submit.fetch_add(1);
                    if (g_persist_window.contains(sub, diagnostic_elapsed_ms())) {
                        const char* dd = getenv("PROSPER_FRAME_DIR");
                        fprintf(stderr, "[persist] submit=%llu present: front=%d/%d front_va=0x%llx "
                                "selected=%s vo:", (unsigned long long)sub, vo_front, vo_n,
                                (unsigned long long)front_va,
                                prosper::frontend::present_source_name(present_choice));
                        for (int i = 0; i < vo_n && i < 8; i++)
                            fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
                        fprintf(stderr, "\n");
                        for (auto& kv : g_rtt) {
                            RttSurf& s = kv.second;
                            if (!s.gpu_valid || !s.w || !s.h) continue;
                            if (static_cast<uint64_t>(s.w) * s.h < 64u * 64u) continue;
                            const VkFormat fmt = prosper::test::backend_color_format(s.format);
                            const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(fmt);
                            const uint64_t expected = static_cast<uint64_t>(s.w) * s.h * bpp;
                            std::vector<uint8_t> px; std::string err;
                            if (!prosper::test::readback_persistent_color_target(
                                    kv.first, s.w, s.h, fmt, px, err) || px.size() != expected)
                                continue;
                            const std::vector<uint8_t> rgba = inspection_rgba8(px, s.w, s.h, fmt);
                            size_t rgbnz = 0;
                            for (size_t p = 0; p + 3 < rgba.size(); p += 4)
                                if (rgba[p] || rgba[p + 1] || rgba[p + 2]) rgbnz++;
                            // "Black" and "empty" are different findings, and `rgb_nonblack` alone
                            // cannot tell them apart: an HDR target whose texels are negative,
                            // non-finite or below 1/510 converts to black from bits that are NOT
                            // zero, which is a different defect signature from a zero-filled buffer
                            // (#1905 recorded the wrong one from the converted count alone). Report
                            // the PRE-conversion evidence beside it: how many raw bytes are non-zero,
                            // and the first non-zero texel's own bytes, so the row a lane writes down
                            // is a measurement.
                            size_t rawnz = 0;
                            size_t first_nz_texel = 0;
                            bool have_first = false;
                            const uint32_t texel_bytes = px.size() && s.w && s.h
                                ? (uint32_t)(px.size() / ((size_t)s.w * s.h)) : 0;
                            for (size_t i = 0; i < px.size(); i++) {
                                if (!px[i]) continue;
                                rawnz++;
                                if (!have_first && texel_bytes) {
                                    first_nz_texel = i / texel_bytes;
                                    have_first = true;
                                }
                            }
                            // Two distinct sentinels, not one: "none" is the finding "no byte of this
                            // target is non-zero", while an unprintable texel stride is the
                            // instrument declining to answer. Sharing one string would let the
                            // second read as the first — a silent zero, which is the class of lie
                            // this whole diagnostic exists to stop reporting. Unreachable today
                            // (every format the census dumps is <= 16 B/texel), so the point is that
                            // it stays unambiguous if a wider one ever appears.
                            char first_text[96] = "none";
                            if (have_first && (!texel_bytes || texel_bytes > 16))
                                std::snprintf(first_text, sizeof first_text,
                                              "unprintable (%u bytes/texel)", texel_bytes);
                            if (have_first && texel_bytes && texel_bytes <= 16) {
                                int at = std::snprintf(first_text, sizeof first_text,
                                                       "texel %zu =", first_nz_texel);
                                for (uint32_t b = 0; b < texel_bytes && at > 0 &&
                                                    at < (int)sizeof first_text - 4; b++)
                                    at += std::snprintf(first_text + at, sizeof first_text - at,
                                                        " %02x", px[first_nz_texel * texel_bytes + b]);
                            }
                            char fn[512];
                            std::snprintf(fn, sizeof fn, "%s/persist_s%04llu_%llx_%ux%u.bmp",
                                          dd ? dd : ".", (unsigned long long)sub,
                                          (unsigned long long)kv.first, s.w, s.h);
                            prosper::test::dump_bmp(fn, rgba, s.w, s.h);
                            fprintf(stderr, "[persist] submit=%llu addr=0x%llx %ux%u fmt=%u "
                                    "rgb_nonblack=%zu/%u raw_nonzero_bytes=%zu/%zu first_nonzero=%s\n",
                                    (unsigned long long)sub,
                                    (unsigned long long)kv.first, s.w, s.h, (unsigned)s.format,
                                    rgbnz, s.w * s.h, rawnz, px.size(), first_text);
                        }
                    }
                }
            } else {
                // Single-framebuffer path: render_draws_rgba composites every draw into ONE framebuffer.
                std::vector<const prosper::gpu::DrawItem*> all; all.reserve(items.size());
                for (const auto& it : items) all.push_back(&it);
                const auto build_start = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                auto backend_draws = build_bds(all);
                const auto build_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                auto rendered = prosper::test::render_draws_rgba(
                    backend_draws, w, h, nullptr, clear_for(all), true);
                const auto backend_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
                    ? prosper::test::backend_render_timing_stats()
                    : prosper::test::BackendRenderTimingStats{};
                const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
                    ? prosper::test::backend_pipeline_cache_stats()
                    : prosper::test::BackendPipelineCacheStats{};
                // Captured HERE, beside the pipeline stats, and PASSED IN -- not read inside
                // record_backend_timing. These are thread_local and published by the backend call;
                // reading them from wherever the recorder happens to run returns a different,
                // nearly empty instance. The first draft did exactly that and reported refs=11 for
                // a 2,102-draw submit -- the kind of small plausible number that gets believed.
                const prosper::test::BackendResourceReuseStats backend_reuse_stats = timing_enabled
                    ? prosper::test::backend_resource_reuse_stats()
                    : prosper::test::BackendResourceReuseStats{};
                if (timing_enabled) {
                    pending_timing.build_resources_ms +=
                        std::chrono::duration<double, std::milli>(build_done - build_start).count();
                    pending_timing.backend_ms +=
                        std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                    record_backend_timing(backend_call_timing, backend_pipeline_stats,
                                          backend_reuse_stats);
                }
                // RTT (#167): cache these rendered pixels under this submit's render-target base, so a later
                // composite pass that samples that address gets the scene we drew (not empty guest memory).
                selected_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(rendered));
                if (rtt_on && !selected_pixels->empty()) {
                    uint64_t tgt = 0;
                    for (const auto& it : items) if (it.color0_base) { tgt = it.color0_base; break; }
                    if (tgt) {
                        RttSurf& s = g_rtt[tgt];
                        s.rgba = selected_pixels; s.w = w; s.h = h;
                        s.format = VK_FORMAT_R8G8B8A8_UNORM; s.gpu_valid = false;
                        s.guest_format = items.empty()
                            ? VK_FORMAT_R8G8B8A8_UNORM
                            : static_cast<VkFormat>(
                                  prosper::frontend::mrt_raw_format(items.front(), 0));
                        s.has_uniform_color = false;
                        s.dcc_metadata_dirty = false;
                        prosper::test::invalidate_persistent_color_target(tgt);
                    }
                    if (rtt_log) { size_t nz=0;
                        for (uint8_t byte : *selected_pixels) nz += byte != 0;
                        fprintf(stderr, "[rtt] store target=0x%llx (%zu items, color0s:", (unsigned long long)tgt, items.size());
                        for (const auto& it : items) fprintf(stderr, " 0x%llx", (unsigned long long)it.color0_base);
                        fprintf(stderr, ") px_nonzero=%zu cache_size=%zu\n", nz, g_rtt.size()); }
                }
            }
            if (timing_enabled)
                pending_timing.pass_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - pass_timing_start).count();
            if (timing_enabled)
                pending_timing.pass_tail_ms +=
                    std::chrono::duration<double, std::milli>(
                        RenderClock::now() - pass_tail_start).count() -
                    (pending_timing.build_resources_ms + pending_timing.backend_ms -
                     pass_tail_measured_before);
            // Ordered submits may invoke this callback for several graphics spans separated by
            // compute dispatches. Intermediate spans only update g_rtt. At the final span, recover
            // the flipped scanout from that persistent cache even when it was rendered earlier in
            // the transaction, and advance frame/dump state exactly once.
            const auto output_copy_start = timing_enabled && phase.final_span
                ? RenderClock::now() : RenderClock::time_point{};
            if (phase.final_span && pertarget) {
                auto cached_scanout = [&](uint64_t addr) -> const RttSurf* {
                    auto it = g_rtt.find(addr);
                    if (it == g_rtt.end() || it->second.w != w || it->second.h != h ||
                        it->second.format != VK_FORMAT_R8G8B8A8_UNORM)
                        return nullptr;
                    RttSurf& surface = it->second;
                    const size_t expected = static_cast<size_t>(w) * h * 4;
                    if ((!surface.rgba || surface.rgba->size() != expected) &&
                        surface.has_uniform_color)
                        materialize_uniform_rtt(surface);
                    if ((!surface.rgba || surface.rgba->size() != expected) && surface.gpu_valid) {
                        std::vector<uint8_t> materialized;
                        std::string error;
                        if (prosper::test::readback_persistent_color_target(
                                addr, w, h, surface.format, materialized, error) &&
                            materialized.size() == expected) {
                            surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                                std::move(materialized));
                        } else {
                            static std::atomic<int> warned{0};
                            if (warned.fetch_add(1) < 24)
                                std::fprintf(stderr,
                                             "[rtt] final scanout readback failed: base=0x%llx "
                                             "extent=%ux%u error=%s\n",
                                             static_cast<unsigned long long>(addr), w, h,
                                             error.c_str());
                            return nullptr;
                        }
                    }
                    return surface.rgba && surface.rgba->size() == expected ? &surface : nullptr;
                };
                const int front = prosper::gpu::present_front_index();
                static uint64_t last_gpu_publish_flip = UINT64_MAX;
                const uint64_t current_flip = prosper_vo_flip_count();
                const bool new_gpu_flip = prosper::frontend::present_blit_has_new_flip(
                    last_gpu_publish_flip, current_flip);
                // GPU present (#1270): when prosper-app has adopted this device and is consuming the
                // front-buffer image directly, blit it into a scanout slot on the GPU and SKIP the CPU
                // readback+reupload entirely. gpu_present_active() is false in every headless/test/
                // screenshot process, so this whole branch is inert there and the CPU path below is
                // byte-for-byte unchanged. On a MISS (image not resident/valid this frame) this falls
                // through to the CPU readback below, which still publishes a CPU frame; prosper-app
                // presents that CPU frame when no GPU frame was published (main.cpp), so a miss degrades to
                // the CPU present path rather than freezing the window.
                if (front >= 0 && prosper::gpu::gpu_present_active() && !new_gpu_flip) {
                    // The previously published slot remains the correct scanout for this guest
                    // flip. Treat it as a successful GPU publication so intermediate render
                    // submissions do not fall through to the expensive CPU readback path.
                    published_gpu = true;
                } else if (front >= 0 && prosper::gpu::gpu_present_active()) {
                    const uint64_t front_va = prosper_vo_buffer_addr(front);
                    auto rit = g_rtt.find(front_va);
                    if (rit != g_rtt.end() && rit->second.gpu_valid && rit->second.w && rit->second.h) {
                        const VkFormat fmt = prosper::test::backend_color_format(rit->second.format);
                        prosper::test::PersistentColorTargetImage* tgt =
                            prosper::test::find_persistent_color_target(
                                front_va, rit->second.w, rit->second.h, fmt);
                        if (tgt && tgt->image && tgt->layout != VK_IMAGE_LAYOUT_UNDEFINED) {
                            published_gpu = prosper::frontend::present_blit_publish(
                                tgt->image, tgt->layout, fmt, rit->second.w, rit->second.h,
                                current_flip);
                            if (published_gpu) last_gpu_publish_flip = current_flip;
                        }
                    }
                }
                const RttSurf* scanout = (!published_gpu && front >= 0)
                    ? cached_scanout(prosper_vo_buffer_addr(front)) : nullptr;
                if (!published_gpu && !scanout) {
                    for (int i = 0; i < prosper_vo_buffer_count(); ++i)
                        if ((scanout = cached_scanout(prosper_vo_buffer_addr(i)))) break;
                }
                if (scanout) selected_pixels = scanout->rgba;
                // Last resort before the retained frame: the flipped buffer's own guest memory.
                // The decision itself lives in guest_scanout_present.hpp so it can be unit-tested;
                // read that header for why each condition is there. Only the mechanism is here.
                //
                // Read once per guest flip, not once per submit: this callback runs for every span
                // of every submit (eleven per frame on Frontiers), the buffer only changes when the
                // guest flips a new one, and a 4K copy plus de-swizzle is not free. These statics
                // have the same single-present-thread contract as `last_scanout_present` below.
                static uint64_t guest_scanout_flip = UINT64_MAX;
                static std::shared_ptr<const std::vector<uint8_t>> guest_scanout_pixels;
                const size_t display_bytes =
                    static_cast<size_t>(prosper::gpu::present_width()) *
                    prosper::gpu::present_height() * 4u;
                if (prosper::frontend::guest_scanout_read_warranted(
                        published_gpu, scanout != nullptr, (bool)selected_pixels,
                        present_extent_bytes, display_bytes) ==
                    prosper::frontend::GuestScanoutDecision::Publish) {
                    if (guest_scanout_flip != current_flip) {
                        guest_scanout_flip = current_flip;
                        guest_scanout_pixels.reset();
                        prosper::VideoOutLinearRead read;
                        const bool got = prosper::videoout_read_front_linear(read);
                        // Named rather than inlined so the #2932 probe below can REPORT it. Whether
                        // the renderer owns a target at the flipped VA is the first thing
                        // guest_scanout_publishable tests, and it short-circuits ahead of
                        // guest_authored -- so a decline of SkipNotAuthored is itself evidence that
                        // this was false at that flip. Reading that off the decision name is
                        // inference; printing the input is measurement.
                        const bool renderer_owns_flip =
                            got && g_rtt.find(read.metadata.address) != g_rtt.end();
                        const auto decision = prosper::frontend::guest_scanout_publishable(
                            got ? read.pixels.size() : 0u, present_extent_bytes,
                            got && read.metadata.address != 0,
                            renderer_owns_flip,
                            read.guest_authored);
                        if (decision == prosper::frontend::GuestScanoutDecision::Publish)
                            guest_scanout_pixels = std::make_shared<const std::vector<uint8_t>>(
                                std::move(read.pixels));
                        // Report the OUTCOME, publish or decline, and name which. A negative
                        // control that can only observe silence cannot tell "correctly declined"
                        // from "never reached" — and a decline is exactly what the Bendy
                        // (PPSA27616) reused-memory case must produce, so it has to be legible.
                        //
                        // Budgeted PER DECISION, which is the whole point on this line and not
                        // ceremony (diag_ratelimit.hpp: "budget per key, or a noisy key exhausts the
                        // log before the one under investigation gets a line"). A title that
                        // declines on every flip would otherwise push the first PUBLISH past the
                        // cap, and "no publish line" is exactly the reading the negative control
                        // rests on. With its own counter the first publish is ordinal 1 and always
                        // prints, so the absence of a `publish` line means zero publishes.
                        static std::atomic<uint64_t>
                            guest_scanout_reports[(size_t)
                                prosper::frontend::GuestScanoutDecision::SkipNotAuthored + 1]{};
                        const uint64_t ord =
                            guest_scanout_reports[(size_t)decision].fetch_add(1) + 1;
                        if (prosper::diag_should_print(ord))
                            fprintf(stderr,
                                    "[rtt] GUEST SCANOUT #%llu: no present source and no renderer "
                                    "target at the flipped buffer 0x%llx — %s (%ux%u tiling=%u "
                                    "authored=%d footprint=%s)\n",
                                    (unsigned long long)ord,
                                    (unsigned long long)read.metadata.address,
                                    prosper::frontend::guest_scanout_decision_name(decision), w, h,
                                    read.metadata.tiling_mode, (int)read.guest_authored,
                                    read.padded_footprint ? "padded" : "nominal");
                        // #2932: what is actually true of the flipped buffer at the moment we
                        // decline to publish it? Three facts, printed together because separately
                        // each one invites a wrong inference:
                        //
                        //   phys=      the physical page behind the flipped VA. This began as a test
                        //              of "scanout and the render targets are two mappings of one
                        //              allocation"; that is FALSE (STRAY_STATUS.md, Ruled out).
                        //   aliases=N  the SEARCHED DOMAIN. The table is populated only while
                        //              page-protection watches are armed, so N=0 means "nothing was
                        //              searched" and says nothing about this VA. A miss reported
                        //              without its domain is how an unarmed instrument gets recorded
                        //              as a negative result.
                        //   rtt_owns=  whether g_rtt holds a target at this VA -- the FIRST input
                        //              guest_scanout_publishable tests, short-circuiting ahead of
                        //              guest_authored. Measured, not inferred from the decision name.
                        //
                        // Deliberately NOT asserted here: that no VA-keyed lookup can connect the
                        // flipped buffer to a render target. Stray's scanout VAs appear in the
                        // colour-target census as 4K attachments in passes carrying draws, so that
                        // sentence -- which earlier revisions of this comment stated as fact -- is
                        // withdrawn. rtt_owns is printed precisely so nobody has to assume it again.
                        if (read.metadata.address && prosper::diag_should_print(ord)) {
                            uint64_t flip_phys = 0;
                            size_t alias_n = 0;
                            const bool ok = prosper::host::guest_write_watch_va_to_phys(
                                read.metadata.address, flip_phys, &alias_n);
                            if (ok)
                                fprintf(stderr,
                                        "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> phys=0x%llx "
                                        "(aliases=%zu rtt_owns=%d decision=%s)\n",
                                        (unsigned long long)ord,
                                        (unsigned long long)read.metadata.address,
                                        (unsigned long long)flip_phys, alias_n,
                                        (int)renderer_owns_flip,
                                        prosper::frontend::guest_scanout_decision_name(decision));
                            else
                                fprintf(stderr,
                                        "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> UNRESOLVED "
                                        "(aliases=%zu rtt_owns=%d decision=%s; %s)\n",
                                        (unsigned long long)ord,
                                        (unsigned long long)read.metadata.address, alias_n,
                                        (int)renderer_owns_flip,
                                        prosper::frontend::guest_scanout_decision_name(decision),
                                        alias_n ? "no alias range covers this VA"
                                                : "alias table EMPTY -- nothing was searched");
                        }
                    }
                    // Re-check the size on the cached path too: a two-set geometry switch can change
                    // the present extent between spans of one flip, and the cache is keyed on the
                    // flip alone.
                    if (guest_scanout_pixels && guest_scanout_pixels->size() == present_extent_bytes) {
                        selected_pixels = guest_scanout_pixels;
                        frame_origin = prosper::gpu::PresentFrameOrigin::GuestScanout;
                    }
                }
                // Hold the last good scanout across VideoOut buffer rotation. Bendy (PPSA27616) rapidly
                // re-registers its scanout buffers; on the frames where the guest has registered new
                // buffers but not yet rendered+flipped them, none is gpu_valid and the present would be
                // a black flicker. Presenting the previous frame instead keeps a stable image (the new
                // buffers get drawn and flipped within a frame or two). CONFIDENCE: MED.
                // Thread-safety: this static is a plain (non-atomic) shared_ptr, correct only under the
                // renderer's single present thread (this callback and its sibling statics — dp_submit,
                // warned, frame_no — all assume the one serialized present path). It must not be read or
                // assigned from another thread; a concurrent present would race the object assignment.
                // The net's own test used to be `!empty()`, which is not the publish predicate (#1986):
                // a non-empty WRONG-EXTENT frame took the store branch, overwrote the retained good
                // frame, and made the recovery branch unreachable for the rest of the process. Sonic
                // Frontiers' 4K publish wall was permanent for exactly that reason. Retaining is now
                // the same predicate as publishing, so a frame prosper could not publish can never
                // become the frame it serves later.
                static std::shared_ptr<const std::vector<uint8_t>> last_scanout_present;
                // Provenance is a property of the RETAINED frame, not of the submit that re-serves
                // it. Post-wall, Frontiers re-serves one retained frame thousands of times (fresh +1
                // against retained +2,048 between two shortfall ordinals), so a label that reset on
                // re-serve would report the guest's own buffer as a composited frame for all but the
                // first of them — and `--require-composited-frame` would pass on the strength of the
                // copies.
                static bool last_scanout_present_from_guest = false;
                if (!published_gpu) {
                    const size_t current_bytes = selected_pixels ? selected_pixels->size() : 0u;
                    const size_t retained_bytes =
                        last_scanout_present ? last_scanout_present->size() : 0u;
                    switch (prosper::frontend::retained_frame_action(
                                current_bytes, retained_bytes, present_extent_bytes)) {
                        case prosper::frontend::RetainedFrameAction::StoreCurrent:
                            last_scanout_present = selected_pixels;
                            last_scanout_present_from_guest =
                                frame_origin == prosper::gpu::PresentFrameOrigin::GuestScanout;
                            present_frames_stored.fetch_add(1, std::memory_order_relaxed);
                            // `retained_frame_action` decides on BYTE SIZES ALONE -- by design, since
                            // it cannot know what a frame should look like. The consequence is that a
                            // correctly-sized but CONTENTLESS frame is retained just as readily as a
                            // good one, and is then re-served on every later submit that produces no
                            // present source. Little Nightmares III presents a uniform (255,255,0)
                            // frame on ~2/3 of samples for exactly that reason (#2014): the retained
                            // frame is itself uniform.
                            //
                            // This names the ORIGIN of a uniform retained frame, which is the one
                            // fact the existing counters cannot supply -- `fresh=N retained=M` says a
                            // substitution happened, not what was substituted or where it came from.
                            // Sampled, not exhaustive: a full uniformity scan of a 33 MB frame on the
                            // present path would cost more than the render.
                            if (const char* uni = PROSPER_ENV_VALUE("PROSPER_UNIFORMLOG")) {
                                if (uni[0] == '1' && uni[1] == '\0' && selected_pixels &&
                                    selected_pixels->size() >= 4) {
                                    const uint8_t* px = selected_pixels->data();
                                    const size_t n = selected_pixels->size();
                                    bool uniform = true;
                                    for (size_t off = 4; off + 4 <= n; off += ((n / 4096) | 4u) & ~3u)
                                        if (px[off] != px[0] || px[off+1] != px[1] ||
                                            px[off+2] != px[2] || px[off+3] != px[3]) {
                                            uniform = false; break;
                                        }
                                    if (uniform) {
                                        static std::atomic<uint64_t> uniform_stores{0};
                                        const uint64_t ord =
                                            uniform_stores.fetch_add(1, std::memory_order_relaxed) + 1;
                                        // WHICH pass handed us the uniform frame. `origin=` only
                                        // separates a composite from a republished guest scanout; it
                                        // cannot say which colour target the composite read back, and
                                        // that is the one fact needed to go from "a uniform frame was
                                        // retained" to "this surface is uniform" (#2014).
                                        using prosper::frontend::PresentSourceChoice;
                                        const uint32_t src_w =
                                            present_choice == PresentSourceChoice::Front ? px_front_w
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_w
                                          : present_choice == PresentSourceChoice::Last  ? px_last_w : 0u;
                                        const uint32_t src_h =
                                            present_choice == PresentSourceChoice::Front ? px_front_h
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_h
                                          : present_choice == PresentSourceChoice::Last  ? px_last_h : 0u;
                                        const uint64_t src_base =
                                            present_choice == PresentSourceChoice::Front ? px_front_base
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_base
                                          : present_choice == PresentSourceChoice::Last  ? px_last_base : 0u;
                                        const uint32_t src_fmt =
                                            present_choice == PresentSourceChoice::Front ? px_front_fmt
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
                                          : present_choice == PresentSourceChoice::Last  ? px_last_fmt
                                                                                        : kNoPassFormat;
                                        // A VkFormat, not the raw CB_COLOR0_INFO.FORMAT:
                                        // `ResolvedPipelineState::color0_format` is already
                                        // `vk_color_format(...)`'s output (render_state.cpp:789).
                                        // Naming it `cbfmt` cost a wrong reading once -- 44 is
                                        // VK_FORMAT_B8G8R8A8_UNORM, which is not a CB format value
                                        // at all.
                                        char fmt_text[16];
                                        if (src_fmt == kNoPassFormat) snprintf(fmt_text, sizeof fmt_text, "none");
                                        else snprintf(fmt_text, sizeof fmt_text, "%u", src_fmt);
                                        if (prosper::diag_should_print(ord))
                                            fprintf(stderr,
                                                    "[uniformlog] #%llu retaining a UNIFORM frame "
                                                    "rgba=(%u,%u,%u,%u) bytes=%zu origin=%s src=%s "
                                                    "%ux%u@0x%llx vkfmt=%s\n",
                                                    (unsigned long long)ord,
                                                    px[0], px[1], px[2], px[3], n,
                                                    frame_origin ==
                                                        prosper::gpu::PresentFrameOrigin::GuestScanout
                                                        ? "GuestScanout" : "Composited",
                                                    prosper::frontend::present_source_name(present_choice),
                                                    src_w, src_h, (unsigned long long)src_base, fmt_text);
                                    }
                                }
                            }
                            break;
                        case prosper::frontend::RetainedFrameAction::ServeRetained:
                            selected_pixels = last_scanout_present;
                            frame_origin = last_scanout_present_from_guest
                                ? prosper::gpu::PresentFrameOrigin::GuestScanout
                                : prosper::gpu::PresentFrameOrigin::Composited;
                            present_frames_served.fetch_add(1, std::memory_order_relaxed);
                            // A stale-but-correct frame IS what the guest sees, so say so rather than
                            // letting a silent substitution read as a healthy present. Under no extent
                            // contract this branch is the historical empty-frame recovery and is not
                            // itself a defect, so it stays quiet there.
                            if (present_extent_bytes)
                                report_present_extent_shortfall(
                                    "serving the retained frame instead", current_bytes);
                            break;
                        case prosper::frontend::RetainedFrameAction::NoUsableFrame:
                            // Nothing publishable this submit and nothing retained to fall back on, so
                            // the screen does not advance. selected_pixels is left as selection found
                            // it — normally empty on this path, since selection already refused every
                            // wrong-extent candidate — which keeps the caller's `[render] … Vulkan
                            // render FAILED` and `[agc] PUBLISH DROPPED` backstops intact. This is the
                            // one outcome that is a lost frame rather than a substituted one, so it is
                            // reported whenever the contract applies, including when nothing rendered
                            // at all: "no pass produced a present-extent source" is the finding either
                            // way, and Frontiers' wall was invisible for exactly as long as it was not
                            // stated anywhere.
                            if (present_extent_bytes)
                                report_present_extent_shortfall(
                                    "nothing is published for this submit", current_bytes);
                            break;
                    }
                }
                if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
                    size_t nb = 0;
                    if (selected_pixels)
                        for (size_t p = 0; p + 3 < selected_pixels->size(); p += 4)
                            if ((*selected_pixels)[p] || (*selected_pixels)[p+1] || (*selected_pixels)[p+2]) nb++;
                    fprintf(stderr, "[persist] present: front=%d front_va=0x%llx scanout=%s rgb_nonblack=%zu\n",
                            front, (unsigned long long)(front >= 0 ? prosper_vo_buffer_addr(front) : 0),
                            scanout ? "HIT" : "MISS", nb);
                }
            }
            if (timing_enabled && phase.final_span)
                pending_timing.output_copy_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - output_copy_start).count();
            if (phase.final_span) {
                release_pinned_scanouts();
                release_consumed_renderer_mip_targets();
            }
            if (phase.final_span && lightweight_rtt_timing && rtt_log_in_range &&
                pending_timing.backend_draws >= rtt_timing_min_draws) {
                std::string output;
                output.reserve(pending_rtt_timing.size() * 320);
                for (const RttTimingRecord& record : pending_rtt_timing)
                    append_rtt_timing(output, record);
                if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
            }
            if (!phase.final_span) {
                if (timing_enabled) {
                    pending_timing.callbacks++;
                    pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                        RenderClock::now() - callback_timing_start).count();
                }
                prosper::frontend::reset_performance_timing_after_span(
                    pending_timing, timing_enabled, phase.final_span);
                return {};
            }
            static const std::vector<uint8_t> empty_pixels;
            const std::vector<uint8_t>& px = selected_pixels ? *selected_pixels : empty_pixels;
            int n = frame_no++;
            // PROSPER_DUMP_CONTENT=<min-nonzero-bytes>: dump ONLY frames whose framebuffer has at least
            // that many nonzero bytes — catches the intermittent content submits the periodic dump misses.
            size_t content_thr = 0; if (const char* c = PROSPER_ENV_VALUE("PROSPER_DUMP_CONTENT")) content_thr = (size_t)atol(c);
            // Sparse long-route captures can override the default first-60/every-10 cadence. This is
            // particularly useful for 4K titles, where capturing itself would otherwise add gigabytes
            // of readback I/O before the scene under investigation is reached. Zero disables a phase.
            static const int dump_first = [] { const char* e = getenv("PROSPER_FRAME_DUMP_FIRST");
                                                return e ? (int)atol(e) : 60; }();
            static const int dump_every = [] { const char* e = getenv("PROSPER_FRAME_DUMP_EVERY");
                                                return e ? (int)atol(e) : 10; }();
            // PROSPER_PRESENT_NZLOG=N: log the presented frame's nonzero-byte count every N frames WITHOUT
            // writing any image. A memory-safe content proxy for long progression runs — dumping BMPs to a
            // tmpfs frame dir exhausts RAM, this does not. 0/unset disables.
            static const int nzlog_every = [] { const char* e = getenv("PROSPER_PRESENT_NZLOG");
                                                return e ? (int)atol(e) : 0; }();
            size_t px_nz = 0;
            if (dump_bmps || nzlog_every) for (uint8_t b : px) px_nz += (b != 0);
            if (px.empty() && !published_gpu) {
                fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
            } else if (dump_bmps && ((content_thr && px_nz >= content_thr) ||
                       (!content_thr && ((dump_first > 0 && n < dump_first) ||
                                         (dump_every > 0 && n % dump_every == 0))))) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
                prosper::test::dump_bmp(fn, px, w, h);
                fprintf(stderr, "[render] frame %d rendered (%ux%u) nz=%zu -> %s\n", n, w, h, px_nz, fn);
            } else if (nzlog_every && !px.empty() && (n % nzlog_every == 0)) {
                fprintf(stderr, "[render-nz] frame %d (%ux%u) nz=%zu\n", n, w, h, px_nz);
            }
            if (timing_enabled) {
                pending_timing.callbacks++;
                pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - callback_timing_start).count();
                // One record per complete semantic submit. A submit may be split into several
                // graphics spans by interleaved compute/DMA; recording every callback would count
                // the growing pending total repeatedly and make the capture itself misattribute time.
                if (perf_capture_timing && phase.final_span) {
                    prosper::perf::RendererTimingRecord record;
                    record.callbacks = pending_timing.callbacks;
                    record.draws = pending_timing.backend_draws;
                    record.texture_bytes = pending_timing.texture_bytes;
                    record.buffer_bytes = pending_timing.buffer_bytes;
                    record.total_ms = pending_timing.total_ms;
                    record.prelude_ms = pending_timing.prelude_ms;
                    record.pass_ms = pending_timing.pass_ms;
                    record.build_resources_ms = pending_timing.build_resources_ms;
                    record.backend_ms = pending_timing.backend_ms;
                    record.output_copy_ms = pending_timing.output_copy_ms;
                    record.pass_head_ms = pending_timing.pass_head_ms;
                    record.pass_loop_ms = pending_timing.pass_loop_ms;
                    record.pass_pre_ms = pending_timing.pass_pre_ms;
                    record.pass_post_ms = pending_timing.pass_post_ms;
                    record.post_stats_ms = pending_timing.post_stats_ms;
                    record.post_slot0_ms = pending_timing.post_slot0_ms;
                    record.post_mrt_ms = pending_timing.post_mrt_ms;
                    record.post_rest_ms = pending_timing.post_rest_ms;
                    record.pass_tail_ms = pending_timing.pass_tail_ms;
                    record.resolve_stall_ms = pending_timing.resolve_stall_ms;
                    record.resolve_read_ms = pending_timing.resolve_read_ms;
                    record.resolve_copy_stall_ms = pending_timing.resolve_copy_stall_ms;
                    record.resolve_copy_ms = pending_timing.resolve_copy_ms;
                    record.resolve_count = pending_timing.resolve_n;
                    record.resolve_read_count = pending_timing.resolve_read_n;
                    record.resolve_bytes = pending_timing.resolve_bytes;
                    record.gpu_wait_ms = pending_timing.backend_gpu_wait_ms;
                    record.gpu_timestamp_samples =
                        pending_timing.backend_gpu_timestamp_samples;
                    record.gpu_device_ms = pending_timing.backend_gpu_device_ms;
                    record.readback_ms = pending_timing.backend_readback_ms;
                    record.backend_target_ms = pending_timing.backend_target_ms;
                    record.backend_draw_setup_ms = pending_timing.backend_draw_setup_ms;
                    record.backend_record_upload_ms = pending_timing.backend_record_upload_ms;
                    record.backend_cleanup_ms = pending_timing.backend_cleanup_ms;
                    record.backend_setup_shader_ms = pending_timing.backend_setup_shader_ms;
                    record.backend_setup_fixed_ms = pending_timing.backend_setup_fixed_ms;
                    record.setup_resources_ms = pending_timing.backend_setup_resources_ms;
                    record.backend_setup_pipeline_ms =
                        pending_timing.backend_setup_pipeline_ms;
                    record.backend_pipeline_refs = pending_timing.backend_pipeline_refs;
                    record.backend_pipeline_hits = pending_timing.backend_pipeline_hits;
                    record.backend_pipeline_misses = pending_timing.backend_pipeline_misses;
                    record.backend_pipeline_bypasses = pending_timing.backend_pipeline_bypasses;
                    record.backend_pipeline_entries = pending_timing.backend_pipeline_entries;
                    record.backend_pipeline_evictions = pending_timing.backend_pipeline_evictions;
                    // Frontend and backend, kept apart by name. These two are build_resources' own
                    // texture/buffer time and are NOT parts of setup_resources_ms above.
                    record.frontend_texture_ms = pending_timing.texture_ms;
                    record.frontend_buffer_ms = pending_timing.buffer_ms;
                    record.frontend_tex_rtt_ms = pending_timing.tex_rtt_ms;
                    record.frontend_tex_compute_ms = pending_timing.tex_compute_ms;
                    record.frontend_tex_local_ms = pending_timing.tex_local_ms;
                    record.frontend_tex_persist_hit_ms = pending_timing.tex_persist_hit_ms;
                    record.frontend_tex_persist_reuse_ms = pending_timing.tex_persist_reuse_ms;
                    record.frontend_tex_persist_miss_ms = pending_timing.tex_persist_miss_ms;
                    record.frontend_tex_persist_invalid_ms = pending_timing.tex_persist_invalid_ms;
                    record.frontend_tex_persist_invalid_n = pending_timing.tex_persist_invalid_n;
                    record.frontend_tex_other_n = pending_timing.tex_other_n;
                    record.frontend_tex_other_slowest_ms =
                        pending_timing.tex_other_slowest_ms;
                    record.frontend_tex_other_addr = pending_timing.tex_other_addr;
                    record.frontend_tex_other_source_bytes =
                        pending_timing.tex_other_source_bytes;
                    record.frontend_tex_other_width = pending_timing.tex_other_width;
                    record.frontend_tex_other_height = pending_timing.tex_other_height;
                    record.frontend_tex_other_depth = pending_timing.tex_other_depth;
                    record.frontend_tex_other_format = pending_timing.tex_other_format;
                    record.frontend_tex_other_components = pending_timing.tex_other_components;
                    record.frontend_tex_other_tile_mode = pending_timing.tex_other_tile_mode;
                    record.frontend_tex_other_img_dim = pending_timing.tex_other_img_dim;
                    record.frontend_tex_other_class = pending_timing.tex_other_class;
                    record.frontend_tex_other_compute_candidate =
                        pending_timing.tex_other_compute_candidate;
                    record.frontend_tex_other_persistent_candidate =
                        pending_timing.tex_other_persistent_candidate;
                    record.frontend_tex_other_compressed =
                        pending_timing.tex_other_compressed;
                    record.frontend_tex_other_depth_compare =
                        pending_timing.tex_other_depth_compare;
                    record.frontend_tex_other_host_backed =
                        pending_timing.tex_other_host_backed;
                    record.frontend_build_draw_ms = pending_timing.build_r_ms;
                    record.frontend_validate_ms = pending_timing.build_validate_ms;
                    record.frontend_poison_ms = pending_timing.build_poison_ms;
                    record.frontend_indices_ms = pending_timing.build_indices_ms;
                    record.frontend_reflect_ms = pending_timing.build_reflect_ms;
                    // These four DO decompose setup_resources_ms, so an offline report can attribute
                    // the largest bucket in the capture instead of leaving a plausible residue.
                    record.res_texture_ms = pending_timing.backend_res_texture_ms;
                    record.res_buffer_ms = pending_timing.backend_res_buffer_ms;
                    record.res_buffer_copy_ms = pending_timing.backend_res_buffer_copy_ms;
                    record.res_buffer_create_ms = pending_timing.backend_res_buffer_create_ms;
                    record.res_buffer_index_find_ms = pending_timing.backend_res_buffer_index_find_ms;
                    record.res_buffer_index_insert_ms = pending_timing.backend_res_buffer_index_insert_ms;
                    record.res_buffer_hash_ms = pending_timing.backend_res_buffer_hash_ms;
                    record.res_descriptor_ms = pending_timing.backend_res_descriptor_ms;
                    prosper::perf::interactive_performance_capture().record_renderer(record);
                }
                if (!timing_mode.log) {
                    // F8 consumes the structured record above, then leaves without touching the
                    // lifetime aggregates or their large periodic stderr summaries.
                    prosper::frontend::reset_performance_timing_after_span(
                        pending_timing, timing_enabled, phase.final_span);
                    prosper::gpu::RenderedFrame frame(std::move(selected_pixels));
                    frame.origin = frame_origin;
                    return frame;
                }
                struct TimingTotals {
                    uint64_t submits = 0, callbacks = 0;
                    double total_ms = 0, prelude_ms = 0, pass_ms = 0;
                    double build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
                    double dcc_materialize_ms = 0;
                    uint64_t dcc_materialize_surfaces = 0, dcc_materialize_bytes = 0;
                    uint64_t backend_calls = 0, backend_draws = 0;
                    uint64_t backend_command_buffers = 0, backend_queue_submits = 0;
                    uint64_t backend_fence_waits = 0;
                    uint64_t flush_no_batch = 0, flush_readback = 0;
                    uint64_t flush_storage_writeback = 0, flush_explicit = 0;
                    uint64_t backend_gpu_timestamp_samples = 0;
                    double backend_target_ms = 0, backend_draw_setup_ms = 0;
                    double backend_record_upload_ms = 0, backend_gpu_wait_ms = 0;
                    double backend_gpu_device_ms = 0;
                    double backend_readback_ms = 0, backend_cleanup_ms = 0;
                    double backend_setup_shader_ms = 0, backend_setup_fixed_ms = 0;
                    double backend_res_fixed_index_upload_ms = 0;
                    double backend_res_fixed_blend_ms = 0;
                    double backend_res_fixed_depth_stencil_ms = 0;
                    double backend_res_fixed_viewport_ms = 0;
                    double backend_res_fixed_stages_ms = 0;
                    double backend_res_fixed_prologue_ms = 0;
                    double backend_res_prologue_subgroup_scan_ms = 0;
                    double backend_setup_resources_ms = 0, backend_setup_pipeline_ms = 0;
                    double backend_res_texture_ms = 0, backend_res_texture_upload_ms = 0;
                    double backend_res_texture_bind_ms = 0, backend_res_buffer_ms = 0;
                    double backend_res_buffer_acquire_ms = 0, backend_res_buffer_copy_ms = 0;
                    double backend_res_buffer_create_ms = 0;
                    double backend_res_buffer_index_find_ms = 0;
                    double backend_res_buffer_index_insert_ms = 0;
                    double backend_res_buffer_hash_ms = 0;
                    double backend_res_descriptor_ms = 0;
                    uint64_t backend_pipeline_refs = 0, backend_pipeline_hits = 0;
                    uint64_t backend_pipeline_misses = 0, backend_pipeline_bypasses = 0;
                    uint64_t backend_pipeline_entries = 0, backend_pipeline_evictions = 0;
                    uint64_t color_target_writes = 0, color_target_write_hits = 0;
                    uint64_t color_target_sample_hits = 0, color_target_readbacks = 0;
                    uint64_t color_target_cached_bytes = 0, color_target_cached_entries = 0;
                    uint64_t textures = 0, texture_reuses = 0, buffers = 0, buffer_views = 0;
                    uint64_t persistent_hits = 0, persistent_misses = 0, persistent_invalidations = 0;
                    uint64_t persistent_submit_reuses = 0, persistent_validations = 0;
                    uint64_t persistent_validation_bytes = 0;
                    uint64_t persistent_watch_reuses = 0, persistent_watch_dirty = 0;
                    uint64_t persistent_watch_unknown = 0, persistent_watch_disabled = 0;
                    uint64_t persistent_submit_unknown = 0, persistent_submit_overlap = 0;
                    uint64_t persistent_submit_unarmed = 0, persistent_submit_stale = 0;
                    uint64_t publish_selected = 0, publish_selected_fmt0 = 0;
                    uint64_t publish_selected_unknown = 0, publish_candidate_fmt0 = 0;
                    uint64_t texture_bytes = 0, buffer_bytes = 0, buffer_materialized_bytes = 0;
                    double texture_ms = 0, buffer_ms = 0;
                    // Attribution of texture_ms by OUTCOME class (#2262). After #2259 removed the
                    // reflection cost this is the largest leaf of build_R -- 22.56 ms/submit on a Blue
                    // Prince gameplay window, 55% of build_R -- and it is one span with nothing inside
                    // it, so "4,449 references at an 89% reuse rate cost 22.56 ms" is as far as anyone
                    // can currently get.
                    //
                    // An attribution rather than a phase partition, deliberately: these classes are
                    // already computed as locals on the per-resource path and are decided in a fixed
                    // order, so classifying in that same order makes them mutually exclusive by
                    // construction -- whereas carving a 2,800-line branch into timed phases invites the
                    // double-counting #2246 and #2250 had to be careful about.
                    double tex_rtt_ms = 0, tex_compute_ms = 0, tex_local_ms = 0;
                    double tex_persist_hit_ms = 0, tex_persist_reuse_ms = 0, tex_persist_miss_ms = 0;
                    double tex_persist_invalid_ms = 0;
                    uint64_t tex_persist_invalid_n = 0;
                    uint64_t tex_rtt_n = 0, tex_compute_n = 0, tex_local_n = 0;
                    uint64_t tex_persist_hit_n = 0, tex_persist_reuse_n = 0, tex_persist_miss_n = 0;
                    uint64_t tex_other_n = 0;
                    double pass_pre_ms = 0, pass_post_ms = 0;
                    double pass_head_ms = 0, pass_tail_ms = 0;
                    double pass_loop_ms = 0;
                    double resolve_stall_ms = 0, resolve_read_ms = 0, resolve_copy_ms = 0;
                    double resolve_copy_stall_ms = 0;
                    uint64_t resolve_n = 0, resolve_read_n = 0, resolve_bytes = 0;
                    uint64_t pass_groups_seen = 0;
                    uint64_t pass_groups = 0;
                    double build_r_ms = 0, build_validate_ms = 0;
                    double build_poison_ms = 0, build_indices_ms = 0;
                    uint64_t build_draws = 0, build_rejected = 0;
                    // A COUNT here, not a set: the aggregate sums each submit's distinct count so
                    // the report can divide by submits. A lifetime union would answer a different
                    // question (how many shaders the title has) and would over-state the ceiling
                    // for a memo whose scope is one submit.
                    double build_reflect_ms = 0;
                    uint64_t build_reflect_calls = 0, build_reflect_words = 0;
                    uint64_t build_reflect_unidentified = 0, build_reflect_distinct = 0;
                    uint64_t build_reflect_memo_hits = 0, build_reflect_memo_clears = 0;
                    uint64_t build_reflect_memo_mismatch = 0;
                };
                static TimingTotals totals;
                static TimingTotals window;
                auto accumulate = [&](TimingTotals& timing) {
                    timing.submits++;
                    timing.callbacks += pending_timing.callbacks;
                    timing.total_ms += pending_timing.total_ms;
                    timing.prelude_ms += pending_timing.prelude_ms;
                    timing.pass_ms += pending_timing.pass_ms;
                    timing.build_resources_ms += pending_timing.build_resources_ms;
                    timing.backend_ms += pending_timing.backend_ms;
                    timing.output_copy_ms += pending_timing.output_copy_ms;
                    timing.dcc_materialize_ms += pending_timing.dcc_materialize_ms;
                    timing.dcc_materialize_surfaces += pending_timing.dcc_materialize_surfaces;
                    timing.dcc_materialize_bytes += pending_timing.dcc_materialize_bytes;
                    timing.backend_calls += pending_timing.backend_calls;
                    timing.backend_draws += pending_timing.backend_draws;
                    timing.backend_command_buffers += pending_timing.backend_command_buffers;
                    timing.backend_queue_submits += pending_timing.backend_queue_submits;
                    timing.backend_fence_waits += pending_timing.backend_fence_waits;
                    timing.flush_no_batch += pending_timing.flush_no_batch;
                    timing.flush_readback += pending_timing.flush_readback;
                    timing.flush_storage_writeback += pending_timing.flush_storage_writeback;
                    timing.flush_explicit += pending_timing.flush_explicit;
                    timing.backend_gpu_timestamp_samples += pending_timing.backend_gpu_timestamp_samples;
                    timing.backend_target_ms += pending_timing.backend_target_ms;
                    timing.backend_draw_setup_ms += pending_timing.backend_draw_setup_ms;
                    timing.backend_record_upload_ms += pending_timing.backend_record_upload_ms;
                    timing.backend_gpu_wait_ms += pending_timing.backend_gpu_wait_ms;
                    timing.backend_gpu_device_ms += pending_timing.backend_gpu_device_ms;
                    timing.backend_readback_ms += pending_timing.backend_readback_ms;
                    timing.backend_cleanup_ms += pending_timing.backend_cleanup_ms;
                    timing.backend_setup_shader_ms += pending_timing.backend_setup_shader_ms;
                    timing.backend_setup_fixed_ms += pending_timing.backend_setup_fixed_ms;
                    timing.backend_res_fixed_index_upload_ms += pending_timing.backend_res_fixed_index_upload_ms;
                    timing.backend_res_fixed_blend_ms += pending_timing.backend_res_fixed_blend_ms;
                    timing.backend_res_fixed_depth_stencil_ms += pending_timing.backend_res_fixed_depth_stencil_ms;
                    timing.backend_res_fixed_viewport_ms += pending_timing.backend_res_fixed_viewport_ms;
                    timing.backend_res_fixed_stages_ms += pending_timing.backend_res_fixed_stages_ms;
                    timing.backend_res_fixed_prologue_ms += pending_timing.backend_res_fixed_prologue_ms;
                    timing.backend_res_prologue_subgroup_scan_ms += pending_timing.backend_res_prologue_subgroup_scan_ms;
                    timing.backend_setup_resources_ms += pending_timing.backend_setup_resources_ms;
                    timing.backend_res_texture_ms += pending_timing.backend_res_texture_ms;
                    timing.backend_res_texture_upload_ms += pending_timing.backend_res_texture_upload_ms;
                    timing.backend_res_texture_bind_ms += pending_timing.backend_res_texture_bind_ms;
                    timing.backend_res_buffer_ms += pending_timing.backend_res_buffer_ms;
                    timing.backend_res_buffer_acquire_ms += pending_timing.backend_res_buffer_acquire_ms;
                    timing.backend_res_buffer_copy_ms += pending_timing.backend_res_buffer_copy_ms;
                    timing.backend_res_buffer_create_ms += pending_timing.backend_res_buffer_create_ms;
                    timing.backend_res_buffer_index_find_ms += pending_timing.backend_res_buffer_index_find_ms;
                    timing.backend_res_buffer_index_insert_ms += pending_timing.backend_res_buffer_index_insert_ms;
                    timing.backend_res_buffer_hash_ms += pending_timing.backend_res_buffer_hash_ms;
                    timing.backend_res_descriptor_ms += pending_timing.backend_res_descriptor_ms;
                    timing.backend_setup_pipeline_ms += pending_timing.backend_setup_pipeline_ms;
                    timing.backend_pipeline_refs += pending_timing.backend_pipeline_refs;
                    timing.backend_pipeline_hits += pending_timing.backend_pipeline_hits;
                    timing.backend_pipeline_misses += pending_timing.backend_pipeline_misses;
                    timing.backend_pipeline_bypasses += pending_timing.backend_pipeline_bypasses;
                    timing.backend_pipeline_entries = pending_timing.backend_pipeline_entries;
                    timing.backend_pipeline_evictions += pending_timing.backend_pipeline_evictions;
                    timing.color_target_writes += pending_timing.color_target_writes;
                    timing.color_target_write_hits += pending_timing.color_target_write_hits;
                    timing.color_target_sample_hits += pending_timing.color_target_sample_hits;
                    timing.color_target_readbacks += pending_timing.color_target_readbacks;
                    timing.color_target_cached_bytes = pending_timing.color_target_cached_bytes;
                    timing.color_target_cached_entries = pending_timing.color_target_cached_entries;
                    timing.textures += pending_timing.textures;
                    timing.texture_reuses += pending_timing.texture_reuses;
                    timing.persistent_hits += pending_timing.persistent_hits;
                    timing.persistent_misses += pending_timing.persistent_misses;
                    timing.persistent_invalidations += pending_timing.persistent_invalidations;
                    timing.persistent_submit_reuses += pending_timing.persistent_submit_reuses;
                    timing.persistent_validations += pending_timing.persistent_validations;
                    timing.persistent_validation_bytes += pending_timing.persistent_validation_bytes;
                    timing.persistent_watch_reuses += pending_timing.persistent_watch_reuses;
                    timing.persistent_watch_dirty += pending_timing.persistent_watch_dirty;
                    timing.persistent_watch_unknown += pending_timing.persistent_watch_unknown;
                    timing.persistent_submit_unknown += pending_timing.persistent_submit_unknown;
                    timing.persistent_submit_overlap += pending_timing.persistent_submit_overlap;
                    timing.persistent_submit_unarmed += pending_timing.persistent_submit_unarmed;
                    timing.persistent_submit_stale += pending_timing.persistent_submit_stale;
                    timing.publish_selected += pending_timing.publish_selected;
                    timing.publish_selected_fmt0 += pending_timing.publish_selected_fmt0;
                    timing.publish_selected_unknown += pending_timing.publish_selected_unknown;
                    timing.publish_candidate_fmt0 += pending_timing.publish_candidate_fmt0;
                    timing.persistent_watch_disabled += pending_timing.persistent_watch_disabled;
                    timing.buffers += pending_timing.buffers;
                    timing.buffer_views += pending_timing.buffer_views;
                    timing.texture_bytes += pending_timing.texture_bytes;
                    timing.buffer_bytes += pending_timing.buffer_bytes;
                    timing.buffer_materialized_bytes += pending_timing.buffer_materialized_bytes;
                    timing.texture_ms += pending_timing.texture_ms;
                    timing.tex_rtt_ms += pending_timing.tex_rtt_ms;
                    timing.tex_compute_ms += pending_timing.tex_compute_ms;
                    timing.tex_local_ms += pending_timing.tex_local_ms;
                    timing.tex_persist_hit_ms += pending_timing.tex_persist_hit_ms;
                    timing.tex_persist_reuse_ms += pending_timing.tex_persist_reuse_ms;
                    timing.tex_persist_miss_ms += pending_timing.tex_persist_miss_ms;
                    timing.tex_persist_invalid_ms += pending_timing.tex_persist_invalid_ms;
                    timing.tex_persist_invalid_n += pending_timing.tex_persist_invalid_n;
                    timing.tex_rtt_n += pending_timing.tex_rtt_n;
                    timing.tex_compute_n += pending_timing.tex_compute_n;
                    timing.tex_local_n += pending_timing.tex_local_n;
                    timing.tex_persist_hit_n += pending_timing.tex_persist_hit_n;
                    timing.tex_persist_reuse_n += pending_timing.tex_persist_reuse_n;
                    timing.tex_persist_miss_n += pending_timing.tex_persist_miss_n;
                    timing.tex_other_n += pending_timing.tex_other_n;
                    timing.pass_pre_ms += pending_timing.pass_pre_ms;
                    timing.pass_head_ms += pending_timing.pass_head_ms;
                    timing.pass_loop_ms += pending_timing.pass_loop_ms;
                    timing.resolve_stall_ms += pending_timing.resolve_stall_ms;
                    timing.resolve_copy_stall_ms += pending_timing.resolve_copy_stall_ms;
                    timing.resolve_read_ms += pending_timing.resolve_read_ms;
                    timing.resolve_copy_ms += pending_timing.resolve_copy_ms;
                    timing.resolve_n += pending_timing.resolve_n;
                    timing.resolve_read_n += pending_timing.resolve_read_n;
                    timing.resolve_bytes += pending_timing.resolve_bytes;
                    timing.pass_groups_seen += pending_timing.pass_groups_seen;
                    timing.pass_tail_ms += pending_timing.pass_tail_ms;
                    timing.pass_post_ms += pending_timing.pass_post_ms;
                    timing.pass_groups += pending_timing.pass_groups;
                    timing.build_r_ms += pending_timing.build_r_ms;
                    timing.build_validate_ms += pending_timing.build_validate_ms;
                    timing.build_poison_ms += pending_timing.build_poison_ms;
                    timing.build_indices_ms += pending_timing.build_indices_ms;
                    timing.build_draws += pending_timing.build_draws;
                    timing.build_rejected += pending_timing.build_rejected;
                    timing.build_reflect_ms += pending_timing.build_reflect_ms;
                    timing.build_reflect_calls += pending_timing.build_reflect_calls;
                    timing.build_reflect_words += pending_timing.build_reflect_words;
                    timing.build_reflect_unidentified += pending_timing.build_reflect_unidentified;
                    timing.build_reflect_distinct += pending_timing.build_reflect_identities.size();
                    timing.build_reflect_memo_hits += pending_timing.build_reflect_memo_hits;
                    timing.build_reflect_memo_clears += pending_timing.build_reflect_memo_clears;
                    timing.build_reflect_memo_mismatch += pending_timing.build_reflect_memo_mismatch;
                    timing.buffer_ms += pending_timing.buffer_ms;
                };
                accumulate(totals);
                accumulate(window);
                if (totals.submits % 25 == 0) {
                    const double nsub = static_cast<double>(totals.submits);
                    const double pass_control = totals.pass_ms - totals.build_resources_ms -
                                                totals.backend_ms;
                    const double other = totals.total_ms - totals.prelude_ms - totals.pass_ms -
                                         totals.output_copy_ms;
                    fprintf(stderr,
                            "[render-timing] frontend submits=%llu callbacks=%llu avg_ms: total=%.2f "
                            "prelude=%.2f build_resources=%.2f backend=%.2f pass_control=%.2f "
                            "output_copy=%.2f other=%.2f dcc=%.2f/%0.2f/%.1fMiB\n",
                            (unsigned long long)totals.submits, (unsigned long long)totals.callbacks,
                            totals.total_ms / nsub, totals.prelude_ms / nsub,
                            totals.build_resources_ms / nsub, totals.backend_ms / nsub,
                            pass_control / nsub, totals.output_copy_ms / nsub, other / nsub,
                            totals.dcc_materialize_ms / nsub,
                            static_cast<double>(totals.dcc_materialize_surfaces) / nsub,
                            totals.dcc_materialize_bytes / (nsub * 1024.0 * 1024.0));
                    // #2395: the shader-analysis cache's counters existed and were read by NOTHING
                    // outside two unit tests, so at runtime the cache was unobservable. A profile of
                    // Blue Prince gameplay put ~17% of all CPU in rdna2_walk / make_shader_compile_key /
                    // resolve_dynamic_fetch -- i.e. shaders being re-analysed during steady-state play --
                    // and there was no way to tell WHY without rebuilding with instrumentation.
                    //
                    // `invalidations` is the discriminator and the reason this line exists. The cache is
                    // keyed on the guest CODE ADDRESS and validated by memcmp, so:
                    //   misses high, invalidations ~0  -> cold or capacity-bound (see evict/entries)
                    //   invalidations high             -> the guest reuses addresses for DIFFERENT code,
                    //                                     so every reuse costs a full re-analysis. That
                    //                                     is a keying problem, not a sizing one, and no
                    //                                     amount of cache MB fixes it.
                    // Those two call for opposite fixes, which is exactly why the number has to be
                    // visible rather than inferred from a flame graph.
                    {
                        const prosper::gpu::ShaderAnalysisCacheStats sa =
                            prosper::gpu::shader_analysis_cache_stats();
                        const uint64_t lookups = sa.hits + sa.misses;
                        fprintf(stderr,
                                "[render-timing] shader_analysis hits=%llu misses=%llu (%.1f%% hit) "
                                "invalidations=%llu evictions=%llu bypasses=%llu entries=%llu %.1f MiB\n",
                                (unsigned long long)sa.hits, (unsigned long long)sa.misses,
                                lookups ? 100.0 * (double)sa.hits / (double)lookups : 0.0,
                                (unsigned long long)sa.invalidations, (unsigned long long)sa.evictions,
                                (unsigned long long)sa.bypasses, (unsigned long long)sa.entries,
                                sa.bytes / (1024.0 * 1024.0));
                    }
                    // pass_control, split (#2266). `pre` and `post` are the per-group work either
                    // side of build_resources+backend; `other` is the SIGNED remainder -- groups that
                    // returned before reaching the render, the tail after the group loop, and the
                    // loop's own overhead. Published rather than clamped, because an unmeasured
                    // region does not read as unmeasured: it reads as somebody else's cost, which is
                    // how 85 ms/submit sat here attributed to nothing.
                    {
                        const double pc_other = pass_control - totals.pass_pre_ms -
                                                totals.pass_post_ms - totals.pass_head_ms -
                                                totals.pass_tail_ms -
                                                (totals.pass_loop_ms - totals.pass_pre_ms -
                                                 totals.pass_post_ms);
                        fprintf(stderr,
                                "[render-timing] pass_control %.2f ms/submit [head=%.2f loop=%.2f "
                                "(pre=%.2f post=%.2f) tail=%.2f other=%+.2f] "
                                "groups=%.1f rendered of %.1f seen\n",
                                pass_control / nsub, totals.pass_head_ms / nsub,
                                totals.pass_loop_ms / nsub,
                                totals.pass_pre_ms / nsub, totals.pass_post_ms / nsub,
                                totals.pass_tail_ms / nsub, pc_other / nsub,
                                (double)totals.pass_groups / nsub,
                                (double)totals.pass_groups_seen / nsub);
                        if (pc_other < -0.05 * nsub)
                            fprintf(stderr,
                                    "[render-timing] *** pass_control leaves EXCEED their parent by %.2f "
                                    "ms/submit -- instrument defect, not renderer\n",
                                    -pc_other / nsub);
                    }
                    // The group loop's ONE early exit is the MSAA resolve, so the gap between
                    // `loop` and pre+post+build+backend is resolve work. Split by what a resolve
                    // does, because the three have different fixes: `stall` is submit_and_wait (a
                    // full GPU pipeline flush), `read` is a whole-surface GPU->CPU readback done
                    // only to materialise a CPU pixel mirror, `copy` is the GPU-side destination
                    // copy that has to happen either way.
                    fprintf(stderr,
                            "[render-timing] resolve %.1f/submit [stall=%.2f read=%.2f "
                            "copy_stall=%.2f copy=%.2f] "
                            "reads=%.1f %.1f MiB/submit\n",
                            (double)totals.resolve_n / nsub, totals.resolve_stall_ms / nsub,
                            totals.resolve_read_ms / nsub, totals.resolve_copy_stall_ms / nsub,
                            totals.resolve_copy_ms / nsub,
                            (double)totals.resolve_read_n / nsub,
                            totals.resolve_bytes / (nsub * 1024.0 * 1024.0));
                    const double backend_detail_ms = totals.backend_target_ms +
                        totals.backend_draw_setup_ms + totals.backend_record_upload_ms +
                        totals.backend_gpu_wait_ms + totals.backend_readback_ms +
                        totals.backend_cleanup_ms;
                    fprintf(stderr,
                            // detail is the SIX-term partition (target draw_setup record_upload
                            // gpu_wait readback cleanup) and `other` is measured - detail, so the
                            // row IS exhaustive. What it used to invite was summing NINE terms:
                            // gpu_device and gpu_overhead are printed here too, and
                            // gpu_overhead is COMPUTED as gpu_wait - gpu_device, so those two
                            // add up to gpu_wait exactly and a nine-term sum double-counts it.
                            // That is why the residual equalled gpu_wait to the hundredth
                            // (#2248) -- not because gpu_wait was nested in a neighbour, but
                            // because its own two CHILDREN were printed as its siblings.
                            //
                            // Parenthesised now, so the nesting is in the syntax rather than
                            // recoverable only by arithmetic, and `other` is signed so the row
                            // states whether it balances instead of leaving that to be derived.
                            "[render-timing] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                            "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                            "gpu_wait=%.2f (device=%.2f overhead=%.2f) readback=%.2f "
                            "cleanup=%.2f other=%+.2f\n",
                            totals.backend_calls / nsub, totals.backend_draws / nsub,
                            totals.backend_ms / nsub, backend_detail_ms / nsub,
                            totals.backend_target_ms / nsub, totals.backend_draw_setup_ms / nsub,
                            totals.backend_record_upload_ms / nsub, totals.backend_gpu_wait_ms / nsub,
                            totals.backend_gpu_device_ms / nsub,
                            std::max(0.0, totals.backend_gpu_wait_ms - totals.backend_gpu_device_ms) / nsub,
                            totals.backend_readback_ms / nsub, totals.backend_cleanup_ms / nsub,
                            (totals.backend_ms - backend_detail_ms) / nsub);
                    fprintf(stderr,
                            "[render-timing] backend-submit synchronization command_buffers=%.2f "
                            "queue_submits=%.2f fence_waits=%.2f timestamps=%.2f  "
                            "flush{readback=%.2f storage_wb=%.2f explicit=%.2f "
                            "no_batch=%.2f}\n",
                            totals.backend_command_buffers / nsub,
                            totals.backend_queue_submits / nsub,
                            totals.backend_fence_waits / nsub,
                            totals.backend_gpu_timestamp_samples / nsub,
                            (double)totals.flush_readback / nsub,
                            (double)totals.flush_storage_writeback / nsub,
                            (double)totals.flush_explicit / nsub,
                            (double)totals.flush_no_batch / nsub);
                    fprintf(stderr,
                            // draw_setup is the PARENT one line above; these four are its leaves
                            // and they did not add up to it -- 75.16 against 82.46 in the window
                            // that found this, a 7.30 ms gap nothing named (#2249). An unmeasured
                            // region does not read as unmeasured, it reads as somebody else's
                            // cost, and 7.30 ms is small enough that nobody chases it and large
                            // enough to hide an instrument -- which is exactly how trap 130's
                            // 305 ms/submit got billed to the renderer.
                            "[render-timing] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                            "resources=%.2f pipeline=%.2f other=%+.2f  fixed{index_upload=%.2f blend=%.2f "
                            "depth_stencil=%.2f viewport=%.2f stages=%.2f pre_index_unsplit=%.2f "
                            "other=%+.2f} pre_index{subgroup_scan=%.2f other=%+.2f}\n",
                            totals.backend_setup_shader_ms / nsub,
                            totals.backend_setup_fixed_ms / nsub,
                            totals.backend_setup_resources_ms / nsub,
                            totals.backend_setup_pipeline_ms / nsub,
                            (totals.backend_draw_setup_ms - totals.backend_setup_shader_ms -
                             totals.backend_setup_fixed_ms - totals.backend_setup_resources_ms -
                             totals.backend_setup_pipeline_ms) / nsub,
                            totals.backend_res_fixed_index_upload_ms / nsub,
                            totals.backend_res_fixed_blend_ms / nsub,
                            totals.backend_res_fixed_depth_stencil_ms / nsub,
                            totals.backend_res_fixed_viewport_ms / nsub,
                            totals.backend_res_fixed_stages_ms / nsub,
                            totals.backend_res_fixed_prologue_ms / nsub,
                            (totals.backend_setup_fixed_ms - totals.backend_res_fixed_index_upload_ms -
                             totals.backend_res_fixed_blend_ms - totals.backend_res_fixed_depth_stencil_ms -
                             totals.backend_res_fixed_viewport_ms - totals.backend_res_fixed_stages_ms -
                             totals.backend_res_fixed_prologue_ms) / nsub,
                            totals.backend_res_prologue_subgroup_scan_ms / nsub,
                            (totals.backend_res_fixed_prologue_ms - totals.backend_res_prologue_subgroup_scan_ms) / nsub);
                    fprintf(stderr,
                            "[render-timing] backend-submit resources avg_ms: texture=%.2f "
                            "(upload=%.2f bind=%.2f lookup=%.2f) buffer=%.2f "
                            "(acquire=%.2f copy=%.2f create=%.2f index_find=%.2f "
                            "index_insert=%.2f hash=%.2f "
                            "other=%+.2f) descriptor=%.2f other=%.2f\n",
                            totals.backend_res_texture_ms / nsub,
                            totals.backend_res_texture_upload_ms / nsub,
                            totals.backend_res_texture_bind_ms / nsub,
                            (totals.backend_res_texture_ms - totals.backend_res_texture_upload_ms -
                             totals.backend_res_texture_bind_ms) / nsub,
                            totals.backend_res_buffer_ms / nsub,
                            totals.backend_res_buffer_acquire_ms / nsub,
                            totals.backend_res_buffer_copy_ms / nsub,
                            totals.backend_res_buffer_create_ms / nsub,
                            totals.backend_res_buffer_index_find_ms / nsub,
                            totals.backend_res_buffer_index_insert_ms / nsub,
                            totals.backend_res_buffer_hash_ms / nsub,
                            (totals.backend_res_buffer_ms - totals.backend_res_buffer_acquire_ms -
                             totals.backend_res_buffer_copy_ms - totals.backend_res_buffer_create_ms -
                             totals.backend_res_buffer_index_find_ms -
                             totals.backend_res_buffer_index_insert_ms - totals.backend_res_buffer_hash_ms) / nsub,
                            totals.backend_res_descriptor_ms / nsub,
                            (totals.backend_setup_resources_ms - totals.backend_res_texture_ms -
                             totals.backend_res_buffer_ms - totals.backend_res_descriptor_ms) / nsub);
                    fprintf(stderr,
                            "[render-timing] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                            "bypass=%.1f entries=%llu evictions=%.1f\n",
                            totals.backend_pipeline_refs / nsub,
                            totals.backend_pipeline_hits / nsub,
                            totals.backend_pipeline_misses / nsub,
                            totals.backend_pipeline_bypasses / nsub,
                            (unsigned long long)totals.backend_pipeline_entries,
                            totals.backend_pipeline_evictions / nsub);
                    const auto backend_buffers =
                        prosper::test::render_host_buffer_pool_stats();
                    fprintf(stderr,
                            "[render-timing] backend_buffer_pool hits=%llu misses=%llu cached=%zu "
                            "%.1f MiB evictions=%llu\n",
                            (unsigned long long)backend_buffers.hits,
                            (unsigned long long)backend_buffers.misses,
                            backend_buffers.cached_buffers,
                            backend_buffers.cached_bytes / (1024.0 * 1024.0),
                            (unsigned long long)backend_buffers.evictions);
                    fprintf(stderr,
                            "[render-timing] publish_source selected=%llu fmt0=%llu unknown=%llu passes_fmt0=%llu\n",
                            (unsigned long long)totals.publish_selected,
                            (unsigned long long)totals.publish_selected_fmt0,
                            (unsigned long long)totals.publish_selected_unknown,
                            (unsigned long long)totals.publish_candidate_fmt0);
                    fprintf(stderr,
                            "[render-timing] color_targets writes=%llu load_hits=%llu sample_hits=%llu "
                            "readbacks=%llu deferred=%llu cached=%llu %.1f MiB\n",
                            (unsigned long long)totals.color_target_writes,
                            (unsigned long long)totals.color_target_write_hits,
                            (unsigned long long)totals.color_target_sample_hits,
                            (unsigned long long)totals.color_target_readbacks,
                            (unsigned long long)(totals.color_target_writes -
                                                 totals.color_target_readbacks),
                            (unsigned long long)totals.color_target_cached_entries,
                            totals.color_target_cached_bytes / (1024.0 * 1024.0));
                    fprintf(stderr,
                            "[render-timing] resources textures=%llu reused=%llu %.1f MiB %.2f ms/submit; "
                            "buffers=%llu views=%llu logical=%.1f MiB materialized=%.1f MiB "
                            "%.2f ms/submit\n",
                            (unsigned long long)totals.textures,
                            (unsigned long long)totals.texture_reuses,
                            totals.texture_bytes / (1024.0 * 1024.0), totals.texture_ms / nsub,
                            (unsigned long long)totals.buffers,
                            (unsigned long long)totals.buffer_views,
                            totals.buffer_bytes / (1024.0 * 1024.0),
                            totals.buffer_materialized_bytes / (1024.0 * 1024.0),
                            totals.buffer_ms / nsub);
                    // texture_ms attributed by OUTCOME class (#2262). The classes are decided in
                    // a fixed order on the per-resource path and classified in that same order, so
                    // they are mutually exclusive and every reference lands in exactly one -- or in
                    // `unclassified`, which is a COUNT of references nothing above claimed. `other`
                    // is the SIGNED millisecond remainder, published rather than clamped for the
                    // reason #2246 records. The two residuals are independent: a non-zero `other`
                    // with `unclassified=0` means the classification is losing time that a claimed
                    // class should hold, which is a defect in this instrument and not in the
                    // renderer.
                    {
                        const double tex_other = totals.texture_ms - totals.tex_rtt_ms -
                                                 totals.tex_compute_ms - totals.tex_local_ms -
                                                 totals.tex_persist_hit_ms - totals.tex_persist_reuse_ms -
                                                 totals.tex_persist_miss_ms -
                                                 totals.tex_persist_invalid_ms;
                        fprintf(stderr,
                                "[render-timing] texture %.2f ms/submit [rtt=%.2f/%.1f compute=%.2f/%.1f "
                                "persist_reuse=%.2f/%.1f persist_hit=%.2f/%.1f "
                                "persist_miss=%.2f/%.1f persist_invalid=%.2f/%.1f "
                                "local=%.2f/%.1f] other=%+.2f "
                                "unclassified=%.1f  (ms/submit per class, refs/submit after the slash)\n",
                                totals.texture_ms / nsub,
                                totals.tex_rtt_ms / nsub, (double)totals.tex_rtt_n / nsub,
                                totals.tex_compute_ms / nsub, (double)totals.tex_compute_n / nsub,
                                totals.tex_persist_reuse_ms / nsub, (double)totals.tex_persist_reuse_n / nsub,
                                totals.tex_persist_hit_ms / nsub, (double)totals.tex_persist_hit_n / nsub,
                                totals.tex_persist_miss_ms / nsub, (double)totals.tex_persist_miss_n / nsub,
                                totals.tex_persist_invalid_ms / nsub, (double)totals.tex_persist_invalid_n / nsub,
                                totals.tex_local_ms / nsub, (double)totals.tex_local_n / nsub,
                                tex_other / nsub, (double)totals.tex_other_n / nsub);
                        if (tex_other < -0.05 * nsub)
                            fprintf(stderr,
                                    "[render-timing] *** texture classes EXCEED their parent by %.2f "
                                    "ms/submit -- the classification double-counts; instrument "
                                    "defect, not renderer\n", -tex_other / nsub);
                    }
                    // build_resources, exhaustively (#2215). The leaves are per-draw spans inside
                    // build_bds; `other` is the SIGNED residual -- everything in build_resources
                    // that is not one of them, which on a healthy partition is the loop overhead
                    // and the BackendDraw field copies. A large positive residual means work is
                    // going somewhere unnamed; a NEGATIVE one means the leaves over-count their
                    // parent, which is a defect in this instrument and not in the renderer. Both
                    // are printed because a partition that cannot report its own incompleteness
                    // attributes the gap to the subject -- the failure that cost #2246 two hours
                    // and produced a 305 ms phantom.
                    //
                    // texture/buffer above are NOT leaves of this partition: they are measured
                    // INSIDE build_R and are therefore a sub-split of `build_R` below. Adding them
                    // here would double-count. That is the cross-layer subtraction #2243 fixed,
                    // and this comment exists so nobody re-derives it from the field names.
                    {
                        // build_reflect_ms is INSIDE build_r_ms, so it is a sub-split of the
                        // build_R leaf and must not be subtracted from build_resources here --
                        // only from build_R's own residual above. Subtracting it at both levels
                        // would drive the outer residual negative and trip the guard on a run
                        // where nothing is wrong.
                        const double br_other = totals.build_resources_ms - totals.build_r_ms -
                                                totals.build_validate_ms - totals.build_poison_ms -
                                                totals.build_indices_ms;
                        fprintf(stderr,
                                "[render-timing] build_resources %.2f ms/submit [build_R=%.2f "
                                "(texture=%.2f buffer=%.2f reflect=%.2f other=%+.2f) "
                                "validate=%.2f indices=%.2f "
                                "poison=%.2f other=%+.2f] draws=%.1f (+%.1f rejected; build_R and validate cover both, poison and indices only the accepted)\n",
                                totals.build_resources_ms / nsub, totals.build_r_ms / nsub,
                                totals.texture_ms / nsub, totals.buffer_ms / nsub,
                                totals.build_reflect_ms / nsub,
                                // build_R's OWN residual, published for the same reason as its
                                // parent's. texture+buffer covered 3.71 of 5.20 ms on the first run
                                // of this instrument -- 29% of build_R with nothing naming it. A
                                // nested partition that reports only the outer remainder just moves
                                // the blind spot one level down, which is the mistake this whole
                                // line exists to stop repeating.
                                (totals.build_r_ms - totals.texture_ms - totals.buffer_ms -
                                 totals.build_reflect_ms) / nsub,
                                totals.build_validate_ms / nsub, totals.build_indices_ms / nsub,
                                totals.build_poison_ms / nsub, br_other / nsub,
                                (double)totals.build_draws / nsub,
                                (double)totals.build_rejected / nsub);
                        // A negative residual has one KNOWN trigger, and naming it here saves
                        // the reader hunting a renderer defect: PROSPER_TARGET_STEP_HASH_DIM calls
                        // build_bds() again at :5815, OUTSIDE the build_start/build_done span, so
                        // every prefix rebuild lands in the leaves without landing in the parent --
                        // and that loop is O(N^2) in draws. Zero occurrences on a default run.
                        // The reflect leaf's own supporting numbers (#2256). `words` is what
                        // the memo's key derivation and its equality check each traverse, so the
                        // ms is checkable against it rather than merely assertable. `distinct` is
                        // the per-submit ceiling on what an identity-keyed memo could remove:
                        // calls - distinct - unidentified is the number of calls it could serve
                        // from cache, and if that difference is small the memo is not worth
                        // building however large the ms turns out to be.
                        fprintf(stderr,
                                "[render-timing] build_R reflect %.3f ms/submit calls=%.1f "
                                "distinct=%.1f unidentified=%.1f words=%.0f/submit "
                                "memo_hits=%.1f clears=%.1f mismatch=%.1f\n",
                                totals.build_reflect_ms / nsub,
                                (double)totals.build_reflect_calls / nsub,
                                (double)totals.build_reflect_distinct / nsub,
                                (double)totals.build_reflect_unidentified / nsub,
                                (double)totals.build_reflect_words / nsub,
                                (double)totals.build_reflect_memo_hits / nsub,
                                (double)totals.build_reflect_memo_clears / nsub,
                                (double)totals.build_reflect_memo_mismatch / nsub);
                        if (br_other < -0.05 * nsub)
                            fprintf(stderr,
                                    "[render-timing] *** build_resources leaves EXCEED their parent "
                                    "by %.2f ms/submit -- the partition above is not trustworthy; "
                                    "this is a defect in the instrument, not the renderer. Known "
                                    "trigger: PROSPER_TARGET_STEP_HASH_DIM rebuilds draws outside "
                                    "the measured span\n",
                                    -br_other / nsub);
                    }
                    size_t persistent_source_bytes = 0;
                    size_t persistent_pixel_bytes = 0;
                    size_t persistent_watch_only_entries = 0;
                    size_t persistent_watch_only_saved_bytes = 0;
                    for (const auto& [key, texture] : persistent_decoded_textures) {
                        (void)key;
                        persistent_source_bytes += texture.source_prefix.size();
                        persistent_pixel_bytes += texture.pixels ? texture.pixels->size() : 0;
                        if (texture.source_watch_only) {
                            ++persistent_watch_only_entries;
                            persistent_watch_only_saved_bytes += texture.source_prefix_size;
                        }
                    }
                    fprintf(stderr,
                            "[render-timing] texture_cache hits=%llu submit_reuse=%llu "
                            "(submit_unknown=%llu [unarmed=%llu stale=%llu] submit_overlap=%llu) misses=%llu "
                            "watch_reuse=%llu watch_dirty=%llu watch_unknown=%llu watch_disabled=%llu "
                            "invalid=%llu "
                            "validations=%llu %.1f GiB entries=%zu %.1f MiB "
                            "(source=%.1f pixels=%.1f MiB watch-only=%zu saved=%.1f MiB)\n",
                            (unsigned long long)totals.persistent_hits,
                            (unsigned long long)totals.persistent_submit_reuses,
                            (unsigned long long)totals.persistent_submit_unknown,
                            (unsigned long long)totals.persistent_submit_unarmed,
                            (unsigned long long)totals.persistent_submit_stale,
                            (unsigned long long)totals.persistent_submit_overlap,
                            (unsigned long long)totals.persistent_misses,
                            (unsigned long long)totals.persistent_watch_reuses,
                            (unsigned long long)totals.persistent_watch_dirty,
                            (unsigned long long)totals.persistent_watch_unknown,
                            (unsigned long long)totals.persistent_watch_disabled,
                            (unsigned long long)totals.persistent_invalidations,
                            (unsigned long long)totals.persistent_validations,
                            totals.persistent_validation_bytes / (1024.0 * 1024.0 * 1024.0),
                            persistent_decoded_textures.size(),
                            persistent_decoded_texture_bytes / (1024.0 * 1024.0),
                            persistent_source_bytes / (1024.0 * 1024.0),
                            persistent_pixel_bytes / (1024.0 * 1024.0),
                            persistent_watch_only_entries,
                            persistent_watch_only_saved_bytes / (1024.0 * 1024.0));
                    // Identity-scope accounting (#1691). `cross_span` is the reuse submit scope adds
                    // over the historical span-scoped map; `invalidated` is entries the in-submit
                    // journal refused to carry across a span boundary.
                    fprintf(stderr,
                            "[render-timing] decode_scope decodes=%llu same_span=%llu "
                            "cross_span=%llu invalidated=%llu pinned=%llu scope=%s\n",
                            (unsigned long long)g_texture_decode_scope.decodes,
                            (unsigned long long)g_texture_decode_scope.same_span_reuses,
                            (unsigned long long)g_texture_decode_scope.cross_span_reuses,
                            (unsigned long long)g_texture_decode_scope.invalidations,
                            (unsigned long long)g_texture_decode_scope.scratch_pins,
                            submit_decode_scope_disabled ? "span" : "submit");
                    size_t rtt_bytes = 0;
                    for (const auto& [addr, surface] : g_rtt) {
                        (void)addr;
                        if (surface.rgba) rtt_bytes += surface.rgba->size();
                    }
                    size_t scratch_bytes = 0;
                    for (const auto& scratch : texstore)
                        scratch_bytes += scratch.capacity();
                    // The decode INTERMEDIATE pool is retained memory too, and a census that omits
                    // it under-reports the frontend's footprint by exactly the amount this change
                    // stopped churning -- which is the number a reader would want to see.
                    //
                    // It is THIS THREAD's pool, and the label says so. The pool is thread_local, so
                    // a process running N decoding threads retains up to N times this; printing a
                    // per-thread figure under a heading that reads as a process total is the kind
                    // of number that gets quoted as the whole footprint.
                    const auto& intermediates = prosper::frontend::decode_scratch_pool();
                    fprintf(stderr,
                            "[render-timing] host_cache rtt=%zu %.1f MiB decode_scratch=%zu %.1f MiB "
                            "intermediates(this thread)=%zu %.1f MiB validation=%.1f MiB\n",
                            g_rtt.size(), rtt_bytes / (1024.0 * 1024.0),
                            texstore.size(), scratch_bytes / (1024.0 * 1024.0),
                            intermediates.retained_buffers(),
                            intermediates.retained_bytes() / (1024.0 * 1024.0),
                            persistent_validation_scratch.capacity() / (1024.0 * 1024.0));
                    const auto write_watch = prosper::host::guest_write_watch_stats();
                    fprintf(stderr,
                            // `protect_fail` was spelled `protect` until #3317 added a field
                            // counting mprotect CALLS to the same line. A reader then has
                            // `protect=0` and `mprotect=N` side by side, and `protect=0` reads as
                            // "nothing was protected" -- which is the opposite of what it means
                            // (zero FAILURES is the healthy value). That misreading happened on the
                            // first run of the new field; see GAME_COMPAT_ORCHESTRATION.md.
                            "[render-timing] write_watch create=%llu ok=%llu pages=%llu no_map=%llu "
                            "alias=%llu oversized=%llu sizes=%llu/%llu/%llu/%llu protect_fail=%llu "
                            "query=%llu unchanged=%llu dirty=%llu "
                            "unknown=%llu faults=%llu stale=%llu physical=%llu rearms=%llu "
                            "host_write=%llu notifies/%llu no_alias pages_hit=%llu "
                            "lock_contended=%llu page_protect=%llu calls/%.1f MiB\n",
                            (unsigned long long)write_watch.create_attempts,
                            (unsigned long long)write_watch.registrations,
                            (unsigned long long)write_watch.registered_pages,
                            (unsigned long long)write_watch.create_no_mapping,
                            (unsigned long long)write_watch.create_incomplete_aliases,
                            (unsigned long long)write_watch.create_oversized,
                            (unsigned long long)write_watch.create_bytes_le_1m,
                            (unsigned long long)write_watch.create_bytes_le_8m,
                            (unsigned long long)write_watch.create_bytes_le_32m,
                            (unsigned long long)write_watch.create_bytes_gt_32m,
                            (unsigned long long)write_watch.create_protect_failures,
                            (unsigned long long)write_watch.queries,
                            (unsigned long long)write_watch.unchanged,
                            (unsigned long long)write_watch.dirty,
                            (unsigned long long)write_watch.unknown,
                            (unsigned long long)write_watch.faults,
                            (unsigned long long)write_watch.stale_faults,
                            (unsigned long long)write_watch.physical_writes,
                            (unsigned long long)write_watch.rearms,
                            (unsigned long long)write_watch.host_write_notifies,
                            (unsigned long long)write_watch.host_write_no_alias,
                            (unsigned long long)write_watch.host_write_pages_hit,
                            (unsigned long long)write_watch.host_write_lock_contended,
                            (unsigned long long)write_watch.protect_calls,
                            write_watch.protect_bytes / (1024.0 * 1024.0));
                    const double wn = static_cast<double>(window.submits);
                    const double window_pass_control = window.pass_ms -
                        window.build_resources_ms - window.backend_ms;
                    const double window_other = window.total_ms - window.prelude_ms -
                        window.pass_ms - window.output_copy_ms;
                    fprintf(stderr,
                            "[render-window] frontend submits=%llu callbacks=%.1f avg_ms: total=%.2f "
                            "prelude=%.2f build_resources=%.2f backend=%.2f pass_control=%.2f "
                            "output_copy=%.2f other=%.2f dcc=%.2f/%0.2f/%.1fMiB; "
                            "resources textures=%.1f "
                            "reused=%.1f %.1f MiB %.2f ms buffers=%.1f views=%.1f "
                            "logical=%.1f MiB materialized=%.1f MiB %.2f ms\n",
                            (unsigned long long)window.submits, window.callbacks / wn,
                            window.total_ms / wn, window.prelude_ms / wn,
                            window.build_resources_ms / wn, window.backend_ms / wn,
                            window_pass_control / wn, window.output_copy_ms / wn, window_other / wn,
                            window.dcc_materialize_ms / wn,
                            static_cast<double>(window.dcc_materialize_surfaces) / wn,
                            window.dcc_materialize_bytes / (wn * 1024.0 * 1024.0),
                            window.textures / wn,
                            window.texture_reuses / wn,
                            window.texture_bytes / (wn * 1024.0 * 1024.0), window.texture_ms / wn,
                            window.buffers / wn, window.buffer_views / wn,
                            window.buffer_bytes / (wn * 1024.0 * 1024.0),
                            window.buffer_materialized_bytes / (wn * 1024.0 * 1024.0),
                            window.buffer_ms / wn);
                    {
                        const double pc_other = window_pass_control - window.pass_pre_ms -
                                                window.pass_post_ms - window.pass_head_ms -
                                                window.pass_tail_ms -
                                                (window.pass_loop_ms - window.pass_pre_ms -
                                                 window.pass_post_ms);
                        fprintf(stderr,
                                "[render-window] pass_control %.2f ms/submit [head=%.2f loop=%.2f "
                                "(pre=%.2f post=%.2f) tail=%.2f other=%+.2f] "
                                "groups=%.1f rendered of %.1f seen\n",
                                window_pass_control / wn, window.pass_head_ms / wn,
                                window.pass_loop_ms / wn,
                                window.pass_pre_ms / wn, window.pass_post_ms / wn,
                                window.pass_tail_ms / wn, pc_other / wn,
                                (double)window.pass_groups / wn,
                                (double)window.pass_groups_seen / wn);
                        if (pc_other < -0.05 * wn)
                            fprintf(stderr,
                                    "[render-window] *** pass_control leaves EXCEED their parent by "
                                    "%.2f ms/submit -- instrument defect, not renderer\n",
                                    -pc_other / wn);
                    }
                    fprintf(stderr,
                            "[render-window] resolve %.1f/submit [stall=%.2f read=%.2f "
                            "copy_stall=%.2f copy=%.2f] "
                            "reads=%.1f %.1f MiB/submit\n",
                            (double)window.resolve_n / wn, window.resolve_stall_ms / wn,
                            window.resolve_read_ms / wn, window.resolve_copy_stall_ms / wn,
                            window.resolve_copy_ms / wn,
                            (double)window.resolve_read_n / wn,
                            window.resolve_bytes / (wn * 1024.0 * 1024.0));
                    // texture_ms attributed by OUTCOME class (#2262). The classes are decided in
                    // a fixed order on the per-resource path and classified in that same order, so
                    // they are mutually exclusive and every reference lands in exactly one -- or in
                    // `unclassified`, which is a COUNT of references nothing above claimed. `other`
                    // is the SIGNED millisecond remainder, published rather than clamped for the
                    // reason #2246 records. The two residuals are independent: a non-zero `other`
                    // with `unclassified=0` means the classification is losing time that a claimed
                    // class should hold, which is a defect in this instrument and not in the
                    // renderer.
                    {
                        const double tex_other = window.texture_ms - window.tex_rtt_ms -
                                                 window.tex_compute_ms - window.tex_local_ms -
                                                 window.tex_persist_hit_ms - window.tex_persist_reuse_ms -
                                                 window.tex_persist_miss_ms -
                                                 window.tex_persist_invalid_ms;
                        fprintf(stderr,
                                "[render-window] texture %.2f ms/submit [rtt=%.2f/%.1f compute=%.2f/%.1f "
                                "persist_reuse=%.2f/%.1f persist_hit=%.2f/%.1f "
                                "persist_miss=%.2f/%.1f persist_invalid=%.2f/%.1f "
                                "local=%.2f/%.1f] other=%+.2f "
                                "unclassified=%.1f  (ms/submit per class, refs/submit after the slash)\n",
                                window.texture_ms / wn,
                                window.tex_rtt_ms / wn, (double)window.tex_rtt_n / wn,
                                window.tex_compute_ms / wn, (double)window.tex_compute_n / wn,
                                window.tex_persist_reuse_ms / wn, (double)window.tex_persist_reuse_n / wn,
                                window.tex_persist_hit_ms / wn, (double)window.tex_persist_hit_n / wn,
                                window.tex_persist_miss_ms / wn, (double)window.tex_persist_miss_n / wn,
                                window.tex_persist_invalid_ms / wn, (double)window.tex_persist_invalid_n / wn,
                                window.tex_local_ms / wn, (double)window.tex_local_n / wn,
                                tex_other / wn, (double)window.tex_other_n / wn);
                        if (tex_other < -0.05 * wn)
                            fprintf(stderr,
                                    "[render-window] *** texture classes EXCEED their parent by %.2f "
                                    "ms/submit -- the classification double-counts; instrument "
                                    "defect, not renderer\n", -tex_other / wn);
                    }
                    const double window_backend_detail_ms = window.backend_target_ms +
                        window.backend_draw_setup_ms + window.backend_record_upload_ms +
                        window.backend_gpu_wait_ms + window.backend_readback_ms +
                        window.backend_cleanup_ms;
                    fprintf(stderr,
                            // detail is the SIX-term partition (target draw_setup record_upload
                            // gpu_wait readback cleanup) and `other` is measured - detail, so the
                            // row IS exhaustive. What it used to invite was summing NINE terms:
                            // gpu_device and gpu_overhead are printed here too, and
                            // gpu_overhead is COMPUTED as gpu_wait - gpu_device, so those two
                            // add up to gpu_wait exactly and a nine-term sum double-counts it.
                            // That is why the residual equalled gpu_wait to the hundredth
                            // (#2248) -- not because gpu_wait was nested in a neighbour, but
                            // because its own two CHILDREN were printed as its siblings.
                            //
                            // Parenthesised now, so the nesting is in the syntax rather than
                            // recoverable only by arithmetic, and `other` is signed so the row
                            // states whether it balances instead of leaving that to be derived.
                            "[render-window] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                            "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                            "gpu_wait=%.2f (device=%.2f overhead=%.2f) readback=%.2f "
                            "cleanup=%.2f other=%+.2f\n",
                            window.backend_calls / wn, window.backend_draws / wn,
                            window.backend_ms / wn, window_backend_detail_ms / wn,
                            window.backend_target_ms / wn, window.backend_draw_setup_ms / wn,
                            window.backend_record_upload_ms / wn, window.backend_gpu_wait_ms / wn,
                            window.backend_gpu_device_ms / wn,
                            std::max(0.0, window.backend_gpu_wait_ms - window.backend_gpu_device_ms) / wn,
                            window.backend_readback_ms / wn, window.backend_cleanup_ms / wn,
                            (window.backend_ms - window_backend_detail_ms) / wn);
                    fprintf(stderr,
                            "[render-window] backend-submit synchronization command_buffers=%.2f "
                            "queue_submits=%.2f fence_waits=%.2f timestamps=%.2f  "
                            "flush{readback=%.2f storage_wb=%.2f explicit=%.2f "
                            "no_batch=%.2f}\n",
                            window.backend_command_buffers / wn,
                            window.backend_queue_submits / wn,
                            window.backend_fence_waits / wn,
                            window.backend_gpu_timestamp_samples / wn,
                            (double)window.flush_readback / wn,
                            (double)window.flush_storage_writeback / wn,
                            (double)window.flush_explicit / wn,
                            (double)window.flush_no_batch / wn);
                    fprintf(stderr,
                            "[render-window] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                            "resources=%.2f pipeline=%.2f other=%+.2f  fixed{index_upload=%.2f blend=%.2f "
                            "depth_stencil=%.2f viewport=%.2f stages=%.2f pre_index_unsplit=%.2f "
                            "other=%+.2f} pre_index{subgroup_scan=%.2f other=%+.2f}\n",
                            window.backend_setup_shader_ms / wn,
                            window.backend_setup_fixed_ms / wn,
                            window.backend_setup_resources_ms / wn,
                            window.backend_setup_pipeline_ms / wn,
                            (window.backend_draw_setup_ms - window.backend_setup_shader_ms -
                             window.backend_setup_fixed_ms - window.backend_setup_resources_ms -
                             window.backend_setup_pipeline_ms) / wn,
                            window.backend_res_fixed_index_upload_ms / wn,
                            window.backend_res_fixed_blend_ms / wn,
                            window.backend_res_fixed_depth_stencil_ms / wn,
                            window.backend_res_fixed_viewport_ms / wn,
                            window.backend_res_fixed_stages_ms / wn,
                            window.backend_res_fixed_prologue_ms / wn,
                            (window.backend_setup_fixed_ms - window.backend_res_fixed_index_upload_ms -
                             window.backend_res_fixed_blend_ms - window.backend_res_fixed_depth_stencil_ms -
                             window.backend_res_fixed_viewport_ms - window.backend_res_fixed_stages_ms -
                             window.backend_res_fixed_prologue_ms) / wn,
                            window.backend_res_prologue_subgroup_scan_ms / wn,
                            (window.backend_res_fixed_prologue_ms - window.backend_res_prologue_subgroup_scan_ms) / wn);
                    fprintf(stderr,
                            "[render-window] backend-submit resources avg_ms: texture=%.2f "
                            "(upload=%.2f bind=%.2f lookup=%.2f) buffer=%.2f "
                            "(acquire=%.2f copy=%.2f create=%.2f index_find=%.2f "
                            "index_insert=%.2f hash=%.2f "
                            "other=%+.2f) descriptor=%.2f other=%.2f\n",
                            window.backend_res_texture_ms / wn,
                            window.backend_res_texture_upload_ms / wn,
                            window.backend_res_texture_bind_ms / wn,
                            (window.backend_res_texture_ms - window.backend_res_texture_upload_ms -
                             window.backend_res_texture_bind_ms) / wn,
                            window.backend_res_buffer_ms / wn,
                            window.backend_res_buffer_acquire_ms / wn,
                            window.backend_res_buffer_copy_ms / wn,
                            window.backend_res_buffer_create_ms / wn,
                            window.backend_res_buffer_index_find_ms / wn,
                            window.backend_res_buffer_index_insert_ms / wn,
                            window.backend_res_buffer_hash_ms / wn,
                            (window.backend_res_buffer_ms - window.backend_res_buffer_acquire_ms -
                             window.backend_res_buffer_copy_ms - window.backend_res_buffer_create_ms -
                             window.backend_res_buffer_index_find_ms -
                             window.backend_res_buffer_index_insert_ms - window.backend_res_buffer_hash_ms) / wn,
                            window.backend_res_descriptor_ms / wn,
                            (window.backend_setup_resources_ms - window.backend_res_texture_ms -
                             window.backend_res_buffer_ms - window.backend_res_descriptor_ms) / wn);
                    // Same partition on the WINDOW path, so it is present wherever the buffer
                    // and fixed ones are (#2246, #2252). The window path isolates a PHASE, which
                    // for a title whose draw count swings 351 -> 2,106 between phases is the more
                    // useful axis than a lifetime total -- and a partition visible on one path and
                    // not the other invites exactly the cross-aggregation comparison that has cost
                    // us two wrong numbers today.
                    {
                        const double br_other_w = window.build_resources_ms - window.build_r_ms -
                                                  window.build_validate_ms - window.build_poison_ms -
                                                  window.build_indices_ms;
                        fprintf(stderr,
                                "[render-window] build_resources %.2f ms/submit [build_R=%.2f "
                                "(texture=%.2f buffer=%.2f reflect=%.2f other=%+.2f) "
                                "validate=%.2f indices=%.2f "
                                "poison=%.2f other=%+.2f] draws=%.1f (+%.1f rejected)\n",
                                window.build_resources_ms / wn, window.build_r_ms / wn,
                                window.texture_ms / wn, window.buffer_ms / wn,
                                window.build_reflect_ms / wn,
                                (window.build_r_ms - window.texture_ms - window.buffer_ms -
                                 window.build_reflect_ms) / wn,
                                window.build_validate_ms / wn, window.build_indices_ms / wn,
                                window.build_poison_ms / wn, br_other_w / wn,
                                (double)window.build_draws / wn,
                                (double)window.build_rejected / wn);
                        fprintf(stderr,
                                "[render-window] build_R reflect %.3f ms/submit calls=%.1f "
                                "distinct=%.1f unidentified=%.1f words=%.0f/submit "
                                "memo_hits=%.1f clears=%.1f mismatch=%.1f\n",
                                window.build_reflect_ms / wn,
                                (double)window.build_reflect_calls / wn,
                                (double)window.build_reflect_distinct / wn,
                                (double)window.build_reflect_unidentified / wn,
                                (double)window.build_reflect_words / wn,
                                (double)window.build_reflect_memo_hits / wn,
                                (double)window.build_reflect_memo_clears / wn,
                                (double)window.build_reflect_memo_mismatch / wn);
                        if (br_other_w < -0.05 * wn)
                            fprintf(stderr,
                                    "[render-window] *** build_resources leaves EXCEED their parent "
                                    "by %.2f ms/submit -- instrument defect, not renderer. Known "
                                    "trigger: PROSPER_TARGET_STEP_HASH_DIM\n",
                                    -br_other_w / wn);
                    }
                    fprintf(stderr,
                            "[render-window] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                            "bypass=%.1f entries=%llu evictions=%.1f\n",
                            window.backend_pipeline_refs / wn,
                            window.backend_pipeline_hits / wn,
                            window.backend_pipeline_misses / wn,
                            window.backend_pipeline_bypasses / wn,
                            (unsigned long long)window.backend_pipeline_entries,
                            window.backend_pipeline_evictions / wn);
                    fprintf(stderr,
                            "[render-window] color_targets writes=%.1f load_hits=%.1f sample_hits=%.1f "
                            "readbacks=%.1f deferred=%.1f cached=%llu %.1f MiB\n",
                            window.color_target_writes / wn,
                            window.color_target_write_hits / wn,
                            window.color_target_sample_hits / wn,
                            window.color_target_readbacks / wn,
                            (window.color_target_writes - window.color_target_readbacks) / wn,
                            (unsigned long long)window.color_target_cached_entries,
                            window.color_target_cached_bytes / (1024.0 * 1024.0));
                    fprintf(stderr,
                            "[render-window] texture_cache hits=%.1f submit_reuse=%.1f misses=%.1f "
                            "watch_reuse=%.1f watch_dirty=%.1f watch_unknown=%.1f watch_disabled=%.1f "
                            "invalid=%.1f "
                            "validations=%.1f %.1f MiB\n",
                            window.persistent_hits / wn, window.persistent_submit_reuses / wn,
                            window.persistent_misses / wn, window.persistent_watch_reuses / wn,
                            window.persistent_watch_dirty / wn, window.persistent_watch_unknown / wn,
                            window.persistent_watch_disabled / wn,
                            window.persistent_invalidations / wn,
                            window.persistent_validations / wn,
                            window.persistent_validation_bytes / (wn * 1024.0 * 1024.0));
                    window = {};
                }
                // The complete semantic submit has now been recorded and added to aggregate logs.
                // Intermediate spans take the earlier return and retain this same population.
                prosper::frontend::reset_performance_timing_after_span(
                    pending_timing, timing_enabled, phase.final_span);
            }
            prosper::gpu::RenderedFrame frame(std::move(selected_pixels));
            frame.origin = frame_origin;
            return frame;
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend
